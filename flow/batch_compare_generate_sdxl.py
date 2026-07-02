#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDXL 批量对比生成工具
- 支持多个 LoRA 模型横向对比
- 支持多组 Prompt 纵向对比
- 自动拼接网格大图

Usage:
    python batch_compare_generate_sdxl.py --config configs/batch_compare_sdxl_config.json
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from einops import repeat

# 添加 sd-scripts 到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SD_SCRIPTS_DIR = SCRIPT_DIR.parent / "sd-scripts"
sys.path.insert(0, str(SD_SCRIPTS_DIR))

from library import sdxl_model_util, device_utils
from library.device_utils import get_preferred_device
from library.utils import setup_logging, str_to_dtype
import networks.lora as lora
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler

setup_logging()
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# Scheduler 常量
# ============================================================================
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDULER_SCHEDULE = "scaled_linear"


# ============================================================================
# 工具函数
# ============================================================================

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_random_seeds(count: int) -> List[int]:
    """生成指定数量的随机种子"""
    return [random.randint(0, 2**32 - 1) for _ in range(count)]


def get_best_grid_layout(count: int) -> Tuple[int, int]:
    """
    计算最佳网格布局（尽量接近正方形）
    返回 (rows, cols)
    """
    if count <= 0:
        return (1, 1)

    sqrt_val = int(math.sqrt(count))
    for cols in range(sqrt_val, count + 1):
        if count % cols == 0:
            return (count // cols, cols)

    # 如果不能整除，向上取整
    cols = sqrt_val + 1
    rows = (count + cols - 1) // cols
    return (rows, cols)


def try_load_font(size: int) -> ImageFont.FreeTypeFont:
    """尝试加载字体，失败则使用默认字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    # 回退到默认字体
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def get_timestep_embedding(x, outdim):
    """获取时间步嵌入"""
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_scheduler(scheduler_name: str, num_train_timesteps: int = 1000):
    """根据名称获取调度器"""
    scheduler_name = scheduler_name.lower()

    if scheduler_name in ["euler", "euler_discrete"]:
        return EulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=SCHEDULER_LINEAR_START,
            beta_end=SCHEDULER_LINEAR_END,
            beta_schedule=SCHEDULER_SCHEDULE,
        )
    elif scheduler_name in ["euler_a", "euler_ancestral"]:
        from diffusers import EulerAncestralDiscreteScheduler
        return EulerAncestralDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=SCHEDULER_LINEAR_START,
            beta_end=SCHEDULER_LINEAR_END,
            beta_schedule=SCHEDULER_SCHEDULE,
        )
    elif scheduler_name in ["dpm", "dpm++", "dpm_solver", "dpmpp_2m"]:
        return DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=SCHEDULER_LINEAR_START,
            beta_end=SCHEDULER_LINEAR_END,
            beta_schedule=SCHEDULER_SCHEDULE,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )
    elif scheduler_name in ["ddim"]:
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=SCHEDULER_LINEAR_START,
            beta_end=SCHEDULER_LINEAR_END,
            beta_schedule=SCHEDULER_SCHEDULE,
        )
    else:
        logger.warning(f"Unknown scheduler '{scheduler_name}', using euler")
        return EulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=SCHEDULER_LINEAR_START,
            beta_end=SCHEDULER_LINEAR_END,
            beta_schedule=SCHEDULER_SCHEDULE,
        )


# ============================================================================
# SDXL 推理核心
# ============================================================================

class SDXLGenerator:
    """SDXL 图像生成器"""

    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.base_model = config["base_model"]
        self.gen_config = config["generation"]

        # 数据类型
        self.dtype = str_to_dtype(self.gen_config.get("dtype", "float16"))
        # UNet 可以用 fp8 加速推理；Text Encoder 需要 bf16 保语义精度；VAE 需要 float32 保像素精度
        if self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            self.unet_dtype = self.dtype
            self.te_dtype = torch.bfloat16
            self.vae_dtype = torch.float32
            logger.info(f"FP8 mode: UNet={self.unet_dtype}, TextEncoder={self.te_dtype}, VAE={self.vae_dtype}")
        else:
            self.unet_dtype = self.dtype
            self.te_dtype = self.dtype
            self.vae_dtype = torch.float32 if self.dtype == torch.float16 else self.dtype
        self.offload = self.gen_config.get("offload", True)

        # 尺寸信息（用于SDXL的额外条件嵌入）
        self.target_height = self.gen_config.get("height", 1024)
        self.target_width = self.gen_config.get("width", 1024)
        self.original_height = self.gen_config.get("original_height", self.target_height)
        self.original_width = self.gen_config.get("original_width", self.target_width)
        self.crop_top = self.gen_config.get("crop_top", 0)
        self.crop_left = self.gen_config.get("crop_left", 0)

        # 加载模型
        self._load_base_models()

        # 当前加载的 LoRA
        self.current_lora = None
        self.lora_model = None

    def _load_base_models(self):
        """加载基础模型"""
        loading_device = "cpu" if self.offload else self.device

        logger.info(f"Loading SDXL model from {self.base_model['ckpt_path']}...")

        # 加载SDXL模型
        (
            self.text_model1,
            self.text_model2,
            self.vae,
            self.unet,
            _,
            _
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0,
            self.base_model["ckpt_path"],
            str(loading_device)
        )

        # 设置模型为评估模式
        self.text_model1.eval()
        self.text_model2.eval()
        self.vae.eval()
        self.unet.eval()

        # 设置内存优化 - 使用 SDPA (PyTorch 原生 attention) 而不是 xformers
        # set_use_memory_efficient_attention(xformers, sdpa)
        self.unet.set_use_memory_efficient_attention(False, True)
        # 不使用 xformers for VAE，避免 GPU 架构不兼容问题
        # if torch.__version__ >= "2.0.0":
        #     self.vae.set_use_memory_efficient_attention_xformers(True)

        # 加载Tokenizers
        text_encoder_1_name = "openai/clip-vit-large-patch14"
        text_encoder_2_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

        logger.info("Loading tokenizers...")
        self.tokenizer1 = CLIPTokenizer.from_pretrained(text_encoder_1_name)
        self.tokenizer2 = CLIPTokenizer.from_pretrained(text_encoder_2_name)

        # 创建调度器
        scheduler_name = self.gen_config.get("scheduler", "euler")
        self.scheduler = get_scheduler(scheduler_name)

        logger.info(f"SDXL model loaded successfully (scheduler: {scheduler_name})")

    def _reload_unet(self):
        """重新加载 UNet（用于切换 LoRA 时确保干净状态）"""
        loading_device = "cpu" if self.offload else self.device

        # 清理旧模型
        if hasattr(self, 'unet') and self.unet is not None:
            del self.unet
            device_utils.clean_memory()

        logger.info(f"Reloading UNet from {self.base_model['ckpt_path']}...")
        (
            self.text_model1,
            self.text_model2,
            self.vae,
            self.unet,
            _,
            _
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0,
            self.base_model["ckpt_path"],
            str(loading_device)
        )

        self.text_model1.eval()
        self.text_model2.eval()
        self.vae.eval()
        self.unet.eval()

        # 使用 SDPA (PyTorch 原生 attention) 而不是 xformers
        self.unet.set_use_memory_efficient_attention(False, True)
        # 不使用 xformers for VAE
        # if torch.__version__ >= "2.0.0":
        #     self.vae.set_use_memory_efficient_attention_xformers(True)

    def load_lora(self, lora_config: Optional[Dict]):
        """加载 LoRA 模型"""
        # 每次切换 LoRA 时都重新加载模型，确保干净状态
        if self.lora_model is not None or self.current_lora is not None:
            logger.info("Reloading models for clean LoRA switch...")
            self.lora_model = None
            self._reload_unet()
            device_utils.clean_memory()

        if lora_config is None or lora_config.get("path") is None:
            self.current_lora = None
            logger.info("No LoRA loaded (base model only)")
            return

        lora_path = lora_config["path"]
        lora_weight = lora_config.get("weight", 1.0)

        logger.info(f"Loading LoRA: {lora_path} (weight={lora_weight})")

        # 如果启用了 offload，需要先将模型移到 GPU 进行合并
        # LoRA merge 使用较高精度（te_dtype），fp8 精度不够会导致合并误差
        merge_dtype = self.te_dtype
        if self.offload:
            logger.info("Moving models to GPU for LoRA merge...")
            self.text_model1 = self.text_model1.float().to(self.device, dtype=merge_dtype)
            self.text_model2 = self.text_model2.float().to(self.device, dtype=merge_dtype)
            self.unet = self.unet.float().to(self.device, dtype=merge_dtype)

        self.lora_model, weights_sd = lora.create_network_from_weights(
            lora_weight, lora_path, self.vae,
            [self.text_model1, self.text_model2], self.unet, None, True
        )

        self.lora_model.merge_to(
            [self.text_model1, self.text_model2],
            self.unet,
            weights_sd,
            merge_dtype,
            self.device
        )

        # 如果启用了 offload，合并后将模型移回 CPU
        if self.offload:
            logger.info("Moving models back to CPU after LoRA merge...")
            self.text_model1 = self.text_model1.cpu()
            self.text_model2 = self.text_model2.cpu()
            self.unet = self.unet.cpu()
            device_utils.clean_memory()

        self.current_lora = lora_config

    def _encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码文本（双编码器）"""
        # Text Encoder 1 (CLIP-L)
        batch_encoding1 = self.tokenizer1(
            text,
            truncation=True,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        tokens1 = batch_encoding1["input_ids"].to(self.device)

        with torch.no_grad():
            enc_out1 = self.text_model1(tokens1, output_hidden_states=True, return_dict=True)
            text_embedding1 = enc_out1["hidden_states"][11]  # 第11层隐藏状态

        # Text Encoder 2 (OpenCLIP BigG)
        batch_encoding2 = self.tokenizer2(
            text,
            truncation=True,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        tokens2 = batch_encoding2["input_ids"].to(self.device)

        with torch.no_grad():
            enc_out2 = self.text_model2(tokens2, output_hidden_states=True, return_dict=True)
            text_embedding2_penu = enc_out2["hidden_states"][-2]  # 倒数第二层
            text_embedding2_pool = enc_out2["text_embeds"]  # 池化输出

        # 连接两个编码器的输出
        text_embedding = torch.cat([text_embedding1, text_embedding2_penu], dim=2)

        return text_embedding, text_embedding2_pool

    def _prepare_vector_embedding(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """准备SDXL的向量嵌入（包含尺寸信息）"""
        emb1 = get_timestep_embedding(
            torch.FloatTensor([self.original_height, self.original_width]).unsqueeze(0), 256
        )
        emb2 = get_timestep_embedding(
            torch.FloatTensor([self.crop_top, self.crop_left]).unsqueeze(0), 256
        )
        emb3 = get_timestep_embedding(
            torch.FloatTensor([self.target_height, self.target_width]).unsqueeze(0), 256
        )

        size_emb = torch.cat([emb1, emb2, emb3], dim=1).to(self.device, dtype=self.unet_dtype)
        vector_emb = torch.cat([pooled_output, size_emb], dim=1)

        return vector_emb

    def generate_batch(
        self,
        prompt: str,
        negative_prompt: str,
        seeds: List[int],
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
    ) -> List[Tuple[int, Image.Image]]:
        """批量生成多张图片（同一个 prompt，不同种子）"""
        batch_size = len(seeds)
        logger.info(f"Batch generating {batch_size} images with seeds={seeds}, size={width}x{height}")

        # 更新目标尺寸
        self.target_height = height
        self.target_width = width

        # 移动模型到设备（先 float32 再转目标 dtype，避免 FP8 直转失败）
        self.text_model1 = self.text_model1.float().to(self.device, dtype=self.te_dtype)
        self.text_model2 = self.text_model2.float().to(self.device, dtype=self.te_dtype)

        # 编码正向提示词
        c_ctx, c_ctx_pool = self._encode_text(prompt)
        c_vector = self._prepare_vector_embedding(c_ctx_pool)

        # 编码负向提示词
        uc_ctx, uc_ctx_pool = self._encode_text(negative_prompt)
        uc_vector = self._prepare_vector_embedding(uc_ctx_pool)

        if self.offload:
            self.text_model1 = self.text_model1.cpu()
            self.text_model2 = self.text_model2.cpu()
        device_utils.clean_memory()

        # 移动UNet到设备
        self.unet = self.unet.to(self.device, dtype=self.unet_dtype)

        # 生成图片列表
        images = []

        for seed in seeds:
            # 设置随机种子
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # 生成初始噪声
            latents_shape = (1, 4, height // 8, width // 8)
            latents = torch.randn(
                latents_shape,
                device="cpu",
                dtype=torch.float32,
            ).to(self.device, dtype=self.unet_dtype)

            # 设置调度器
            self.scheduler.set_timesteps(steps, self.device)
            latents = latents * self.scheduler.init_noise_sigma

            timesteps = self.scheduler.timesteps.to(self.device)

            # 合并条件嵌入（用于CFG），转为 UNet 的 dtype
            text_embeddings = torch.cat([uc_ctx, c_ctx]).to(dtype=self.unet_dtype)
            vector_embeddings = torch.cat([uc_vector, c_vector]).to(dtype=self.unet_dtype)

            # 去噪循环
            with torch.no_grad():
                for t in tqdm(timesteps, desc=f"Sampling (seed={seed})", leave=False):
                    # 扩展latents用于CFG
                    latent_model_input = latents.repeat((2, 1, 1, 1))
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # 预测噪声
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        text_embeddings,
                        vector_embeddings
                    )

                    # CFG
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # 调度器步进
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # 存储当前latent用于后续解码
            images.append((seed, latents.clone()))

        if self.offload:
            self.unet = self.unet.cpu()
        device_utils.clean_memory()

        # VAE解码
        self.vae = self.vae.to(self.device, dtype=self.vae_dtype)

        decoded_images = []
        for seed, latent in images:
            with torch.no_grad():
                # SDXL VAE缩放因子
                latent = latent / sdxl_model_util.VAE_SCALE_FACTOR
                latent = latent.to(self.vae_dtype)

                image = self.vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)

            # 转换为PIL图片
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            img = Image.fromarray(image[0])

            decoded_images.append((seed, img))

        if self.offload:
            self.vae = self.vae.cpu()
        device_utils.clean_memory()

        return decoded_images


# ============================================================================
# 拼图功能
# ============================================================================

class GridComposer:
    """网格拼图生成器"""

    def __init__(self, config: Dict):
        self.output_config = config["output"]
        self.padding = self.output_config.get("grid_padding", 8)
        self.add_labels = self.output_config.get("add_labels", True)
        self.label_font_size = self.output_config.get("label_font_size", 24)
        self.label_bg_color = tuple(self.output_config.get("label_bg_color", [0, 0, 0, 180]))
        self.label_text_color = tuple(self.output_config.get("label_text_color", [255, 255, 255]))

        self.font = try_load_font(self.label_font_size)

    def add_label_to_image(
        self,
        img: Image.Image,
        model_name: str,
        prompt_name: str,
        seed: int
    ) -> Image.Image:
        """在图片上添加标签"""
        if not self.add_labels:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 标签文本
        label_text = f"{model_name} | {prompt_name} | seed:{seed}"

        # 计算文本大小
        if self.font:
            bbox = draw.textbbox((0, 0), label_text, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(label_text) * 8
            text_height = 16

        # 绘制背景
        margin = 8
        bg_rect = [
            margin,
            img.height - text_height - margin * 3,
            text_width + margin * 3,
            img.height - margin
        ]
        draw.rectangle(bg_rect, fill=self.label_bg_color)

        # 绘制文本
        text_pos = (margin * 2, img.height - text_height - margin * 2)
        draw.text(text_pos, label_text, fill=self.label_text_color, font=self.font)

        return Image.alpha_composite(img, overlay).convert("RGB")

    def create_prompt_grid(
        self,
        all_images: Dict[int, Dict[str, Dict[str, Image.Image]]],
        model_names: List[str],
        prompt_name: str,
        seeds: List[int],
    ) -> Image.Image:
        """
        创建单个 Prompt 的对比网格（per-prompt 模式）

        all_images: {seed: {model_name: {prompt_name: Image}}}
        横向：模型
        纵向：种子
        """
        if not all_images or not model_names or not seeds:
            raise ValueError("Empty images, model names, or seeds")

        # 获取单张图片尺寸
        sample_seed = seeds[0]
        sample_model = model_names[0]
        sample_img = all_images[sample_seed][sample_model][prompt_name]
        img_width, img_height = sample_img.size

        # 计算网格尺寸
        num_cols = len(model_names)
        num_rows = len(seeds)

        grid_width = num_cols * img_width + (num_cols + 1) * self.padding
        grid_height = num_rows * img_height + (num_rows + 1) * self.padding

        # 创建网格画布
        grid = Image.new("RGB", (grid_width, grid_height), color=(32, 32, 32))

        # 放置图片
        for row_idx, seed in enumerate(seeds):
            for col_idx, model_name in enumerate(model_names):
                img = all_images[seed][model_name][prompt_name]

                x = self.padding + col_idx * (img_width + self.padding)
                y = self.padding + row_idx * (img_height + self.padding)

                grid.paste(img, (x, y))

        return grid

    def create_comparison_grid(
        self,
        images: Dict[str, Dict[str, Image.Image]],
        model_names: List[str],
        prompt_names: List[str],
        seed: int,
    ) -> Image.Image:
        """
        创建单个种子的对比网格

        images: {model_name: {prompt_name: Image}}
        横向：模型
        纵向：Prompt
        """
        if not images or not model_names or not prompt_names:
            raise ValueError("Empty images or names")

        # 获取单张图片尺寸
        sample_img = images[model_names[0]][prompt_names[0]]
        img_width, img_height = sample_img.size

        # 计算网格尺寸
        num_cols = len(model_names)
        num_rows = len(prompt_names)

        grid_width = num_cols * img_width + (num_cols + 1) * self.padding
        grid_height = num_rows * img_height + (num_rows + 1) * self.padding

        # 创建网格画布
        grid = Image.new("RGB", (grid_width, grid_height), color=(32, 32, 32))

        # 放置图片
        for row_idx, prompt_name in enumerate(prompt_names):
            for col_idx, model_name in enumerate(model_names):
                img = images[model_name][prompt_name]

                x = self.padding + col_idx * (img_width + self.padding)
                y = self.padding + row_idx * (img_height + self.padding)

                grid.paste(img, (x, y))

        return grid

    def create_final_grid(
        self,
        seed_grids: List[Tuple[int, Image.Image]]
    ) -> Image.Image:
        """
        将多个种子的网格拼成最终大图

        seed_grids: [(seed, grid_image), ...]
        """
        if not seed_grids:
            raise ValueError("No grids to compose")

        if len(seed_grids) == 1:
            return seed_grids[0][1]

        # 获取单个网格尺寸
        grid_width, grid_height = seed_grids[0][1].size

        # 计算最佳布局
        count = len(seed_grids)
        rows, cols = get_best_grid_layout(count)

        # 计算最终尺寸
        final_width = cols * grid_width + (cols + 1) * self.padding
        final_height = rows * grid_height + (rows + 1) * self.padding

        # 创建画布
        final_grid = Image.new("RGB", (final_width, final_height), color=(16, 16, 16))
        draw = ImageDraw.Draw(final_grid)

        # 放置网格并添加种子标签
        for idx, (seed, grid_img) in enumerate(seed_grids):
            row = idx // cols
            col = idx % cols

            x = self.padding + col * (grid_width + self.padding)
            y = self.padding + row * (grid_height + self.padding)

            final_grid.paste(grid_img, (x, y))

            # 在左上角添加种子标签
            seed_label = f"Seed: {seed}"
            if self.font:
                bbox = draw.textbbox((0, 0), seed_label, font=self.font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(seed_label) * 8
                text_height = 16

            label_x = x + 8
            label_y = y + 8
            draw.rectangle(
                [label_x - 4, label_y - 4, label_x + text_width + 4, label_y + text_height + 4],
                fill=self.label_bg_color
            )
            draw.text((label_x, label_y), seed_label, fill=self.label_text_color, font=self.font)

        return final_grid


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SDXL 批量对比生成工具")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--seeds", type=str, default=None, help="指定种子列表（逗号分隔），覆盖随机生成")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # 准备输出目录
    output_dir = Path(config["output"]["dir"])
    images_dir = output_dir / "images"
    grids_dir = output_dir / "grids"
    images_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    # 生成或解析种子
    repeat_count = config["generation"]["repeat_count"]
    seeds_by_prompt = config.get("seeds_by_prompt", None)

    if seeds_by_prompt:
        logger.info("Using per-prompt seeds from config")
        seeds = None
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        logger.info(f"Using seeds from command line: {seeds}")
    else:
        seeds = generate_random_seeds(repeat_count)
        logger.info(f"Using randomly generated seeds: {seeds}")

    # 保存种子到日志
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "seeds": seeds if seeds else "per-prompt (see seeds_by_prompt)",
        "seeds_by_prompt": seeds_by_prompt,
        "generated_images": []
    }

    # 初始化设备
    device = get_preferred_device()
    logger.info(f"Using device: {device}")

    # 初始化生成器
    generator = SDXLGenerator(config, device)
    composer = GridComposer(config)

    # 获取模型和提示词列表
    lora_models = config["lora_models"]
    prompts = config["prompts"]
    gen_config = config["generation"]

    # 批量生成大小
    batch_size = gen_config.get("batch_size", 1)  # SDXL默认batch_size=1，内存消耗较大

    model_names = [m["name"] for m in lora_models]
    prompt_names = [p["name"] for p in prompts]

    # 存储所有生成的图片
    all_images: Dict[int, Dict[str, Dict[str, Image.Image]]] = {}

    # 获取所有需要的种子
    if seeds_by_prompt:
        all_seeds = set()
        for prompt_config in prompts:
            prompt_id = prompt_config["id"]
            prompt_seeds = seeds_by_prompt.get(prompt_id, [])
            all_seeds.update(prompt_seeds)
        all_seeds = sorted(all_seeds)
        total_images = sum(len(seeds_by_prompt.get(p["id"], [])) for p in prompts) * len(lora_models)
    else:
        all_seeds = seeds
        total_images = len(lora_models) * len(prompts) * len(seeds)

    # 初始化存储结构
    for seed in all_seeds:
        all_images[seed] = {}

    logger.info(f"Total images to generate: {total_images}")
    logger.info(f"Batch size: {batch_size}")

    with tqdm(total=total_images, desc="Generating") as pbar:
        for lora_config in lora_models:
            model_name = lora_config["name"]

            # 加载 LoRA
            generator.load_lora(lora_config)

            # 创建模型目录
            model_dir = images_dir / model_name
            model_dir.mkdir(exist_ok=True)

            # 初始化模型的存储
            for seed in all_seeds:
                all_images[seed][model_name] = {}

            for prompt_config in prompts:
                prompt_name = prompt_config["name"]
                prompt_id = prompt_config["id"]
                positive = prompt_config["positive"]
                negative = prompt_config.get("negative", "")

                # 获取该 prompt 对应的种子列表
                if seeds_by_prompt:
                    current_seeds = seeds_by_prompt.get(prompt_id, [])
                    if not current_seeds:
                        logger.warning(f"No seeds found for prompt '{prompt_id}', skipping...")
                        continue
                else:
                    current_seeds = seeds

                # 创建 Prompt 目录
                prompt_dir = model_dir / prompt_name
                prompt_dir.mkdir(exist_ok=True)

                # 按 batch 处理种子
                for batch_start in range(0, len(current_seeds), batch_size):
                    batch_seeds = current_seeds[batch_start:batch_start + batch_size]

                    # 批量生成图片
                    batch_results = generator.generate_batch(
                        prompt=positive,
                        negative_prompt=negative,
                        seeds=batch_seeds,
                        width=gen_config["width"],
                        height=gen_config["height"],
                        steps=gen_config["steps"],
                        guidance_scale=gen_config["guidance_scale"],
                    )

                    # 处理批量结果
                    for seed, img in batch_results:
                        # 添加标签
                        img_labeled = composer.add_label_to_image(
                            img, model_name, prompt_name, seed
                        )

                        # 保存单张图片
                        img_path = prompt_dir / f"seed_{seed}.png"
                        img_labeled.save(img_path)

                        # 存储用于后续拼图
                        all_images[seed][model_name][prompt_name] = img_labeled

                        # 记录日志
                        log_data["generated_images"].append({
                            "model": model_name,
                            "prompt": prompt_name,
                            "seed": seed,
                            "path": str(img_path)
                        })

                        pbar.update(1)

    logger.info("All images generated. Creating grids...")

    if seeds_by_prompt:
        # Per-prompt 模式
        logger.info("Creating per-prompt comparison grids...")
        for prompt_config in tqdm(prompts, desc="Creating prompt grids"):
            prompt_id = prompt_config["id"]
            prompt_name = prompt_config["name"]
            prompt_seeds = seeds_by_prompt.get(prompt_id, [])

            if not prompt_seeds:
                continue

            grid = composer.create_prompt_grid(
                all_images,
                model_names,
                prompt_name,
                prompt_seeds
            )

            grid_path = grids_dir / f"grid_{prompt_name}.png"
            grid.save(grid_path)
            logger.info(f"Grid for {prompt_name} saved to {grid_path}")
    else:
        # 传统模式
        seed_grids = []
        for seed in tqdm(seeds, desc="Creating grids"):
            grid = composer.create_comparison_grid(
                all_images[seed],
                model_names,
                prompt_names,
                seed
            )

            grid_path = grids_dir / f"grid_seed_{seed}.png"
            grid.save(grid_path)

            seed_grids.append((seed, grid))

        # 生成最终大图
        if len(seed_grids) > 1:
            logger.info("Creating final comparison grid...")
            final_grid = composer.create_final_grid(seed_grids)
            final_path = output_dir / "final_comparison.png"
            final_grid.save(final_path)
            logger.info(f"Final grid saved to {final_path}")

    # 保存日志
    log_path = output_dir / "generation_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Generation log saved to {log_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
