#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX Guidance Sweep 批量对比生成工具
- 支持 Guidance 参数扫描（如 3.6-4.5）
- 支持多个 LoRA 模型横向对比
- 支持多组 Prompt 和种子
- 自动生成两种对比网格图：
  1. 按 Prompt+Seed 组织（横轴 Guidance，纵轴 Model）
  2. 按 Model+Prompt 组织（横轴 Guidance，纵轴 Seed）

Usage:
    python batch_guidance_sweep.py --config /tmp/guidance_sweep_config.json
"""

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 添加 sd-scripts 到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SD_SCRIPTS_DIR = SCRIPT_DIR.parent / "sd-scripts"
sys.path.insert(0, str(SD_SCRIPTS_DIR))

from library import flux_utils, strategy_flux, device_utils
from library.device_utils import get_preferred_device
from library.utils import setup_logging, str_to_dtype
import networks.lora_flux as lora_flux
import accelerate

setup_logging()
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数
# ============================================================================

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def try_load_font(size: int) -> ImageFont.FreeTypeFont:
    """尝试加载字体，失败则使用默认字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# ============================================================================
# FLUX 推理核心
# ============================================================================

class FluxGenerator:
    """FLUX 图像生成器"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.base_model = config["base_model"]
        self.gen_config = config["generation"]
        
        # 数据类型
        self.dtype = str_to_dtype(self.gen_config.get("dtype", "bfloat16"))
        self.flux_dtype = str_to_dtype(self.gen_config.get("flux_dtype", "bfloat16"), self.dtype)
        self.offload = self.gen_config.get("offload", True)
        logger.info(f"FLUX dtype: transformer={self.flux_dtype}, text_encoders/ae={self.dtype}, offload={self.offload}")
        
        # 加载模型
        self._load_base_models()
        
        # 当前加载的 LoRA
        self.current_lora = None
        self.lora_model = None
    
    def _load_base_models(self):
        """加载基础模型"""
        loading_device = "cpu" if self.offload else self.device
        
        logger.info(f"Loading CLIP-L from {self.base_model['clip_l']}...")
        self.clip_l = flux_utils.load_clip_l(
            self.base_model["clip_l"], self.dtype, loading_device
        )
        self.clip_l.eval()
        
        logger.info(f"Loading T5XXL from {self.base_model['t5xxl']}...")
        self.t5xxl = flux_utils.load_t5xxl(
            self.base_model["t5xxl"], self.dtype, loading_device
        )
        self.t5xxl.eval()
        
        logger.info(f"Loading FLUX model from {self.base_model['ckpt_path']}...")
        self.is_schnell, self.model = flux_utils.load_flow_model(
            self.base_model["ckpt_path"], None, loading_device
        )
        self.model.eval()
        self.model.to(self.flux_dtype)
        
        logger.info(f"Loading AE from {self.base_model['ae']}...")
        self.ae = flux_utils.load_ae(
            self.base_model["ae"], self.dtype, loading_device
        )
        self.ae.eval()
        
        # 初始化 tokenizer 和 encoding strategy
        t5xxl_max_length = 256 if self.is_schnell else 512
        self.tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
        self.encoding_strategy = strategy_flux.FluxTextEncodingStrategy()
        
        # Accelerator for fp8
        self.accelerator = accelerate.Accelerator(mixed_precision="bf16")
    
    def _reload_flux_model(self):
        """重新加载 FLUX 模型（用于切换 LoRA 时确保干净状态）"""
        loading_device = "cpu" if self.offload else self.device
        
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            device_utils.clean_memory()
        
        logger.info(f"Reloading FLUX model from {self.base_model['ckpt_path']}...")
        self.is_schnell, self.model = flux_utils.load_flow_model(
            self.base_model["ckpt_path"], None, loading_device
        )
        self.model.eval()
        self.model.to(self.flux_dtype)
    
    def load_lora(self, lora_config: Optional[Dict]):
        """加载 LoRA 模型"""
        if self.lora_model is not None or self.current_lora is not None:
            logger.info("Reloading FLUX model for clean LoRA switch...")
            self.lora_model = None
            self._reload_flux_model()
            device_utils.clean_memory()
        
        if lora_config is None or lora_config.get("path") is None:
            self.current_lora = None
            logger.info("No LoRA loaded (base model only)")
            return
        
        lora_path = lora_config["path"]
        lora_weight = lora_config.get("weight", 1.0)
        
        logger.info(f"Loading LoRA: {lora_path} (weight={lora_weight})")
        
        from safetensors.torch import load_file
        weights_sd = load_file(lora_path)
        
        self.lora_model, _ = lora_flux.create_network_from_weights(
            lora_weight, None, self.ae, [self.clip_l, self.t5xxl], 
            self.model, weights_sd, True
        )
        
        self.lora_model.apply_to([self.clip_l, self.t5xxl], self.model)
        self.lora_model.load_state_dict(weights_sd, strict=True)
        self.lora_model.eval()
        self.lora_model.to(self.device)
        
        self.current_lora = lora_config
    
    def generate_batch(
        self,
        prompt: str,
        negative_prompt: str,
        seeds: List[int],
        width: int,
        height: int,
        steps: int,
        guidance: float,
        cfg_scale: float,
    ) -> List[Tuple[int, Image.Image]]:
        """批量生成多张图片（同一个 prompt，不同种子）"""
        import einops
        
        batch_size = len(seeds)
        logger.info(f"Batch generating {batch_size} images with guidance={guidance}, seeds={seeds[:3]}...")
        
        # 准备 latent
        packed_latent_height = math.ceil(height / 16)
        packed_latent_width = math.ceil(width / 16)
        
        # 为每个种子生成独立的 noise
        noise_list = []
        for seed in seeds:
            noise = torch.randn(
                1,
                packed_latent_height * packed_latent_width,
                16 * 2 * 2,
                device=self.device,
                dtype=torch.float32,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
            noise_list.append(noise)
        
        noise = torch.cat(noise_list, dim=0)
        img_ids = flux_utils.prepare_img_ids(batch_size, packed_latent_height, packed_latent_width)
        
        # 编码文本
        self.clip_l = self.clip_l.to(self.device)
        self.t5xxl = self.t5xxl.to(self.device)
        
        def encode(text: str):
            tokens_and_masks = self.tokenize_strategy.tokenize(text)
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    l_pooled, _, _, _ = self.encoding_strategy.encode_tokens(
                        self.tokenize_strategy, [self.clip_l, None], tokens_and_masks
                    )
                    _, t5_out, txt_ids, t5_attn_mask = self.encoding_strategy.encode_tokens(
                        self.tokenize_strategy, [None, self.t5xxl], tokens_and_masks,
                        self.gen_config.get("apply_t5_attn_mask", True)
                    )
            return l_pooled, t5_out, txt_ids, t5_attn_mask
        
        l_pooled, t5_out, txt_ids, t5_attn_mask = encode(prompt)
        
        # 扩展文本编码到 batch_size
        l_pooled = l_pooled.repeat(batch_size, 1)
        t5_out = t5_out.repeat(batch_size, 1, 1)
        txt_ids = txt_ids.repeat(batch_size, 1, 1)
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.repeat(batch_size, 1)
        
        # 负面提示词
        neg_l_pooled, neg_t5_out, neg_t5_attn_mask = None, None, None
        if negative_prompt and cfg_scale > 1.0:
            neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode(negative_prompt)
            neg_l_pooled = neg_l_pooled.repeat(batch_size, 1)
            neg_t5_out = neg_t5_out.repeat(batch_size, 1, 1)
            if neg_t5_attn_mask is not None:
                neg_t5_attn_mask = neg_t5_attn_mask.repeat(batch_size, 1)
        
        if self.offload:
            self.clip_l = self.clip_l.cpu()
            self.t5xxl = self.t5xxl.cpu()
        device_utils.clean_memory()
        
        # 生成
        self.model = self.model.to(self.device)

        img_ids = img_ids.to(self.device)
        t5_attn_mask = t5_attn_mask.to(self.device) if t5_attn_mask is not None else None
        l_pooled = l_pooled.to(self.device)
        t5_out = t5_out.to(self.device)
        txt_ids = txt_ids.to(self.device)
        
        if neg_l_pooled is not None:
            neg_l_pooled = neg_l_pooled.to(self.device)
        if neg_t5_out is not None:
            neg_t5_out = neg_t5_out.to(self.device)
        if neg_t5_attn_mask is not None:
            neg_t5_attn_mask = neg_t5_attn_mask.to(self.device)
        
        # 采样
        from flux_minimal_inference import do_sample
        
        x = do_sample(
            self.accelerator,
            self.model,
            noise,
            img_ids,
            l_pooled,
            t5_out,
            txt_ids,
            steps,
            guidance,
            t5_attn_mask,
            self.is_schnell,
            self.device,
            self.flux_dtype,
            neg_l_pooled,
            neg_t5_out,
            neg_t5_attn_mask,
            cfg_scale if cfg_scale > 1.0 else None,
        )
        
        if self.offload:
            self.model = self.model.cpu()
        device_utils.clean_memory()
        
        # 解码
        x = x.float()
        x = einops.rearrange(
            x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=packed_latent_height, w=packed_latent_width, ph=2, pw=2
        )
        
        self.ae = self.ae.to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                x = self.ae.decode(x)
        
        if self.offload:
            self.ae = self.ae.cpu()
        
        x = x.clamp(-1, 1)
        x = x.permute(0, 2, 3, 1)
        
        # 转换为 PIL 图片列表
        images = []
        x_np = (127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)
        for i, seed in enumerate(seeds):
            img = Image.fromarray(x_np[i])
            images.append((seed, img))
        
        return images


# ============================================================================
# 网格图生成器
# ============================================================================

class GuidanceSweepGridComposer:
    """Guidance Sweep 专用网格图生成器"""
    
    def __init__(self, config: Dict):
        grid_config = config.get("grid", {})
        self.padding = grid_config.get("grid_padding", 8)
        self.add_labels = grid_config.get("add_labels", True)
        self.label_font_size = grid_config.get("label_font_size", 24)
        self.label_bg_color = tuple(grid_config.get("label_bg_color", [0, 0, 0, 180]))
        self.label_text_color = tuple(grid_config.get("label_text_color", [255, 255, 255]))
        
        self.font = try_load_font(self.label_font_size)
        self.small_font = try_load_font(max(12, self.label_font_size - 8))
    
    def add_label_to_image(
        self,
        img: Image.Image,
        guidance: float,
        seed: int,
        model_name: str = None,
        prompt_name: str = None,
    ) -> Image.Image:
        """在图片上添加标签"""
        if not self.add_labels:
            return img
        
        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 构建标签文本
        parts = [f"g{guidance}"]
        if model_name:
            parts.append(model_name)
        parts.append(f"s{seed}")
        label_text = " | ".join(parts)
        
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
    
    def create_grid_type1(
        self,
        images: Dict[float, Dict[str, Image.Image]],
        guidance_values: List[float],
        model_names: List[str],
        prompt_name: str,
        seed: int,
    ) -> Image.Image:
        """
        类型1：按 Prompt+Seed 组织的网格
        横轴：Guidance 值
        纵轴：Model
        
        images: {guidance: {model_name: Image}}
        """
        if not images or not guidance_values or not model_names:
            raise ValueError("Empty images, guidance values, or model names")
        
        # 获取单张图片尺寸
        sample_guidance = guidance_values[0]
        sample_model = model_names[0]
        sample_img = images[sample_guidance][sample_model]
        img_width, img_height = sample_img.size
        
        num_cols = len(guidance_values)
        num_rows = len(model_names)
        
        # 标题区域高度
        header_height = 40
        row_label_width = 200
        
        grid_width = row_label_width + num_cols * img_width + (num_cols + 1) * self.padding
        grid_height = header_height + num_rows * img_height + (num_rows + 1) * self.padding
        
        grid = Image.new("RGB", (grid_width, grid_height), color=(32, 32, 32))
        draw = ImageDraw.Draw(grid)
        
        # 绘制列标题（Guidance 值）
        for col_idx, guidance in enumerate(guidance_values):
            x = row_label_width + self.padding + col_idx * (img_width + self.padding) + img_width // 2
            y = header_height // 2
            label = f"g={guidance}"
            if self.small_font:
                bbox = draw.textbbox((0, 0), label, font=self.small_font)
                text_width = bbox[2] - bbox[0]
                draw.text((x - text_width // 2, y - 8), label, fill=(255, 255, 255), font=self.small_font)
            else:
                draw.text((x - 30, y - 8), label, fill=(255, 255, 255))
        
        # 绘制行标签和图片
        for row_idx, model_name in enumerate(model_names):
            # 行标签
            y_center = header_height + self.padding + row_idx * (img_height + self.padding) + img_height // 2
            label = model_name[:25]  # 截断过长名称
            if self.small_font:
                draw.text((10, y_center - 8), label, fill=(200, 200, 200), font=self.small_font)
            else:
                draw.text((10, y_center - 8), label, fill=(200, 200, 200))
            
            # 图片
            for col_idx, guidance in enumerate(guidance_values):
                img = images[guidance][model_name]
                x = row_label_width + self.padding + col_idx * (img_width + self.padding)
                y = header_height + self.padding + row_idx * (img_height + self.padding)
                grid.paste(img, (x, y))
        
        # 添加标题
        title = f"Prompt: {prompt_name} | Seed: {seed}"
        if self.font:
            draw.text((row_label_width + 10, 5), title, fill=(255, 255, 0), font=self.small_font)
        
        return grid
    
    def create_grid_type2(
        self,
        images: Dict[float, Dict[int, Image.Image]],
        guidance_values: List[float],
        seeds: List[int],
        model_name: str,
        prompt_name: str,
    ) -> Image.Image:
        """
        类型2：按 Model+Prompt 组织的网格
        横轴：Guidance 值
        纵轴：Seed
        
        images: {guidance: {seed: Image}}
        """
        if not images or not guidance_values or not seeds:
            raise ValueError("Empty images, guidance values, or seeds")
        
        # 获取单张图片尺寸
        sample_guidance = guidance_values[0]
        sample_seed = seeds[0]
        sample_img = images[sample_guidance][sample_seed]
        img_width, img_height = sample_img.size
        
        num_cols = len(guidance_values)
        num_rows = len(seeds)
        
        # 标题区域高度
        header_height = 40
        row_label_width = 150
        
        grid_width = row_label_width + num_cols * img_width + (num_cols + 1) * self.padding
        grid_height = header_height + num_rows * img_height + (num_rows + 1) * self.padding
        
        grid = Image.new("RGB", (grid_width, grid_height), color=(32, 32, 32))
        draw = ImageDraw.Draw(grid)
        
        # 绘制列标题（Guidance 值）
        for col_idx, guidance in enumerate(guidance_values):
            x = row_label_width + self.padding + col_idx * (img_width + self.padding) + img_width // 2
            y = header_height // 2
            label = f"g={guidance}"
            if self.small_font:
                bbox = draw.textbbox((0, 0), label, font=self.small_font)
                text_width = bbox[2] - bbox[0]
                draw.text((x - text_width // 2, y - 8), label, fill=(255, 255, 255), font=self.small_font)
            else:
                draw.text((x - 30, y - 8), label, fill=(255, 255, 255))
        
        # 绘制行标签（种子）和图片
        for row_idx, seed in enumerate(seeds):
            # 行标签
            y_center = header_height + self.padding + row_idx * (img_height + self.padding) + img_height // 2
            label = f"s{seed}"
            if self.small_font:
                draw.text((10, y_center - 8), label, fill=(200, 200, 200), font=self.small_font)
            else:
                draw.text((10, y_center - 8), label, fill=(200, 200, 200))
            
            # 图片
            for col_idx, guidance in enumerate(guidance_values):
                img = images[guidance][seed]
                x = row_label_width + self.padding + col_idx * (img_width + self.padding)
                y = header_height + self.padding + row_idx * (img_height + self.padding)
                grid.paste(img, (x, y))
        
        # 添加标题
        title = f"Model: {model_name} | Prompt: {prompt_name}"
        if self.font:
            draw.text((row_label_width + 10, 5), title, fill=(255, 255, 0), font=self.small_font)
        
        return grid


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FLUX Guidance Sweep 批量对比生成工具")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # 获取配置项
    guidance_sweep = config["guidance_sweep"]
    guidance_values = guidance_sweep["values"]
    seeds_by_prompt = config["seeds_by_prompt"]
    lora_models = config["lora_models"]
    prompts = config["prompts"]
    gen_config = config["generation"]
    output_config = config["output"]
    grid_config = config.get("grid", {})
    
    base_dir = Path(output_config["base_dir"])
    server_name = output_config["server_name"]
    output_dir = base_dir / server_name
    
    model_names = [m["name"] for m in lora_models]
    prompt_names = [p["name"] for p in prompts]
    batch_size = gen_config.get("batch_size", 4)
    
    # 计算总图片数
    num_seeds = len(list(seeds_by_prompt.values())[0])
    total_images = len(lora_models) * len(prompts) * len(guidance_values) * num_seeds
    
    logger.info(f"=== Guidance Sweep Generation ===")
    logger.info(f"Server: {server_name}")
    logger.info(f"Guidance values: {guidance_values}")
    logger.info(f"Models: {len(lora_models)}, Prompts: {len(prompts)}, Seeds: {num_seeds}")
    logger.info(f"Total images to generate: {total_images}")
    
    # 创建目录结构
    grids_dir = output_dir / "grids"
    grids_type1_dir = grids_dir / "by_prompt_seed"
    grids_type2_dir = grids_dir / "by_model_prompt"
    grids_type1_dir.mkdir(parents=True, exist_ok=True)
    grids_type2_dir.mkdir(parents=True, exist_ok=True)
    
    for guidance in guidance_values:
        guidance_dir = output_dir / f"guidance_{guidance}"
        for model in lora_models:
            model_dir = guidance_dir / model["name"]
            for prompt in prompts:
                prompt_dir = model_dir / prompt["name"]
                prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化设备和生成器
    device = get_preferred_device()
    logger.info(f"Using device: {device}")
    
    generator = FluxGenerator(config, device)
    composer = GuidanceSweepGridComposer(config)
    
    # 存储所有图片用于生成网格
    # all_images[guidance][model_name][prompt_name][seed] = Image
    all_images: Dict[float, Dict[str, Dict[str, Dict[int, Image.Image]]]] = {}
    for guidance in guidance_values:
        all_images[guidance] = {}
        for model_name in model_names:
            all_images[guidance][model_name] = {}
            for prompt_name in prompt_names:
                all_images[guidance][model_name][prompt_name] = {}
    
    # 日志数据
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config_summary": {
            "guidance_values": guidance_values,
            "models": model_names,
            "prompts": prompt_names,
            "seeds_per_prompt": num_seeds,
        },
        "generated_images": []
    }
    
    # 生成图片
    # 循环顺序：Model -> Guidance -> Prompt -> Seeds (batch)
    # 这样可以减少 LoRA 切换次数
    with tqdm(total=total_images, desc="Generating") as pbar:
        for lora_config in lora_models:
            model_name = lora_config["name"]
            
            # 加载 LoRA（每个模型只加载一次）
            generator.load_lora(lora_config)
            
            for guidance in guidance_values:
                guidance_dir = output_dir / f"guidance_{guidance}"
                
                for prompt_config in prompts:
                    prompt_name = prompt_config["name"]
                    prompt_id = prompt_config["id"]
                    positive = prompt_config["positive"]
                    negative = prompt_config["negative"]
                    
                    current_seeds = seeds_by_prompt.get(prompt_id, [])
                    if not current_seeds:
                        logger.warning(f"No seeds for prompt '{prompt_id}', skipping...")
                        continue
                    
                    # 输出目录
                    img_dir = guidance_dir / model_name / prompt_name
                    
                    # 按 batch 生成
                    for batch_start in range(0, len(current_seeds), batch_size):
                        batch_seeds = current_seeds[batch_start:batch_start + batch_size]
                        
                        batch_results = generator.generate_batch(
                            prompt=positive,
                            negative_prompt=negative,
                            seeds=batch_seeds,
                            width=gen_config["width"],
                            height=gen_config["height"],
                            steps=gen_config["steps"],
                            guidance=guidance,  # 使用当前 guidance 值
                            cfg_scale=gen_config["cfg_scale"],
                        )
                        
                        for seed, img in batch_results:
                            # 添加标签
                            img_labeled = composer.add_label_to_image(
                                img, guidance, seed, model_name, prompt_name
                            )
                            
                            # 保存单张图片
                            img_path = img_dir / f"seed_{seed}.png"
                            img_labeled.save(img_path)
                            
                            # 存储用于网格
                            all_images[guidance][model_name][prompt_name][seed] = img_labeled
                            
                            # 记录日志
                            log_data["generated_images"].append({
                                "guidance": guidance,
                                "model": model_name,
                                "prompt": prompt_name,
                                "seed": seed,
                                "path": str(img_path)
                            })
                            
                            pbar.update(1)
    
    logger.info("All images generated. Creating grids...")
    
    # 生成网格图类型1：按 Prompt+Seed 组织（横轴 Guidance，纵轴 Model）
    logger.info("Creating Type 1 grids (by Prompt+Seed)...")
    for prompt_config in tqdm(prompts, desc="Type 1 grids"):
        prompt_name = prompt_config["name"]
        prompt_id = prompt_config["id"]
        current_seeds = seeds_by_prompt.get(prompt_id, [])
        
        for seed in current_seeds:
            # 收集该 prompt+seed 下所有 guidance x model 的图片
            grid_images = {}  # {guidance: {model_name: Image}}
            for guidance in guidance_values:
                grid_images[guidance] = {}
                for model_name in model_names:
                    if seed in all_images[guidance][model_name][prompt_name]:
                        grid_images[guidance][model_name] = all_images[guidance][model_name][prompt_name][seed]
            
            # 生成网格
            try:
                grid = composer.create_grid_type1(
                    grid_images, guidance_values, model_names, prompt_name, seed
                )
                grid_path = grids_type1_dir / f"{prompt_name}_seed{seed}_grid.png"
                grid.save(grid_path)
            except Exception as e:
                logger.warning(f"Failed to create Type 1 grid for {prompt_name} seed {seed}: {e}")
    
    # 生成网格图类型2：按 Model+Prompt 组织（横轴 Guidance，纵轴 Seed）
    logger.info("Creating Type 2 grids (by Model+Prompt)...")
    for lora_config in tqdm(lora_models, desc="Type 2 grids"):
        model_name = lora_config["name"]
        
        for prompt_config in prompts:
            prompt_name = prompt_config["name"]
            prompt_id = prompt_config["id"]
            current_seeds = seeds_by_prompt.get(prompt_id, [])
            
            # 收集该 model+prompt 下所有 guidance x seed 的图片
            grid_images = {}  # {guidance: {seed: Image}}
            for guidance in guidance_values:
                grid_images[guidance] = {}
                for seed in current_seeds:
                    if seed in all_images[guidance][model_name][prompt_name]:
                        grid_images[guidance][seed] = all_images[guidance][model_name][prompt_name][seed]
            
            # 生成网格
            try:
                grid = composer.create_grid_type2(
                    grid_images, guidance_values, current_seeds, model_name, prompt_name
                )
                grid_path = grids_type2_dir / f"{model_name}_{prompt_name}_grid.png"
                grid.save(grid_path)
            except Exception as e:
                logger.warning(f"Failed to create Type 2 grid for {model_name} {prompt_name}: {e}")
    
    # 保存日志
    log_path = output_dir / "generation_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Generation log saved to {log_path}")
    logger.info(f"Type 1 grids saved to {grids_type1_dir}")
    logger.info(f"Type 2 grids saved to {grids_type2_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
