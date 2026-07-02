#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX 批量对比生成工具
- 支持多个 LoRA 模型横向对比
- 支持多组 Prompt 纵向对比
- 自动拼接网格大图

Usage:
    python batch_compare_generate.py --config configs/batch_compare_config.json
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
        
        # 清理旧模型
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
        # 每次切换 LoRA 时都重新加载 FLUX 模型，确保干净状态
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
        logger.info(f"Batch generating {batch_size} images with seeds={seeds}, size={width}x{height}")
        
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
        
        # 合并成 batch
        noise = torch.cat(noise_list, dim=0)  # [batch_size, h*w, c]
        
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
            # 扩展到 batch_size
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
        
        # 移动所有 tensor 到设备
        img_ids = img_ids.to(self.device)
        t5_attn_mask = t5_attn_mask.to(self.device) if t5_attn_mask is not None else None
        l_pooled = l_pooled.to(self.device)
        t5_out = t5_out.to(self.device)
        txt_ids = txt_ids.to(self.device)
        
        # 负面提示词 tensors 也需要移到设备
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
    parser = argparse.ArgumentParser(description="FLUX 批量对比生成工具")
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
    # 支持两种模式：
    # 1. 传统模式：所有 prompt 共用同一组种子
    # 2. Per-prompt 模式：每个 prompt 使用独立的种子列表（通过配置文件的 seeds_by_prompt 字段）
    repeat_count = config["generation"]["repeat_count"]
    seeds_by_prompt = config.get("seeds_by_prompt", None)
    
    if seeds_by_prompt:
        # Per-prompt 模式：从配置文件读取每个 prompt 的种子
        logger.info("Using per-prompt seeds from config")
        seeds = None  # 将在循环中按 prompt 获取
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
    generator = FluxGenerator(config, device)
    composer = GridComposer(config)
    
    # 获取模型和提示词列表
    lora_models = config["lora_models"]
    prompts = config["prompts"]
    gen_config = config["generation"]
    
    # 批量生成大小（默认为 4）
    batch_size = gen_config.get("batch_size", 4)
    
    model_names = [m["name"] for m in lora_models]
    prompt_names = [p["name"] for p in prompts]
    
    # 存储所有生成的图片
    # all_images[seed][model_name][prompt_name] = Image
    all_images: Dict[int, Dict[str, Dict[str, Image.Image]]] = {}
    
    # 获取所有需要的种子（用于初始化存储结构）
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
    logger.info(f"Batch size: {batch_size} (generating up to {batch_size} images per batch)")
    
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
                negative = prompt_config["negative"]
                
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
                        guidance=gen_config["guidance"],
                        cfg_scale=gen_config["cfg_scale"],
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
        # Per-prompt 模式：为每个 prompt 生成独立的网格（横向模型，纵向种子）
        logger.info("Creating per-prompt comparison grids...")
        for prompt_config in tqdm(prompts, desc="Creating prompt grids"):
            prompt_id = prompt_config["id"]
            prompt_name = prompt_config["name"]
            prompt_seeds = seeds_by_prompt.get(prompt_id, [])
            
            if not prompt_seeds:
                continue
            
            # 为该 prompt 创建网格：横向是模型，纵向是种子
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
        # 传统模式：为每个种子生成网格（横向模型，纵向 prompt）
        seed_grids = []
        for seed in tqdm(seeds, desc="Creating grids"):
            grid = composer.create_comparison_grid(
                all_images[seed],
                model_names,
                prompt_names,
                seed
            )
            
            # 保存单个种子的网格
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

