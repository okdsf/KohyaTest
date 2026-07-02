#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采样图拼接工具

功能：
1. 将每个 prompt 的不同 step 采样图横向拼接
2. 将所有 prompt 纵向拼接成大图
3. 行 = Prompt，列 = Step

文件名格式: {output_name}_{step}_{prompt_idx}_{timestamp}_{seed}.png
例如: dusk_lora_test_001000_02_20251225143052_799633003.png
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


def try_load_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """尝试加载字体"""
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


def parse_sample_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    解析采样图文件名，返回 (step, prompt_idx)
    
    文件名格式: {output_name}_{step}_{prompt_idx}_{timestamp}_{seed}.png
    例如: dusk_lora_test_001000_02_20251225143052_799633003.png
    """
    # 去掉扩展名
    name = filename.rsplit('.', 1)[0]
    
    # 从后往前分割
    parts = name.rsplit('_', 3)
    
    if len(parts) >= 3:
        try:
            # parts[-3] 是 prompt_idx
            # parts[-4] 之前的部分需要进一步解析 step
            prompt_idx = int(parts[-3])
            
            # 找 step：在 prompt_idx 之前
            prefix = '_'.join(parts[:-3])
            prefix_parts = prefix.rsplit('_', 1)
            if len(prefix_parts) >= 1:
                step_str = prefix_parts[-1]
                step = int(step_str)
                return (step, prompt_idx)
        except (ValueError, IndexError):
            pass
    
    return None


def collect_sample_images(sample_dir: Path, num_prompts: int = 4) -> Dict[int, Dict[int, Path]]:
    """
    收集采样图，按 prompt 和 step 分类
    
    返回: {prompt_idx: {step: image_path}}
    """
    result: Dict[int, Dict[int, Path]] = {}
    
    # 遍历 prompt 子文件夹
    for prompt_idx in range(num_prompts):
        prompt_dir = sample_dir / f'prompt_{prompt_idx + 1}'
        if not prompt_dir.exists():
            continue
        
        result[prompt_idx] = {}
        
        for img_file in prompt_dir.glob('*.png'):
            parsed = parse_sample_filename(img_file.name)
            if parsed:
                step, parsed_prompt_idx = parsed
                # 验证 prompt_idx 是否匹配
                if parsed_prompt_idx == prompt_idx:
                    result[prompt_idx][step] = img_file
    
    return result


def create_sample_grid(
    sample_dir: Path,
    output_path: Path,
    num_prompts: int = 4,
    padding: int = 8,
    add_labels: bool = True,
    label_font_size: int = 20,
    prompt_names: Optional[List[str]] = None,
) -> Optional[Image.Image]:
    """
    创建采样图网格
    
    行 = Prompt，列 = Step
    """
    # 收集图片
    images = collect_sample_images(sample_dir, num_prompts)
    
    if not images:
        print(f"没有找到采样图: {sample_dir}")
        return None
    
    # 获取所有 steps（排序）
    all_steps = set()
    for prompt_images in images.values():
        all_steps.update(prompt_images.keys())
    
    if not all_steps:
        print("没有找到有效的采样图")
        return None
    
    steps = sorted(all_steps)
    
    # 获取可用的 prompts（排序）
    prompts = sorted(images.keys())
    
    print(f"找到 {len(prompts)} 个 prompts, {len(steps)} 个 steps")
    print(f"Steps: {steps}")
    
    # 加载第一张图获取尺寸
    sample_img_path = None
    for prompt_idx in prompts:
        if images[prompt_idx]:
            sample_img_path = list(images[prompt_idx].values())[0]
            break
    
    if not sample_img_path:
        return None
    
    sample_img = Image.open(sample_img_path)
    img_width, img_height = sample_img.size
    sample_img.close()
    
    # 计算网格尺寸
    num_cols = len(steps)
    num_rows = len(prompts)
    
    # 标签高度
    label_height = label_font_size + 10 if add_labels else 0
    header_height = label_font_size + 10 if add_labels else 0  # 列标题（step）
    
    grid_width = padding + num_cols * (img_width + padding)
    grid_height = header_height + padding + num_rows * (img_height + label_height + padding)
    
    # 如果添加行标签，需要额外宽度
    row_label_width = 150 if add_labels else 0
    grid_width += row_label_width
    
    # 创建画布
    grid = Image.new("RGB", (grid_width, grid_height), color=(32, 32, 32))
    draw = ImageDraw.Draw(grid)
    font = try_load_font(label_font_size)
    
    # 绘制列标题（Step）
    if add_labels:
        for col_idx, step in enumerate(steps):
            x = row_label_width + padding + col_idx * (img_width + padding) + img_width // 2
            y = padding
            step_label = f"Step {step}"
            
            if font:
                bbox = draw.textbbox((0, 0), step_label, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text((x - text_width // 2, y), step_label, fill=(200, 200, 200), font=font)
            else:
                draw.text((x, y), step_label, fill=(200, 200, 200))
    
    # 放置图片
    for row_idx, prompt_idx in enumerate(prompts):
        y_base = header_height + padding + row_idx * (img_height + label_height + padding)
        
        # 绘制行标签（Prompt）
        if add_labels:
            if prompt_names and prompt_idx < len(prompt_names):
                prompt_label = prompt_names[prompt_idx]
            else:
                prompt_label = f"Prompt {prompt_idx + 1}"
            
            label_y = y_base + img_height // 2
            if font:
                draw.text((padding, label_y), prompt_label, fill=(200, 200, 200), font=font)
            else:
                draw.text((padding, label_y), prompt_label, fill=(200, 200, 200))
        
        # 放置该 prompt 的所有图片
        for col_idx, step in enumerate(steps):
            x = row_label_width + padding + col_idx * (img_width + padding)
            y = y_base
            
            if step in images.get(prompt_idx, {}):
                img_path = images[prompt_idx][step]
                try:
                    img = Image.open(img_path)
                    grid.paste(img, (x, y))
                    img.close()
                except Exception as e:
                    print(f"无法加载图片 {img_path}: {e}")
                    # 绘制占位符
                    draw.rectangle([x, y, x + img_width, y + img_height], fill=(64, 64, 64))
            else:
                # 绘制空白占位符
                draw.rectangle([x, y, x + img_width, y + img_height], fill=(64, 64, 64))
                draw.text((x + 10, y + 10), "N/A", fill=(128, 128, 128), font=font)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"网格图已保存: {output_path}")
    
    return grid


def create_grid_for_branch(
    branch_dir: Path,
    num_prompts: int = 4,
    prompt_names: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    为单个训练分支创建采样图网格
    
    返回生成的网格图路径
    """
    sample_dir = branch_dir / 'sample'
    
    if not sample_dir.exists():
        print(f"采样目录不存在: {sample_dir}")
        return None
    
    output_path = sample_dir / 'training_progress_grid.png'
    
    grid = create_sample_grid(
        sample_dir=sample_dir,
        output_path=output_path,
        num_prompts=num_prompts,
        prompt_names=prompt_names,
    )
    
    if grid:
        return output_path
    return None


def main():
    parser = argparse.ArgumentParser(description='采样图拼接工具')
    parser.add_argument('sample_dir', type=str, help='采样图目录（包含 prompt_1, prompt_2 等子文件夹）')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件路径')
    parser.add_argument('--num-prompts', type=int, default=4, help='Prompt 数量（默认 4）')
    parser.add_argument('--padding', type=int, default=8, help='图片间距（默认 8）')
    parser.add_argument('--no-labels', action='store_true', help='不添加标签')
    parser.add_argument('--font-size', type=int, default=20, help='标签字体大小（默认 20）')
    
    args = parser.parse_args()
    
    sample_dir = Path(args.sample_dir).resolve()
    
    if not sample_dir.exists():
        print(f"错误：目录不存在: {sample_dir}")
        return
    
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = sample_dir / 'training_progress_grid.png'
    
    create_sample_grid(
        sample_dir=sample_dir,
        output_path=output_path,
        num_prompts=args.num_prompts,
        padding=args.padding,
        add_labels=not args.no_labels,
        label_font_size=args.font_size,
    )


if __name__ == '__main__':
    main()

