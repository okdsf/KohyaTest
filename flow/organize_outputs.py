#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出整理工具

功能：
1. 将采样图按 prompt 索引分类到子文件夹
2. 将模型文件移动到 models 子文件夹

可单独运行，也可被 orchestrator.py 调用
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def organize_samples(sample_dir: Path, num_prompts: int = 4, verbose: bool = True) -> int:
    """
    整理采样图到按 prompt 分类的子文件夹
    
    文件名格式: {output_name}_{step}_{prompt_idx}_{timestamp}_{seed}.png
    例如: dusk_lora_test_001000_02_20251225143052_799633003.png
    
    返回移动的文件数量
    """
    if not sample_dir.exists():
        if verbose:
            print(f"采样目录不存在: {sample_dir}")
        return 0
    
    # 创建子文件夹
    for i in range(num_prompts):
        (sample_dir / f'prompt_{i+1}').mkdir(exist_ok=True)
    
    moved_count = 0
    
    # 移动文件
    for img_file in sample_dir.glob('*.png'):
        filename = img_file.name
        
        # 跳过已经在子文件夹中的文件
        if img_file.parent.name.startswith('prompt_'):
            continue
        
        # 解析 prompt 索引
        # 格式: xxx_xxxxxx_XX_timestamp_seed.png
        # 从后往前解析更可靠
        parts = filename.rsplit('_', 3)
        
        if len(parts) >= 3:
            try:
                # parts[-3] 应该是 prompt 索引（如 "00", "01", "02", "03"）
                prompt_idx_str = parts[-3]
                prompt_idx = int(prompt_idx_str)
                
                if 0 <= prompt_idx < num_prompts:
                    dest_dir = sample_dir / f'prompt_{prompt_idx + 1}'
                    dest_path = dest_dir / filename
                    
                    if verbose:
                        print(f"  {filename} -> prompt_{prompt_idx + 1}/")
                    
                    shutil.move(str(img_file), str(dest_path))
                    moved_count += 1
                else:
                    if verbose:
                        print(f"  跳过 {filename}: prompt 索引 {prompt_idx} 超出范围")
            except (ValueError, IndexError) as e:
                if verbose:
                    print(f"  跳过 {filename}: 无法解析 ({e})")
    
    return moved_count


def organize_models(output_dir: Path, verbose: bool = True) -> int:
    """
    将模型文件移动到 models 子文件夹
    
    返回移动的文件数量
    """
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    
    for model_file in output_dir.glob('*.safetensors'):
        dest_path = models_dir / model_file.name
        
        if verbose:
            print(f"  {model_file.name} -> models/")
        
        shutil.move(str(model_file), str(dest_path))
        moved_count += 1
    
    return moved_count


def organize_directory(target_dir: Path, num_prompts: int = 4, verbose: bool = True) -> None:
    """整理单个输出目录"""
    print(f"\n处理目录: {target_dir}")
    
    # 整理采样图
    sample_dir = target_dir / 'sample'
    if sample_dir.exists():
        print("整理采样图...")
        count = organize_samples(sample_dir, num_prompts, verbose)
        print(f"  移动了 {count} 个采样图")
    
    # 整理模型
    print("整理模型文件...")
    count = organize_models(target_dir, verbose)
    print(f"  移动了 {count} 个模型文件")


def organize_batch(output_root: Path, num_prompts: int = 4, verbose: bool = True) -> None:
    """
    批量整理输出目录
    
    预期目录结构:
    output_root/
    ├── Generate_Dataset1/
    │   ├── branch1/
    │   │   ├── sample/
    │   │   └── *.safetensors
    │   └── branch2/
    │       └── ...
    └── Generate_Dataset2/
        └── ...
    """
    for generate_dir in output_root.glob('Generate_*'):
        if not generate_dir.is_dir():
            continue
        
        print(f"\n{'='*50}")
        print(f"数据集: {generate_dir.name}")
        print(f"{'='*50}")
        
        for branch_dir in generate_dir.iterdir():
            if branch_dir.is_dir():
                organize_directory(branch_dir, num_prompts, verbose)


def main():
    parser = argparse.ArgumentParser(description='输出整理工具')
    parser.add_argument('target', type=str,
                        help='目标目录（可以是单个输出目录或批量输出根目录）')
    parser.add_argument('--num-prompts', type=int, default=4,
                        help='prompt 数量（默认 4）')
    parser.add_argument('--batch', action='store_true',
                        help='批量模式（遍历 Generate_* 目录）')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式（减少输出）')
    
    args = parser.parse_args()
    
    target = Path(args.target).resolve()
    verbose = not args.quiet
    
    if not target.exists():
        print(f"错误：目录不存在: {target}")
        sys.exit(1)
    
    if args.batch:
        organize_batch(target, args.num_prompts, verbose)
    else:
        organize_directory(target, args.num_prompts, verbose)
    
    print("\n整理完成！")


if __name__ == '__main__':
    main()

