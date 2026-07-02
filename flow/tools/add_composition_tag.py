#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注文件构图标签添加工具

功能：
根据图片文件名后缀，自动在对应的 txt 标注文件开头添加构图标签：
- 文件名以 _halfbody 结尾 → 添加 "halfbody, "
- 文件名以 _head 结尾 → 添加 "head, "
- 其他情况 → 添加 "full, "

用法：
python add_composition_tag.py /path/to/dataset
python add_composition_tag.py /path/to/dataset --dry-run  # 只预览不修改
"""

import os
import sys
import argparse
from pathlib import Path


def get_composition_tag(filename_stem: str) -> str:
    """
    根据文件名（不含扩展名）判断构图标签
    
    Args:
        filename_stem: 文件名（不含 .txt 扩展名）
    
    Returns:
        构图标签字符串
    """
    if filename_stem.endswith('_halfbody'):
        return 'halfbody'
    elif filename_stem.endswith('_head'):
        return 'head'
    else:
        return 'full'


def process_txt_file(txt_path: Path, dry_run: bool = False) -> bool:
    """
    处理单个 txt 文件，在开头添加构图标签
    
    Args:
        txt_path: txt 文件路径
        dry_run: 如果为 True，只打印不修改
    
    Returns:
        是否成功处理
    """
    # 获取文件名（不含扩展名）
    filename_stem = txt_path.stem
    
    # 判断构图标签
    tag = get_composition_tag(filename_stem)
    
    try:
        # 读取原内容
        with open(txt_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 检查是否已经有构图标签（避免重复添加）
        if original_content.startswith(('halfbody,', 'head,', 'full,')):
            print(f"  跳过 {txt_path.name}: 已有构图标签")
            return False
        
        # 构建新内容
        new_content = f"{tag}, {original_content}"
        
        if dry_run:
            print(f"  [DRY RUN] {txt_path.name}: 添加 '{tag}' 标签")
        else:
            # 写入新内容
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  {txt_path.name}: 添加 '{tag}' 标签")
        
        return True
        
    except Exception as e:
        print(f"  错误 {txt_path.name}: {e}")
        return False


def process_directory(root_dir: Path, dry_run: bool = False) -> dict:
    """
    递归处理目录下的所有 txt 文件
    
    Args:
        root_dir: 根目录
        dry_run: 如果为 True，只打印不修改
    
    Returns:
        统计信息字典
    """
    stats = {
        'total': 0,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'by_tag': {'halfbody': 0, 'head': 0, 'full': 0}
    }
    
    # 递归遍历所有 txt 文件
    for txt_path in root_dir.rglob('*.txt'):
        stats['total'] += 1
        
        # 获取相对路径用于显示
        rel_path = txt_path.relative_to(root_dir)
        parent_dir = rel_path.parent
        
        # 打印当前处理的目录（仅在目录变化时）
        if not hasattr(process_directory, '_last_dir') or process_directory._last_dir != parent_dir:
            process_directory._last_dir = parent_dir
            print(f"\n📁 {parent_dir}/")
        
        # 处理文件
        tag = get_composition_tag(txt_path.stem)
        success = process_txt_file(txt_path, dry_run)
        
        if success:
            stats['processed'] += 1
            stats['by_tag'][tag] += 1
        else:
            stats['skipped'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='根据文件名后缀在标注文件开头添加构图标签',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python add_composition_tag.py /path/to/dataset
  python add_composition_tag.py /path/to/dataset --dry-run

文件名规则：
  image_001_halfbody.txt  → 添加 "halfbody, " 到开头
  image_002_head.txt      → 添加 "head, " 到开头
  image_003.txt           → 添加 "full, " 到开头
        """
    )
    parser.add_argument('directory', type=str,
                        help='要处理的数据集目录')
    parser.add_argument('--dry-run', action='store_true',
                        help='只预览不实际修改文件')
    
    args = parser.parse_args()
    
    root_dir = Path(args.directory).resolve()
    
    if not root_dir.exists():
        print(f"错误：目录不存在: {root_dir}")
        sys.exit(1)
    
    if not root_dir.is_dir():
        print(f"错误：不是目录: {root_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("标注文件构图标签添加工具")
    print("=" * 60)
    print(f"目标目录: {root_dir}")
    if args.dry_run:
        print("模式: DRY RUN（只预览不修改）")
    print("=" * 60)
    
    # 处理目录
    stats = process_directory(root_dir, args.dry_run)
    
    # 打印统计
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总文件数: {stats['total']}")
    print(f"已处理: {stats['processed']}")
    print(f"已跳过: {stats['skipped']}")
    print(f"\n标签统计:")
    print(f"  halfbody: {stats['by_tag']['halfbody']}")
    print(f"  head: {stats['by_tag']['head']}")
    print(f"  full: {stats['by_tag']['full']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

