#!/usr/bin/env python3
"""
训练执行器 - 精简版

功能：
1. 执行后处理（整理采样图、模型文件、生成拼图）
2. 接收命令行参数，不读取 JSON 配置

用法：
    # 只执行后处理
    python train_runner.py --post-process --output-dir /path/to/output --num-prompts 4

    # 后处理 + 生成拼图
    python train_runner.py --post-process --output-dir /path/to/output --num-prompts 4 --create-grid
"""

import argparse
import sys
from pathlib import Path

# 添加当前目录到路径，以便导入同目录下的模块
sys.path.insert(0, str(Path(__file__).parent))

from organize_outputs import organize_samples, organize_models
from create_sample_grid import create_grid_for_branch


def run_post_process(args):
    """执行后处理"""
    output_dir = Path(args.output_dir).resolve()
    sample_dir = output_dir / 'sample'

    print(f"\n{'='*60}")
    print(f"开始后处理: {output_dir}")
    print(f"{'='*60}\n")

    # 1. 整理采样图
    if sample_dir.exists():
        print(f"[1/3] 整理采样图...")
        moved_samples = organize_samples(sample_dir, args.num_prompts, verbose=not args.quiet)
        print(f"      移动了 {moved_samples} 个采样图文件")
    else:
        print(f"[1/3] 跳过采样图整理（目录不存在: {sample_dir}）")
        moved_samples = 0

    # 2. 整理模型文件
    print(f"\n[2/3] 整理模型文件...")
    moved_models = organize_models(output_dir, verbose=not args.quiet)
    print(f"      移动了 {moved_models} 个模型文件")

    # 3. 生成拼图
    if args.create_grid and sample_dir.exists():
        print(f"\n[3/3] 生成训练进度拼图...")
        grid_path = create_grid_for_branch(output_dir, args.num_prompts)
        if grid_path:
            print(f"      拼图已保存: {grid_path}")
        else:
            print(f"      未能生成拼图（可能没有足够的采样图）")
    else:
        print(f"\n[3/3] 跳过拼图生成")
        grid_path = None

    # 输出结果
    print(f"\n{'='*60}")
    print(f"后处理完成!")
    print(f"  - 采样图: {moved_samples} 个")
    print(f"  - 模型文件: {moved_models} 个")
    print(f"  - 拼图: {'已生成' if grid_path else '未生成'}")
    print(f"{'='*60}\n")

    return {
        'samples': moved_samples,
        'models': moved_models,
        'grid': grid_path is not None
    }


def main():
    parser = argparse.ArgumentParser(
        description='训练执行器 - 后处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 后处理（整理文件 + 生成拼图）
    python train_runner.py --post-process --output-dir /root/outputs/test --num-prompts 4

    # 只整理文件，不生成拼图
    python train_runner.py --post-process --output-dir /root/outputs/test --num-prompts 4 --no-grid

    # 安静模式（减少输出）
    python train_runner.py --post-process --output-dir /root/outputs/test --num-prompts 4 --quiet
        """
    )

    # 模式选择
    parser.add_argument('--post-process', action='store_true',
                        help='执行后处理（整理采样图、模型、生成拼图）')

    # 后处理参数
    parser.add_argument('--output-dir', type=str, required=True,
                        help='训练输出目录（包含 sample/ 和 *.safetensors）')
    parser.add_argument('--num-prompts', type=int, default=4,
                        help='采样提示词数量（默认: 4）')
    parser.add_argument('--create-grid', action='store_true', default=True,
                        help='生成训练进度拼图（默认: True）')
    parser.add_argument('--no-grid', action='store_true',
                        help='不生成拼图')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='安静模式，减少输出')

    args = parser.parse_args()

    # 处理 --no-grid 参数
    if args.no_grid:
        args.create_grid = False

    # 检查必需的模式
    if not args.post_process:
        print("错误: 请指定 --post-process 参数")
        parser.print_help()
        sys.exit(1)

    # 检查输出目录
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        sys.exit(1)

    # 执行后处理
    if args.post_process:
        result = run_post_process(args)
        sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
