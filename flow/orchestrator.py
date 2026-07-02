#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量训练管理器 (Orchestrator)

功能：
1. 遍历数据集目录，对每个子文件夹执行多个训练分支
2. 支持局部配置覆盖（override.json）
3. 动态生成输出目录结构
4. 训练后整理采样图到分类文件夹
5. 生成训练进度拼图（行=Prompt，列=Step）
6. 完整的日志记录和训练报告
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# 添加当前目录到 sys.path 以便导入同目录模块
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 导入拼图功能
try:
    from create_sample_grid import create_grid_for_branch
    GRID_AVAILABLE = True
except ImportError as e:
    GRID_AVAILABLE = False
    create_grid_for_branch = None


# ============================================================
# 日志系统
# ============================================================

class LogManager:
    """日志管理器 - 同时输出到终端和文件"""
    
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.log_file = output_root / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.report_file = output_root / "training_report.txt"
        
        # 确保目录存在
        output_root.mkdir(parents=True, exist_ok=True)
        
        # 配置 logging
        self.logger = logging.getLogger('orchestrator')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的 handlers
        self.logger.handlers.clear()
        
        # 文件 handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 终端 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def exception(self, msg: str):
        self.logger.exception(msg)


# ============================================================
# 任务状态追踪
# ============================================================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # 训练可能失败，但有输出


@dataclass
class TaskResult:
    """单个训练任务的结果"""
    dataset: str
    branch: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    sample_count: int = 0
    model_count: int = 0
    grid_generated: bool = False
    
    @property
    def duration(self) -> str:
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "N/A"


@dataclass
class TrainingReport:
    """训练报告"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    tasks: List[TaskResult] = field(default_factory=list)
    
    def add_task(self, task: TaskResult):
        self.tasks.append(task)
    
    def generate_report(self) -> str:
        self.end_time = datetime.now()
        
        lines = []
        lines.append("=" * 70)
        lines.append("训练报告")
        lines.append("=" * 70)
        lines.append(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"结束时间: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_duration = self.end_time - self.start_time
        hours, remainder = divmod(int(total_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        lines.append(f"总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
        lines.append("")
        
        # 统计
        success = sum(1 for t in self.tasks if t.status == TaskStatus.SUCCESS)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        partial = sum(1 for t in self.tasks if t.status == TaskStatus.PARTIAL)
        
        lines.append(f"任务统计:")
        lines.append(f"  总计: {len(self.tasks)}")
        lines.append(f"  成功: {success}")
        lines.append(f"  部分成功: {partial}")
        lines.append(f"  失败: {failed}")
        lines.append("")
        
        # 详细任务列表
        lines.append("-" * 70)
        lines.append("任务详情:")
        lines.append("-" * 70)
        
        for task in self.tasks:
            status_icon = {
                TaskStatus.SUCCESS: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.PARTIAL: "⚠️",
                TaskStatus.PENDING: "⏳",
                TaskStatus.RUNNING: "🔄",
            }.get(task.status, "?")
            
            lines.append(f"\n{status_icon} {task.dataset} / {task.branch}")
            lines.append(f"   状态: {task.status.value}")
            lines.append(f"   耗时: {task.duration}")
            lines.append(f"   采样图: {task.sample_count} 张")
            lines.append(f"   模型: {task.model_count} 个")
            lines.append(f"   拼图: {'已生成' if task.grid_generated else '未生成'}")
            
            if task.error_message:
                lines.append(f"   错误: {task.error_message}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())


# ============================================================
# 配置加载与合并
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    """加载 JSON 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典，override 覆盖 base"""
    result = base.copy()
    for key, value in override.items():
        if key.startswith('_'):
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_branch_config(config_dir: Path, branch_file: str) -> Dict[str, Any]:
    """加载分支配置并合并基础配置"""
    branch_path = config_dir / 'branches' / branch_file
    branch_config = load_json(branch_path)
    
    base_name = branch_config.get('_base', '')
    if base_name:
        base_path = config_dir / base_name
        base_config = load_json(base_path)
        merged = deep_merge(base_config, branch_config)
    else:
        merged = branch_config.copy()
    
    merged['_script'] = branch_config.get('_script', 'sdxl_train_network.py')
    merged['_branch_id'] = branch_config.get('_branch_id', '')
    merged['_base'] = base_name
    
    return merged


def apply_override(config: Dict, override: Dict, branch_id: str) -> Dict:
    """应用文件夹级别的覆盖配置"""
    result = config.copy()
    
    global_overrides = override.get('global_overrides', {})
    result = deep_merge(result, global_overrides)
    
    branch_overrides = override.get('branch_overrides', {})
    if branch_id in branch_overrides:
        result = deep_merge(result, branch_overrides[branch_id])
    
    return result


# ============================================================
# 采样提示词生成
# ============================================================

def generate_sample_prompts_file(
    prompts_config: Dict, 
    output_path: Path,
    trigger_word: Optional[str] = None
) -> None:
    """根据配置生成采样提示词文件"""
    placeholder = prompts_config.get('trigger_word_placeholder', '{trigger_word}')
    
    lines = []
    for prompt in prompts_config.get('prompts', []):
        positive = prompt['positive']
        negative = prompt['negative']
        
        if trigger_word:
            positive = positive.replace(placeholder, trigger_word)
            negative = negative.replace(placeholder, trigger_word)
        
        line = (
            f"{positive}, "
            f"--n {negative}, "
            f"--s {prompt['steps']} "
            f"--ss {prompt['sampler']} "
            f"--l {prompt['cfg_scale']} "
            f"--d {prompt['seed']} "
            f"--w {prompt['width']} "
            f"--h {prompt['height']}"
        )
        lines.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================
# 测试模式：生成假数据
# ============================================================

def generate_test_sample_image(
    output_path: Path, 
    step: int, 
    prompt_idx: int,
    output_name: str,
    prompt_name: str = "",
    width: int = 256,
    height: int = 384
) -> Path:
    """生成测试用的假采样图"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # 如果没有 PIL，创建一个空文件
        output_path.touch()
        return output_path
    
    # 创建带有信息的测试图
    colors = [
        (255, 100, 100),  # 红
        (100, 255, 100),  # 绿
        (100, 100, 255),  # 蓝
        (255, 255, 100),  # 黄
    ]
    bg_color = colors[prompt_idx % len(colors)]
    
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # 添加文字信息
    text_lines = [
        f"TEST IMAGE",
        f"Step: {step}",
        f"Prompt: {prompt_idx}",
        f"{prompt_name}",
        f"{output_name}",
    ]
    
    y = 20
    for line in text_lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 30
    
    # 添加边框
    draw.rectangle([0, 0, width-1, height-1], outline=(0, 0, 0), width=3)
    
    img.save(output_path)
    return output_path


def generate_test_model(output_path: Path) -> Path:
    """生成测试用的假模型文件"""
    # 创建一个最小的 safetensors 文件
    # safetensors 格式需要特定的头部，这里创建一个简单的空文件
    output_path.write_bytes(b'\x00' * 100)
    return output_path


def run_test_mode(
    branch_output_dir: Path,
    output_name: str,
    prompts_config: Dict,
    test_steps: List[int],
    log_file: Path,
    logger: LogManager
) -> Tuple[bool, Optional[str]]:
    """
    测试模式：生成假的采样图和模型
    
    返回: (success, error_message)
    """
    import time
    import random
    
    sample_dir = branch_output_dir / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = prompts_config.get('prompts', [])
    num_prompts = len(prompts)
    
    # 写入分支日志文件
    log_lines = []
    log_lines.append(f"=" * 60)
    log_lines.append(f"测试模式日志")
    log_lines.append(f"开始时间: {datetime.now()}")
    log_lines.append(f"输出目录: {branch_output_dir}")
    log_lines.append(f"输出名称: {output_name}")
    log_lines.append(f"测试步数: {test_steps}")
    log_lines.append(f"提示词数量: {num_prompts}")
    log_lines.append(f"=" * 60)
    log_lines.append("")
    
    logger.info(f"🧪 测试模式：生成 {len(test_steps)} 个步数 × {num_prompts} 个提示词 = {len(test_steps) * num_prompts} 张假采样图")
    log_lines.append(f"生成 {len(test_steps)} 个步数 × {num_prompts} 个提示词 = {len(test_steps) * num_prompts} 张假采样图")
    
    try:
        # 生成采样图
        for step in test_steps:
            for prompt_idx, prompt in enumerate(prompts):
                prompt_name = prompt.get('name', f'prompt_{prompt_idx}')
                seed = prompt.get('seed', random.randint(100000, 999999))
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                
                # 文件名格式: {output_name}_{step:06d}_{prompt_idx:02d}_{timestamp}_{seed}.png
                filename = f"{output_name}_{step:06d}_{prompt_idx:02d}_{timestamp}_{seed}.png"
                img_path = sample_dir / filename
                
                generate_test_sample_image(
                    img_path, step, prompt_idx, output_name, prompt_name,
                    width=256, height=384
                )
                log_lines.append(f"  生成: {filename}")
            
            logger.info(f"  生成 step {step} 的采样图...")
            log_lines.append(f"Step {step} 完成")
            time.sleep(0.1)  # 模拟一点延迟
        
        # 生成模型文件
        logger.info(f"🧪 测试模式：生成 {len(test_steps)} 个假模型")
        log_lines.append(f"\n生成 {len(test_steps)} 个假模型:")
        for step in test_steps:
            model_filename = f"{output_name}-step{step:08d}.safetensors"
            model_path = branch_output_dir / model_filename
            generate_test_model(model_path)
            log_lines.append(f"  生成: {model_filename}")
        
        logger.info("🧪 测试数据生成完成")
        log_lines.append(f"\n结束时间: {datetime.now()}")
        log_lines.append("状态: 成功")
        
        # 写入日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines))
        
        logger.info(f"📄 分支日志已保存: {log_file}")
        
        return True, None
        
    except Exception as e:
        error_msg = f"测试模式生成失败: {str(e)}"
        logger.error(error_msg)
        log_lines.append(f"\n错误: {error_msg}")
        log_lines.append(f"结束时间: {datetime.now()}")
        log_lines.append("状态: 失败")
        
        # 写入日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines))
        
        return False, error_msg


# ============================================================
# 命令行参数构建
# ============================================================

def build_command_args(config: Dict, train_data_dir: Path, output_dir: Path,
                       output_name: str, sample_prompts_path: Path) -> List[str]:
    """将配置转换为命令行参数列表"""
    args = []
    skip_keys = {'_description', '_model_type', '_branch_id', '_base', '_script'}
    
    for key, value in config.items():
        if key in skip_keys:
            continue
        
        arg_name = f"--{key}"
        
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        elif isinstance(value, list):
            args.append(arg_name)
            for item in value:
                args.append(str(item))
        else:
            args.append(arg_name)
            args.append(str(value))
    
    args.extend([
        '--train_data_dir', str(train_data_dir),
        '--output_dir', str(output_dir),
        '--logging_dir', str(output_dir / 'logs'),
        '--output_name', output_name,
        '--sample_prompts', str(sample_prompts_path),
    ])
    
    return args


# ============================================================
# 输出整理
# ============================================================

def organize_samples(sample_dir: Path, num_prompts: int = 4, logger: Optional[LogManager] = None) -> int:
    """整理采样图到按 prompt 分类的子文件夹，返回移动的文件数"""
    if not sample_dir.exists():
        return 0
    
    # 创建子文件夹
    for i in range(num_prompts):
        (sample_dir / f'prompt_{i+1}').mkdir(exist_ok=True)
    
    moved_count = 0
    for img_file in sample_dir.glob('*.png'):
        filename = img_file.name
        parts = filename.rsplit('_', 3)
        if len(parts) >= 3:
            try:
                prompt_idx = int(parts[-3])
                if 0 <= prompt_idx < num_prompts:
                    dest_dir = sample_dir / f'prompt_{prompt_idx + 1}'
                    shutil.move(str(img_file), str(dest_dir / filename))
                    moved_count += 1
            except (ValueError, IndexError) as e:
                if logger:
                    logger.debug(f"无法解析文件名 {filename}: {e}")
    
    return moved_count


def organize_models(output_dir: Path) -> int:
    """将模型文件移动到 models 子文件夹，返回移动的文件数"""
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    for model_file in output_dir.glob('*.safetensors'):
        shutil.move(str(model_file), str(models_dir / model_file.name))
        moved_count += 1
    
    return moved_count


def count_samples(sample_dir: Path) -> int:
    """统计采样图数量"""
    if not sample_dir.exists():
        return 0
    
    # 统计根目录和子文件夹中的图片
    count = len(list(sample_dir.glob('*.png')))
    for sub_dir in sample_dir.glob('prompt_*'):
        count += len(list(sub_dir.glob('*.png')))
    
    return count


def count_models(output_dir: Path) -> int:
    """统计模型数量"""
    count = len(list(output_dir.glob('*.safetensors')))
    models_dir = output_dir / 'models'
    if models_dir.exists():
        count += len(list(models_dir.glob('*.safetensors')))
    return count


# ============================================================
# 主流程
# ============================================================

def discover_datasets(datasets_root: Path) -> List[Path]:
    """发现所有数据集文件夹"""
    datasets = []
    for item in datasets_root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            datasets.append(item)
    return sorted(datasets)


def discover_branches(config_dir: Path, model_type: Optional[str] = None) -> List[str]:
    """发现所有分支配置文件"""
    branches_dir = config_dir / 'branches'
    branches = []
    
    for f in branches_dir.glob('*.json'):
        if model_type:
            config = load_json(f)
            base = config.get('_base', '')
            if model_type == 'sdxl' and 'sdxl' not in base:
                continue
            if model_type == 'flux' and 'flux' not in base:
                continue
        branches.append(f.name)
    
    return sorted(branches)


def run_training(script_path: Path, args: List[str], log_file: Path, 
                 logger: LogManager, dry_run: bool = False) -> Tuple[bool, Optional[str]]:
    """
    执行训练脚本，使用 script 命令创建伪终端
    同时输出到终端和日志文件，完美保留所有格式
    
    返回: (success, error_message)
    """
    cmd = ['accelerate', 'launch', str(script_path)] + args
    
    logger.info(f"\n{'='*60}")
    logger.info(f"执行命令：")
    logger.debug(f"完整命令: {' '.join(cmd)}")
    
    # 简化显示
    logger.info(f"  accelerate launch {script_path.name}")
    logger.info(f"{'='*60}\n")
    
    if dry_run:
        logger.info("[DRY RUN] 跳过实际执行")
        return True, None
    
    start_time = datetime.now()
    
    # 写入日志头部
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"命令: {' '.join(cmd)}\n")
        f.write(f"开始时间: {start_time}\n")
        f.write("=" * 60 + "\n\n")
    
    try:
        # 使用 script 命令创建伪终端
        # script 会让程序认为自己在真正的终端中运行
        # 所有输出（包括颜色、进度条）都会正常显示并保存到文件
        inner_cmd = ' '.join([f'"{c}"' if ' ' in c else c for c in cmd])
        # -q: 安静模式，不显示 script 自己的消息
        # -a: 追加模式
        # -c: 执行命令
        shell_cmd = f'script -q -a "{log_file}" -c "{inner_cmd}"'
        
        result = subprocess.run(
            shell_cmd,
            shell=True,
            executable='/bin/bash'
        )
        
        # 写入日志尾部
        end_time = datetime.now()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"结束时间: {end_time}\n")
            f.write(f"耗时: {end_time - start_time}\n")
            f.write(f"退出代码: {result.returncode}\n")
        
        if result.returncode == 0:
            return True, None
        else:
            return False, f"训练进程退出代码: {result.returncode}"
            
    except Exception as e:
        error_msg = f"训练执行异常: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n错误: {error_msg}\n")
        
        return False, error_msg


def process_task_output(branch_output_dir: Path, num_prompts: int, 
                        prompts_config: Dict, logger: LogManager) -> Tuple[int, int, bool]:
    """
    处理训练输出：整理文件并生成拼图
    
    返回: (sample_count, model_count, grid_generated)
    """
    sample_dir = branch_output_dir / 'sample'
    
    # 统计并整理采样图
    sample_count = count_samples(sample_dir)
    if sample_count > 0:
        logger.info(f"发现 {sample_count} 张采样图，开始整理...")
        moved = organize_samples(sample_dir, num_prompts, logger)
        logger.info(f"已移动 {moved} 张图片到子文件夹")
    
    # 整理模型
    model_count = count_models(branch_output_dir)
    if model_count > 0:
        moved_models = organize_models(branch_output_dir)
        logger.info(f"已移动 {moved_models} 个模型文件")
    
    # 生成拼图
    grid_generated = False
    if GRID_AVAILABLE and sample_count > 0:
        logger.info("生成训练进度拼图...")
        try:
            prompt_names = [p.get('name', f'Prompt {i+1}') for i, p in enumerate(prompts_config.get('prompts', []))]
            grid_path = create_grid_for_branch(
                branch_output_dir,
                num_prompts=num_prompts,
                prompt_names=prompt_names,
            )
            if grid_path:
                logger.info(f"拼图已保存: {grid_path}")
                grid_generated = True
        except Exception as e:
            logger.error(f"生成拼图失败: {e}")
    elif not GRID_AVAILABLE:
        logger.warning("拼图模块不可用，跳过")
    
    return sample_count, model_count, grid_generated


def main():
    parser = argparse.ArgumentParser(description='批量训练管理器')
    parser.add_argument('--datasets-root', type=str, required=True,
                        help='数据集根目录（包含多个子文件夹）')
    parser.add_argument('--output-root', type=str, required=True,
                        help='输出根目录')
    parser.add_argument('--config-dir', type=str, default=None,
                        help='配置目录（默认为脚本同目录下的 configs）')
    parser.add_argument('--kohya-root', type=str, default='/root/kohya_ss',
                        help='kohya_ss 安装目录')
    parser.add_argument('--model-type', type=str, choices=['sdxl', 'flux', 'all'],
                        default='all', help='模型类型筛选')
    parser.add_argument('--branches', type=str, nargs='+', default=None,
                        help='指定要执行的分支（默认执行全部）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印命令，不实际执行')
    parser.add_argument('--skip-organize', action='store_true',
                        help='跳过输出整理步骤')
    parser.add_argument('--validate', action='store_true',
                        help='验证配置链条（不执行训练）')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式：生成假数据验证整个流程')
    parser.add_argument('--test-steps', type=int, nargs='+', default=[500, 1000, 1500, 2000],
                        help='测试模式下模拟的训练步数（默认：500 1000 1500 2000）')
    
    args = parser.parse_args()
    
    # 路径设置
    datasets_root = Path(args.datasets_root).resolve()
    output_root = Path(args.output_root).resolve()
    kohya_root = Path(args.kohya_root).resolve()
    
    if args.config_dir:
        config_dir = Path(args.config_dir).resolve()
    else:
        config_dir = Path(__file__).parent / 'configs'
    
    # 验证路径
    if not datasets_root.exists():
        print(f"错误：数据集目录不存在: {datasets_root}")
        sys.exit(1)
    
    if not config_dir.exists():
        print(f"错误：配置目录不存在: {config_dir}")
        sys.exit(1)
    
    # 创建输出根目录
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志系统
    logger = LogManager(output_root)
    report = TrainingReport()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"批量训练管理器")
    if args.test_mode:
        logger.info(f"🧪🧪🧪 测试模式 🧪🧪🧪")
    logger.info(f"{'='*60}")
    logger.info(f"日志文件: {logger.log_file}")
    
    # 加载采样提示词配置
    try:
        prompts_configs = {
            'sdxl': load_json(config_dir / 'sample_prompts.json'),
            'flux': load_json(config_dir / 'sample_prompts_flux.json'),
        }
    except Exception as e:
        logger.error(f"加载提示词配置失败: {e}")
        sys.exit(1)
    
    def get_prompts_config_for_branch(branch_config: Dict) -> Tuple[Dict, str]:
        base = branch_config.get('_base', '')
        if 'flux' in base.lower():
            return prompts_configs['flux'], 'flux'
        else:
            return prompts_configs['sdxl'], 'sdxl'
    
    # 发现数据集和分支
    datasets = discover_datasets(datasets_root)
    
    model_type_filter = None if args.model_type == 'all' else args.model_type
    if args.branches:
        branches = [b if b.endswith('.json') else f'{b}.json' for b in args.branches]
    else:
        branches = discover_branches(config_dir, model_type_filter)
    
    logger.info(f"数据集目录: {datasets_root}")
    logger.info(f"输出目录: {output_root}")
    logger.info(f"配置目录: {config_dir}")
    logger.info(f"发现 {len(datasets)} 个数据集:")
    for ds in datasets:
        logger.info(f"  - {ds.name}")
    logger.info(f"将执行 {len(branches)} 个训练分支:")
    for b in branches:
        logger.info(f"  - {b}")
    logger.info(f"{'='*60}\n")
    
    # ============================================================
    # 验证模式
    # ============================================================
    if args.validate:
        logger.info("=" * 60)
        logger.info("🔍 验证模式：检查配置链条")
        logger.info("=" * 60)
        
        all_valid = True
        validation_errors = []
        
        for branch_file in branches:
            branch_id = branch_file.replace('.json', '')
            logger.info(f"\n--- 验证分支: {branch_id} ---")
            
            try:
                config = load_branch_config(config_dir, branch_file)
                logger.info(f"  ✅ 分支配置加载成功")
                
                base = config.get('_base', '')
                if base:
                    logger.info(f"  ✅ _base 字段: {base}")
                else:
                    logger.info(f"  ⚠️ 未设置 _base 字段")
                
                prompts_config, model_type = get_prompts_config_for_branch(config)
                logger.info(f"  ✅ 模型类型: {model_type}")
                
                num_prompts = len(prompts_config.get('prompts', []))
                logger.info(f"  ✅ 提示词数量: {num_prompts}")
                
                if num_prompts == 0:
                    validation_errors.append(f"{branch_id}: 提示词数量为 0")
                    all_valid = False
                
                script = config.get('_script', 'sdxl_train_network.py')
                logger.info(f"  ✅ 训练脚本: {script}")
                
                max_train_steps = config.get('max_train_steps', 'N/A')
                logger.info(f"  ✅ max_train_steps: {max_train_steps}")
                
                sample_every_n_steps = config.get('sample_every_n_steps', 'N/A')
                logger.info(f"  ✅ sample_every_n_steps: {sample_every_n_steps}")
                
                logger.info(f"  提示词预览:")
                for i, prompt in enumerate(prompts_config.get('prompts', [])[:2]):
                    name = prompt.get('name', f'prompt_{i+1}')
                    positive = prompt.get('positive', '')[:60] + '...'
                    logger.info(f"    - {name}: {positive}")
                
            except Exception as e:
                logger.error(f"  ❌ 错误: {e}")
                validation_errors.append(f"{branch_id}: {e}")
                all_valid = False
        
        logger.info(f"\n--- 验证拼图功能 ---")
        if GRID_AVAILABLE:
            logger.info(f"  ✅ 拼图模块可用")
        else:
            logger.info(f"  ⚠️ 拼图模块不可用")
        
        logger.info(f"\n{'='*60}")
        if all_valid:
            logger.info("✅ 所有配置验证通过！")
        else:
            logger.info("❌ 配置验证失败：")
            for err in validation_errors:
                logger.info(f"   - {err}")
        logger.info(f"{'='*60}\n")
        
        return
    
    # ============================================================
    # 主训练循环
    # ============================================================
    total_tasks = len(datasets) * len(branches)
    current_task = 0
    
    for dataset_path in datasets:
        dataset_name = dataset_path.name
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"# 处理数据集: {dataset_name}")
        logger.info(f"{'#'*60}")
        
        trigger_word = dataset_name
        
        override_path = dataset_path / 'override.json'
        override_config = {}
        if override_path.exists():
            override_config = load_json(override_path)
            logger.info(f"发现局部覆盖配置: {override_path}")
        
        for branch_file in branches:
            branch_id = branch_file.replace('.json', '')
            current_task += 1
            
            logger.info(f"\n--- [{current_task}/{total_tasks}] 分支: {branch_id} ---")
            
            # 创建任务记录
            task = TaskResult(dataset=dataset_name, branch=branch_id, status=TaskStatus.RUNNING)
            task.start_time = datetime.now()
            
            try:
                # 加载配置
                config = load_branch_config(config_dir, branch_file)
                prompts_config, model_type = get_prompts_config_for_branch(config)
                num_prompts = len(prompts_config.get('prompts', []))
                logger.info(f"模型类型: {model_type}, 使用 {num_prompts} 个提示词")
                
                config = apply_override(config, override_config, branch_id)
                
                script_name = config.get('_script', 'sdxl_train_network.py')
                script_path = kohya_root / 'sd-scripts' / script_name
                
                # 构建输出目录
                branch_output_dir = output_root / f"Generate_{dataset_name}" / branch_id
                branch_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 训练日志文件
                train_log_file = branch_output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                
                # 生成采样提示词
                sample_prompts_path = branch_output_dir / 'sample_prompts.txt'
                generate_sample_prompts_file(prompts_config, sample_prompts_path, trigger_word=trigger_word)
                logger.info(f"触发词替换: {{trigger_word}} -> {trigger_word}")
                
                output_name = f"{dataset_name}_{branch_id}"
                
                cmd_args = build_command_args(
                    config,
                    train_data_dir=dataset_path,
                    output_dir=branch_output_dir,
                    output_name=output_name,
                    sample_prompts_path=sample_prompts_path
                )
                
                # 执行训练或测试
                if args.test_mode:
                    # 测试模式：生成假数据
                    success, error_msg = run_test_mode(
                        branch_output_dir, output_name, prompts_config, 
                        args.test_steps, train_log_file, logger
                    )
                else:
                    # 正常训练模式
                    success, error_msg = run_training(script_path, cmd_args, train_log_file, logger, dry_run=args.dry_run)
                
                task.end_time = datetime.now()
                
                # 无论成功失败，都尝试处理输出（测试模式或非 dry-run 模式）
                if (args.test_mode or not args.dry_run) and not args.skip_organize:
                    sample_count, model_count, grid_generated = process_task_output(
                        branch_output_dir, num_prompts, prompts_config, logger
                    )
                    task.sample_count = sample_count
                    task.model_count = model_count
                    task.grid_generated = grid_generated
                
                # 判断最终状态
                if success:
                    task.status = TaskStatus.SUCCESS
                    logger.info(f"✅ 任务完成: {dataset_name}/{branch_id}")
                elif task.sample_count > 0 or task.model_count > 0:
                    # 有输出但训练失败，标记为部分成功
                    task.status = TaskStatus.PARTIAL
                    task.error_message = error_msg
                    logger.warning(f"⚠️ 任务部分完成: {dataset_name}/{branch_id} (有输出但训练异常)")
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = error_msg
                    logger.error(f"❌ 任务失败: {dataset_name}/{branch_id}")
                
            except Exception as e:
                task.end_time = datetime.now()
                task.status = TaskStatus.FAILED
                task.error_message = f"{str(e)}\n{traceback.format_exc()}"
                logger.exception(f"任务执行异常: {dataset_name}/{branch_id}")
            
            report.add_task(task)
    
    # ============================================================
    # 生成报告
    # ============================================================
    report_text = report.generate_report()
    report.save(logger.report_file)
    
    logger.info(f"\n{report_text}")
    logger.info(f"\n📄 详细报告已保存: {logger.report_file}")
    logger.info(f"📄 完整日志已保存: {logger.log_file}")


if __name__ == '__main__':
    main()
