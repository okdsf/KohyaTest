#!/usr/bin/env python3
"""
数据集预处理工具
用于确保图片分组满足 BATCH_SIZE 整除条件

功能:
- scan: 扫描并统计（只看不动）
- exile: 执行剔除
- restore: 恢复文件
- verify: 验证数据集
"""

import os
import re
import json
import random
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from PIL import Image


@dataclass
class ImageInfo:
    """单张图片的信息"""
    file_path: str
    txt_path: str
    root_name: str
    shot_type: str  # 'head', 'halfbody', 'full'
    resolution: Tuple[int, int]
    weight: int  # 来自目录前缀的权重
    parent_dir: str  # 所在的父目录路径


@dataclass
class SeriesInfo:
    """一个系列（同一 root_name）的信息"""
    root_name: str
    parent_dir: str
    weight: int
    images: Dict[str, ImageInfo] = field(default_factory=dict)  # shot_type -> ImageInfo
    
    @property
    def config_signature(self) -> Tuple[bool, bool, bool]:
        """构型签名: (有head, 有halfbody, 有full)"""
        return (
            'head' in self.images,
            'halfbody' in self.images,
            'full' in self.images
        )
    
    @property
    def res_signature(self) -> Tuple[Optional[Tuple[int, int]], ...]:
        """分辨率签名: (head分辨率, halfbody分辨率, full分辨率)"""
        return (
            self.images['head'].resolution if 'head' in self.images else None,
            self.images['halfbody'].resolution if 'halfbody' in self.images else None,
            self.images['full'].resolution if 'full' in self.images else None
        )
    
    @property
    def category_key(self) -> Tuple:
        """分类标识: (config_signature, res_signature)"""
        return (self.config_signature, self.res_signature)
    
    @property
    def group_type_name(self) -> str:
        """组类型名称: 三元组/二元组/一元组"""
        count = sum(self.config_signature)
        if count == 3:
            return "三元组"
        elif count == 2:
            return "二元组"
        else:
            return "一元组"


def parse_directory_weight(dir_name: str) -> int:
    """从目录名解析权重，例如 '4_portraits' -> 4"""
    match = re.match(r'^(\d+)_', dir_name)
    if match:
        return int(match.group(1))
    return 1  # 默认权重


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    解析文件名，提取 root_name 和 shot_type
    
    例如:
    - 'A_head.png' -> ('A', 'head')
    - 'A_B_head.png' -> ('A_B', 'head')
    - 'A_halfbody.png' -> ('A', 'halfbody')
    - 'A.png' -> ('A', 'full')
    """
    stem = Path(filename).stem
    
    if stem.endswith('_head'):
        return (stem[:-5], 'head')
    elif stem.endswith('_halfbody'):
        return (stem[:-9], 'halfbody')
    else:
        return (stem, 'full')


def get_image_resolution(image_path: str) -> Tuple[int, int]:
    """读取图片分辨率"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def scan_directory(base_dir: str) -> Dict[str, SeriesInfo]:
    """
    扫描目录，收集所有系列信息
    
    注意：同一 root_name 的图片可能分布在不同子目录（如 head/halfbody/full 分开存放）
    
    重要：为了保证可重复性，所有遍历都使用 sorted() 确保顺序一致
    
    返回: {root_name: SeriesInfo}
    """
    base_path = Path(base_dir)
    series_map: Dict[str, SeriesInfo] = {}
    
    # 图片扩展名
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    # 遍历一级子目录（排序确保顺序一致）
    for subdir in sorted(base_path.iterdir(), key=lambda p: p.name):
        if not subdir.is_dir():
            continue
        
        dir_name = subdir.name
        weight = parse_directory_weight(dir_name)
        parent_dir = str(subdir)
        
        # 扫描该子目录下的所有图片（排序确保顺序一致）
        for file_path in sorted(subdir.iterdir(), key=lambda p: p.name):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in image_extensions:
                continue
            
            # 解析文件名
            root_name, shot_type = parse_filename(file_path.name)
            
            # 对应的 txt 文件
            txt_path = file_path.with_suffix('.txt')
            if not txt_path.exists():
                print(f"[警告] 缺少对应的txt文件: {txt_path}")
                continue
            
            # 读取分辨率
            try:
                resolution = get_image_resolution(str(file_path))
            except Exception as e:
                print(f"[警告] 无法读取图片分辨率: {file_path}, 错误: {e}")
                continue
            
            # 创建 ImageInfo
            img_info = ImageInfo(
                file_path=str(file_path),
                txt_path=str(txt_path),
                root_name=root_name,
                shot_type=shot_type,
                resolution=resolution,
                weight=weight,
                parent_dir=parent_dir
            )
            
            # 添加到系列（跨目录，只用 root_name 作为 key）
            if root_name not in series_map:
                series_map[root_name] = SeriesInfo(
                    root_name=root_name,
                    parent_dir=parent_dir,  # 记录第一次遇到的目录
                    weight=weight
                )
            
            # 检查权重一致性
            if series_map[root_name].weight != weight:
                print(f"[警告] 系列 {root_name} 的权重不一致: "
                      f"已有 {series_map[root_name].weight}, 当前 {weight} (来自 {parent_dir})")
            
            series_map[root_name].images[shot_type] = img_info
    
    return series_map


def categorize_series(series_map: Dict[str, SeriesInfo]) -> Dict[Tuple, List[SeriesInfo]]:
    """
    按 category_key 分类系列
    
    重要：为了保证可重复性，所有列表都按 root_name 排序
    
    返回: {category_key: [SeriesInfo, ...]}（列表已按 root_name 排序）
    """
    categories: Dict[Tuple, List[SeriesInfo]] = defaultdict(list)
    
    # 按 root_name 排序后遍历，确保顺序一致
    for root_name in sorted(series_map.keys()):
        series = series_map[root_name]
        categories[series.category_key].append(series)
    
    # 每个分类内部也按 root_name 排序（虽然上面已经保证了，但显式排序更安全）
    for key in categories:
        categories[key] = sorted(categories[key], key=lambda s: s.root_name)
    
    return dict(categories)


def calculate_category_stats(categories: Dict[Tuple, List[SeriesInfo]], batch_size: int) -> List[Dict]:
    """
    计算每个分类的统计信息
    
    返回统计列表，每项包含:
    - category_key
    - group_type: 三元组/二元组/一元组
    - res_signature
    - series_list: 系列列表
    - total_weighted_count: 加权总数
    - remainder: 余数
    - need_to_remove: 需要剔除的权重和
    """
    stats = []
    
    for category_key, series_list in categories.items():
        config_sig, res_sig = category_key
        
        # 计算加权总数
        total_weighted = sum(s.weight for s in series_list)
        remainder = total_weighted % batch_size
        
        # 组类型名称
        count = sum(config_sig)
        if count == 3:
            group_type = "三元组"
        elif count == 2:
            group_type = "二元组"
        else:
            group_type = "一元组"
        
        stats.append({
            'category_key': category_key,
            'group_type': group_type,
            'config_signature': config_sig,
            'res_signature': res_sig,
            'series_list': series_list,
            'total_weighted_count': total_weighted,
            'remainder': remainder,
            'need_to_remove': remainder
        })
    
    return stats


def format_res_signature(res_sig: Tuple) -> str:
    """格式化分辨率签名为可读字符串"""
    parts = []
    for r in res_sig:
        if r is None:
            parts.append("None")
        else:
            parts.append(f"{r[0]}x{r[1]}")
    return f"({', '.join(parts)})"


def print_scan_report(stats: List[Dict], batch_size: int):
    """打印扫描报告"""
    print("\n" + "=" * 70)
    print(f"扫描报告 (BATCH_SIZE = {batch_size})")
    print("=" * 70)
    
    total_series = 0
    total_weighted = 0
    categories_with_remainder = []
    
    for stat in stats:
        total_series += len(stat['series_list'])
        total_weighted += stat['total_weighted_count']
        
        print(f"\n【{stat['group_type']}】分辨率签名: {format_res_signature(stat['res_signature'])}")
        print(f"  系列数量: {len(stat['series_list'])}")
        print(f"  加权总数: {stat['total_weighted_count']}")
        print(f"  余数: {stat['remainder']}")
        
        if stat['remainder'] > 0:
            print(f"  ⚠ 需要剔除权重和: {stat['need_to_remove']}")
            categories_with_remainder.append(stat)
        else:
            print(f"  ✓ 已满足整除条件")
    
    print("\n" + "-" * 70)
    print(f"总计: {total_series} 个系列, 加权总数 {total_weighted}")
    print(f"需要处理的分类: {len(categories_with_remainder)} 个")
    print("=" * 70)
    
    return categories_with_remainder


def find_subset_sum_exact(series_list: List[SeriesInfo], target: int, seed: int = 42) -> Optional[List[SeriesInfo]]:
    """
    子集和问题：找到权重和恰好等于 target 的系列子集
    使用随机化贪心 + 回溯
    
    重要：为了保证可重复性：
    1. 输入列表必须已经是有序的（按 root_name）
    2. 使用固定种子的随机数生成器
    3. 排序时使用 (weight, root_name) 作为复合键，确保完全确定
    
    返回: 满足条件的系列列表，或 None（无法凑出）
    """
    if target <= 0:
        return [] if target == 0 else None
    
    rng = random.Random(seed)
    
    # 按 (weight, root_name) 排序，确保完全确定性
    sorted_series = sorted(series_list, key=lambda s: (s.weight, s.root_name))
    
    # 按权重分组，然后在同权重内部用固定种子 shuffle
    weights = defaultdict(list)
    for s in sorted_series:
        weights[s.weight].append(s)
    
    # 对每个权重组内部 shuffle（因为输入已按 root_name 排序，shuffle 结果是确定的）
    for w in sorted(weights.keys()):
        rng.shuffle(weights[w])
    
    # 按权重从小到大重新组装
    shuffled = []
    for w in sorted(weights.keys()):
        shuffled.extend(weights[w])
    
    # 回溯搜索
    result = []
    
    def backtrack(index: int, current_sum: int) -> bool:
        if current_sum == target:
            return True
        if current_sum > target:
            return False
        if index >= len(shuffled):
            return False
        
        # 剪枝：剩余最小的都超过需要的
        if shuffled[index].weight > target - current_sum:
            return False
        
        # 选择当前系列
        result.append(shuffled[index])
        if backtrack(index + 1, current_sum + shuffled[index].weight):
            return True
        result.pop()
        
        # 不选择当前系列
        return backtrack(index + 1, current_sum)
    
    if backtrack(0, 0):
        return result
    return None


def find_subset_sum_upward(
    series_list: List[SeriesInfo], 
    original_target: int, 
    batch_size: int,
    total_weighted: int,
    seed: int = 42
) -> Tuple[Optional[List[SeriesInfo]], int, int]:
    """
    向上删除策略：如果精确凑不出，尝试凑 target + batch_size, target + 2*batch_size, ...
    
    返回: (系列列表, 实际凑出的值, 多删了多少)
           或 (None, 0, 0) 如果完全失败
    """
    max_possible = sum(s.weight for s in series_list)
    
    # 从原始目标开始，每次增加 batch_size
    for attempt_target in range(original_target, max_possible + 1, batch_size):
        result = find_subset_sum_exact(series_list, attempt_target, seed)
        if result is not None:
            extra_removed = attempt_target - original_target
            return (result, attempt_target, extra_removed)
    
    # 理论上不会到这里（最坏情况删光）
    return (None, 0, 0)


def execute_rescue(
    categories_to_process: List[Dict],
    dataset_dir: str,
    batch_size: int,
    seed: int = 42,
    dry_run: bool = False
) -> Tuple[bool, Dict]:
    """
    执行救济操作：把需要处理的系列移动到 {B}_rescue 文件夹
    
    核心思想：
    - 被移动的系列权重自动变成 B（batch_size）
    - B % B = 0，永远整除，不会产生余数问题
    - 数据不丢失，只是重新组织
    
    返回: (成功与否, rescue_record)
    """
    dataset_path = Path(dataset_dir)
    rescue_dir_name = f"{batch_size}_rescue"
    rescue_path = dataset_path / rescue_dir_name
    
    if not dry_run:
        rescue_path.mkdir(parents=True, exist_ok=True)
    
    rescue_record = {
        'created_at': datetime.now().isoformat(),
        'seed': seed,
        'dataset_dir': str(dataset_path),
        'rescue_dir': str(rescue_path),
        'batch_size': batch_size,
        'rescued_series': []
    }
    
    all_success = True
    
    for stat in categories_to_process:
        target = stat['need_to_remove']
        if target == 0:
            continue
        
        series_list = stat['series_list']
        total = stat['total_weighted_count']
        
        # 打印分类信息
        print(f"\n{'─' * 60}")
        print(f"[分类] {stat['group_type']} {format_res_signature(stat['res_signature'])}")
        print(f"  加权总数: {total}")
        print(f"  batch_size: {batch_size}")
        print(f"  需处理余数: {target}")
        print(f"  可用系列权重: {[s.weight for s in series_list]}")
        
        # 计算所有可能凑出的值
        weights = [s.weight for s in series_list]
        possible_sums = set([0])
        for w in weights:
            possible_sums = possible_sums | {s + w for s in possible_sums}
        possible_sums = sorted(possible_sums)
        print(f"  可凑出的值: {possible_sums[:15]}{'...' if len(possible_sums) > 15 else ''}")
        
        # 尝试精确凑数
        to_rescue = find_subset_sum_exact(series_list, target, seed)
        
        if to_rescue is not None:
            # 精确凑出
            actual_rescued = target
            print(f"\n  ✓ 精确凑数成功!")
            print(f"    目标: {target}")
            print(f"    凑法: {' + '.join(str(s.weight) for s in to_rescue)} = {actual_rescued}")
        else:
            # 需要向上凑数
            print(f"\n  ⚠ 无法精确凑出 {target}，尝试向上凑数...")
            
            found = False
            for k in range(1, len(series_list) + 1):
                new_target = target + k * batch_size
                if new_target > sum(weights):
                    break
                
                to_rescue = find_subset_sum_exact(series_list, new_target, seed)
                if to_rescue is not None:
                    actual_rescued = new_target
                    extra = new_target - target
                    found = True
                    
                    print(f"\n  → 向上凑数策略:")
                    print(f"    原目标: {target}")
                    print(f"    尝试: {target} + {k}×{batch_size} = {new_target}")
                    print(f"    凑法: {' + '.join(str(s.weight) for s in to_rescue)} = {actual_rescued}")
                    break
            
            if not found:
                # 最坏情况：全部移入救济站
                to_rescue = series_list.copy()
                actual_rescued = sum(weights)
                print(f"\n  → 全部移入救济站:")
                print(f"    移动全部 {len(to_rescue)} 个系列，权重和 {actual_rescued}")
                found = True
                
            if not found:
                print(f"\n  ✗ 错误：无法找到有效的方案")
                all_success = False
                continue
        
        # 计算救济后的效果
        remaining_weight = total - actual_rescued
        rescued_new_weight = len(to_rescue) * batch_size  # 每个系列权重变成 B
        new_total = remaining_weight + rescued_new_weight
        
        print(f"\n  📊 救济效果:")
        print(f"    原位置剩余: {remaining_weight}")
        print(f"    救济站贡献: {len(to_rescue)} 系列 × {batch_size} = {rescued_new_weight}")
        print(f"    新总数: {remaining_weight} + {rescued_new_weight} = {new_total}")
        print(f"    验证: {new_total} % {batch_size} = {new_total % batch_size}")
        
        # 打印选中的系列
        print(f"\n  选中系列 (移入 {rescue_dir_name}/):")
        for s in to_rescue:
            print(f"    - {s.root_name} (原权重: {s.weight} → 新权重: {batch_size})")
        
        # 执行移动
        for series in to_rescue:
            series_record = {
                'root_name': series.root_name,
                'original_weight': series.weight,
                'new_weight': batch_size,
                'category': f"{stat['group_type']}_{format_res_signature(stat['res_signature'])}",
                'files': []
            }
            
            for shot_type, img_info in series.images.items():
                # 目标路径：救济站目录
                new_img_path = rescue_path / Path(img_info.file_path).name
                new_txt_path = rescue_path / Path(img_info.txt_path).name
                
                file_record = {
                    'shot_type': shot_type,
                    'original_image_path': img_info.file_path,
                    'original_txt_path': img_info.txt_path,
                    'rescued_image_path': str(new_img_path),
                    'rescued_txt_path': str(new_txt_path)
                }
                series_record['files'].append(file_record)
                
                if not dry_run:
                    shutil.move(img_info.file_path, str(new_img_path))
                    shutil.move(img_info.txt_path, str(new_txt_path))
            
            rescue_record['rescued_series'].append(series_record)
        
        if not dry_run:
            print(f"\n  ✓ 文件已移动到: {rescue_path}")
    
    return all_success, rescue_record


def restore_from_rescue_record(record_path: str, dry_run: bool = False) -> bool:
    """从救济记录恢复文件到原位置"""
    with open(record_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}恢复文件")
    print(f"记录文件: {record_path}")
    print(f"创建时间: {record['created_at']}")
    print(f"救济站: {record['rescue_dir']}")
    
    success = True
    
    for series_record in record['rescued_series']:
        print(f"\n[恢复] {series_record['root_name']} (权重: {series_record['new_weight']} → {series_record['original_weight']})")
        
        for file_record in series_record['files']:
            rescued_img = file_record['rescued_image_path']
            rescued_txt = file_record['rescued_txt_path']
            original_img = file_record['original_image_path']
            original_txt = file_record['original_txt_path']
            
            if not dry_run:
                try:
                    shutil.move(rescued_img, original_img)
                    shutil.move(rescued_txt, original_txt)
                    print(f"  ✓ {Path(original_img).name}")
                except Exception as e:
                    print(f"  ✗ {Path(original_img).name}: {e}")
                    success = False
            else:
                print(f"  [DRY-RUN] {Path(rescued_img).name} → {original_img}")
    
    return success


def restore_from_record(record_path: str, dry_run: bool = False) -> bool:
    """从记录恢复文件（兼容旧版 exile 和新版 rescue）"""
    with open(record_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    
    # 判断是旧版 exile 还是新版 rescue
    if 'rescued_series' in record:
        return restore_from_rescue_record(record_path, dry_run)
    
    # 旧版 exile 逻辑
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}恢复文件 (旧版exile格式)")
    print(f"记录文件: {record_path}")
    print(f"创建时间: {record['created_at']}")
    
    success = True
    
    for series_record in record.get('removed_series', []):
        print(f"\n[恢复] {series_record['root_name']}")
        
        for file_record in series_record['files']:
            exiled_img = file_record['exiled_image_path']
            exiled_txt = file_record['exiled_txt_path']
            original_img = file_record['original_image_path']
            original_txt = file_record['original_txt_path']
            
            if not dry_run:
                try:
                    shutil.move(exiled_img, original_img)
                    shutil.move(exiled_txt, original_txt)
                    print(f"  ✓ {Path(original_img).name}")
                except Exception as e:
                    print(f"  ✗ {Path(original_img).name}: {e}")
                    success = False
            else:
                print(f"  [DRY-RUN] {Path(exiled_img).name} -> {original_img}")
    
    return success


def verify_dataset(base_dir: str, batch_size: int) -> bool:
    """验证数据集是否满足整除条件"""
    print(f"\n验证数据集: {base_dir}")
    print(f"BATCH_SIZE: {batch_size}")
    
    # 扫描
    series_map = scan_directory(base_dir)
    categories = categorize_series(series_map)
    stats = calculate_category_stats(categories, batch_size)
    
    # 检查
    all_pass = True
    failed_categories = []
    
    for stat in stats:
        if stat['remainder'] != 0:
            all_pass = False
            failed_categories.append(stat)
    
    # 报告
    print("\n" + "-" * 50)
    if all_pass:
        print("✓ 验证通过！所有分类都满足整除条件。")
    else:
        print("✗ 验证失败！以下分类不满足整除条件:")
        for stat in failed_categories:
            print(f"  - {stat['group_type']} {format_res_signature(stat['res_signature'])}")
            print(f"    加权总数: {stat['total_weighted_count']}, 余数: {stat['remainder']}")
    
    return all_pass


def batch_scan(base_dir: str, batch_size: int):
    """批量扫描所有子数据集"""
    base_path = Path(base_dir)
    
    print("\n" + "=" * 70)
    print(f"批量扫描 (根目录: {base_dir}, BATCH_SIZE = {batch_size})")
    print("=" * 70)
    
    results = []
    
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 检查是否是数据集目录（包含带数字前缀的子目录）
        has_dataset_structure = any(
            re.match(r'^\d+_', d.name) for d in subdir.iterdir() if d.is_dir()
        )
        
        if not has_dataset_structure:
            continue
        
        print(f"\n{'─' * 70}")
        print(f"数据集: {subdir.name}")
        print(f"{'─' * 70}")
        
        series_map = scan_directory(str(subdir))
        categories = categorize_series(series_map)
        stats = calculate_category_stats(categories, batch_size)
        
        need_process = []
        for stat in stats:
            status = "⚠ 需处理" if stat['remainder'] > 0 else "✓"
            print(f"  {status} {stat['group_type']} {format_res_signature(stat['res_signature'])}: "
                  f"加权={stat['total_weighted_count']}, 余数={stat['remainder']}")
            if stat['remainder'] > 0:
                need_process.append(stat)
        
        results.append({
            'name': subdir.name,
            'path': str(subdir),
            'total_series': len(series_map),
            'categories_need_process': len(need_process)
        })
    
    # 汇总
    print("\n" + "=" * 70)
    print("批量扫描汇总")
    print("=" * 70)
    
    total_need = 0
    for r in results:
        status = "⚠" if r['categories_need_process'] > 0 else "✓"
        print(f"  {status} {r['name']}: {r['total_series']} 系列, "
              f"{r['categories_need_process']} 个分类需处理")
        total_need += r['categories_need_process']
    
    print(f"\n总计: {len(results)} 个数据集, {total_need} 个分类需要处理")
    
    return results


def batch_rescue(base_dir: str, batch_size: int, seed: int = 42, dry_run: bool = False):
    """批量救济所有子数据集"""
    base_path = Path(base_dir)
    
    print("\n" + "=" * 70)
    print(f"{'[DRY-RUN] ' if dry_run else ''}批量救济")
    print(f"根目录: {base_dir}")
    print(f"BATCH_SIZE = {batch_size}")
    print(f"救济站目录名: {batch_size}_rescue/")
    print("=" * 70)
    
    all_records = {
        'created_at': datetime.now().isoformat(),
        'base_dir': str(base_path),
        'batch_size': batch_size,
        'seed': seed,
        'datasets': []
    }
    
    all_success = True
    
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 检查是否是数据集目录
        has_dataset_structure = any(
            re.match(r'^\d+_', d.name) for d in subdir.iterdir() if d.is_dir()
        )
        
        if not has_dataset_structure:
            continue
        
        dataset_name = subdir.name
        print(f"\n{'─' * 70}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'─' * 70}")
        
        # 扫描
        series_map = scan_directory(str(subdir))
        categories = categorize_series(series_map)
        stats = calculate_category_stats(categories, batch_size)
        
        categories_to_process = [s for s in stats if s['remainder'] > 0]
        
        if not categories_to_process:
            print(f"  ✓ 无需处理")
            continue
        
        # 执行救济
        success, record = execute_rescue(
            categories_to_process,
            str(subdir),
            batch_size,
            seed,
            dry_run
        )
        
        if not success:
            all_success = False
        
        # 记录
        all_records['datasets'].append({
            'name': dataset_name,
            'path': str(subdir),
            'rescue_dir': str(subdir / f'{batch_size}_rescue'),
            'rescued_series': record.get('rescued_series', [])
        })
    
    # 保存总记录
    if not dry_run:
        record_path = base_path / 'batch_rescue_record.json'
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"\n总记录已保存: {record_path}")
    
    if all_success:
        print("\n批量救济完成！")
    else:
        print("\n批量救济过程中遇到错误，请检查上方输出。")
    
    return all_success


def batch_restore(record_path: str, dry_run: bool = False) -> bool:
    """从批量记录恢复所有文件"""
    with open(record_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}批量恢复")
    print(f"记录文件: {record_path}")
    print(f"创建时间: {record['created_at']}")
    print(f"数据集数量: {len(record['datasets'])}")
    
    all_success = True
    
    for dataset in record['datasets']:
        print(f"\n{'─' * 70}")
        print(f"恢复数据集: {dataset['name']}")
        print(f"{'─' * 70}")
        
        # 支持新版 rescue 和旧版 exile 格式
        series_list = dataset.get('rescued_series', dataset.get('removed_series', []))
        
        for series_record in series_list:
            print(f"\n  [恢复] {series_record['root_name']}")
            
            for file_record in series_record['files']:
                # 支持新版和旧版的字段名
                rescued_img = file_record.get('rescued_image_path', file_record.get('exiled_image_path'))
                rescued_txt = file_record.get('rescued_txt_path', file_record.get('exiled_txt_path'))
                original_img = file_record['original_image_path']
                original_txt = file_record['original_txt_path']
                
                if not dry_run:
                    try:
                        shutil.move(rescued_img, original_img)
                        shutil.move(rescued_txt, original_txt)
                        print(f"    ✓ {Path(original_img).name}")
                    except Exception as e:
                        print(f"    ✗ {Path(original_img).name}: {e}")
                        all_success = False
                else:
                    print(f"    [DRY-RUN] {Path(rescued_img).name} → {original_img}")
    
    if all_success:
        print("\n批量恢复完成！")
    else:
        print("\n批量恢复过程中遇到错误。")
    
    return all_success


def batch_verify(base_dir: str, batch_size: int) -> bool:
    """批量验证所有子数据集"""
    base_path = Path(base_dir)
    
    print("\n" + "=" * 70)
    print(f"批量验证 (根目录: {base_dir}, BATCH_SIZE = {batch_size})")
    print("=" * 70)
    
    all_pass = True
    results = []
    
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 检查是否是数据集目录
        has_dataset_structure = any(
            re.match(r'^\d+_', d.name) for d in subdir.iterdir() if d.is_dir()
        )
        
        if not has_dataset_structure:
            continue
        
        # 验证
        series_map = scan_directory(str(subdir))
        categories = categorize_series(series_map)
        stats = calculate_category_stats(categories, batch_size)
        
        failed = [s for s in stats if s['remainder'] > 0]
        passed = len(failed) == 0
        
        if not passed:
            all_pass = False
        
        results.append({
            'name': subdir.name,
            'passed': passed,
            'failed_count': len(failed)
        })
    
    # 汇总
    print("\n验证结果:")
    for r in results:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['name']}" + (f" ({r['failed_count']} 个分类未通过)" if not r['passed'] else ""))
    
    print(f"\n{'✓ 全部通过！' if all_pass else '✗ 部分数据集未通过验证'}")
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description='数据集预处理工具 - 确保图片分组满足 BATCH_SIZE 整除条件',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # ========== 单数据集命令 ==========
    
    # scan 命令
    scan_parser = subparsers.add_parser('scan', help='扫描并统计（只看不动）')
    scan_parser.add_argument('--dir', required=True, help='数据集目录路径')
    scan_parser.add_argument('--batch-size', type=int, required=True, help='批次大小')
    
    # rescue 命令（原 exile，现改为救济模式）
    rescue_parser = subparsers.add_parser('rescue', help='执行救济（移动到 B_rescue 文件夹）')
    rescue_parser.add_argument('--dir', required=True, help='数据集目录路径')
    rescue_parser.add_argument('--batch-size', type=int, required=True, help='批次大小')
    rescue_parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    rescue_parser.add_argument('--dry-run', action='store_true', help='只显示会做什么，不实际移动')
    
    # restore 命令
    restore_parser = subparsers.add_parser('restore', help='恢复文件')
    restore_parser.add_argument('--record', required=True, help='rescue_record.json 路径')
    restore_parser.add_argument('--dry-run', action='store_true', help='只显示会做什么，不实际移动')
    
    # verify 命令
    verify_parser = subparsers.add_parser('verify', help='验证数据集')
    verify_parser.add_argument('--dir', required=True, help='数据集目录路径')
    verify_parser.add_argument('--batch-size', type=int, required=True, help='批次大小')
    
    # ========== 批量命令 ==========
    
    # batch-scan 命令
    batch_scan_parser = subparsers.add_parser('batch-scan', help='批量扫描所有子数据集')
    batch_scan_parser.add_argument('--dir', required=True, help='包含多个数据集的根目录')
    batch_scan_parser.add_argument('--batch-size', type=int, required=True, help='批次大小')
    
    # batch-rescue 命令（原 batch-exile）
    batch_rescue_parser = subparsers.add_parser('batch-rescue', help='批量执行救济')
    batch_rescue_parser.add_argument('--dir', required=True, help='包含多个数据集的根目录')
    batch_rescue_parser.add_argument('--batch-size', type=int, required=True, help='批次大小')
    batch_rescue_parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    batch_rescue_parser.add_argument('--dry-run', action='store_true', help='只显示会做什么，不实际移动')
    
    # batch-restore 命令
    batch_restore_parser = subparsers.add_parser('batch-restore', help='批量恢复文件')
    batch_restore_parser.add_argument('--record', required=True, help='batch_rescue_record.json 路径')
    batch_restore_parser.add_argument('--dry-run', action='store_true', help='只显示会做什么，不实际移动')
    
    # batch-verify 命令
    batch_verify_parser = subparsers.add_parser('batch-verify', help='批量验证所有子数据集')
    batch_verify_parser.add_argument('--dir', required=True, help='包含多个数据集的根目录')
    batch_verify_parser.add_argument('--batch-size', type=int, required=True, help='批次大小')
    
    args = parser.parse_args()
    
    # ========== 单数据集命令处理 ==========
    
    if args.command == 'scan':
        series_map = scan_directory(args.dir)
        categories = categorize_series(series_map)
        stats = calculate_category_stats(categories, args.batch_size)
        print_scan_report(stats, args.batch_size)
        
    elif args.command == 'rescue':
        series_map = scan_directory(args.dir)
        categories = categorize_series(series_map)
        stats = calculate_category_stats(categories, args.batch_size)
        categories_to_process = print_scan_report(stats, args.batch_size)
        
        if not categories_to_process:
            print("\n无需救济，数据集已满足条件。")
            return
        
        print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}开始救济...")
        print(f"救济站: {args.batch_size}_rescue/")
        
        success, record = execute_rescue(
            categories_to_process,
            args.dir,
            args.batch_size,
            args.seed,
            args.dry_run
        )
        
        if not args.dry_run:
            record_path = Path(args.dir) / f'{args.batch_size}_rescue' / 'rescue_record.json'
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"\n记录已保存: {record_path}")
        
        if success:
            print("\n救济完成！")
            if not args.dry_run:
                print("运行 verify 命令确认结果。")
        else:
            print("\n救济过程中遇到错误，请检查上方输出。")
            
    elif args.command == 'restore':
        success = restore_from_record(args.record, args.dry_run)
        if success:
            print("\n恢复完成！")
        else:
            print("\n恢复过程中遇到错误。")
            
    elif args.command == 'verify':
        verify_dataset(args.dir, args.batch_size)
    
    # ========== 批量命令处理 ==========
    
    elif args.command == 'batch-scan':
        batch_scan(args.dir, args.batch_size)
        
    elif args.command == 'batch-rescue':
        batch_rescue(
            args.dir,
            args.batch_size,
            args.seed,
            args.dry_run
        )
        
    elif args.command == 'batch-restore':
        batch_restore(args.record, args.dry_run)
        
    elif args.command == 'batch-verify':
        batch_verify(args.dir, args.batch_size)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
