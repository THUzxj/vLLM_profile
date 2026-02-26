#!/usr/bin/env python3
"""
分析 component_times 数据，按 batch_size 汇总每个 component 的时间统计信息
"""

import json
import os
import re
import csv
from collections import defaultdict
from pathlib import Path
import statistics
from typing import Dict, List, Any


def parse_count_number(filename: str) -> int:
    """从文件名中解析 count number

    Args:
        filename: 文件名，格式如 'count_0_promptlenshape_torch.Size([10])_time3989154.886429787.json'

    Returns:
        count_number: 解析出的 count number
    """
    match = re.search(r'count_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"无法从文件名中解析 count_number: {filename}")


def parse_batch_size(filename: str) -> int:
    """从文件名中解析 batch_size

    Args:
        filename: 文件名，格式如 'count_11_promptlenshape_torch.Size([4])_time3949059.841547339.json'

    Returns:
        batch_size: 解析出的 batch size
    """
    match = re.search(r'torch\.Size\(\[(\d+)\]\)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"无法从文件名中解析 batch_size: {filename}")


def extract_components(data: Dict[str, Any]) -> Dict[str, float]:
    """从 JSON 数据中提取所有 component 的时间

    Args:
        data: JSON 数据字典

    Returns:
        components: 字典，key 为 component 名称，value 为时间值
    """
    components = {}

    # 提取 model_time
    if 'model_time' in data:
        components['model_time'] = data['model_time']

    # 提取 layer_times
    if 'layer_times' in data:
        for layer in data['layer_times']:
            layer_idx = layer.get('layer_idx', 0)

            # 提取 total_layer_time
            if 'total_layer_time' in layer:
                components[f'layer_{layer_idx}_total'] = layer['total_layer_time']

            # 提取 layer_details
            if 'layer_details' in layer:
                details = layer['layer_details']
                for detail_name, detail_time in details.items():
                    components[f'layer_{layer_idx}_{detail_name}'] = detail_time

    return components


def filter_outliers(values: List[float]) -> List[float]:
    """筛选掉明显大于其他值的离群值

    使用 IQR (Interquartile Range) 方法检测上界离群值。
    如果没有离群值，则返回原始数据。

    Args:
        values: 数值列表

    Returns:
        filtered_values: 筛选后的数值列表（如果没有离群值则返回原始列表）
    """
    if not values or len(values) < 4:
        # 数据太少，无法有效检测离群值
        return values

    sorted_values = sorted(values)
    n = len(sorted_values)

    # 计算第一四分位数 (Q1) 和第三四分位数 (Q3)
    # 使用标准方法：Q1 是下中位数的中位数，Q3 是上中位数的中位数
    mid = n // 2
    if n % 2 == 0:
        # 偶数个元素
        lower_half = sorted_values[:mid]
        upper_half = sorted_values[mid:]
    else:
        # 奇数个元素，中位数不包含在任一半中
        lower_half = sorted_values[:mid]
        upper_half = sorted_values[mid + 1:]

    # Q1 是下中位数
    q1 = statistics.median(lower_half)
    # Q3 是上中位数
    q3 = statistics.median(upper_half)

    # 计算 IQR
    iqr = q3 - q1

    # 如果 IQR 为 0，说明数据没有分散性，不需要筛选
    if iqr == 0:
        return values

    # 计算上界阈值 (只检测明显大于其他值的离群值)
    upper_bound = q3 + 1.5 * iqr

    # 筛选掉大于阈值的值
    filtered_values = [v for v in values if v <= upper_bound]

    # 如果没有筛选掉任何值，返回原始数据
    if len(filtered_values) == len(values):
        return values
    else:
        print(f"filtered out values: {[v for v in values if v > upper_bound]}")

    return filtered_values


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """计算统计信息

    在计算平均值时，会自动筛选掉明显大于其他值的离群值。

    Args:
        values: 数值列表

    Returns:
        stats: 包含平均值、最大值、最小值、方差、标准差、中位数等的字典
    """
    if not values:
        return {}

    # 筛选离群值用于计算平均值
    filtered_values = filter_outliers(values)

    # 使用原始值计算其他统计信息（min, max, median）
    # 使用筛选后的值计算平均值
    stats = {
        'count': len(values),  # 原始数据数量
        'mean': statistics.mean(filtered_values),  # 使用筛选后的数据计算平均值
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values),
    }

    # 如果筛选掉了离群值，记录相关信息
    if len(filtered_values) < len(values):
        stats['outliers_removed'] = len(values) - len(filtered_values)
        stats['filtered_count'] = len(filtered_values)
    else:
        stats['outliers_removed'] = 0
        stats['filtered_count'] = len(values)

    if len(values) > 1:
        stats['variance'] = statistics.variance(values)
        stats['stdev'] = statistics.stdev(values)
    else:
        stats['variance'] = 0.0
        stats['stdev'] = 0.0

    return stats


def write_component_csv(output_file: Path, component_name: str, batch_data: Dict[int, Dict[str, float]]):
    """将 component 统计信息写入 CSV 文件

    Args:
        output_file: CSV 文件路径
        component_name: component 名称
        batch_data: {batch_size: stats} 字典
    """
    sorted_batch_sizes = sorted(batch_data.keys())

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['component_name', 'batch_size', 'count', 'mean', 'min', 'max',
                         'median', 'variance', 'stdev'])

        # 写入数据行
        for batch_size in sorted_batch_sizes:
            stats = batch_data[batch_size]
            writer.writerow([
                component_name,
                batch_size,
                stats['count'],
                stats['mean'],
                stats['min'],
                stats['max'],
                stats['median'],
                stats['variance'],
                stats['stdev']
            ])


def analyze_component_times(data_dir: str, output_dir: str = None, output_len: int = 4, warmup_steps: int = 3):
    """分析 component_times 数据

    Args:
        data_dir: 包含 JSON 文件的目录路径
        output_dir: 输出目录，如果为 None 则在 data_dir 下创建 'analysis' 子目录
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"目录不存在: {data_dir}")

    # 设置输出目录
    if output_dir is None:
        output_dir = data_path / 'analysis'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 存储数据: {component_name: {batch_size: [values]}}
    component_data = defaultdict(lambda: defaultdict(list))

    # 遍历所有 JSON 文件
    json_files = sorted(data_path.glob('*.json'))
    print(f"找到 {len(json_files)} 个 JSON 文件")

    for json_file in json_files:
        count_number = parse_count_number(json_file.name)

        if count_number <= output_len * warmup_steps:
            continue

        try:
            # 解析 batch_size
            batch_size = parse_batch_size(json_file.name)

            # 读取 JSON 数据
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 提取 components
            components = extract_components(data)

            # 存储到 component_data
            for component_name, time_value in components.items():
                component_data[component_name][batch_size].append(time_value)

        except Exception as e:
            print(f"处理文件 {json_file.name} 时出错: {e}")
            continue

    # 计算统计信息并输出
    print(f"\n找到 {len(component_data)} 个不同的 component")

    # 为每个 component 生成统计报告
    for component_name, batch_data in component_data.items():
        # 准备输出数据
        output_data = {
            'component_name': component_name,
            'statistics_by_batch_size': {}
        }

        # 按 batch_size 排序
        sorted_batch_sizes = sorted(batch_data.keys())

        # 存储统计信息用于 CSV 导出
        stats_by_batch = {}

        for batch_size in sorted_batch_sizes:
            values = batch_data[batch_size]
            stats = calculate_statistics(values)
            output_data['statistics_by_batch_size'][batch_size] = stats
            stats_by_batch[batch_size] = stats

        # 文件名中替换可能的不合法字符
        safe_component_name = component_name.replace(
            '/', '_').replace('\\', '_')

        # 保存 JSON 文件
        json_file = output_dir / f'{safe_component_name}_statistics.json'
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"已生成: {json_file}")

        # 保存 CSV 文件
        csv_file = output_dir / f'{safe_component_name}_statistics.csv'
        write_component_csv(csv_file, component_name, stats_by_batch)
        print(f"已生成: {csv_file}")

    # 生成汇总报告（所有 component 的汇总）
    summary = {
        'total_components': len(component_data),
        'components': {}
    }

    for component_name, batch_data in component_data.items():
        safe_component_name = component_name.replace(
            '/', '_').replace('\\', '_')
        summary['components'][component_name] = {
            'batch_sizes': sorted(batch_data.keys()),
            'total_samples': sum(len(values) for values in batch_data.values())
        }

    # 保存 JSON 汇总报告
    summary_json_file = output_dir / 'summary.json'
    with open(summary_json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n已生成汇总报告: {summary_json_file}")

    # 保存 CSV 汇总报告
    summary_csv_file = output_dir / 'summary.csv'
    with open(summary_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['component_name', 'batch_sizes', 'total_samples'])
        for component_name, batch_data in component_data.items():
            batch_sizes_str = ','.join(map(str, sorted(batch_data.keys())))
            total_samples = sum(len(values) for values in batch_data.values())
            writer.writerow([component_name, batch_sizes_str, total_samples])
    print(f"已生成汇总报告: {summary_csv_file}")

    print(f"所有分析结果保存在: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='分析 component_times 数据')
    parser.add_argument('data_dir', type=str, help='包含 JSON 文件的目录路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出目录（默认为 data_dir/analysis）')
    parser.add_argument('--output-len', type=int, default=4, help='输出长度，默认为 4')
    parser.add_argument('--warmup-steps', type=int,
                        default=3, help='预热步骤数，默认为 3')
    args = parser.parse_args()

    analyze_component_times(args.data_dir, args.output,
                            args.output_len, args.warmup_steps)


if __name__ == '__main__':
    main()
