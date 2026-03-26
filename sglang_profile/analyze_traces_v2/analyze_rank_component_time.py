#!/usr/bin/env python3
"""
Analyze rank component time - 对每个rank的分析结果进行组件时间可视化
"""
import os
import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def load_csv_data(csv_file):
    """
    加载CSV数据

    Args:
        csv_file: CSV文件路径

    Returns:
        数据列表
    """
    data = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'trace_file': row['trace_file'],
                'tp_rank': int(row['tp_rank']),
                'step_idx': int(row['step_idx']),
                'layer_idx': int(row['layer_idx']),
                'layer_type': row['layer_type'],
                'stage': row['stage'],
                'dur_ms': float(row['dur_ms']),
            })

    return data


def aggregate_by_layer(data):
    """
    在层维度上聚合各个component的时间

    Args:
        data: 原始数据

    Returns:
        聚合后的数据: {step_idx: {stage: avg_dur_ms}}
    """
    # {step_idx: {stage: [durations]}}
    step_stage_durations = defaultdict(lambda: defaultdict(list))

    for row in data:
        step_idx = row['step_idx']
        stage = row['stage']
        dur_ms = row['dur_ms']

        step_stage_durations[step_idx][stage].append(dur_ms)

    # 计算每个step每个stage的平均值
    aggregated = {}
    for step_idx, stages in step_stage_durations.items():
        aggregated[step_idx] = {}
        for stage, durations in stages.items():
            avg_dur = sum(durations) / len(durations)
            aggregated[step_idx][stage] = avg_dur

    return aggregated


def plot_component_time(aggregated_data, output_dir, tp_rank):
    """
    绘制每个component时间随step变化的折线图

    Args:
        aggregated_data: 聚合后的数据 {step_idx: {stage: avg_dur_ms}}
        output_dir: 输出目录
        tp_rank: TP rank
    """
    # 收集所有stage，排除dense
    all_stages = set()
    for stages in aggregated_data.values():
        all_stages.update(stages.keys())
    all_stages = sorted([s for s in all_stages if s != 'dense'])

    # 按step排序
    sorted_steps = sorted(aggregated_data.keys())

    # 为每个stage准备数据
    stage_data = {stage: [] for stage in all_stages}
    for step in sorted_steps:
        for stage in all_stages:
            stage_data[stage].append(aggregated_data[step].get(stage, 0.0))

    # 创建图形
    plt.figure(figsize=(14, 8))

    # 为每个stage绘制折线
    colors = plt.cm.tab10(range(len(all_stages)))
    for i, stage in enumerate(all_stages):
        plt.plot(sorted_steps, stage_data[stage],
                marker='o',
                linewidth=2,
                markersize=4,
                label=stage,
                color=colors[i])

    # 设置图表属性
    plt.xlabel('Step Index', fontsize=12, fontweight='bold')
    plt.ylabel('Average Duration (ms)', fontsize=12, fontweight='bold')
    plt.title(f'Component Time per Step - Rank {tp_rank}',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)

    # 优化布局
    plt.tight_layout()

    # 保存图表
    output_file = os.path.join(output_dir, f'rank_{tp_rank}_component_time.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot to: {output_file}")


def analyze_single_rank(csv_file, output_dir):
    """
    分析单个rank的数据

    Args:
        csv_file: CSV文件路径
        output_dir: 输出目录
    """
    filename = os.path.basename(csv_file)
    print(f"\nProcessing: {filename}")

    # 加载数据
    data = load_csv_data(csv_file)
    if not data:
        print(f"  Warning: No data in {filename}")
        return

    tp_rank = data[0]['tp_rank']
    print(f"  TP Rank: {tp_rank}")
    print(f"  Total records: {len(data)}")

    # 聚合数据
    aggregated = aggregate_by_layer(data)
    num_steps = len(aggregated)
    print(f"  Steps: {num_steps}")

    # 绘制图表
    plot_component_time(aggregated, output_dir, tp_rank)


def analyze_all_ranks(input_dir, output_dir):
    """
    分析所有rank的数据

    Args:
        input_dir: 输入目录（包含analysis结果的CSV文件）
        output_dir: 输出目录
    """
    # 查找所有analysis CSV文件
    csv_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.analysis.csv'):
            csv_files.append(os.path.join(input_dir, file))

    if not csv_files:
        print(f"Error: No .analysis.csv files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} analysis files")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个文件
    for i, csv_file in enumerate(sorted(csv_files), 1):
        print(f"\n[{i}/{len(csv_files)}]")
        try:
            analyze_single_rank(csv_file, output_dir)
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_rank_component_time.py <input_dir> [output_dir]")
        print("  input_dir: Directory containing .analysis.csv files")
        print("  output_dir: Output directory for plots (optional, defaults to input_dir/plots)")
        sys.exit(1)

    input_dir = sys.argv[1]

    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.join(input_dir, 'plots')

    analyze_all_ranks(input_dir, output_dir)


if __name__ == '__main__':
    main()
