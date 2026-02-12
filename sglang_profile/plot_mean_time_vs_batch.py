#!/usr/bin/env python3
"""
分析脚本：读取各个组件的统计数据，绘制mean time与batch size的关系折线图
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_component_statistics(stats_dir):
    """
    读取指定目录下所有的*_statistics.csv文件，提取component_name, batch_size, mean, min数据
    """
    csv_files = glob.glob(os.path.join(stats_dir, '*_statistics.csv'))

    if not csv_files:
        print(f"未找到CSV文件在目录: {stats_dir}")
        return None

    data_dict = {}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # 提取component_name（从第一行）和batch_size, mean, min数据
        if 'component_name' in df.columns and 'batch_size' in df.columns and 'mean' in df.columns:
            component_name = df['component_name'].iloc[0]
            batch_sizes = df['batch_size'].values
            means = df['mean'].values
            # 如果没有min列，用mean代替
            mins = df['min'].values if 'min' in df.columns else means

            data_dict[component_name] = {
                'batch_sizes': batch_sizes,
                'means': means,
                'mins': mins
            }

    return data_dict


def plot_mean_time_vs_batch_size(data_dict, output_dir=None):
    """
    绘制各个组件的mean time与batch size的关系
    """
    if not data_dict:
        print("没有数据可绘制")
        return

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 排除model_time，只绘制layer相关的组件
    for component_name, data in sorted(data_dict.items()):
        if 'model_time' not in component_name:  # 排除总的model_time
            batch_sizes = data['batch_sizes']
            means = data['means']

            # 转换为毫秒（从秒转换）
            means_ms = [m * 1000 for m in means]

            ax.plot(batch_sizes, means_ms, marker='o',
                    label=component_name, linewidth=2, markersize=6)

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Time vs Batch Size for Different Components',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, 'mean_time_vs_batch_size.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")

    plt.show()


def plot_individual_components(data_dict, output_dir=None):
    """
    为每个主要组件绘制单独的图表（更清晰的视图）
    """
    if not data_dict:
        print("没有数据可绘制")
        return

    # 分组绘制主要组件
    main_components = [
        'layer_0_total',
        'layer_0_self_attention',
        'layer_0_mlp',
        'layer_0_attention_prepare',
        'layer_0_attention_core',
        'layer_0_mlp_gate',
        'layer_0_mlp_experts'
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, component in enumerate(main_components):
        if component in data_dict:
            ax = axes[idx]
            batch_sizes = data_dict[component]['batch_sizes']
            means = data_dict[component]['means']
            means_ms = [m * 1000 for m in means]

            ax.plot(batch_sizes, means_ms, marker='o',
                    color='blue', linewidth=2, markersize=6)
            # ax.fill_between(range(len(batch_sizes)), means_ms, alpha=0.3)
            ax.set_xlabel('Batch Size', fontsize=10)
            ax.set_ylabel('Mean Time (ms)', fontsize=10)
            ax.set_title(component, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(b) for b in batch_sizes], rotation=45)
            ax.set_ylim(bottom=0)  # 从0开始显示Y轴

    # 隐藏最后一个未使用的子图
    if len(main_components) < len(axes):
        axes[-1].axis('off')

    plt.suptitle('Mean Time vs Batch Size - Individual Components',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(
        output_dir, 'individual_components_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"个别组件分析图已保存到: {output_path}")

    plt.show()


def plot_min_time_vs_batch_size(data_dict, output_dir=None):
    """
    绘制各个组件的min time与batch size的关系
    """
    if not data_dict:
        print("没有数据可绘制")
        return

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 排除model_time，只绘制layer相关的组件
    for component_name, data in sorted(data_dict.items()):
        if 'model_time' not in component_name:  # 排除总的model_time
            batch_sizes = data['batch_sizes']
            mins = data['mins']

            # 转换为毫秒（从秒转换）
            mins_ms = [m * 1000 for m in mins]

            ax.plot(batch_sizes, mins_ms, marker='s',
                    label=component_name, linewidth=2, markersize=6)

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Min Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Min Time vs Batch Size for Different Components',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, 'min_time_vs_batch_size.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Min Time图表已保存到: {output_path}")

    plt.show()


def plot_individual_components_min_time(data_dict, output_dir=None):
    """
    为每个主要组件绘制单独的min time图表
    """
    if not data_dict:
        print("没有数据可绘制")
        return

    # 分组绘制主要组件
    main_components = [
        'layer_0_total',
        'layer_0_self_attention',
        'layer_0_mlp',
        'layer_0_attention_prepare',
        'layer_0_attention_core',
        'layer_0_mlp_gate',
        'layer_0_mlp_experts'
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, component in enumerate(main_components):
        if component in data_dict:
            ax = axes[idx]
            batch_sizes = data_dict[component]['batch_sizes']
            mins = data_dict[component]['mins']
            mins_ms = [m * 1000 for m in mins]

            ax.plot(batch_sizes, mins_ms, marker='s',
                    color='green', linewidth=2, markersize=6)
            # ax.fill_between(range(len(batch_sizes)), mins_ms,
            # alpha=0.3, color='green')
            ax.set_xlabel('Batch Size', fontsize=10)
            ax.set_ylabel('Min Time (ms)', fontsize=10)
            ax.set_title(component, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(b) for b in batch_sizes], rotation=45)
            ax.set_ylim(bottom=0)  # 从0开始显示Y轴

    # 隐藏最后一个未使用的子图
    if len(main_components) < len(axes):
        axes[-1].axis('off')

    plt.suptitle('Min Time vs Batch Size - Individual Components',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(
        output_dir, 'individual_components_min_time_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Min Time个别组件分析图已保存到: {output_path}")

    plt.show()


def print_summary(data_dict):
    """
    打印数据摘要
    """
    print("\n" + "="*80)
    print("数据摘要")
    print("="*80)

    for component_name, data in sorted(data_dict.items()):
        batch_sizes = data['batch_sizes']
        means = data['means']
        mins = data['mins']
        means_ms = [m * 1000 for m in means]
        mins_ms = [m * 1000 for m in mins]

        print(f"\n{component_name}:")
        print(f"  Batch sizes: {list(batch_sizes)}")
        print(f"  Mean times (ms): {[f'{m:.4f}' for m in means_ms]}")
        print(f"  Min times (ms):  {[f'{m:.4f}' for m in mins_ms]}")
        print(
            f"  Mean - Min time: {min(means_ms):.4f} ms (batch_size={batch_sizes[means_ms.index(min(means_ms))]})")
        print(
            f"  Mean - Max time: {max(means_ms):.4f} ms (batch_size={batch_sizes[means_ms.index(max(means_ms))]})")
        print(
            f"  Min  - Min time: {min(mins_ms):.4f} ms (batch_size={batch_sizes[mins_ms.index(min(mins_ms))]})")
        print(
            f"  Min  - Max time: {max(mins_ms):.4f} ms (batch_size={batch_sizes[mins_ms.index(max(mins_ms))]})")


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 检查是否有命令行参数指定分析目录
    if len(sys.argv) > 1:
        stats_dir = sys.argv[1]
        if not os.path.exists(stats_dir):
            print(f"指定的目录不存在: {stats_dir}")
            sys.exit(1)
    else:
        # 查找分析结果目录
        results_dir = os.path.join(script_dir, 'results')

        # 找到最新的component_times_output目录
        if os.path.exists(results_dir):
            # 使用glob搜索（处理长目录名被截断的情况）
            result_dirs = []
            for entry in os.listdir(results_dir):
                full_path = os.path.join(results_dir, entry)
                if os.path.isdir(full_path) and entry.startswith('component_times_output_'):
                    test_stats_dir = os.path.join(
                        full_path, 'cputime', 'analysis')
                    if os.path.exists(test_stats_dir):
                        result_dirs.append((full_path, test_stats_dir))

            if result_dirs:
                # 找到最新的结果
                latest_result, stats_dir = sorted(result_dirs)[-1]
            else:
                print(f"未找到有效的component_times_output目录在: {results_dir}")
                sys.exit(1)
        else:
            print(f"结果目录不存在: {results_dir}")
            sys.exit(1)

    print(f"正在分析目录: {stats_dir}")

    # 获取输出目录（在数据目录的同级目录下创建output文件夹）
    # 如果数据在 .../results/component_times_output_xxx/cputime/analysis/
    # 则输出到 .../results/component_times_output_xxx/cputime/analysis/
    output_dir = stats_dir

    # 如果没有写入权限，尝试在脚本目录创建output子目录
    if not os.access(output_dir, os.W_OK):
        output_dir = os.path.join(script_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        print(f"原始输出目录无写入权限，改为输出到: {output_dir}")

    # 读取统计数据
    data_dict = read_component_statistics(stats_dir)

    if data_dict:
        # 打印摘要
        print_summary(data_dict)

        # 绘制mean time合并图表
        print("\n生成Mean Time折线图...")
        plot_mean_time_vs_batch_size(data_dict, output_dir)

        # 绘制mean time单独图表
        print("生成Mean Time个别组件分析图...")
        plot_individual_components(data_dict, output_dir)

        # 绘制min time合并图表
        print("生成Min Time折线图...")
        plot_min_time_vs_batch_size(data_dict, output_dir)

        # 绘制min time单独图表
        print("生成Min Time个别组件分析图...")
        plot_individual_components_min_time(data_dict, output_dir)

        print("\n分析完成！")
