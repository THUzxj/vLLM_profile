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


def plot_time_vs_batch_size(data_dict, output_dir=None, time_type='mean', layer_num=None):
    """
    绘制各个组件的 mean/min time 与 batch size 的关系

    :param data_dict: 组件统计数据
    :param output_dir: 输出目录
    :param time_type: 'mean' 或 'min'，控制绘制哪种时间
    :param layer_num: 指定layer编号（整数），如果指定则只绘制该layer的组件，并在文件名中包含layer number
    """
    if not data_dict:
        print("没有数据可绘制")
        return

    assert time_type in ('mean', 'min'), "time_type 必须是 'mean' 或 'min'"

    # 配置不同类型的绘图样式
    if time_type == 'mean':
        value_key = 'means'
        ylabel = 'Mean Time (ms)'
        title = 'Mean Time vs Batch Size for Different Components'
        filename = 'mean_time_vs_batch_size.png'
        marker = 'o'
    else:
        value_key = 'mins'
        ylabel = 'Min Time (ms)'
        title = 'Min Time vs Batch Size for Different Components'
        filename = 'min_time_vs_batch_size.png'
        marker = 's'

    # 如果指定了layer_num，在文件名和标题中添加layer信息
    if layer_num is not None:
        title = f"{title} - Layer {layer_num}"
        base, ext = os.path.splitext(filename)
        filename = f"{base}_layer{layer_num}{ext}"

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 排除 model_time 和各层 total，只绘制具体子组件
    for component_name, data in sorted(data_dict.items()):
        if 'model_time' in component_name or component_name.endswith('_total'):
            continue
        
        # 如果指定了layer_num，只绘制该layer的组件
        if layer_num is not None:
            if not component_name.startswith(f'layer_{layer_num}_'):
                continue
        
        batch_sizes = data['batch_sizes']
        values = data[value_key]

        # 转换为毫秒（从秒转换）
        values_ms = [v * 1000 for v in values]

        ax.plot(batch_sizes, values_ms, marker=marker,
                label=component_name, linewidth=2, markersize=6)

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")

    plt.show()


def plot_individual_components_time(data_dict, output_dir=None, time_type='mean', layer_num=None):
    """
    为每个主要组件绘制单独的 mean/min time 图表（更清晰的视图）

    :param data_dict: 组件统计数据
    :param output_dir: 输出目录
    :param time_type: 'mean' 或 'min'，控制绘制哪种时间
    :param layer_num: 指定layer编号（整数），如果指定则只绘制该layer的组件，并在文件名中包含layer number
    """
    if not data_dict:
        print("没有数据可绘制")
        return

    assert time_type in ('mean', 'min'), "time_type 必须是 'mean' 或 'min'"

    # 如果没有指定layer_num，默认使用layer_0
    if layer_num is None:
        layer_num = 0

    # 分组绘制主要组件
    # 参考 JSON 中的字段，新增 MoE 相关的三个组件：
    # moe_dispatch, moe_combine, moe_core -> 对应 layer_X_moe_dispatch 等
    main_components = [
        f'layer_{layer_num}_total',
        f'layer_{layer_num}_self_attention',
        f'layer_{layer_num}_mlp',
        f'layer_{layer_num}_attention_prepare',
        f'layer_{layer_num}_attention_core',
        f'layer_{layer_num}_mlp_gate',
        f'layer_{layer_num}_mlp_experts',
        f'layer_{layer_num}_moe_dispatch',
        f'layer_{layer_num}_moe_combine',
        f'layer_{layer_num}_moe_core',
    ]

    # 根据 time_type 配置
    if time_type == 'mean':
        value_key = 'means'
        ylabel = 'Mean Time (ms)'
        suptitle = f'Mean Time vs Batch Size - Individual Components - Layer {layer_num}'
        filename = f'individual_components_analysis_layer{layer_num}.png'
        marker = 'o'
        color = 'blue'
    else:
        value_key = 'mins'
        ylabel = 'Min Time (ms)'
        suptitle = f'Min Time vs Batch Size - Individual Components - Layer {layer_num}'
        filename = f'individual_components_min_time_analysis_layer{layer_num}.png'
        marker = 's'
        color = 'green'

    # 根据组件数量自适应子图网格
    import math
    n_components = len(main_components)
    n_cols = 4
    n_rows = math.ceil(n_components / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = axes.flatten()

    for idx, component in enumerate(main_components):
        if component in data_dict:
            ax = axes[idx]
            batch_sizes = data_dict[component]['batch_sizes']
            values = data_dict[component][value_key]
            values_ms = [v * 1000 for v in values]

            ax.plot(batch_sizes, values_ms, marker=marker,
                    color=color, linewidth=2, markersize=6)
            ax.set_xlabel('Batch Size', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(component, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(b) for b in batch_sizes], rotation=45)
            ax.set_ylim(bottom=0)  # 从0开始显示Y轴

    # 隐藏未使用的子图
    if len(main_components) < len(axes):
        for ax in axes[len(main_components):]:
            ax.axis('off')

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"个别组件分析图已保存到: {output_path}")

    plt.show()


def plot_time_components_bar(
    data_dict,
    output_dir=None,
    time_type='mean',
    components=None,
    label='',
    layer_num=None,
):
    """
    绘制柱状图：不同 batch size 下各个组件的 mean/min time
    x 轴为 batch size，每个 batch size 这一组里有多个 component 的柱子

    :param data_dict: 组件统计数据
    :param output_dir: 输出目录
    :param time_type: 'mean' 或 'min'，控制绘制哪种时间
    :param components: 要绘制的组件名列表；为 None 时默认使用所有符合条件的组件
    :param label: 额外加入到标题和文件名中的标签，用于区分不同层级/分组
    :param layer_num: 指定layer编号（整数），如果指定则只绘制该layer的组件，并在文件名中包含layer number
    """
    if not data_dict:
        print("没有数据可绘制")
        return

    assert time_type in ('mean', 'min'), "time_type 必须是 'mean' 或 'min'"

    import numpy as np

    # 排除 model_time 和各层 total，只绘制具体组件（例如 layer_x_xxx）
    if components is None:
        components = sorted([
                name
                for name in data_dict.keys()
                if 'model_time' not in name and not name.endswith('_total')
                  and not name.endswith('_self_attention') and not name.endswith('_mlp')
            ])
    else:
        # 只保留在 data_dict 中存在的组件
        components = [name for name in components if name in data_dict]

    # 如果指定了layer_num，只保留该layer的组件
    if layer_num is not None:
        components = [name for name in components if name.startswith(f'layer_{layer_num}_')]

    if not components:
        print("没有可用的组件数据")
        return

    # 收集所有 batch size（去重后排序）
    all_batch_sizes = set()
    for name in components:
        bs = data_dict[name]['batch_sizes']
        all_batch_sizes.update(bs)
    all_batch_sizes = sorted(all_batch_sizes)

    # 根据 time_type 选择数据和配置
    value_key = 'means' if time_type == 'mean' else 'mins'
    ylabel = 'Mean Time (ms)' if time_type == 'mean' else 'Min Time (ms)'
    title = (
        'Mean Time of Components for Each Batch Size (Bar Plot)'
        if time_type == 'mean'
        else 'Min Time of Components for Each Batch Size (Bar Plot)'
    )
    filename = (
        'mean_time_components_bar.png'
        if time_type == 'mean'
        else 'min_time_components_bar.png'
    )

    # 如果传入了 label，则追加到标题和文件名中
    if label:
        title = f"{title} - {label}"
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{label}{ext}"
    
    # 如果指定了layer_num，在文件名和标题中添加layer信息
    if layer_num is not None:
        title = f"{title} - Layer {layer_num}"
        base, ext = os.path.splitext(filename)
        filename = f"{base}_layer{layer_num}{ext}"

    # 构建值矩阵 (n_components, n_batch)
    value_matrix = []
    for name in components:
        bs = data_dict[name]['batch_sizes']
        values = data_dict[name][value_key]
        values_ms = [v * 1000 for v in values]
        value_map = {b: v for b, v in zip(bs, values_ms)}
        row = [value_map.get(b, 0.0) for b in all_batch_sizes]
        value_matrix.append(row)

    value_matrix = np.array(value_matrix)  # shape: (n_components, n_batch)

    # 现在以 batch size 作为 x 轴，一个 group 是一个 batch size，组内是多个 component
    x = np.arange(len(all_batch_sizes))
    total_width = 0.8
    n_components = len(components)
    bar_width = total_width / n_components

    fig, ax = plt.subplots(figsize=(max(12, len(all_batch_sizes) * 0.8), 8))

    for i, name in enumerate(components):
        positions = x - total_width / 2 + i * bar_width + bar_width / 2
        ax.bar(positions, value_matrix[i, :], width=bar_width, label=name)

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(
        title,
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in all_batch_sizes], rotation=0, ha='center')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(title='Component', fontsize=9)

    plt.tight_layout()

    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{'Mean' if time_type == 'mean' else 'Min'} Time 柱状图已保存到: {output_path}")

    plt.show()


def extract_layer_numbers(data_dict):
    """
    从data_dict中提取所有layer编号
    
    :param data_dict: 组件统计数据
    :return: 排序后的layer编号列表
    """
    import re
    layer_numbers = set()
    
    for component_name in data_dict.keys():
        # 匹配 layer_X_ 格式
        match = re.match(r'layer_(\d+)_', component_name)
        if match:
            layer_numbers.add(int(match.group(1)))
    
    return sorted(layer_numbers)


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

    # 检查是否有命令行参数指定分析目录和layer number
    layer_num = None
    if len(sys.argv) > 1:
        stats_dir = sys.argv[1]
        if not os.path.exists(stats_dir):
            print(f"指定的目录不存在: {stats_dir}")
            sys.exit(1)
        # 检查是否有第二个参数指定layer number
        if len(sys.argv) > 2:
            try:
                layer_num = int(sys.argv[2])
            except ValueError:
                print(f"无效的layer number: {sys.argv[2]}，应为整数")
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

        # 如果没有指定layer_num，自动检测所有layer
        if layer_num is None:
            layer_numbers = extract_layer_numbers(data_dict)
            if not layer_numbers:
                print("未检测到layer编号，将绘制所有组件")
                layer_numbers = [None]  # 使用None表示绘制所有layer
        else:
            layer_numbers = [layer_num]

        # 为每个layer绘制图表
        for current_layer_num in layer_numbers:
            layer_suffix = f" (Layer {current_layer_num})" if current_layer_num is not None else ""
            print(f"\n{'='*80}")
            if current_layer_num is not None:
                print(f"正在处理 Layer {current_layer_num}")
            else:
                print("正在处理所有Layer")
            print(f"{'='*80}")

            # 绘制 mean time 合并折线图
            print("\n生成Mean Time折线图...")
            plot_time_vs_batch_size(data_dict, output_dir, time_type='mean', layer_num=current_layer_num)

            # 绘制 mean time 个别组件折线图
            print("生成Mean Time个别组件分析图...")
            plot_individual_components_time(data_dict, output_dir, time_type='mean', layer_num=current_layer_num)

            # 绘制 min time 合并折线图
            print("生成Min Time折线图...")
            plot_time_vs_batch_size(data_dict, output_dir, time_type='min', layer_num=current_layer_num)

            # 绘制 min time 个别组件折线图
            print("生成Min Time个别组件分析图...")
            plot_individual_components_time(data_dict, output_dir, time_type='min', layer_num=current_layer_num)

            # 按层级绘制 mean / min time 柱状图（组件 × batch size）
            # 自动根据 component name 中的 layer 前缀进行分组
            if current_layer_num is not None:
                components_groups = {
                    "coarse": [f"layer_{current_layer_num}_self_attention", f"layer_{current_layer_num}_mlp"],
                    "detailed": [
                        f"layer_{current_layer_num}_attention_prepare",
                        f"layer_{current_layer_num}_attention_core",
                        f"layer_{current_layer_num}_mlp_gate",
                        f"layer_{current_layer_num}_moe_dispatch",
                        f"layer_{current_layer_num}_moe_core",
                        f"layer_{current_layer_num}_moe_combine"
                    ],
                }
            else:
                # 如果没有指定layer，使用layer_0作为默认值
                components_groups = {
                    "coarse": ["layer_0_self_attention", "layer_0_mlp"],
                    "detailed": ["layer_0_attention_prepare", "layer_0_attention_core", "layer_0_mlp_gate", "layer_0_moe_dispatch", "layer_0_moe_core", "layer_0_moe_combine"],
                }
            
            for label, components_group in components_groups.items():
                plot_time_components_bar(
                    data_dict,
                    output_dir,
                    time_type='mean',
                    components=components_group,
                    label=label,
                    layer_num=current_layer_num,
                )

        print("\n分析完成！")
