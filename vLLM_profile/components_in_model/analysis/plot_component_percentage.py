#!/usr/bin/env python3
"""
绘制组件百分比分析图脚本（基于 component_percentage_summary.csv）

- 按 batch_size 分组：每个 batch_size 一张图，显示不同 input_len 的 component 百分比（柱状图）
- 按 input_len 分组：每个 input_len 一张图，显示不同 batch_size 的 component 百分比（柱状图）
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(csv_file):
    """读取CSV文件"""
    print(f"读取 CSV 文件: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"数据行数: {len(df)}")
    print(f"列: {list(df.columns)}")
    return df


def plot_by_batch_size(df, batch_size, output_dir):
    """为指定的 batch_size 绘制柱状图，比较不同 input_len 的 component 百分比
    
    Args:
        df: 数据框
        batch_size: 要绘制的 batch_size
        output_dir: 输出目录
    """
    # 筛选该 batch_size 的数据
    data = df[df['batch_size'] == batch_size].copy()
    
    if len(data) == 0:
        print(f"  警告: batch_size={batch_size} 没有数据")
        return
    
    # 按 input_len 排序
    data = data.sort_values('input_len')
    input_lens = data['input_len'].values
    
    # 定义组件列名和显示名称
    components = {
        'mlp_percentage': 'MLP',
        'qkv_projection_and_rope_percentage': 'QKV & RoPE',
        'attention_forward_percentage': 'Attention Forward',
        'output_projection_percentage': 'Output Projection'
    }
    
    # 准备数据
    component_data = {}
    for col, label in components.items():
        component_data[label] = data[col].values
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 设置柱状图参数
    x = np.arange(len(input_lens))
    width = 0.2  # 每个柱子的宽度
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 不同颜色
    
    # 绘制每个组件的柱状图
    for idx, (label, values) in enumerate(component_data.items()):
        offset = (idx - len(components) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=colors[idx % len(colors)], alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Input Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Component Percentage Comparison (Batch Size = {batch_size})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{il}' for il in input_lens])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # 添加数值标签（只在柱子足够高时显示）
    for idx, (label, values) in enumerate(component_data.items()):
        offset = (idx - len(components) / 2 + 0.5) * width
        for i, v in enumerate(values):
            if v > 5:  # 只在百分比大于5%时显示标签，避免标签过多
                ax.text(i + offset, v + 2, f'{v:.1f}%', 
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'component_percentage_batch_size_{batch_size}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {os.path.basename(output_path)}")
    plt.close(fig)


def plot_by_input_len(df, input_len, output_dir):
    """为指定的 input_len 绘制柱状图，比较不同 batch_size 的 component 百分比
    
    Args:
        df: 数据框
        input_len: 要绘制的 input_len
        output_dir: 输出目录
    """
    # 筛选该 input_len 的数据
    data = df[df['input_len'] == input_len].copy()
    
    if len(data) == 0:
        print(f"  警告: input_len={input_len} 没有数据")
        return
    
    # 按 batch_size 排序
    data = data.sort_values('batch_size')
    batch_sizes = data['batch_size'].values
    
    # 定义组件列名和显示名称
    components = {
        'mlp_percentage': 'MLP',
        'qkv_projection_and_rope_percentage': 'QKV & RoPE',
        'attention_forward_percentage': 'Attention Forward',
        'output_projection_percentage': 'Output Projection'
    }
    
    # 准备数据
    component_data = {}
    for col, label in components.items():
        component_data[label] = data[col].values
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 设置柱状图参数
    x = np.arange(len(batch_sizes))
    width = 0.2  # 每个柱子的宽度
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 不同颜色
    
    # 绘制每个组件的柱状图
    for idx, (label, values) in enumerate(component_data.items()):
        offset = (idx - len(components) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=colors[idx % len(colors)], alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Component Percentage Comparison (Input Length = {input_len})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{bs}' for bs in batch_sizes])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # 添加数值标签（只在柱子足够高时显示）
    for idx, (label, values) in enumerate(component_data.items()):
        offset = (idx - len(components) / 2 + 0.5) * width
        for i, v in enumerate(values):
            if v > 5:  # 只在百分比大于5%时显示标签，避免标签过多
                ax.text(i + offset, v + 2, f'{v:.1f}%', 
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'component_percentage_input_len_{input_len}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {os.path.basename(output_path)}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_component_percentage.py <csv_file> [output_dir]")
        print("示例: python plot_component_percentage.py component_percentage_summary.csv ./component_percentage_plots/")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(csv_file)[0] + '_plots'
    
    # 读取数据
    df = read_data(csv_file)
    
    # 获取所有唯一的 batch_size 和 input_len
    batch_sizes = sorted(df['batch_size'].dropna().unique())
    input_lens = sorted(df['input_len'].dropna().unique())
    
    print(f"\n发现 {len(batch_sizes)} 个 batch sizes: {batch_sizes}")
    print(f"发现 {len(input_lens)} 个 input lengths: {input_lens}")
    
    # 创建输出目录
    batch_size_dir = os.path.join(output_dir, 'by_batch_size')
    input_len_dir = os.path.join(output_dir, 'by_input_len')
    
    # 按 batch_size 分组绘制
    print(f"\n按 batch_size 分组绘制图表...")
    for batch_size in batch_sizes:
        print(f"  生成 batch_size={batch_size} 的图表...")
        plot_by_batch_size(df, batch_size, batch_size_dir)
    
    # 按 input_len 分组绘制
    print(f"\n按 input_len 分组绘制图表...")
    for input_len in input_lens:
        print(f"  生成 input_len={input_len} 的图表...")
        plot_by_input_len(df, input_len, input_len_dir)
    
    print(f"\n完成！所有图表已保存到: {output_dir}")
    print(f"  - 按 batch_size 分组的图表: {batch_size_dir}")
    print(f"  - 按 input_len 分组的图表: {input_len_dir}")


if __name__ == '__main__':
    main()

