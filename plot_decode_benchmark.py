#!/usr/bin/env python3
"""
绘制 TPOT 折线图脚本（基于 decode_benchmark.csv）

- 三个图：
    1. TPOT vs Batch Size（每条线对应一个 input_tokens）
    2. TPOT vs Input Tokens（每条线对应一个 batch_size）
    3. TPOT vs (batch_size × input_tokens)（每条线对应一个 batch_size）

- 对于相同组合，会选取 std_tpot_ms 最小的那条数据
- 每张图分别保存线性 x 轴与对数 x 轴两个版本
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(df):
    """准备数据：按 (bs, actual_input_len) 分组，计算 mean 和 std"""
    # 重命名列以匹配参考代码的命名
    df_prep = df.copy()
    df_prep['batch_size'] = df_prep['bs']
    df_prep['input_tokens'] = df_prep['actual_input_len']
    df_prep['tpot_ms'] = df_prep['tpot']  # tpot 已经是毫秒单位
    
    # 按 (batch_size, input_tokens) 分组，计算统计量
    grouped = df_prep.groupby(['batch_size', 'input_tokens'])['tpot_ms'].agg([
        ('mean_tpot_ms', 'mean'),
        ('std_tpot_ms', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    # 如果只有一个数据点，std 会是 NaN，设为 0
    grouped['std_tpot_ms'] = grouped['std_tpot_ms'].fillna(0)
    
    return grouped


def plot_tpot_by_batch_size(df, use_log=False):
    """按 batch_size 绘制折线图，同一 input_tokens 为一条线
    
    对于相同的 (batch_size, input_tokens)，只取 std_tpot_ms 最小的那条数据
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 按 (batch_size, input_tokens) 分组，每组只保留 std_tpot_ms 最小的一条
    df_filtered = df.loc[df.groupby(['batch_size', 'input_tokens'])[
        'std_tpot_ms'].idxmin()]
    
    # 按 input_tokens 分组
    input_tokens_list = sorted(df_filtered['input_tokens'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(input_tokens_list))))
    
    for idx, input_tokens in enumerate(input_tokens_list):
        data = df_filtered[df_filtered['input_tokens']
                           == input_tokens].sort_values('batch_size')
        ax.plot(data['batch_size'], data['mean_tpot_ms'],
                marker='o', label=f'input_tokens={input_tokens}',
                color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean TPOT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('TPOT vs Batch Size (each input_tokens is a separate line, selecting the minimum std_tpot_ms)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_tpot_by_input_tokens(df, use_log=False):
    """按 input_tokens 绘制折线图，同一 batch_size 为一条线
    
    对于相同的 (batch_size, input_tokens)，只取 std_tpot_ms 最小的那条数据
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 按 (batch_size, input_tokens) 分组，每组只保留 std_tpot_ms 最小的一条
    df_filtered = df.loc[df.groupby(['batch_size', 'input_tokens'])[
        'std_tpot_ms'].idxmin()]
    
    # 按 batch_size 分组
    batch_sizes = sorted(df_filtered['batch_size'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(batch_sizes))))
    
    for idx, bs in enumerate(batch_sizes):
        data = df_filtered[df_filtered['batch_size']
                           == bs].sort_values('input_tokens')
        ax.plot(data['input_tokens'], data['mean_tpot_ms'],
                marker='s', label=f'batch_size={bs}',
                color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('Input Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean TPOT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('TPOT vs Input Tokens (each batch_size is a separate line, selecting the minimum std_tpot_ms)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_tpot_by_combined(df, use_log=False):
    """按 batch_size × input_tokens 绘制折线图
    
    对于相同的 (batch_size × input_tokens)，只取 std_tpot_ms 最小的那条数据
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算 combined = batch_size × input_tokens
    df_copy = df.copy()
    df_copy['combined'] = df_copy['batch_size'] * df_copy['input_tokens']
    
    # 按 (batch_size, combined) 分组，每组只保留 std_tpot_ms 最小的一条
    df_filtered = df_copy.loc[df_copy.groupby(['batch_size', 'combined'])[
        'std_tpot_ms'].idxmin()]
    
    # 按 batch_size 分组，用不同的线型和颜色区分
    batch_sizes = sorted(df_filtered['batch_size'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(batch_sizes))))
    
    for idx, bs in enumerate(batch_sizes):
        data = df_filtered[df_filtered['batch_size']
                           == bs].sort_values('combined')
        ax.plot(data['combined'], data['mean_tpot_ms'],
                marker='^', label=f'batch_size={bs}',
                color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('batch_size × input_tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean TPOT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('TPOT vs (batch_size × input_tokens) (each batch_size is a separate line, selecting the minimum std_tpot_ms)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def save_versions(plot_func, df, output_dir, base_name):
    """对给定 plot_func，生成 linear 与 log 两个版本并保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    for use_log, suffix in [(False, 'linear'), (True, 'log')]:
        fig = plot_func(df, use_log=use_log)
        path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"已保存: {path}")
        plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_decode_benchmark.py <csv_file> [output_dir]")
        print("示例: python plot_decode_benchmark.py decode_benchmark.csv ./plots/")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(csv_file)[0] + '_plots'
    
    # 读取 CSV
    print(f"读取 CSV 文件: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"数据行数: {len(df)}")
    
    # 准备数据
    print("准备数据...")
    df_prepared = prepare_data(df)
    print(f"准备后的数据行数: {len(df_prepared)}")
    
    if 'input_tokens' in df_prepared.columns:
        print(f"输入 tokens 种类: {sorted(df_prepared['input_tokens'].unique())}")
    if 'batch_size' in df_prepared.columns:
        print(f"Batch sizes: {sorted(df_prepared['batch_size'].unique())}")
    
    # 保存所有图的两个版本
    print("\n生成图表1: TPOT vs Batch Size (linear & log)...")
    save_versions(plot_tpot_by_batch_size, df_prepared,
                  output_dir, 'tpot_vs_batch_size')
    
    print("生成图表2: TPOT vs Input Tokens (linear & log)...")
    save_versions(plot_tpot_by_input_tokens, df_prepared,
                  output_dir, 'tpot_vs_input_tokens')
    
    print("生成图表3: TPOT vs (batch_size × input_tokens) (linear & log)...")
    save_versions(plot_tpot_by_combined, df_prepared, output_dir, 'tpot_vs_combined')
    
    print("\n完成！")


if __name__ == '__main__':
    main()




