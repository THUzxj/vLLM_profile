#!/usr/bin/env python3
"""
绘制组件时间分析图脚本（基于 all_layer_times_statistics.csv）

- 按 Component 分组
- 对每个 Component 生成三个图：
    1. Mean vs Batch Size（每条线对应一个 input_len）
    2. Mean vs Input Len（每条线对应一个 batch_size）
    3. Mean vs (batch_size × input_len)（每条线对应一个 batch_size）

- 每张图分别保存线性 x 轴与对数 x 轴两个版本（matplotlib 版本）
- 另外生成 Plotly 版本（交互式 HTML 图表，方便交互分析）
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go


def prepare_data(df):
    """准备数据：按 (model_name, input_len, batch_size, component) 分组"""
    df_prep = df.copy()
    
    # 确保关键列存在
    required_cols = ['model_name', 'input_len', 'batch_size', 'Component', 'Mean']
    for col in required_cols:
        if col not in df_prep.columns:
            print(f"警告: 缺少列 {col}")
            return None
    
    return df_prep


def calculate_component_ratios(df):
    """计算 attention_forward 与其他部分占 total_layer_time 的比例
    
    返回新的 dataframe，包含：
    - attention_forward_ratio: attention_forward / total_layer_time
    - others_ratio: (total_layer_time - attention_forward) / total_layer_time
    """
    ratios_list = []
    
    # 获取每个 (model_name, input_len, batch_size) 组合的 attention_forward 和 total_layer_time
    for (model_name, input_len, batch_size), group in df.groupby(['model_name', 'input_len', 'batch_size']):
        total_row = group[group['Component'] == 'total_layer_time']
        attn_row = group[group['Component'] == 'attention_forward']
        mlp_row = group[group['Component'] == 'mlp']
        
        if len(total_row) > 0 and len(attn_row) > 0:
            total_time = total_row['Mean'].values[0]
            attn_time = attn_row['Mean'].values[0]
            mlp_time = mlp_row['Mean'].values[0] if len(mlp_row) > 0 else 0.0
            
            if total_time > 0:
                attn_ratio = attn_time / total_time
                mlp_ratio = mlp_time / total_time if mlp_time > 0 else 0.0
                others_ratio = 1.0 - attn_ratio
            else:
                attn_ratio = 0
                mlp_ratio = 0
                others_ratio = 0
            
            ratios_list.append({
                'model_name': model_name,
                'input_len': input_len,
                'batch_size': batch_size,
                'attention_forward_ratio': attn_ratio,
                'others_ratio': others_ratio,
                'mlp_ratio': mlp_ratio
            })
    
    return pd.DataFrame(ratios_list)


def plot_mean_by_batch_size(df, component, use_log=False):
    """按 batch_size 绘制折线图，同一 input_len 为一条线
    
    Args:
        df: 包含该 component 的数据
        component: 组件名称
        use_log: 是否使用对数 x 轴
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 按 input_len 分组
    input_lens = sorted(df['input_len'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(input_lens))))
    
    for idx, input_len in enumerate(input_lens):
        data = df[df['input_len'] == input_len].sort_values('batch_size')
        if len(data) > 0:
            ax.plot(data['batch_size'], data['Mean'],
                    marker='o', label=f'input_len={input_len}',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{component} Mean Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'{component}: Mean vs Batch Size',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_mean_by_input_len(df, component, use_log=False):
    """按 input_len 绘制折线图，同一 batch_size 为一条线
    
    Args:
        df: 包含该 component 的数据
        component: 组件名称
        use_log: 是否使用对数 x 轴
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 按 batch_size 分组
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(batch_sizes))))
    
    for idx, bs in enumerate(batch_sizes):
        data = df[df['batch_size'] == bs].sort_values('input_len')
        if len(data) > 0:
            ax.plot(data['input_len'], data['Mean'],
                    marker='s', label=f'batch_size={bs}',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('Input Length', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{component} Mean Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'{component}: Mean vs Input Length',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_mean_by_combined(df, component, use_log=False):
    """按 batch_size × input_len 绘制折线图
    
    Args:
        df: 包含该 component 的数据
        component: 组件名称
        use_log: 是否使用对数 x 轴
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算 combined = batch_size × input_len
    df_copy = df.copy()
    df_copy['combined'] = df_copy['batch_size'] * df_copy['input_len']
    
    # 按 batch_size 分组
    batch_sizes = sorted(df_copy['batch_size'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(batch_sizes))))
    
    for idx, bs in enumerate(batch_sizes):
        data = df_copy[df_copy['batch_size'] == bs].sort_values('combined')
        if len(data) > 0:
            ax.plot(data['combined'], data['Mean'],
                    marker='^', label=f'batch_size={bs}',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('batch_size × input_len', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{component} Mean Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'{component}: Mean vs (batch_size × input_len)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_ratio_by_batch_size(df, use_log=False):
    """按 batch_size 绘制占比折线图，attention_forward 和 others
    
    Args:
        df: 包含占比数据的 dataframe
        use_log: 是否使用对数 x 轴
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 按 input_len 分组
    input_lens = sorted(df['input_len'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(input_lens))))
    
    for idx, input_len in enumerate(input_lens):
        data = df[df['input_len'] == input_len].sort_values('batch_size')
        if len(data) > 0:
            ax.plot(data['batch_size'], data['attention_forward_ratio'] * 100,
                    marker='o', label=f'attention_forward (input_len={input_len})',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='-')
            ax.plot(data['batch_size'], data['mlp_ratio'] * 100,
                    marker='s', label=f'MLP (input_len={input_len})',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='--')
            # ax.plot(data['batch_size'], data['others_ratio'] * 100,
            #         marker='s', label=f'others (input_len={input_len})',
            #         color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='--')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Ratio vs Batch Size (attention_forward vs others)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_ratio_by_input_len(df, use_log=False):
    """按 input_len 绘制占比折线图，attention_forward 和 others
    
    Args:
        df: 包含占比数据的 dataframe
        use_log: 是否使用对数 x 轴
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 按 batch_size 分组
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(batch_sizes))))
    
    for idx, bs in enumerate(batch_sizes):
        data = df[df['batch_size'] == bs].sort_values('input_len')
        if len(data) > 0:
            ax.plot(data['input_len'], data['attention_forward_ratio'] * 100,
                    marker='o', label=f'attention_forward (batch_size={bs})',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='-')
            ax.plot(data['input_len'], data['mlp_ratio'] * 100,
                    marker='s', label=f'MLP (batch_size={bs})',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='--')
            # ax.plot(data['input_len'], data['others_ratio'] * 100,
            #         marker='s', label=f'others (batch_size={bs})',
            #         color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='--')
    
    ax.set_xlabel('Input Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Ratio vs Input Length (attention_forward vs others)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_ratio_by_combined(df, use_log=False):
    """按 batch_size × input_len 绘制占比折线图
    
    Args:
        df: 包含占比数据的 dataframe
        use_log: 是否使用对数 x 轴
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算 combined = batch_size × input_len
    df_copy = df.copy()
    df_copy['combined'] = df_copy['batch_size'] * df_copy['input_len']
    
    # 按 batch_size 分组
    batch_sizes = sorted(df_copy['batch_size'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(batch_sizes))))
    
    for idx, bs in enumerate(batch_sizes):
        data = df_copy[df_copy['batch_size'] == bs].sort_values('combined')
        if len(data) > 0:
            ax.plot(data['combined'], data['attention_forward_ratio'] * 100,
                    marker='^', label=f'attention_forward (batch_size={bs})',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='-')
            ax.plot(data['combined'], data['mlp_ratio'] * 100,
                    marker='v', label=f'MLP (batch_size={bs})',
                    color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='--')
            # ax.plot(data['combined'], data['others_ratio'] * 100,
            #         marker='v', label=f'others (batch_size={bs})',
            #         color=colors[idx % len(colors)], linewidth=2, markersize=6, linestyle='--')
    
    ax.set_xlabel('batch_size × input_len', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Ratio vs (batch_size × input_len) (attention_forward vs others)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    if use_log:
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_mean_by_batch_size_sns(df, component):
    """Seaborn 版本：按 batch_size 绘制折线图，同一 input_len 为一条线"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 准备数据，将 input_len 转换为字符串以便 seaborn 识别为分类变量
    df_plot = df.copy()
    df_plot['input_len'] = df_plot['input_len'].astype(str)
    
    sns.lineplot(data=df_plot, x='batch_size', y='Mean', hue='input_len',
                 marker='o', markersize=8, linewidth=2.5, ax=ax)
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{component} Mean Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'{component}: Mean vs Batch Size (Seaborn)', fontsize=14, fontweight='bold')
    ax.legend(title='Input Length', loc='best', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mean_by_input_len_sns(df, component):
    """Seaborn 版本：按 input_len 绘制折线图，同一 batch_size 为一条线"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 准备数据，将 batch_size 转换为字符串以便 seaborn 识别为分类变量
    df_plot = df.copy()
    df_plot['batch_size'] = df_plot['batch_size'].astype(str)
    
    sns.lineplot(data=df_plot, x='input_len', y='Mean', hue='batch_size',
                 marker='s', markersize=8, linewidth=2.5, ax=ax)
    
    ax.set_xlabel('Input Length', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{component} Mean Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'{component}: Mean vs Input Length (Seaborn)', fontsize=14, fontweight='bold')
    ax.legend(title='Batch Size', loc='best', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mean_by_combined_sns(df, component):
    """Seaborn 版本：按 batch_size × input_len 绘制折线图"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算 combined = batch_size × input_len
    df_plot = df.copy()
    df_plot['combined'] = df_plot['batch_size'] * df_plot['input_len']
    df_plot['batch_size'] = df_plot['batch_size'].astype(str)
    
    sns.lineplot(data=df_plot, x='combined', y='Mean', hue='batch_size',
                 marker='^', markersize=8, linewidth=2.5, ax=ax)
    
    ax.set_xlabel('batch_size × input_len', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{component} Mean Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'{component}: Mean vs (batch_size × input_len) (Seaborn)', fontsize=14, fontweight='bold')
    ax.legend(title='Batch Size', loc='best', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_ratio_by_batch_size_sns(df):
    """Seaborn 版本：按 batch_size 绘制占比折线图，attention_forward 和 MLP"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 准备数据：将数据从宽格式转换为长格式
    df_plot = df.copy()
    df_plot['input_len'] = df_plot['input_len'].astype(str)
    
    # 创建长格式数据
    ratio_data = []
    for _, row in df_plot.iterrows():
        ratio_data.append({
            'batch_size': row['batch_size'],
            'input_len': row['input_len'],
            'Ratio (%)': row['attention_forward_ratio'] * 100,
            'Component': 'attention_forward'
        })
        ratio_data.append({
            'batch_size': row['batch_size'],
            'input_len': row['input_len'],
            'Ratio (%)': row['mlp_ratio'] * 100,
            'Component': 'MLP'
        })
    
    df_long = pd.DataFrame(ratio_data)
    
    # 使用 hue 区分 Component，使用 style 区分 input_len
    sns.lineplot(data=df_long, x='batch_size', y='Ratio (%)', 
                 hue='Component', style='input_len',
                 marker='o', markersize=8, linewidth=2.5, ax=ax)
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Ratio vs Batch Size (attention_forward vs MLP) (Seaborn)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_ratio_by_input_len_sns(df):
    """Seaborn 版本：按 input_len 绘制占比折线图，attention_forward 和 MLP"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 准备数据：将数据从宽格式转换为长格式
    df_plot = df.copy()
    df_plot['batch_size'] = df_plot['batch_size'].astype(str)
    
    # 创建长格式数据
    ratio_data = []
    for _, row in df_plot.iterrows():
        ratio_data.append({
            'input_len': row['input_len'],
            'batch_size': row['batch_size'],
            'Ratio (%)': row['attention_forward_ratio'] * 100,
            'Component': 'attention_forward'
        })
        ratio_data.append({
            'input_len': row['input_len'],
            'batch_size': row['batch_size'],
            'Ratio (%)': row['mlp_ratio'] * 100,
            'Component': 'MLP'
        })
    
    df_long = pd.DataFrame(ratio_data)
    
    # 使用 hue 区分 Component，使用 style 区分 batch_size
    sns.lineplot(data=df_long, x='input_len', y='Ratio (%)', 
                 hue='Component', style='batch_size',
                 marker='s', markersize=8, linewidth=2.5, ax=ax)
    
    ax.set_xlabel('Input Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Ratio vs Input Length (attention_forward vs MLP) (Seaborn)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_ratio_by_combined_sns(df):
    """Seaborn 版本：按 batch_size × input_len 绘制占比折线图"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算 combined = batch_size × input_len
    df_plot = df.copy()
    df_plot['combined'] = df_plot['batch_size'] * df_plot['input_len']
    df_plot['batch_size'] = df_plot['batch_size'].astype(str)
    
    # 创建长格式数据
    ratio_data = []
    for _, row in df_plot.iterrows():
        ratio_data.append({
            'combined': row['combined'],
            'batch_size': row['batch_size'],
            'Ratio (%)': row['attention_forward_ratio'] * 100,
            'Component': 'attention_forward'
        })
        ratio_data.append({
            'combined': row['combined'],
            'batch_size': row['batch_size'],
            'Ratio (%)': row['mlp_ratio'] * 100,
            'Component': 'MLP'
        })
    
    df_long = pd.DataFrame(ratio_data)
    
    # 使用 hue 区分 Component，使用 style 区分 batch_size
    sns.lineplot(data=df_long, x='combined', y='Ratio (%)', 
                 hue='Component', style='batch_size',
                 marker='^', markersize=8, linewidth=2.5, ax=ax)
    
    ax.set_xlabel('batch_size × input_len', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Ratio vs (batch_size × input_len) (attention_forward vs MLP) (Seaborn)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ========== Plotly 版本的绘图函数 ==========

def plot_mean_by_batch_size_plotly(df, component):
    """Plotly 版本：按 batch_size 绘制折线图，同一 input_len 为一条线"""
    fig = go.Figure()
    
    # 按 input_len 分组
    input_lens = sorted(df['input_len'].unique())
    
    for idx, input_len in enumerate(input_lens):
        data = df[df['input_len'] == input_len].sort_values('batch_size')
        if len(data) > 0:
            fig.add_trace(go.Scatter(
                x=data['batch_size'],
                y=data['Mean'],
                mode='lines+markers',
                name=f'input_len={input_len}',
                marker=dict(size=8, symbol='circle'),
                line=dict(width=2.5),
                hovertemplate=f'<b>input_len={input_len}</b><br>' +
                              'Batch Size: %{x}<br>' +
                              f'{component} Mean Time: %{{y:.6f}}s<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=f'{component}: Mean vs Batch Size (Plotly)', font=dict(size=16, family='Arial Black')),
        xaxis=dict(title='Batch Size'),
        yaxis=dict(title=f'{component} Mean Time (s)'),
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=1000,
        legend=dict(font=dict(size=10))
    )
    
    return fig


def plot_mean_by_input_len_plotly(df, component):
    """Plotly 版本：按 input_len 绘制折线图，同一 batch_size 为一条线"""
    fig = go.Figure()
    
    # 按 batch_size 分组
    batch_sizes = sorted(df['batch_size'].unique())
    
    for idx, bs in enumerate(batch_sizes):
        data = df[df['batch_size'] == bs].sort_values('input_len')
        if len(data) > 0:
            fig.add_trace(go.Scatter(
                x=data['input_len'],
                y=data['Mean'],
                mode='lines+markers',
                name=f'batch_size={bs}',
                marker=dict(size=8, symbol='square'),
                line=dict(width=2.5),
                hovertemplate=f'<b>batch_size={bs}</b><br>' +
                              'Input Length: %{x}<br>' +
                              f'{component} Mean Time: %{{y:.6f}}s<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=f'{component}: Mean vs Input Length (Plotly)', font=dict(size=16, family='Arial Black')),
        xaxis=dict(title='Input Length'),
        yaxis=dict(title=f'{component} Mean Time (s)'),
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=1000,
        legend=dict(font=dict(size=10))
    )
    
    return fig


def plot_mean_by_combined_plotly(df, component):
    """Plotly 版本：按 batch_size × input_len 绘制折线图"""
    fig = go.Figure()
    
    # 计算 combined = batch_size × input_len
    df_copy = df.copy()
    df_copy['combined'] = df_copy['batch_size'] * df_copy['input_len']
    
    # 按 batch_size 分组
    batch_sizes = sorted(df_copy['batch_size'].unique())
    
    for idx, bs in enumerate(batch_sizes):
        data = df_copy[df_copy['batch_size'] == bs].sort_values('combined')
        if len(data) > 0:
            fig.add_trace(go.Scatter(
                x=data['combined'],
                y=data['Mean'],
                mode='lines+markers',
                name=f'batch_size={bs}',
                marker=dict(size=8, symbol='triangle-up'),
                line=dict(width=2.5),
                hovertemplate=f'<b>batch_size={bs}</b><br>' +
                              'batch_size × input_len: %{x}<br>' +
                              f'{component} Mean Time: %{{y:.6f}}s<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=f'{component}: Mean vs (batch_size × input_len) (Plotly)', font=dict(size=16, family='Arial Black')),
        xaxis=dict(title='batch_size × input_len'),
        yaxis=dict(title=f'{component} Mean Time (s)'),
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=1000,
        legend=dict(font=dict(size=10))
    )
    
    return fig


def plot_ratio_by_batch_size_plotly(df):
    """Plotly 版本：按 batch_size 绘制占比折线图，attention_forward 和 MLP"""
    fig = go.Figure()
    
    # 按 input_len 分组
    input_lens = sorted(df['input_len'].unique())
    
    for idx, input_len in enumerate(input_lens):
        data = df[df['input_len'] == input_len].sort_values('batch_size')
        if len(data) > 0:
            # attention_forward
            fig.add_trace(go.Scatter(
                x=data['batch_size'],
                y=data['attention_forward_ratio'] * 100,
                mode='lines+markers',
                name=f'attention_forward (input_len={input_len})',
                marker=dict(size=8, symbol='circle'),
                line=dict(width=2.5, dash='solid'),
                hovertemplate=f'<b>attention_forward (input_len={input_len})</b><br>' +
                              'Batch Size: %{x}<br>' +
                              'Ratio: %{y:.2f}%<extra></extra>'
            ))
            # MLP
            fig.add_trace(go.Scatter(
                x=data['batch_size'],
                y=data['mlp_ratio'] * 100,
                mode='lines+markers',
                name=f'MLP (input_len={input_len})',
                marker=dict(size=8, symbol='square'),
                line=dict(width=2.5, dash='dash'),
                hovertemplate=f'<b>MLP (input_len={input_len})</b><br>' +
                              'Batch Size: %{x}<br>' +
                              'Ratio: %{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text='Component Ratio vs Batch Size (attention_forward vs MLP) (Plotly)', 
                  font=dict(size=16, family='Arial Black')),
        xaxis=dict(title='Batch Size'),
        yaxis=dict(title='Ratio (%)', range=[0, 100]),
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=1000,
        legend=dict(font=dict(size=9), orientation='v')
    )
    
    return fig


def plot_ratio_by_input_len_plotly(df):
    """Plotly 版本：按 input_len 绘制占比折线图，attention_forward 和 MLP"""
    fig = go.Figure()
    
    # 按 batch_size 分组
    batch_sizes = sorted(df['batch_size'].unique())
    
    for idx, bs in enumerate(batch_sizes):
        data = df[df['batch_size'] == bs].sort_values('input_len')
        if len(data) > 0:
            # attention_forward
            fig.add_trace(go.Scatter(
                x=data['input_len'],
                y=data['attention_forward_ratio'] * 100,
                mode='lines+markers',
                name=f'attention_forward (batch_size={bs})',
                marker=dict(size=8, symbol='circle'),
                line=dict(width=2.5, dash='solid'),
                hovertemplate=f'<b>attention_forward (batch_size={bs})</b><br>' +
                              'Input Length: %{x}<br>' +
                              'Ratio: %{y:.2f}%<extra></extra>'
            ))
            # MLP
            fig.add_trace(go.Scatter(
                x=data['input_len'],
                y=data['mlp_ratio'] * 100,
                mode='lines+markers',
                name=f'MLP (batch_size={bs})',
                marker=dict(size=8, symbol='square'),
                line=dict(width=2.5, dash='dash'),
                hovertemplate=f'<b>MLP (batch_size={bs})</b><br>' +
                              'Input Length: %{x}<br>' +
                              'Ratio: %{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text='Component Ratio vs Input Length (attention_forward vs MLP) (Plotly)', 
                  font=dict(size=16, family='Arial Black')),
        xaxis=dict(title='Input Length'),
        yaxis=dict(title='Ratio (%)', range=[0, 100]),
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=1000,
        legend=dict(font=dict(size=9), orientation='v')
    )
    
    return fig


def plot_ratio_by_combined_plotly(df):
    """Plotly 版本：按 batch_size × input_len 绘制占比折线图"""
    fig = go.Figure()
    
    # 计算 combined = batch_size × input_len
    df_copy = df.copy()
    df_copy['combined'] = df_copy['batch_size'] * df_copy['input_len']
    
    # 按 batch_size 分组
    batch_sizes = sorted(df_copy['batch_size'].unique())
    
    for idx, bs in enumerate(batch_sizes):
        data = df_copy[df_copy['batch_size'] == bs].sort_values('combined')
        if len(data) > 0:
            # attention_forward
            fig.add_trace(go.Scatter(
                x=data['combined'],
                y=data['attention_forward_ratio'] * 100,
                mode='lines+markers',
                name=f'attention_forward (batch_size={bs})',
                marker=dict(size=8, symbol='triangle-up'),
                line=dict(width=2.5, dash='solid'),
                hovertemplate=f'<b>attention_forward (batch_size={bs})</b><br>' +
                              'batch_size × input_len: %{x}<br>' +
                              'Ratio: %{y:.2f}%<extra></extra>'
            ))
            # MLP
            fig.add_trace(go.Scatter(
                x=data['combined'],
                y=data['mlp_ratio'] * 100,
                mode='lines+markers',
                name=f'MLP (batch_size={bs})',
                marker=dict(size=8, symbol='triangle-down'),
                line=dict(width=2.5, dash='dash'),
                hovertemplate=f'<b>MLP (batch_size={bs})</b><br>' +
                              'batch_size × input_len: %{x}<br>' +
                              'Ratio: %{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text='Component Ratio vs (batch_size × input_len) (attention_forward vs MLP) (Plotly)', 
                  font=dict(size=16, family='Arial Black')),
        xaxis=dict(title='batch_size × input_len'),
        yaxis=dict(title='Ratio (%)', range=[0, 100]),
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=1000,
        legend=dict(font=dict(size=9), orientation='v')
    )
    
    return fig


def save_interactive_html(fig, html_path: str):
    """将 plotly 图保存为可交互 HTML"""
    try:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        if isinstance(fig, go.Figure):
            fig.write_html(html_path)
            print(f"  已保存交互式图: {os.path.basename(html_path)}")
        else:
            print(f"  警告: 传入的不是 plotly 图表对象")
    except Exception as e:
        print(f"  保存交互式 HTML 时出错: {e}")


def save_versions(plot_func, df, component, output_dir, base_name):
    """对给定 plot_func，生成 linear 与 log 两个版本并保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    for use_log, suffix in [(False, 'linear'), (True, 'log')]:
        try:
            fig = plot_func(df, component, use_log=use_log)
            # 清理组件名称中的特殊字符，用于文件名
            safe_component = component.replace('/', '_').replace('\\', '_')
            path = os.path.join(output_dir, f"{base_name}_{safe_component}_{suffix}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  已保存: {os.path.basename(path)}")
            plt.close(fig)
        except Exception as e:
            print(f"  错误: {e}")
            plt.close('all')


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_component_analysis.py <csv_file> [output_dir]")
        print("示例: python plot_component_analysis.py all_layer_times_statistics.csv ./component_plots/")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(csv_file)[0] + '_plots'
    
    # 读取 CSV
    print(f"读取 CSV 文件: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"数据行数: {len(df)}")
    print(f"列: {list(df.columns)}")
    
    # 准备数据
    print("\n准备数据...")
    df_prepared = prepare_data(df)
    
    if df_prepared is None:
        sys.exit(1)
    
    # 获取所有 components
    components = sorted(df_prepared['Component'].unique())
    print(f"发现 {len(components)} 个 components:")
    for comp in components:
        print(f"  - {comp}")
    
    print(f"\nInput lengths: {sorted(df_prepared['input_len'].dropna().unique())}")
    print(f"Batch sizes: {sorted(df_prepared['batch_size'].dropna().unique())}")
    
    # 为每个 component 生成图表
    for component in components:
        print(f"\n生成 {component} 的图表...")
        comp_data = df_prepared[df_prepared['Component'] == component]
        
        if len(comp_data) == 0:
            print(f"  警告: 没有找到 {component} 的数据")
            continue
        
        comp_output_dir = os.path.join(output_dir, component.replace('/', '_').replace('\\', '_'))
        
        print(f"  生成图表1: Mean vs Batch Size (linear & log)...")
        save_versions(plot_mean_by_batch_size, comp_data, component,
                      comp_output_dir, 'mean_vs_batch_size')
        
        print(f"  生成图表2: Mean vs Input Length (linear & log)...")
        save_versions(plot_mean_by_input_len, comp_data, component,
                      comp_output_dir, 'mean_vs_input_len')
        
        print(f"  生成图表3: Mean vs (batch_size × input_len) (linear & log)...")
        save_versions(plot_mean_by_combined, comp_data, component,
                      comp_output_dir, 'mean_vs_combined')
        
        # 生成 Plotly 版本（交互式图表）
        print(f"  生成 Plotly 版本图表1: Mean vs Batch Size...")
        try:
            fig = plot_mean_by_batch_size_plotly(comp_data, component)
            safe_component = component.replace('/', '_').replace('\\', '_')
            html_path = os.path.join(comp_output_dir, f'mean_vs_batch_size_{safe_component}_plotly.html')
            os.makedirs(comp_output_dir, exist_ok=True)
            save_interactive_html(fig, html_path)
        except Exception as e:
            print(f"  错误: {e}")
        
        print(f"  生成 Plotly 版本图表2: Mean vs Input Length...")
        try:
            fig = plot_mean_by_input_len_plotly(comp_data, component)
            safe_component = component.replace('/', '_').replace('\\', '_')
            html_path = os.path.join(comp_output_dir, f'mean_vs_input_len_{safe_component}_plotly.html')
            os.makedirs(comp_output_dir, exist_ok=True)
            save_interactive_html(fig, html_path)
        except Exception as e:
            print(f"  错误: {e}")
        
        print(f"  生成 Plotly 版本图表3: Mean vs (batch_size × input_len)...")
        try:
            fig = plot_mean_by_combined_plotly(comp_data, component)
            safe_component = component.replace('/', '_').replace('\\', '_')
            html_path = os.path.join(comp_output_dir, f'mean_vs_combined_{safe_component}_plotly.html')
            os.makedirs(comp_output_dir, exist_ok=True)
            save_interactive_html(fig, html_path)
        except Exception as e:
            print(f"  错误: {e}")
    
    # 生成占比分析图表
    print(f"\n计算 attention_forward 占比...")
    ratio_df = calculate_component_ratios(df_prepared)
    
    if len(ratio_df) > 0:
        ratio_output_dir = os.path.join(output_dir, 'component_ratios')
        
        print(f"生成占比图表1: Ratio vs Batch Size (linear & log)...")
        for use_log, suffix in [(False, 'linear'), (True, 'log')]:
            try:
                fig = plot_ratio_by_batch_size(ratio_df, use_log=use_log)
                path = os.path.join(ratio_output_dir, f'ratio_vs_batch_size_{suffix}.png')
                os.makedirs(ratio_output_dir, exist_ok=True)
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f"  已保存: {os.path.basename(path)}")
                plt.close(fig)
            except Exception as e:
                print(f"  错误: {e}")
                plt.close('all')
        
        print(f"生成占比图表2: Ratio vs Input Length (linear & log)...")
        for use_log, suffix in [(False, 'linear'), (True, 'log')]:
            try:
                fig = plot_ratio_by_input_len(ratio_df, use_log=use_log)
                path = os.path.join(ratio_output_dir, f'ratio_vs_input_len_{suffix}.png')
                os.makedirs(ratio_output_dir, exist_ok=True)
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f"  已保存: {os.path.basename(path)}")
                plt.close(fig)
            except Exception as e:
                print(f"  错误: {e}")
                plt.close('all')
        
        print(f"生成占比图表3: Ratio vs (batch_size × input_len) (linear & log)...")
        for use_log, suffix in [(False, 'linear'), (True, 'log')]:
            try:
                fig = plot_ratio_by_combined(ratio_df, use_log=use_log)
                path = os.path.join(ratio_output_dir, f'ratio_vs_combined_{suffix}.png')
                os.makedirs(ratio_output_dir, exist_ok=True)
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f"  已保存: {os.path.basename(path)}")
                plt.close(fig)
            except Exception as e:
                print(f"  错误: {e}")
                plt.close('all')
        
        # 生成 Plotly 版本的占比图表（交互式图表）
        print(f"生成 Plotly 版本占比图表1: Ratio vs Batch Size...")
        try:
            fig = plot_ratio_by_batch_size_plotly(ratio_df)
            html_path = os.path.join(ratio_output_dir, f'ratio_vs_batch_size_plotly.html')
            os.makedirs(ratio_output_dir, exist_ok=True)
            save_interactive_html(fig, html_path)
        except Exception as e:
            print(f"  错误: {e}")
        
        print(f"生成 Plotly 版本占比图表2: Ratio vs Input Length...")
        try:
            fig = plot_ratio_by_input_len_plotly(ratio_df)
            html_path = os.path.join(ratio_output_dir, f'ratio_vs_input_len_plotly.html')
            os.makedirs(ratio_output_dir, exist_ok=True)
            save_interactive_html(fig, html_path)
        except Exception as e:
            print(f"  错误: {e}")
        
        print(f"生成 Plotly 版本占比图表3: Ratio vs (batch_size × input_len)...")
        try:
            fig = plot_ratio_by_combined_plotly(ratio_df)
            html_path = os.path.join(ratio_output_dir, f'ratio_vs_combined_plotly.html')
            os.makedirs(ratio_output_dir, exist_ok=True)
            save_interactive_html(fig, html_path)
        except Exception as e:
            print(f"  错误: {e}")
    
    print(f"\n完成！所有图表已保存到: {output_dir}")


if __name__ == '__main__':
    main()
