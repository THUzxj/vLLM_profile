#!/usr/bin/env python3
"""
根据 flash_attn_kernel_metrics.csv 生成20个曲线图：
Duration相关：
1. duration vs batch_size
2. duration vs kv_len
3. duration vs batch_size*kv_len
4. duration vs batch_size*kv_len (log scale)
Memory Throughput相关：
5. memory_throughput vs batch_size
6. memory_throughput vs kv_len
7. memory_throughput vs batch_size*kv_len
8. memory_throughput vs batch_size*kv_len (log scale)
Device Memory相关：
9. device_memory_mb vs batch_size
10. device_memory_mb vs kv_len
11. device_memory_mb vs batch_size*kv_len
12. device_memory_mb vs batch_size*kv_len (log scale)
Compute Throughput相关：
13. compute_throughput vs batch_size
14. compute_throughput vs kv_len
15. compute_throughput vs batch_size*kv_len
16. compute_throughput vs batch_size*kv_len (log scale)
Device Memory Bandwidth相关：
17. device_memory_bandwidth_gbs vs batch_size
18. device_memory_bandwidth_gbs vs kv_len
19. device_memory_bandwidth_gbs vs batch_size*kv_len
20. device_memory_bandwidth_gbs vs batch_size*kv_len (log scale)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_duration_vs_batch_size(df, output_dir):
    """绘制 duration vs batch_size 的曲线图，按 kv_len 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 kv_len 分组绘制
    kv_lens = sorted(df['kv_len'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(kv_lens)))
    
    for kv_len, color in zip(kv_lens, colors):
        df_subset = df[df['kv_len'] == kv_len].sort_values('batch_size')
        ax.plot(df_subset['batch_size'], df_subset['duration'], 
                marker='o', label=f'kv_len={kv_len}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('Duration vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    # ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'duration_vs_batch_size.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_duration_vs_kv_len(df, output_dir):
    """绘制 duration vs kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('kv_len')
        ax.plot(df_subset['kv_len'], df_subset['duration'], 
                marker='s', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('KV Length', fontsize=12)
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('Duration vs KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    # ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'duration_vs_kv_len.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_duration_vs_batch_kv_product(df, output_dir):
    """绘制 duration vs batch_size*kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len
    df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['duration'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('Duration vs Batch Size × KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    # ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'duration_vs_batch_kv_product.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_memory_throughput_vs_batch_size(df, output_dir):
    """绘制 memory_throughput vs batch_size 的曲线图，按 kv_len 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 kv_len 分组绘制
    kv_lens = sorted(df['kv_len'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(kv_lens)))
    
    for kv_len, color in zip(kv_lens, colors):
        df_subset = df[df['kv_len'] == kv_len].sort_values('batch_size')
        ax.plot(df_subset['batch_size'], df_subset['memory_throughput'], 
                marker='o', label=f'kv_len={kv_len}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Memory Throughput (GB/s)', fontsize=12)
    ax.set_title('Memory Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'memory_throughput_vs_batch_size.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_memory_throughput_vs_kv_len(df, output_dir):
    """绘制 memory_throughput vs kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('kv_len')
        ax.plot(df_subset['kv_len'], df_subset['memory_throughput'], 
                marker='s', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('KV Length', fontsize=12)
    ax.set_ylabel('Memory Throughput (GB/s)', fontsize=12)
    ax.set_title('Memory Throughput vs KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'memory_throughput_vs_kv_len.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_memory_throughput_vs_batch_kv_product(df, output_dir):
    """绘制 memory_throughput vs batch_size*kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['memory_throughput'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Memory Throughput (GB/s)', fontsize=12)
    ax.set_title('Memory Throughput vs Batch Size × KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'memory_throughput_vs_batch_kv_product.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_vs_batch_size(df, output_dir):
    """绘制 device_memory_mb vs batch_size 的曲线图，按 kv_len 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 kv_len 分组绘制
    kv_lens = sorted(df['kv_len'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(kv_lens)))
    
    for kv_len, color in zip(kv_lens, colors):
        df_subset = df[df['kv_len'] == kv_len].sort_values('batch_size')
        ax.plot(df_subset['batch_size'], df_subset['device_memory_mb'], 
                marker='o', label=f'kv_len={kv_len}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Device Memory (MB)', fontsize=12)
    ax.set_title('Device Memory vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_vs_batch_size.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_vs_kv_len(df, output_dir):
    """绘制 device_memory_mb vs kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('kv_len')
        ax.plot(df_subset['kv_len'], df_subset['device_memory_mb'], 
                marker='s', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('KV Length', fontsize=12)
    ax.set_ylabel('Device Memory (MB)', fontsize=12)
    ax.set_title('Device Memory vs KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_vs_kv_len.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_vs_batch_kv_product(df, output_dir):
    """绘制 device_memory_mb vs batch_size*kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['device_memory_mb'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Device Memory (MB)', fontsize=12)
    ax.set_title('Device Memory vs Batch Size × KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_vs_batch_kv_product.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_compute_throughput_vs_batch_size(df, output_dir):
    """绘制 compute_throughput vs batch_size 的曲线图，按 kv_len 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 kv_len 分组绘制
    kv_lens = sorted(df['kv_len'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(kv_lens)))
    
    for kv_len, color in zip(kv_lens, colors):
        df_subset = df[df['kv_len'] == kv_len].sort_values('batch_size')
        ax.plot(df_subset['batch_size'], df_subset['compute_throughput'], 
                marker='o', label=f'kv_len={kv_len}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Compute Throughput (TFLOP/s)', fontsize=12)
    ax.set_title('Compute Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'compute_throughput_vs_batch_size.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_compute_throughput_vs_kv_len(df, output_dir):
    """绘制 compute_throughput vs kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('kv_len')
        ax.plot(df_subset['kv_len'], df_subset['compute_throughput'], 
                marker='s', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('KV Length', fontsize=12)
    ax.set_ylabel('Compute Throughput (TFLOP/s)', fontsize=12)
    ax.set_title('Compute Throughput vs KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'compute_throughput_vs_kv_len.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_compute_throughput_vs_batch_kv_product(df, output_dir):
    """绘制 compute_throughput vs batch_size*kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['compute_throughput'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Compute Throughput (TFLOP/s)', fontsize=12)
    ax.set_title('Compute Throughput vs Batch Size × KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'compute_throughput_vs_batch_kv_product.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_bandwidth_vs_batch_size(df, output_dir):
    """绘制 device_memory_bandwidth_gbs vs batch_size 的曲线图，按 kv_len 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 kv_len 分组绘制
    kv_lens = sorted(df['kv_len'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(kv_lens)))
    
    for kv_len, color in zip(kv_lens, colors):
        df_subset = df[df['kv_len'] == kv_len].sort_values('batch_size')
        ax.plot(df_subset['batch_size'], df_subset['device_memory_bandwidth_gbs'], 
                marker='o', label=f'kv_len={kv_len}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Device Memory Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Device Memory Bandwidth vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_bandwidth_vs_batch_size.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_bandwidth_vs_kv_len(df, output_dir):
    """绘制 device_memory_bandwidth_gbs vs kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('kv_len')
        ax.plot(df_subset['kv_len'], df_subset['device_memory_bandwidth_gbs'], 
                marker='s', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('KV Length', fontsize=12)
    ax.set_ylabel('Device Memory Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Device Memory Bandwidth vs KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_bandwidth_vs_kv_len.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_bandwidth_vs_batch_kv_product(df, output_dir):
    """绘制 device_memory_bandwidth_gbs vs batch_size*kv_len 的曲线图，按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['device_memory_bandwidth_gbs'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Device Memory Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Device Memory Bandwidth vs Batch Size × KV Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_bandwidth_vs_batch_kv_product.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_duration_vs_batch_kv_product_log(df, output_dir):
    """绘制 duration vs batch_size*kv_len 的曲线图（对数横坐标），按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['duration'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Duration (ms)', fontsize=12)
    ax.set_title('Duration vs Batch Size × KV Length (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'duration_vs_batch_kv_product_log.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_memory_throughput_vs_batch_kv_product_log(df, output_dir):
    """绘制 memory_throughput vs batch_size*kv_len 的曲线图（对数横坐标），按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['memory_throughput'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Memory Throughput (GB/s)', fontsize=12)
    ax.set_title('Memory Throughput vs Batch Size × KV Length (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'memory_throughput_vs_batch_kv_product_log.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_vs_batch_kv_product_log(df, output_dir):
    """绘制 device_memory_mb vs batch_size*kv_len 的曲线图（对数横坐标），按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['device_memory_mb'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Device Memory (MB)', fontsize=12)
    ax.set_title('Device Memory vs Batch Size × KV Length (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_vs_batch_kv_product_log.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_compute_throughput_vs_batch_kv_product_log(df, output_dir):
    """绘制 compute_throughput vs batch_size*kv_len 的曲线图（对数横坐标），按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['compute_throughput'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Compute Throughput (TFLOP/s)', fontsize=12)
    ax.set_title('Compute Throughput vs Batch Size × KV Length (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'compute_throughput_vs_batch_kv_product_log.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_device_memory_bandwidth_vs_batch_kv_product_log(df, output_dir):
    """绘制 device_memory_bandwidth_gbs vs batch_size*kv_len 的曲线图（对数横坐标），按 batch_size 分组"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算 batch_size * kv_len（如果还没有计算）
    if 'batch_kv_product' not in df.columns:
        df['batch_kv_product'] = df['batch_size'] * df['kv_len']
    
    # 按 batch_size 分组绘制
    batch_sizes = sorted(df['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for batch_size, color in zip(batch_sizes, colors):
        df_subset = df[df['batch_size'] == batch_size].sort_values('batch_kv_product')
        ax.plot(df_subset['batch_kv_product'], df_subset['device_memory_bandwidth_gbs'], 
                marker='o', label=f'batch_size={batch_size}', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Batch Size × KV Length', fontsize=12)
    ax.set_ylabel('Device Memory Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Device Memory Bandwidth vs Batch Size × KV Length (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'device_memory_bandwidth_vs_batch_kv_product_log.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    # 获取CSV文件路径
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # 默认使用当前目录下的文件
        csv_file = 'ncu_profile_result_v4_4B/flash_attn_kernel_metrics.csv'
        if not os.path.exists(csv_file):
            csv_file = 'flash_attn_kernel_metrics.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    # 读取CSV文件
    print(f"Reading: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 确定输出目录（保存到CSV文件所在目录）
    output_dir = os.path.dirname(os.path.abspath(csv_file)) if os.path.dirname(csv_file) else '.'
    
    # 生成15个图表
    print("\nGenerating duration plots...")
    plot_duration_vs_batch_size(df, output_dir)
    plot_duration_vs_kv_len(df, output_dir)
    plot_duration_vs_batch_kv_product(df, output_dir)
    
    print("\nGenerating memory throughput plots...")
    plot_memory_throughput_vs_batch_size(df, output_dir)
    plot_memory_throughput_vs_kv_len(df, output_dir)
    plot_memory_throughput_vs_batch_kv_product(df, output_dir)
    
    print("\nGenerating device memory plots...")
    plot_device_memory_vs_batch_size(df, output_dir)
    plot_device_memory_vs_kv_len(df, output_dir)
    plot_device_memory_vs_batch_kv_product(df, output_dir)
    
    print("\nGenerating compute throughput plots...")
    plot_compute_throughput_vs_batch_size(df, output_dir)
    plot_compute_throughput_vs_kv_len(df, output_dir)
    plot_compute_throughput_vs_batch_kv_product(df, output_dir)
    
    print("\nGenerating device memory bandwidth plots...")
    plot_device_memory_bandwidth_vs_batch_size(df, output_dir)
    plot_device_memory_bandwidth_vs_kv_len(df, output_dir)
    plot_device_memory_bandwidth_vs_batch_kv_product(df, output_dir)
    
    print("\nGenerating log-scale batch_kv_product plots...")
    plot_duration_vs_batch_kv_product_log(df, output_dir)
    plot_memory_throughput_vs_batch_kv_product_log(df, output_dir)
    plot_device_memory_vs_batch_kv_product_log(df, output_dir)
    plot_compute_throughput_vs_batch_kv_product_log(df, output_dir)
    plot_device_memory_bandwidth_vs_batch_kv_product_log(df, output_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()

