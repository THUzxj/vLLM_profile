#!/usr/bin/env python3
"""
Script to plot memory bandwidth vs batch size for flash_attn and flash_infer
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def extract_batch_size(filename):
    """Extract batch size from filename"""
    match = re.search(r'bs(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def read_bandwidth_value(filepath):
    """Read bandwidth value from CSV or TXT file"""
    try:
        df = pd.read_csv(filepath)
        if 'bandwidth' in df.columns and len(df) > 0:
            # Get the bandwidth value (usually the first data row)
            bandwidth = df['bandwidth'].iloc[0]
            # Check if it's a valid number (not NaN)
            if pd.notna(bandwidth):
                return float(bandwidth)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def collect_data(data_dir, file_extension):
    """Collect batch size and bandwidth data from files"""
    data = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist")
        return data
    
    for filepath in data_path.glob(f'*{file_extension}'):
        batch_size = extract_batch_size(filepath.name)
        if batch_size is not None:
            bandwidth = read_bandwidth_value(filepath)
            if bandwidth is not None:
                data.append({'batch_size': batch_size, 'bandwidth': bandwidth})
    
    # Sort by batch size
    data.sort(key=lambda x: x['batch_size'])
    return data

def plot_bandwidth_vs_batchsize():
    """Plot bandwidth vs batch size for both flash_attn and flash_infer"""
    base_dir = Path(__file__).parent.parent
    flash_attn_dir = base_dir / 'vLLM_profile' / 'bandwidth' / 'flash_attn'
    flash_infer_dir = base_dir / 'vLLM_profile' / 'bandwidth' / 'flash_infer'
    
    # Collect data
    flash_attn_data = collect_data(flash_attn_dir, '.csv')
    flash_infer_data = collect_data(flash_infer_dir, '.txt')
    
    if not flash_attn_data and not flash_infer_data:
        print("No data found!")
        return
    
    # Extract batch sizes and bandwidths
    flash_attn_batch_sizes = [d['batch_size'] for d in flash_attn_data]
    flash_attn_bandwidths = [d['bandwidth'] for d in flash_attn_data]
    
    flash_infer_batch_sizes = [d['batch_size'] for d in flash_infer_data]
    flash_infer_bandwidths = [d['bandwidth'] for d in flash_infer_data]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    if flash_attn_data:
        plt.plot(flash_attn_batch_sizes, flash_attn_bandwidths, 
                marker='o', label='flash_attn', linewidth=2, markersize=8)
    
    if flash_infer_data:
        plt.plot(flash_infer_batch_sizes, flash_infer_bandwidths, 
                marker='s', label='flash_infer', linewidth=2, markersize=8)
    
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Memory Bandwidth (GB/s)', fontsize=12)
    plt.title('Memory Bandwidth vs Batch Size (Llama2-7B, seq_len=1024)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent
    output_path = output_dir / 'bandwidth_vs_batchsize.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()

if __name__ == '__main__':
    plot_bandwidth_vs_batchsize()

