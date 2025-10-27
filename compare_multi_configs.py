#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import glob
from pathlib import Path

def parse_filename_info(filename):
    """Parse model info from filename"""
    # Expected format: ModelName_tp{N}_seqlen{L}_*_decode_bench_*.csv
    parts = Path(filename).stem.split('_')
    
    model_name = "Unknown"
    tp_size = 1
    seq_len = 2048
    
    for i, part in enumerate(parts):
        if part.startswith('tp') and len(part) > 2:
            try:
                tp_size = int(part[2:])
            except:
                pass
        elif part.startswith('seqlen') and len(part) > 6:
            try:
                seq_len = int(part[6:])
            except:
                pass
        elif i == 0:  # First part is usually model name
            model_name = part
    
    return model_name, tp_size, seq_len

def clean_single_csv_data(csv_path):
    """Clean a single CSV file and return DataFrame with metadata"""
    print(f"Processing: {csv_path}")
    
    # Parse file info
    model_name, tp_size, seq_len = parse_filename_info(csv_path)
    
    # Read the raw file as text
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the data manually
    cleaned_data = []
    header_found = False
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Split by comma
        parts = line.split(',')
        
        if not header_found and 'context' in line and 'bs' in line:
            # This is the header
            header_found = True
            continue
        
        # Try to extract the first 4 numeric values that make sense
        if header_found and len(parts) >= 4:
            try:
                # Try to parse as: context, bs, repeat_idx, tpot
                context = float(parts[0])
                bs = int(float(parts[1]))  # Convert to int for batch size
                repeat_idx = int(float(parts[2])) if parts[2] != '' else 0
                tpot = float(parts[3])
                
                # Sanity checks and filter for context that are multiples of 512
                if context > 0 and bs > 0 and tpot > 0 and context % 512 == 0:
                    cleaned_data.append({
                        'context': context,
                        'bs': bs,
                        'repeat_idx': repeat_idx,
                        'tpot': tpot,
                        'model_name': model_name,
                        'tp_size': tp_size,
                        'seq_len': seq_len,
                        'config': f"{model_name}_TP{tp_size}",
                        'file_path': csv_path
                    })
            except (ValueError, IndexError):
                continue
    
    if not cleaned_data:
        print(f"  Warning: No valid data found in {csv_path}")
        return pd.DataFrame()
    
    df = pd.DataFrame(cleaned_data)
    print(f"  Parsed {len(df)} valid data points for {model_name} TP{tp_size}")
    
    return df

def load_multiple_csv_files(file_pattern):
    """Load and combine multiple CSV files"""
    
    # Find all matching files
    if isinstance(file_pattern, str):
        csv_files = glob.glob(file_pattern)
    else:
        csv_files = file_pattern
    
    if not csv_files:
        raise ValueError(f"No CSV files found matching pattern: {file_pattern}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Process each file
    all_dataframes = []
    
    for csv_path in csv_files:
        df = clean_single_csv_data(csv_path)
        if not df.empty:
            all_dataframes.append(df)
    
    if not all_dataframes:
        raise ValueError("No valid data found in any CSV files")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Total data points: {len(combined_df)}")
    print(f"  Configurations: {sorted(combined_df['config'].unique())}")
    print(f"  Batch sizes: {sorted(combined_df['bs'].unique())}")
    print(f"  Context range: {combined_df['context'].min():.0f} - {combined_df['context'].max():.0f}")
    print(f"  TPOT range: {combined_df['tpot'].min():.2f} - {combined_df['tpot'].max():.2f} ms")
    
    return combined_df

def plot_tpot_vs_context_multi_config(df, output_dir=None):
    """Plot TPOT vs Context Length for different configurations"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    configs = sorted(df['config'].unique())
    batch_sizes = sorted(df['bs'].unique())
    
    # Select key batch sizes to display (avoid overcrowding)
    key_batch_sizes = [bs for bs in [1, 4, 16, 32] if bs in batch_sizes]
    if not key_batch_sizes:
        key_batch_sizes = batch_sizes[:4]  # Take first 4 if our defaults aren't available
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(configs)))
    
    for plot_idx, bs in enumerate(key_batch_sizes):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        for config_idx, config in enumerate(configs):
            config_data = df[(df['config'] == config) & (df['bs'] == bs)].sort_values('context')
            
            if len(config_data) > 0:
                # Calculate mean and std for each context length
                stats = config_data.groupby('context')['tpot'].agg(['mean', 'std', 'count']).reset_index()
                
                if len(stats) > 1:
                    # Plot individual repeats with transparency
                    for repeat_idx in sorted(config_data['repeat_idx'].unique()):
                        repeat_data = config_data[config_data['repeat_idx'] == repeat_idx]
                        if len(repeat_data) > 1:
                            ax.plot(repeat_data['context'], repeat_data['tpot'], 
                                   color=colors[config_idx], alpha=0.2, linewidth=0.8)
                    
                    # Plot mean with error bars
                    yerr = stats['std'].fillna(0)
                    yerr = yerr.where(stats['count'] > 1, 0)
                    
                    ax.errorbar(stats['context'], stats['mean'], 
                               yerr=yerr,
                               color=colors[config_idx], linewidth=2.5, 
                               marker='o', markersize=4,
                               label=config, capsize=3, alpha=0.8)
        
        ax.set_xlabel('Context Length (tokens)', fontsize=12)
        ax.set_ylabel('TPOT (ms)', fontsize=12)
        ax.set_title(f'TPOT vs Context Length - Batch Size {bs}', fontsize=14)
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(key_batch_sizes), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Performance Comparison Across Different Configurations\n(TPOT vs Context Length)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tpot_vs_context_multi_config.png')
    else:
        output_path = 'tpot_vs_context_multi_config.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-config context plot saved to: {output_path}")
    
    return fig

def plot_all_contexts_combined(df, output_dir=None):
    """Plot all context lengths combined in one figure, showing all batch sizes for each configuration"""
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    configs = sorted(df['config'].unique())
    batch_sizes = sorted(df['bs'].unique())
    
    # Select representative batch sizes to avoid overcrowding
    key_batch_sizes = []
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        if bs in batch_sizes:
            key_batch_sizes.append(bs)
    if not key_batch_sizes:
        key_batch_sizes = batch_sizes[:8]  # Take first 8 if defaults not available
    
    # Create color map for configurations and line styles for batch sizes
    config_colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Cycle through styles
    
    for config_idx, config in enumerate(configs):
        for bs_idx, bs in enumerate(key_batch_sizes):
            # Get all data for this configuration and batch size
            config_bs_data = df[(df['config'] == config) & (df['bs'] == bs)]
            
            if len(config_bs_data) > 0:
                # Calculate mean TPOT for each context length
                stats = config_bs_data.groupby('context')['tpot'].agg(['mean', 'std', 'count']).reset_index()
                
                if len(stats) > 1:
                    # Plot individual measurements with high transparency
                    for repeat_idx in sorted(config_bs_data['repeat_idx'].unique()):
                        repeat_data = config_bs_data[config_bs_data['repeat_idx'] == repeat_idx].sort_values('context')
                        if len(repeat_data) > 1:
                            ax.plot(repeat_data['context'], repeat_data['tpot'], 
                                   color=config_colors[config_idx], alpha=0.1, linewidth=0.5,
                                   linestyle=line_styles[bs_idx % len(line_styles)])
                    
                    # Plot mean line
                    line_style = line_styles[bs_idx % len(line_styles)]
                    label = f"{config} BS{bs}"
                    
                    ax.plot(stats['context'], stats['mean'], 
                           color=config_colors[config_idx], linewidth=2.5, 
                           marker='o', markersize=4, alpha=0.8,
                           linestyle=line_style, label=label)
    
    ax.set_xlabel('Context Length (tokens)', fontsize=14)
    ax.set_ylabel('TPOT (ms)', fontsize=14)
    ax.set_title('TPOT vs Context Length - All Configurations and Batch Sizes Combined', fontsize=16)
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    
    # Create a more organized legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Group by configuration
    legend_dict = {}
    for handle, label in zip(handles, labels):
        config_name = label.split(' BS')[0]
        if config_name not in legend_dict:
            legend_dict[config_name] = []
        legend_dict[config_name].append((handle, label))
    
    # Create legend with better organization
    if len(handles) <= 20:  # If not too many lines, show all
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    else:  # If too many, create a simplified legend
        # Show only configuration colors
        config_handles = []
        config_labels = []
        for config_idx, config in enumerate(configs):
            config_handles.append(plt.Line2D([0], [0], color=config_colors[config_idx], linewidth=3))
            config_labels.append(config)
        
        legend1 = ax.legend(config_handles, config_labels, 
                           title='Configurations', loc='upper left', fontsize=10)
        
        # Add batch size line styles legend
        bs_handles = []
        bs_labels = []
        for bs_idx, bs in enumerate(key_batch_sizes[:len(line_styles)]):
            bs_handles.append(plt.Line2D([0], [0], color='gray', 
                                       linestyle=line_styles[bs_idx % len(line_styles)], linewidth=2))
            bs_labels.append(f'BS {bs}')
        
        legend2 = ax.legend(bs_handles, bs_labels, 
                           title='Batch Sizes', loc='upper right', fontsize=10)
        ax.add_artist(legend1)  # Add back the first legend
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tpot_vs_context_all_combined.png')
    else:
        output_path = 'tpot_vs_context_all_combined.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined context plot saved to: {output_path}")
    
    return fig

def plot_all_batch_sizes_combined(df, output_dir=None):
    """Plot all context lengths combined in one figure, showing all batch sizes for each configuration"""
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    configs = sorted(df['config'].unique())
    context_lengths = sorted(df['context'].unique())
    
    # Select representative context lengths to avoid overcrowding
    key_context_lengths = []
    for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        if ctx in context_lengths:
            key_context_lengths.append(ctx)
    if not key_context_lengths:
        key_context_lengths = context_lengths[:8]  # Take first 8 if no standard ones found
    
    # Create color map for configurations and line styles for context lengths
    config_colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Cycle through styles
    
    for config_idx, config in enumerate(configs):
        for ctx_idx, ctx in enumerate(key_context_lengths):
            # Get data for this configuration and context length
            config_ctx_data = df[(df['config'] == config) & (df['context'] == ctx)]
            
            if len(config_ctx_data) > 0:
                # Calculate mean TPOT for each batch size
                stats = config_ctx_data.groupby('bs')['tpot'].agg(['mean', 'std', 'count']).reset_index()
                
                if len(stats) > 1:  # Need at least 2 points to draw a line
                    line_style = line_styles[ctx_idx % len(line_styles)]
                    
                    # Plot mean line
                    ax.plot(stats['bs'], stats['mean'], 
                           color=config_colors[config_idx], 
                           linestyle=line_style,
                           linewidth=2.5, 
                           marker='o', markersize=4,
                           label=f'{config} (ctx={ctx})', 
                           alpha=0.8)
    
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('TPOT (ms)', fontsize=14)
    ax.set_title('TPOT vs Batch Size - All Configurations and Context Lengths Combined', fontsize=16)
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    
    # Create a more organized legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Group by configuration
    legend_dict = {}
    for handle, label in zip(handles, labels):
        config_name = label.split(' (ctx=')[0]
        if config_name not in legend_dict:
            legend_dict[config_name] = []
        legend_dict[config_name].append((handle, label))
    
    # Create legend with better organization
    if len(handles) <= 20:  # If not too many, show all
        ax.legend(handles, labels, fontsize=10, loc='best', ncol=2)
    else:  # If too many, create a more compact legend
        # Group by configuration and show only representative lines
        new_handles = []
        new_labels = []
        for config_name in sorted(legend_dict.keys()):
            items = legend_dict[config_name]
            if items:
                # Take the first item as representative
                new_handles.append(items[0][0])
                ctx_list = [item[1].split('(ctx=')[1].rstrip(')') for item in items[:3]]
                if len(items) > 3:
                    ctx_list.append('...')
                new_labels.append(f"{config_name} (ctx: {', '.join(ctx_list)})")
        
        ax.legend(new_handles, new_labels, fontsize=10, loc='best', ncol=1)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tpot_vs_bs_all_combined.png')
    else:
        output_path = 'tpot_vs_bs_all_combined.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined batch size plot saved to: {output_path}")
    
    return fig

def plot_tpot_vs_bs_multi_config(df, output_dir=None):
    """Plot TPOT vs Batch Size for different configurations"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    configs = sorted(df['config'].unique())
    context_lengths = sorted(df['context'].unique())
    
    # Select key context lengths to display
    key_contexts = []
    for ctx in context_lengths:
        if ctx in [512, 1024, 2048, 4096, 8192, 16384]:
            key_contexts.append(ctx)
    
    # Take up to 4 context lengths
    if len(key_contexts) > 4:
        key_contexts = key_contexts[:4]
    elif len(key_contexts) == 0:
        key_contexts = context_lengths[:4] if len(context_lengths) >= 4 else context_lengths
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(configs)))
    
    for plot_idx, ctx in enumerate(key_contexts):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        for config_idx, config in enumerate(configs):
            config_data = df[(df['config'] == config) & (df['context'] == ctx)].sort_values('bs')
            
            if len(config_data) > 0:
                # Calculate mean and std for each batch size
                stats = config_data.groupby('bs')['tpot'].agg(['mean', 'std', 'count']).reset_index()
                
                if len(stats) > 1:
                    # Plot individual repeats with transparency
                    for repeat_idx in sorted(config_data['repeat_idx'].unique()):
                        repeat_data = config_data[config_data['repeat_idx'] == repeat_idx]
                        if len(repeat_data) > 1:
                            ax.plot(repeat_data['bs'], repeat_data['tpot'], 
                                   color=colors[config_idx], alpha=0.2, linewidth=0.8)
                    
                    # Plot mean with error bars
                    yerr = stats['std'].fillna(0)
                    yerr = yerr.where(stats['count'] > 1, 0)
                    
                    ax.errorbar(stats['bs'], stats['mean'], 
                               yerr=yerr,
                               color=colors[config_idx], linewidth=2.5, 
                               marker='o', markersize=4,
                               label=config, capsize=3, alpha=0.8)
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('TPOT (ms)', fontsize=12)
        ax.set_title(f'TPOT vs Batch Size - Context Length {int(ctx)}', fontsize=14)
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(key_contexts), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Performance Comparison Across Different Configurations\n(TPOT vs Batch Size)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tpot_vs_bs_multi_config.png')
    else:
        output_path = 'tpot_vs_bs_multi_config.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-config batch size plot saved to: {output_path}")
    
    return fig

def plot_performance_heatmap(df, output_dir=None):
    """Create heatmaps showing performance across configurations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    configs = sorted(df['config'].unique())
    
    # Heatmap 1: Average TPOT by Batch Size and Config
    bs_config_data = df.groupby(['config', 'bs'])['tpot'].mean().reset_index()
    bs_pivot = bs_config_data.pivot(index='config', columns='bs', values='tpot')
    
    sns.heatmap(bs_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, 
                cbar_kws={'label': 'Average TPOT (ms)'})
    ax1.set_title('Average TPOT by Configuration and Batch Size', fontsize=14)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Configuration', fontsize=12)
    
    # Heatmap 2: Performance improvement relative to baseline
    if len(configs) > 1:
        baseline_config = configs[0]  # Use first config as baseline
        
        improvement_data = []
        for config in configs:
            config_stats = df[df['config'] == config].groupby('bs')['tpot'].mean()
            baseline_stats = df[df['config'] == baseline_config].groupby('bs')['tpot'].mean()
            
            for bs in config_stats.index:
                if bs in baseline_stats.index:
                    improvement = (baseline_stats[bs] - config_stats[bs]) / baseline_stats[bs] * 100
                    improvement_data.append({
                        'config': config,
                        'bs': bs,
                        'improvement_pct': improvement
                    })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            improvement_pivot = improvement_df.pivot(index='config', columns='bs', values='improvement_pct')
            
            sns.heatmap(improvement_pivot, annot=True, fmt='.1f', cmap='RdBu_r', 
                       center=0, ax=ax2, cbar_kws={'label': 'Improvement vs Baseline (%)'})
            ax2.set_title(f'Performance Improvement vs {baseline_config}', fontsize=14)
            ax2.set_xlabel('Batch Size', fontsize=12)
            ax2.set_ylabel('Configuration', fontsize=12)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'performance_heatmap.png')
    else:
        output_path = 'performance_heatmap.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance heatmap saved to: {output_path}")
    
    return fig

def generate_comparison_report(df, output_dir=None):
    """Generate a comprehensive comparison report"""
    
    report = []
    report.append("=== Multi-Configuration Performance Analysis Report ===")
    report.append("")
    
    configs = sorted(df['config'].unique())
    
    # Basic statistics
    report.append(f"Configurations analyzed: {len(configs)}")
    for config in configs:
        config_data = df[df['config'] == config]
        report.append(f"  - {config}: {len(config_data)} measurements")
    
    report.append(f"Batch sizes tested: {sorted(df['bs'].unique())}")
    report.append(f"Context length range: {df['context'].min():.0f} - {df['context'].max():.0f} tokens")
    report.append(f"Overall TPOT range: {df['tpot'].min():.2f} - {df['tpot'].max():.2f} ms")
    report.append("")
    
    # Performance by configuration
    report.append("=== Average Performance by Configuration ===")
    config_stats = df.groupby('config')['tpot'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    report.append(config_stats.to_string())
    report.append("")
    
    # Best configuration analysis
    report.append("=== Performance Analysis ===")
    best_config = config_stats['mean'].idxmin()
    best_tpot = config_stats.loc[best_config, 'mean']
    worst_config = config_stats['mean'].idxmax()
    worst_tpot = config_stats.loc[worst_config, 'mean']
    
    report.append(f"• Best overall performance: {best_config} with {best_tpot:.2f} ms average TPOT")
    report.append(f"• Worst overall performance: {worst_config} with {worst_tpot:.2f} ms average TPOT")
    report.append(f"• Performance gap: {((worst_tpot / best_tpot) - 1) * 100:.1f}% slower")
    
    # Scaling analysis by batch size
    report.append("")
    report.append("=== Batch Size Scaling Analysis ===")
    for config in configs:
        config_data = df[df['config'] == config]
        bs_stats = config_data.groupby('bs')['tpot'].mean()
        if len(bs_stats) > 1:
            min_bs, max_bs = bs_stats.index.min(), bs_stats.index.max()
            scaling_factor = bs_stats[max_bs] / bs_stats[min_bs]
            report.append(f"• {config}: {scaling_factor:.2f}x slower at BS {max_bs} vs BS {min_bs}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'multi_config_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nComparison report saved to: {report_path}")
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description='Compare performance across multiple model configurations')
    parser.add_argument('file_pattern', 
                       help='Pattern to match CSV files (e.g., "*_decode_bench_*.csv") or space-separated list of files')
    parser.add_argument('--output-dir', '-o', default='./multi_config_analysis', 
                       help='Output directory for plots')
    parser.add_argument('--files', nargs='+', 
                       help='Explicit list of CSV files to process')
    
    args = parser.parse_args()
    
    try:
        # Determine input files
        if args.files:
            csv_files = args.files
        else:
            csv_files = args.file_pattern
        
        # Load and combine data
        print("Loading and processing CSV files...")
        df = load_multiple_csv_files(csv_files)
        
        if df.empty:
            print("No valid data found. Exiting.")
            return 1
        
        # Generate all comparison plots
        print("\nGenerating comparison plots...")
        
        # TPOT vs Context Length comparison
        plot_tpot_vs_context_multi_config(df, args.output_dir)
        
        # All contexts combined in one plot
        plot_all_contexts_combined(df, args.output_dir)
        
        # TPOT vs Batch Size comparison
        plot_tpot_vs_bs_multi_config(df, args.output_dir)
        
        # All batch sizes combined in one plot
        plot_all_batch_sizes_combined(df, args.output_dir)
        
        # Performance heatmaps
        plot_performance_heatmap(df, args.output_dir)
        
        # Generate comparison report
        generate_comparison_report(df, args.output_dir)
        
        # Save combined cleaned data
        cleaned_csv_path = os.path.join(args.output_dir, 'combined_cleaned_data.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(cleaned_csv_path, index=False)
        print(f"Combined cleaned data saved to: {cleaned_csv_path}")
        
        print("\nAll multi-configuration analysis completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())