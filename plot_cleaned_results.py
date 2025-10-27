#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def clean_csv_data(csv_path):
    """Clean the malformed CSV data"""
    print(f"Cleaning CSV data from: {csv_path}")
    
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
            print(f"Found header at line {line_num + 1}")
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
                        'tpot': tpot
                    })
            except (ValueError, IndexError):
                print(f"Skipping malformed line {line_num + 1}: {line[:100]}...")
                continue
    
    if not cleaned_data:
        raise ValueError("No valid data found in CSV")
    
    df = pd.DataFrame(cleaned_data)
    print(f"Successfully parsed {len(df)} valid data points")
    print(f"Batch sizes found: {sorted(df['bs'].unique())}")
    print(f"Context range: {df['context'].min():.0f} - {df['context'].max():.0f}")
    print(f"TPOT range: {df['tpot'].min():.2f} - {df['tpot'].max():.2f} ms")
    
    return df

def plot_tpot_vs_context_by_bs(df, output_dir=None):
    """Create main plot with batch sizes as series"""
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    batch_sizes = sorted(df['bs'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(batch_sizes)))
    
    # Plot 1: All repeats with transparency
    for i, bs in enumerate(batch_sizes):
        bs_data = df[df['bs'] == bs].sort_values('context')
        
        if len(bs_data) > 0:
            # Plot individual repeats with transparency
            for repeat_idx in sorted(bs_data['repeat_idx'].unique()):
                repeat_data = bs_data[bs_data['repeat_idx'] == repeat_idx]
                if len(repeat_data) > 1:  # Only plot if we have multiple points
                    ax1.plot(repeat_data['context'], repeat_data['tpot'], 
                            color=colors[i], alpha=0.3, linewidth=1)
            
            # Plot mean line
            mean_data = bs_data.groupby('context')['tpot'].mean().reset_index()
            if len(mean_data) > 1:
                ax1.plot(mean_data['context'], mean_data['tpot'], 
                        color=colors[i], linewidth=2.5, alpha=0.9,
                        label=f'BS {bs}', marker='o', markersize=3)
    
    ax1.set_xlabel('Context Length', fontsize=12)
    ax1.set_ylabel('TPOT (ms)', fontsize=12)
    ax1.set_title('TPOT vs Context Length by Batch Size\n(Individual Repeats + Mean)', fontsize=14)
    ax1.set_xscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean with error bars
    for i, bs in enumerate(batch_sizes):
        bs_data = df[df['bs'] == bs]
        
        stats = bs_data.groupby('context')['tpot'].agg(['mean', 'std', 'count']).reset_index()
        
        if len(stats) > 1:
            # Only show error bars if we have multiple measurements
            yerr = stats['std'].fillna(0)
            yerr = yerr.where(stats['count'] > 1, 0)
            
            ax2.errorbar(stats['context'], stats['mean'], 
                        yerr=yerr, 
                        color=colors[i], linewidth=2, 
                        marker='o', markersize=4,
                        label=f'BS {bs}',
                        capsize=3, alpha=0.8)
    
    ax2.set_xlabel('Context Length', fontsize=12)
    ax2.set_ylabel('TPOT (ms)', fontsize=12)
    ax2.set_title('TPOT vs Context Length by Batch Size\n(Mean ± Standard Deviation)', fontsize=14)
    ax2.set_xscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tpot_vs_context_by_bs.png')
    else:
        output_path = 'tpot_vs_context_by_bs.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Main plot saved to: {output_path}")
    
    return fig

def plot_repeat_variability_detailed(df, output_dir=None):
    """Create detailed plots showing repeat variability"""
    
    batch_sizes = sorted(df['bs'].unique())
    n_plots = min(6, len(batch_sizes))  # Show up to 6 batch sizes
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx in range(n_plots):
        bs = batch_sizes[idx]
        bs_data = df[df['bs'] == bs].sort_values(['repeat_idx', 'context'])
        
        # Plot each repeat as a separate line
        colors = plt.cm.Set1(np.linspace(0, 1, len(bs_data['repeat_idx'].unique())))
        
        for repeat_idx, color in zip(sorted(bs_data['repeat_idx'].unique()), colors):
            repeat_data = bs_data[bs_data['repeat_idx'] == repeat_idx]
            if len(repeat_data) > 1:
                axes[idx].plot(repeat_data['context'], repeat_data['tpot'], 
                              marker='o', markersize=3, alpha=0.8, color=color,
                              label=f'Repeat {repeat_idx}', linewidth=2)
        
        axes[idx].set_xlabel('Context Length')
        axes[idx].set_ylabel('TPOT (ms)')
        axes[idx].set_title(f'Batch Size {bs} - Repeat Variability')
        axes[idx].set_xscale('log')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'repeat_variability_detailed.png')
    else:
        output_path = 'repeat_variability_detailed.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Repeat variability plot saved to: {output_path}")
    
    return fig

def plot_tpot_vs_bs_by_context(df, output_dir=None):
    """Create plot with context lengths as series, batch size as x-axis (log scale), tpot as y-axis"""
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get unique context lengths and sort them
    context_lengths = sorted(df['context'].unique())
    
    # Select a subset of context lengths for better visualization (every few values)
    # Show key context lengths: powers of 2 and some intermediate values
    key_contexts = []
    for ctx in context_lengths:
        if ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768] or ctx % 5120 == 0:
            key_contexts.append(ctx)
    
    # If we have too many, take a subset
    if len(key_contexts) > 10:
        step = len(key_contexts) // 10
        key_contexts = key_contexts[::step]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(key_contexts)))
    
    # Plot 1: All repeats with transparency
    for i, ctx in enumerate(key_contexts):
        ctx_data = df[df['context'] == ctx].sort_values('bs')
        
        if len(ctx_data) > 0:
            # Plot individual repeats with transparency
            for repeat_idx in sorted(ctx_data['repeat_idx'].unique()):
                repeat_data = ctx_data[ctx_data['repeat_idx'] == repeat_idx]
                if len(repeat_data) > 1:  # Only plot if we have multiple points
                    ax1.plot(repeat_data['bs'], repeat_data['tpot'], 
                            color=colors[i], alpha=0.3, linewidth=1)
            
            # Plot mean line
            mean_data = ctx_data.groupby('bs')['tpot'].mean().reset_index()
            if len(mean_data) > 1:
                ax1.plot(mean_data['bs'], mean_data['tpot'], 
                        color=colors[i], linewidth=2.5, alpha=0.9,
                        label=f'Context {int(ctx)}', marker='o', markersize=3)
    
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('TPOT (ms)', fontsize=12)
    ax1.set_title('TPOT vs Batch Size by Context Length\n(Individual Repeats + Mean)', fontsize=14)
    ax1.set_xscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean with error bars
    for i, ctx in enumerate(key_contexts):
        ctx_data = df[df['context'] == ctx]
        
        stats = ctx_data.groupby('bs')['tpot'].agg(['mean', 'std', 'count']).reset_index()
        
        if len(stats) > 1:
            # Only show error bars if we have multiple measurements
            yerr = stats['std'].fillna(0)
            yerr = yerr.where(stats['count'] > 1, 0)
            
            ax2.errorbar(stats['bs'], stats['mean'], 
                        yerr=yerr, 
                        color=colors[i], linewidth=2, 
                        marker='o', markersize=4,
                        label=f'Context {int(ctx)}',
                        capsize=3, alpha=0.8)
    
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('TPOT (ms)', fontsize=12)
    ax2.set_title('TPOT vs Batch Size by Context Length\n(Mean ± Standard Deviation)', fontsize=14)
    ax2.set_xscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tpot_vs_bs_by_context.png')
    else:
        output_path = 'tpot_vs_bs_by_context.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TPOT vs BS by Context plot saved to: {output_path}")
    
    return fig

def plot_summary_statistics(df, output_dir=None):
    """Generate summary statistics and plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: TPOT distribution by batch size
    batch_sizes = sorted(df['bs'].unique())
    tpot_by_bs = [df[df['bs'] == bs]['tpot'].values for bs in batch_sizes]
    
    ax1.boxplot(tpot_by_bs, labels=batch_sizes)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('TPOT (ms)')
    ax1.set_title('TPOT Distribution by Batch Size')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Mean TPOT vs Batch Size
    bs_stats = df.groupby('bs')['tpot'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(bs_stats['bs'], bs_stats['mean'], yerr=bs_stats['std'], 
                marker='o', capsize=5, linewidth=2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Mean TPOT (ms)')
    ax2.set_title('Mean TPOT vs Batch Size')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Context length effect
    context_stats = df.groupby('context')['tpot'].agg(['mean', 'std']).reset_index()
    ax3.errorbar(context_stats['context'], context_stats['mean'], yerr=context_stats['std'], 
                marker='o', capsize=3, alpha=0.7)
    ax3.set_xlabel('Context Length')
    ax3.set_ylabel('Mean TPOT (ms)')
    ax3.set_title('Mean TPOT vs Context Length (All Batch Sizes)')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coefficient of variation
    cv_data = []
    for bs in batch_sizes:
        bs_data = df[df['bs'] == bs]
        for context in bs_data['context'].unique():
            context_data = bs_data[bs_data['context'] == context]['tpot']
            if len(context_data) > 1:
                cv = context_data.std() / context_data.mean()
                cv_data.append({'bs': bs, 'context': context, 'cv': cv})
    
    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        cv_by_bs = cv_df.groupby('bs')['cv'].mean().reset_index()
        ax4.bar(range(len(cv_by_bs)), cv_by_bs['cv'])
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Coefficient of Variation')
        ax4.set_title('Average Variability by Batch Size')
        ax4.set_xticks(range(len(cv_by_bs)))
        ax4.set_xticklabels([f'{int(bs)}' for bs in cv_by_bs['bs']], rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'summary_statistics.png')
    else:
        output_path = 'summary_statistics.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary statistics plot saved to: {output_path}")
    
    # Print text summary
    print("\n=== Data Summary ===")
    print(f"Total measurements: {len(df)}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Context lengths: {sorted(df['context'].unique())}")
    print(f"Repeats per (bs, context): {df.groupby(['bs', 'context']).size().describe()}")
    
    print("\n=== TPOT Statistics by Batch Size ===")
    print(df.groupby('bs')['tpot'].agg(['count', 'mean', 'std', 'min', 'max']).round(2))
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot cleaned benchmark results')
    parser.add_argument('csv_path', help='Path to the benchmark CSV file')
    parser.add_argument('--output-dir', '-o', default='./cleaned_plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    try:
        # Clean and load data
        df = clean_csv_data(args.csv_path)
        
        # Generate plots
        print("\nGenerating plots...")
        
        # Main analysis plot
        plot_tpot_vs_context_by_bs(df, args.output_dir)
        
        # New plot: TPOT vs Batch Size by Context Length
        plot_tpot_vs_bs_by_context(df, args.output_dir)
        
        # Repeat variability analysis
        plot_repeat_variability_detailed(df, args.output_dir)
        
        # Summary statistics
        plot_summary_statistics(df, args.output_dir)
        
        # Save cleaned data
        cleaned_csv_path = os.path.join(args.output_dir, 'cleaned_data.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(cleaned_csv_path, index=False)
        print(f"Cleaned data saved to: {cleaned_csv_path}")
        
        print("\nAll plots and analysis completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())