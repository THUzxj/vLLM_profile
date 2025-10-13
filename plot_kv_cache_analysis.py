#!/usr/bin/env python3
"""
KV Cache Analysis Plotter

This script creates scatter plots and trend lines for KV cache analysis data.
Supports both single file plotting and batch processing of multiple CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys
import glob
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def load_csv_data(csv_file: str) -> dict:
    """Load CSV data and convert types"""
    try:
        df = pd.read_csv(csv_file)
        
        # Convert relative_time_seconds to numeric
        df['relative_time_seconds'] = pd.to_numeric(df['relative_time_seconds'], errors='coerce')
        df['free_blocks'] = pd.to_numeric(df['free_blocks'], errors='coerce')
        df['used_blocks'] = pd.to_numeric(df['used_blocks'], errors='coerce')
        df['global_usage_percent'] = pd.to_numeric(df['global_usage_percent'], errors='coerce')
        df['memory_est_mb'] = pd.to_numeric(df['memory_est_mb'], errors='coerce')
        
        # Remove any rows with NaN values in key columns
        df = df.dropna(subset=['relative_time_seconds', 'free_blocks'])
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def create_scatter_plot_with_trend(df: pd.DataFrame, output_file: str = None):
    """Create scatter plot with trend line for free blocks over time"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('KV Cache Usage Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Free Blocks vs Time (main requested plot)
    ax1 = axes[0, 0]
    
    # Scatter plot
    scatter = ax1.scatter(df['relative_time_seconds'], df['free_blocks'], 
                         c=df.index, cmap='viridis', alpha=0.7, s=50)
    
    # Fit trend line (polynomial regression for smooth curve)
    if len(df) > 1:
        x = df['relative_time_seconds'].values.reshape(-1, 1)
        y = df['free_blocks'].values
        
        # Use polynomial features for better fit
        degree = min(3, len(df) - 1)  # Avoid overfitting
        poly_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_reg.fit(x, y)
        
        # Generate smooth line
        x_smooth = np.linspace(df['relative_time_seconds'].min(), 
                              df['relative_time_seconds'].max(), 300).reshape(-1, 1)
        y_smooth = poly_reg.predict(x_smooth)
        
        ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.8, label='Trend Line')
        
        # Calculate R² score
        r2_score = poly_reg.score(x, y)
        ax1.text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    ax1.set_xlabel('Relative Time (seconds)')
    ax1.set_ylabel('Free Blocks')
    ax1.set_title('Free Blocks vs Time')
    ax1.grid(True, alpha=0.3)
    # ax1.legend()  # Legend disabled
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Request Sequence')
    
    # Plot 2: Memory Usage vs Time
    ax2 = axes[0, 1]
    ax2.scatter(df['relative_time_seconds'], df['memory_est_mb'], 
               c='orange', alpha=0.7, s=50)
    
    if len(df) > 1:
        # Linear trend for memory
        x = df['relative_time_seconds'].values.reshape(-1, 1)
        y = df['memory_est_mb'].values
        
        reg = LinearRegression()
        reg.fit(x, y)
        
        x_line = np.linspace(df['relative_time_seconds'].min(), 
                            df['relative_time_seconds'].max(), 100).reshape(-1, 1)
        y_line = reg.predict(x_line)
        
        ax2.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8)
        
        r2_score = reg.score(x, y)
        ax2.text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    ax2.set_xlabel('Relative Time (seconds)')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage vs Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Global Usage Percentage vs Time
    ax3 = axes[1, 0]
    ax3.scatter(df['relative_time_seconds'], df['global_usage_percent'], 
               c='green', alpha=0.7, s=50)
    
    if len(df) > 1:
        x = df['relative_time_seconds'].values.reshape(-1, 1)
        y = df['global_usage_percent'].values
        
        reg = LinearRegression()
        reg.fit(x, y)
        
        x_line = np.linspace(df['relative_time_seconds'].min(), 
                            df['relative_time_seconds'].max(), 100).reshape(-1, 1)
        y_line = reg.predict(x_line)
        
        ax3.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8)
        
        r2_score = reg.score(x, y)
        ax3.text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    ax3.set_xlabel('Relative Time (seconds)')
    ax3.set_ylabel('Global Usage (%)')
    ax3.set_title('Global Usage Percentage vs Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Operation Distribution
    ax4 = axes[1, 1]
    operation_counts = df['operation'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(operation_counts)))
    
    wedges, texts, autotexts = ax4.pie(operation_counts.values, 
                                      labels=operation_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    ax4.set_title('Operation Distribution')
    
    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        # Generate default filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"kv_cache_analysis_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    return fig


def create_detailed_free_blocks_plot(df: pd.DataFrame, output_file: str = None):
    """Create detailed plot focusing on free blocks trend"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color points by request_id
    unique_requests = df['request_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_requests)))
    
    for i, req_id in enumerate(unique_requests):
        req_data = df[df['request_id'] == req_id]
        ax.scatter(req_data['relative_time_seconds'], req_data['free_blocks'],
                  c=[colors[i]], label=f'Request {req_id}', s=100, alpha=0.7)
    
    # Overall trend line
    if len(df) > 1:
        x = df['relative_time_seconds'].values.reshape(-1, 1)
        y = df['free_blocks'].values
        
        # Polynomial fit
        degree = min(3, len(df) - 1)
        poly_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_reg.fit(x, y)
        
        x_smooth = np.linspace(df['relative_time_seconds'].min(), 
                              df['relative_time_seconds'].max(), 300).reshape(-1, 1)
        y_smooth = poly_reg.predict(x_smooth)
        
        ax.plot(x_smooth, y_smooth, 'red', linewidth=3, alpha=0.8, 
                label=f'Polynomial Trend (degree={degree})', linestyle='--')
        
        # R² score
        r2_score = poly_reg.score(x, y)
        ax.text(0.02, 0.98, f'R² = {r2_score:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                verticalalignment='top', fontsize=12, fontweight='bold')
    
    # Annotations for first and last points
    if len(df) > 0:
        first_point = df.iloc[0]
        last_point = df.iloc[-1]
        
        ax.annotate(f'Start: {first_point["free_blocks"]} blocks', 
                   xy=(first_point['relative_time_seconds'], first_point['free_blocks']),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if len(df) > 1:
            ax.annotate(f'End: {last_point["free_blocks"]} blocks', 
                       xy=(last_point['relative_time_seconds'], last_point['free_blocks']),
                       xytext=(-10, -30), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Relative Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Free Blocks', fontsize=12, fontweight='bold')
    ax.set_title('KV Cache Free Blocks Over Time\n(Scatter Plot with Polynomial Trend Line)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend disabled
    
    # Add statistics box
    stats_text = f"""Statistics:
Total Points: {len(df)}
Time Range: {df['relative_time_seconds'].max():.3f}s
Free Blocks Range: {df['free_blocks'].min()} - {df['free_blocks'].max()}
Mean Free Blocks: {df['free_blocks'].mean():.1f}
Std Free Blocks: {df['free_blocks'].std():.1f}"""
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
            verticalalignment='bottom', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save detailed plot
    if output_file:
        detail_file = output_file.replace('.png', '_detailed.png')
    else:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        detail_file = f"kv_cache_free_blocks_detailed_{timestamp}.png"
    
    plt.savefig(detail_file, dpi=300, bbox_inches='tight')
    print(f"Detailed plot saved to: {detail_file}")
    
    plt.show()
    
    return fig


def print_data_summary(df: pd.DataFrame):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("KV CACHE DATA SUMMARY")
    print("="*60)
    
    print(f"Total log entries: {len(df)}")
    print(f"Time range: {df['relative_time_seconds'].min():.3f}s to {df['relative_time_seconds'].max():.3f}s")
    print(f"Duration: {df['relative_time_seconds'].max() - df['relative_time_seconds'].min():.3f}s")
    print()
    
    print("Free Blocks Statistics:")
    print(f"  Min: {df['free_blocks'].min()}")
    print(f"  Max: {df['free_blocks'].max()}")
    print(f"  Mean: {df['free_blocks'].mean():.1f}")
    print(f"  Std: {df['free_blocks'].std():.1f}")
    print()
    
    print("Operations:")
    for op, count in df['operation'].value_counts().items():
        print(f"  {op}: {count} times")
    print()
    
    print("Requests:")
    for req_id in sorted(df['request_id'].unique()):
        req_data = df[df['request_id'] == req_id]
        print(f"  Request {req_id}: {len(req_data)} operations")


def process_single_file(csv_file: str, output_dir: str = None, plot_type: str = "detailed") -> bool:
    """Process a single CSV file and create plot"""
    try:
        # Load data
        df = load_csv_data(csv_file)
        if df is None or len(df) == 0:
            print(f"✗ No valid data found in {csv_file}")
            return False
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(csv_file).parent / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        input_path = Path(csv_file)
        
        # Create different types of plots
        success = False
        if plot_type in ["detailed", "both"]:
            output_file = output_path / f"{input_path.stem}_detailed_plot.png"
            create_detailed_free_blocks_plot(df, str(output_file))
            print(f"✓ {input_path.name} -> {output_file.name} (detailed)")
            success = True
            
        if plot_type in ["scatter", "both"]:
            output_file = output_path / f"{input_path.stem}_scatter_plot.png"
            create_scatter_plot_with_trend(df, str(output_file))
            print(f"✓ {input_path.name} -> {output_file.name} (scatter)")
            success = True
        
        return success
        
    except Exception as e:
        print(f"✗ Error processing {csv_file}: {e}")
        return False


def batch_process(input_pattern: str, output_dir: str = None, plot_type: str = "detailed") -> dict:
    """Batch process multiple CSV files"""
    
    # Find CSV files
    if os.path.isdir(input_pattern):
        # If directory, find all CSV files
        csv_files = glob.glob(os.path.join(input_pattern, "*_kv_cache.csv"))
    else:
        # Use as glob pattern
        csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching: {input_pattern}")
        return {'processed': 0, 'successful': 0, 'failed': 0}
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Use first file's directory
        first_file_dir = Path(csv_files[0]).parent
        output_path = first_file_dir / "plots"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(csv_files)} CSV files...")
    print(f"Output directory: {output_path}")
    print()
    
    # Process files
    successful = 0
    failed = 0
    
    for csv_file in sorted(csv_files):
        success = process_single_file(
            csv_file, 
            str(output_path), 
            plot_type=plot_type
        )
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    results = {
        'processed': len(csv_files),
        'successful': successful, 
        'failed': failed,
        'output_dir': str(output_path)
    }
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {results['processed']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {successful/len(csv_files)*100:.1f}%")
    print(f"Plots saved in: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Create scatter plots for KV cache analysis data"
    )
    parser.add_argument("input", 
                       help="Input CSV file or directory pattern")
    parser.add_argument("--output-dir", "-o", 
                       help="Output directory for plots (default: auto-generate)")
    parser.add_argument("--plot-type", "-p", 
                       choices=["detailed", "scatter", "both"], 
                       default="detailed",
                       help="Type of plot to generate (default: detailed)")
    parser.add_argument("--batch", "-b", action="store_true",
                       help="Batch process multiple files")
    
    args = parser.parse_args()
    
    # Check if batch processing
    if args.batch or os.path.isdir(args.input) or '*' in args.input:
        # Batch processing
        results = batch_process(
            args.input,
            args.output_dir, 
            plot_type=args.plot_type
        )
        return 0 if results['failed'] == 0 else 1
    else:
        # Single file processing
        if not Path(args.input).exists():
            print(f"Error: CSV file not found: {args.input}")
            return 1

        print("Processing single file...")
        success = process_single_file(
            args.input,
            args.output_dir,
            plot_type=args.plot_type
        )
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())