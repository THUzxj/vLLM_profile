#!/usr/bin/env python3
"""
Script to generate bar charts comparing component times across different parallel strategies.
Each chart represents a different batch size, showing multiple components at different parallelization levels.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate bar charts comparing component times across different parallel strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 plot_component_times_comparison.py \\
    --dp1-path /path/to/dp1_results \\
    --dp2-path /path/to/dp2_results \\
    --dp4-path /path/to/dp4_results \\
    --output-dir /path/to/output

  python3 plot_component_times_comparison.py \\
    --results-dir /path/to/results_folder \\
    --output-dir /path/to/output \\
    --dp1 "component_times_output_dp1_..." \\
    --dp2 "component_times_output_dp2_..." \\
    --dp4 "component_times_output_dp4_..."
        """
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/data/xjzhang/vLLM_profile_v1/sglang_profile/results"),
        help="Base results directory (default: /data/xjzhang/vLLM_profile_v1/sglang_profile/results)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated plots (required)"
    )
    
    parser.add_argument(
        "--dp1-path",
        type=Path,
        help="Full path to DP1 results directory (overrides --dp1 when combined with --results-dir)"
    )
    
    parser.add_argument(
        "--dp2-path",
        type=Path,
        help="Full path to DP2 results directory (overrides --dp2 when combined with --results-dir)"
    )
    
    parser.add_argument(
        "--dp4-path",
        type=Path,
        help="Full path to DP4 results directory (overrides --dp4 when combined with --results-dir)"
    )
    
    parser.add_argument(
        "--dp1",
        type=str,
        default="component_times_output_Qwen3-30B-A3B-1layer_il40000_dp1_ep1_tp1_random_20260210_042123",
        help="DP1 folder name relative to results-dir"
    )
    
    parser.add_argument(
        "--dp2",
        type=str,
        default="component_times_output_Qwen3-30B-A3B-1layer_il40000_dp2_ep2_tp2_random_20260210_042627",
        help="DP2 folder name relative to results-dir"
    )
    
    parser.add_argument(
        "--dp4",
        type=str,
        default="component_times_output_Qwen3-30B-A3B-1layer_il40000_dp4_ep4_tp4_random_20260210_050340",
        help="DP4 folder name relative to results-dir"
    )
    
    return parser.parse_args()


def setup_configuration(args):
    """Setup RESULTS_DIR, OUTPUT_DIR, and CONFIGS based on command line arguments."""
    results_dir = args.results_dir
    output_dir = args.output_dir
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Build CONFIGS dictionary
    configs = {}
    
    if args.dp1_path:
        configs["DP1"] = args.dp1_path
    else:
        configs["DP1"] = results_dir / args.dp1
    
    if args.dp2_path:
        configs["DP2"] = args.dp2_path
    else:
        configs["DP2"] = results_dir / args.dp2
    
    if args.dp4_path:
        configs["DP4"] = args.dp4_path
    else:
        configs["DP4"] = results_dir / args.dp4
    
    return results_dir, output_dir, configs

# Define which components to plot
COMPONENTS = [
    "layer_0_self_attention_statistics",
    "layer_0_mlp_statistics",
    "layer_0_attention_core_statistics",
    "layer_0_attention_prepare_statistics",
    "layer_0_mlp_gate_statistics",
    "layer_0_mlp_experts_statistics",
    "model_time_statistics",
]


def load_component_data(config_name, component_name, configs, results_dir):
    """Load component statistics from CSV file."""
    csv_path = results_dir / configs[config_name] / \
        "cuda" / "analysis" / f"{component_name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def get_all_batch_sizes(configs, results_dir):
    """Get all unique batch sizes across all configurations and components."""
    batch_sizes = set()
    for config_name in configs.keys():
        for component_name in COMPONENTS:
            df = load_component_data(config_name, component_name, configs, results_dir)
            if df is not None:
                batch_sizes.update(df['batch_size'].unique())
    return sorted(list(batch_sizes))


def create_batch_size_comparison(batch_size, configs, results_dir, components_to_show=None):
    """
    Create a bar chart comparing component times at a specific batch size
    across different parallel strategies.
    """
    if components_to_show is None:
        components_to_show = COMPONENTS

    # Prepare data structure
    data_dict = {config: {} for config in configs.keys()}

    # Load data for this batch size
    for config_name in configs.keys():
        for component_name in components_to_show:
            df = load_component_data(config_name, component_name, configs, results_dir)
            if df is not None:
                # Filter for the specific batch size
                filtered = df[df['batch_size'] == batch_size]
                if not filtered.empty:
                    mean_time = filtered['mean'].values[0]
                    # Extract component short name
                    short_name = component_name.replace(
                        "layer_0_", "").replace("_statistics", "")
                    data_dict[config_name][short_name] = mean_time * \
                        1000  # Convert to ms

    # Prepare data for plotting
    components_found = set()
    for config_data in data_dict.values():
        components_found.update(config_data.keys())

    if not components_found:
        print(f"No data found for batch size {batch_size}")
        return None

    components_found = sorted(list(components_found))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(components_found))
    width = 0.25  # Width of bars

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Plot bars for each configuration
    for idx, (config_name, color) in enumerate(zip(sorted(configs.keys()), colors)):
        values = [data_dict[config_name].get(
            comp, 0) for comp in components_found]
        ax.bar(x + idx * width, values, width,
               label=config_name, color=color, alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Component Times Comparison at Batch Size {batch_size}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(components_found, rotation=45, ha='right')
    ax.legend(title='Parallel Strategy', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, components_found


def create_all_comparison_charts(configs, results_dir, output_dir):
    """Create comparison charts for all batch sizes."""
    batch_sizes = get_all_batch_sizes(configs, results_dir)

    print(f"Found {len(batch_sizes)} unique batch sizes: {batch_sizes}")

    # Create charts for specific batch sizes (to avoid too many charts)
    # Show charts for every 5 batch sizes or min/max
    batch_sizes_to_plot = set()

    # Always include first and last
    if batch_sizes:
        batch_sizes_to_plot.add(batch_sizes[0])
        batch_sizes_to_plot.add(batch_sizes[-1])

        # Add every nth batch size for readability
        step = max(1, len(batch_sizes) // 5)  # Target ~5 charts
        for i in range(0, len(batch_sizes), step):
            batch_sizes_to_plot.add(batch_sizes[i])

    batch_sizes_to_plot = sorted(list(batch_sizes_to_plot))

    print(f"Creating charts for batch sizes: {batch_sizes_to_plot}")

    for batch_size in batch_sizes_to_plot:
        print(f"Creating chart for batch size {batch_size}...")
        result = create_batch_size_comparison(batch_size, configs, results_dir)

        if result is not None:
            fig, components_found = result
            output_path = output_dir / \
                f"batch_size_{batch_size:05d}_comparison.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close(fig)


def create_component_comparison_chart(configs, results_dir, output_dir):
    """
    Create a single comprehensive chart showing all components at different batch sizes.
    This shows mean times for each component across different parallel strategies.
    """
    batch_sizes = get_all_batch_sizes(configs, results_dir)

    # Create subplots for each configuration
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Mean Component Times vs Batch Size (Across Parallel Strategies)',
                 fontsize=16, fontweight='bold')

    config_names = sorted(configs.keys())

    for ax_idx, config_name in enumerate(config_names):
        ax = axes[ax_idx]

        # Prepare data for this config
        for component_name in COMPONENTS[:5]:  # Limit to first 5 for clarity
            times = []
            valid_batch_sizes = []

            for batch_size in batch_sizes:
                df = load_component_data(config_name, component_name, configs, results_dir)
                if df is not None:
                    filtered = df[df['batch_size'] == batch_size]
                    if not filtered.empty:
                        # Convert to ms
                        mean_time = filtered['mean'].values[0] * 1000
                        times.append(mean_time)
                        valid_batch_sizes.append(batch_size)

            if times:
                short_name = component_name.replace(
                    "layer_0_", "").replace("_statistics", "")
                ax.plot(valid_batch_sizes, times, marker='o',
                        label=short_name, linewidth=2)

        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.set_title(f'{config_name} Configuration',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()
    output_path = output_dir / "component_times_vs_batch_size.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive comparison: {output_path}")
    plt.close(fig)


def print_summary_statistics(configs, results_dir):
    """Print summary statistics for all components and configurations."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for config_name in sorted(configs.keys()):
        print(f"\n{config_name} Configuration:")
        print("-" * 60)

        for component_name in COMPONENTS:
            df = load_component_data(config_name, component_name, configs, results_dir)
            if df is not None:
                print(f"\n  {component_name}:")
                print(f"    Min time: {df['min'].min()*1000:.4f} ms")
                print(f"    Max time: {df['max'].max()*1000:.4f} ms")
                print(f"    Mean time: {df['mean'].mean()*1000:.4f} ms")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup configuration based on arguments
    results_dir, output_dir, configs = setup_configuration(args)
    
    print("Starting component times comparison analysis...")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Configurations:")
    for name, path in configs.items():
        print(f"  {name}: {path}")
    
    # Create individual batch size comparison charts
    print("\n1. Creating batch size comparison charts...")
    create_all_comparison_charts(configs, results_dir, output_dir)
    
    # Create comprehensive comparison chart
    print("\n2. Creating comprehensive component vs batch size chart...")
    create_component_comparison_chart(configs, results_dir, output_dir)
    
    # Print summary statistics
    print("\n3. Summary statistics:")
    print_summary_statistics(configs, results_dir)
    
    print(f"\n✓ Analysis complete! Charts saved to: {output_dir}")
