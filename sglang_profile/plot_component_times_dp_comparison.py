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
  python3 plot_component_times_dp_comparison.py \\
    --dp1-path /path/to/dp1_results \\
    --dp2-path /path/to/dp2_results \\
    --dp4-path /path/to/dp4_results \\
    --output-dir /path/to/output

  python3 plot_component_times_dp_comparison.py \\
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
    "layer_0_moe_core_statistics",
    "layer_0_moe_combine_statistics",
    "layer_0_moe_dispatch_statistics",
]

# Component groups for bar charts
# 1) moe_gate, moe_combine, moe_dispatch, moe_core in one bar chart
#    (there is no explicit "moe_gate" component in the CSV list, so we only
#     include the existing MoE-related components here)
MOE_COMPONENTS = [
    "layer_0_moe_core_statistics",
    "layer_0_moe_combine_statistics",
    "layer_0_moe_dispatch_statistics",
]

# 2) self_attention and mlp in one bar chart
SELF_MLP_COMPONENTS = [
    "layer_0_self_attention_statistics",
    "layer_0_mlp_statistics",
]

# 3) attention_core, attention_prepare, mlp_experts, mlp_gate in one bar chart
ATTN_MLP_DETAIL_COMPONENTS = [
    "layer_0_attention_core_statistics",
    "layer_0_attention_prepare_statistics",
    "layer_0_mlp_experts_statistics",
    "layer_0_mlp_gate_statistics",
]


def load_component_data(config_name, component_name, configs, results_dir):
    """Load component statistics from CSV file."""
    csv_path = configs[config_name] / \
        "cuda" / "analysis" / f"{component_name}.csv"
    print(f"Loading data for {config_name} - {component_name} from {csv_path}")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def get_dp_number(config_name):
    """Extract DP number from config name (e.g., 'DP1' -> 1, 'DP2' -> 2)."""
    import re
    match = re.search(r'DP(\d+)', config_name)
    if match:
        return int(match.group(1))
    return 1  # Default to 1 if not found


def get_all_batch_sizes(configs, results_dir):
    """Get all unique batch sizes across all configurations and components."""
    batch_sizes = set()
    for config_name in configs.keys():
        for component_name in COMPONENTS:
            df = load_component_data(
                config_name, component_name, configs, results_dir)
            if df is not None:
                batch_sizes.update(df['batch_size'].unique())
    return sorted(list(batch_sizes))


def get_all_total_batch_sizes(configs, results_dir):
    """Get all unique total batch sizes (DP * batch_size) across all configurations."""
    total_batch_sizes = set()
    for config_name in configs.keys():
        dp_number = get_dp_number(config_name)
        for component_name in COMPONENTS:
            df = load_component_data(
                config_name, component_name, configs, results_dir)
            if df is not None:
                # Calculate total_batch_size = DP * batch_size
                df_copy = df.copy()
                df_copy['total_batch_size'] = dp_number * df_copy['batch_size']
                total_batch_sizes.update(df_copy['total_batch_size'].unique())
    return sorted(list(total_batch_sizes))


def create_batch_size_comparison(batch_size, configs, results_dir, components_to_show=None, stat_type='min'):
    """
    Create a bar chart comparing component times at a specific batch size
    across different parallel strategies.

    Args:
        batch_size: The batch size to plot
        configs: Dictionary of configurations
        results_dir: Results directory
        components_to_show: List of components to show (None for all)
        stat_type: 'min' or 'mean' - which statistic to use
    """
    if components_to_show is None:
        components_to_show = COMPONENTS

    # Prepare data structure
    data_dict = {config: {} for config in configs.keys()}

    # Load data for this batch size
    for config_name in configs.keys():
        for component_name in components_to_show:
            df = load_component_data(
                config_name, component_name, configs, results_dir)
            if df is not None:
                # Filter for the specific batch size
                filtered = df[df['batch_size'] == batch_size]
                if not filtered.empty:
                    # Use min or mean based on stat_type
                    time_value = filtered[stat_type].values[0]
                    # Extract component short name
                    short_name = component_name.replace(
                        "layer_0_", "").replace("_statistics", "")
                    data_dict[config_name][short_name] = time_value * \
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
    stat_label = stat_type.capitalize()
    ax.set_title(
        f'Component Times Comparison at Batch Size {batch_size} ({stat_label} Time)',
        fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(components_found, rotation=45, ha='right')
    ax.legend(title='Parallel Strategy', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, components_found


def create_total_batch_size_comparison(total_batch_size, configs, results_dir, components_to_show=None, stat_type='min'):
    """
    Create a bar chart comparing component times at a specific total_batch_size
    (DP * batch_size) across different parallel strategies.

    Args:
        total_batch_size: The total batch size (DP * batch_size) to plot
        configs: Dictionary of configurations
        results_dir: Results directory
        components_to_show: List of components to show (None for all)
        stat_type: 'min' or 'mean' - which statistic to use
    """
    if components_to_show is None:
        components_to_show = COMPONENTS

    # Prepare data structure
    data_dict = {config: {} for config in configs.keys()}

    # Load data for this total_batch_size
    for config_name in configs.keys():
        dp_number = get_dp_number(config_name)
        for component_name in components_to_show:
            df = load_component_data(
                config_name, component_name, configs, results_dir)
            if df is not None:
                # Calculate total_batch_size for each row
                df_copy = df.copy()
                df_copy['total_batch_size'] = dp_number * df_copy['batch_size']
                # Filter for the specific total_batch_size
                filtered = df_copy[df_copy['total_batch_size']
                                   == total_batch_size]
                if not filtered.empty:
                    # Use min or mean based on stat_type
                    time_value = filtered[stat_type].values[0]
                    # Extract component short name
                    short_name = component_name.replace(
                        "layer_0_", "").replace("_statistics", "")
                    data_dict[config_name][short_name] = time_value * \
                        1000  # Convert to ms

    # Prepare data for plotting
    components_found = set()
    for config_data in data_dict.values():
        components_found.update(config_data.keys())

    if not components_found:
        print(f"No data found for total_batch_size {total_batch_size}")
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
    stat_label = stat_type.capitalize()
    ax.set_title(
        f'Component Times Comparison at Total Batch Size {total_batch_size} ({stat_label} Time)\n(DP × batch_size)',
        fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(components_found, rotation=45, ha='right')
    ax.legend(title='Parallel Strategy', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, components_found


def create_all_comparison_charts(configs, results_dir, output_dir):
    """Create comparison charts for all batch sizes (both min and mean).

    For每个 batch_size，本函数会按照如下分组各生成一张柱状图：
      1) MoE 相关: moe_core, moe_combine, moe_dispatch
      2) 主计算: self_attention, mlp
      3) 细分算子: attention_core, attention_prepare, mlp_experts, mlp_gate
    """
    batch_sizes = get_all_batch_sizes(configs, results_dir)

    print(f"Found {len(batch_sizes)} unique batch sizes: {batch_sizes}")

    # Create charts for specific batch sizes (to avoid too many charts)
    # Show charts for every 5 batch sizes or min/max
    batch_sizes_to_plot = set()

    # Always include first and last
    # if batch_sizes:
    #     batch_sizes_to_plot.add(batch_sizes[0])
    #     batch_sizes_to_plot.add(batch_sizes[-1])

    #     # Add every nth batch size for readability
    #     step = max(1, len(batch_sizes) // 5)  # Target ~5 charts
    #     for i in range(0, len(batch_sizes), step):
    #         batch_sizes_to_plot.add(batch_sizes[i])

    batch_sizes_to_plot = batch_sizes

    batch_sizes_to_plot = sorted(list(batch_sizes_to_plot))

    print(f"Creating charts for batch sizes: {batch_sizes_to_plot}")

    # 分别为三类组件生成柱状图（同时包含 DP1/DP2/DP4 的对比）
    group_specs = [
        ("moe", MOE_COMPONENTS),
        ("self_mlp", SELF_MLP_COMPONENTS),
        ("attn_mlp_detail", ATTN_MLP_DETAIL_COMPONENTS),
    ]

    # Create charts for both min and mean
    for stat_type in ['min', 'mean']:
        for batch_size in batch_sizes_to_plot:
            for group_key, components_to_show in group_specs:
                print(
                    f"Creating {stat_type} chart for batch size {batch_size} "
                    f"(group: {group_key})..."
                )
                result = create_batch_size_comparison(
                    batch_size,
                    configs,
                    results_dir,
                    components_to_show=components_to_show,
                    stat_type=stat_type,
                )

                if result is not None:
                    fig, components_found = result
                    output_path = output_dir / (
                        f"batch_size_{batch_size:05d}_comparison_"
                        f"{group_key}_{stat_type}.png"
                    )
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"Saved: {output_path}")
                    plt.close(fig)


def create_all_total_batch_size_comparison_charts(configs, results_dir, output_dir):
    """Create comparison charts for all total batch sizes (DP * batch_size) (both min and mean).

    与 create_all_comparison_charts 类似，这里按照 total_batch_size（DP × batch_size）
    为三类组件分别生成柱状图：
      1) MoE 相关: moe_core, moe_combine, moe_dispatch
      2) 主计算: self_attention, mlp
      3) 细分算子: attention_core, attention_prepare, mlp_experts, mlp_gate
    """
    total_batch_sizes = get_all_total_batch_sizes(configs, results_dir)

    print(
        f"Found {len(total_batch_sizes)} unique total batch sizes: {total_batch_sizes}")

    total_batch_sizes_to_plot = sorted(list(total_batch_sizes))

    print(
        f"Creating charts for total batch sizes: {total_batch_sizes_to_plot}")

    group_specs = [
        ("moe", MOE_COMPONENTS),
        ("self_mlp", SELF_MLP_COMPONENTS),
        ("attn_mlp_detail", ATTN_MLP_DETAIL_COMPONENTS),
    ]

    # Create charts for both min and mean
    for stat_type in ['min', 'mean']:
        for total_batch_size in total_batch_sizes_to_plot:
            for group_key, components_to_show in group_specs:
                print(
                    f"Creating {stat_type} chart for total_batch_size "
                    f"{total_batch_size} (group: {group_key})..."
                )
                result = create_total_batch_size_comparison(
                    total_batch_size,
                    configs,
                    results_dir,
                    components_to_show=components_to_show,
                    stat_type=stat_type,
                )

                if result is not None:
                    fig, components_found = result
                    output_path = output_dir / (
                        f"total_batch_size_{total_batch_size:05d}_comparison_"
                        f"{group_key}_{stat_type}.png"
                    )
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"Saved: {output_path}")
                    plt.close(fig)


def create_component_comparison_chart(configs, results_dir, output_dir):
    """
    Create comprehensive charts showing all components at different batch sizes.
    Creates separate charts for min and mean times.
    """
    batch_sizes = get_all_batch_sizes(configs, results_dir)

    # Create charts for both min and mean
    for stat_type in ['min', 'mean']:
        # Create subplots for each configuration
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        stat_label = stat_type.capitalize()
        fig.suptitle(f'{stat_label} Component Times vs Batch Size (Across Parallel Strategies)',
                     fontsize=16, fontweight='bold')

        config_names = sorted(configs.keys())

        for ax_idx, config_name in enumerate(config_names):
            ax = axes[ax_idx]

            # Prepare data for this config
            # Limit to first 5 for clarity
            for component_name in COMPONENTS[:5]:
                times = []
                valid_batch_sizes = []

                for batch_size in batch_sizes:
                    df = load_component_data(
                        config_name, component_name, configs, results_dir)
                    if df is not None:
                        filtered = df[df['batch_size'] == batch_size]
                        if not filtered.empty:
                            # Use min or mean based on stat_type
                            # Convert to ms
                            time_value = filtered[stat_type].values[0] * 1000
                            times.append(time_value)
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
        output_path = output_dir / \
            f"component_times_vs_batch_size_{stat_type}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comprehensive comparison ({stat_type}): {output_path}")
        plt.close(fig)


def create_total_batch_size_comparison_chart(configs, results_dir, output_dir):
    """
    Create comprehensive charts comparing component times at same total_batch_size
    (DP * batch_size) across different parallel strategies.
    Creates separate charts for min and mean times.
    """
    total_batch_sizes = get_all_total_batch_sizes(configs, results_dir)

    # Create charts for both min and mean
    for stat_type in ['min', 'mean']:
        # Create subplots for each component
        num_components = min(5, len(COMPONENTS))  # Limit to first 5 components
        fig, axes = plt.subplots(
            1, num_components, figsize=(5*num_components, 6))
        if num_components == 1:
            axes = [axes]

        stat_label = stat_type.capitalize()
        fig.suptitle(f'{stat_label} Component Times vs Total Batch Size (DP × batch_size) Comparison',
                     fontsize=16, fontweight='bold')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        config_names = sorted(configs.keys())

        for comp_idx, component_name in enumerate(COMPONENTS[:num_components]):
            ax = axes[comp_idx]

            # Prepare data for each configuration
            for config_name, color in zip(config_names, colors):
                dp_number = get_dp_number(config_name)
                times = []
                valid_total_batch_sizes = []

                for total_batch_size in total_batch_sizes:
                    df = load_component_data(
                        config_name, component_name, configs, results_dir)
                    if df is not None:
                        # Calculate total_batch_size for each row
                        df_copy = df.copy()
                        df_copy['total_batch_size'] = dp_number * \
                            df_copy['batch_size']
                        # Filter for the specific total_batch_size
                        filtered = df_copy[df_copy['total_batch_size']
                                           == total_batch_size]
                        if not filtered.empty:
                            # Use min or mean based on stat_type
                            # Convert to ms
                            time_value = filtered[stat_type].values[0] * 1000
                            times.append(time_value)
                            valid_total_batch_sizes.append(total_batch_size)

                if times:
                    ax.plot(valid_total_batch_sizes, times, marker='o',
                            label=config_name, linewidth=2, color=color, markersize=6)

            short_name = component_name.replace(
                "layer_0_", "").replace("_statistics", "")
            ax.set_xlabel('Total Batch Size (DP × batch_size)', fontsize=11)
            ax.set_ylabel('Time (ms)', fontsize=11)
            ax.set_title(short_name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')

        plt.tight_layout()
        output_path = output_dir / \
            f"component_times_vs_total_batch_size_{stat_type}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(
            f"Saved total batch size comparison ({stat_type}): {output_path}")
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
            df = load_component_data(
                config_name, component_name, configs, results_dir)
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

    # Create total batch size comparison charts (DP * batch_size)
    print("\n2. Creating total batch size comparison charts (DP × batch_size)...")
    create_all_total_batch_size_comparison_charts(
        configs, results_dir, output_dir)

    # Create comprehensive comparison chart
    print("\n3. Creating comprehensive component vs batch size chart...")
    create_component_comparison_chart(configs, results_dir, output_dir)

    # Create comprehensive total batch size comparison chart
    print("\n4. Creating comprehensive component vs total batch size chart...")
    create_total_batch_size_comparison_chart(configs, results_dir, output_dir)

    # Print summary statistics
    print("\n5. Summary statistics:")
    print_summary_statistics(configs, results_dir)

    print(f"\n✓ Analysis complete! Charts saved to: {output_dir}")
