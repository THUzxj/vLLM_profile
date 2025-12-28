# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Script to plot Attention Layer and MLP component benchmarks on the same figure.
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_data(
    attention_csv: str,
    mlp_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV data for both components."""
    attn_df = pd.read_csv(attention_csv)
    mlp_df = pd.read_csv(mlp_csv)

    # Filter out rows with missing data
    attn_df = attn_df.dropna(subset=["batch_size", "kv_len", "mean_time_ms"])
    mlp_df = mlp_df.dropna(subset=["batch_size", "mean_time_ms"])

    # Convert to appropriate types
    attn_df = attn_df.astype({"batch_size": int, "kv_len": int})
    mlp_df = mlp_df.astype({"batch_size": int})

    return attn_df, mlp_df


def plot_batch_size_comparison(
    attn_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    output_prefix: str = "benchmark_combined_components",
):
    """Plot batch_size vs mean_time for both components."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot MLP (seq_len=1, decode phase)
    mlp_df_sorted = mlp_df.sort_values("batch_size")
    plt.plot(
        mlp_df_sorted["batch_size"],
        mlp_df_sorted["mean_time_ms"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="MLP (seq_len=1)",
        color="blue",
    )

    # Plot Attention Layer for different kv_len values
    colors = ["red", "orange", "green", "purple"]
    kv_lens = sorted(attn_df["kv_len"].unique())

    for i, kv_len in enumerate(kv_lens):
        subset = attn_df[attn_df["kv_len"] == kv_len]
        subset = subset.sort_values("batch_size")
        color = colors[i % len(colors)]
        plt.plot(
            subset["batch_size"],
            subset["mean_time_ms"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=f"Attention Layer (kv_len={kv_len})",
            color=color,
            linestyle="--",
        )

    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms)", fontsize=12)
    plt.title(
        "Component Comparison: Batch Size vs Mean Time",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    # plt.xscale("log", base=2)
    # plt.yscale("log")
    plt.tight_layout()

    plot_path = (
        os.path.join(output_dir, f"{output_prefix}_batch_size_vs_time.png")
        if output_dir
        else f"{output_prefix}_batch_size_vs_time.png"
    )
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()


def plot_kv_len_comparison(
    attn_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    output_prefix: str = "benchmark_combined_components",
):
    """Plot kv_len vs mean_time for attention layer, with MLP as reference."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot MLP as horizontal reference line (constant time for different kv_len)
    # Use average MLP time across all batch sizes, or show for each batch size
    for batch_size in sorted(mlp_df["batch_size"].unique()):
        mlp_subset = mlp_df[mlp_df["batch_size"] == batch_size]
        if not mlp_subset.empty:
            mlp_time = mlp_subset["mean_time_ms"].iloc[0]
            plt.axhline(
                y=mlp_time,
                xmin=0,
                xmax=1,
                color="blue",
                linestyle=":",
                linewidth=1.5,
                alpha=0.6,
                label=f"MLP (batch_size={batch_size})" if batch_size in [
                    1, 8, 32, 128, 256] else "",
            )

    # Plot Attention Layer for different batch sizes
    colors = ["red", "orange", "green", "purple",
              "brown", "pink", "gray", "olive", "cyan"]
    batch_sizes = sorted(attn_df["batch_size"].unique())

    for i, batch_size in enumerate(batch_sizes):
        subset = attn_df[attn_df["batch_size"] == batch_size]
        subset = subset.sort_values("kv_len")
        color = colors[i % len(colors)]
        plt.plot(
            subset["kv_len"],
            subset["mean_time_ms"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"Attention Layer (batch_size={batch_size})",
            color=color,
        )

    plt.xlabel("KV Length", fontsize=12)
    plt.ylabel("Mean Time (ms)", fontsize=12)
    plt.title(
        "Attention Layer: KV Length vs Mean Time (with MLP reference)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=9, loc="best", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.tight_layout()

    plot_path = (
        os.path.join(output_dir, f"{output_prefix}_kv_len_vs_time.png")
        if output_dir
        else f"{output_prefix}_kv_len_vs_time.png"
    )
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()


def plot_throughput_comparison(
    attn_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    output_prefix: str = "benchmark_combined_components",
):
    """Plot batch_size vs throughput for both components."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot MLP throughput
    mlp_df_sorted = mlp_df.sort_values("batch_size")
    plt.plot(
        mlp_df_sorted["batch_size"],
        mlp_df_sorted["throughput_tokens_per_sec"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="MLP (seq_len=1)",
        color="blue",
    )

    # Plot Attention Layer throughput for different kv_len values
    colors = ["red", "orange", "green", "purple"]
    kv_lens = sorted(attn_df["kv_len"].unique())

    for i, kv_len in enumerate(kv_lens):
        subset = attn_df[attn_df["kv_len"] == kv_len]
        subset = subset.sort_values("batch_size")
        color = colors[i % len(colors)]
        plt.plot(
            subset["batch_size"],
            subset["throughput_tokens_per_sec"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=f"Attention Layer (kv_len={kv_len})",
            color=color,
            linestyle="--",
        )

    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Throughput (tokens/sec)", fontsize=12)
    plt.title(
        "Component Comparison: Batch Size vs Throughput",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.tight_layout()

    plot_path = (
        os.path.join(
            output_dir, f"{output_prefix}_batch_size_vs_throughput.png")
        if output_dir
        else f"{output_prefix}_batch_size_vs_throughput.png"
    )
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot combined Attention Layer and MLP benchmark results"
    )
    parser.add_argument(
        "--attention-csv",
        type=str,
        default="sweep_profile_result/benchmark_attention_layer_results.csv",
        help="Path to attention layer results CSV file",
    )
    parser.add_argument(
        "--mlp-csv",
        type=str,
        default="sweep_profile_result/benchmark_qwen3_mlp_results.csv",
        help="Path to MLP results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sweep_profile_result",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="benchmark_combined_components",
        help="Prefix for output plot filenames",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    attn_df, mlp_df = load_data(args.attention_csv, args.mlp_csv)

    print(f"Loaded {len(attn_df)} attention layer records")
    print(f"Loaded {len(mlp_df)} MLP records")

    # Create plots
    print("\nCreating plots...")
    plot_batch_size_comparison(
        attn_df, mlp_df, args.output_dir, args.output_prefix)
    plot_kv_len_comparison(
        attn_df, mlp_df, args.output_dir, args.output_prefix)
    plot_throughput_comparison(
        attn_df, mlp_df, args.output_dir, args.output_prefix)

    print("\nAll plots created successfully!")


if __name__ == "__main__":
    main()
