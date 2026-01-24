# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for flash_attn_varlen_func performance measurement.
Compatible with latest vLLM version.
"""

import argparse
import csv
import os
import time
from typing import Optional
import multiprocessing as mp
from multiprocessing import Barrier, Lock
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import yaml

from vllm.platforms import current_platform
from vllm.attention.utils.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

# Qwen3 model configurations
QWEN3_MODEL_CONFIGS = {
    "0.6B": {
        "num_heads": (16, 8),
        "head_size": 128,
    },
    "4B": {
        "num_heads": (32, 8),
        "head_size": 128,
    },
    "32B": {
        "num_heads": (64, 8),
        "head_size": 128,
    },
}

# Default to Qwen3-4B for backward compatibility
DEFAULT_MODEL = "4B"

# Test configurations
NUM_HEADS = [(4, 4), (8, 2), (32, 8), (64, 8)]
HEAD_SIZES = [64, 128, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16, torch.float16]
QDTYPES = [None, torch.float8_e4m3fn]
NUM_BLOCKS = [2048, 8192, 32768]
SOFT_CAPS = [None, 50.0]
SLIDING_WINDOWS = [None, 256]
FA_VERSIONS = [2, 3]

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def benchmark_flash_attn_varlen(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    sliding_window: Optional[int],
    fa_version: Optional[int],
    q_dtype: Optional[torch.dtype],
    use_out: bool = True,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    process_id: int = 0,
    barrier: Optional[Barrier] = None,
) -> dict:
    """
    Benchmark flash_attn_varlen_func function (compatible with latest vLLM).

    Returns:
        dict: Benchmark results including mean, min, max times and throughput.
    """
    # Check if flash attention is available
    if not is_flash_attn_varlen_func_available():
        return {
            "error": "Flash attention varlen function is not available on this platform"
        }

    # Get FA version if not specified
    if fa_version is None:
        fa_version = get_flash_attn_version()
        if fa_version is None:
            return {
                "error": "Flash attention version could not be determined"
            }

    if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
        return {
            "error": "Flash attention with quantized inputs is only "
            "supported on version 3 with bfloat16 base type"
        }

    # Setup
    # Different seed for each process
    current_platform.seed_everything(42 + process_id)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    window_size = (
        [sliding_window - 1, 0] if sliding_window is not None else None
    )

    # Create tensors
    # Query shape: (num_tokens, num_query_heads, head_size)
    # For decode, each sequence has query_len=1
    query_len = 1
    num_tokens = num_seqs * query_len
    query = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)

    # KV cache shape: (num_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    # Create cu_seqlens_q: cumulative sequence lengths for queries
    # Shape: (num_seqs + 1,)
    # For decode with query_len=1, this is simply [0, 1, 2, ..., num_seqs]
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32)

    # seqused_k: KV cache sequence lengths
    # Shape: (num_seqs,)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32
    )

    out = torch.empty_like(query) if use_out else None

    # Handle quantization
    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)
        k_descale = torch.ones(scale_shape, dtype=torch.float32)
        v_descale = torch.ones(scale_shape, dtype=torch.float32)

    # Warmup
    for _ in range(warmup_iterations):
        _ = flash_attn_varlen_func(
            q=maybe_quantized_query,
            k=maybe_quantized_key_cache,
            v=maybe_quantized_value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            seqused_k=seqused_k,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    # If in multi-process mode, synchronize all processes before starting benchmark
    if barrier is not None:
        barrier.wait()

    total_time_start = time.perf_counter()

    # Synchronize before benchmarking
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        _ = flash_attn_varlen_func(
            q=maybe_quantized_query,
            k=maybe_quantized_key_cache,
            v=maybe_quantized_value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            seqused_k=seqused_k,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds


    if barrier is not None:
        barrier.wait()
    total_time_end = time.perf_counter()
    total_time = (total_time_end - total_time_start) * 1000

    # Calculate statistics
    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()

    # Calculate throughput (tokens per second)
    total_tokens = sum(kv_lens)
    throughput = total_tokens / (mean_time / 1000)  # tokens per second

    return {
        "total_time_ms": total_time,
        "mean_time_ms": mean_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "throughput_tokens_per_sec": throughput,
        "num_seqs": num_seqs,
        "total_tokens": total_tokens,
        "config": {
            "num_heads": num_heads,
            "head_size": head_size,
            "dtype": str(dtype),
            "block_size": block_size,
            "soft_cap": soft_cap,
            "num_blocks": num_blocks,
            "sliding_window": sliding_window,
            "fa_version": fa_version,
            "q_dtype": str(q_dtype) if q_dtype is not None else None,
            "kv_lens": kv_lens,
        }
    }


def print_benchmark_results(results: dict):
    """Print benchmark results in a formatted way."""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    config = results["config"]
    print("\n" + "="*80)
    print("Flash Attention Varlen Function Benchmark Results")
    print("="*80)
    print(f"Configuration:")
    print(f"  Num heads (Q, KV): {config['num_heads']}")
    print(f"  Head size: {config['head_size']}")
    print(f"  Dtype: {config['dtype']}")
    print(f"  Q dtype: {config['q_dtype']}")
    print(f"  Block size: {config['block_size']}")
    print(f"  Num blocks: {config['num_blocks']}")
    if "batch_size" in config and "kv_len" in config:
        print(f"  Batch size: {config['batch_size']}")
        print(f"  KV length: {config['kv_len']}")
    print(f"  KV lengths: {config['kv_lens']}")
    print(f"  Soft cap: {config['soft_cap']}")
    print(f"  Sliding window: {config['sliding_window']}")
    print(f"  FA version: {config['fa_version']}")
    print(f"\nPerformance:")
    print(f"  Mean time: {results['mean_time_ms']:.4f} ms")
    print(f"  Min time:  {results['min_time_ms']:.4f} ms")
    print(f"  Max time:  {results['max_time_ms']:.4f} ms")
    print(f"  Std dev:   {results['std_time_ms']:.4f} ms")
    print(f"  Total time: {results['total_time_ms']:.4f} ms")
    print(
        f"  Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Total tokens: {results['total_tokens']}")
    print("="*80 + "\n")


def save_results_to_csv(
    results: list,
    filename: str = "benchmark_flash_attn_v2_results.csv",
    output_dir: Optional[str] = None,
    file_lock: Optional[Lock] = None,
    process_id: int = 0,
    barrier: Optional[Barrier] = None,
    num_processes: int = 1,
    gpu_percentage: Optional[int] = None,
):
    """Save benchmark results to CSV file.

    Args:
        results: List of benchmark results
        filename: Output CSV filename
        output_dir: Output directory
        file_lock: Optional multiprocessing lock for file writing
        process_id: Process ID for multi-process mode
        barrier: Optional multiprocessing barrier for synchronization
        num_processes: Total number of processes
        gpu_percentage: GPU resource allocation percentage for this process
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Use process-specific log file if multiple processes
    if num_processes > 1:
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        data_path = os.path.join(
            output_dir, f"{base_name}_proc{process_id}{ext}") if output_dir else f"{base_name}_proc{process_id}{ext}"
    else:
        data_path = os.path.join(
            output_dir, filename) if output_dir else filename

    rows = []
    for result in results:
        if "error" in result:
            continue
        config = result["config"]
        row = {
            "batch_size": config.get("batch_size", ""),
            "kv_len": config.get("kv_len", ""),
            "num_heads_q": config["num_heads"][0] if isinstance(config["num_heads"], tuple) else "",
            "num_heads_kv": config["num_heads"][1] if isinstance(config["num_heads"], tuple) else "",
            "head_size": config["head_size"],
            "dtype": config["dtype"],
            "block_size": config["block_size"],
            "num_blocks": config["num_blocks"],
            "fa_version": config["fa_version"],
            "mean_time_ms": result["mean_time_ms"],
            "min_time_ms": result["min_time_ms"],
            "max_time_ms": result["max_time_ms"],
            "std_time_ms": result["std_time_ms"],
            "total_time_ms": result["total_time_ms"],
            "throughput_tokens_per_sec": result["throughput_tokens_per_sec"],
            "total_tokens": result["total_tokens"],
            "process_id": process_id,
            "gpu_percentage": gpu_percentage if gpu_percentage is not None else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(data_path):
        if file_lock:
            with file_lock:
                # Double check after acquiring lock
                if not os.path.exists(data_path):
                    df.to_csv(data_path, index=False)
        else:
            df.to_csv(data_path, index=False)
    else:
        # Append to existing file
        if file_lock:
            with file_lock:
                df.to_csv(data_path, mode='a', header=False, index=False)
        else:
            df.to_csv(data_path, mode='a', header=False, index=False)

    print(f"[Process {process_id}] Results saved to {data_path}")
    return df


def plot_results(
    df: pd.DataFrame,
    output_prefix: str = "benchmark_flash_attn_v2",
    output_dir: Optional[str] = None,
):
    """Create three interactive plots from benchmark results using Plotly."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter out rows with missing data
    df = df.dropna(subset=["batch_size", "kv_len", "mean_time_ms"])
    df = df.astype({"batch_size": int, "kv_len": int})

    # Normalize mean_time to start from 0
    df["mean_time_normalized"] = df["mean_time_ms"] - df["mean_time_ms"].min()

    # Create kv_len_batch_size column for plotting
    df["kv_len_batch_size"] = df["kv_len"] * df["batch_size"]

    # Create a subplot figure with 3 plots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "KV Length vs Mean Time (by Batch Size)",
            "Batch Size vs Mean Time (by KV Length)",
            "KV Length × Batch Size vs Mean Time (by Batch Size)"
        ),
        horizontal_spacing=0.1
    )

    # Plot 1: kv_len vs mean_time, grouped by batch_size
    colors = px.colors.qualitative.Set1
    for idx, batch_size in enumerate(sorted(df["batch_size"].unique())):
        subset = df[df["batch_size"] == batch_size].sort_values("kv_len")
        fig.add_trace(
            go.Scatter(
                x=subset["kv_len"],
                y=subset["mean_time_ms"],
                mode="lines+markers",
                name=f"batch_size={batch_size}",
                legendgroup="plot1",
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate="<b>Batch Size:</b> %{fullData.name}<br>" +
                              "<b>KV Length:</b> %{x}<br>" +
                              "<b>Mean Time:</b> %{y:.4f} ms<extra></extra>"
            ),
            row=1, col=1
        )

    fig.update_xaxes(title_text="KV Length", row=1, col=1)
    fig.update_yaxes(title_text="Mean Time (ms, normalized)", row=1, col=1)

    # Plot 2: batch_size vs mean_time, grouped by kv_len
    for idx, kv_len in enumerate(sorted(df["kv_len"].unique())):
        subset = df[df["kv_len"] == kv_len].sort_values("batch_size")
        fig.add_trace(
            go.Scatter(
                x=subset["batch_size"],
                y=subset["mean_time_ms"],
                mode="lines+markers",
                name=f"kv_len={kv_len}",
                legendgroup="plot2",
                line=dict(color=colors[idx %
                          len(colors)], width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate="<b>KV Length:</b> %{fullData.name}<br>" +
                              "<b>Batch Size:</b> %{x}<br>" +
                              "<b>Mean Time:</b> %{y:.4f} ms<extra></extra>",
                showlegend=True
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Batch Size", row=1, col=2)
    fig.update_yaxes(title_text="Mean Time (ms, normalized)", row=1, col=2)

    # Plot 3: kv_len*batch_size vs mean_time, grouped by batch_size
    for idx, batch_size in enumerate(sorted(df["batch_size"].unique())):
        subset = df[df["batch_size"] == batch_size].sort_values(
            "kv_len_batch_size")
        fig.add_trace(
            go.Scatter(
                x=subset["kv_len_batch_size"],
                y=subset["mean_time_ms"],
                mode="lines+markers",
                name=f"batch_size={batch_size}",
                legendgroup="plot3",
                line=dict(color=colors[idx %
                          len(colors)], width=2, dash="dot"),
                marker=dict(size=6),
                hovertemplate="<b>Batch Size:</b> %{fullData.name}<br>" +
                              "<b>KV Length × Batch Size:</b> %{x}<br>" +
                              "<b>Mean Time:</b> %{y:.4f} ms<extra></extra>",
                showlegend=True
            ),
            row=1, col=3
        )

    fig.update_xaxes(title_text="KV Length × Batch Size", row=1, col=3)
    fig.update_yaxes(title_text="Mean Time (ms, normalized)", row=1, col=3)

    # Update layout
    fig.update_layout(
        title_text="Flash Attention Benchmark Results",
        title_x=0.5,
        height=600,
        width=1800,
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    # Save as HTML
    html_path = os.path.join(
        output_dir, f"{output_prefix}_interactive.html") if output_dir else f"{output_prefix}_interactive.html"
    fig.write_html(html_path)
    print(f"Saved interactive plot: {html_path}")

    # Save as PNG using matplotlib (three subplots)
    fig_combined, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: kv_len vs mean_time, grouped by batch_size
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size].sort_values("kv_len")
        axes[0].plot(subset["kv_len"], subset["mean_time_normalized"],
                     marker="o", label=f"batch_size={batch_size}")
    axes[0].set_xlabel("KV Length", fontsize=10)
    axes[0].set_ylabel("Mean Time (ms, normalized)", fontsize=10)
    axes[0].set_title("KV Length vs Mean Time (by Batch Size)", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: batch_size vs mean_time, grouped by kv_len
    for kv_len in sorted(df["kv_len"].unique()):
        subset = df[df["kv_len"] == kv_len].sort_values("batch_size")
        axes[1].plot(subset["batch_size"], subset["mean_time_normalized"],
                     marker="o", label=f"kv_len={kv_len}")
    axes[1].set_xlabel("Batch Size", fontsize=10)
    axes[1].set_ylabel("Mean Time (ms, normalized)", fontsize=10)
    axes[1].set_title("Batch Size vs Mean Time (by KV Length)", fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: kv_len*batch_size vs mean_time, grouped by batch_size
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size].sort_values(
            "kv_len_batch_size")
        axes[2].plot(subset["kv_len_batch_size"], subset["mean_time_normalized"],
                     marker="o", label=f"batch_size={batch_size}")
    axes[2].set_xlabel("KV Length × Batch Size", fontsize=10)
    axes[2].set_ylabel("Mean Time (ms, normalized)", fontsize=10)
    axes[2].set_title(
        "KV Length × Batch Size vs Mean Time (by Batch Size)", fontsize=11)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Flash Attention Benchmark Results", fontsize=14, y=1.02)
    plt.tight_layout()
    png_path = os.path.join(
        output_dir, f"{output_prefix}_interactive.png") if output_dir else f"{output_prefix}_interactive.png"
    plt.savefig(png_path, dpi=300)
    print(f"Saved PNG plot: {png_path}")
    plt.close()

    # Also create individual HTML files for each plot
    # Plot 1: kv_len vs mean_time, grouped by batch_size
    fig1 = go.Figure()
    for idx, batch_size in enumerate(sorted(df["batch_size"].unique())):
        subset = df[df["batch_size"] == batch_size].sort_values("kv_len")
        fig1.add_trace(
            go.Scatter(
                x=subset["kv_len"],
                y=subset["mean_time_ms"],
                mode="lines+markers",
                name=f"batch_size={batch_size}",
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8),
                hovertemplate="<b>Batch Size:</b> %{fullData.name}<br>" +
                              "<b>KV Length:</b> %{x}<br>" +
                              "<b>Mean Time:</b> %{y:.4f} ms<extra></extra>"
            )
        )
    fig1.update_layout(
        title="Flash Attention: KV Length vs Mean Time (by Batch Size)",
        xaxis_title="KV Length",
        yaxis_title="Mean Time (ms, normalized)",
        height=600,
        width=1000,
        hovermode="closest",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    plot1_path = os.path.join(
        output_dir, f"{output_prefix}_kv_len_vs_time.html") if output_dir else f"{output_prefix}_kv_len_vs_time.html"
    fig1.write_html(plot1_path)
    print(f"Saved plot: {plot1_path}")

    # Save as PNG using matplotlib
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size].sort_values("kv_len")
        plt.plot(subset["kv_len"], subset["mean_time_normalized"],
                 marker="o", label=f"batch_size={batch_size}")
    plt.xlabel("KV Length", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Flash Attention: KV Length vs Mean Time (by Batch Size)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_png_path = os.path.join(
        output_dir, f"{output_prefix}_kv_len_vs_time.png") if output_dir else f"{output_prefix}_kv_len_vs_time.png"
    plt.savefig(plot1_png_path, dpi=300)
    print(f"Saved PNG plot: {plot1_png_path}")
    plt.close()

    # Plot 2: batch_size vs mean_time, grouped by kv_len
    fig2 = go.Figure()
    for idx, kv_len in enumerate(sorted(df["kv_len"].unique())):
        subset = df[df["kv_len"] == kv_len].sort_values("batch_size")
        fig2.add_trace(
            go.Scatter(
                x=subset["batch_size"],
                y=subset["mean_time_ms"],
                mode="lines+markers",
                name=f"kv_len={kv_len}",
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8),
                hovertemplate="<b>KV Length:</b> %{fullData.name}<br>" +
                              "<b>Batch Size:</b> %{x}<br>" +
                              "<b>Mean Time:</b> %{y:.4f} ms<extra></extra>"
            )
        )
    fig2.update_layout(
        title="Flash Attention: Batch Size vs Mean Time (by KV Length)",
        xaxis_title="Batch Size",
        yaxis_title="Mean Time (ms, normalized)",
        height=600,
        width=1000,
        hovermode="closest",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    plot2_path = os.path.join(
        output_dir, f"{output_prefix}_batch_size_vs_time.html") if output_dir else f"{output_prefix}_batch_size_vs_time.html"
    fig2.write_html(plot2_path)
    print(f"Saved plot: {plot2_path}")

    # Save as PNG using matplotlib
    plt.figure(figsize=(12, 8))
    for kv_len in sorted(df["kv_len"].unique()):
        subset = df[df["kv_len"] == kv_len].sort_values("batch_size")
        plt.plot(subset["batch_size"], subset["mean_time_normalized"],
                 marker="o", label=f"kv_len={kv_len}")
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Flash Attention: Batch Size vs Mean Time (by KV Length)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_png_path = os.path.join(
        output_dir, f"{output_prefix}_batch_size_vs_time.png") if output_dir else f"{output_prefix}_batch_size_vs_time.png"
    plt.savefig(plot2_png_path, dpi=300)
    print(f"Saved PNG plot: {plot2_png_path}")
    plt.close()

    # Plot 3: kv_len*batch_size vs mean_time, grouped by batch_size
    fig3 = go.Figure()
    for idx, batch_size in enumerate(sorted(df["batch_size"].unique())):
        subset = df[df["batch_size"] == batch_size].sort_values(
            "kv_len_batch_size")
        fig3.add_trace(
            go.Scatter(
                x=subset["kv_len_batch_size"],
                y=subset["mean_time_ms"],
                mode="lines+markers",
                name=f"batch_size={batch_size}",
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8),
                hovertemplate="<b>Batch Size:</b> %{fullData.name}<br>" +
                              "<b>KV Length × Batch Size:</b> %{x}<br>" +
                              "<b>Mean Time:</b> %{y:.4f} ms<extra></extra>"
            )
        )
    fig3.update_layout(
        title="Flash Attention: KV Length × Batch Size vs Mean Time (by Batch Size)",
        xaxis_title="KV Length × Batch Size",
        yaxis_title="Mean Time (ms, normalized)",
        height=600,
        width=1000,
        hovermode="closest",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    plot3_path = os.path.join(
        output_dir, f"{output_prefix}_kv_len_batch_size_vs_time.html") if output_dir else f"{output_prefix}_kv_len_batch_size_vs_time.html"
    fig3.write_html(plot3_path)
    print(f"Saved plot: {plot3_path}")

    # Save as PNG using matplotlib
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size].sort_values(
            "kv_len_batch_size")
        plt.plot(subset["kv_len_batch_size"], subset["mean_time_normalized"],
                 marker="o", label=f"batch_size={batch_size}")
    plt.xlabel("KV Length × Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title(
        "Flash Attention: KV Length × Batch Size vs Mean Time (by Batch Size)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_png_path = os.path.join(
        output_dir, f"{output_prefix}_kv_len_batch_size_vs_time.png") if output_dir else f"{output_prefix}_kv_len_batch_size_vs_time.png"
    plt.savefig(plot3_png_path, dpi=300)
    print(f"Saved PNG plot: {plot3_png_path}")
    plt.close()


def run_benchmark_suite(
    batch_sizes: Optional[list[int]] = None,
    kv_lens: Optional[list[int]] = None,
    output_dir: Optional[str] = None,
    model_size: str = DEFAULT_MODEL,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    barrier: Optional[Barrier] = None,
    file_lock: Optional[Lock] = None,
    process_id: int = 0,
    num_processes: int = 1,
    filename: str = "benchmark_flash_attn_v2_results.csv",
    gpu_percentage: Optional[int] = None,
):
    """Run a comprehensive benchmark suite.

    Args:
        batch_sizes: List of batch sizes to test. If None, uses default values.
        kv_lens: List of KV lengths to test. If None, uses default values.
        output_dir: Directory to save output files. If None, saves to current directory.
        model_size: Model size to use ("0.6B", "4B", or "32B"). Defaults to "4B".
        warmup_iterations: Number of warmup iterations.
        benchmark_iterations: Number of benchmark iterations.
        barrier: Optional multiprocessing barrier for synchronization.
        file_lock: Optional multiprocessing lock for file writing.
        process_id: Process ID for multi-process mode.
        num_processes: Total number of processes.
        filename: Output CSV filename.
        gpu_percentage: GPU resource allocation percentage for this process.
    """
    # Get model configuration
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}"
        )
    # Synchronize all processes at the start to ensure they begin together
    if barrier:
        barrier.wait()

    model_config = QWEN3_MODEL_CONFIGS[model_size]
    num_heads = model_config["num_heads"]
    head_size = model_config["head_size"]

    print(
        f"[Process {process_id}] Starting Flash Attention Varlen Function Benchmark Suite...")
    print(f"[Process {process_id}] Model: Qwen3-{model_size}")
    print(f"[Process {process_id}]   Num heads (Q, KV): {num_heads}")
    print(f"[Process {process_id}]   Head size: {head_size}")
    print(f"[Process {process_id}] Warmup iterations: {warmup_iterations}")
    print(
        f"[Process {process_id}] Benchmark iterations: {benchmark_iterations}")
    print(f"[Process {process_id}] Number of processes: {num_processes}\n")

    torch.set_default_device("cuda")

    # Test different combinations of batch_size and kv_len
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if kv_lens is None:
        kv_lens = [1024, 2048, 4096, 8192]

    # Example benchmark configurations
    # Use batch_size and kv_len instead of kv_lens list
    test_configs = [
        {
            "batch_size": batch_size,
            "kv_len": kv_len,
            "num_heads": num_heads,
            "head_size": head_size,
            "dtype": torch.bfloat16,
            "block_size": 16,
            "soft_cap": None,
            "num_blocks": 8192,
            "sliding_window": None,
            "fa_version": 2,  # Fixed to 2
            "q_dtype": None,
        }
        for batch_size in batch_sizes
        for kv_len in kv_lens
    ]

    # All processes execute the same configurations synchronously
    total_configs = len(test_configs)

    print(f"[Process {process_id}] Processing {total_configs} configurations "
          f"(all processes execute the same configurations synchronously)")

    if barrier:
        barrier.wait()  # Synchronize before starting benchmarks

    print(f"[Process {process_id}] Synchronized before starting benchmarks")
    all_results = []
    for i, config in enumerate(test_configs, 1):
        print(
            f"\n[Process {process_id}] Running benchmark {i}/{total_configs}...")

        # Create a copy to avoid modifying the original config
        config_copy = config.copy()

        # Extract batch_size and kv_len, generate kv_lens list
        batch_size = config_copy.pop("batch_size")
        kv_len = config_copy.pop("kv_len")
        kv_lens_list = [kv_len] * batch_size

        # Synchronize all processes before executing this configuration
        if barrier:
            barrier.wait()

        print(
            f"[Process {process_id}] Synchronized before running benchmark {i}")
        # Run benchmark with generated kv_lens
        results = benchmark_flash_attn_varlen(
            kv_lens=kv_lens_list,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            process_id=process_id,
            barrier=barrier,
            **config_copy)

        # Add batch_size and kv_len to results config
        if "config" in results:
            results["config"]["batch_size"] = batch_size
            results["config"]["kv_len"] = kv_len

        all_results.append(results)
        print_benchmark_results(results)

    # Save results to CSV (plots will be created after all processes complete)
    df = save_results_to_csv(
        all_results,
        filename=filename,
        output_dir=output_dir,
        file_lock=file_lock,
        process_id=process_id,
        barrier=barrier,
        num_processes=num_processes,
        gpu_percentage=gpu_percentage
    )

    return all_results


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got: {type(data)}")
    return data


def _get_process_sweep_from_yaml(
    yaml_cfg: dict[str, Any],
    process_id: int,
    cli_batch_sizes: Optional[list[int]],
    cli_kv_lens: Optional[list[int]],
) -> tuple[Optional[list[int]], Optional[list[int]]]:
    """
    Resolve (batch_sizes, kv_lens) for a given process.

    Priority:
      1) YAML per-process override
      2) YAML default
      3) CLI args (which may be None -> fall back to run_benchmark_suite defaults)

    Supported YAML shapes:
      - default: {batch_sizes: [...], kv_lens: [...]}
        processes:
          - id: 0
            batch_sizes: [...]
            kv_lens: [...]
          - id: 1
            ...

      - default: ...
        process_overrides:
          "0": {batch_sizes: [...], kv_lens: [...]}
          "1": {batch_sizes: [...], kv_lens: [...]}
    """
    default_cfg = yaml_cfg.get("default", {}) if isinstance(yaml_cfg.get("default", {}), dict) else {}

    batch_sizes: Optional[list[int]] = default_cfg.get("batch_sizes", cli_batch_sizes)
    kv_lens: Optional[list[int]] = default_cfg.get("kv_lens", cli_kv_lens)

    # dict-style overrides
    proc_overrides = yaml_cfg.get("process_overrides", None)
    if isinstance(proc_overrides, dict):
        p = proc_overrides.get(str(process_id), proc_overrides.get(process_id))
        if isinstance(p, dict):
            if "batch_sizes" in p:
                batch_sizes = p["batch_sizes"]
            if "kv_lens" in p:
                kv_lens = p["kv_lens"]

    # list-style overrides
    procs_list = yaml_cfg.get("processes", None)
    if isinstance(procs_list, list):
        for p in procs_list:
            if not isinstance(p, dict):
                continue
            if p.get("id") == process_id:
                if "batch_sizes" in p:
                    batch_sizes = p["batch_sizes"]
                if "kv_lens" in p:
                    kv_lens = p["kv_lens"]
                break

    # Basic validation if provided
    def _validate_int_list(name: str, v: Any) -> Optional[list[int]]:
        if v is None:
            return None
        if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
            raise ValueError(f"{name} must be a list[int] or null, got: {v!r}")
        return v

    return _validate_int_list("batch_sizes", batch_sizes), _validate_int_list("kv_lens", kv_lens)


def _process_worker(
    *,
    batch_sizes: Optional[list[int]],
    kv_lens: Optional[list[int]],
    output_dir: Optional[str],
    model_size: str,
    warmup_iterations: int,
    benchmark_iterations: int,
    barrier: Optional[Barrier],
    file_lock: Optional[Lock],
    process_id: int,
    num_processes: int,
    filename: str,
    process_gpu_percentage: Optional[int],
):
    # NOTE: multiprocessing.Process doesn't support env=...; set env in the child process.
    if process_gpu_percentage is not None:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(process_gpu_percentage)

    run_benchmark_suite(
        batch_sizes=batch_sizes,
        kv_lens=kv_lens,
        output_dir=output_dir,
        model_size=model_size,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        barrier=barrier,
        file_lock=file_lock,
        process_id=process_id,
        num_processes=num_processes,
        filename=filename,
        gpu_percentage=process_gpu_percentage,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark flash_attn_varlen_func performance (compatible with latest vLLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="List of batch sizes to test (e.g., --batch-sizes 1 2 4 8). "
        "If not specified, uses default: [1, 2, 4, 8, 16, 32, 64, 128, 256]",
    )
    parser.add_argument(
        "--kv-lens",
        type=int,
        nargs="+",
        default=None,
        help="List of KV lengths to test (e.g., --kv-lens 1024 2048 4096). "
        "If not specified, uses default: [1024, 2048, 4096, 8192]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (CSV and plots). "
        "If not specified, saves to current directory.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(QWEN3_MODEL_CONFIGS.keys()),
        help=f"Model size to benchmark. Options: {list(QWEN3_MODEL_CONFIGS.keys())}. "
        f"Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=WARMUP_ITERATIONS,
        help=f"Number of warmup iterations. Default: {WARMUP_ITERATIONS}",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=BENCHMARK_ITERATIONS,
        help=f"Number of benchmark iterations. Default: {BENCHMARK_ITERATIONS}",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes to run in parallel. Default: 1",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="benchmark_flash_attn_v2_results.csv",
        help="Output CSV filename. Default: benchmark_flash_attn_v2_results.csv",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (useful for multi-process mode)",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="YAML config file to define default + per-process sweep parameters "
        "(e.g., different batch_sizes/kv_lens per process).",
    )


    # Process GPU resource allocation
    parser.add_argument('--process-gpu-percentage', type=int, nargs='+', default=None,
                        help='GPU resource allocation percentage for each process (list of integers)')
    args = parser.parse_args()

    print("="*60)
    print("Flash Attention Varlen Function Benchmark (vLLM Compatible)")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(
        f"Batch sizes: {args.batch_sizes if args.batch_sizes else 'default'}")
    print(f"KV lengths: {args.kv_lens if args.kv_lens else 'default'}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")
    print(f"Number of processes: {args.num_processes}")
    print(
        f"Output directory: {args.output_dir if args.output_dir else 'current directory'}")
    if args.process_gpu_percentage is not None:
        print(f"GPU Percentage Per Process: {args.process_gpu_percentage}")
    print("="*60)

    yaml_cfg: dict[str, Any] = {}
    if args.config_yaml:
        yaml_cfg = _load_yaml_config(args.config_yaml)

    # Validate GPU percentage configuration
    if args.process_gpu_percentage is not None:
        if len(args.process_gpu_percentage) != args.num_processes:
            raise ValueError(
                f"process_gpu_percentage length ({len(args.process_gpu_percentage)}) must match "
                f"num_processes ({args.num_processes})"
            )

    # Create barrier and lock for multiprocessing (unified for both single and multi-process)
    barrier = Barrier(args.num_processes) if args.num_processes > 1 else None
    file_lock = Lock() if args.num_processes > 1 else None

    # Create and start processes (unified execution path)
    processes = []
    for i in range(args.num_processes):
        proc_batch_sizes, proc_kv_lens = _get_process_sweep_from_yaml(
            yaml_cfg=yaml_cfg,
            process_id=i,
            cli_batch_sizes=args.batch_sizes,
            cli_kv_lens=args.kv_lens,
        )
        proc_gpu_pct = (
            args.process_gpu_percentage[i]
            if args.process_gpu_percentage is not None
            else None
        )
        p = mp.Process(
            target=_process_worker,
            kwargs={
                "batch_sizes": proc_batch_sizes,
                "kv_lens": proc_kv_lens,
                "output_dir": args.output_dir,
                "model_size": args.model_size,
                "warmup_iterations": args.warmup_iterations,
                "benchmark_iterations": args.benchmark_iterations,
                "barrier": barrier,
                "file_lock": file_lock,
                "process_id": i,
                "num_processes": args.num_processes,
                "filename": args.filename,
                "process_gpu_percentage": proc_gpu_pct,
            },
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"\nAll {args.num_processes} process(es) completed.")
    if args.output_dir:
        print(
            f"Results saved to: {os.path.join(args.output_dir, args.filename)}")
    else:
        print(f"Results saved to: {args.filename}")
    if args.num_processes > 1:
        print(f"Note: Each process wrote to a separate file (proc0, proc1, ...)")

    # Generate plots from combined results if not skipped
    if not args.skip_plots:
        print("\nGenerating plots from combined results...")
        # Collect all results from process-specific files
        all_dfs = []
        for i in range(args.num_processes):
            if args.num_processes > 1:
                base_name = os.path.splitext(args.filename)[0]
                ext = os.path.splitext(args.filename)[1]
                proc_filename = f"{base_name}_proc{i}{ext}"
            else:
                proc_filename = args.filename
            proc_path = os.path.join(
                args.output_dir, proc_filename) if args.output_dir else proc_filename
            if os.path.exists(proc_path):
                df = pd.read_csv(proc_path)
                all_dfs.append(df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            plot_results(combined_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
