# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for flash_attn_with_kvcache performance measurement.
"""

import argparse
import csv
import os
import time
from typing import Optional
import multiprocessing as mp
from multiprocessing import Barrier, Lock

import matplotlib.pyplot as plt
import pandas as pd
import torch

from vllm.platforms import current_platform
from vllm.vllm_flash_attn.flash_attn_interface import (fa_version_unsupported_reason,
                                  flash_attn_with_kvcache,
                                  is_fa_version_supported)

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


def benchmark_flash_attn_with_kvcache(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    sliding_window: Optional[int],
    fa_version: int,
    q_dtype: Optional[torch.dtype],
    use_out: bool = True,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    process_id: int = 0,
) -> dict:
    """
    Benchmark flash_attn_with_kvcache function.

    Returns:
        dict: Benchmark results including mean, min, max times and throughput.
    """
    # Check if FA version is supported
    if not is_fa_version_supported(fa_version):
        return {
            "error": f"Flash attention version {fa_version} not supported: "
            f"{fa_version_unsupported_reason(fa_version)}"
        }

    if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
        return {
            "error": "Flash attention with quantized inputs is only "
            "supported on version 3 with bfloat16 base type"
        }

    # Setup
    current_platform.seed_everything(42 + process_id)  # Different seed for each process
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    window_size = ((sliding_window - 1, 0) if sliding_window is not None else
                   (-1, -1))

    # Create tensors
    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    q = query.unsqueeze(1)
    out = torch.empty_like(q) if use_out else None

    # Handle quantization
    maybe_quantized_query = q
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        maybe_quantized_query = q.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)
        k_descale = torch.ones(scale_shape, dtype=torch.float32)
        v_descale = torch.ones(scale_shape, dtype=torch.float32)

    # Warmup
    for _ in range(warmup_iterations):
        _ = flash_attn_with_kvcache(
            q=maybe_quantized_query,
            k_cache=maybe_quantized_key_cache,
            v_cache=maybe_quantized_value_cache,
            out=out,
            softmax_scale=scale,
            causal=True,
            block_table=block_tables,
            cache_seqlens=kv_lens_tensor,
            softcap=soft_cap if soft_cap is not None else 0,
            window_size=window_size,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    # Synchronize before benchmarking
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        _ = flash_attn_with_kvcache(
            q=maybe_quantized_query,
            k_cache=maybe_quantized_key_cache,
            v_cache=maybe_quantized_value_cache,
            out=out,
            softmax_scale=scale,
            causal=True,
            block_table=block_tables,
            cache_seqlens=kv_lens_tensor,
            softcap=soft_cap if soft_cap is not None else 0,
            window_size=window_size,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

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
    print("Flash Attention with KV Cache Benchmark Results")
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
    print(
        f"  Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Total tokens: {results['total_tokens']}")
    print("="*80 + "\n")


def save_results_to_csv(
    results: list,
    filename: str = "benchmark_flash_attn_results.csv",
    output_dir: Optional[str] = None,
    file_lock: Optional[Lock] = None,
    process_id: int = 0,
    barrier: Optional[Barrier] = None,
    num_processes: int = 1,
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
        data_path = os.path.join(output_dir, filename) if output_dir else filename

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
            "throughput_tokens_per_sec": result["throughput_tokens_per_sec"],
            "total_tokens": result["total_tokens"],
            "process_id": process_id,
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
    output_prefix: str = "benchmark_flash_attn",
    output_dir: Optional[str] = None,
):
    """Create three plots from benchmark results."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter out rows with missing data
    df = df.dropna(subset=["batch_size", "kv_len", "mean_time_ms"])
    df = df.astype({"batch_size": int, "kv_len": int})

    # Normalize mean_time to start from 0
    df["mean_time_normalized"] = df["mean_time_ms"] - df["mean_time_ms"].min()

    # Plot 1: kv_len vs mean_time, grouped by batch_size
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size]
        subset = subset.sort_values("kv_len")
        plt.plot(subset["kv_len"], subset["mean_time_normalized"],
                 marker="o", label=f"batch_size={batch_size}")
    plt.xlabel("KV Length", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Flash Attention: KV Length vs Mean Time (by Batch Size)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_kv_len_vs_time.png") if output_dir else f"{output_prefix}_kv_len_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 2: batch_size vs mean_time, grouped by kv_len
    plt.figure(figsize=(12, 8))
    for kv_len in sorted(df["kv_len"].unique()):
        subset = df[df["kv_len"] == kv_len]
        subset = subset.sort_values("batch_size")
        plt.plot(subset["batch_size"], subset["mean_time_normalized"],
                 marker="o", label=f"kv_len={kv_len}")
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Flash Attention: Batch Size vs Mean Time (by KV Length)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_batch_size_vs_time.png") if output_dir else f"{output_prefix}_batch_size_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 3: kv_len*batch_size vs mean_time, grouped by batch_size
    df["kv_len_batch_size"] = df["kv_len"] * df["batch_size"]
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size]
        subset = subset.sort_values("kv_len_batch_size")
        plt.plot(subset["kv_len_batch_size"], subset["mean_time_normalized"],
                 marker="o", label=f"batch_size={batch_size}")
    plt.xlabel("KV Length × Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title(
        "Flash Attention: KV Length × Batch Size vs Mean Time (by Batch Size)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_kv_len_batch_size_vs_time.png") if output_dir else f"{output_prefix}_kv_len_batch_size_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
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
    filename: str = "benchmark_flash_attn_results.csv",
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
    """
    # Get model configuration
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}"
        )
    model_config = QWEN3_MODEL_CONFIGS[model_size]
    num_heads = model_config["num_heads"]
    head_size = model_config["head_size"]

    print(f"[Process {process_id}] Starting Flash Attention with KV Cache Benchmark Suite...")
    print(f"[Process {process_id}] Model: Qwen3-{model_size}")
    print(f"[Process {process_id}]   Num heads (Q, KV): {num_heads}")
    print(f"[Process {process_id}]   Head size: {head_size}")
    print(f"[Process {process_id}] Warmup iterations: {warmup_iterations}")
    print(f"[Process {process_id}] Benchmark iterations: {benchmark_iterations}")
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
            "fa_version": 3,
            "q_dtype": None,
        }
        for batch_size in batch_sizes
        for kv_len in kv_lens
    ]

    # Distribute configurations across processes
    total_configs = len(test_configs)
    configs_per_process = total_configs // num_processes
    remainder = total_configs % num_processes
    
    # Calculate start and end indices for this process
    start_idx = process_id * configs_per_process + min(process_id, remainder)
    end_idx = start_idx + configs_per_process + (1 if process_id < remainder else 0)
    
    process_configs = test_configs[start_idx:end_idx]
    process_total_configs = len(process_configs)

    print(f"[Process {process_id}] Processing {process_total_configs} configurations "
          f"(indices {start_idx} to {end_idx-1} out of {total_configs} total)")

    if barrier:
        barrier.wait()  # Synchronize before starting benchmarks

    print(f"[Process {process_id}] Synchronized before starting benchmarks")
    all_results = []
    for i, config in enumerate(process_configs, 1):
        print(f"\n[Process {process_id}] Running benchmark {i}/{process_total_configs} "
              f"(global {start_idx + i}/{total_configs})...")

        # Create a copy to avoid modifying the original config
        config_copy = config.copy()

        # Extract batch_size and kv_len, generate kv_lens list
        batch_size = config_copy.pop("batch_size")
        kv_len = config_copy.pop("kv_len")
        kv_lens_list = [kv_len] * batch_size

        if barrier:
            barrier.wait()

        print(f"[Process {process_id}] Synchronized before running benchmark {i}")
        # Run benchmark with generated kv_lens
        results = benchmark_flash_attn_with_kvcache(
            kv_lens=kv_lens_list,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            process_id=process_id,
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
        num_processes=num_processes
    )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark flash_attn_with_kvcache performance",
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
        default="benchmark_flash_attn_results.csv",
        help="Output CSV filename. Default: benchmark_flash_attn_results.csv",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (useful for multi-process mode)",
    )
    args = parser.parse_args()

    print("="*60)
    print("Flash Attention with KV Cache Benchmark")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(f"Batch sizes: {args.batch_sizes if args.batch_sizes else 'default'}")
    print(f"KV lengths: {args.kv_lens if args.kv_lens else 'default'}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")
    print(f"Number of processes: {args.num_processes}")
    print(f"Output directory: {args.output_dir if args.output_dir else 'current directory'}")
    print("="*60)

    if args.num_processes > 1:
        # Create barrier and lock for multiprocessing
        barrier = Barrier(args.num_processes)
        file_lock = Lock()

        # Create and start processes
        processes = []
        for i in range(args.num_processes):
            p = mp.Process(
                target=run_benchmark_suite,
                args=(
                    args.batch_sizes,
                    args.kv_lens,
                    args.output_dir,
                    args.model_size,
                    args.warmup_iterations,
                    args.benchmark_iterations,
                    barrier,
                    file_lock,
                    i,
                    args.num_processes,
                    args.filename,
                )
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print(f"\nAll {args.num_processes} processes completed.")
        if args.output_dir:
            print(f"Results saved to: {os.path.join(args.output_dir, args.filename)}")
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
                base_name = os.path.splitext(args.filename)[0]
                ext = os.path.splitext(args.filename)[1]
                proc_filename = f"{base_name}_proc{i}{ext}"
                proc_path = os.path.join(args.output_dir, proc_filename) if args.output_dir else proc_filename
                if os.path.exists(proc_path):
                    df = pd.read_csv(proc_path)
                    all_dfs.append(df)
            
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                plot_results(combined_df, output_dir=args.output_dir)
    else:
        # Single process mode
        results = run_benchmark_suite(
            batch_sizes=args.batch_sizes,
            kv_lens=args.kv_lens,
            output_dir=args.output_dir,
            model_size=args.model_size,
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
            barrier=None,
            file_lock=None,
            process_id=0,
            num_processes=1,
            filename=args.filename,
        )
        
        # Generate plots if not skipped
        if not args.skip_plots:
            # Load the saved CSV and create plots
            data_path = os.path.join(args.output_dir, args.filename) if args.output_dir else args.filename
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                plot_results(df, output_dir=args.output_dir)
        
        print(f"\nResults saved to: {os.path.join(args.output_dir, args.filename) if args.output_dir else args.filename}")


if __name__ == "__main__":
    main()
