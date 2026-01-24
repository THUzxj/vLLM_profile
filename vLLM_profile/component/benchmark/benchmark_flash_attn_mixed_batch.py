# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for flash_attn_with_kvcache performance measurement with mixed KV lengths in a batch.
This script allows specifying different KV lengths and their counts in a single batch.
"""

import argparse
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

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

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def parse_kv_len_counts(kv_len_counts_str: list[str]) -> dict[int, int]:
    """
    Parse KV length and count pairs from command line arguments.
    
    Args:
        kv_len_counts_str: List of strings in format "kv_len:count" (e.g., ["1024:5", "2048:3"])
    
    Returns:
        Dictionary mapping kv_len to count (e.g., {1024: 5, 2048: 3})
    
    Example:
        parse_kv_len_counts(["1024:5", "2048:3", "4096:2"])
        -> {1024: 5, 2048: 3, 4096: 2}
    """
    kv_len_counts = {}
    for item in kv_len_counts_str:
        try:
            kv_len_str, count_str = item.split(":")
            kv_len = int(kv_len_str)
            count = int(count_str)
            if count <= 0:
                raise ValueError(f"Count must be positive, got {count}")
            if kv_len <= 0:
                raise ValueError(f"KV length must be positive, got {kv_len}")

            if kv_len in kv_len_counts:
                kv_len_counts[kv_len] += count
            else:
                kv_len_counts[kv_len] = count
        except ValueError as e:
            raise ValueError(
                f"Invalid format '{item}'. Expected format: 'kv_len:count' (e.g., '1024:5'). "
                f"Error: {e}"
            )
    return kv_len_counts


def build_kv_lens_from_counts(kv_len_counts: dict[int, int]) -> list[int]:
    """
    Build kv_lens list from kv_len_counts dictionary.
    
    Args:
        kv_len_counts: Dictionary mapping kv_len to count
    
    Returns:
        List of kv_lens (e.g., [1024, 1024, 1024, 2048, 2048, 4096, 4096])
    
    Example:
        build_kv_lens_from_counts({1024: 3, 2048: 2, 4096: 1})
        -> [1024, 1024, 1024, 2048, 2048, 4096]
    """
    kv_lens = []
    for kv_len, count in sorted(kv_len_counts.items()):
        kv_lens.extend([kv_len] * count)
    return kv_lens


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
) -> dict:
    """
    Benchmark flash_attn_with_kvcache function.

    Returns:
        dict: Benchmark results including mean, min, max times and throughput.
    """
    # Check if flash attention varlen function is available
    if not is_flash_attn_varlen_func_available():
        return {
            "error": "Flash attention varlen function is not available on this platform"
        }

    # Get FA version if not specified (kept for parity with benchmark_flash_attn_v2)
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
    current_platform.seed_everything(0)
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
    # For decode, each sequence has query_len=1
    query_len = 1
    num_tokens = num_seqs * query_len
    # Query shape: (num_tokens, num_query_heads, head_size)
    query = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)

    # KV cache shape: (num_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

    # cu_seqlens_q: cumulative sequence lengths for queries
    # Shape: (num_seqs + 1,)
    # For decode with query_len=1, this is simply [0, 1, 2, ..., num_seqs]
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
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
    kv_lens = config["kv_lens"]
    
    # Count occurrences of each kv_len
    kv_len_counts = {}
    for kv_len in kv_lens:
        kv_len_counts[kv_len] = kv_len_counts.get(kv_len, 0) + 1
    
    print("\n" + "="*80)
    print("Flash Attention with KV Cache Benchmark Results (Mixed Batch)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Num heads (Q, KV): {config['num_heads']}")
    print(f"  Head size: {config['head_size']}")
    print(f"  Dtype: {config['dtype']}")
    print(f"  Q dtype: {config['q_dtype']}")
    print(f"  Block size: {config['block_size']}")
    print(f"  Num blocks: {config['num_blocks']}")
    print(f"  Batch size: {len(kv_lens)}")
    print(f"  KV length distribution:")
    for kv_len, count in sorted(kv_len_counts.items()):
        print(f"    {kv_len}: {count} request(s)")
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
    filename: str = "benchmark_flash_attn_mixed_batch_results.csv",
    output_dir: Optional[str] = None,
):
    """Save benchmark results to CSV file."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)

    rows = []
    for result in results:
        if "error" in result:
            continue
        config = result["config"]
        kv_lens = config["kv_lens"]
        
        # Count occurrences of each kv_len
        kv_len_counts = {}
        for kv_len in kv_lens:
            kv_len_counts[kv_len] = kv_len_counts.get(kv_len, 0) + 1
        
        # Create a summary string for kv_len distribution
        kv_len_dist = ", ".join([f"{k}:{v}" for k, v in sorted(kv_len_counts.items())])
        
        row = {
            "batch_size": len(kv_lens),
            "kv_len_distribution": kv_len_dist,
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
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    return df


def run_single_benchmark(
    kv_len_counts: dict[int, int],
    model_size: str,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    sliding_window: Optional[int],
    fa_version: int,
    q_dtype: Optional[torch.dtype],
    warmup_iterations: int,
    benchmark_iterations: int,
) -> dict:
    """
    Run a single benchmark with mixed KV lengths in a batch.

    Args:
        kv_len_counts: Dictionary mapping kv_len to count (e.g., {1024: 5, 2048: 3})
        model_size: Model size to use ("0.6B", "4B", or "32B").
        dtype: Data type for tensors.
        block_size: Block size for KV cache.
        soft_cap: Soft cap value.
        num_blocks: Number of blocks.
        sliding_window: Sliding window size.
        fa_version: Flash attention version.
        q_dtype: Quantization dtype.
        warmup_iterations: Number of warmup iterations.
        benchmark_iterations: Number of benchmark iterations.

    Returns:
        Benchmark results dictionary.
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

    # Build kv_lens list from kv_len_counts
    kv_lens = build_kv_lens_from_counts(kv_len_counts)
    batch_size = len(kv_lens)

    # Run benchmark
    results = benchmark_flash_attn_with_kvcache(
        kv_lens=kv_lens,
        num_heads=num_heads,
        head_size=head_size,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        sliding_window=sliding_window,
        fa_version=fa_version,
        q_dtype=q_dtype,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
    )

    return results


def run_benchmark_suite(
    batch_configs: list[dict[int, int]],
    output_dir: Optional[str] = None,
    model_size: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    soft_cap: Optional[float] = None,
    num_blocks: int = 8192,
    sliding_window: Optional[int] = None,
    # Default to FA v2 to match benchmark_flash_attn_v2.py behavior
    fa_version: int = 2,
    q_dtype: Optional[torch.dtype] = None,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
):
    """
    Run benchmark suite with multiple batch configurations.

    Args:
        batch_configs: List of dictionaries, each mapping kv_len to count (e.g., [{1024: 5, 2048: 3}, {1024: 10, 2048: 5}])
        output_dir: Directory to save output files. If None, saves to current directory.
        model_size: Model size to use ("0.6B", "4B", or "32B"). Defaults to "4B".
        dtype: Data type for tensors. Defaults to torch.bfloat16.
        block_size: Block size for KV cache. Defaults to 16.
        soft_cap: Soft cap value. Defaults to None.
        num_blocks: Number of blocks. Defaults to 8192.
        sliding_window: Sliding window size. Defaults to None.
        fa_version: Flash attention version. Defaults to 3.
        q_dtype: Quantization dtype. Defaults to None.
        warmup_iterations: Number of warmup iterations. Defaults to WARMUP_ITERATIONS.
        benchmark_iterations: Number of benchmark iterations. Defaults to BENCHMARK_ITERATIONS.

    Returns:
        List of benchmark results.
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

    print("Starting Flash Attention with KV Cache Benchmark Suite (Mixed Batch)...")
    print(f"Model: Qwen3-{model_size}")
    print(f"  Num heads (Q, KV): {num_heads}")
    print(f"  Head size: {head_size}")
    print(f"  Number of batch configurations: {len(batch_configs)}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Benchmark iterations: {benchmark_iterations}\n")

    torch.set_default_device("cuda")

    all_results = []
    total_configs = len(batch_configs)
    
    for i, kv_len_counts in enumerate(batch_configs, 1):
        batch_size = sum(kv_len_counts.values())
        print(f"\n{'='*80}")
        print(f"Running benchmark {i}/{total_configs}...")
        print(f"Batch size: {batch_size}")
        print(f"KV length distribution:")
        for kv_len, count in sorted(kv_len_counts.items()):
            print(f"  {kv_len}: {count} request(s)")

        # Run benchmark
        results = run_single_benchmark(
            kv_len_counts=kv_len_counts,
            model_size=model_size,
            dtype=dtype,
            block_size=block_size,
            soft_cap=soft_cap,
            num_blocks=num_blocks,
            sliding_window=sliding_window,
            fa_version=fa_version,
            q_dtype=q_dtype,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )

        print_benchmark_results(results)
        all_results.append(results)

    # Save all results to CSV
    df = save_results_to_csv(all_results, output_dir=output_dir)
    
    print(f"\n{'='*80}")
    print(f"Completed {total_configs} benchmark(s). All results saved to CSV.")
    print(f"{'='*80}\n")

    return all_results


def run_benchmark(
    kv_len_counts: dict[int, int],
    output_dir: Optional[str] = None,
    model_size: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    soft_cap: Optional[float] = None,
    num_blocks: int = 8192,
    sliding_window: Optional[int] = None,
    fa_version: int = 3,
    q_dtype: Optional[torch.dtype] = None,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
):
    """
    Run benchmark with mixed KV lengths in a single batch (backward compatibility).

    Args:
        kv_len_counts: Dictionary mapping kv_len to count (e.g., {1024: 5, 2048: 3})
        output_dir: Directory to save output files. If None, saves to current directory.
        model_size: Model size to use ("0.6B", "4B", or "32B"). Defaults to "4B".
        dtype: Data type for tensors. Defaults to torch.bfloat16.
        block_size: Block size for KV cache. Defaults to 16.
        soft_cap: Soft cap value. Defaults to None.
        num_blocks: Number of blocks. Defaults to 8192.
        sliding_window: Sliding window size. Defaults to None.
        fa_version: Flash attention version. Defaults to 3.
        q_dtype: Quantization dtype. Defaults to None.
        warmup_iterations: Number of warmup iterations. Defaults to WARMUP_ITERATIONS.
        benchmark_iterations: Number of benchmark iterations. Defaults to BENCHMARK_ITERATIONS.
    """
    return run_benchmark_suite(
        batch_configs=[kv_len_counts],
        output_dir=output_dir,
        model_size=model_size,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        sliding_window=sliding_window,
        fa_version=fa_version,
        q_dtype=q_dtype,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark flash_attn_with_kvcache performance with mixed KV lengths in a batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single batch with 5 requests of kv_len=1024, 3 requests of kv_len=2048, and 2 requests of kv_len=4096
  python benchmark_flash_attn_mixed_batch.py --kv-len-counts 1024:5 2048:3 4096:2

  # Run multiple batch configurations in one run (all results saved to one CSV)
  python benchmark_flash_attn_mixed_batch.py --batch-configs "1024:5 2048:3" "1024:10 2048:5 4096:2" "1024:8 2048:4"

  # Run with custom model size and output directory
  python benchmark_flash_attn_mixed_batch.py --batch-configs "1024:10 2048:5" "1024:20 2048:10" --model-size 32B --output-dir ./results

  # Run with custom iterations
  python benchmark_flash_attn_mixed_batch.py --batch-configs "1024:8 2048:4" "1024:16 2048:8" --warmup-iterations 20 --benchmark-iterations 200
        """
    )
    parser.add_argument(
        "--kv-len-counts",
        type=str,
        nargs="+",
        default=None,
        help="KV length and count pairs in format 'kv_len:count' (e.g., '1024:5 2048:3 4096:2'). "
        "This specifies how many requests of each KV length to include in a single batch. "
        "Use --batch-configs for multiple batch configurations.",
    )
    parser.add_argument(
        "--batch-configs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple batch configurations, each as a quoted string with space-separated 'kv_len:count' pairs. "
        "Each configuration will be tested separately and all results saved to one CSV file. "
        "Example: --batch-configs \"1024:5 2048:3\" \"1024:10 2048:5 4096:2\"",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (CSV). "
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
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for tensors. Options: bfloat16, float16. Default: bfloat16",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size for KV cache. Default: 16",
    )
    parser.add_argument(
        "--soft-cap",
        type=float,
        default=None,
        help="Soft cap value. Default: None",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=8192,
        help="Number of blocks. Default: 8192",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Sliding window size. Default: None",
    )
    parser.add_argument(
        "--fa-version",
        type=int,
        # Default to 2 to match benchmark_flash_attn_v2.py
        default=2,
        choices=[2, 3],
        help="Flash attention version. Options: 2, 3. Default: 2",
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
    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    # Determine which mode to use
    if args.batch_configs is not None:
        # Multiple batch configurations mode
        batch_configs = []
        for config_str in args.batch_configs:
            # Split the config string by spaces to get individual kv_len:count pairs
            kv_len_count_pairs = config_str.split()
            kv_len_counts = parse_kv_len_counts(kv_len_count_pairs)
            batch_configs.append(kv_len_counts)
        
        # Run benchmark suite
        results = run_benchmark_suite(
            batch_configs=batch_configs,
            output_dir=args.output_dir,
            model_size=args.model_size,
            dtype=dtype,
            block_size=args.block_size,
            soft_cap=args.soft_cap,
            num_blocks=args.num_blocks,
            sliding_window=args.sliding_window,
            fa_version=args.fa_version,
            q_dtype=None,  # Can be extended to support quantization if needed
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
    elif args.kv_len_counts is not None:
        # Single batch configuration mode (backward compatibility)
        kv_len_counts = parse_kv_len_counts(args.kv_len_counts)
        results = run_benchmark(
            kv_len_counts=kv_len_counts,
            output_dir=args.output_dir,
            model_size=args.model_size,
            dtype=dtype,
            block_size=args.block_size,
            soft_cap=args.soft_cap,
            num_blocks=args.num_blocks,
            sliding_window=args.sliding_window,
            fa_version=args.fa_version,
            q_dtype=None,  # Can be extended to support quantization if needed
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
    else:
        parser.error("Either --kv-len-counts or --batch-configs must be provided.")

