# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for Attention layer decode performance measurement.
"""

import argparse
import csv
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
from vllm.attention.layer import Attention

# Qwen3 model configurations
QWEN3_MODEL_CONFIGS = {
    "0.6B": {
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_size": 128,
        "hidden_size": 2048,
    },
    "4B": {
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "hidden_size": 4096,
    },
    "32B": {
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_size": 128,
        "hidden_size": 8192,
    },
}

# Default to Qwen3-4B for backward compatibility
DEFAULT_MODEL = "4B"
NUM_HEADS = QWEN3_MODEL_CONFIGS[DEFAULT_MODEL]["num_heads"]
NUM_KV_HEADS = QWEN3_MODEL_CONFIGS[DEFAULT_MODEL]["num_kv_heads"]
HEAD_SIZE = QWEN3_MODEL_CONFIGS[DEFAULT_MODEL]["head_size"]
HIDDEN_SIZE = QWEN3_MODEL_CONFIGS[DEFAULT_MODEL]["hidden_size"]

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def create_decode_metadata(
    batch_size: int,
    kv_lens: list[int],
    block_size: int,
    num_blocks: int,
    device: torch.device,
    vllm_config: VllmConfig,
    layer_name: str,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> FlashAttentionMetadata:
    """Create FlashAttentionMetadata for decode phase using the builder."""
    max_kv_len = max(kv_lens)
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    # Create block tables: [batch_size, max_blocks_per_seq]
    # Assign blocks sequentially to each sequence, ensuring we don't exceed num_blocks
    block_tables = []
    current_block = 0
    for i in range(batch_size):
        kv_len = kv_lens[i]
        num_blocks_needed = (kv_len + block_size - 1) // block_size
        # Assign sequential blocks, wrapping around if needed
        block_table = []
        for j in range(num_blocks_needed):
            block_table.append(current_block % num_blocks)
            current_block += 1
        # Pad to max_blocks_per_seq
        block_table += [0] * (max_blocks_per_seq - len(block_table))
        block_tables.append(block_table)

    block_table_tensor = torch.tensor(
        block_tables, dtype=torch.int32, device=device
    )

    # For decode phase, query_len is 1 for each sequence
    query_lens = [1] * batch_size
    seq_lens = kv_lens  # Current sequence lengths (including new token)

    # Create seq_lens_tensor
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens_tensor.cpu()

    # Create query_start_loc: cumulative sum of query lengths
    query_start_loc = [0]
    for q_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + q_len)
    query_start_loc_tensor = torch.tensor(
        query_start_loc, dtype=torch.int32, device=device
    )
    query_start_loc_cpu = query_start_loc_tensor.cpu()

    # Create slot_mapping for decode (not used with block tables, but required)
    # For decode with block tables, we use a dummy slot_mapping
    num_actual_tokens = batch_size  # Each decode token is 1
    slot_mapping = torch.zeros(
        num_actual_tokens, dtype=torch.int64, device=device
    )

    # num_computed_tokens_cpu: CPU tensor with number of computed tokens
    # For decode phase, this is batch_size (each sequence processes 1 token)
    num_computed_tokens_cpu = torch.tensor(num_actual_tokens, dtype=torch.int32, device="cpu")

    # Create CommonAttentionMetadata
    common_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_tensor,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_tensor,
        seq_lens_cpu=seq_lens_cpu,
        num_reqs=batch_size,
        num_actual_tokens=num_actual_tokens,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        max_query_len=1,
        max_seq_len=max_kv_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    # Get KV cache spec - use the dtype from cache config
    kv_cache_dtype = vllm_config.cache_config.cache_dtype
    if kv_cache_dtype == "auto":
        kv_cache_torch_dtype = dtype
    else:
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
        kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )

    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=kv_cache_torch_dtype,
    )

    # Create FlashAttentionMetadataBuilder and build metadata
    builder = FlashAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=device,
    )

    # Build the metadata (common_prefix_len=0 for decode)
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
        fast_build=False,
    )

    return metadata


def create_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create KV cache tensor.

    Flash Attention backend expects shape: (2, num_blocks, block_size, num_kv_heads, head_size)
    where the first dimension 2 represents key and value caches.
    """
    # Verify block_size is divisible by 16 (required by Flash Attention)
    if block_size % 16 != 0:
        raise ValueError(
            f"Block size {block_size} must be divisible by 16 for Flash Attention"
        )

    # KV cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
    # First dimension: 0 = key cache, 1 = value cache
    kv_cache = torch.randn(
        2,  # key and value
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    return kv_cache


def benchmark_attention_layer(
    batch_size: int,
    kv_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
) -> dict:
    """
    Benchmark Attention layer forward pass in decode phase.

    Returns:
        dict: Benchmark results including mean, min, max times and throughput.
    """
    device = torch.device("cuda")
    torch.set_default_device(device)
    current_platform.seed_everything(0)

    # Create CacheConfig
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        sliding_window=None,
    )

    # Create VllmConfig for compilation_config
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    # Create a minimal VllmConfig first (needed before creating Attention layer)
    compilation_config = CompilationConfig()
    model_config = ModelConfig(
        model="/nfs/xjzhang/Qwen/Qwen3-4B",
        trust_remote_code=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        cache_config=cache_config,
    )

    # Set as current vllm config (needed for get_current_vllm_config)
    set_current_vllm_config(vllm_config)

    # Create Attention layer with Flash Attention 2 backend
    # Note: Attention.__init__ will register itself in compilation_config.static_forward_context
    scale = head_size ** -0.5
    attention = Attention(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        cache_config=cache_config,
        prefix="test_attn",
        attn_backend=FlashAttentionBackend,  # Force Flash Attention 2
    )
    attention = attention.to(device)
    attention.eval()

    # Verify backend
    print(f"Using attention backend: {attention.attn_backend.get_name()}")

    # Create KV cache
    kv_cache = create_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
    )

    # Bind KV cache to attention layer
    attention.kv_cache[0] = kv_cache

    # Create kv_lens (all sequences have the same kv_len)
    kv_lens = [kv_len] * batch_size

    # Create decode metadata
    attn_metadata = create_decode_metadata(
        batch_size=batch_size,
        kv_lens=kv_lens,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        vllm_config=vllm_config,
        layer_name="test_attn",
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )

    # Create query tensor for decode: [batch_size, num_heads, head_size]
    query = torch.randn(
        batch_size, num_heads, head_size, dtype=dtype, device=device
    )

    # For decode phase, key and value are None (using KV cache)
    key = None
    value = None

    # attn_metadata should be a dict mapping layer_name to metadata
    attn_metadata_dict = {"test_attn": attn_metadata}

    # Register the attention layer in no_compile_layers for the forward context
    # The layer is already registered in compilation_config.static_forward_context,
    # but we need to also register it in the forward context's no_compile_layers
    no_compile_layers = {"test_attn": attention}

    # Warmup
    with set_forward_context(
        attn_metadata=attn_metadata_dict,
        vllm_config=vllm_config,
        virtual_engine=0,
        no_compile_layers=no_compile_layers,
    ):
        for _ in range(warmup_iterations):
            _ = attention(query, key, value)

    # Synchronize before benchmarking
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with set_forward_context(
        attn_metadata=attn_metadata_dict,
        vllm_config=vllm_config,
        virtual_engine=0,
        no_compile_layers=no_compile_layers,
    ):
        for _ in range(benchmark_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = attention(query, key, value)

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            # Convert to milliseconds
            times.append((end_time - start_time) * 1000)

    # Calculate statistics
    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()

    # Calculate throughput (tokens per second)
    total_tokens = batch_size  # Each decode processes batch_size tokens
    throughput = total_tokens / (mean_time / 1000)  # tokens per second

    return {
        "mean_time_ms": mean_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "throughput_tokens_per_sec": throughput,
        "batch_size": batch_size,
        "kv_len": kv_len,
        "config": {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "dtype": str(dtype),
            "block_size": block_size,
            "num_blocks": num_blocks,
            "batch_size": batch_size,
            "kv_len": kv_len,
        }
    }


def print_benchmark_results(results: dict):
    """Print benchmark results in a formatted way."""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    config = results["config"]
    print("\n" + "="*80)
    print("Attention Layer Decode Benchmark Results")
    print("="*80)
    print(f"Configuration:")
    print(
        f"  Num heads (Q, KV): ({config['num_heads']}, {config['num_kv_heads']})")
    print(f"  Head size: {config['head_size']}")
    print(f"  Dtype: {config['dtype']}")
    print(f"  Block size: {config['block_size']}")
    print(f"  Num blocks: {config['num_blocks']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  KV length: {config['kv_len']}")
    print(f"\nPerformance:")
    print(f"  Mean time: {results['mean_time_ms']:.4f} ms")
    print(f"  Min time:  {results['min_time_ms']:.4f} ms")
    print(f"  Max time:  {results['max_time_ms']:.4f} ms")
    print(f"  Std dev:   {results['std_time_ms']:.4f} ms")
    print(
        f"  Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print("="*80 + "\n")


def save_results_to_csv(
    results: list,
    filename: str = "benchmark_attention_layer_results.csv",
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
        row = {
            "batch_size": config["batch_size"],
            "kv_len": config["kv_len"],
            "num_heads": config["num_heads"],
            "num_kv_heads": config["num_kv_heads"],
            "head_size": config["head_size"],
            "dtype": config["dtype"],
            "block_size": config["block_size"],
            "num_blocks": config["num_blocks"],
            "mean_time_ms": result["mean_time_ms"],
            "min_time_ms": result["min_time_ms"],
            "max_time_ms": result["max_time_ms"],
            "std_time_ms": result["std_time_ms"],
            "throughput_tokens_per_sec": result["throughput_tokens_per_sec"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    return df


def plot_results(
    df: pd.DataFrame,
    output_prefix: str = "benchmark_attention_layer",
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
        plt.plot(
            subset["kv_len"],
            subset["mean_time_normalized"],
            marker="o",
            label=f"batch_size={batch_size}",
        )
    plt.xlabel("KV Length", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title(
        "Attention Layer Decode: KV Length vs Mean Time (by Batch Size)",
        fontsize=14,
    )
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
        plt.plot(
            subset["batch_size"],
            subset["mean_time_normalized"],
            marker="o",
            label=f"kv_len={kv_len}",
        )
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title(
        "Attention Layer Decode: Batch Size vs Mean Time (by KV Length)",
        fontsize=14,
    )
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
        plt.plot(
            subset["kv_len_batch_size"],
            subset["mean_time_normalized"],
            marker="o",
            label=f"batch_size={batch_size}",
        )
    plt.xlabel("KV Length × Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title(
        "Attention Layer Decode: KV Length × Batch Size vs Mean Time (by Batch Size)",
        fontsize=14,
    )
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
):
    """Run a comprehensive benchmark suite.

    Args:
        batch_sizes: List of batch sizes to test. If None, uses default values.
        kv_lens: List of KV lengths to test. If None, uses default values.
        output_dir: Directory to save output files. If None, saves to current directory.
        model_size: Model size to use ("0.6B", "4B", or "32B"). Defaults to "4B".
    """
    # Get model configuration
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}"
        )
    model_config = QWEN3_MODEL_CONFIGS[model_size]
    num_heads = model_config["num_heads"]
    num_kv_heads = model_config["num_kv_heads"]
    head_size = model_config["head_size"]
    hidden_size = model_config["hidden_size"]

    print("Starting Attention Layer Decode Benchmark Suite...")
    print(f"Model: Qwen3-{model_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num KV heads: {num_kv_heads}")
    print(f"  Head size: {head_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}\n")

    torch.set_default_device("cuda")

    # Test different combinations of batch_size and kv_len
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if kv_lens is None:
        kv_lens = [1024, 2048, 4096, 8192]

    # Benchmark configurations
    test_configs = [
        {
            "batch_size": batch_size,
            "kv_len": kv_len,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "dtype": torch.bfloat16,
            "block_size": 16,
            "num_blocks": 8192,
        }
        for batch_size in batch_sizes
        for kv_len in kv_lens
    ]

    all_results = []
    total_configs = len(test_configs)
    for i, config in enumerate(test_configs, 1):
        print(f"\nRunning benchmark {i}/{total_configs}...")
        print(
            f"  Batch size: {config['batch_size']}, KV len: {config['kv_len']}")

        try:
            results = benchmark_attention_layer(**config)
            all_results.append(results)
            print_benchmark_results(results)
        except Exception as e:
            print(f"ERROR: Failed to run benchmark: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"error": str(e), "config": config})

    # Save results to CSV and create plots
    if all_results:
        df = save_results_to_csv(all_results, output_dir=output_dir)
        if not df.empty:
            plot_results(df, output_dir=output_dir)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Attention layer decode performance"
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
    args = parser.parse_args()

    # Run benchmark suite
    results = run_benchmark_suite(
        batch_sizes=args.batch_sizes,
        kv_lens=args.kv_lens,
        output_dir=args.output_dir,
        model_size=args.model_size,
    )

    # You can also run individual benchmarks
    # Example:
    # results = benchmark_attention_layer(
    #     batch_size=8,
    #     kv_len=2048,
    #     num_heads=NUM_HEADS,
    #     num_kv_heads=NUM_KV_HEADS,
    #     head_size=HEAD_SIZE,
    #     dtype=torch.bfloat16,
    #     block_size=16,
    #     num_blocks=8192,
    # )
    # print_benchmark_results(results)
