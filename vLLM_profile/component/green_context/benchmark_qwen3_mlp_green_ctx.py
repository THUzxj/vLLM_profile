# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for Qwen3MLP performance measurement with green context and SM allocation.
Supports both single process and multiprocess modes.
"""

import argparse
import os
import tempfile
import time
from typing import Optional, List, Tuple
import multiprocessing as mp
from multiprocessing import Barrier, Lock

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from vllm.distributed.parallel_state import (cleanup_dist_env_and_memory,
                                             init_distributed_environment,
                                             initialize_model_parallel,
                                             model_parallel_is_initialized)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform

# Import green context functions from test_green_context
from test_green_context import (
    split_device_green_ctx,
    get_cudevice,
    get_device_resource,
    split_resource,
    checkCudaErrors,
)
import cuda.bindings.driver as driver

# Qwen3 model configurations
QWEN3_MODEL_CONFIGS = {
    "0.6B": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
    },
    "4B": {
        "hidden_size": 2560,
        "intermediate_size": 9728,
    },
    "14B": {
        "hidden_size": 5120,
        "intermediate_size": 17408,
    },
    "32B": {
        "hidden_size": 5120,
        "intermediate_size": 25600,
    },
}

# Default to Qwen3-4B for backward compatibility
DEFAULT_MODEL = "4B"

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Global flag to track if distributed environment is initialized
_dist_env_initialized = False
_dist_init_temp_file = None


def ensure_distributed_initialized(process_id: int = 0):
    """Ensure distributed environment and model parallel groups are initialized."""
    global _dist_env_initialized, _dist_init_temp_file

    if _dist_env_initialized:
        return

    if not torch.distributed.is_initialized():
        # Create a temporary file for distributed initialization
        # Use process_id to create unique temp files for each process
        _dist_init_temp_file = tempfile.mkstemp(suffix=f"_proc{process_id}")[1]

        backend = "nccl"
        if current_platform.is_cpu() or current_platform.is_tpu():
            backend = "gloo"

        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{_dist_init_temp_file}",
            local_rank=0,
            backend=backend,
        )

    # Initialize model parallel groups (tensor_parallel_size=1, pipeline_parallel_size=1)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    _dist_env_initialized = True


def cleanup_distributed():
    """Clean up distributed environment if it was initialized by this script."""
    global _dist_env_initialized, _dist_init_temp_file

    if _dist_env_initialized:
        cleanup_dist_env_and_memory(shutdown_ray=False)
        if _dist_init_temp_file and os.path.exists(_dist_init_temp_file):
            try:
                os.unlink(_dist_init_temp_file)
            except Exception:
                pass
        _dist_env_initialized = False
        _dist_init_temp_file = None


def create_single_green_ctx_stream(
    dev: torch.device,
    num_groups: int,
    min_sm_count: int,
    stream_id: int,
) -> Tuple[torch.Stream, int]:
    """
    Create a single green context stream for the specified stream_id.
    This is more efficient than creating all streams when only one is needed.
    
    Note: CUDA streams cannot be shared across processes, so each process must
    create its own stream. However, this function only creates the specific
    stream needed, rather than creating all streams and selecting one.
    
    Args:
        dev: The device to use
        num_groups: Number of green context groups
        min_sm_count: Minimum number of SMs per group
        stream_id: Which stream to create (0 to num_groups)
    
    Returns:
        Tuple of (stream, sm_count)
    """
    cu_dev = get_cudevice(dev)
    resource = get_device_resource(cu_dev)
    results, remaining = split_resource(resource, num_groups, min_sm_count)
    resources = results + [remaining]
    
    if stream_id >= len(resources):
        raise ValueError(
            f"stream_id {stream_id} exceeds available streams ({len(resources)})"
        )
    
    # Only create the stream we need
    selected_resource = resources[stream_id]
    sm_count = selected_resource.sm.smCount
    
    # Create green context for this specific resource
    desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([selected_resource], 1))
    green_ctx = checkCudaErrors(
        driver.cuGreenCtxCreate(
            desc, cu_dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        )
    )
    stream = checkCudaErrors(
        driver.cuGreenCtxStreamCreate(
            green_ctx,
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            0,  # priority
        )
    )
    
    torch_stream = torch.cuda.get_stream_from_external(stream)
    return torch_stream, sm_count


class Qwen3MLP(nn.Module):
    """Qwen3MLP module for benchmarking."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,  # Disable tensor parallelism for benchmarking
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,  # Disable tensor parallelism for benchmarking
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def benchmark_qwen3_mlp_with_green_ctx(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    device: str = "cuda:0",
    process_id: int = 0,
    num_groups: int = 1,
    min_sm_count: int = 16,
    stream_id: int = 0,
    sm_count: Optional[int] = None,
) -> dict:
    """
    Benchmark Qwen3MLP function with green context and specified SM allocation.

    Args:
        batch_size: Batch size for the input tensor
        seq_len: Sequence length for the input tensor
        hidden_size: Hidden size dimension
        intermediate_size: Intermediate size dimension
        dtype: Data type for tensors
        warmup_iterations: Number of warmup iterations
        benchmark_iterations: Number of benchmark iterations
        device: Device to run on (default: "cuda:0")
        process_id: Process ID for multi-process mode
        num_groups: Number of green context groups to create
        min_sm_count: Minimum number of SMs per group
        stream_id: Which stream to use (0 to num_groups-1)
        sm_count: Expected SM count for this stream (for validation). If None, will be retrieved from resource.

    Returns:
        dict: Benchmark results including mean, min, max times and throughput.
    """
    # Ensure distributed environment is initialized
    ensure_distributed_initialized(process_id=process_id)

    # Setup
    current_platform.seed_everything(42 + process_id)  # Different seed for each process
    torch.set_default_device(device)

    # Create green context stream (only the one we need, not all of them)
    dev = torch.device(device)
    try:
        # Only create the specific stream we need, not all streams
        selected_stream, actual_sm_count = create_single_green_ctx_stream(
            dev, num_groups, min_sm_count, stream_id
        )
        
        # Validate SM count if provided
        expected_sm_count = sm_count
        if expected_sm_count is not None and actual_sm_count != expected_sm_count:
            print(f"[Process {process_id}] WARNING: SM count mismatch. Expected {expected_sm_count}, got {actual_sm_count}")
        
        sm_count = actual_sm_count
        print(f"[Process {process_id}] Using stream {stream_id} with {sm_count} SMs")
    except Exception as e:
        print(f"[Process {process_id}] Failed to create green context: {e}")
        raise

    # Create MLP module
    mlp = Qwen3MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    ).to(device).to(dtype)

    # Create input tensor: [batch_size, seq_len, hidden_size]
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=dtype, device=device
    )

    # Warmup
    mlp.eval()
    with torch.cuda.stream(selected_stream):
        with torch.inference_mode():
            for _ in range(warmup_iterations):
                _ = mlp(input_tensor)

    # Synchronize before benchmarking
    selected_stream.synchronize()

    # Benchmark
    times = []
    with torch.cuda.stream(selected_stream):
        with torch.inference_mode():
            for _ in range(benchmark_iterations):
                start_time = time.perf_counter()

                _ = mlp(input_tensor)

                selected_stream.synchronize()
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
    total_tokens = batch_size * seq_len
    throughput = total_tokens / (mean_time / 1000)  # tokens per second

    # Calculate FLOPs (approximate)
    # gate_up_proj: batch_size * seq_len * hidden_size * intermediate_size * 2
    # act_fn: batch_size * seq_len * intermediate_size (negligible)
    # down_proj: batch_size * seq_len * intermediate_size * hidden_size
    flops = (
        batch_size * seq_len * hidden_size * intermediate_size * 2 +  # gate_up_proj
        batch_size * seq_len * intermediate_size * hidden_size  # down_proj
    )
    tflops = flops / (mean_time / 1000) / 1e12  # TFLOPs

    return {
        "mean_time_ms": mean_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "throughput_tokens_per_sec": throughput,
        "tflops": tflops,
        "total_tokens": total_tokens,
        "sm_count": sm_count,
        "stream_id": stream_id,
        "num_groups": num_groups,
        "min_sm_count": min_sm_count,
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "dtype": str(dtype),
            "device": device,
        }
    }


def print_benchmark_results(results: dict):
    """Print benchmark results in a formatted way."""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    config = results["config"]
    print("\n" + "="*80)
    print("Qwen3MLP Benchmark Results (with Green Context)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Intermediate size: {config['intermediate_size']}")
    print(f"  Dtype: {config['dtype']}")
    print(f"  Device: {config['device']}")
    print(f"  SM count: {results.get('sm_count', 'N/A')}")
    print(f"  Stream ID: {results.get('stream_id', 'N/A')}")
    print(f"  Number of groups: {results.get('num_groups', 'N/A')}")
    print(f"  Min SM count: {results.get('min_sm_count', 'N/A')}")
    print(f"\nPerformance:")
    print(f"  Mean time: {results['mean_time_ms']:.4f} ms")
    print(f"  Min time:  {results['min_time_ms']:.4f} ms")
    print(f"  Max time:  {results['max_time_ms']:.4f} ms")
    print(f"  Std dev:   {results['std_time_ms']:.4f} ms")
    print(
        f"  Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  TFLOPs: {results['tflops']:.4f}")
    print(f"  Total tokens: {results['total_tokens']}")
    print("="*80 + "\n")


def save_results_to_csv(
    results: list,
    filename: str = "benchmark_qwen3_mlp_green_ctx_results.csv",
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
            "batch_size": config["batch_size"],
            "seq_len": config["seq_len"],
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "dtype": config["dtype"],
            "mean_time_ms": result["mean_time_ms"],
            "min_time_ms": result["min_time_ms"],
            "max_time_ms": result["max_time_ms"],
            "std_time_ms": result["std_time_ms"],
            "throughput_tokens_per_sec": result["throughput_tokens_per_sec"],
            "tflops": result["tflops"],
            "total_tokens": result["total_tokens"],
            "sm_count": result.get("sm_count", None),
            "stream_id": result.get("stream_id", None),
            "num_groups": result.get("num_groups", None),
            "min_sm_count": result.get("min_sm_count", None),
            "process_id": process_id,
        }
        rows.append(row)

    if not rows:
        print(f"[Process {process_id}] No results to save (all results had errors)")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Check if file exists and has content
    file_exists = os.path.exists(data_path)
    file_has_content = False
    if file_exists:
        try:
            # Check if file has content (more than just whitespace)
            with open(data_path, 'r') as f:
                content = f.read().strip()
                file_has_content = len(content) > 0
        except Exception:
            file_has_content = False
    
    # Write with header if file doesn't exist or is empty
    if not file_exists or not file_has_content:
        if file_lock:
            with file_lock:
                # Double check after acquiring lock
                file_exists_check = os.path.exists(data_path)
                file_has_content_check = False
                if file_exists_check:
                    try:
                        with open(data_path, 'r') as f:
                            content = f.read().strip()
                            file_has_content_check = len(content) > 0
                    except Exception:
                        file_has_content_check = False
                
                if not file_exists_check or not file_has_content_check:
                    df.to_csv(data_path, index=False)
                else:
                    # File was created by another process, append without header
                    df.to_csv(data_path, mode='a', header=False, index=False)
        else:
            df.to_csv(data_path, index=False)
    else:
        # Append to existing file without header
        if file_lock:
            with file_lock:
                df.to_csv(data_path, mode='a', header=False, index=False)
        else:
            df.to_csv(data_path, mode='a', header=False, index=False)
    
    print(f"[Process {process_id}] Results saved to {data_path}")
    return df


def plot_results(
    df: pd.DataFrame,
    output_prefix: str = "benchmark_qwen3_mlp_green_ctx",
    output_dir: Optional[str] = None,
):
    """Create plots from benchmark results."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter out rows with missing data
    df = df.dropna(subset=["batch_size", "seq_len", "mean_time_ms"])
    df = df.astype({"batch_size": int, "seq_len": int})

    # Normalize mean_time to start from 0
    df["mean_time_normalized"] = df["mean_time_ms"] - df["mean_time_ms"].min()

    # Plot 1: seq_len vs mean_time, grouped by batch_size and SM count
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        for sm_count in sorted(df["sm_count"].dropna().unique()):
            subset = df[(df["batch_size"] == batch_size) & (df["sm_count"] == sm_count)]
            if len(subset) > 0:
                subset = subset.sort_values("seq_len")
                plt.plot(subset["seq_len"], subset["mean_time_normalized"],
                         marker="o", label=f"batch={batch_size}, SM={sm_count}")
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Qwen3MLP with Green Context: Sequence Length vs Mean Time", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_seq_len_vs_time.png") if output_dir else f"{output_prefix}_seq_len_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 2: SM count vs mean_time, grouped by batch_size
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size]
        subset = subset.sort_values("sm_count")
        plt.plot(subset["sm_count"], subset["mean_time_normalized"],
                 marker="o", label=f"batch_size={batch_size}")
    plt.xlabel("SM Count", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Qwen3MLP with Green Context: SM Count vs Mean Time", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_sm_count_vs_time.png") if output_dir else f"{output_prefix}_sm_count_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 3: batch_size vs mean_time, grouped by SM count
    plt.figure(figsize=(12, 8))
    for sm_count in sorted(df["sm_count"].dropna().unique()):
        subset = df[df["sm_count"] == sm_count]
        subset = subset.sort_values("batch_size")
        plt.plot(subset["batch_size"], subset["mean_time_normalized"],
                 marker="o", label=f"SM_count={sm_count}")
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Qwen3MLP with Green Context: Batch Size vs Mean Time (by SM Count)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_batch_size_vs_time.png") if output_dir else f"{output_prefix}_batch_size_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()


def run_benchmark_suite(
    batch_sizes: Optional[list[int]] = None,
    seq_lens: Optional[list[int]] = None,
    output_dir: Optional[str] = None,
    model_size: str = DEFAULT_MODEL,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    barrier: Optional[Barrier] = None,
    file_lock: Optional[Lock] = None,
    process_id: int = 0,
    num_processes: int = 1,
    filename: str = "benchmark_qwen3_mlp_green_ctx_results.csv",
    num_groups: int = 1,
    min_sm_count: int = 16,
    stream_id: Optional[int] = None,
    sm_count: Optional[int] = None,
):
    """Run a comprehensive benchmark suite with green context and SM allocation.

    Args:
        batch_sizes: List of batch sizes to test. If None, uses default values.
        seq_lens: List of sequence lengths to test. If None, uses default values.
        output_dir: Directory to save output files. If None, saves to current directory.
        model_size: Model size to use ("0.6B", "4B", "14B", or "32B"). Defaults to "4B".
        warmup_iterations: Number of warmup iterations.
        benchmark_iterations: Number of benchmark iterations.
        barrier: Optional multiprocessing barrier for synchronization.
        file_lock: Optional multiprocessing lock for file writing.
        process_id: Process ID for multi-process mode.
        num_processes: Total number of processes.
        filename: Output CSV filename.
        num_groups: Number of green context groups to create.
        min_sm_count: Minimum number of SMs per group.
        stream_id: Which stream to use. If None, uses process_id % num_groups.
        sm_count: Expected SM count for this stream (for validation). If None, will be retrieved from resource.
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

    # Determine which stream to use for this process
    if stream_id is None:
        stream_id = process_id % num_groups

    print(f"[Process {process_id}] Starting Qwen3MLP Benchmark Suite with Green Context...")
    print(f"[Process {process_id}] Model: Qwen3-{model_size}")
    print(f"[Process {process_id}]   Hidden size: {model_config['hidden_size']}")
    print(f"[Process {process_id}]   Intermediate size: {model_config['intermediate_size']}")
    print(f"[Process {process_id}] Warmup iterations: {warmup_iterations}")
    print(f"[Process {process_id}] Benchmark iterations: {benchmark_iterations}")
    print(f"[Process {process_id}] Number of processes: {num_processes}")
    print(f"[Process {process_id}] Green context groups: {num_groups}")
    print(f"[Process {process_id}] Min SM count per group: {min_sm_count}")
    print(f"[Process {process_id}] Using stream ID: {stream_id}\n")

    # Use the selected model configuration
    model_configs = [model_config]

    # Test different combinations of batch_size and seq_len
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if seq_lens is None:
        seq_lens = [1,]

    # Different input size configurations
    # Format: (batch_size, seq_len)
    input_configs = [
        (batch_size, seq_len)
        for batch_size in batch_sizes
        for seq_len in seq_lens
    ]

    dtype = torch.bfloat16

    # All processes execute the same configurations synchronously
    total_configs = len(input_configs)

    print(f"[Process {process_id}] Processing {total_configs} configurations "
          f"(all processes execute the same configurations synchronously)")

    if barrier:
        barrier.wait()  # Synchronize before starting benchmarks

    print(f"[Process {process_id}] Synchronized before starting benchmarks")
    all_results = []
    for model_idx, model_config in enumerate(model_configs, 1):
        print(f"\n[Process {process_id}] {'='*80}")
        print(f"[Process {process_id}] Model Config {model_idx}/{len(model_configs)}: "
              f"hidden_size={model_config['hidden_size']}, "
              f"intermediate_size={model_config['intermediate_size']}")
        print(f"[Process {process_id}] {'='*80}\n")

        for input_idx, (batch_size, seq_len) in enumerate(input_configs, 1):
            # Synchronize all processes before executing this configuration
            if barrier:
                barrier.wait()

            print(f"[Process {process_id}] Running benchmark {input_idx}/{total_configs}: "
                  f"batch_size={batch_size}, seq_len={seq_len}...")

            try:
                results = benchmark_qwen3_mlp_with_green_ctx(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=model_config["hidden_size"],
                    intermediate_size=model_config["intermediate_size"],
                    dtype=dtype,
                    warmup_iterations=warmup_iterations,
                    benchmark_iterations=benchmark_iterations,
                    process_id=process_id,
                    num_groups=num_groups,
                    min_sm_count=min_sm_count,
                    stream_id=stream_id,
                    sm_count=sm_count,
                )
                all_results.append(results)
                print_benchmark_results(results)
            except Exception as e:
                print(f"[Process {process_id}] ERROR: Failed to run benchmark: {e}\n")
                import traceback
                print(traceback.format_exc())
                all_results.append({"error": str(e), "config": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    **model_config
                }})

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

    # Clean up distributed environment
    cleanup_distributed()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3MLP performance with green context and SM allocation"
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
        "--seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="List of sequence lengths to test (e.g., --seq-lens 1 1024 2048). "
        "If not specified, uses default: [1]",
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
        default="benchmark_qwen3_mlp_green_ctx_results.csv",
        help="Output CSV filename. Default: benchmark_qwen3_mlp_green_ctx_results.csv",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (useful for multi-process mode)",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=1,
        help="Number of green context groups to create. Default: 1",
    )
    parser.add_argument(
        "--min-sm-count",
        type=int,
        default=16,
        help="Minimum number of SMs per group. Default: 16",
    )
    parser.add_argument(
        "--stream-id",
        type=int,
        default=None,
        help="Which stream to use (0 to num_groups-1). "
        "If not specified, uses process_id %% num_groups",
    )
    args = parser.parse_args()

    print("="*60)
    print("Qwen3MLP Benchmark with Green Context")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(f"Batch sizes: {args.batch_sizes if args.batch_sizes else 'default'}")
    print(f"Sequence lengths: {args.seq_lens if args.seq_lens else 'default'}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")
    print(f"Number of processes: {args.num_processes}")
    print(f"Green context groups: {args.num_groups}")
    print(f"Min SM count per group: {args.min_sm_count}")
    print(f"Stream ID: {args.stream_id if args.stream_id is not None else 'auto (process_id % num_groups)'}")
    print(f"Output directory: {args.output_dir if args.output_dir else 'current directory'}")
    print("="*60)

    # Pre-create green context in main process to validate configuration and get SM counts
    device = "cuda:0"
    dev = torch.device(device)
    stream_sm_counts = {}  # Map from stream_id to sm_count
    
    try:
        print(f"\nPre-creating green context in main process to validate configuration...")
        streams, resources = split_device_green_ctx(dev, args.num_groups, args.min_sm_count)
        print(f"Successfully created {len(streams)} streams with {args.num_groups} groups")
        for i, resource in enumerate(resources):
            sm_count = resource.sm.smCount
            stream_sm_counts[i] = sm_count
            print(f"  Stream {i}: {sm_count} SMs")
    except Exception as e:
        print(f"ERROR: Failed to create green context in main process: {e}")
        import traceback
        print(traceback.format_exc())
        raise

    if args.num_processes > 1:
        # Create barrier and lock for multiprocessing
        barrier = Barrier(args.num_processes)
        file_lock = Lock()

        # Create and start processes
        processes = []
        for i in range(args.num_processes):
            # Determine stream_id for this process
            stream_id = args.stream_id if args.stream_id is not None else (i % args.num_groups)
            
            # Get the expected SM count for this stream
            sm_count = stream_sm_counts.get(stream_id)
            if sm_count is None:
                print(f"WARNING: No SM count found for stream_id {stream_id}, using None")
            
            p = mp.Process(
                target=run_benchmark_suite,
                args=(
                    args.batch_sizes,
                    args.seq_lens,
                    args.output_dir,
                    args.model_size,
                    args.warmup_iterations,
                    args.benchmark_iterations,
                    barrier,
                    file_lock,
                    i,
                    args.num_processes,
                    args.filename,
                    args.num_groups,
                    args.min_sm_count,
                    stream_id,
                    sm_count,
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
        stream_id = args.stream_id if args.stream_id is not None else 0
        sm_count = stream_sm_counts.get(stream_id)
        
        results = run_benchmark_suite(
            batch_sizes=args.batch_sizes,
            seq_lens=args.seq_lens,
            output_dir=args.output_dir,
            model_size=args.model_size,
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
            barrier=None,
            file_lock=None,
            process_id=0,
            num_processes=1,
            filename=args.filename,
            num_groups=args.num_groups,
            min_sm_count=args.min_sm_count,
            stream_id=stream_id,
            sm_count=sm_count,
        )
        
        # Generate plots if not skipped
        if not args.skip_plots:
            # Load the saved CSV and create plots
            data_path = os.path.join(args.output_dir, args.filename) if args.output_dir else args.filename
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                plot_results(df, output_dir=args.output_dir)
        
        print(f"\nResults saved to: {os.path.join(args.output_dir, args.filename) if args.output_dir else args.filename}")

