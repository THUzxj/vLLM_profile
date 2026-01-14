# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for Qwen3MLP performance measurement with different input sizes.
"""

import argparse
import os
import tempfile
import time
from typing import Optional
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


def benchmark_qwen3_mlp(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    device: str = "cuda",
    process_id: int = 0,
) -> dict:
    """
    Benchmark Qwen3MLP function with given input dimensions.

    Args:
        batch_size: Batch size for the input tensor
        seq_len: Sequence length for the input tensor
        hidden_size: Hidden size dimension
        intermediate_size: Intermediate size dimension
        dtype: Data type for tensors
        warmup_iterations: Number of warmup iterations
        benchmark_iterations: Number of benchmark iterations
        device: Device to run on (default: "cuda")

    Returns:
        dict: Benchmark results including mean, min, max times and throughput.
    """
    # Ensure distributed environment is initialized
    ensure_distributed_initialized(process_id=process_id)

    # Setup
    # Different seed for each process
    current_platform.seed_everything(42 + process_id)
    torch.set_default_device(device)

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
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = mlp(input_tensor)

    # Synchronize before benchmarking
    torch.cuda.synchronize()

    # Benchmark
    times = []  # Host-side timing using time.perf_counter
    cuda_event_times = []  # Device-side timing using CUDA events

    # CUDA events for more accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for _ in range(benchmark_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Record CUDA start event
            start_event.record()

            _ = mlp(input_tensor)

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            # Record CUDA end event and measure elapsed time (in ms)
            end_event.record()
            torch.cuda.synchronize()

            # Convert to milliseconds
            times.append((end_time - start_time) * 1000)
            cuda_event_times.append(start_event.elapsed_time(end_event))

    # Calculate statistics
    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()

    # CUDA event based statistics
    cuda_times_tensor = torch.tensor(cuda_event_times)
    mean_time_cuda = cuda_times_tensor.mean().item()
    min_time_cuda = cuda_times_tensor.min().item()
    max_time_cuda = cuda_times_tensor.max().item()
    std_time_cuda = cuda_times_tensor.std().item()

    # Calculate throughput (tokens per second)
    total_tokens = batch_size * seq_len
    throughput = total_tokens / (mean_time / 1000)  # tokens per second
    throughput_cuda = total_tokens / (mean_time_cuda / 1000)  # tokens per second using CUDA timing

    # Calculate FLOPs (approximate)
    # gate_up_proj: batch_size * seq_len * hidden_size * intermediate_size * 2
    # act_fn: batch_size * seq_len * intermediate_size (negligible)
    # down_proj: batch_size * seq_len * intermediate_size * hidden_size
    flops = (
        batch_size * seq_len * hidden_size * intermediate_size * 2 +  # gate_up_proj
        batch_size * seq_len * intermediate_size * hidden_size  # down_proj
    )
    tflops = flops / (mean_time / 1000) / 1e12  # TFLOPs
    tflops_cuda = flops / (mean_time_cuda / 1000) / 1e12  # TFLOPs using CUDA timing

    return {
        # Host-side timing results
        "mean_time_ms": mean_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "throughput_tokens_per_sec": throughput,
        "tflops": tflops,
        # CUDA event timing results
        "mean_time_ms_cuda": mean_time_cuda,
        "min_time_ms_cuda": min_time_cuda,
        "max_time_ms_cuda": max_time_cuda,
        "std_time_ms_cuda": std_time_cuda,
        "throughput_tokens_per_sec_cuda": throughput_cuda,
        "tflops_cuda": tflops_cuda,
        "total_tokens": total_tokens,
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
    print("Qwen3MLP Benchmark Results")
    print("="*80)
    print(f"Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Intermediate size: {config['intermediate_size']}")
    print(f"  Dtype: {config['dtype']}")
    print(f"  Device: {config['device']}")
    print(f"\nPerformance:")
    print(f"  Mean time (host): {results['mean_time_ms']:.4f} ms")
    print(f"  Min time  (host): {results['min_time_ms']:.4f} ms")
    print(f"  Max time  (host): {results['max_time_ms']:.4f} ms")
    print(f"  Std dev   (host): {results['std_time_ms']:.4f} ms")
    print(f"  Mean time (CUDA): {results['mean_time_ms_cuda']:.4f} ms")
    print(f"  Min time  (CUDA): {results['min_time_ms_cuda']:.4f} ms")
    print(f"  Max time  (CUDA): {results['max_time_ms_cuda']:.4f} ms")
    print(f"  Std dev   (CUDA): {results['std_time_ms_cuda']:.4f} ms")
    print(
        f"  Throughput (host): {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(
        f"  Throughput (CUDA): {results['throughput_tokens_per_sec_cuda']:.2f} tokens/sec")
    print(f"  TFLOPs (host): {results['tflops']:.4f}")
    print(f"  TFLOPs (CUDA): {results['tflops_cuda']:.4f}")
    print(f"  Total tokens: {results['total_tokens']}")
    print("="*80 + "\n")


def save_results_to_csv(
    results: list,
    filename: str = "benchmark_qwen3_mlp_results.csv",
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
        data_path = os.path.join(
            output_dir, filename) if output_dir else filename

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
            # Host-side timing columns
            "mean_time_ms": result["mean_time_ms"],
            "min_time_ms": result["min_time_ms"],
            "max_time_ms": result["max_time_ms"],
            "std_time_ms": result["std_time_ms"],
            "throughput_tokens_per_sec": result["throughput_tokens_per_sec"],
            "tflops": result["tflops"],
            # CUDA event timing columns
            "mean_time_ms_cuda": result["mean_time_ms_cuda"],
            "min_time_ms_cuda": result["min_time_ms_cuda"],
            "max_time_ms_cuda": result["max_time_ms_cuda"],
            "std_time_ms_cuda": result["std_time_ms_cuda"],
            "throughput_tokens_per_sec_cuda": result["throughput_tokens_per_sec_cuda"],
            "tflops_cuda": result["tflops_cuda"],
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
    output_prefix: str = "benchmark_qwen3_mlp",
    output_dir: Optional[str] = None,
):
    """Create three plots from benchmark results."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter out rows with missing data
    df = df.dropna(subset=["batch_size", "seq_len", "mean_time_ms"])
    df = df.astype({"batch_size": int, "seq_len": int})

    # Normalize mean_time to start from 0
    df["mean_time_normalized"] = df["mean_time_ms"] - df["mean_time_ms"].min()

    # Plot 1: seq_len vs mean_time, grouped by batch_size
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size]
        subset = subset.sort_values("seq_len")
        plt.plot(subset["seq_len"], subset["mean_time_normalized"],
                 marker="o", label=f"batch_size={batch_size}")
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Qwen3MLP: Sequence Length vs Mean Time (by Batch Size)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_seq_len_vs_time.png") if output_dir else f"{output_prefix}_seq_len_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 2: batch_size vs mean_time, grouped by seq_len
    plt.figure(figsize=(12, 8))
    for seq_len in sorted(df["seq_len"].unique()):
        subset = df[df["seq_len"] == seq_len]
        subset = subset.sort_values("batch_size")
        plt.plot(subset["batch_size"], subset["mean_time_normalized"],
                 marker="o", label=f"seq_len={seq_len}")
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title("Qwen3MLP: Batch Size vs Mean Time (by Sequence Length)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_batch_size_vs_time.png") if output_dir else f"{output_prefix}_batch_size_vs_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 3: seq_len*batch_size vs mean_time, grouped by batch_size
    df["seq_len_batch_size"] = df["seq_len"] * df["batch_size"]
    plt.figure(figsize=(12, 8))
    for batch_size in sorted(df["batch_size"].unique()):
        subset = df[df["batch_size"] == batch_size]
        subset = subset.sort_values("seq_len_batch_size")
        plt.plot(subset["seq_len_batch_size"], subset["mean_time_normalized"],
                 marker="o", label=f"batch_size={batch_size}")
    plt.xlabel("Sequence Length × Batch Size", fontsize=12)
    plt.ylabel("Mean Time (ms, normalized)", fontsize=12)
    plt.title(
        "Qwen3MLP: Sequence Length × Batch Size vs Mean Time (by Batch Size)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{output_prefix}_seq_len_batch_size_vs_time.png") if output_dir else f"{output_prefix}_seq_len_batch_size_vs_time.png"
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
    filename: str = "benchmark_qwen3_mlp_results.csv",
):
    """Run a comprehensive benchmark suite with different input sizes.

    Args:
        batch_sizes: List of batch sizes to test. If None, uses default values.
        seq_lens: List of sequence lengths to test. If None, uses default values.
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
    # Synchronize all processes at the start to ensure they begin together
    if barrier:
        barrier.wait()

    model_config = QWEN3_MODEL_CONFIGS[model_size]

    print(f"[Process {process_id}] Starting Qwen3MLP Benchmark Suite...")
    print(f"[Process {process_id}] Model: Qwen3-{model_size}")
    print(
        f"[Process {process_id}]   Hidden size: {model_config['hidden_size']}")
    print(
        f"[Process {process_id}]   Intermediate size: {model_config['intermediate_size']}")
    print(f"[Process {process_id}] Warmup iterations: {warmup_iterations}")
    print(
        f"[Process {process_id}] Benchmark iterations: {benchmark_iterations}")
    print(f"[Process {process_id}] Number of processes: {num_processes}\n")

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
                results = benchmark_qwen3_mlp(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=model_config["hidden_size"],
                    intermediate_size=model_config["intermediate_size"],
                    dtype=dtype,
                    warmup_iterations=warmup_iterations,
                    benchmark_iterations=benchmark_iterations,
                    process_id=process_id,
                )
                all_results.append(results)
                print_benchmark_results(results)
            except Exception as e:
                print(
                    f"[Process {process_id}] ERROR: Failed to run benchmark: {e}\n")
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


def run_single_benchmark():
    """Run a single benchmark with custom configuration."""
    results = benchmark_qwen3_mlp(
        batch_size=1,
        seq_len=128,
        hidden_size=4096,
        intermediate_size=11008,
        dtype=torch.bfloat16,
    )
    print_benchmark_results(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3MLP performance with different input sizes"
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
        default="benchmark_qwen3_mlp_results.csv",
        help="Output CSV filename. Default: benchmark_qwen3_mlp_results.csv",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (useful for multi-process mode)",
    )
    args = parser.parse_args()

    print("="*60)
    print("Qwen3MLP Benchmark")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(
        f"Batch sizes: {args.batch_sizes if args.batch_sizes else 'default'}")
    print(f"Sequence lengths: {args.seq_lens if args.seq_lens else 'default'}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")
    print(f"Number of processes: {args.num_processes}")
    print(
        f"Output directory: {args.output_dir if args.output_dir else 'current directory'}")
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
                )
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print(f"\nAll {args.num_processes} processes completed.")
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
                base_name = os.path.splitext(args.filename)[0]
                ext = os.path.splitext(args.filename)[1]
                proc_filename = f"{base_name}_proc{i}{ext}"
                proc_path = os.path.join(
                    args.output_dir, proc_filename) if args.output_dir else proc_filename
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
        )

        # Generate plots if not skipped
        if not args.skip_plots:
            # Load the saved CSV and create plots
            data_path = os.path.join(
                args.output_dir, args.filename) if args.output_dir else args.filename
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                plot_results(df, output_dir=args.output_dir)

        print(
            f"\nResults saved to: {os.path.join(args.output_dir, args.filename) if args.output_dir else args.filename}")

    # You can also run a single benchmark
    # Example:
    # results = run_single_benchmark()
