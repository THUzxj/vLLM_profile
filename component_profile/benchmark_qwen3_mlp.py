# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for Qwen3MLP performance measurement with different input sizes.
"""

import argparse
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


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
    # Setup
    current_platform.seed_everything(0)
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
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.inference_mode():
        for _ in range(benchmark_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = mlp(input_tensor)

            if device == "cuda":
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
    filename: str = "benchmark_qwen3_mlp_results.csv",
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
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
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
):
    """Run a comprehensive benchmark suite with different input sizes.

    Args:
        batch_sizes: List of batch sizes to test. If None, uses default values.
        seq_lens: List of sequence lengths to test. If None, uses default values.
        output_dir: Directory to save output files. If None, saves to current directory.
    """
    print("Starting Qwen3MLP Benchmark Suite...")
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}\n")

    # Common model configurations (Qwen3 typical sizes)
    model_configs = [
        {"hidden_size": 2560, "intermediate_size": 9728},   # Qwen3-4B
        # {"hidden_size": 4096, "intermediate_size": 11008},  # Qwen3-7B
        # {"hidden_size": 8192, "intermediate_size": 28672},  # Qwen3-14B
        # {"hidden_size": 3584, "intermediate_size": 9728},   # Qwen3-1.5B
    ]

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

    all_results = []
    for model_idx, model_config in enumerate(model_configs, 1):
        print(f"\n{'='*80}")
        print(f"Model Config {model_idx}/{len(model_configs)}: "
              f"hidden_size={model_config['hidden_size']}, "
              f"intermediate_size={model_config['intermediate_size']}")
        print(f"{'='*80}\n")

        for input_idx, (batch_size, seq_len) in enumerate(input_configs, 1):
            print(f"Running benchmark {input_idx}/{len(input_configs)}: "
                  f"batch_size={batch_size}, seq_len={seq_len}...")

            try:
                results = benchmark_qwen3_mlp(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=model_config["hidden_size"],
                    intermediate_size=model_config["intermediate_size"],
                    dtype=dtype,
                )
                all_results.append(results)
                print_benchmark_results(results)
            except Exception as e:
                print(f"ERROR: Failed to run benchmark: {e}\n")
                all_results.append({"error": str(e), "config": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    **model_config
                }})

    # Save results to CSV and create plots
    df = save_results_to_csv(all_results, output_dir=output_dir)
    plot_results(df, output_dir=output_dir)

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
    args = parser.parse_args()

    # Run comprehensive benchmark suite
    results = run_benchmark_suite(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        output_dir=args.output_dir,
    )

    # You can also run a single benchmark
    # Example:
    # results = run_single_benchmark()
