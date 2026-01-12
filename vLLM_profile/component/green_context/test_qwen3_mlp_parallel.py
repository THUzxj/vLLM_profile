"""
测试脚本：在多个 green context streams 上并行运行相同的 Qwen3MLP 任务。
基于 test_parallel_streams 的逻辑，但使用 Qwen3MLP 模型。
"""

import os
import tempfile
from typing import Optional, List, Dict

import pandas as pd
import torch
from torch import nn

from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform

from test_green_context import split_device_green_ctx

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

# Default to Qwen3-4B
DEFAULT_MODEL = "4B"

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Global flag to track if distributed environment is initialized
_dist_env_initialized = False
_dist_init_temp_file = None


def ensure_distributed_initialized():
    """Ensure distributed environment and model parallel groups are initialized."""
    global _dist_env_initialized, _dist_init_temp_file

    if _dist_env_initialized:
        return

    if not torch.distributed.is_initialized():
        _dist_init_temp_file = tempfile.mkstemp(
            suffix="_qwen3_mlp_parallel")[1]

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

    # Initialize model parallel groups
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
        if _dist_init_temp_file:
            try:
                if os.path.exists(_dist_init_temp_file):
                    os.unlink(_dist_init_temp_file)
            except Exception:
                pass
        _dist_env_initialized = False
        _dist_init_temp_file = None


def save_results_to_csv(
    results: List[Dict],
    csv_path: str,
):
    """
    将测试结果保存到CSV文件。如果文件不存在则创建，存在则追加。

    Args:
        results: 结果字典列表
        csv_path: CSV文件路径
    """
    # 确保目录存在
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 检查文件是否存在
    file_exists = os.path.exists(csv_path)

    if file_exists:
        # 追加模式，不包含header
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"\nResults appended to {csv_path}")
    else:
        # 创建新文件，包含header
        df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"\nResults saved to new file: {csv_path}")


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
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. " "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def test_qwen3_mlp_parallel_streams(
    num_groups: int = 2,
    min_count: int = 16,
    num_mlps_per_stream: int = 1,
    batch_size: int = 1,
    seq_len: int = 1,
    model_size: str = DEFAULT_MODEL,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    output_csv: Optional[str] = None,
):
    """
    测试不同 stream 内的 Qwen3MLP 任务并行地同时开启执行。

    Args:
        num_groups: 要创建的 green context 组数
        min_count: 每个组的最小 SM 数量
        num_mlps_per_stream: 每个 stream 上初始化的 MLP 个数
        batch_size: 输入 batch size
        seq_len: 输入 sequence length
        model_size: 模型大小 ("0.6B", "4B", "14B", "32B")
        warmup_iterations: Warmup 迭代次数
        benchmark_iterations: 基准测试迭代次数
        device: 设备
        dtype: 数据类型
    """
    # 确保分布式环境已初始化
    ensure_distributed_initialized()

    # 获取模型配置
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}"
        )
    model_config = QWEN3_MODEL_CONFIGS[model_size]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]

    dev = torch.device(device)
    torch.set_default_device(device)
    current_platform.seed_everything(42)

    try:
        # 创建 green context streams
        streams, resources = split_device_green_ctx(dev, num_groups, min_count)
        print(
            f"num_groups={num_groups}, min_count={min_count}, "
            f"SM counts: {[r.sm.smCount for r in resources]}"
        )
        print(
            f"Model: Qwen3-{model_size}, batch_size={batch_size}, seq_len={seq_len}, "
            f"hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        )

        streams = streams[:num_groups]
        resources = resources[:num_groups]

        num_streams = len(streams)

        # 为每个 stream 创建若干个 Qwen3MLP 模型
        # 结构：mlps[stream_id][mlp_id]，input_tensors[stream_id][mlp_id]
        mlps = [[] for _ in range(num_streams)]
        input_tensors = [[] for _ in range(num_streams)]
        for i in range(num_streams):
            for j in range(num_mlps_per_stream):
                mlp = Qwen3MLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act="silu",
                ).to(device).to(dtype)
                mlp.eval()
                mlps[i].append(mlp)

                # 为该 MLP 创建对应的输入张量
                input_tensor = torch.randn(
                    batch_size, seq_len, hidden_size, dtype=dtype, device=device
                )
                input_tensors[i].append(input_tensor)

        # Warm up
        print("\nWarming up...")
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                with torch.inference_mode():
                    for it in range(warmup_iterations):
                        # 通过迭代次数对 MLP 数量取模，选择当前要跑的 MLP
                        mlp_idx = it % num_mlps_per_stream
                        _ = mlps[i][mlp_idx](input_tensors[i][mlp_idx])

        # 同步以确保 warm up 完成
        for stream in streams:
            stream.synchronize()
        torch.cuda.synchronize()
        print("Warmup completed.\n")

        # 为每个 stream 创建事件
        # start_events = [torch.cuda.Event(enable_timing=True)
        #                 for _ in range(num_streams)]
        # end_events = [torch.cuda.Event(enable_timing=True)
        #               for _ in range(num_streams)]

        # 记录总体开始时间
        overall_start = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(streams[0]):
            overall_start.record()

        # 在所有 streams 上同时启动任务（并行执行）
        # print("Starting parallel execution on all streams...")
        # for i, stream in enumerate(streams):
        #     with torch.cuda.stream(stream):
        #         start_events[i].record(stream)
        #         with torch.inference_mode():
        #             for it in range(benchmark_iterations):
        #                 # 通过迭代次数对 MLP 数量取模，选择当前要跑的 MLP
        #                 mlp_idx = it % num_mlps_per_stream
        #                 _ = mlps[i][mlp_idx](input_tensors[i][mlp_idx])
        #         end_events[i].record(stream)

        for it in range(benchmark_iterations):
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    # 通过迭代次数对 MLP 数量取模，选择当前要跑的 MLP
                    mlp_idx = it % num_mlps_per_stream
                    _ = mlps[i][mlp_idx](input_tensors[i][mlp_idx])

        # 记录总体结束时间（在默认 stream 上）
        overall_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(streams[0]):
            overall_end.record()

        # 等待所有 streams 完成
        for stream in streams:
            stream.synchronize()

        # 同步以确保所有事件都已完成
        torch.cuda.synchronize()

        # 计算并打印每个 stream 的执行时间
        print("\n" + "=" * 80)
        print("Parallel Qwen3MLP Execution Results")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Model: Qwen3-{model_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Intermediate size: {intermediate_size}")
        print(f"  Number of streams: {num_streams}")
        print(f"  Benchmark iterations: {benchmark_iterations}")
        print(f"\nPer-stream results:")

        total_tokens = batch_size * seq_len
        for i in range(num_streams):
            # elapsed_time = start_events[i].elapsed_time(end_events[i])
            elapsed_time = 1.0
            # 计算平均时间（单次迭代）
            avg_time = elapsed_time / benchmark_iterations
            throughput = total_tokens / (avg_time / 1000)  # tokens per second

            print(
                f"  Stream {i}: {resources[i].sm.smCount} SMs, "
                f"Total time: {elapsed_time:.3f} ms, "
                f"Avg time: {avg_time:.3f} ms, "
                f"Throughput: {throughput:.2f} tokens/sec"
            )

        # 计算总体执行时间
        overall_time = overall_start.elapsed_time(overall_end)
        avg_overall_time = overall_time / benchmark_iterations
        overall_throughput = total_tokens / (avg_overall_time / 1000)
        print(f"\nOverall parallel execution:")
        print(f"  Total time: {overall_time:.3f} ms")
        print(f"  Avg time per iteration: {avg_overall_time:.3f} ms")
        print(f"  Throughput: {overall_throughput:.2f} tokens/sec")
        print("=" * 80 + "\n")

        # 准备结果数据
        results = []
        for i in range(num_streams):
            # elapsed_time = start_events[i].elapsed_time(end_events[i])
            elapsed_time = 1.0
            avg_time = elapsed_time / benchmark_iterations
            throughput = total_tokens / (avg_time / 1000)

            results.append({
                "num_groups": num_groups,
                "min_count": min_count,
                "stream_id": i,
                "sm_count": resources[i].sm.smCount,
                "model_size": model_size,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "dtype": str(dtype),
                "device": device,
                "warmup_iterations": warmup_iterations,
                "benchmark_iterations": benchmark_iterations,
                "total_time_ms": elapsed_time,
                "avg_time_ms": avg_time,
                "throughput_tokens_per_sec": throughput,
            })

        # 添加总体结果
        results.append({
            "num_groups": num_groups,
            "min_count": min_count,
            "stream_id": "overall",
            "sm_count": sum([r.sm.smCount for r in resources]),
            "model_size": model_size,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "dtype": str(dtype),
            "device": device,
            "warmup_iterations": warmup_iterations,
            "benchmark_iterations": benchmark_iterations,
            "total_time_ms": overall_time,
            "avg_time_ms": avg_overall_time,
            "throughput_tokens_per_sec": overall_throughput,
        })

        # 保存结果到CSV
        if output_csv:
            save_results_to_csv(results, output_csv)

        return results

    except RuntimeError as e:
        print(f"num_groups={num_groups}, min_count={min_count}, Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None
    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test parallel Qwen3MLP execution on multiple green context streams"
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=2,
        help="Number of green context groups to create. Default: 2",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=16,
        help="Minimum number of SMs per group. Default: 16",
    )
    parser.add_argument(
        "--num-mlps-per-stream",
        type=int,
        default=3,
        help="Number of Qwen3MLP instances per stream. Default: 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for input tensor. Default: 1",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Sequence length for input tensor. Default: 1",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(QWEN3_MODEL_CONFIGS.keys()),
        help=f"Model size. Options: {list(QWEN3_MODEL_CONFIGS.keys())}. Default: {DEFAULT_MODEL}",
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
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on. Default: cuda:0",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type. Default: bfloat16",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file. If not specified, results are not saved. "
        "If file exists, results will be appended.",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 80)
    print("Qwen3MLP Parallel Stream Test")
    print("=" * 80)
    print(f"Number of groups: {args.num_groups}")
    print(f"Min SM count per group: {args.min_count}")
    print(f"Number of MLPs per stream: {args.num_mlps_per_stream}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Model size: {args.model_size}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    if args.output_csv:
        print(f"Output CSV: {args.output_csv}")
    print("=" * 80 + "\n")

    test_qwen3_mlp_parallel_streams(
        num_groups=args.num_groups,
        min_count=args.min_count,
        num_mlps_per_stream=args.num_mlps_per_stream,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        model_size=args.model_size,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
        device=args.device,
        dtype=dtype,
        output_csv=args.output_csv,
    )
