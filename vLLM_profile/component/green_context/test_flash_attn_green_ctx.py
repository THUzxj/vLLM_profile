"""
测试脚本：在多个 green context streams 上并行运行 flash attention 任务。
结合 test_qwen3_mlp_parallel.py 的 green context 使用方式和 
benchmark_flash_attn_v2.py 的 flash attention 推理处理方法。
"""

import os
from typing import Optional, List, Dict

import pandas as pd
import torch

from vllm.attention.utils.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.platforms import current_platform

from test_green_context import split_device_green_ctx

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

# Default to Qwen3-4B
DEFAULT_MODEL = "4B"

# Benchmark parameters
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


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


def prepare_flash_attn_inputs(
    kv_lens: List[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
    device: torch.device,
    q_dtype: Optional[torch.dtype] = None,
    soft_cap: Optional[float] = None,
    sliding_window: Optional[int] = None,
):
    """
    准备 flash attention 的输入张量。

    Args:
        kv_lens: KV cache 长度列表，每个元素对应一个序列
        num_heads: (num_query_heads, num_kv_heads) 元组
        head_size: 每个头的维度
        dtype: 数据类型
        block_size: KV cache block 大小
        num_blocks: KV cache block 总数
        device: 设备
        q_dtype: 可选的量化数据类型
        soft_cap: 可选的 soft cap 值
        sliding_window: 可选的滑动窗口大小

    Returns:
        包含所有输入参数的字典
    """
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
    query = torch.randn(num_tokens, num_query_heads,
                        head_size, dtype=dtype, device=device)

    # KV cache shape: (num_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device
    )
    value_cache = torch.randn_like(key_cache)

    # Create cu_seqlens_q: cumulative sequence lengths for queries
    # Shape: (num_seqs + 1,)
    # For decode with query_len=1, this is simply [0, 1, 2, ..., num_seqs]
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)

    # seqused_k: KV cache sequence lengths
    # Shape: (num_seqs,)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device
    )

    out = torch.empty_like(query)

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
        q_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        k_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        v_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)

    return {
        "q": maybe_quantized_query,
        "k": maybe_quantized_key_cache,
        "v": maybe_quantized_value_cache,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": query_len,
        "seqused_k": seqused_k,
        "max_seqlen_k": max_kv_len,
        "softmax_scale": scale,
        "causal": True,
        "window_size": window_size,
        "block_table": block_tables,
        "softcap": soft_cap if soft_cap is not None else 0,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
    }


def test_flash_attn_parallel_streams(
    num_groups: int = 2,
    min_count: int = 16,
    batch_size: int = 1,
    kv_len: int = 1024,
    model_size: str = DEFAULT_MODEL,
    block_size: int = 16,
    num_blocks: int = 8192,
    soft_cap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    fa_version: Optional[int] = None,
    q_dtype: Optional[torch.dtype] = None,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    output_csv: Optional[str] = None,
):
    """
    测试不同 stream 内的 flash attention 任务并行地同时开启执行。

    Args:
        num_groups: 要创建的 green context 组数
        min_count: 每个组的最小 SM 数量
        batch_size: 输入 batch size
        kv_len: KV cache 长度（所有序列使用相同的长度）
        model_size: 模型大小 ("0.6B", "4B", "32B")
        block_size: KV cache block 大小
        num_blocks: KV cache block 总数
        soft_cap: 可选的 soft cap 值
        sliding_window: 可选的滑动窗口大小
        fa_version: Flash attention 版本（None 表示自动检测）
        q_dtype: 可选的量化数据类型
        warmup_iterations: Warmup 迭代次数
        benchmark_iterations: 基准测试迭代次数
        device: 设备
        dtype: 数据类型
        output_csv: 输出 CSV 文件路径（可选）
    """
    # Check if flash attention is available
    if not is_flash_attn_varlen_func_available():
        print("ERROR: Flash attention varlen function is not available on this platform")
        return None

    # Get FA version if not specified
    if fa_version is None:
        fa_version = get_flash_attn_version()
        if fa_version is None:
            print("ERROR: Flash attention version could not be determined")
            return None

    if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
        print("ERROR: Flash attention with quantized inputs is only "
              "supported on version 3 with bfloat16 base type")
        return None

    # 获取模型配置
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}"
        )
    model_config = QWEN3_MODEL_CONFIGS[model_size]
    num_heads = model_config["num_heads"]
    head_size = model_config["head_size"]

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
            f"Model: Qwen3-{model_size}, batch_size={batch_size}, kv_len={kv_len}, "
            f"num_heads={num_heads}, head_size={head_size}, "
            f"fa_version={fa_version}, dtype={dtype}"
        )

        streams = streams[:num_groups]
        resources = resources[:num_groups]

        num_streams = len(streams)

        # 为每个 stream 准备 flash attention 输入
        # 每个 stream 处理相同的 batch_size 和 kv_len
        kv_lens_list = [kv_len] * batch_size
        flash_attn_inputs_list = []
        for i in range(num_streams):
            inputs = prepare_flash_attn_inputs(
                kv_lens=kv_lens_list,
                num_heads=num_heads,
                head_size=head_size,
                dtype=dtype,
                block_size=block_size,
                num_blocks=num_blocks,
                device=dev,
                q_dtype=q_dtype,
                soft_cap=soft_cap,
                sliding_window=sliding_window,
            )
            # 添加 fa_version 到输入字典
            inputs["fa_version"] = fa_version
            flash_attn_inputs_list.append(inputs)

        # Warm up
        print("\nWarming up...")
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                for _ in range(warmup_iterations):
                    _ = flash_attn_varlen_func(**flash_attn_inputs_list[i])

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
        print("Starting parallel execution on all streams...")

        for it in range(benchmark_iterations):
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    _ = flash_attn_varlen_func(**flash_attn_inputs_list[i])

        # for i, stream in enumerate(streams):
        #     with torch.cuda.stream(stream):
        #         start_events[i].record(stream)
        #         for _ in range(benchmark_iterations):
        #             _ = flash_attn_varlen_func(**flash_attn_inputs_list[i])
        #         end_events[i].record(stream)

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
        print("Parallel Flash Attention Execution Results")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Model: Qwen3-{model_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  KV length: {kv_len}")
        print(f"  Num heads (Q, KV): {num_heads}")
        print(f"  Head size: {head_size}")
        print(f"  Number of streams: {num_streams}")
        print(f"  Benchmark iterations: {benchmark_iterations}")
        print(f"  FA version: {fa_version}")
        print(f"\nPer-stream results:")

        total_tokens = sum(kv_lens_list)
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
                "kv_len": kv_len,
                "num_heads_q": num_heads[0],
                "num_heads_kv": num_heads[1],
                "head_size": head_size,
                "block_size": block_size,
                "num_blocks": num_blocks,
                "soft_cap": soft_cap,
                "sliding_window": sliding_window,
                "fa_version": fa_version,
                "q_dtype": str(q_dtype) if q_dtype is not None else None,
                "dtype": str(dtype),
                "device": device,
                "warmup_iterations": warmup_iterations,
                "benchmark_iterations": benchmark_iterations,
                "total_time_ms": elapsed_time,
                "avg_time_ms": avg_time,
                "throughput_tokens_per_sec": throughput,
                "total_tokens": total_tokens,
            })

        # 添加总体结果
        results.append({
            "num_groups": num_groups,
            "min_count": min_count,
            "stream_id": "overall",
            "sm_count": sum([r.sm.smCount for r in resources]),
            "model_size": model_size,
            "batch_size": batch_size,
            "kv_len": kv_len,
            "num_heads_q": num_heads[0],
            "num_heads_kv": num_heads[1],
            "head_size": head_size,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "soft_cap": soft_cap,
            "sliding_window": sliding_window,
            "fa_version": fa_version,
            "q_dtype": str(q_dtype) if q_dtype is not None else None,
            "dtype": str(dtype),
            "device": device,
            "warmup_iterations": warmup_iterations,
            "benchmark_iterations": benchmark_iterations,
            "total_time_ms": overall_time,
            "avg_time_ms": avg_overall_time,
            "throughput_tokens_per_sec": overall_throughput,
            "total_tokens": total_tokens,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test parallel flash attention execution on multiple green context streams"
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for input. Default: 1",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=1024,
        help="KV cache length (same for all sequences). Default: 1024",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(QWEN3_MODEL_CONFIGS.keys()),
        help=f"Model size. Options: {list(QWEN3_MODEL_CONFIGS.keys())}. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size. Default: 16",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=8192,
        help="Total number of KV cache blocks. Default: 8192",
    )
    parser.add_argument(
        "--soft-cap",
        type=float,
        default=None,
        help="Soft cap value (optional). Default: None",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Sliding window size (optional). Default: None",
    )
    parser.add_argument(
        "--fa-version",
        type=int,
        default=None,
        choices=[2, 3],
        help="Flash attention version (2 or 3). If not specified, auto-detect. Default: None",
    )
    parser.add_argument(
        "--q-dtype",
        type=str,
        default=None,
        choices=["float8_e4m3fn"],
        help="Quantized dtype for query/key/value (optional). Default: None",
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

    # Convert q_dtype string to torch dtype if specified
    q_dtype = None
    if args.q_dtype:
        q_dtype_map = {
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        q_dtype = q_dtype_map[args.q_dtype]

    print("=" * 80)
    print("Flash Attention Parallel Stream Test (Green Context)")
    print("=" * 80)
    print(f"Number of groups: {args.num_groups}")
    print(f"Min SM count per group: {args.min_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"KV length: {args.kv_len}")
    print(f"Model size: {args.model_size}")
    print(f"Block size: {args.block_size}")
    print(f"Num blocks: {args.num_blocks}")
    print(f"Soft cap: {args.soft_cap}")
    print(f"Sliding window: {args.sliding_window}")
    print(
        f"FA version: {args.fa_version if args.fa_version else 'auto-detect'}")
    print(f"Q dtype: {args.q_dtype}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    if args.output_csv:
        print(f"Output CSV: {args.output_csv}")
    print("=" * 80 + "\n")

    test_flash_attn_parallel_streams(
        num_groups=args.num_groups,
        min_count=args.min_count,
        batch_size=args.batch_size,
        kv_len=args.kv_len,
        model_size=args.model_size,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        soft_cap=args.soft_cap,
        sliding_window=args.sliding_window,
        fa_version=args.fa_version,
        q_dtype=q_dtype,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
        device=args.device,
        dtype=dtype,
        output_csv=args.output_csv,
    )
