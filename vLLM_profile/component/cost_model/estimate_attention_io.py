#!/usr/bin/env python3
"""
根据batch size、kv_len和model size估计attention的IO量，
并根据A100的IO吞吐量估计时间
"""

import argparse
import json
import os
from typing import Dict, Tuple


# Qwen3模型配置
QWEN3_MODEL_CONFIGS = {
    "0.6B": {
        "num_heads": (16, 8),  # (num_query_heads, num_kv_heads)
        "head_size": 128,
        "hidden_size": 1024,
    },
    "4B": {
        "num_heads": (32, 8),
        "head_size": 128,
        "hidden_size": 2560,
    },
    "32B": {
        "num_heads": (64, 8),
        "head_size": 128,
        "hidden_size": 5120,
    },
}

# A100 GPU配置
# A100 80GB的HBM带宽：理论峰值约2039 GB/s，实际有效带宽约1500-1800 GB/s
A100_HBM_BANDWIDTH_GBPS = 1555.0  # GB/s，保守估计
A100_HBM_BANDWIDTH_BPS = A100_HBM_BANDWIDTH_GBPS * 1024 * 1024 * 1024  # bytes/s

# 数据类型大小（字节）
DTYPE_SIZES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}


def load_model_config(model_size: str, config_path: str = None) -> Dict:
    """加载模型配置"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return {
            "num_heads": (config["num_attention_heads"], config["num_key_value_heads"]),
            "head_size": config["head_dim"],
            "hidden_size": config["hidden_size"],
        }
    elif model_size in QWEN3_MODEL_CONFIGS:
        return QWEN3_MODEL_CONFIGS[model_size]
    else:
        raise ValueError(f"Unknown model size: {model_size}. "
                         f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}")


def calculate_attention_io(
    batch_size: int,
    kv_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype_size: int = 2,
    include_intermediate: bool = True,
) -> Dict[str, float]:
    """
    计算Flash Attention with KV Cache的IO量

    Args:
        batch_size: 批次大小
        kv_len: KV cache的长度
        num_query_heads: Query头的数量
        num_kv_heads: Key-Value头的数量
        head_size: 每个头的维度
        dtype_size: 数据类型大小（字节）
        include_intermediate: 是否包含中间结果（如QK点积结果）

    Returns:
        包含各项IO量的字典（单位：字节）
    """
    # 1. 读取Q矩阵
    # Q shape: (batch_size, num_query_heads, head_size)
    q_read = batch_size * num_query_heads * head_size * dtype_size

    # 2. 读取K cache
    # K cache shape: (batch_size, kv_len, num_kv_heads, head_size)
    # 注意：实际Flash Attention会分块读取，这里计算理论IO量
    k_read = batch_size * kv_len * num_kv_heads * head_size * dtype_size

    # 3. 读取V cache
    # V cache shape: (batch_size, kv_len, num_kv_heads, head_size)
    v_read = batch_size * kv_len * num_kv_heads * head_size * dtype_size

    # 4. 写入输出
    # Output shape: (batch_size, num_query_heads, head_size)
    output_write = batch_size * num_query_heads * head_size * dtype_size

    # 5. 中间结果（可选）
    # QK点积结果: (batch_size, num_query_heads, kv_len)
    # 注意：Flash Attention使用online softmax，可能不需要存储完整的QK矩阵
    intermediate_io = 0
    if include_intermediate:
        # 假设需要存储softmax结果和attention weights
        # 实际Flash Attention可能不需要，这里给出上界估计
        qk_intermediate = batch_size * num_query_heads * kv_len * 4  # float32 for softmax
        intermediate_io = qk_intermediate

    # 总IO量
    total_io = q_read + k_read + v_read + output_write + intermediate_io

    return {
        "q_read_bytes": q_read,
        "k_read_bytes": k_read,
        "v_read_bytes": v_read,
        "output_write_bytes": output_write,
        "intermediate_bytes": intermediate_io,
        "total_io_bytes": total_io,
    }


def estimate_time_from_io(io_bytes: float, bandwidth_bps: float = A100_HBM_BANDWIDTH_BPS) -> float:
    """
    根据IO量估计时间

    Args:
        io_bytes: IO量（字节）
        bandwidth_bps: 内存带宽（字节/秒）

    Returns:
        估计时间（秒）
    """
    return io_bytes / bandwidth_bps


def format_bytes(bytes_value: float) -> str:
    """格式化字节数为可读格式"""
    for unit in ['B', 'KB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} MB"


def main():
    parser = argparse.ArgumentParser(
        description="根据batch size、kv_len和model size估计attention的IO量和时间"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        required=True,
        help="批次大小（可以指定多个值，例如：--batch-size 1 2 4 8）"
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        nargs="+",
        required=True,
        help="KV cache的长度（可以指定多个值，例如：--kv-len 1024 2048 4096）"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="4B",
        choices=list(QWEN3_MODEL_CONFIGS.keys()),
        help=f"模型大小。选项: {list(QWEN3_MODEL_CONFIGS.keys())}。默认: 4B"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="模型配置文件路径（JSON格式）。如果提供，将覆盖--model-size"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_SIZES.keys()),
        help=f"数据类型。选项: {list(DTYPE_SIZES.keys())}。默认: bfloat16"
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=A100_HBM_BANDWIDTH_GBPS,
        help=f"A100 HBM带宽（GB/s）。默认: {A100_HBM_BANDWIDTH_GBPS}"
    )
    parser.add_argument(
        "--include-intermediate",
        action="store_true",
        default=False,
        help="是否包含中间结果的IO量（如QK点积结果）"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="输出CSV文件路径（可选）"
    )

    args = parser.parse_args()

    # 加载模型配置
    model_config = load_model_config(args.model_size, args.config_path)
    num_query_heads, num_kv_heads = model_config["num_heads"]
    head_size = model_config["head_size"]
    hidden_size = model_config.get("hidden_size", num_query_heads * head_size)

    # 获取数据类型大小
    dtype_size = DTYPE_SIZES[args.dtype]

    # 批量计算所有配置组合
    batch_sizes = args.batch_size if isinstance(
        args.batch_size, list) else [args.batch_size]
    kv_lens = args.kv_len if isinstance(args.kv_len, list) else [args.kv_len]

    all_results = []
    bandwidth_bps = args.bandwidth_gbps * 1e9

    # 如果只有一个配置，详细打印；否则批量计算并保存到CSV
    if len(batch_sizes) == 1 and len(kv_lens) == 1:
        # 单个配置，详细输出
        batch_size = batch_sizes[0]
        kv_len = kv_lens[0]

        # 计算IO量
        io_results = calculate_attention_io(
            batch_size=batch_size,
            kv_len=kv_len,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype_size=dtype_size,
            include_intermediate=args.include_intermediate,
        )

        print(io_results)

        # 估计时间
        estimated_time_sec = estimate_time_from_io(
            io_results["total_io_bytes"], bandwidth_bps)
        estimated_time_ms = estimated_time_sec * 1000

        # 打印结果
        print("=" * 80)
        print("Attention IO量估计")
        print("=" * 80)
        print(f"配置:")
        print(f"  批次大小 (batch_size): {batch_size}")
        print(f"  KV长度 (kv_len): {kv_len}")
        print(f"  模型大小: {args.model_size}")
        print(f"  Query头数: {num_query_heads}")
        print(f"  KV头数: {num_kv_heads}")
        print(f"  头维度 (head_size): {head_size}")
        print(f"  隐藏层维度 (hidden_size): {hidden_size}")
        print(f"  数据类型: {args.dtype} ({dtype_size} 字节)")
        print(f"  A100 HBM带宽: {args.bandwidth_gbps} GB/s")
        print()
        print(f"IO量分析:")
        print(f"  读取Q矩阵: {format_bytes(io_results['q_read_bytes'])}")
        print(f"  读取K cache: {format_bytes(io_results['k_read_bytes'])}")
        print(f"  读取V cache: {format_bytes(io_results['v_read_bytes'])}")
        print(f"  写入输出: {format_bytes(io_results['output_write_bytes'])}")
        if args.include_intermediate:
            print(f"  中间结果: {format_bytes(io_results['intermediate_bytes'])}")
        print(f"  总IO量: {format_bytes(io_results['total_io_bytes'])}")
        print()
        print(f"时间估计:")
        print(
            f"  估计时间: {estimated_time_ms:.4f} ms ({estimated_time_sec:.6f} s)")
        print(
            f"  理论吞吐量: {batch_size * kv_len / estimated_time_sec:.2f} tokens/s")
        print("=" * 80)

        all_results.append({
            "batch_size": batch_size,
            "kv_len": kv_len,
            "io_results": io_results,
            "estimated_time_ms": estimated_time_ms,
        })
    else:
        # 批量计算
        print("=" * 80)
        print("批量计算Attention IO量估计")
        print("=" * 80)
        print(f"模型配置:")
        print(f"  模型大小: {args.model_size}")
        print(f"  Query头数: {num_query_heads}")
        print(f"  KV头数: {num_kv_heads}")
        print(f"  头维度 (head_size): {head_size}")
        print(f"  隐藏层维度 (hidden_size): {hidden_size}")
        print(f"  数据类型: {args.dtype} ({dtype_size} 字节)")
        print(f"  A100 HBM带宽: {args.bandwidth_gbps} GB/s")
        print(f"  批次大小: {batch_sizes}")
        print(f"  KV长度: {kv_lens}")
        print("=" * 80)
        print()

        total_configs = len(batch_sizes) * len(kv_lens)
        current = 0

        for batch_size in batch_sizes:
            for kv_len in kv_lens:
                current += 1
                # 计算IO量
                io_results = calculate_attention_io(
                    batch_size=batch_size,
                    kv_len=kv_len,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype_size=dtype_size,
                    include_intermediate=args.include_intermediate,
                )

                # 估计时间
                estimated_time_sec = estimate_time_from_io(
                    io_results["total_io_bytes"], bandwidth_bps)
                estimated_time_ms = estimated_time_sec * 1000

                all_results.append({
                    "batch_size": batch_size,
                    "kv_len": kv_len,
                    "io_results": io_results,
                    "estimated_time_ms": estimated_time_ms,
                })

                print(f"[{current}/{total_configs}] bs={batch_size:4d}, kv_len={kv_len:5d}: "
                      f"IO={format_bytes(io_results['total_io_bytes']):>12s}, "
                      f"time={estimated_time_ms:8.4f} ms")

    # 保存到CSV（如果指定）
    if args.output_csv or len(all_results) > 1:
        import csv
        output_file = args.output_csv or "attention_io_estimates.csv"
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "batch_size", "kv_len", "model_size", "num_query_heads",
                    "num_kv_heads", "head_size", "dtype", "q_read_bytes",
                    "k_read_bytes", "v_read_bytes", "output_write_bytes",
                    "intermediate_bytes", "total_io_bytes", "estimated_time_ms",
                    "bandwidth_gbps"
                ])
            for result in all_results:
                writer.writerow([
                    result["batch_size"], result["kv_len"], args.model_size, num_query_heads,
                    num_kv_heads, head_size, args.dtype, result["io_results"]['q_read_bytes'],
                    result["io_results"]['k_read_bytes'], result["io_results"]['v_read_bytes'],
                    result["io_results"]['output_write_bytes'], result["io_results"]['intermediate_bytes'],
                    result["io_results"]['total_io_bytes'], result["estimated_time_ms"], args.bandwidth_gbps
                ])
        print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
