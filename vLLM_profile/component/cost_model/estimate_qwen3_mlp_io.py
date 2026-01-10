#!/usr/bin/env python3
"""
根据 batch size、seq_len 和 Qwen3 MLP 的 hidden_size / intermediate_size
估计一次 MLP 前向的 IO 量，并根据 A100 的 HBM 吞吐量估计时间。

计算的 IO 主要是激活（activation）和权重（weight）的读写量，
给出的都是「上界」或「简化模型」下的估计，方便和实际 benchmark 做对比。
"""

import argparse
import json
import os
from typing import Dict, Tuple


# Qwen3 MLP 模型配置（与 benchmark_qwen3_mlp.py 保持一致）
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


# A100 GPU配置（与 estimate_attention_io.py 保持风格一致）
# A100 80GB 的 HBM 带宽：理论峰值约 2039 GB/s，实际有效带宽约 1500-1800 GB/s
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


def load_mlp_config(model_size: str, config_path: str | None = None) -> Dict:
    """加载 MLP 配置（hidden_size / intermediate_size）。"""
    if config_path and os.path.exists(config_path):
        # 期望 JSON 中至少包含 hidden_size / intermediate_size
        with open(config_path, "r") as f:
            config = json.load(f)
        if "hidden_size" not in config or "intermediate_size" not in config:
            raise ValueError(
                "Config JSON must contain 'hidden_size' and 'intermediate_size'."
            )
        return {
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
        }
    elif model_size in QWEN3_MODEL_CONFIGS:
        return QWEN3_MODEL_CONFIGS[model_size]
    else:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}"
        )


def calculate_qwen3_mlp_io(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_size: int = 2,
    include_intermediate: bool = True,
) -> Dict[str, float]:
    """
    估计一次 Qwen3 MLP 前向的 IO 量。

    假设计算图如下（与 benchmark_qwen3_mlp 中的实现保持一致）：
        x: [B, S, H]
        gate_up = x @ W_gate_up          # [B, S, 2I]
        gate_up -> SiluAndMul -> h_act   # [B, S, I]
        y = h_act @ W_down               # [B, S, H]

    IO 设计原则：
    - 激活：
        * 读输入 x
        * 写 gate_up (2I)
        * 读 gate_up 做激活
        * 写激活后的中间结果 h_act (I)
        * 读 h_act 做 down_proj
        * 写最终输出 y
      如果 kernel 完全融合，上述中间激活可能不会全部落到 HBM，这里将其作为「上界估计」，
      可以通过 include_intermediate 控制。
    - 权重：
        * gate_up 权重： [H, 2I]
        * down_proj 权重： [I, H]
      假设每次前向各读一遍。

    Returns:
        dict: 各项 IO 量（单位：字节）
    """
    tokens = batch_size * seq_len

    # 1. 激活相关 IO
    # 1.1 读取输入 x
    x_read = tokens * hidden_size * dtype_size

    # 1.2 gate_up 输出 [B, S, 2I]
    gate_up_write = tokens * (2 * intermediate_size) * dtype_size

    # 1.3 激活及中间结果
    intermediate_io = 0
    if include_intermediate:
        # 读 gate_up 做激活（SiluAndMul）
        gate_up_read_for_act = tokens * (2 * intermediate_size) * dtype_size
        # 写激活后的结果 h_act [B, S, I]
        act_write = tokens * intermediate_size * dtype_size
        # 读 h_act 做 down_proj
        act_read_for_down = tokens * intermediate_size * dtype_size
        intermediate_io = (
            gate_up_read_for_act + act_write + act_read_for_down
        )

    # 1.4 down_proj 输出 y [B, S, H]
    y_write = tokens * hidden_size * dtype_size

    activation_io = x_read + gate_up_write + y_write + intermediate_io

    # 2. 权重相关 IO（一次前向各读一遍）
    gate_up_weight_read = hidden_size * (2 * intermediate_size) * dtype_size
    down_weight_read = intermediate_size * hidden_size * dtype_size
    weight_io = gate_up_weight_read + down_weight_read

    total_io = activation_io + weight_io

    return {
        "tokens": tokens,
        "x_read_bytes": x_read,
        "gate_up_write_bytes": gate_up_write,
        "intermediate_bytes": intermediate_io,
        "y_write_bytes": y_write,
        "activation_io_bytes": activation_io,
        "gate_up_weight_read_bytes": gate_up_weight_read,
        "down_weight_read_bytes": down_weight_read,
        "weight_io_bytes": weight_io,
        "total_io_bytes": total_io,
    }


def estimate_time_from_io(
    io_bytes: float, bandwidth_bps: float = A100_HBM_BANDWIDTH_BPS
) -> float:
    """
    根据 IO 量估计时间（秒）。
    """
    return io_bytes / bandwidth_bps


def format_bytes(bytes_value: float) -> str:
    """格式化字节数为可读格式。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="根据 batch size、seq_len 和 model size 估计 Qwen3 MLP 的 IO 量和时间"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        required=True,
        help="批次大小（可以指定多个值，例如：--batch-size 1 2 4 8）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        required=True,
        help="序列长度（可以指定多个值，例如：--seq-len 1 128 1024）",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="4B",
        choices=list(QWEN3_MODEL_CONFIGS.keys()),
        help=f"模型大小。选项: {list(QWEN3_MODEL_CONFIGS.keys())}。默认: 4B",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="MLP 配置文件路径（JSON 格式，需包含 hidden_size / intermediate_size）。"
        "如果提供，将覆盖 --model-size",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_SIZES.keys()),
        help=f"数据类型。选项: {list(DTYPE_SIZES.keys())}。默认: bfloat16",
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=A100_HBM_BANDWIDTH_GBPS,
        help=f"A100 HBM 带宽（GB/s）。默认: {A100_HBM_BANDWIDTH_GBPS}",
    )
    parser.add_argument(
        "--include-intermediate",
        action="store_true",
        default=False,
        help="是否包含中间激活（gate_up、act 等）的 IO 量，上界估计。",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="输出 CSV 文件路径（可选）。如不指定但有多个配置，将自动写入 qwen3_mlp_io_estimates.csv",
    )

    args = parser.parse_args()

    # 加载 MLP 配置
    mlp_config = load_mlp_config(args.model_size, args.config_path)
    hidden_size = mlp_config["hidden_size"]
    intermediate_size = mlp_config["intermediate_size"]

    # 获取数据类型大小
    dtype_size = DTYPE_SIZES[args.dtype]

    batch_sizes = args.batch_size if isinstance(args.batch_size, list) else [args.batch_size]
    seq_lens = args.seq_len if isinstance(args.seq_len, list) else [args.seq_len]

    all_results = []
    bandwidth_bps = args.bandwidth_gbps * 1e9

    # 单配置：详细输出；多配置：批量打印 + CSV
    if len(batch_sizes) == 1 and len(seq_lens) == 1:
        batch_size = batch_sizes[0]
        seq_len = seq_lens[0]

        io_results = calculate_qwen3_mlp_io(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype_size=dtype_size,
            include_intermediate=args.include_intermediate,
        )

        estimated_time_sec = estimate_time_from_io(
            io_results["total_io_bytes"], bandwidth_bps
        )
        estimated_time_ms = estimated_time_sec * 1000

        print(io_results)

        print("=" * 80)
        print("Qwen3 MLP IO 量估计")
        print("=" * 80)
        print("配置:")
        print(f"  批次大小 (batch_size): {batch_size}")
        print(f"  序列长度 (seq_len): {seq_len}")
        print(f"  模型大小: {args.model_size}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  intermediate_size: {intermediate_size}")
        print(f"  数据类型: {args.dtype} ({dtype_size} 字节)")
        print(f"  A100 HBM 带宽: {args.bandwidth_gbps} GB/s")
        print()
        print("IO 量分析（激活）:")
        print(f"  读取输入 x: {format_bytes(io_results['x_read_bytes'])}")
        print(
            f"  写入 gate_up 输出: {format_bytes(io_results['gate_up_write_bytes'])}"
        )
        if args.include_intermediate:
            print(
                f"  中间激活 (gate_up 读 + act 写 + act 读): "
                f"{format_bytes(io_results['intermediate_bytes'])}"
            )
        print(f"  写入输出 y: {format_bytes(io_results['y_write_bytes'])}")
        print(f"  激活总 IO: {format_bytes(io_results['activation_io_bytes'])}")
        print()
        print("IO 量分析（权重）:")
        print(
            f"  读取 gate_up 权重: "
            f"{format_bytes(io_results['gate_up_weight_read_bytes'])}"
        )
        print(
            f"  读取 down_proj 权重: "
            f"{format_bytes(io_results['down_weight_read_bytes'])}"
        )
        print(f"  权重总 IO: {format_bytes(io_results['weight_io_bytes'])}")
        print()
        print(f"整体 IO 总量: {format_bytes(io_results['total_io_bytes'])}")
        print()
        print("时间估计:")
        print(f"  估计时间: {estimated_time_ms:.4f} ms ({estimated_time_sec:.6f} s)")
        print(
            f"  理论吞吐量: {io_results['tokens'] / estimated_time_sec:.2f} tokens/s"
        )
        print("=" * 80)

        all_results.append(
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "io_results": io_results,
                "estimated_time_ms": estimated_time_ms,
            }
        )
    else:
        # 批量计算
        print("=" * 80)
        print("批量计算 Qwen3 MLP IO 量估计")
        print("=" * 80)
        print("模型配置:")
        print(f"  模型大小: {args.model_size}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  intermediate_size: {intermediate_size}")
        print(f"  数据类型: {args.dtype} ({dtype_size} 字节)")
        print(f"  A100 HBM 带宽: {args.bandwidth_gbps} GB/s")
        print(f"  批次大小: {batch_sizes}")
        print(f"  序列长度: {seq_lens}")
        print("=" * 80)
        print()

        total_configs = len(batch_sizes) * len(seq_lens)
        current = 0

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                current += 1
                io_results = calculate_qwen3_mlp_io(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dtype_size=dtype_size,
                    include_intermediate=args.include_intermediate,
                )

                estimated_time_sec = estimate_time_from_io(
                    io_results["total_io_bytes"], bandwidth_bps
                )
                estimated_time_ms = estimated_time_sec * 1000

                all_results.append(
                    {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "io_results": io_results,
                        "estimated_time_ms": estimated_time_ms,
                    }
                )

                print(
                    f"[{current}/{total_configs}] "
                    f"bs={batch_size:4d}, seq_len={seq_len:5d}: "
                    f"IO={format_bytes(io_results['total_io_bytes']):>12s}, "
                    f"time={estimated_time_ms:8.4f} ms"
                )

    # 保存到 CSV（如果指定，或者有多个配置）
    if args.output_csv or len(all_results) > 1:
        import csv

        output_file = args.output_csv or "qwen3_mlp_io_estimates.csv"
        file_exists = os.path.exists(output_file)
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "batch_size",
                        "seq_len",
                        "model_size",
                        "hidden_size",
                        "intermediate_size",
                        "dtype",
                        "x_read_bytes",
                        "gate_up_write_bytes",
                        "intermediate_bytes",
                        "y_write_bytes",
                        "activation_io_bytes",
                        "gate_up_weight_read_bytes",
                        "down_weight_read_bytes",
                        "weight_io_bytes",
                        "total_io_bytes",
                        "estimated_time_ms",
                        "bandwidth_gbps",
                    ]
                )
            for result in all_results:
                writer.writerow(
                    [
                        result["batch_size"],
                        result["seq_len"],
                        args.model_size,
                        hidden_size,
                        intermediate_size,
                        args.dtype,
                        result["io_results"]["x_read_bytes"],
                        result["io_results"]["gate_up_write_bytes"],
                        result["io_results"]["intermediate_bytes"],
                        result["io_results"]["y_write_bytes"],
                        result["io_results"]["activation_io_bytes"],
                        result["io_results"]["gate_up_weight_read_bytes"],
                        result["io_results"]["down_weight_read_bytes"],
                        result["io_results"]["weight_io_bytes"],
                        result["io_results"]["total_io_bytes"],
                        result["estimated_time_ms"],
                        args.bandwidth_gbps,
                    ]
                )
        print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()


