# borrowed from flashinfer bandwidth profiler
# adapted for flash attention varlen function

import argparse
import torch
import random
import time
from typing import Optional
from pathlib import Path
from typing import Any

from torch import multiprocessing
from multiprocessing.managers import DictProxy
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.math_utils import cdiv
from vllm.utils.system_utils import update_environment_variables
from vllm.config import ModelConfig, ParallelConfig
from vllm.platforms import current_platform
from vllm.attention.utils.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

import csv
import os


def save_to_csv(
    items: list[dict[str, Any]],
    headers: list[str],
    output_path: str,
):
    output_dir = Path(output_path).parent
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        w = csv.DictWriter(f, headers)
        w.writeheader()
        [w.writerow(item) for item in items]


def get_token_size(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    kv_dtype: torch.dtype,
    num_layers: int = 1
):
    """
    Calculate the KV cache size per token in MB.
    If num_layers is provided, use it; otherwise get from model_config.
    """
    if num_layers is None:
        num_layers = model_config.get_num_layers(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_dim = model_config.get_head_size()
    # KV cache: K and V, each is num_kv_heads * head_dim
    token_size = num_layers * 2 * num_kv_heads * \
        head_dim * kv_dtype.itemsize  # bytes
    token_size /= 1024 * 1024  # convert to MB
    return token_size


def calculate_data_transfer(
    batch_size: int,
    kv_lens: list[int],
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> float:
    """
    Calculate the total data transfer in MB for flash attention operation.

    Data transferred includes:
    1. Query: batch_size * num_query_heads * head_dim
    2. Key cache: sum(kv_lens) * num_kv_heads * head_dim
    3. Value cache: sum(kv_lens) * num_kv_heads * head_dim
    4. Output: batch_size * num_query_heads * head_dim

    Returns:
        Total data transfer in MB
    """
    dtype_size = dtype.itemsize

    # Query data
    query_size = batch_size * num_query_heads * head_dim * dtype_size

    # KV cache data (read)
    total_kv_len = sum(kv_lens)
    key_size = total_kv_len * num_kv_heads * head_dim * dtype_size
    value_size = total_kv_len * num_kv_heads * head_dim * dtype_size

    # Output data
    output_size = batch_size * num_query_heads * head_dim * dtype_size

    # Total: read (Q + K + V) + write (O)
    total_size = query_size + key_size + value_size + output_size
    total_size_mb = total_size / (1024 * 1024)

    return total_size_mb


def profile_bandwidth(
    model: str,
    batch_size: int,
    avg_seq_len: int,
    tensor_parallel_size: int = 1,
    num_layers: int = 1,
    fa_version: Optional[int] = None,
):
    """
    Profile memory bandwidth for flash attention varlen function.

    Args:
        model: Model name or path
        batch_size: Batch size
        avg_seq_len: Average sequence length (all sequences use this length)
        tensor_parallel_size: Tensor parallel size
        num_layers: Number of layers to simulate (for calculating token size)
        fa_version: Flash attention version (None for auto-detect)

    Returns:
        Bandwidth in GB/s
    """
    # Check if flash attention is available
    if not is_flash_attn_varlen_func_available():
        raise RuntimeError(
            "Flash attention varlen function is not available on this platform")

    # Get FA version if not specified
    if fa_version is None:
        fa_version = get_flash_attn_version()
        if fa_version is None:
            raise RuntimeError(
                "Flash attention version could not be determined")

    engine_args = EngineArgs(
        model, load_format="dummy", tensor_parallel_size=tensor_parallel_size
    )
    config = engine_args.create_engine_config()

    num_query_heads = config.model_config.get_num_attention_heads(
        config.parallel_config)
    num_kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_dim = config.model_config.get_head_size()
    block_size = config.cache_config.block_size
    dtype = torch.bfloat16  # Use bfloat16 for consistency

    # Generate uniform sequence lengths
    seq_lens = [avg_seq_len] * batch_size
    print(f"DEBUG: seq_lens = {seq_lens[:5]}... (total: {len(seq_lens)})")
    print(
        f"DEBUG: sum(seq_lens) = {sum(seq_lens)}, avg = {sum(seq_lens) / len(seq_lens)}")

    # Setup flash attention parameters
    num_seqs = batch_size
    max_kv_len = max(seq_lens)
    scale = head_dim ** -0.5

    # Create tensors
    # Query shape: (num_tokens, num_query_heads, head_dim)
    # For decode, each sequence has query_len=1
    query_len = 1
    num_tokens = num_seqs * query_len
    query = torch.randn(num_tokens, num_query_heads,
                        head_dim, dtype=dtype, device="cuda")

    # KV cache shape: (num_blocks, block_size, num_kv_heads, head_dim)
    block_lens = [cdiv(seq_len, block_size) for seq_len in seq_lens]
    max_num_pages = sum(block_lens)
    num_blocks = max(max_num_pages, 8192)  # Ensure enough blocks

    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda"
    )
    value_cache = torch.randn_like(key_cache)

    # Create cu_seqlens_q: cumulative sequence lengths for queries
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")

    # seqused_k: KV cache sequence lengths
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda"
    )

    out = torch.empty_like(query)

    # Warmup
    for _ in range(3):
        _ = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            seqused_k=seqused_k,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=None,
            block_table=block_tables,
            softcap=0,
            fa_version=fa_version,
        )

    torch.cuda.synchronize()

    # Wait for GPU to stabilize
    torch.cuda._sleep(int(1e7))

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):  # Run multiple iterations for better accuracy
        _ = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            seqused_k=seqused_k,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=None,
            block_table=block_tables,
            softcap=0,
            fa_version=fa_version,
        )
    end_event.record()
    torch.cuda.synchronize()

    attn_time = start_event.elapsed_time(
        end_event)  # ms (total for 100 iterations)
    attn_time_per_iter = attn_time / 100.0  # ms per iteration

    # Calculate data transfer
    data_transfer_mb = calculate_data_transfer(
        batch_size=batch_size,
        kv_lens=seq_lens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # Calculate bandwidth: data_transfer (MB) / time (ms) = GB/s
    # Note: 1 GB = 1000 MB, so MB/ms = (MB/ms) * (1000 ms/s) / (1000 MB/GB) = GB/s
    bandwidth = data_transfer_mb / attn_time_per_iter  # GB/s

    print(f"DEBUG: data_transfer = {data_transfer_mb:.2f} MB")
    print(f"DEBUG: attn_time = {attn_time_per_iter:.4f} ms")
    print(f"DEBUG: bandwidth = {bandwidth:.2f} GB/s")

    return bandwidth


def worker(
    model: str,
    batch_size: int,
    avg_seq_len: int,
    sm_pct: int,
    bandwidths: DictProxy,
    tensor_parallel_size: int = 1,
    num_layers: int = 1,
    fa_version: Optional[int] = None,
):
    """Worker function for profiling bandwidth with specific SM percentage."""
    print(f"profiling bandwidth using {sm_pct}% SM")
    try:
        bandwidth = profile_bandwidth(
            model,
            batch_size,
            avg_seq_len,
            tensor_parallel_size=tensor_parallel_size,
            num_layers=num_layers,
            fa_version=fa_version,
        )
        bandwidths[sm_pct] = bandwidth
    except Exception as e:
        print(f"Error profiling with {sm_pct}% SM: {e}")
        bandwidths[sm_pct] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile memory bandwidth for flash attention varlen function"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--avg-seq-len", type=int, required=True,
                        help="Average sequence length")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Batch size")
    parser.add_argument("--sm-pcts", type=int, nargs="+", default=[100],
                        help="List of SM percentages to test")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of layers (for token size calculation)")
    parser.add_argument("--fa-version", type=int, default=None,
                        help="Flash attention version (None for auto-detect)")
    parser.add_argument("--output-file", type=str,
                        default="flash_attn_bandwidth.csv")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model: str = args.model
    batch_size: int = args.batch_size
    avg_seq_len: int = args.avg_seq_len
    sm_pcts: list[int] = args.sm_pcts
    output_file: str = args.output_file
    tensor_parallel_size: int = args.tensor_parallel_size
    num_layers: int = args.num_layers
    fa_version: Optional[int] = args.fa_version

    manager = multiprocessing.Manager()
    bandwidths = manager.dict()
    ctx = multiprocessing.get_context("spawn")

    for sm_pct in sm_pcts:
        process_args = (
            model,
            batch_size,
            avg_seq_len,
            sm_pct,
            bandwidths,
            tensor_parallel_size,
            num_layers,
            fa_version,
        )
        p = ctx.Process(target=worker, args=process_args)
        env_dicts = {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(sm_pct)}
        update_environment_variables(env_dicts)
        p.start()
        p.join()

    # Filter out None values
    valid_results = {
        sm_pct: bandwidth
        for sm_pct, bandwidth in bandwidths.items()
        if bandwidth is not None
    }

    save_to_csv(
        items=[
            {"sm_pct": sm_pct, "bandwidth": bandwidth}
            for sm_pct, bandwidth in valid_results.items()
        ],
        headers=["sm_pct", "bandwidth"],
        output_path=output_file,
    )

    print(f"\nResults saved to {output_file}")
    print("\nBandwidth Results:")
    for sm_pct, bandwidth in sorted(valid_results.items()):
        print(f"  {sm_pct}% SM: {bandwidth:.2f} GB/s")
