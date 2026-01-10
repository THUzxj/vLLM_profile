# borrowed from flashinfer
# https://docs.flashinfer.ai/api/decode.html#batch-decoding

import argparse
import torch
import flashinfer
import random
from typing import Optional


from torch import multiprocessing
from multiprocessing.managers import DictProxy
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.math_utils import cdiv
from vllm.utils.system_utils import update_environment_variables
from vllm.config import ModelConfig, ParallelConfig

import csv
import os

from pathlib import Path
from typing import Any


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


def get_token_size(model_config: ModelConfig, parallel_config: ParallelConfig, kv_dtype: torch.dtype):
    num_layers = model_config.get_num_layers(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_dim = model_config.get_head_size()
    token_size = num_layers * 2 * num_kv_heads * \
        head_dim * kv_dtype.itemsize  # bytes
    token_size /= 1024 * 1024  # convert to MB
    return token_size


def _adjust_random_nums_by_max(nums: list[int], max_num: int):
    exceed_len = sum([max(seq_len - max_num, 0) for seq_len in nums])
    for i in range(len(nums)):
        if nums[i] > max_num:
            nums[i] = max_num

        if exceed_len > 0 and nums[i] < max_num:
            delta = min(max_num - nums[i], exceed_len)
            nums[i] += delta
            exceed_len -= delta

    return nums


def _create_random_intergers_by_sum(
    sum_value: int, num: int, start: int = 1
) -> list[int]:
    """
    generate (num - 1) points from range [0, sum_value - start * num],
    so the mean value of intervals between adjacent points is (sum_value / num - start).
    And we add start later to get list with mean value of avg_value.
    """
    if num == 0:
        return []
    partition_length = sum_value - start * num
    partitions = [random.randint(0, partition_length) for _ in range(num - 1)]
    partitions.sort()
    partitions = [0] + partitions + [partition_length]
    samples = [partitions[i + 1] - partitions[i]
               for i in range(len(partitions) - 1)]
    nums = [sample + start for sample in samples]
    return nums


def _create_random_integers_by_max(num_ints: int, max_int: int, min_int: int = 1) -> list[int]:
    assert max_int > 0 and num_ints > 0
    seq_lens = [random.randint(min_int, max_int) for _ in range(num_ints)]
    seq_lens[-1] = max_int
    return seq_lens


def _create_random_integers_by_avg(
    avg_seq_len: int, batch: int, max_seq_len: Optional[int] = None
) -> list[int]:
    seq_lens = _create_random_intergers_by_sum(avg_seq_len * batch, batch)
    # generate random intergers by average only ensure the average value,
    # i.e. the sum of intergers is (avg_seq_len x batch), however,
    # there may be some intergers larger then max_seq_len, there we need to
    # adjust seq_lens by max_seq_len
    if max_seq_len is not None:
        if isinstance(max_seq_len, int):
            assert avg_seq_len <= max_seq_len
            seq_lens = _adjust_random_nums_by_max(seq_lens, max_seq_len)

    return seq_lens


def create_seq_lens(
    batch: int,
    avg_seq_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    config: Optional[VllmConfig] = None,
) -> list[int]:
    if max_seq_len is None:
        assert config is not None
        max_seq_len = config.model_config.max_model_len
    if avg_seq_len is None:
        seq_lens = _create_random_integers_by_max(batch, max_seq_len)
    else:
        seq_lens = _create_random_integers_by_avg(
            avg_seq_len, batch, max_seq_len)
    return seq_lens


def profile_bandwidth(model: str, batch_size: int, avg_seq_len: int, tensor_parallel_size: int = 1):
    engine_args = EngineArgs(
        model, load_format="dummy", tensor_parallel_size=tensor_parallel_size
    )
    config = engine_args.create_engine_config()

    num_layers = config.model_config.get_num_layers(config.parallel_config)
    num_qo_heads = config.model_config.get_num_attention_heads(
        config.parallel_config)
    num_kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_dim = config.model_config.get_head_size()
    block_size = config.cache_config.block_size

    # Generate uniform sequence lengths to avoid flashinfer group_size issues
    # Use all the same length for simplicity and compatibility
    seq_lens = [avg_seq_len] * batch_size
    print(f"DEBUG: seq_lens = {seq_lens[:5]}... (total: {len(seq_lens)})")
    print(
        f"DEBUG: sum(seq_lens) = {sum(seq_lens)}, avg = {sum(seq_lens) / len(seq_lens)}")
    block_lens = [cdiv(seq_len, block_size) for seq_len in seq_lens]

    max_num_pages = sum(block_lens)
    prefix_block_lens = [sum(block_lens[:i])
                         for i in range(len(block_lens) + 1)]
    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    kv_page_indices = torch.arange(
        max_num_pages, dtype=torch.int32, device="cuda")
    kv_page_indptr = torch.tensor(
        prefix_block_lens, dtype=torch.int32, device="cuda"
    )
    # 1 <= kv_last_page_len <= page_size
    kv_last_page_len = torch.randint(
        1, block_size + 1, (batch_size,), dtype=torch.int32, device="cuda"
    )
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16
    )
    kv_cache_for_layers = [
        torch.randn(
            max_num_pages,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        for _ in range(num_layers)
    ]
    q_for_layers = [
        torch.randn(batch_size, num_qo_heads, head_dim,
                    dtype=torch.half, device="cuda")
        for _ in range(num_layers)
    ]
    for q, kv_cache in zip(q_for_layers[-3:], kv_cache_for_layers[-3:]):
        o = decode_wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(int(1e7))
    start.record()
    for q, kv_cache in zip(q_for_layers, kv_cache_for_layers):
        o = decode_wrapper.run(q, kv_cache)
    end.record()
    torch.cuda.synchronize()
    attn_time = start.elapsed_time(end)  # ms
    token_size = get_token_size(
        config.model_config, config.parallel_config, torch.float16)
    size = avg_seq_len * batch_size * token_size  # MB
    bandwidth = size / attn_time  # GB/s
    return bandwidth


def worker(model: str, batch_size: int, avg_seq_len: int, sm_pct: int, bandwidths: DictProxy):
    print(f"profiling bandwidth using {sm_pct}% SM")
    bandwidth = profile_bandwidth(model, batch_size, avg_seq_len)
    bandwidths[sm_pct] = bandwidth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--avg-seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sm-pcts", type=int, nargs="+", default=[100])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-file", type=str,
                        default="attention_bandwidth.csv")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model: str = args.model
    batch_size: int = args.batch_size
    avg_seq_len: int = args.avg_seq_len
    sm_pcts: list[int] = args.sm_pcts
    output_file: str = args.output_file
    manager = multiprocessing.Manager()
    bandwidths = manager.dict()
    ctx = multiprocessing.get_context("spawn")
    for sm_pct in sm_pcts:
        process_args = (model, batch_size, avg_seq_len, sm_pct, bandwidths)
        p = ctx.Process(target=worker, args=process_args)
        env_dicts = {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(sm_pct)}
        update_environment_variables(env_dicts)
        p.start()
        p.join()

    save_to_csv(
        items=[
            {"sm_pct": sm_pct, "bandwidth": bandwidth}
            for sm_pct, bandwidth in bandwidths.items()
        ],
        headers=["sm_pct", "bandwidth"],
        output_path=output_file,
    )
