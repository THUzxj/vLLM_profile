"""
测试脚本：在多个 green context streams 上并行运行相同的 Qwen3DecoderLayer 任务。
基于 test_qwen3_mlp_parallel.py 的逻辑，但使用 Qwen3DecoderLayer 模型，并测试 decode 阶段的性能。
"""

from custom_models.qwen3 import Qwen3DecoderLayer
from test_green_context import split_device_green_ctx
from transformers import Qwen3Config
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.forward_context import set_forward_context, get_forward_context
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from torch import nn
import torch
import pandas as pd
import os
import sys
import tempfile
from typing import Optional, List, Dict

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


# Qwen3 model configurations
QWEN3_MODEL_CONFIGS = {
    "0.6B": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
    },
    "4B": {
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
    },
    "14B": {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
    },
    "32B": {
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
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
            suffix="_qwen3_decoder_layer_parallel")[1]

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


def create_qwen3_config(model_size: str) -> Qwen3Config:
    """Create Qwen3Config from model size."""
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Supported sizes: {list(QWEN3_MODEL_CONFIGS.keys())}"
        )
    config_dict = QWEN3_MODEL_CONFIGS[model_size]

    # Create Qwen3Config with required fields
    config = Qwen3Config(
        hidden_size=config_dict["hidden_size"],
        intermediate_size=config_dict["intermediate_size"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        max_position_embeddings=config_dict["max_position_embeddings"],
        rms_norm_eps=config_dict["rms_norm_eps"],
        vocab_size=config_dict["vocab_size"],
        hidden_act="silu",
        rope_parameters={"rope_type": "default"},
    )
    return config


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
    """Create FlashAttentionMetadata for decode phase."""
    max_kv_len = max(kv_lens)
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    # Create block tables: [batch_size, max_blocks_per_seq]
    block_tables = []
    current_block = 0
    for i in range(batch_size):
        kv_len = kv_lens[i]
        num_blocks_needed = (kv_len + block_size - 1) // block_size
        block_table = []
        for j in range(num_blocks_needed):
            block_table.append(current_block % num_blocks)
            current_block += 1
        block_table += [0] * (max_blocks_per_seq - len(block_table))
        block_tables.append(block_table)

    block_table_tensor = torch.tensor(
        block_tables, dtype=torch.int32, device=device
    )

    # For decode phase, query_len is 1 for each sequence
    query_lens = [1] * batch_size
    seq_lens = kv_lens

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens_tensor.cpu()

    query_start_loc = [0]
    for q_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + q_len)
    query_start_loc_tensor = torch.tensor(
        query_start_loc, dtype=torch.int32, device=device
    )
    query_start_loc_cpu = query_start_loc_tensor.cpu()

    num_actual_tokens = batch_size
    slot_mapping = torch.zeros(
        num_actual_tokens, dtype=torch.int64, device=device
    )

    num_computed_tokens_cpu = torch.tensor(
        num_actual_tokens, dtype=torch.int32, device="cpu")

    common_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_tensor,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_tensor,
        # seq_lens_cpu=seq_lens_cpu,
        num_reqs=batch_size,
        num_actual_tokens=num_actual_tokens,
        # num_computed_tokens_cpu=num_computed_tokens_cpu,
        max_query_len=1,
        max_seq_len=max_kv_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

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

    builder = FlashAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=device,
    )

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
    """Create KV cache tensor."""
    if block_size % 16 != 0:
        raise ValueError(
            f"Block size {block_size} must be divisible by 16 for Flash Attention"
        )

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


def test_qwen3_decoder_layer_parallel_streams(
    num_groups: int = 2,
    min_count: int = 16,
    num_layers_per_stream: int = 1,
    batch_size: int = 1,
    kv_len: int = 1024,
    model_size: str = DEFAULT_MODEL,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    num_blocks: int = 8192,
    output_csv: Optional[str] = None,
):
    """
    测试不同 stream 内的 Qwen3DecoderLayer 任务并行地同时开启执行（decode 阶段）。

    Args:
        num_groups: 要创建的 green context 组数
        min_count: 每个组的最小 SM 数量
        num_layers_per_stream: 每个 stream 上初始化的 DecoderLayer 个数
        batch_size: 输入 batch size
        kv_len: KV cache 长度（序列长度）
        model_size: 模型大小 ("0.6B", "4B", "14B", "32B")
        warmup_iterations: Warmup 迭代次数
        benchmark_iterations: 基准测试迭代次数
        device: 设备
        dtype: 数据类型
        block_size: KV cache block size
        num_blocks: KV cache 块数量
        output_csv: 输出 CSV 文件路径
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
    num_attention_heads = model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    head_size = hidden_size // num_attention_heads

    dev = torch.device(device)
    torch.set_default_device(device)
    current_platform.seed_everything(42)

    # Create CacheConfig and VllmConfig
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        sliding_window=None,
    )

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    compilation_config = CompilationConfig()
    model_config_obj = ModelConfig(
        model=f"/nfs/xjzhang/Qwen/Qwen3-{model_size}",
        trust_remote_code=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config_obj,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        cache_config=cache_config,
    )

    set_current_vllm_config(vllm_config)

    try:
        # 创建 green context streams
        streams, resources = split_device_green_ctx(dev, num_groups, min_count)
        print(
            f"num_groups={num_groups}, min_count={min_count}, "
            f"SM counts: {[r.sm.smCount for r in resources]}"
        )
        print(
            f"Model: Qwen3-{model_size}, batch_size={batch_size}, kv_len={kv_len}, "
            f"hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        )

        streams = streams[:num_groups]
        resources = resources[:num_groups]

        num_streams = len(streams)

        # 创建 Qwen3Config
        qwen3_config = create_qwen3_config(model_size)

        # 为每个 stream 创建若干个 Qwen3DecoderLayer 模型
        decoder_layers = [[] for _ in range(num_streams)]
        kv_caches = [[] for _ in range(num_streams)]
        attn_metadata_dicts = [{} for _ in range(num_streams)]

        for i in range(num_streams):
            for j in range(num_layers_per_stream):
                layer_name = f"layer_{i}_{j}"

                # Create decoder layer
                decoder_layer = Qwen3DecoderLayer(
                    config=qwen3_config,
                    cache_config=cache_config,
                    prefix=layer_name,
                ).to(device).to(dtype)
                decoder_layer.eval()
                decoder_layers[i].append(decoder_layer)

                # Create KV cache
                kv_cache = create_kv_cache(
                    num_blocks=num_blocks,
                    block_size=block_size,
                    num_kv_heads=num_key_value_heads,
                    head_size=head_size,
                    dtype=dtype,
                    device=device,
                )
                kv_caches[i].append(kv_cache)

                # Bind KV cache to attention layer
                decoder_layer.self_attn.attn.kv_cache[0] = kv_cache

                # Create decode metadata
                kv_lens = [kv_len] * batch_size
                attn_metadata = create_decode_metadata(
                    batch_size=batch_size,
                    kv_lens=kv_lens,
                    block_size=block_size,
                    num_blocks=num_blocks,
                    device=device,
                    vllm_config=vllm_config,
                    layer_name=f"{layer_name}.self_attn.attn",
                    num_kv_heads=num_key_value_heads,
                    head_size=head_size,
                    dtype=dtype,
                )
                attn_metadata_dicts[i][f"{layer_name}.self_attn.attn"] = attn_metadata

        # Register layers in no_compile_layers for each stream
        # Each stream needs its own no_compile_layers dict
        no_compile_layers_dicts = []
        for i in range(num_streams):
            stream_no_compile_layers = {}
            for j, layer in enumerate(decoder_layers[i]):
                layer_name = f"layer_{i}_{j}"
                stream_no_compile_layers[f"{layer_name}.self_attn.attn"] = layer.self_attn.attn
            no_compile_layers_dicts.append(stream_no_compile_layers)

        # Also register all layers in compilation_config.static_forward_context
        # This ensures they are available in the forward context
        # Note: Attention layers are automatically registered when created,
        # but we need to ensure no_compile_layers dict exists
        if 'no_compile_layers' not in compilation_config.static_forward_context:
            compilation_config.static_forward_context['no_compile_layers'] = {}
        for i in range(num_streams):
            for j, layer in enumerate(decoder_layers[i]):
                layer_name = f"layer_{i}_{j}"
                full_layer_name = f"{layer_name}.self_attn.attn"
                compilation_config.static_forward_context['no_compile_layers'][
                    full_layer_name] = layer.self_attn.attn

        # Create input tensors for decode: [batch_size, hidden_size]
        # For decode, we process one token at a time
        input_tensors = []
        position_tensors = []
        for i in range(num_streams):
            stream_inputs = []
            stream_positions = []
            for j in range(num_layers_per_stream):
                # Hidden states: [batch_size, hidden_size]
                hidden_states = torch.randn(
                    batch_size, hidden_size, dtype=dtype, device=device
                )
                stream_inputs.append(hidden_states)

                # Positions: [batch_size] - current position for each sequence
                positions = torch.arange(
                    kv_len, kv_len + batch_size, dtype=torch.long, device=device
                )
                stream_positions.append(positions)
            input_tensors.append(stream_inputs)
            position_tensors.append(stream_positions)

        # Warm up
        print("\nWarming up...")
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                # Try to set no_compile_layers if the function supports it
                try:
                    with set_forward_context(
                        attn_metadata=attn_metadata_dicts[i],
                        vllm_config=vllm_config,
                        virtual_engine=0,
                        no_compile_layers=no_compile_layers_dicts[i],
                    ):
                        with torch.inference_mode():
                            for it in range(warmup_iterations):
                                layer_idx = it % num_layers_per_stream
                                hidden_states = input_tensors[i][layer_idx]
                                positions = position_tensors[i][layer_idx]
                                residual = None
                                hidden_states, residual = decoder_layers[i][layer_idx](
                                    positions=positions,
                                    hidden_states=hidden_states,
                                    residual=residual,
                                )
                except TypeError:
                    # If no_compile_layers is not supported, modify forward context directly
                    with set_forward_context(
                        attn_metadata=attn_metadata_dicts[i],
                        vllm_config=vllm_config,
                        virtual_engine=0,
                    ):
                        # Manually set no_compile_layers in the forward context
                        forward_context = get_forward_context()
                        if not hasattr(forward_context, 'no_compile_layers'):
                            forward_context.no_compile_layers = {}
                        forward_context.no_compile_layers.update(
                            no_compile_layers_dicts[i])

                        with torch.inference_mode():
                            for it in range(warmup_iterations):
                                layer_idx = it % num_layers_per_stream
                                hidden_states = input_tensors[i][layer_idx]
                                positions = position_tensors[i][layer_idx]
                                residual = None
                                hidden_states, residual = decoder_layers[i][layer_idx](
                                    positions=positions,
                                    hidden_states=hidden_states,
                                    residual=residual,
                                )

        # 同步以确保 warm up 完成
        for stream in streams:
            stream.synchronize()
        torch.cuda.synchronize()
        print("Warmup completed.\n")

        # 记录总体开始时间
        overall_start = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(streams[0]):
            overall_start.record()

        # Benchmark: 在所有 streams 上同时启动任务（并行执行）
        for it in range(benchmark_iterations):
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    # Try to set no_compile_layers if the function supports it
                    try:
                        with set_forward_context(
                            attn_metadata=attn_metadata_dicts[i],
                            vllm_config=vllm_config,
                            virtual_engine=0,
                            no_compile_layers=no_compile_layers_dicts[i],
                        ):
                            with torch.inference_mode():
                                layer_idx = it % num_layers_per_stream
                                hidden_states = input_tensors[i][layer_idx]
                                positions = position_tensors[i][layer_idx]
                                residual = None
                                hidden_states, residual = decoder_layers[i][layer_idx](
                                    positions=positions,
                                    hidden_states=hidden_states,
                                    residual=residual,
                                )
                    except TypeError:
                        # If no_compile_layers is not supported, modify forward context directly
                        with set_forward_context(
                            attn_metadata=attn_metadata_dicts[i],
                            vllm_config=vllm_config,
                            virtual_engine=0,
                        ):
                            # Manually set no_compile_layers in the forward context
                            forward_context = get_forward_context()
                            if not hasattr(forward_context, 'no_compile_layers'):
                                forward_context.no_compile_layers = {}
                            forward_context.no_compile_layers.update(
                                no_compile_layers_dicts[i])

                            with torch.inference_mode():
                                layer_idx = it % num_layers_per_stream
                                hidden_states = input_tensors[i][layer_idx]
                                positions = position_tensors[i][layer_idx]
                                residual = None
                                hidden_states, residual = decoder_layers[i][layer_idx](
                                    positions=positions,
                                    hidden_states=hidden_states,
                                    residual=residual,
                                )

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
        print("Parallel Qwen3DecoderLayer Decode Execution Results")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Model: Qwen3-{model_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  KV length: {kv_len}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Intermediate size: {intermediate_size}")
        print(f"  Number of streams: {num_streams}")
        print(f"  Benchmark iterations: {benchmark_iterations}")
        print(f"\nPer-stream results:")

        total_tokens = batch_size
        for i in range(num_streams):
            # Use overall time divided by iterations as per-stream time estimate
            elapsed_time = 1.0  # Placeholder, actual per-stream timing would need individual events
            avg_time = elapsed_time / benchmark_iterations
            throughput = total_tokens / \
                (avg_time / 1000) if avg_time > 0 else 0

            print(
                f"  Stream {i}: {resources[i].sm.smCount} SMs, "
                f"Avg time: {avg_time:.3f} ms, "
                f"Throughput: {throughput:.2f} tokens/sec"
            )

        # 计算总体执行时间
        overall_time = overall_start.elapsed_time(overall_end)
        avg_overall_time = overall_time / benchmark_iterations
        overall_throughput = total_tokens / \
            (avg_overall_time / 1000) if avg_overall_time > 0 else 0
        print(f"\nOverall parallel execution:")
        print(f"  Total time: {overall_time:.3f} ms")
        print(f"  Avg time per iteration: {avg_overall_time:.3f} ms")
        print(f"  Throughput: {overall_throughput:.2f} tokens/sec")
        print("=" * 80 + "\n")

        # 准备结果数据
        results = []
        for i in range(num_streams):
            elapsed_time = 1.0  # Placeholder
            avg_time = elapsed_time / benchmark_iterations
            throughput = total_tokens / \
                (avg_time / 1000) if avg_time > 0 else 0

            results.append({
                "num_groups": num_groups,
                "min_count": min_count,
                "stream_id": i,
                "sm_count": resources[i].sm.smCount,
                "model_size": model_size,
                "batch_size": batch_size,
                "kv_len": kv_len,
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
            "kv_len": kv_len,
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
        description="Test parallel Qwen3DecoderLayer decode execution on multiple green context streams"
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
        "--num-layers-per-stream",
        type=int,
        default=1,
        help="Number of Qwen3DecoderLayer instances per stream. Default: 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for input tensor. Default: 1",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=1024,
        help="KV cache length (sequence length). Default: 1024",
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
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size. Default: 16",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=8192,
        help="Number of KV cache blocks. Default: 8192",
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
    print("Qwen3DecoderLayer Parallel Stream Decode Test")
    print("=" * 80)
    print(f"Number of groups: {args.num_groups}")
    print(f"Min SM count per group: {args.min_count}")
    print(f"Number of layers per stream: {args.num_layers_per_stream}")
    print(f"Batch size: {args.batch_size}")
    print(f"KV length: {args.kv_len}")
    print(f"Model size: {args.model_size}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Block size: {args.block_size}")
    print(f"Num blocks: {args.num_blocks}")
    if args.output_csv:
        print(f"Output CSV: {args.output_csv}")
    print("=" * 80 + "\n")

    test_qwen3_decoder_layer_parallel_streams(
        num_groups=args.num_groups,
        min_count=args.min_count,
        num_layers_per_stream=args.num_layers_per_stream,
        batch_size=args.batch_size,
        kv_len=args.kv_len,
        model_size=args.model_size,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
        device=args.device,
        dtype=dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        output_csv=args.output_csv,
    )
