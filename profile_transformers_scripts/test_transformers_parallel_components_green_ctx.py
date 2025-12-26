#!/usr/bin/env python3
"""
Parallel Transformers component (Attention & FFN layers) benchmarking on green context streams.

This script:
1. Extracts attention and FFN layers from a Transformers model
2. Splits GPU resources using CUDA Green Contexts into multiple independent streams
3. Distributes components across streams and runs them in parallel
4. Measures per-component latency on each stream with different GPU resource allocations
5. Records results with component name, stream info, batch size, sequence length, latency

Key features:
- Each stream has its own green context with allocated SMs
- Components run independently on separate streams (potentially in parallel)
- Per-component and per-stream latency measurement
- CSV output with incremental per-row appending
- Supports both local modeling_qwen3_2 and standard AutoModel loading
"""

import argparse
import contextlib
import gc
import os
import sys
import threading
import time
from typing import List, Tuple, Dict, Callable, Optional
from random import randint
from pathlib import Path

import cuda.bindings.driver as driver
import cuda.bindings.runtime as runtime
import cuda.cudart as cudart
import cuda.nvrtc as nvrtc
import torch
import numpy as np
import pandas as pd
import inspect
from cuda.bindings.driver import CUdevice, CUdevResource
from transformers import AutoModel, AutoTokenizer

# Add script directory to path for local modeling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import modeling_qwen3_2 as local_qwen
except ImportError:
    local_qwen = None


class BenchmarkCache:
    """Minimal cache object implementing the update API expected by HF attention layers."""

    def __init__(self, past_key: Optional[torch.Tensor], past_value: Optional[torch.Tensor]):
        self.past_key = past_key
        self.past_value = past_value

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.past_key is not None and self.past_value is not None:
            key_states = torch.cat([self.past_key, key_states], dim=2)
            value_states = torch.cat([self.past_value, value_states], dim=2)
        return key_states, value_states

    def __iter__(self):
        yield self.past_key
        yield self.past_value

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0:
            return self.past_key
        if idx == 1:
            return self.past_value
        raise IndexError(idx)


def get_embeddings_from_token_ids(
    model,
    tokenizer,
    batch_size: int,
    seq_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    sample_text: str = None,
):
    """Obtain input embeddings by tokenizing text or sampling token ids and
    passing them through the model's embedding layer.

    Returns a tensor shaped (batch_size, seq_length, hidden_dim) on `device`.
    """
    # Build input_ids via tokenizer if sample_text provided, else sample random ids
    if sample_text:
        toks = tokenizer(
            sample_text,
            return_tensors="pt",
            padding="max_length",
            max_length=seq_length,
            truncation=True,
        )
        input_ids = toks["input_ids"].to(device)
        if input_ids.size(1) != seq_length:
            input_ids = input_ids[:, :seq_length]
        if input_ids.size(0) == 1 and batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
        elif input_ids.size(0) != batch_size:
            # repeat or truncate to match batch_size
            input_ids = input_ids.repeat(
                batch_size // input_ids.size(0) + 1, 1)[:batch_size]
    else:
        # try to determine vocab size
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = tokenizer.model_maximum_features if hasattr(
                    tokenizer, 'model_maximum_features') else None
            except Exception:
                vocab_size = None
        if not vocab_size or vocab_size <= 0:
            vocab_size = 50257
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_length), device=device)

    # Get embedding layer
    emb_layer = None
    if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
        emb_layer = model.get_input_embeddings()
    elif hasattr(model, 'embed_tokens'):
        emb_layer = model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        emb_layer = model.model.embed_tokens

    if emb_layer is None:
        raise RuntimeError(
            "Embedding layer not found on model; cannot produce input embeddings")

    # Ensure input_ids on same device as embedding layer
    input_ids = input_ids.to(next(emb_layer.parameters()).device if any(
        True for _ in emb_layer.parameters()) else device)

    with torch.no_grad():
        embeddings = emb_layer(input_ids)

    return embeddings.to(device=device, dtype=dtype)


def cleanup():
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, runtime.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def get_cudevice(dev: torch.device) -> CUdevice:
    try:
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    except RuntimeError as e:
        runtime.cudaInitDevice(dev.index, 0, 0)
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    return cu_dev


def get_device_resource(cu_dev: CUdevice) -> CUdevResource:
    return checkCudaErrors(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )


def split_resource(
    resource: CUdevResource,
    num_groups: int,
    min_count: int,
) -> Tuple[CUdevResource, CUdevResource]:
    results, _, remaining = checkCudaErrors(
        driver.cuDevSmResourceSplitByCount(
            num_groups,
            resource,
            0,  # useFlags
            min_count,
        )
    )
    return results, remaining


def create_green_ctx_streams(
    cu_dev: CUdevResource, resources: List[CUdevResource]
) -> List[torch.Stream]:
    streams = []
    for split in resources:
        desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([split], 1))
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
        streams.append(torch.cuda.get_stream_from_external(stream))

    return streams


def split_device_green_ctx(
    dev: torch.device, num_groups: int, min_count: int
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    """
    Split the device into multiple green contexts.

    Returns:
        streams: List of torch.Streams for each group (including remaining)
        resources: List of CUdevResource for each group (including remaining)
    """
    cu_dev = get_cudevice(dev)
    resource = get_device_resource(cu_dev)
    results, remaining = split_resource(resource, num_groups, min_count)
    resources = results + [remaining]
    streams = create_green_ctx_streams(cu_dev, resources)
    return streams, resources


def extract_attention_and_ffn_layers(model) -> Dict[str, Callable]:
    """
    Extract attention and FFN layers from the model.
    Returns a dict mapping layer names to callable modules.
    """
    components: Dict[str, Callable] = {}

    # Try common, explicit extraction first (GPT-like, BERT-like, Qwen/LLaMA-like)
    try:
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            for idx, layer in enumerate(model.transformer.h):
                if hasattr(layer, 'attn'):
                    components[f'layer_{idx}_attn'] = layer.attn

                if hasattr(layer, 'mlp'):
                    components[f'layer_{idx}_ffn'] = layer.mlp

        if not components and hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            for idx, layer in enumerate(model.encoder.layer):
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                    components[f'layer_{idx}_attn'] = layer.attention.self
                if hasattr(layer, 'intermediate') and hasattr(layer, 'output'):
                    components[f'layer_{idx}_ffn'] = torch.nn.Sequential(
                        layer.intermediate,
                        layer.output
                    )

        if not components and hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for idx, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn'):
                    components[f'layer_{idx}_attn'] = layer.self_attn
                if hasattr(layer, 'mlp'):
                    components[f'layer_{idx}_ffn'] = layer.mlp
    except Exception:
        # Fall through to generic scan
        pass

    # If explicit extraction found nothing, perform a generic scan of named modules
    if not components:
        for name, module in model.named_modules():
            # skip the top-level module
            if name == "":
                continue

            lname = name.replace('.', '_')

            # Heuristics for attention modules: class name contains 'Attention' or module has q/k/v proj
            cls_name = module.__class__.__name__.lower()
            has_qkv = any(hasattr(module, attr) for attr in (
                'q_proj', 'k_proj', 'v_proj', 'qkv_proj', 'q', 'k', 'v'))
            if 'attention' in cls_name or 'selfattn' in cls_name or 'multihead' in cls_name or has_qkv:
                key = f"{lname}_attn"
                components[key] = module
                continue

            # Heuristics for FFN modules: class name contains 'mlp'/'feed'/'ffn' or sequential of linear+gelu
            if 'mlp' in cls_name or 'ffn' in cls_name or 'feed' in cls_name or 'dense' in cls_name:
                key = f"{lname}_ffn"
                components[key] = module
                continue

            # Detect sequential blocks with Linear layers -> potential FFN
            if isinstance(module, torch.nn.Sequential):
                sub_cls = ' '.join([m.__class__.__name__.lower()
                                   for m in module])
                if 'linear' in sub_cls and ('gelu' in sub_cls or 'relu' in sub_cls or 'silu' in sub_cls):
                    key = f"{lname}_ffn"
                    components[key] = module

    return components


def print_available_components(components: Dict[str, Callable]):
    """
    Print all available components in a formatted table.
    """
    if not components:
        print("No components found!")
        return

    print(f"\n{'='*70}")
    print(f"AVAILABLE COMPONENTS ({len(components)} total)")
    print(f"{'='*70}")
    print(f"{'Component Name':<30} {'Type':<15} {'Layer Index':<15}")
    print(f"{'-'*70}")

    # Sort components by layer index for better readability
    sorted_components = sorted(components.keys(), key=lambda x: (
        int(x.split('_')[1]) if len(x.split('_')) > 1 and x.split(
            '_')[1].isdigit() else 999,
        'attn' in x  # attention layers first
    ))

    for comp_name in sorted_components:
        if 'attn' in comp_name:
            comp_type = 'Attention'
        elif 'ffn' in comp_name:
            comp_type = 'FFN'
        else:
            comp_type = 'Unknown'

        layer_idx = comp_name.split('_')[1] if '_' in comp_name else '-1'

        print(f"{comp_name:<30} {comp_type:<15} {layer_idx:<15}")

    print(f"{'='*70}\n")


def _make_defaults(bsz, seq_len, hidden_dim, dtype, device, config):
    num_heads = getattr(config, 'num_attention_heads',
                        8) if config is not None else 8
    num_kv_heads = getattr(config, 'num_key_value_heads',
                           num_heads) if config is not None else num_heads
    head_dim = getattr(config, 'head_dim', None)
    if not head_dim:
        head_dim = max(hidden_dim // num_heads, 1)

    cos = torch.ones(1, seq_len, head_dim, dtype=dtype, device=device)
    sin = torch.zeros(1, seq_len, head_dim, dtype=dtype, device=device)
    pos_emb_rope = (cos, sin)

    pos_emb_single = torch.zeros(
        bsz, seq_len, hidden_dim, dtype=dtype, device=device)

    past_seq_len = seq_len
    total_kv_len = seq_len + past_seq_len
    mask_dtype = torch.float32 if dtype not in (
        torch.float32, torch.float64) else dtype
    causal_mask = torch.full(
        (seq_len, total_kv_len),
        torch.finfo(mask_dtype).min,
        dtype=mask_dtype,
        device=device,
    )
    row_ids = torch.arange(seq_len, device=device).unsqueeze(1)
    col_ids = torch.arange(total_kv_len, device=device).unsqueeze(0)
    valid = col_ids <= (past_seq_len + row_ids)
    causal_mask = torch.where(valid, torch.zeros(
        1, device=device, dtype=mask_dtype), causal_mask)
    attn_mask = causal_mask.unsqueeze(0).unsqueeze(
        0).expand(bsz, 1, seq_len, total_kv_len).to(dtype)

    past_key = torch.zeros(
        bsz, num_kv_heads, past_seq_len, head_dim, dtype=dtype, device=device)
    past_value = torch.zeros_like(past_key)
    cache = BenchmarkCache(past_key, past_value)
    return pos_emb_rope, pos_emb_single, attn_mask, cache


def prepare_component_args(module: torch.nn.Module, hidden: torch.Tensor, model_config=None, component_name: str = None):
    """Prepare arguments for module forward call.

    Determines component type from component_name suffix (_attn or _ffn).

    Args:
        module: The module to prepare arguments for
        hidden: Input hidden states tensor
        model_config: Optional model config
        component_name: Component name (e.g., 'layer_0_attn' or 'layer_0_ffn')

    Returns:
        tuple: (args, kwargs, call_func) where call_func is a function that calls the module
    """
    # Determine component type from component_name suffix
    is_attention = False
    is_ffn = False

    if component_name:
        if component_name.endswith('_attn') or '_attn_' in component_name:
            is_attention = True
        elif component_name.endswith('_ffn') or '_ffn_' in component_name or 'mlp' in component_name.lower():
            is_ffn = True

    # If component_name not provided, try to infer from module class name
    if not is_attention and not is_ffn:
        class_name_lower = module.__class__.__name__.lower()
        if 'attention' in class_name_lower or 'attn' in class_name_lower:
            is_attention = True
        elif 'mlp' in class_name_lower or 'ffn' in class_name_lower:
            is_ffn = True

    sig = None
    try:
        sig = inspect.signature(module.forward)
    except (ValueError, TypeError):
        try:
            sig = inspect.signature(module)
        except Exception:
            sig = None

    bsz, seq_len, hidden_dim = hidden.shape

    # For FFN modules, use simple calling pattern (only input tensor)
    if is_ffn:
        if sig is not None:
            params = list(sig.parameters.keys())
            # FFN typically only needs the input tensor
            if 'x' in params:
                def call_func():
                    return module(hidden)
            elif 'hidden_states' in params:
                def call_func():
                    return module(hidden_states=hidden)
            elif 'input' in params:
                def call_func():
                    return module(input=hidden)
            else:
                def call_func():
                    return module(hidden)
        else:
            def call_func():
                return module(hidden)
        return ([], {}, call_func)

    # For attention modules, prepare all required arguments
    if is_attention:
        # Prepare all inputs (this is done outside timing)
        pos_emb_rope, pos_emb_single, attn_mask, past_key_values = _make_defaults(
            bsz, seq_len, hidden_dim, hidden.dtype, hidden.device, model_config
        )

        if sig is not None:
            params = list(sig.parameters.keys())
            kwargs = {}
            args = []

            # Build arguments based on parameter names
            if 'hidden_states' in params:
                args.append(hidden)
            elif 'x' in params:
                args.append(hidden)
            elif 'input' in params:
                args.append(hidden)
            else:
                args.append(hidden)  # Default: first positional argument

            # Use keyword arguments for optional parameters to avoid ordering issues
            if 'position_embeddings' in params:
                param_obj = sig.parameters['position_embeddings']
                if param_obj.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwargs['position_embeddings'] = pos_emb_rope
                else:
                    args.append(pos_emb_rope)

            if 'attention_mask' in params:
                param_obj = sig.parameters['attention_mask']
                if param_obj.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwargs['attention_mask'] = attn_mask
                else:
                    args.append(attn_mask)

            if 'return_dict' in params:
                kwargs['return_dict'] = False

            if 'past_key_values' in params:
                kwargs['past_key_values'] = past_key_values
            if 'past_key_value' in params:
                kwargs['past_key_value'] = past_key_values
            if 'use_cache' in params:
                kwargs['use_cache'] = True
            if 'cache_position' in params:
                cache_position = torch.arange(
                    seq_len, device=hidden.device, dtype=torch.long)
                kwargs['cache_position'] = cache_position

            # Create call function that only calls forward
            def call_func():
                return module(*args, **kwargs)

            return (args, kwargs, call_func)
        else:
            # No signature available, use common pattern for attention
            pos_emb_rope, pos_emb_single, attn_mask, past_key_values = _make_defaults(
                bsz, seq_len, hidden_dim, hidden.dtype, hidden.device, model_config
            )

            def call_func():
                return module(hidden, position_embeddings=pos_emb_rope, attention_mask=attn_mask)
            return ([], {}, call_func)

    # Unknown component type - raise error instead of fallback
    raise RuntimeError(
        f"Cannot determine component type for '{component_name}'. "
        f"Component name should end with '_attn' or '_ffn', or module class should contain 'attention'/'attn' or 'mlp'/'ffn'."
    )


def invoke_component(module: torch.nn.Module, hidden: torch.Tensor, model_config=None, component_name: str = None):
    """Call the module with sensible defaults depending on its signature.

    This function is kept for backward compatibility and warmup.
    For timing measurements, use prepare_component_args + call_func instead.
    """
    _, _, call_func = prepare_component_args(
        module, hidden, model_config, component_name)
    return call_func()


def worker_run_components_on_stream(
    components: Dict[str, Callable],
    stream_idx: int,
    stream: torch.cuda.Stream,
    sm_count: int,
    sm_partition_count: int,
    batch_sizes: List[int],
    seq_lengths: List[int],
    hidden_dim: int,
    device: torch.device,
    model_config,
    num_repeats: int,
    warmup: int,
    tokenizer,
    model,
    results: List[dict],
    lock: threading.Lock,
    output_path: str,
    barrier: threading.Barrier = None,
):
    """
    Worker thread that runs components on a dedicated green context stream.

    Each worker:
    - Uses a specific stream with allocated SMs
    - Runs all components sequentially on that stream
    - Measures per-component latency
    - Appends results to CSV (incremental per-row)
    """
    try:
        print(f"[worker {stream_idx}] starting on stream with {sm_count} SMs")
        print(
            f"[worker {stream_idx}] batch_sizes={batch_sizes}, seq_lengths={seq_lengths}, hidden_dim={hidden_dim}")

        with torch.cuda.stream(stream):
            # Prepare warmup on the first batch/seq combination
            warmup_bsz = batch_sizes[0]
            warmup_seqlen = seq_lengths[0]
            if tokenizer is not None:
                hidden_states = get_embeddings_from_token_ids(
                    model,
                    tokenizer,
                    warmup_bsz,
                    warmup_seqlen,
                    device,
                    dtype=torch.bfloat16,
                )
            else:
                hidden_states = torch.randn(
                    warmup_bsz, warmup_seqlen, hidden_dim,
                    dtype=torch.bfloat16,
                    device=device
                )

            # Warmup for this stream (first component only)
            print(f"[worker {stream_idx}] warmup ({warmup} iterations)")
            with torch.no_grad():
                for comp_name, component in list(components.items())[:1]:
                    for _ in range(warmup):
                        try:
                            _ = invoke_component(
                                component, hidden_states, model_config, comp_name)
                        except Exception as e:
                            print(
                                f"[worker {stream_idx}] warmup error on {comp_name}: {e}")

            torch.cuda.synchronize()

            # Synchronize all workers before benchmark starts
            if barrier is not None:
                try:
                    print(
                        f"[worker {stream_idx}] waiting at barrier before benchmark")
                    barrier.wait()
                    print(
                        f"[worker {stream_idx}] barrier released, starting benchmark")
                except Exception as e:
                    print(
                        f"[worker {stream_idx}] barrier error (may indicate worker crash): {e}")
                    raise

            # Iterate over batch sizes and sequence lengths
            for batch_size in batch_sizes:
                for seq_length in seq_lengths:
                    # Synchronize all workers before starting this batch_size/seq_length combination
                    if barrier is not None:
                        try:
                            print(
                                f"[worker {stream_idx}] waiting at barrier before (bs={batch_size}, seqlen={seq_length})")
                            barrier.wait()
                            print(
                                f"[worker {stream_idx}] barrier released, starting (bs={batch_size}, seqlen={seq_length})")
                        except Exception as e:
                            print(f"[worker {stream_idx}] barrier error: {e}")
                            raise

                    # Create input hidden states for this combination
                    if tokenizer is not None:
                        hidden_states = get_embeddings_from_token_ids(
                            model,
                            tokenizer,
                            batch_size,
                            seq_length,
                            device,
                            dtype=torch.bfloat16,
                        )
                    else:
                        hidden_states = torch.randn(
                            batch_size, seq_length, hidden_dim,
                            dtype=torch.bfloat16,
                            device=device
                        )

                    # Benchmark each component on this stream
                    for comp_name, component in components.items():
                        times = []
                        cuda_times = []

                        print(
                            f"[worker {stream_idx}] benchmarking {comp_name} (bs={batch_size}, seqlen={seq_length})")

                        # Prepare arguments once (outside timing)
                        try:
                            _, _, call_func = prepare_component_args(
                                component, hidden_states, model_config, comp_name)
                        except Exception as e:
                            print(
                                f"[worker {stream_idx}] component {comp_name} error preparing args: {e}")
                            import traceback
                            traceback.print_exc()
                            raise

                        with torch.no_grad():
                            for repeat_idx in range(num_repeats):
                                if True:
                                    if barrier is not None:
                                        try:
                                            print(
                                                f"[worker {stream_idx}] waiting at barrier before (bs={batch_size}, seqlen={seq_length})")
                                            barrier.wait()
                                            print(
                                                f"[worker {stream_idx}] barrier released, starting (bs={batch_size}, seqlen={seq_length})")
                                        except Exception as e:
                                            print(
                                                f"[worker {stream_idx}] barrier error: {e}")
                                            raise

                                start_event = torch.cuda.Event(
                                    enable_timing=True)
                                end_event = torch.cuda.Event(
                                    enable_timing=True)

                                # Synchronize before timing to ensure clean measurement
                                torch.cuda.synchronize()

                                start = time.perf_counter()
                                start_event.record()

                                # Only measure the forward call time
                                try:
                                    _ = call_func()
                                except Exception as e:
                                    print(
                                        f"[worker {stream_idx}] component {comp_name} error: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise

                                end_event.record()
                                torch.cuda.synchronize()
                                end = time.perf_counter()

                                times.append((end - start) * 1000)
                                cuda_times.append(
                                    start_event.elapsed_time(end_event))

                        times = np.array(times)
                        cuda_times = np.array(cuda_times)

                        # Synchronize all workers after completing this component test
                        # This ensures all streams finish the same test before writing results
                        if barrier is not None:
                            try:
                                barrier.wait()
                            except Exception as e:
                                print(
                                    f"[worker {stream_idx}] barrier error after component test: {e}")
                                raise

                        # Extract component type and layer index
                        if 'attn' in comp_name:
                            comp_type = 'attention'
                        elif 'ffn' in comp_name:
                            comp_type = 'ffn'
                        else:
                            comp_type = 'unknown'

                        layer_idx = comp_name.split(
                            '_')[1] if '_' in comp_name else '-1'

                        # Record result
                        result_row = {
                            "stream_idx": stream_idx,
                            "sm_partition_count": sm_partition_count,
                            "sm_count": sm_count,
                            "batch_size": batch_size,
                            "seq_length": seq_length,
                            "component_type": comp_type,
                            "layer_idx": layer_idx,
                            "hidden_dim": hidden_dim,
                            "min_time_ms": float(np.min(times)),
                            "max_time_ms": float(np.max(times)),
                            "mean_time_ms": float(np.mean(times)),
                            "median_time_ms": float(np.median(times)),
                            "stdev_time_ms": float(np.std(times)),
                            "cuda_max_time_ms": float(np.max(cuda_times)),
                            "cuda_min_time_ms": float(np.min(cuda_times)),
                            "cuda_mean_time_ms": float(np.mean(cuda_times)),
                            "cuda_median_time_ms": float(np.median(cuda_times)),
                            "cuda_stdev_time_ms": float(np.std(cuda_times)),
                        }

                        with lock:
                            pd.DataFrame([result_row]).to_csv(
                                output_path, mode='a', header=False, index=False)
                            results.append(result_row)

                        print(
                            f"[worker {stream_idx}] {comp_name} (bs={batch_size}, seqlen={seq_length}): "
                            f"mean={result_row['mean_time_ms']:.4f}ms, "
                            f"median={result_row['median_time_ms']:.4f}ms"
                        )

            torch.cuda.empty_cache()
            cleanup()
            torch.cuda.synchronize()
            print(f"[worker {stream_idx}] finished")

    except Exception as e:
        print(f"[worker {stream_idx}] error: {e}")
        import traceback
        traceback.print_exc()
        # Try to release barrier even if error occurred
        if barrier is not None:
            try:
                print(
                    f"[worker {stream_idx}] attempting to release barrier after error")
                barrier.abort()
            except Exception:
                pass


def benchmark_parallel_components_green_ctx(args):
    """
    Benchmark components in parallel on different green context streams.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.log_dir, exist_ok=True)

    dev = torch.device("cuda:0")

    # Create output CSV
    output_path = os.path.join(args.log_dir, args.log_path)
    df_header = pd.DataFrame(columns=[
        "stream_idx",
        "sm_partition_count",
        "sm_count",
        "batch_size",
        "seq_length",
        "component_type",
        "layer_idx",
        "hidden_dim",
        "min_time_ms",
        "max_time_ms",
        "mean_time_ms",
        "median_time_ms",
        "stdev_time_ms",
        "cuda_max_time_ms",
        "cuda_min_time_ms",
        "cuda_mean_time_ms",
        "cuda_median_time_ms",
        "cuda_stdev_time_ms",
    ])
    df_header.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"Output will be saved to: {output_path}")
    print(f"{'='*70}")

    # Load model and tokenizer
    print(f"\nLoading model from {args.model}...")

    # Try to use local modeling_qwen3_2 if available and model path matches Qwen3
    model = None
    if local_qwen is not None and 'qwen3' in args.model.lower():
        try:
            print("Using local modeling_qwen3_2...")
            model = local_qwen.Qwen3ForCausalLM.from_pretrained(
                args.model,
                device_map="cuda:0",
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
            model.eval()
            if hasattr(args, 'debug_shapes') and args.debug_shapes:
                local_qwen.set_debug_shapes(True)
        except Exception as e:
            print(
                f"Failed to load with local modeling_qwen3_2: {e}, falling back to AutoModel")
            model = None

    # Fallback to AutoModel if local loading failed or not applicable
    if model is None:
        model_kwargs = {
            "attn_implementation": args.attention_impl
        }
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            **model_kwargs
        )
        model.eval()

    # Load tokenizer if available
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True)
    except Exception:
        tokenizer = None

    # Get model config
    if hasattr(model, 'config'):
        hidden_dim = model.config.hidden_size
        print(f"Model hidden dimension: {hidden_dim}")
    else:
        hidden_dim = None

    # Extract components
    components = extract_attention_and_ffn_layers(model)
    print(f"Found {len(components)} components (attention + FFN layers)")
    # print_available_components(components)

    # Filter components if component_name is specified
    if args.component_name:
        if args.component_name not in components:
            print(f"ERROR: Component '{args.component_name}' not found!")
            print(f"Available components: {list(components.keys())}")
            return
        components = {args.component_name: components[args.component_name]}
        print(f"Benchmarking only component: {args.component_name}")

    # Create green context streams
    num_streams = args.num_streams
    min_sm_per_stream = args.min_sm_per_stream

    print(f"\n{'='*70}")
    print(
        f"Creating {num_streams} green context streams with min {min_sm_per_stream} SMs each")
    print(f"{'='*70}\n")

    streams, resources = split_device_green_ctx(
        dev, num_streams, min_sm_per_stream)

    # Get SM counts for each resource
    sm_counts = []
    for res in resources:
        try:
            sm_count = res.sm.smCount if hasattr(
                res, 'sm') and hasattr(res.sm, 'smCount') else -1
        except Exception:
            sm_count = -1
        sm_counts.append(sm_count)
        print(f"Stream resource allocated: {sm_count} SMs")

    print(f"streams number: {len(streams)}, SM counts: {sm_counts}\n")

    # Shared results list and lock
    results = []
    results_lock = threading.Lock()
    threads = []

    # Calculate actual number of streams (excluding leftover if requested)
    if args.no_leftover_mode:
        actual_num_streams = len(streams) - 1  # Don't use the remaining stream
        streams_to_use = streams[:-1]
        sm_counts_to_use = sm_counts[:-1]
    else:
        # Use all streams including remaining
        actual_num_streams = len(streams)
        streams_to_use = streams
        sm_counts_to_use = sm_counts

    # Create barrier for synchronizing all worker threads
    barrier = threading.Barrier(actual_num_streams)

    # Launch worker threads for each stream
    print(
        f"Launching {actual_num_streams} worker threads (one per stream)...\n")
    for stream_idx, (stream, sm_count) in enumerate(zip(streams_to_use, sm_counts_to_use)):
        t = threading.Thread(
            target=worker_run_components_on_stream,
            args=(
                components,
                stream_idx,
                stream,
                sm_count,
                actual_num_streams,  # sm_partition_count
                args.batch_sizes,
                args.seq_lengths,
                hidden_dim,
                dev,
                model.config if hasattr(model, 'config') else None,
                args.num_repeat,
                args.warmup,
                tokenizer,
                model,
                results,
                results_lock,
                output_path,
                barrier
            ),
            daemon=False
        )
        t.start()
        threads.append(t)

    # Wait for all workers to finish
    print("Waiting for workers to finish...\n")
    for t in threads:
        t.join()

    # Summary
    if results:
        print(f"\n{'='*70}")
        print(
            f"Collected {len(results)} measurements (saved to {output_path})")
        print(f"{'='*70}\n")
    else:
        print("No results were collected.")

    # Cleanup
    try:
        del model
    except Exception:
        pass
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Transformers component benchmarking with green context streams'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--component-name', type=str, default=None,
                        help='Specific component to benchmark (e.g., layer_0_attn, layer_0_ffn). If not specified, benchmark all components.')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Batch sizes to test')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[128, 256, 512, 1024],
                        help='Sequence lengths to test')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Backward compatible: single batch size')
    parser.add_argument('--seq-length', type=int, default=None,
                        help='Backward compatible: single sequence length')
    parser.add_argument('--num-streams', type=int, default=2,
                        help='Number of parallel streams to create')
    parser.add_argument('--min-sm-per-stream', type=int, default=32,
                        help='Minimum SMs per stream')
    parser.add_argument('--num-repeat', type=int, default=5,
                        help='Number of repeats for each component benchmark')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup iterations')
    parser.add_argument('--log-dir', type=str, default='parallel_components_green_ctx',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str,
                        default='transformers_parallel_components.csv',
                        help='Output CSV filename')
    parser.add_argument('--attention-impl', type=str, default='eager',
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help='Preferred attention implementation')
    parser.add_argument('--debug-shapes', action='store_true',
                        help='Enable debug shapes for local Qwen model (if applicable)')
    parser.add_argument('--no-leftover-mode', default=False, action='store_true',
                        help='Do not include the leftover SM allocation (use only main splits)')

    args = parser.parse_args()

    # Backward compatibility for single-value flags
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        args.batch_sizes = [args.batch_size]
    if hasattr(args, 'seq_length') and args.seq_length is not None:
        args.seq_lengths = [args.seq_length]

    print("="*70)
    print("Parallel Transformers Component Benchmarking (Green Context Streams)")
    print("="*70)
    print(f"Model: {args.model}")
    if args.component_name:
        print(f"Component: {args.component_name}")
    else:
        print(f"Component: All components")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Num streams: {args.num_streams}")
    print(f"Min SMs per stream: {args.min_sm_per_stream}")
    print(f"Num repeats: {args.num_repeat}")
    print(f"Attention implementation: {args.attention_impl}")
    print("="*70)

    benchmark_parallel_components_green_ctx(args)


if __name__ == "__main__":
    main()
