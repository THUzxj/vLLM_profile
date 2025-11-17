#!/usr/bin/env python3
"""
Benchmark Transformers model components (Attention and FFN layers) 
with different GPU resource partitioning using Green Contexts.

This script:
1. Splits GPU resources using CUDA Green Contexts
2. Creates Transformers model instances on different streams
3. Extracts and benchmarks individual attention and FFN layers
4. Measures latency and throughput for each component
5. Compares performance across different resource configurations
"""

import argparse
import contextlib
import gc
import os
import sys
import time
from typing import List, Tuple, Dict, Callable
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
            input_ids = input_ids.repeat(batch_size // input_ids.size(0) + 1, 1)[:batch_size]
    else:
        # try to determine vocab size
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = tokenizer.model_maximum_features if hasattr(tokenizer, 'model_maximum_features') else None
            except Exception:
                vocab_size = None
        if not vocab_size or vocab_size <= 0:
            vocab_size = 50257
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    # Get embedding layer
    emb_layer = None
    if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
        emb_layer = model.get_input_embeddings()
    elif hasattr(model, 'embed_tokens'):
        emb_layer = model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        emb_layer = model.model.embed_tokens

    if emb_layer is None:
        raise RuntimeError("Embedding layer not found on model; cannot produce input embeddings")

    # Ensure input_ids on same device as embedding layer
    input_ids = input_ids.to(next(emb_layer.parameters()).device if any(True for _ in emb_layer.parameters()) else device)

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
            has_qkv = any(hasattr(module, attr) for attr in ('q_proj', 'k_proj', 'v_proj', 'qkv_proj', 'q', 'k', 'v'))
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
                sub_cls = ' '.join([m.__class__.__name__.lower() for m in module])
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
        int(x.split('_')[1]) if len(x.split('_')) > 1 and x.split('_')[1].isdigit() else 999,
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
    print("Use --component-name to benchmark a specific component:")
    print(f"  Example: python test_transformers_component_benchmark.py --component-name {sorted_components[0]}\n")


def benchmark_component(
    component: torch.nn.Module,
    component_name: str,
    hidden_states: torch.Tensor,
    num_repeats: int = 10,
    warmup: int = 2,
    model_config = None,
) -> Dict[str, float]:
    """
    Benchmark a single component (attention or FFN layer).
    
    Args:
        component: The module to benchmark
        component_name: Name of the component for logging
        hidden_states: Dummy input tensor (batch_size, seq_length, hidden_dim)
        num_repeats: Number of timing iterations
        warmup: Number of warmup iterations
        model_config: Optional model config to get num_attention_heads, hidden_size, etc.
    
    Returns:
        dict with keys: min_time_ms, max_time_ms, mean_time_ms, stdev_time_ms
    """
    times = []

    def _make_defaults(bsz, seq_len, hidden_dim, dtype, device):
        # bsz, seq_len, hidden_dim = hidden.shape
        # For RoPE (Qwen3, LLaMA): return (cos, sin) tuple
        # Standard RoPE: cos and sin have shape (1, seq_len, 1, head_dim)
        # Calculate head_dim from model config if available, otherwise estimate
        if model_config is not None and hasattr(model_config, 'head_dim'):
            head_dim = model_config.head_dim
        elif model_config is not None and hasattr(model_config, 'num_attention_heads'):
            num_heads = model_config.num_attention_heads
            head_dim = hidden_dim // num_heads
        else:
            head_dim = hidden_dim // 8  # fallback estimate

        cos = torch.ones(1, seq_len, head_dim, dtype=dtype, device=device)
        sin = torch.zeros(1, seq_len, head_dim, dtype=dtype, device=device)
        pos_emb_rope = (cos, sin)
        
        # Also provide a single tensor fallback for non-RoPE models
        pos_emb_single = torch.zeros(bsz, seq_len, hidden_dim, dtype=dtype, device=device)
        
        # Generate proper 4D causal attention mask: (bsz, 1, seq_len, seq_len)
        # Causal mask has -inf (or very negative value) for future positions, 0 for valid positions
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device),
            diagonal=1
        )
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
        return pos_emb_rope, pos_emb_single, attn_mask

    def invoke_component(module: torch.nn.Module, hidden: torch.Tensor):
        """Call the module with sensible defaults depending on its signature.

        Tries multiple reasonable argument combinations to support different module APIs
        (some expect position_embeddings as (cos, sin) tuple for RoPE, others as single tensor).
        """
        sig = None
        try:
            sig = inspect.signature(module.forward)
        except (ValueError, TypeError):
            try:
                sig = inspect.signature(module)
            except Exception:
                sig = None

        bsz, seq_len, hidden_dim = hidden.shape
        pos_emb_rope, pos_emb_single, attn_mask = _make_defaults(bsz, seq_len, hidden_dim, hidden.dtype, hidden.device)

        # Log main tensor shapes for debugging (q/k/v projections, pos embeddings, masks)
        # try:
        #     print(f"shapes: hidden={hidden.shape}, pos_emb_rope=({pos_emb_rope[0].shape}, {pos_emb_rope[1].shape}), pos_emb_single={pos_emb_single.shape}, attn_mask={attn_mask.shape}")
        # except Exception:
        #     pass

        # Build kwargs based on parameter names when possible
        if sig is not None:
            params = list(sig.parameters.keys())
            kwargs = {}
            args = []
            # print(f'params: {params}')
            # common param names mapping
            if 'hidden_states' in params or 'x' in params or 'input' in params:
                args.append(hidden)

            if 'position_embeddings' in params:
                # Try RoPE first (cos, sin tuple), then fall back to single tensor
                args.append(pos_emb_rope)

            if 'attention_mask' in params:
                args.append(attn_mask)

            # pass return_dict=False when present to avoid structured outputs
            if 'return_dict' in params:
                kwargs['return_dict'] = False

            # Some modules accept past_key_values; pass None
            if 'past_key_values' in params:
                kwargs['past_key_values'] = None

            try:
                return module(*args, **kwargs)
            except (TypeError, ValueError) as e:
                # If RoPE failed (unpacking error), try with single tensor
                if 'position_embeddings' in params and 'unpack' in str(e).lower() or 'tuple' in str(e).lower():
                    args_fallback = []
                    if 'hidden_states' in params or 'x' in params or 'input' in params:
                        args_fallback.append(hidden)
                    if 'position_embeddings' in params:
                        args_fallback.append(pos_emb_single)
                    if 'attention_mask' in params:
                        args_fallback.append(attn_mask)
                    try:
                        return module(*args_fallback, **kwargs)
                    except (TypeError, ValueError):
                        pass
                # Fallbacks below
                pass

        # Try common call patterns with RoPE
        try:
            return module(hidden, pos_emb_rope, attn_mask)
        except (TypeError, ValueError):
            pass

        # Try common call patterns with single pos_emb
        # try:
        #     return module(hidden, pos_emb_single, attn_mask)
        # except (TypeError, ValueError):
        #     pass

        try:
            return module(hidden, attention_mask=attn_mask)
        except (TypeError, ValueError):
            pass

        try:
            return module(hidden)
        except Exception as e:
            # Reraise with more context
            raise

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            try:
                _ = invoke_component(component, hidden_states)
            except Exception as warmup_err:
                print(f"  Warmup error: {warmup_err}")
                raise

    torch.cuda.synchronize()

    # Actual measurements
    with torch.no_grad():
        for _ in range(num_repeats):
            start = time.perf_counter()
            _ = invoke_component(component, hidden_states)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'mean_time_ms': float(np.mean(times)),
        'median_time_ms': float(np.median(times)),
        'stdev_time_ms': float(np.std(times)),
    }


def benchmark_components_with_green_ctx(args):
    """
    Benchmark model components with different GPU resource partitioning.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.log_dir, exist_ok=True)
    
    dev = torch.device("cuda:0")
    
    # Generate output filename with component name if specified
    if args.component_name:
        # Insert component name into filename
        base_name, ext = os.path.splitext(args.log_path)
        output_filename = f"{base_name}_{args.component_name}{ext}"
    else:
        output_filename = args.log_path
    
    output_path = os.path.join(args.log_dir, output_filename)
    df_header = pd.DataFrame(columns=[
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
    ])
    df_header.to_csv(output_path, index=False)
    
    print(f"Output will be saved to: {output_filename}")
    
    # Load model and tokenizer once
    print(f"Loading model from {args.model}...")
    
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

            local_qwen.set_debug_shapes(args.debug_shapes)
        except Exception as e:
            print(f"Failed to load with local modeling_qwen3_2: {e}, falling back to AutoModel")
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
    
    # Load tokenizer if available (used to produce realistic input embeddings)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    
    # Test different SM partition sizes
    sm_partition_sizes = args.sm_partition_sizes
    
    for partition_idx, sm_size in enumerate(sm_partition_sizes):
        try:
            print(f"\n{'='*70}")
            print(f"Testing SM partition size: {sm_size}")
            print(f"{'='*70}")
            
            # Split GPU resources
            streams, resources = split_device_green_ctx(dev, 1, sm_size)
            sm_counts = [r.sm.smCount for r in resources]
            print(f"SM counts: {sm_counts}")
            
            main_stream = streams[0]
            main_sm_count = sm_counts[0]
            
            with torch.cuda.stream(main_stream):
                # Test different batch sizes and sequence lengths
                for batch_size in args.batch_sizes:
                    for seq_length in args.seq_lengths:
                        print(f"\n  Batch size: {batch_size}, Seq length: {seq_length}")
                        
                        # Create hidden states by tokenizing and using model embeddings when possible
                        # Shape: (batch_size, seq_length, hidden_dim)
                        if tokenizer is not None:
                            hidden_states = get_embeddings_from_token_ids(
                                model,
                                tokenizer,
                                batch_size,
                                seq_length,
                                dev,
                                dtype=torch.bfloat16,
                            )
                        else:
                            hidden_states = torch.randn(
                                batch_size, seq_length, hidden_dim,
                                dtype=torch.bfloat16,
                                device=dev
                            )
                        
                        # Benchmark each component
                        for component_name, component in components.items():
                            # Extract layer type (attn or ffn) and index
                            if 'attn' in component_name:
                                comp_type = 'attention'
                            elif 'ffn' in component_name:
                                comp_type = 'ffn'
                            else:
                                comp_type = 'unknown'
                            
                            layer_idx = component_name.split('_')[1] if '_' in component_name else '-1'
                            
                            try:
                                # Benchmark the component
                                timings = benchmark_component(
                                    component,
                                    component_name,
                                    hidden_states,
                                    num_repeats=args.num_repeat,
                                    warmup=args.warmup,
                                    model_config=model.config if hasattr(model, 'config') else None,
                                )
                                
                                # Log result
                                result_row = pd.DataFrame([{
                                    "sm_partition_count": sm_size,
                                    "sm_count": main_sm_count,
                                    "batch_size": batch_size,
                                    "seq_length": seq_length,
                                    "component_type": comp_type,
                                    "layer_idx": layer_idx,
                                    "hidden_dim": hidden_dim,
                                    "min_time_ms": timings['min_time_ms'],
                                    "max_time_ms": timings['max_time_ms'],
                                    "mean_time_ms": timings['mean_time_ms'],
                                    "median_time_ms": timings['median_time_ms'],
                                    "stdev_time_ms": timings['stdev_time_ms'],
                                }])
                                result_row.to_csv(output_path, mode='a', header=False, index=False)
                                
                                print(f"    {component_name}: mean={timings['mean_time_ms']:.4f}ms, "
                                      f"median={timings['median_time_ms']:.4f}ms, "
                                      f"stdev={timings['stdev_time_ms']:.4f}ms")
                            
                            except Exception as e:
                                print(f"    {component_name}: Error - {e}")
                                # print backtrace
                                import traceback

                                traceback.print_exc()
                
                # Clean up per-partition temporary tensors
                torch.cuda.empty_cache()
                cleanup()
                torch.cuda.synchronize()
        
        except RuntimeError as e:
            print(f"Error with SM partition size {sm_size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final cleanup: delete model once after all partitions
    try:
        del model
    except Exception:
        pass
    torch.cuda.empty_cache()
    cleanup()

    print(f"\n{'='*70}")
    print(f"Benchmark completed! Results saved to: {output_path}")
    print(f"{'='*70}")


def create_comparison_plot(csv_path, output_dir=None):
    """Create visualization of component performance."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Attention vs FFN latency by batch size
    ax1 = axes[0, 0]
    for comp_type in df['component_type'].unique():
        data = df[df['component_type'] == comp_type]
        data_grouped = data.groupby('batch_size')['mean_time_ms'].mean()
        ax1.plot(data_grouped.index, data_grouped.values, marker='o', label=comp_type.upper())
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Component Latency vs Batch Size')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Latency vs SM partition count
    ax2 = axes[0, 1]
    for comp_type in df['component_type'].unique():
        data = df[df['component_type'] == comp_type]
        data_grouped = data.groupby('sm_partition_count')['mean_time_ms'].mean()
        ax2.plot(data_grouped.index, data_grouped.values, marker='s', label=comp_type.upper())
    ax2.set_xlabel('SM Partition Count')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Component Latency vs GPU Partition Size')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Latency vs sequence length
    ax3 = axes[1, 0]
    for comp_type in df['component_type'].unique():
        data = df[df['component_type'] == comp_type]
        data_grouped = data.groupby('seq_length')['mean_time_ms'].mean()
        ax3.plot(data_grouped.index, data_grouped.values, marker='^', label=comp_type.upper())
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Component Latency vs Sequence Length')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Heatmap of attention latency by batch and seq length
    ax4 = axes[1, 1]
    attn_data = df[df['component_type'] == 'attention']
    if not attn_data.empty:
        pivot_data = attn_data.pivot_table(
            values='mean_time_ms',
            index='batch_size',
            columns='seq_length',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4, 
                    cbar_kws={'label': 'Latency (ms)'})
        ax4.set_title('Attention Latency Heatmap: Batch Size vs Sequence Length')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'transformers_component_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Transformers model components (Attention and FFN layers)'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--component-name', type=str, default=None,
                        help='Specific component to benchmark (e.g., layer_0_attn, layer_0_ffn). If not specified, benchmark all components.')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Batch sizes to test')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[128, 256, 512, 1024],
                        help='Sequence lengths to test')
    parser.add_argument('--sm-partition-sizes', type=int, nargs='+',
                        default=[16, 32, 48, 64, 80],
                        help='SM partition sizes to test')
    parser.add_argument('--num-repeat', type=int, default=10,
                        help='Number of repeats for each component benchmark')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup iterations')
    parser.add_argument('--log-dir', type=str, default='profile_transformers_component',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str, 
                        default='transformers_component_benchmark.csv',
                        help='Output CSV filename')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--attention-impl', type=str, default='eager',
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help='Preferred attention implementation')
    parser.add_argument('--dump-modules', action='store_true',
                        help='Dump model.named_modules() to a file (useful for debugging component discovery)')
    
    parser.add_argument('--debug-shapes', action='store_true',
                        help='Enable debug shapes for local Qwen model (if applicable)')
    args = parser.parse_args()
    
    print("="*70)
    print("Transformers Component Benchmark (Attention & FFN Layers)")
    print("="*70)
    print(f"Model: {args.model}")
    if args.component_name:
        print(f"Component: {args.component_name}")
    else:
        print(f"Component: All components")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"SM partition sizes: {args.sm_partition_sizes}")
    print(f"Attention implementation: {args.attention_impl}")
    print("="*70)
    
    # Run benchmark
    # If requested, dump the model.named_modules() for debugging and exit
    if args.dump_modules:
        os.makedirs(args.log_dir, exist_ok=True)
        dump_path = os.path.join(args.log_dir, 'model_named_modules.txt')
        # Load model to inspect modules
        print(f"Loading model to dump named modules: {args.model}")
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation=args.attention_impl,
        )
        with open(dump_path, 'w') as fh:
            for name, module in model.named_modules():
                fh.write(f"{name}\t{module.__class__.__name__}\n")
        print(f"Model named modules written to: {dump_path}")
        return

    benchmark_components_with_green_ctx(args)
    
    # Generate plots if requested
    if args.plot:
        output_path = os.path.join(args.log_dir, args.log_path)
        create_comparison_plot(output_path, args.log_dir)


if __name__ == "__main__":
    main()
