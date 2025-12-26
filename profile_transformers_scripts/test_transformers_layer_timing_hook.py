#!/usr/bin/env python3
"""
Measure timing for each attention and MLP layer in Qwen3-4B using PyTorch hooks.

This script:
1. Loads Qwen3-4B model from transformers
2. Registers forward hooks on attention and MLP layers
3. Runs inference and collects timing data for each layer
4. Outputs results to CSV with format matching test_transformers_green_ctx.py
"""

import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple
from random import randint

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import modeling_qwen3_2 as local_qwen
except ImportError:
    local_qwen = None


class LayerTimingHook:
    """Hook to measure forward pass timing for a layer."""
    
    def __init__(self, layer_name: str, layer_type: str, layer_idx: int):
        self.layer_name = layer_name
        self.layer_type = layer_type  # 'attention' or 'mlp'
        self.layer_idx = layer_idx
        self.times = []
        self.cuda_times = []
        self.start_event = None
        self.end_event = None
        self.start_time = None
        
    def pre_hook(self, module, input):
        """Called before forward pass."""
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_time = time.perf_counter()
        self.start_event.record()
        
    def post_hook(self, module, input, output):
        """Called after forward pass."""
        self.end_event.record()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Record timing
        cpu_time_ms = (end_time - self.start_time) * 1000.0
        cuda_time_ms = self.start_event.elapsed_time(self.end_event)
        
        self.times.append(cpu_time_ms)
        self.cuda_times.append(cuda_time_ms)
        
    def get_stats(self) -> Dict[str, float]:
        """Get statistics for collected timings."""
        if not self.times:
            return {
                'mean_ms': 0.0,
                'median_ms': 0.0,
                'std_ms': 0.0,
                'cuda_mean_ms': 0.0,
                'cuda_median_ms': 0.0,
                'cuda_std_ms': 0.0,
            }
        
        times_arr = np.array(self.times)
        cuda_times_arr = np.array(self.cuda_times)
        
        return {
            'mean_ms': float(np.mean(times_arr)),
            'median_ms': float(np.median(times_arr)),
            'std_ms': float(np.std(times_arr)),
            'cuda_mean_ms': float(np.mean(cuda_times_arr)),
            'cuda_median_ms': float(np.median(cuda_times_arr)),
            'cuda_std_ms': float(np.std(cuda_times_arr)),
        }
    
    def reset(self):
        """Reset collected timings."""
        self.times = []
        self.cuda_times = []


def register_layer_hooks(model, hooks_dict: Dict[str, LayerTimingHook]):
    """Register forward hooks on all attention and MLP layers."""
    hooks_handles = []
    
    # Find all decoder layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        print("Warning: Could not find model layers")
        return hooks_handles
    
    for layer_idx, layer in enumerate(layers):
        # Register hook for attention layer
        if hasattr(layer, 'self_attn'):
            attn_name = f"layer_{layer_idx}_attention"
            if attn_name not in hooks_dict:
                hooks_dict[attn_name] = LayerTimingHook(
                    attn_name, 'attention', layer_idx
                )
            hook = hooks_dict[attn_name]
            handle = layer.self_attn.register_forward_hook(hook.post_hook)
            hooks_handles.append(handle)
            # Also register pre-hook if needed
            layer.self_attn.register_forward_pre_hook(hook.pre_hook)
        
        # Register hook for MLP layer
        if hasattr(layer, 'mlp'):
            mlp_name = f"layer_{layer_idx}_mlp"
            if mlp_name not in hooks_dict:
                hooks_dict[mlp_name] = LayerTimingHook(
                    mlp_name, 'mlp', layer_idx
                )
            hook = hooks_dict[mlp_name]
            handle = layer.mlp.register_forward_hook(hook.post_hook)
            hooks_handles.append(handle)
            # Also register pre-hook if needed
            layer.mlp.register_forward_pre_hook(hook.pre_hook)
    
    print(f"Registered {len(hooks_handles)} hooks on {len(hooks_dict)} layers")
    return hooks_handles


def measure_generation_with_layer_timing(
    model, tokenizer, inputs, output_length: int, device, hooks_dict: Dict[str, LayerTimingHook]
):
    """
    Generate tokens and measure timing for each layer using hooks.
    
    Returns:
        ttft_ms: Time to first token in milliseconds
        tpot_ms: Average time per output token (excluding first token)
        total_time_ms: Total generation time
        generated_tokens: Number of generated tokens
        layer_times: Dict of layer timing statistics
    """
    batch_input_ids = inputs["input_ids"].clone()
    batch_size = batch_input_ids.shape[0]
    
    first_token_times = [None] * batch_size
    subsequent_token_times = [[] for _ in range(batch_size)]
    generated_counts = [0] * batch_size
    
    # Reset all hooks
    for hook in hooks_dict.values():
        hook.reset()
    
    gen_start = time.perf_counter()
    
    with torch.no_grad():
        past_key_values = None
        attention_mask = inputs.get("attention_mask", None)
        current_attention_mask = attention_mask
        
        for token_idx in range(output_length):
            token_gen_start = time.perf_counter()
            
            # First iteration: use full prompt, subsequent iterations: only new token
            if token_idx == 0:
                input_ids = batch_input_ids
            else:
                # Only process the newly generated token
                input_ids = next_token_ids
                # Update attention mask: append 1 for the new token
                if current_attention_mask is not None:
                    new_mask = torch.ones(
                        (batch_size, 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device
                    )
                    current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=-1)
            
            out = model(
                input_ids=input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Save past_key_values for next iteration
            if hasattr(out, 'past_key_values'):
                past_key_values = out.past_key_values
            
            logits = out.logits[:, -1, :]
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            torch.cuda.synchronize()
            token_gen_end = time.perf_counter()
            
            token_time_ms = (token_gen_end - token_gen_start) * 1000
            
            # Record timing for each sample in batch
            for i in range(batch_size):
                if first_token_times[i] is None:
                    first_token_times[i] = token_time_ms
                else:
                    subsequent_token_times[i].append(token_time_ms)
                generated_counts[i] += 1
                
                # Check for EOS token
                if int(next_token_ids[i, 0].item()) == tokenizer.eos_token_id:
                    break
    
    gen_end = time.perf_counter()
    
    # Aggregate metrics
    valid_first = [t for t in first_token_times if t is not None]
    ttft_ms = float(sum(valid_first) / len(valid_first)) if valid_first else 0.0
    
    all_subsequent = [t for lst in subsequent_token_times for t in lst]
    tpot_ms = float(sum(all_subsequent) / len(all_subsequent)) if all_subsequent else 0.0
    
    total_time_ms = (gen_end - gen_start) * 1000
    generated_tokens = float(sum(generated_counts) / len(generated_counts))
    
    # Get layer timing statistics
    layer_times = {}
    for layer_name, hook in hooks_dict.items():
        stats = hook.get_stats()
        layer_times[layer_name] = stats
    
    return ttft_ms, tpot_ms, total_time_ms, int(generated_tokens), layer_times


def benchmark_layers_with_hooks(args):
    """Main benchmarking function."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("Transformers Layer Timing with Hooks")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Prompt length: {args.prompt_length}")
    print(f"Output length: {args.output_length}")
    print(f"Num repeats: {args.num_repeat}")
    print("="*60)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = None
    if local_qwen is not None and 'qwen3' in args.model.lower():
        try:
            model = local_qwen.Qwen3ForCausalLM.from_pretrained(
                args.model,
                device_map="cuda:0",
                trust_remote_code=True,
                dtype=torch.float16,
                use_cache=True,
            )
            model.eval()
        except Exception as e:
            print(f"Failed to load local Qwen model: {e}")
            model = None
    
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()
    
    print("Model loaded successfully")
    
    # Register hooks
    hooks_dict = {}
    hooks_handles = register_layer_hooks(model, hooks_dict)
    
    # Create output directory
    os.makedirs(args.log_dir, exist_ok=True)
    output_path = os.path.join(args.log_dir, args.output_file)
    
    # Create CSV header
    df_header = pd.DataFrame(columns=[
        "sm_partition_count",
        "stream_id",
        "sm_count",
        "batch_size",
        "prompt_length",
        "generation_steps",
        "ttft_ms",
        "tpot_ms",
        "total_time_ms",
        "config_id",
        "layer_idx",
        "layer_type",
        "layer_name",
        "layer_time_mean_ms",
        "layer_time_median_ms",
        "layer_time_std_ms",
        "layer_time_cuda_mean_ms",
        "layer_time_cuda_median_ms",
        "layer_time_cuda_std_ms",
    ])
    df_header.to_csv(output_path, index=False)
    
    # Run benchmarks
    results = []
    
    for batch_size in args.batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        for repeat_idx in range(args.num_repeat):
            print(f"  Repeat {repeat_idx + 1}/{args.num_repeat}")
            
            # Create random prompt tokens
            prompt_token_ids = [[randint(100, 8000) for _ in range(args.prompt_length)]
                               for _ in range(batch_size)]
            
            # Pad sequences to same length
            inputs = tokenizer.pad(
                {"input_ids": prompt_token_ids},
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Reset hooks before each measurement
            for hook in hooks_dict.values():
                hook.reset()
            
            # Warm up (only on first repeat)
            if repeat_idx == 0:
                print("    Warming up...")
                with torch.no_grad():
                    _ = model.generate(
                        input_ids=inputs["input_ids"][:1],
                        max_new_tokens=1,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                # Reset hooks after warmup
                for hook in hooks_dict.values():
                    hook.reset()
            
            # Measure generation with layer timing
            ttft_ms, tpot_ms, total_time_ms, generation_steps, layer_times = \
                measure_generation_with_layer_timing(
                    model, tokenizer, inputs, args.output_length, device, hooks_dict
                )
            
            print(f"    TTFT={ttft_ms:.2f}ms, TPOT={tpot_ms:.2f}ms, "
                  f"Total={total_time_ms:.2f}ms, Generated={generation_steps}")
            
            # Write results for each layer
            for layer_name, stats in layer_times.items():
                # Parse layer info
                parts = layer_name.split('_')
                layer_idx = int(parts[1]) if len(parts) > 1 else -1
                layer_type = parts[2] if len(parts) > 2 else 'unknown'
                
                result_row = {
                    "sm_partition_count": args.sm_partition_count,
                    "stream_id": 0,
                    "sm_count": args.sm_count,
                    "batch_size": batch_size,
                    "prompt_length": args.prompt_length,
                    "generation_steps": generation_steps,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "total_time_ms": total_time_ms,
                    "config_id": f"{args.config_id}",
                    "layer_idx": layer_idx,
                    "layer_type": layer_type,
                    "layer_name": layer_name,
                    "layer_time_mean_ms": stats['mean_ms'],
                    "layer_time_median_ms": stats['median_ms'],
                    "layer_time_std_ms": stats['std_ms'],
                    "layer_time_cuda_mean_ms": stats['cuda_mean_ms'],
                    "layer_time_cuda_median_ms": stats['cuda_median_ms'],
                    "layer_time_cuda_std_ms": stats['cuda_std_ms'],
                }
                
                # Append to CSV
                pd.DataFrame([result_row]).to_csv(
                    output_path, mode='a', header=False, index=False
                )
                results.append(result_row)
    
    # Remove hooks
    for handle in hooks_handles:
        handle.remove()
    
    print(f"\n{'='*60}")
    print(f"Benchmark completed! Results saved to: {output_path}")
    print(f"Total measurements: {len(results)}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Measure timing for each attention and MLP layer using hooks'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--prompt-length', type=int, default=512,
                        help='Prompt length')
    parser.add_argument('--output-length', type=int, default=128,
                        help='Output length to generate')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2],
                        help='Batch sizes to test')
    parser.add_argument('--num-repeat', type=int, default=3,
                        help='Number of repeats for each batch size')
    parser.add_argument('--log-dir', type=str, default='layer_timing_results',
                        help='Output directory for logs')
    parser.add_argument('--output-file', type=str,
                        default='layer_timing_hook_results.csv',
                        help='Output CSV filename')
    parser.add_argument('--sm-partition-count', type=int, default=0,
                        help='SM partition count (for compatibility)')
    parser.add_argument('--sm-count', type=int, default=108,
                        help='SM count (for compatibility)')
    parser.add_argument('--config-id', type=str, default='hook_0',
                        help='Config ID (for compatibility)')
    
    args = parser.parse_args()
    
    benchmark_layers_with_hooks(args)


if __name__ == "__main__":
    main()

