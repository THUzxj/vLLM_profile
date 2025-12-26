#!/usr/bin/env python3
"""
Test Transformers inference with combined batches in a single stream.

This script:
1. Combines multiple batches with different batch_size, prompt_length, and output_length
   into one large batch
2. Runs inference on a single stream (default CUDA stream)
3. Measures per-sub-batch inference latency (TPOT, TTFT, total time)
4. Writes results to CSV file

Key features:
- Combines multiple different batch configurations into one large batch
- Handles different prompt_lengths by padding to max length
- Handles different output_lengths by using minimum output_length
- Measures metrics per original sub-batch configuration
- Used for comparison with parallel green context streams
"""

import argparse
import contextlib
import gc
import os
import time
from typing import List, Tuple
from random import randint

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import modeling_qwen3_2 as local_qwen
except ImportError:
    local_qwen = None


def cleanup():
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


def measure_generation_with_timing_combined(
    model, tokenizer, combined_inputs, sub_batch_configs, output_length, device
):
    """
    Generate tokens for a combined batch and measure metrics per sub-batch.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer
        combined_inputs: Combined inputs with all sub-batches
        sub_batch_configs: List of (start_idx, end_idx, original_batch_size, prompt_length) 
                          for each sub-batch
        output_length: Minimum output length (used for all sub-batches)
        device: Device to run on
    
    Returns:
        results: List of dicts with metrics for each sub-batch
        layer_records: Layer timing records if available
    """
    batch_input_ids = combined_inputs["input_ids"].clone()
    total_batch_size = batch_input_ids.shape[0]
    
    # Initialize tracking for each sample in the combined batch
    first_token_times = [None] * total_batch_size
    subsequent_token_times = [[] for _ in range(total_batch_size)]
    generated_counts = [0] * total_batch_size
    
    gen_start = time.perf_counter()
    
    with torch.no_grad():
        past_key_values = None
        attention_mask = combined_inputs.get("attention_mask", None)
        current_attention_mask = attention_mask
        
        layer_records = []
        for token_idx in range(output_length):
            
            # First iteration: use full prompt, subsequent iterations: only new token
            if token_idx == 0:
                input_ids = batch_input_ids
            else:
                input_ids = next_token_ids
                if current_attention_mask is not None:
                    new_mask = torch.ones(
                        (total_batch_size, 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device
                    )
                    current_attention_mask = torch.cat(
                        [current_attention_mask, new_mask], dim=-1)
            
            token_gen_start = time.perf_counter()
            # Generate one token
            out = model(
                input_ids=input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            torch.cuda.synchronize()
            token_gen_end = time.perf_counter()
            
            if isinstance(out, tuple):
                outputs, layer_record_times = out
            else:
                outputs = out
                layer_record_times = None
            
            past_key_values = outputs.past_key_values
            
            logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            if token_idx != 0:
                token_time_ms = (token_gen_end - token_gen_start) * 1000
                
                # Record timing for each sample in batch
                for batch_idx in range(total_batch_size):
                    if token_idx == 1:  # First generated token (TTFT)
                        first_token_times[batch_idx] = token_time_ms
                    else:
                        subsequent_token_times[batch_idx].append(token_time_ms)
                    
                    # Check for EOS token
                    if next_token_ids[batch_idx].item() == tokenizer.eos_token_id:
                        # Keep padding subsequent tokens for consistent measurement
                        pass
                
                if layer_record_times is not None:
                    layer_records.append(layer_record_times)
            
            for batch_idx in range(total_batch_size):
                generated_counts[batch_idx] += 1
        
        gen_end = time.perf_counter()
        total_time_ms = (gen_end - gen_start) * 1000
        
        # Aggregate metrics per sub-batch
        results = []
        for sub_idx, (start_idx, end_idx, orig_batch_size, prompt_length) in enumerate(sub_batch_configs):
            # Get metrics for this sub-batch
            sub_first_times = first_token_times[start_idx:end_idx]
            sub_subsequent_times = subsequent_token_times[start_idx:end_idx]
            sub_generated_counts = generated_counts[start_idx:end_idx]
            
            valid_first = [t for t in sub_first_times if t is not None]
            ttft_ms = float(np.mean(valid_first)) if valid_first else 0.0
            
            all_subsequent = [t for lst in sub_subsequent_times for t in lst]
            tpot_ms = float(np.mean(all_subsequent)) if all_subsequent else 0.0
            
            generated_tokens = float(np.mean(sub_generated_counts))
            
            results.append({
                "sub_batch_idx": sub_idx,
                "batch_size": orig_batch_size,
                "prompt_length": prompt_length,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "total_time_ms": total_time_ms,  # Same total time for all sub-batches
                "generation_steps": int(generated_tokens),
            })
        
        return results, layer_records


def benchmark_transformers_combined_batch(args):
    """
    Benchmark Transformers with combined batches in a single stream.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create output CSV
    output_path = os.path.join(args.log_dir, args.log_path)
    df_header = pd.DataFrame(columns=[
        "repeat_idx",
        "sub_batch_idx",
        "batch_size",
        "prompt_length",
        "generation_steps",
        "ttft_ms",
        "tpot_ms",
        "total_time_ms",
        "finish_time"
    ])
    df_header.to_csv(output_path, index=False)
    
    device = torch.device("cuda:0")
    
    # Validate array lengths
    num_configs = len(args.batch_sizes)
    if len(args.prompt_lengths) != num_configs:
        raise ValueError(
            f"prompt_lengths length ({len(args.prompt_lengths)}) must match batch_sizes length ({num_configs})"
        )
    if len(args.output_lengths) != num_configs:
        raise ValueError(
            f"output_lengths length ({len(args.output_lengths)}) must match batch_sizes length ({num_configs})"
        )
    
    # Use minimum output_length for all sub-batches
    output_length = min(args.output_lengths)
    max_prompt_length = max(args.prompt_lengths)
    total_batch_size = sum(args.batch_sizes)
    
    print(f"\n{'='*60}")
    print(f"Combined Batch Configuration")
    print(f"{'='*60}")
    print(f"Total batch size: {total_batch_size}")
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Output length (min): {output_length}")
    print(f"Sub-batch configurations:")
    for i, (bs, pl, ol) in enumerate(zip(args.batch_sizes, args.prompt_lengths, args.output_lengths)):
        print(f"  Sub-batch {i}: batch_size={bs}, prompt_length={pl}, output_length={ol}")
    print(f"{'='*60}\n")
    
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
            )
            model.eval()
        except Exception:
            model = None
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()
    print("Model loaded successfully")
    
    # Warm up
    print("Warming up...")
    warmup_prompt = [[randint(100, 8000) for _ in range(max_prompt_length)]]
    warmup_inputs = tokenizer.pad(
        {"input_ids": warmup_prompt},
        padding=True,
        return_tensors="pt"
    )
    warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
    with torch.no_grad():
        _ = model(**warmup_inputs)
    torch.cuda.synchronize()
    print("Warmup complete")
    
    # Run benchmarks
    for repeat_idx in range(args.num_repeat):
        print(f"\n{'='*60}")
        print(f"Repeat {repeat_idx + 1}/{args.num_repeat}")
        print(f"{'='*60}")
        
        # Create combined batch
        all_prompt_token_ids = []
        sub_batch_configs = []
        current_idx = 0
        
        for sub_idx, (batch_size, prompt_length) in enumerate(zip(args.batch_sizes, args.prompt_lengths)):
            # Create prompts for this sub-batch
            sub_prompts = [[randint(100, 8000) for _ in range(prompt_length)]
                          for _ in range(batch_size)]
            all_prompt_token_ids.extend(sub_prompts)
            
            # Record sub-batch configuration
            start_idx = current_idx
            end_idx = current_idx + batch_size
            sub_batch_configs.append((start_idx, end_idx, batch_size, prompt_length))
            current_idx = end_idx
        
        # Pad all sequences to max_prompt_length
        combined_inputs = tokenizer.pad(
            {"input_ids": all_prompt_token_ids},
            padding=True,
            return_tensors="pt"
        )
        combined_inputs = {k: v.to(device) for k, v in combined_inputs.items()}
        
        print(f"Running inference on combined batch (size={total_batch_size})...")
        
        # Measure generation
        results, layer_records = measure_generation_with_timing_combined(
            model, tokenizer, combined_inputs, sub_batch_configs, output_length, device
        )
        
        # Write results
        for sub_result in results:
            result_row = {
                "repeat_idx": repeat_idx,
                "sub_batch_idx": sub_result["sub_batch_idx"],
                "batch_size": sub_result["batch_size"],
                "prompt_length": sub_result["prompt_length"],
                "generation_steps": sub_result["generation_steps"],
                "ttft_ms": sub_result["ttft_ms"],
                "tpot_ms": sub_result["tpot_ms"],
                "total_time_ms": sub_result["total_time_ms"],
                "finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            
            # Append to CSV
            pd.DataFrame([result_row]).to_csv(
                output_path, mode='a', header=False, index=False)
            
            print(f"  Sub-batch {sub_result['sub_batch_idx']}: "
                  f"bs={sub_result['batch_size']} pl={sub_result['prompt_length']} "
                  f"tpot={sub_result['tpot_ms']:.2f}ms ttft={sub_result['ttft_ms']:.2f}ms "
                  f"total={sub_result['total_time_ms']:.2f}ms")
        
        # Save layer records if available
        if layer_records:
            try:
                json_name = (
                    f"layer_times_combined_rep{repeat_idx}_"
                    f"totalbs{total_batch_size}_maxpl{max_prompt_length}_ol{output_length}.json"
                )
                json_path = os.path.join(args.log_dir, json_name)
                with open(json_path, 'w') as fh:
                    json.dump({
                        "repeat_idx": repeat_idx,
                        "total_batch_size": total_batch_size,
                        "max_prompt_length": max_prompt_length,
                        "output_length": output_length,
                        "sub_batch_configs": [
                            {"sub_idx": i, "batch_size": bs, "prompt_length": pl}
                            for i, (_, _, bs, pl) in enumerate(sub_batch_configs)
                        ],
                        "layer_records": layer_records,
                    }, fh)
            except Exception as e:
                print(f"Error writing JSON: {e}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    cleanup()
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete. Results saved to {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Transformers benchmarking with combined batches in a single stream. '
                    'Combines multiple batch configurations into one large batch.'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--prompt-lengths', type=int, nargs='+', required=True,
                        help='Prompt lengths for each sub-batch (array)')
    parser.add_argument('--output-lengths', type=int, nargs='+', required=True,
                        help='Output lengths for each sub-batch (array, min will be used)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', required=True,
                        help='Batch sizes for each sub-batch (array)')
    parser.add_argument('--num-repeat', type=int, default=2,
                        help='Number of repeats')
    parser.add_argument('--log-dir', type=str, default='combined_batch_results',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str,
                        default='transformers_combined_batch_benchmark.csv',
                        help='Output CSV filename')
    parser.add_argument('--attention-impl', type=str, default='default',
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help='Preferred attention implementation to enable on the model (best-effort)')
    args = parser.parse_args()
    
    print("="*60)
    print("Transformers Combined Batch Benchmark")
    print("Single Stream with Combined Batches")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Output lengths: {args.output_lengths}")
    print(f"Num repeats: {args.num_repeat}")
    print("="*60)
    
    benchmark_transformers_combined_batch(args)


if __name__ == "__main__":
    main()

