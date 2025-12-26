#!/usr/bin/env python3
"""
Run Qwen3 model with specified batch size and input length.
Construct inputs and save model's timing records to file.
"""

from transformers import AutoConfig
import modeling_qwen3_2
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

# Add script directory to path for local modeling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_model_and_config(
    model_name: str = "Qwen/Qwen3-0.5B",
    device: str = "cuda",
    attn_implementation: Optional[str] = None
):
    """Load model and configuration."""
    print(f"Loading config from {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Set attention implementation if specified
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
        print(f"Using attention implementation: {attn_implementation}")

    print(f"Creating Qwen3Model...")
    model = modeling_qwen3_2.Qwen3ForCausalLM(config)
    model = model.to(device)
    model.eval()

    return model, config


def construct_inputs(
    batch_size: int,
    input_len: int,
    config,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Construct random input tensors for model."""
    # Random input IDs
    input_ids = torch.randint(
        0, config.vocab_size,
        (batch_size, input_len),
        device=device,
        dtype=torch.long
    )

    # Attention mask (all ones = all tokens attend to all previous tokens)
    attention_mask = torch.ones(
        (batch_size, input_len),
        device=device,
        dtype=torch.long
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,  # Don't use KV cache for simplicity
    }


def run_model_benchmark(
    model,
    config,
    batch_size: int,
    input_len: int,
    num_runs: int = 1,
    warmup_runs: int = 1,
    device: str = "cuda",
):
    """Run model benchmark and collect timing information."""

    print(f"\nBenchmark: batch_size={batch_size}, input_len={input_len}")
    print(f"  Warmup runs: {warmup_runs}, Measurement runs: {num_runs}")

    # Warmup runs
    print("  Warming up GPU...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            inputs = construct_inputs(batch_size, input_len, config, device)
            _ = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

    torch.cuda.synchronize()
    time.sleep(0.5)

    # Measurement runs
    print("  Measuring...")
    all_layer_times = []
    total_times = []

    past_key_values = None

    with torch.no_grad():
        for run in range(num_runs):
            inputs = construct_inputs(batch_size, input_len, config, device)

            # Record total time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            start_time = time.perf_counter()
            result = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )
            # Model returns (CausalLMOutputWithPast, layer_record_times)
            outputs, layer_record_times = result
            end_time = time.perf_counter()

            end_event.record()
            torch.cuda.synchronize()

            total_time = end_time - start_time
            total_times.append(total_time)
            all_layer_times.append(layer_record_times)

            past_key_values = outputs.past_key_values

            if (run + 1) % max(1, num_runs // 5) == 0 or run == num_runs - 1:
                print(
                    f"    Run {run+1}/{num_runs} - Total time: {total_time:.4f}s")

    return all_layer_times, total_times


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3 model and save timing records"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for model input (default: 1)"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Input sequence length (default: 128)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/nfs/xjzhang/Qwen/Qwen3-4B",
        help="Model name or path (default: Qwen/Qwen3-0.5B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of measurement runs (default: 3)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./timing_results",
        help="Output directory for timing results (default: ./timing_results)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="timing_records.json",
        help="Output filename (default: timing_records.json)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="JSON file with multiple configs to run sequentially"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for computation (default: bfloat16)"
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Attention implementation to use (default: None, uses config default). Options: eager, flash_attention_2, sdpa"
    )

    args = parser.parse_args()

    # Select dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model_and_config(
        args.model_name,
        args.device,
        attn_implementation=args.attn_implementation
    )
    model = model.to(dtype)

    # Get configurations to run
    if args.configs:
        # Load from JSON file
        with open(args.configs, 'r') as f:
            configs = json.load(f)
    else:
        # Single configuration from command line args
        configs = [{
            "batch_size": args.batch_size,
            "input_len": args.input_len,
        }]

    # Run benchmarks
    print("="*70)
    print("QWEN3 MODEL TIMING BENCHMARK")
    print("="*70)

    try:
        for config_item in configs:
            batch_size = config_item["batch_size"]
            input_len = config_item["input_len"]

            all_layer_times, total_times = run_model_benchmark(
                model,
                config,
                batch_size=batch_size,
                input_len=input_len,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                device=args.device,
            )

            # Save results to individual file
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "batch_size": batch_size,
                "input_len": input_len,
                "attn_implementation": getattr(config, "_attn_implementation", None),
                "total_times": total_times,
                "avg_total_time": float(np.mean(total_times)),
                "all_layer_details": all_layer_times,
            }

            # Generate filename
            filename = f"timing_bs{batch_size}_len{input_len}.json"
            filepath = output_dir / filename

            with open(filepath, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"✓ Results saved to {filepath}")

            # Print summary for this run
            print(f"\nSummary: batch_size={batch_size}, input_len={input_len}")
            print(f"  Average total time: {np.mean(total_times):.4f}s")
            print(
                f"  Min/Max: {np.min(total_times):.4f}s / {np.max(total_times):.4f}s")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
