#!/usr/bin/env python3
"""
Test Transformers inference with different GPU resource partitioning using Green Contexts.

This script:
1. Splits GPU resources using CUDA Green Contexts
2. Creates Transformers model instances on different streams
3. Measures inference latency under different resource configurations
"""

import argparse
import contextlib
import gc
import os
import time
from typing import List, Tuple
from random import randint

import cuda.bindings.driver as driver
import cuda.bindings.runtime as runtime
import cuda.cudart as cudart
import cuda.nvrtc as nvrtc
import torch
import numpy as np
import pandas as pd
from cuda.bindings.driver import CUdevice, CUdevResource
from transformers import AutoModelForCausalLM, AutoTokenizer

# os.environ["TORCH_PROFILER_DIR"] = "./torch_profile"


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


def measure_generation_with_timing(model, tokenizer, inputs, output_length, device):
    """
    Generate tokens and measure TTFT (Time To First Token) and TPOT (Time Per Output Token).
    
    Returns:
        ttft_ms: Time to first token in milliseconds
        tpot_ms: Average time per output token (excluding first token)
        total_time_ms: Total generation time
        generated_tokens: Number of generated tokens
    """
    # Support batch generation: compute per-sample TTFT and TPOT and return their averages.
    batch_input_ids = inputs["input_ids"].clone()
    batch_size = batch_input_ids.shape[0]

    first_token_times = [None] * batch_size
    subsequent_token_times = [[] for _ in range(batch_size)]
    generated_counts = [0] * batch_size

    gen_start = time.perf_counter()

    with torch.no_grad():
        input_ids = batch_input_ids
        attention_mask = inputs.get("attention_mask", None)

        for token_idx in range(output_length):
            token_gen_start = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
            token_gen_end = time.perf_counter()

            token_time_ms = (token_gen_end - token_gen_start) * 1000

            # Update per-sample timings and input_ids
            for i in range(batch_size):
                nid = int(next_token_ids[i, 0].item())
                if first_token_times[i] is None:
                    first_token_times[i] = token_time_ms
                else:
                    subsequent_token_times[i].append(token_time_ms)

                generated_counts[i] += 1

            # Append generated tokens to input_ids for next step
            input_ids = torch.cat([input_ids, next_token_ids.to(input_ids.device)], dim=-1)

            # If all samples have produced EOS, stop early
            if all((int(next_token_ids[i, 0].item()) == tokenizer.eos_token_id) for i in range(batch_size)):
                break

    gen_end = time.perf_counter()

    # Aggregate metrics across batch
    valid_first = [t for t in first_token_times if t is not None]
    ttft_ms = float(np.mean(valid_first)) if valid_first else 0.0

    all_subsequent = [t for lst in subsequent_token_times for t in lst]
    tpot_ms = float(np.mean(all_subsequent)) if all_subsequent else 0.0

    total_time_ms = (gen_end - gen_start) * 1000
    # Average generated tokens per sample
    generated_tokens = float(np.mean(generated_counts))

    return ttft_ms, tpot_ms, total_time_ms, generated_tokens


def benchmark_transformers_with_green_ctx(args):
    """
    Benchmark Transformers inference with different GPU resource partitioning.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.log_dir, exist_ok=True)
    
    dev = torch.device("cuda:0")
    
    # Create output CSV file
    output_path = os.path.join(args.log_dir, args.log_path)
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
        "config_id"
    ])
    df_header.to_csv(output_path, index=False)
    
    # Load tokenizer once
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test different SM partition sizes
    sm_partition_sizes = args.sm_partition_sizes
    
    for partition_idx, sm_size in enumerate(sm_partition_sizes):
        try:
            print(f"\n{'='*60}")
            print(f"Testing SM partition size: {sm_size}")
            print(f"{'='*60}")
            
            # Split GPU resources
            streams, resources = split_device_green_ctx(dev, 1, sm_size)
            sm_counts = [r.sm.smCount for r in resources]
            print(f"SM counts: {sm_counts}")
            
            # Test on the first stream (main inference stream)
            main_stream = streams[0]
            main_sm_count = sm_counts[0]
            
            # Set the stream as current
            def trace_handler(prof):
                prof.export_chrome_trace(
                    os.path.join(
                        args.log_dir,
                        f"transformers_green_ctx_sm{sm_size}_trace.json"
                    )
                )
                print(f"Trace for SM partition {sm_size} saved.")
            
            with torch.cuda.stream(main_stream):
                # Load model
                print(f"Loading model {args.model}...")

                model_kwargs = {
                    "attn_implementation": args.attention_impl
                }
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.float16,
                    device_map="cuda:0",
                    trust_remote_code=True,
                    **model_kwargs
                )
                model.eval()

                # warm up: perform one inference
                prompt_token_ids = [[randint(100, 8000) for _ in range(args.prompt_length)] 
                                                    for _ in range(1)]

                # Pad sequences to same length
                inputs = tokenizer.pad(
                    {"input_ids": prompt_token_ids},
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(dev) for k, v in inputs.items()}

                ttft_ms, tpot_ms, total_time_ms, generation_steps = measure_generation_with_timing(
                        model, tokenizer, inputs, args.output_length, dev
                    )
                
                # Run profiling only when requested. Otherwise run the same workload
                # without torch profiling to avoid trace export and overhead.
                if args.profile:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA,
                                    torch.profiler.ProfilerActivity.CPU],
                        with_stack=True,
                        record_shapes=True,
                        on_trace_ready=trace_handler
                    ) as prof:
                        print(f"Model on stream with {main_sm_count} SMs...")
                        # Warm up: Test the stream with matrix operations
                        # a = torch.randn(4096*1024, device=torch.cuda.current_device(), dtype=torch.float16)
                        # b = torch.randn(4096*1024, device=torch.cuda.current_device(), dtype=torch.float16)
                        # c = torch.matmul(a, b)

                        # Test different batch sizes
                        for batch_size in args.batch_sizes:
                            prompt_length = args.prompt_length
                            output_length = args.output_length

                            print(f"\n  Batch size: {batch_size}")

                            # Run multiple repeats
                            for repeat_idx in range(args.num_repeat):
                                # Create random prompt tokens
                                prompt_token_ids = [[randint(100, 8000) for _ in range(prompt_length)] 
                                                    for _ in range(batch_size)]
                                
                                # Pad sequences to same length
                                inputs = tokenizer.pad(
                                    {"input_ids": prompt_token_ids},
                                    padding=True,
                                    return_tensors="pt"
                                )
                                inputs = {k: v.to(dev) for k, v in inputs.items()}
                                
                                # Measure generation time with TTFT and TPOT (supports batch)
                                ttft_ms, tpot_ms, total_time_ms, generation_steps = measure_generation_with_timing(
                                    model, tokenizer, inputs, output_length, dev
                                )
                                
                                print(f"    Repeat {repeat_idx + 1}: "
                                      f"TTFT={ttft_ms:.2f}ms, "
                                      f"TPOT={tpot_ms:.2f}ms, "
                                      f"Total={total_time_ms:.2f}ms, "
                                      f"Generated tokens={generation_steps}")

                                # Write result to CSV
                                result_row = pd.DataFrame([{
                                    "sm_partition_count": sm_size,
                                    "stream_id": 0,
                                    "sm_count": main_sm_count,
                                    "batch_size": batch_size,
                                    "prompt_length": prompt_length,
                                    "generation_steps": generation_steps,
                                    "ttft_ms": ttft_ms,
                                    "tpot_ms": tpot_ms,
                                    "total_time_ms": total_time_ms,
                                    "config_id": f"{partition_idx}_{sm_size}"
                                }])
                                result_row.to_csv(output_path, mode='a', header=False, index=False)
                
                else:
                    # No profiling: run the exact same workload without profiler
                    print(f"Model on stream with {main_sm_count} SMs (no profile)...")
                    # Warm up: Test the stream
                    a = torch.randn(4096*1024, device=torch.cuda.current_device(), dtype=torch.float16)
                    b = torch.randn(4096*1024, device=torch.cuda.current_device(), dtype=torch.float16)
                    c = torch.matmul(a, b)

                    # Test different batch sizes
                    for batch_size in args.batch_sizes:
                        prompt_length = args.prompt_length
                        output_length = args.output_length

                        print(f"\n  Batch size: {batch_size}")

                        # Run multiple repeats
                        for repeat_idx in range(args.num_repeat):
                            # Create random prompt tokens
                            prompt_token_ids = [[randint(100, 8000) for _ in range(prompt_length)] 
                                                for _ in range(batch_size)]
                            
                            # Pad sequences to same length
                            inputs = tokenizer.pad(
                                {"input_ids": prompt_token_ids},
                                padding=True,
                                return_tensors="pt"
                            )
                            inputs = {k: v.to(dev) for k, v in inputs.items()}
                            
                            # Measure generation time with TTFT and TPOT (supports batch)
                            ttft_ms, tpot_ms, total_time_ms, generation_steps = measure_generation_with_timing(
                                model, tokenizer, inputs, output_length, dev
                            )
                            
                            print(f"    Repeat {repeat_idx + 1}: "
                                  f"TTFT={ttft_ms:.2f}ms, "
                                  f"TPOT={tpot_ms:.2f}ms, "
                                  f"Total={total_time_ms:.2f}ms, "
                                  f"Generated tokens={generation_steps}")

                            # Write result to CSV
                            result_row = pd.DataFrame([{
                                "sm_partition_count": sm_size,
                                "stream_id": 0,
                                "sm_count": main_sm_count,
                                "batch_size": batch_size,
                                "prompt_length": prompt_length,
                                "generation_steps": generation_steps,
                                "ttft_ms": ttft_ms,
                                "tpot_ms": tpot_ms,
                                "total_time_ms": total_time_ms,
                                "config_id": f"{partition_idx}_{sm_size}"
                            }])
                            result_row.to_csv(output_path, mode='a', header=False, index=False)
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                cleanup()

                torch.cuda.synchronize()
                
        except RuntimeError as e:
            print(f"Error with SM partition size {sm_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Benchmark completed! Results saved to: {output_path}")
    print(f"{'='*60}")


def create_comparison_plot(csv_path, output_dir=None):
    """Create visualization comparing different GPU partitions."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Generation time vs SM partition count (grouped by batch size)
    ax1 = axes[0, 0]
    for bs in sorted(df['batch_size'].unique()):
        data = df[df['batch_size'] == bs]
        data_grouped = data.groupby('sm_partition_count')['tpot_ms'].mean()
        ax1.plot(data_grouped.index, data_grouped.values, marker='o', label=f'BS={bs}')
    ax1.set_xlabel('SM Partition Count')
    ax1.set_ylabel('TPOT (ms)')
    ax1.set_title('TPOT vs GPU Partition Size')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Total time vs SM partition count
    ax2 = axes[0, 1]
    for bs in sorted(df['batch_size'].unique()):
        data = df[df['batch_size'] == bs]
        data_grouped = data.groupby('sm_partition_count')['ttft_ms'].mean()
        ax2.plot(data_grouped.index, data_grouped.values, marker='o', label=f'BS={bs}')
    ax2.set_xlabel('SM Partition Count')
    ax2.set_ylabel('TTFT (ms)')
    ax2.set_title('TTFT vs GPU Partition Size')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Generation time vs batch size (grouped by partition)
    ax3 = axes[1, 0]
    for partition in sorted(df['sm_partition_count'].unique()):
        data = df[df['sm_partition_count'] == partition]
        data_grouped = data.groupby('batch_size')['tpot_ms'].mean()
        ax3.plot(data_grouped.index, data_grouped.values, marker='s', label=f'SM={partition}')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('TPOT (ms)')
    ax3.set_title('TPOT vs Batch Size')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Heatmap of TPOT
    ax4 = axes[1, 1]
    pivot_data = df.pivot_table(
        values='tpot_ms',
        index='batch_size',
        columns='sm_partition_count',
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'TPOT (ms)'})
    ax4.set_title('TPOT Heatmap: Batch Size vs SM Partition')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'transformers_green_ctx_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Transformers with different GPU resource partitioning'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--prompt-length', type=int, default=512,
                        help='Prompt length')
    parser.add_argument('--output-length', type=int, default=128,
                        help='Output length to generate')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Batch sizes to test')
    parser.add_argument('--sm-partition-sizes', type=int, nargs='+',
                        default=[16, 32, 48, 64, 80],
                        help='SM partition sizes to test')
    parser.add_argument('--num-repeat', type=int, default=3,
                        help='Number of repeats for each configuration')
    parser.add_argument('--log-dir', type=str, default='profile_transformers_green_ctx',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str, 
                        default='transformers_green_ctx_benchmark.csv',
                        help='Output CSV filename')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling and trace export')
    parser.add_argument('--attention-impl', type=str, default='eager',
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help='Preferred attention implementation to enable on the model (best-effort)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Transformers Green Context GPU Partitioning Benchmark")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"SM partition sizes: {args.sm_partition_sizes}")
    print(f"Num repeats: {args.num_repeat}")
    print(f"Profiling enabled: {args.profile}")
    print("="*60)
    
    # Run benchmark
    benchmark_transformers_with_green_ctx(args)
    
    # Generate plots if requested
    if args.plot:
        output_path = os.path.join(args.log_dir, args.log_path)
        create_comparison_plot(output_path, args.log_dir)


if __name__ == "__main__":
    main()
