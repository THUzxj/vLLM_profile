#!/usr/bin/env python3
"""
Test vLLM inference with different GPU resource partitioning using Green Contexts.

This script:
1. Splits GPU resources using CUDA Green Contexts
2. Creates vLLM instances on different streams
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
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind

from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"


def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    # if not is_cpu():
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


def benchmark_vllm_with_green_ctx(args):
    """
    Benchmark vLLM inference with different GPU resource partitioning.
    """
    np.random.seed(42)
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
        "decode_steps",
        "tpot_ms",
        "ttft_ms",
        "total_time_ms",
        "config_id"
    ])
    df_header.to_csv(output_path, index=False)
    
    # Test different SM partition sizes
    sm_partition_sizes = args.sm_partition_sizes  # e.g., [16, 32, 48, 64]
    
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
                        f"vllm_green_ctx_sm{sm_size}_trace.json"
                    )
                )
                print(f"Trace for SM partition {sm_size} saved.")
            
            with torch.cuda.stream(main_stream):
                # Create vLLM instance
                llm = LLM(
                    model=args.model,
                    tensor_parallel_size=args.tp_size,
                    max_model_len=args.max_model_len,
                    enforce_eager=False,
                    max_num_seqs=1024,
                    max_num_batched_tokens=args.batched_tokens,
                    gpu_memory_utilization=0.45,
                )
                llm_engine = llm.llm_engine
                # Run profiling only when requested. Otherwise run the same workload
                # without torch/vLLM profiling to avoid trace export and overhead.
                if args.profile:
                    llm.start_profile()

                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA,
                                    torch.profiler.ProfilerActivity.CPU],
                        with_stack=True,
                        record_shapes=True,
                        on_trace_ready=trace_handler
                    ) as prof:
                        print(f"Creating vLLM instance on stream with {main_sm_count} SMs...")
                        # Test the stream
                        a = torch.randn(4096*1024, device=torch.cuda.current_device())
                        b = torch.randn(4096*1024, device=torch.cuda.current_device())
                        c = torch.matmul(a, b)

                        # Test different batch sizes
                        for batch_size in args.batch_sizes:
                            prompt_length = args.prompt_length
                            output_length = args.output_length

                            sampling_params = SamplingParams(
                                temperature=1,
                                ignore_eos=True,
                                max_tokens=output_length,
                                output_kind=RequestOutputKind.CUMULATIVE,
                            )

                            print(f"\n  Batch size: {batch_size}")

                            # Run multiple repeats
                            for repeat_idx in range(args.num_repeat):
                                # Clear previous requests
                                while llm_engine.has_unfinished_requests():
                                    llm_engine.step()

                                # Add requests
                                request_ids = []
                                for i in range(batch_size):
                                    # Create prompt tokens
                                    prompt_token_ids = [randint(0, 8192) for _ in range(prompt_length)]
                                    token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

                                    request_id = f"req_{partition_idx}_{sm_size}_{batch_size}_{repeat_idx}_{i}"
                                    llm_engine.add_request(
                                        request_id=request_id,
                                        prompt=token_prompt,
                                        params=sampling_params,
                                    )
                                    request_ids.append(request_id)

                                ttft = None
                                decode_steps = 0
                                step_times = []

                                # Prefill phase
                                prefill_start = time.perf_counter()
                                while llm_engine.has_unfinished_requests() and ttft is None:
                                    step_start = time.perf_counter()
                                    step_outputs = llm_engine.step()
                                    step_end = time.perf_counter()

                                    for output in step_outputs:
                                        if hasattr(output, 'outputs') and output.outputs:
                                            if len(output.outputs[0].token_ids) > 0:
                                                ttft = (step_end - step_start) * 1000  # ms
                                                break

                                    if ttft is not None:
                                        break

                                # Decode phase
                                decode_start = time.perf_counter()
                                while llm_engine.has_unfinished_requests():
                                    step_start = time.perf_counter()
                                    step_outputs = llm_engine.step()
                                    step_end = time.perf_counter()

                                    decode_steps += 1
                                    step_times.append((step_end - step_start) * 1000)  # ms

                                decode_end = time.perf_counter()

                                # Calculate metrics
                                if decode_steps > 0:
                                    tpot_ms = np.mean(step_times)
                                else:
                                    tpot_ms = 0

                                total_time = (decode_end - prefill_start) * 1000  # ms

                                print(f"    Repeat {repeat_idx + 1}: TPOT={tpot_ms:.2f}ms, "
                                    f"TTFT={ttft:.2f}ms, Total={total_time:.2f}ms")

                            # Write result to CSV
                            result_row = pd.DataFrame([{
                                "sm_partition_count": sm_size,
                                "stream_id": 0,
                                "sm_count": main_sm_count,
                                "batch_size": batch_size,
                                "prompt_length": prompt_length,
                                "decode_steps": decode_steps,
                                "tpot_ms": tpot_ms,
                                "ttft_ms": ttft if ttft else 0,
                                "total_time_ms": total_time,
                                "config_id": f"{partition_idx}_{sm_size}"
                            }])
                            result_row.to_csv(output_path, mode='a', header=False, index=False)

                    llm.stop_profile()
                else:
                    # No profiling: run the exact same workload without profiler
                    print(f"Creating vLLM instance on stream with {main_sm_count} SMs (no profile)...")
                    # Test the stream
                    a = torch.randn(4096*1024, device=torch.cuda.current_device())
                    b = torch.randn(4096*1024, device=torch.cuda.current_device())
                    c = torch.matmul(a, b)

                    # Test different batch sizes
                    for batch_size in args.batch_sizes:
                        prompt_length = args.prompt_length
                        output_length = args.output_length

                        sampling_params = SamplingParams(
                            temperature=1,
                            ignore_eos=True,
                            max_tokens=output_length,
                            output_kind=RequestOutputKind.CUMULATIVE,
                        )

                        print(f"\n  Batch size: {batch_size}")

                        # Run multiple repeats
                        for repeat_idx in range(args.num_repeat):
                            # Clear previous requests
                            while llm_engine.has_unfinished_requests():
                                llm_engine.step()

                            # Add requests
                            request_ids = []
                            for i in range(batch_size):
                                # Create prompt tokens
                                prompt_token_ids = [randint(0, 8192) for _ in range(prompt_length)]
                                token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

                                request_id = f"req_{partition_idx}_{sm_size}_{batch_size}_{repeat_idx}_{i}"
                                llm_engine.add_request(
                                    request_id=request_id,
                                    prompt=token_prompt,
                                    params=sampling_params,
                                )
                                request_ids.append(request_id)

                            ttft = None
                            decode_steps = 0
                            step_times = []

                            # Prefill phase
                            prefill_start = time.perf_counter()
                            while llm_engine.has_unfinished_requests() and ttft is None:
                                step_start = time.perf_counter()
                                step_outputs = llm_engine.step()
                                step_end = time.perf_counter()

                                for output in step_outputs:
                                    if hasattr(output, 'outputs') and output.outputs:
                                        if len(output.outputs[0].token_ids) > 0:
                                            ttft = (step_end - step_start) * 1000  # ms
                                            break

                                if ttft is not None:
                                    break

                            # Decode phase
                            decode_start = time.perf_counter()
                            while llm_engine.has_unfinished_requests():
                                step_start = time.perf_counter()
                                step_outputs = llm_engine.step()
                                step_end = time.perf_counter()

                                decode_steps += 1
                                step_times.append((step_end - step_start) * 1000)  # ms

                            decode_end = time.perf_counter()

                            # Calculate metrics
                            if decode_steps > 0:
                                tpot_ms = np.mean(step_times)
                            else:
                                tpot_ms = 0

                            total_time = (decode_end - prefill_start) * 1000  # ms

                            print(f"    Repeat {repeat_idx + 1}: TPOT={tpot_ms:.2f}ms, "
                                f"TTFT={ttft:.2f}ms, Total={total_time:.2f}ms")

                        # Write result to CSV
                        result_row = pd.DataFrame([{
                            "sm_partition_count": sm_size,
                            "stream_id": 0,
                            "sm_count": main_sm_count,
                            "batch_size": batch_size,
                            "prompt_length": prompt_length,
                            "decode_steps": decode_steps,
                            "tpot_ms": tpot_ms,
                            "ttft_ms": ttft if ttft else 0,
                            "total_time_ms": total_time,
                            "config_id": f"{partition_idx}_{sm_size}"
                        }])
                        result_row.to_csv(output_path, mode='a', header=False, index=False)
                
                # Clean up
                del llm
                torch.cuda.empty_cache()
                cleanup()

                torch.cuda.synchronize()
                
        except RuntimeError as e:
            print(f"Error with SM partition size {sm_size}: {e}")
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
    
    # Plot 1: TPOT vs SM partition count (grouped by batch size)
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
    
    # Plot 2: TTFT vs SM partition count
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
    
    # Plot 3: Total time vs batch size (grouped by partition)
    ax3 = axes[1, 0]
    for partition in sorted(df['sm_partition_count'].unique()):
        data = df[df['sm_partition_count'] == partition]
        data_grouped = data.groupby('batch_size')['total_time_ms'].mean()
        ax3.plot(data_grouped.index, data_grouped.values, marker='s', label=f'SM={partition}')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Total Time (ms)')
    ax3.set_title('Total Inference Time vs Batch Size')
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
    output_file = os.path.join(output_dir, 'vllm_green_ctx_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark vLLM with different GPU resource partitioning'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path')
    parser.add_argument('--tp-size', type=int, default=1,
                        help='Tensor parallel size')
    parser.add_argument('--max-model-len', type=int, default=2048,
                        help='Max model length')
    parser.add_argument('--batched-tokens', type=int, default=4096,
                        help='Max batched tokens')
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
    parser.add_argument('--warmup-iters', type=int, default=2,
                        help='Number of warmup iterations')
    parser.add_argument('--warmup-bs', type=int, default=2,
                        help='Batch size for warmup')
    parser.add_argument('--log-dir', type=str, default='profile_green_ctx',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str, 
                        default='vllm_green_ctx_benchmark.csv',
                        help='Output CSV filename')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling and trace export')
    
    args = parser.parse_args()
    
    print("="*60)
    print("vLLM Green Context GPU Partitioning Benchmark")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tp_size}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"SM partition sizes: {args.sm_partition_sizes}")
    print(f"Num repeats: {args.num_repeat}")
    print(f"Profiling enabled: {args.profile}")
    print("="*60)
    
    # Run benchmark
    benchmark_vllm_with_green_ctx(args)
    
    # Generate plots if requested
    if args.plot:
        output_path = os.path.join(args.log_dir, args.log_path)
        create_comparison_plot(output_path, args.log_dir)


if __name__ == "__main__":
    main()
