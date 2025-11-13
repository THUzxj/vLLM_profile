#!/usr/bin/env python3
"""
Test vLLM inference with multiple independent LLM instances,
each running on a separate CUDA green context stream in parallel.

This script:
1. Splits GPU resources using CUDA Green Contexts into multiple streams
2. Creates a separate vLLM instance on each stream
3. Launches worker threads for each stream to run inference independently
4. Measures per-stream inference latency (TPOT, TTFT, total time)
5. Writes results to CSV file

Key features:
- Each stream has its own LLM instance and worker thread
- Independent inference per stream with separate resource allocation
- Lock-protected CSV writing to avoid conflicts
- Supports testing different SM partition sizes
"""

import argparse
import contextlib
import gc
import os
import threading
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


def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
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


def worker_run_on_stream(
    llm,
    stream_idx: int, 
    stream: torch.cuda.Stream,
    sm_count: int,
    args,
    results: List[dict],
    lock: threading.Lock,
    barrier: threading.Barrier = None
):
    """
    Worker thread that creates an independent LLM instance on a green context stream
    and runs inference benchmarks.
    
    Each worker:
    - Creates its own vLLM instance on the assigned stream
    - Runs inference with different batch sizes
    - Measures TPOT, TTFT, and total time per batch
    - Appends results to shared list (protected by lock)
    """
    try:
        print(f"[worker {stream_idx}] starting on stream with {sm_count} SMs")
        
        # Ensure all operations are on this stream
        with torch.cuda.stream(stream):
            # Create independent LLM instance
            print(f"[worker {stream_idx}] creating LLM instance...")
            # llm = LLM(
            #     model=args.model,
            #     tensor_parallel_size=args.tp_size,
            #     max_model_len=args.max_model_len,
            #     enforce_eager=False,
            #     max_num_seqs=1024,
            #     max_num_batched_tokens=args.batched_tokens,
            #     gpu_memory_utilization=0.45,
            # )
            llm_engine = llm.llm_engine
            
            sampling_params = SamplingParams(
                temperature=1,
                ignore_eos=True,
                max_tokens=args.output_length,
                output_kind=RequestOutputKind.CUMULATIVE,
            )
            
            # Test each batch size
            for batch_size in args.batch_sizes:
                prompt_length = args.prompt_length
                
                # Run multiple repeats
                for repeat_idx in range(args.num_repeat):
                    # Synchronize all workers before starting this repeat
                    if barrier is not None:
                        print(f"[worker {stream_idx}] waiting at barrier for bs={batch_size} rep={repeat_idx}")
                        barrier.wait()
                        print(f"[worker {stream_idx}] barrier released, starting inference")
                    
                    # Clear any unfinished requests
                    while llm_engine.has_unfinished_requests():
                        llm_engine.step()
                    
                    # Submit requests
                    request_ids = []
                    for i in range(batch_size):
                        prompt_token_ids = [randint(0, 8192) for _ in range(prompt_length)]
                        token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
                        request_id = f"w{stream_idx}_bs{batch_size}_r{repeat_idx}_{i}"
                        llm_engine.add_request(
                            request_id=request_id,
                            prompt=token_prompt,
                            params=sampling_params,
                        )
                        request_ids.append(request_id)
                    
                    # Prefill phase - measure TTFT
                    ttft_ms = None
                    prefill_start = time.perf_counter()
                    while llm_engine.has_unfinished_requests() and ttft_ms is None:
                        step_start = time.perf_counter()
                        step_outputs = llm_engine.step()
                        step_end = time.perf_counter()
                        
                        for output in step_outputs:
                            if hasattr(output, 'outputs') and output.outputs:
                                if len(output.outputs[0].token_ids) > 0:
                                    ttft_ms = (step_end - step_start) * 1000  # ms
                                    break
                        
                        if ttft_ms is not None:
                            break
                    
                    # Decode phase - measure TPOT
                    decode_steps = 0
                    step_times = []
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
                    
                    total_time_ms = (decode_end - prefill_start) * 1000  # ms
                    
                    # Record result
                    result_row = {
                        "repeat_idx": repeat_idx,
                        "stream_idx": stream_idx,
                        "sm_count": sm_count,
                        "batch_size": batch_size,
                        "prompt_length": prompt_length,
                        "decode_steps": decode_steps,
                        "tpot_ms": tpot_ms,
                        "ttft_ms": ttft_ms if ttft_ms else 0,
                        "total_time_ms": total_time_ms,
                        "finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    }
                    
                    with lock:
                        results.append(result_row)
                    
                    print(f"[worker {stream_idx}] bs={batch_size} rep={repeat_idx} "
                          f"tpot={tpot_ms:.2f}ms ttft={ttft_ms:.2f}ms total={total_time_ms:.2f}ms")
            
            # Cleanup this worker's LLM instance
            del llm
            torch.cuda.empty_cache()
            cleanup()
            torch.cuda.synchronize()
            print(f"[worker {stream_idx}] finished and cleaned up")
    
    except Exception as e:
        print(f"[worker {stream_idx}] error: {e}")
        import traceback
        traceback.print_exc()


def benchmark_vllm_parallel_green_ctx(args):
    """
    Benchmark vLLM with multiple independent instances on separate green context streams.
    """
    np.random.seed(42)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create output CSV
    output_path = os.path.join(args.log_dir, args.log_path)
    df_header = pd.DataFrame(columns=[
        "repeat_idx",
        "stream_idx", 
        "sm_count", 
        "batch_size", 
        "prompt_length",
        "decode_steps",
        "tpot_ms",
        "ttft_ms",
        "total_time_ms",
        "finish_time"
    ])
    df_header.to_csv(output_path, index=False)
    
    dev = torch.device("cuda:0")
    
    # Create green context streams
    num_streams = args.num_streams
    min_sm_per_stream = args.min_sm_per_stream
    
    print(f"\n{'='*60}")
    print(f"Creating {num_streams} green context streams with min {min_sm_per_stream} SMs each")
    print(f"{'='*60}\n")
    
    streams, resources = split_device_green_ctx(dev, num_streams, min_sm_per_stream)
    
    # Use first num_streams resources/streams
    # streams = streams[:num_streams]
    # resources = resources[:num_streams]
    
    # Get SM counts for each resource
    sm_counts = []
    for res in resources:
        try:
            sm_count = res.sm.smCount if hasattr(res, 'sm') and hasattr(res.sm, 'smCount') else -1
        except Exception:
            sm_count = -1
        sm_counts.append(sm_count)
        print(f"Stream resource allocated: {sm_count} SMs")
    
    print(f"streams number: {len(streams)}, SM counts: {sm_counts}\n")

    # Shared results list and lock
    results = []
    results_lock = threading.Lock()
    threads = []
    
    # Create barrier for synchronizing all worker threads
    # Barrier count = num_threads + 1 (main thread waits too to ensure all ready)
    barrier = threading.Barrier(num_streams)

    # Lauch LLM engines
    print("Launching LLM engines on each stream...")

    llms = []
    for stream_idx, (stream, sm_count) in enumerate(zip(streams, sm_counts)):
        with torch.cuda.stream(stream):
            print(f"[main] creating LLM engine for stream {stream_idx} with {sm_count} SMs")
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tp_size,
                max_model_len=args.max_model_len,
                enforce_eager=False,
                max_num_seqs=1024,
                max_num_batched_tokens=args.batched_tokens,
                gpu_memory_utilization=0.45,
            )
            llms.append(llm)
            print(f"[main] LLM engine for stream {stream_idx} created")

    
    # Launch worker threads
    print(f"\nLaunching {num_streams} worker threads...")
    for stream_idx, (stream, sm_count) in enumerate(zip(streams, sm_counts)):
        t = threading.Thread(
            target=worker_run_on_stream,
            args=(llms[stream_idx], stream_idx, stream, sm_count, args, results, results_lock, barrier),
            daemon=False
        )
        t.start()
        threads.append(t)
    
    # Wait for all workers to finish
    print("Waiting for workers to finish...")
    for t in threads:
        t.join()
    
    # Write all results to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_path, mode='a', header=False, index=False)
        print(f"\n{'='*60}")
        print(f"Saved {len(results)} measurements to {output_path}")
        print(f"{'='*60}\n")
    else:
        print("No results were collected.")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel vLLM benchmarking with independent instances on green context streams'
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
    parser.add_argument('--num-streams', type=int, default=1,
                        help='Number of parallel streams to create')
    parser.add_argument('--min-sm-per-stream', type=int, default=32,
                        help='Minimum SMs per stream')
    parser.add_argument('--num-repeat', type=int, default=2,
                        help='Number of repeats for each batch size')
    parser.add_argument('--log-dir', type=str, default='parallel_green_ctx_results',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str, 
                        default='vllm_parallel_green_ctx_benchmark.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    print("="*60)
    print("vLLM Parallel Green Context Benchmark")
    print("Multiple Independent LLM Instances on Separate Streams")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tp_size}")
    print(f"Num Streams: {args.num_streams}")
    print(f"Min SMs per Stream: {args.min_sm_per_stream}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Num repeats: {args.num_repeat}")
    print("="*60)
    
    benchmark_vllm_parallel_green_ctx(args)


if __name__ == "__main__":
    main()
