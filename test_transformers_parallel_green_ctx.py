#!/usr/bin/env python3
"""
Test Transformers inference with multiple independent model instances,
each running on a separate CUDA green context stream in parallel.

This script:
1. Splits GPU resources using CUDA Green Contexts into multiple streams
2. Creates a separate Transformers model instance on each stream
3. Launches worker threads for each stream to run inference independently
4. Measures per-stream inference latency (TPOT, TTFT, total time)
5. Writes results to CSV file

Key features:
- Each stream has its own model instance and worker thread
- Independent inference per stream with separate resource allocation
- Lock-protected CSV writing to avoid conflicts
- Supports testing different SM partition sizes
- Measures TTFT and TPOT for all batch sizes
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
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    Supports batch inputs and computes per-sample metrics then averages them.
    
    Returns:
        ttft_ms: Average time to first token in milliseconds
        tpot_ms: Average time per output token (excluding first token)
        total_time_ms: Total generation time
        generated_tokens: Average number of generated tokens per sample
    """
    batch_input_ids = inputs["input_ids"].clone()
    batch_size = batch_input_ids.shape[0]

    first_token_times = [None] * batch_size
    subsequent_token_times = [[] for _ in range(batch_size)]
    generated_counts = [0] * batch_size

    gen_start = time.perf_counter()

    with torch.no_grad():
        input_ids = batch_input_ids.clone()
        attention_mask = inputs.get("attention_mask", None)
        
        for token_idx in range(output_length):
            token_gen_start = time.perf_counter()
            
            # Generate one token
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            token_gen_end = time.perf_counter()
            token_time_ms = (token_gen_end - token_gen_start) * 1000
            
            # Record timing for each sample in batch
            for batch_idx in range(batch_size):
                if token_idx == 0:
                    first_token_times[batch_idx] = token_time_ms
                else:
                    subsequent_token_times[batch_idx].append(token_time_ms)
                
                generated_counts[batch_idx] += 1
                
                # Check for EOS token
                if next_token_ids[batch_idx].item() == tokenizer.eos_token_id:
                    # Keep padding subsequent tokens for consistent measurement
                    pass
            
            # Append generated tokens
            input_ids = torch.cat([input_ids, next_token_ids], dim=-1)

    gen_end = time.perf_counter()

    # Aggregate metrics across batch
    valid_first = [t for t in first_token_times if t is not None]
    ttft_ms = float(np.mean(valid_first)) if valid_first else 0.0

    all_subsequent = [t for lst in subsequent_token_times for t in lst]
    tpot_ms = float(np.mean(all_subsequent)) if all_subsequent else 0.0

    total_time_ms = (gen_end - gen_start) * 1000
    # Average generated tokens per sample
    generated_tokens = float(np.mean(generated_counts))

    return ttft_ms, tpot_ms, total_time_ms, int(generated_tokens)


def worker_run_on_stream(
    model,
    tokenizer,
    stream_idx: int, 
    stream: torch.cuda.Stream,
    sm_count: int,
    args,
    results: List[dict],
    lock: threading.Lock,
    barrier: threading.Barrier = None
):
    """
    Worker thread that creates an independent model instance on a green context stream
    and runs inference benchmarks.
    
    Each worker:
    - Uses a model instance on the assigned stream
    - Runs inference with different batch sizes
    - Measures TPOT, TTFT, and total time per batch
    - Appends results to shared list (protected by lock)
    """
    try:
        print(f"[worker {stream_idx}] starting on stream with {sm_count} SMs")
        
        # Ensure all operations are on this stream
        with torch.cuda.stream(stream):
            print(f"[worker {stream_idx}] model ready for inference")
            
            # warm up: perform one inference
            prompt_token_ids = [[randint(100, 8000) for _ in range(args.prompt_length)] 
                                                for _ in range(1)]

            # Pad sequences to same length
            inputs = tokenizer.pad(
                {"input_ids": prompt_token_ids},
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(stream.device) for k, v in inputs.items()}

            ttft_ms, tpot_ms, total_time_ms, generation_steps = measure_generation_with_timing(
                    model, tokenizer, inputs, args.output_length, stream.device
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
                    
                    # Create random prompt tokens
                    prompt_token_ids = [[randint(100, 8000) for _ in range(prompt_length)] 
                                        for _ in range(batch_size)]
                    
                    # Pad sequences to same length
                    inputs = tokenizer.pad(
                        {"input_ids": prompt_token_ids},
                        padding=True,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(stream.device) for k, v in inputs.items()}
                    
                    # Measure generation with TTFT and TPOT
                    ttft_ms, tpot_ms, total_time_ms, generation_steps = measure_generation_with_timing(
                        model, tokenizer, inputs, args.output_length, stream.device
                    )
                    
                    # Record result
                    result_row = {
                        "repeat_idx": repeat_idx,
                        "stream_idx": stream_idx,
                        "sm_count": sm_count,
                        "batch_size": batch_size,
                        "prompt_length": prompt_length,
                        "generation_steps": generation_steps,
                        "ttft_ms": ttft_ms,
                        "tpot_ms": tpot_ms,
                        "total_time_ms": total_time_ms,
                        "finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    }
                    
                    with lock:
                        results.append(result_row)
                    
                    print(f"[worker {stream_idx}] bs={batch_size} rep={repeat_idx} "
                          f"tpot={tpot_ms:.2f}ms ttft={ttft_ms:.2f}ms total={total_time_ms:.2f}ms")
            
            torch.cuda.empty_cache()
            cleanup()
            torch.cuda.synchronize()
            print(f"[worker {stream_idx}] finished and cleaned up")
    
    except Exception as e:
        print(f"[worker {stream_idx}] error: {e}")
        import traceback
        traceback.print_exc()


def benchmark_transformers_parallel_green_ctx(args):
    """
    Benchmark Transformers with multiple independent instances on separate green context streams.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create output CSV
    output_path = os.path.join(args.log_dir, args.log_path)
    df_header = pd.DataFrame(columns=[
        "repeat_idx",
        "stream_idx", 
        "sm_count", 
        "batch_size", 
        "prompt_length",
        "generation_steps",
        "ttft_ms",
        "tpot_ms",
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

    # Load tokenizer once
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "attn_implementation": args.attention_impl if hasattr(args, 'attention_impl') else 'default'
    }

    # Load model once (will be shared across streams)
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        **model_kwargs
    )
    model.eval()
    print("Model loaded successfully")

    # Shared results list and lock
    results = []
    results_lock = threading.Lock()
    threads = []
    
    # Create barrier for synchronizing all worker threads
    barrier = threading.Barrier(num_streams)
    
    # Launch worker threads
    print(f"\nLaunching {num_streams} worker threads...")
    for stream_idx, (stream, sm_count) in enumerate(zip(streams, sm_counts)):
        t = threading.Thread(
            target=worker_run_on_stream,
            args=(model, tokenizer, stream_idx, stream, sm_count, args, results, results_lock, barrier),
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
    
    # Clean up
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Transformers benchmarking with independent instances on green context streams'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path or HuggingFace model ID')
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
    parser.add_argument('--log-dir', type=str, default='parallel_transformers_green_ctx_results',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str, 
                        default='transformers_parallel_green_ctx_benchmark.csv',
                        help='Output CSV filename')
    parser.add_argument('--attention-impl', type=str, default='default',
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help='Preferred attention implementation to enable on the model (best-effort)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Transformers Parallel Green Context Benchmark")
    print("Multiple Independent Model Instances on Separate Streams")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Num Streams: {args.num_streams}")
    print(f"Min SMs per Stream: {args.min_sm_per_stream}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Num repeats: {args.num_repeat}")
    print("="*60)
    
    benchmark_transformers_parallel_green_ctx(args)


if __name__ == "__main__":
    main()
