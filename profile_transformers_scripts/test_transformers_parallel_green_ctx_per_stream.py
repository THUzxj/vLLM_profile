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
- Each stream can have different batch_size, prompt_length, and output_length
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


def measure_generation_with_timing(model, tokenizer, inputs, output_length, device, stream=None):
    """
    Generate tokens and measure TTFT (Time To First Token) and TPOT (Time Per Output Token).
    Supports batch inputs and computes per-sample metrics then averages them.

    Args:
        model: The model to use
        tokenizer: The tokenizer
        inputs: Input dictionary with input_ids and attention_mask
        output_length: Number of tokens to generate
        device: Device to run on
        stream: CUDA stream to use for synchronization (optional)

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
        past_key_values = None
        attention_mask = inputs.get("attention_mask", None)
        current_attention_mask = attention_mask  # Initialize for first iteration

        layer_records = []
        for token_idx in range(output_length):

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

            if stream is not None:
                stream.synchronize()
            else:
                torch.cuda.synchronize()
            token_gen_end = time.perf_counter()
            if isinstance(out, tuple):
                outputs, layer_record_times = out
            else:
                outputs = out
                layer_record_times = None

            # Save past_key_values for next iteration
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

            if token_idx != 0:
                token_time_ms = (token_gen_end - token_gen_start) * 1000

                # Record timing for each sample in batch
                for batch_idx in range(batch_size):
                    if token_idx == 0:
                        first_token_times[batch_idx] = token_time_ms
                    else:
                        subsequent_token_times[batch_idx].append(token_time_ms)

                    # Check for EOS token
                    if next_token_ids[batch_idx].item() == tokenizer.eos_token_id:
                        # Keep padding subsequent tokens for consistent measurement
                        pass

                if layer_record_times is not None:
                    layer_records.append(layer_record_times)

            for batch_idx in range(batch_size):
                generated_counts[batch_idx] += 1

    gen_end = time.perf_counter()

    # Aggregate metrics across batch
    valid_first = [t for t in first_token_times if t is not None]
    ttft_ms = float(np.mean(valid_first)) if valid_first else 0.0

    all_subsequent = [t for lst in subsequent_token_times for t in lst]
    tpot_ms = float(np.mean(all_subsequent)) if all_subsequent else 0.0

    total_time_ms = (gen_end - gen_start) * 1000
    # Average generated tokens per sample
    generated_tokens = float(np.mean(generated_counts))

    return ttft_ms, tpot_ms, total_time_ms, int(generated_tokens), layer_records


def worker_run_on_stream(
    model,
    tokenizer,
    stream_idx: int,
    stream: torch.cuda.Stream,
    sm_count: int,
    batch_size: int,
    prompt_length: int,
    output_length: int,
    num_repeat: int,
    log_dir: str,
    log_path: str,
    results: List[dict],
    lock: threading.Lock,
    barrier: threading.Barrier = None
):
    """
    Worker thread that creates an independent model instance on a green context stream
    and runs inference benchmarks.

    Each worker:
    - Uses a model instance on the assigned stream
    - Runs inference with stream-specific batch_size, prompt_length, and output_length
    - Measures TPOT, TTFT, and total time per batch
    - Appends results to shared list (protected by lock)
    """
    try:
        print(f"[worker {stream_idx}] starting on stream with {sm_count} SMs")
        print(
            f"[worker {stream_idx}] config: batch_size={batch_size}, prompt_length={prompt_length}, output_length={output_length}")

        # Ensure all operations are on this stream
        with torch.cuda.stream(stream):
            print(f"[worker {stream_idx}] model ready for inference")

            # warm up: perform one inference
            prompt_token_ids = [[randint(100, 8000) for _ in range(prompt_length)]
                                for _ in range(1)]

            # Pad sequences to same length
            inputs = tokenizer.pad(
                {"input_ids": prompt_token_ids},
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(stream.device) for k, v in inputs.items()}

            ttft_ms, tpot_ms, total_time_ms, generation_steps, layer_records = measure_generation_with_timing(
                model, tokenizer, inputs, output_length, stream.device, stream
            )

            # Run multiple repeats
            for repeat_idx in range(num_repeat):

                # Create random prompt tokens
                prompt_token_ids = [[randint(100, 8000) for _ in range(prompt_length)]
                                    for _ in range(batch_size)]

                # Pad sequences to same length
                inputs = tokenizer.pad(
                    {"input_ids": prompt_token_ids},
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(stream.device)
                          for k, v in inputs.items()}

                # Synchronize all workers before starting inference
                if barrier is not None:
                    print(
                        f"[worker {stream_idx}] waiting at barrier for rep={repeat_idx}")
                    barrier.wait()
                    print(
                        f"[worker {stream_idx}] barrier released, starting inference")

                # Measure generation with TTFT and TPOT
                ttft_ms, tpot_ms, total_time_ms, generation_steps, layer_records = measure_generation_with_timing(
                    model, tokenizer, inputs, output_length, stream.device, stream
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

                output_path = os.path.join(log_dir, log_path)
                with lock:
                    # Append single-row CSV without header
                    pd.DataFrame([result_row]).to_csv(
                        output_path, mode='a', header=False, index=False)
                    # Keep in-memory copy for summary if desired
                    results.append(result_row)

                try:
                    json_name = (
                        f"layer_times_stream{stream_idx}_sm{sm_count}_bs{batch_size}_rep{repeat_idx}_"
                        f"pl{prompt_length}_ol{output_length}.json"
                    )
                    json_path = os.path.join(log_dir, json_name)
                    print(
                        f"[worker {stream_idx}] writing JSON to {json_path}")
                    with open(json_path, 'w') as fh:
                        json.dump({
                            "stream_idx": stream_idx,
                            "sm_count": sm_count,
                            "batch_size": batch_size,
                            "repeat_idx": repeat_idx,
                            "prompt_length": prompt_length,
                            "output_length": output_length,
                            "layer_records": layer_records,
                        }, fh)
                except Exception as e:
                    print(f"[worker {stream_idx}] error writing JSON: {e}")
                    pass

                print(f"[worker {stream_idx}] bs={batch_size} pl={prompt_length} ol={output_length} rep={repeat_idx} "
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
    Each stream can have different batch_size, prompt_length, and output_length.
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

    # Validate array lengths
    expected_length = num_streams if args.no_leftover_mode else num_streams + 1

    if len(args.batch_sizes) != expected_length:
        raise ValueError(
            f"batch_sizes length ({len(args.batch_sizes)}) must match number of streams ({expected_length})"
        )
    if len(args.prompt_lengths) != expected_length:
        raise ValueError(
            f"prompt_lengths length ({len(args.prompt_lengths)}) must match number of streams ({expected_length})"
        )
    if len(args.output_lengths) != expected_length:
        raise ValueError(
            f"output_lengths length ({len(args.output_lengths)}) must match number of streams ({expected_length})"
        )

    print(f"\n{'='*60}")
    print(
        f"Creating {num_streams} green context streams with min {min_sm_per_stream} SMs each")
    print(f"{'='*60}\n")

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

    # Load tokenizer once
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Shared results list and lock
    results = []
    results_lock = threading.Lock()
    threads = []

    if args.no_leftover_mode:
        streams = streams[:num_streams]
        sm_counts = sm_counts[:num_streams]
        batch_sizes = args.batch_sizes[:num_streams]
        prompt_lengths = args.prompt_lengths[:num_streams]
        output_lengths = args.output_lengths[:num_streams]
    else:
        num_streams = num_streams + 1
        batch_sizes = args.batch_sizes
        prompt_lengths = args.prompt_lengths
        output_lengths = args.output_lengths

    # Create barrier for synchronizing all worker threads
    barrier = threading.Barrier(num_streams)

    # Launch worker threads
    print(f"\nLaunching {num_streams} worker threads...")
    print("Stream configurations:")
    for stream_idx in range(num_streams):
        print(f"  Stream {stream_idx}: batch_size={batch_sizes[stream_idx]}, "
              f"prompt_length={prompt_lengths[stream_idx]}, "
              f"output_length={output_lengths[stream_idx]}")

    for stream_idx, (stream, sm_count, batch_size, prompt_length, output_length) in enumerate(
        zip(streams, sm_counts, batch_sizes, prompt_lengths, output_lengths)
    ):
        t = threading.Thread(
            target=worker_run_on_stream,
            args=(model, tokenizer, stream_idx, stream, sm_count,
                  batch_size, prompt_length, output_length,
                  args.num_repeat, args.log_dir, args.log_path,
                  results, results_lock, barrier),
            daemon=False
        )
        t.start()
        threads.append(t)

    # Wait for all workers to finish
    print("Waiting for workers to finish...")
    for t in threads:
        t.join()

    # Results are flushed incrementally by workers. Report summary.
    if results:
        print(f"\n{'='*60}")
        print(
            f"Collected {len(results)} measurements (already flushed to {output_path})")
        print(f"{'='*60}\n")
    else:
        print("No results were collected.")

    # Clean up
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Transformers benchmarking with independent instances on green context streams. '
                    'Each stream can have different batch_size, prompt_length, and output_length.'
    )
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--prompt-lengths', type=int, nargs='+', required=True,
                        help='Prompt lengths for each stream (array, length must match number of streams)')
    parser.add_argument('--output-lengths', type=int, nargs='+', required=True,
                        help='Output lengths for each stream (array, length must match number of streams)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', required=True,
                        help='Batch sizes for each stream (array, length must match number of streams)')
    parser.add_argument('--num-streams', type=int, default=1,
                        help='Number of parallel streams to create')
    parser.add_argument('--min-sm-per-stream', type=int, default=32,
                        help='Minimum SMs per stream')
    parser.add_argument('--num-repeat', type=int, default=2,
                        help='Number of repeats for each stream')
    parser.add_argument('--log-dir', type=str, default='parallel_transformers_green_ctx_results',
                        help='Output directory for logs')
    parser.add_argument('--log-path', type=str,
                        default='transformers_parallel_green_ctx_benchmark.csv',
                        help='Output CSV filename')
    parser.add_argument('--attention-impl', type=str, default='default',
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help='Preferred attention implementation to enable on the model (best-effort)')
    parser.add_argument('--no-leftover-mode',
                        default=False, action='store_true',
                        help='If set, do not use the leftover stream')
    args = parser.parse_args()

    print("="*60)
    print("Transformers Parallel Green Context Benchmark")
    print("Multiple Independent Model Instances on Separate Streams")
    print("Per-Stream Configuration Support")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Num Streams: {args.num_streams}")
    print(f"Min SMs per Stream: {args.min_sm_per_stream}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Output lengths: {args.output_lengths}")
    print(f"Num repeats: {args.num_repeat}")
    print("="*60)

    benchmark_transformers_parallel_green_ctx(args)


if __name__ == "__main__":
    main()
