#!/usr/bin/env python3
"""
Command-line interface for vLLM decode benchmarking.

This script exposes all parameters from decode_bench as CLI arguments,
allowing flexible configuration without code changes.
"""

import argparse
import os
import time
from random import randint
import multiprocessing as mp
from multiprocessing import Barrier, Lock

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from collections import defaultdict


def decode_bench(args, barrier=None, file_lock=None, process_id=0):
    """Benchmark vLLM decode performance.

    Args:
        args: Arguments object
        barrier: multiprocessing.Barrier for synchronization
        file_lock: multiprocessing.Lock for file writing
        process_id: ID of the current process
    """
    np.random.seed(42 + process_id)  # Different seed for each process

    os.makedirs(args.log_dir, exist_ok=True)

    # Support multiple prompt lengths
    prompt_lengths = args.prompt_length if isinstance(
        args.prompt_length, list) else [args.prompt_length]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enforce_eager=False,  # enable CUDA graph
        max_num_seqs=1024,
        max_num_batched_tokens=args.batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
    )
    llm_engine = llm.llm_engine
    batch_sizes = args.batch_sizes

    # Use process-specific log file if multiple processes
    if barrier is not None and barrier.parties > 1:
        base_name = os.path.splitext(args.log_path)[0]
        ext = os.path.splitext(args.log_path)[1]
        data_path = os.path.join(
            args.log_dir, f"{base_name}_proc{process_id}{ext}")
    else:
        data_path = os.path.join(args.log_dir, args.log_path)

    # Initialize CSV file with headers
    # Check if file exists, if not, write headers
    if not os.path.exists(data_path):
        df_header = pd.DataFrame(columns=[
            "context", "prompt_length", "bs", "repeat_idx", "actual_input_len", "tpot", "decode_time",
            "total_length", "batched_tokens", "ttft", "process_id"
        ])
        if file_lock:
            with file_lock:
                # Double check after acquiring lock
                if not os.path.exists(data_path):
                    df_header.to_csv(data_path, index=False)
        else:
            df_header.to_csv(data_path, index=False)

    draft_tokens = {}
    all_outputs = []

    # Warm up (use first prompt length for warmup)
    print(f"[Process {process_id}] Warming up...")
    warmup_prompt_length = prompt_lengths[0]
    output_len = args.output_len

    sampling_params = SamplingParams(
        temperature=1,
        ignore_eos=True,
        max_tokens=args.output_len,
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    warmup_iters = args.warmup_iters
    warmup_bs = args.warmup_bs

    for _ in range(warmup_iters):
        for i in range(warmup_bs):
            prompt_token_ids = [randint(0, 8192)
                                for _ in range(warmup_prompt_length)]
            token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
            llm_engine.add_request(
                request_id=f"warmup_{process_id}_{i}",
                prompt=token_prompt,
                params=sampling_params,
            )
        while llm_engine.has_unfinished_requests():  # Synchronize before step
            llm_engine.step()

    # Loop over all prompt lengths
    for prompt_token_length in prompt_lengths:
        print(f"\n[Process {process_id}] {'='*60}")
        print(
            f"[Process {process_id}] Testing prompt length: {prompt_token_length}")
        print(f"[Process {process_id}] {'='*60}")

        for bs in batch_sizes:
            output_len = args.output_len

            # actual_input_len = prompt_token_length - output_len // 2
            actual_input_len = prompt_token_length

            if (actual_input_len + output_len) * bs > args.batched_tokens:
                print(
                    f"[Process {process_id}] Skipping batch size {bs} due to insufficient batched tokens, (actual_input_len + output_len) * bs={actual_input_len + output_len} * {bs} > {args.batched_tokens}")
                continue

            sampling_params = SamplingParams(
                temperature=1,
                ignore_eos=True,
                max_tokens=output_len,
                output_kind=RequestOutputKind.CUMULATIVE,
            )

            for repeat_idx in range(args.num_repeat):
                print(
                    f"\n[Process {process_id}] === Running prompt_length {prompt_token_length}, batch size {bs}, repeat {repeat_idx + 1}/{args.num_repeat} output_len {output_len} ===")

                # Clear previous requests if any
                while llm_engine.has_unfinished_requests():
                    llm_engine.step()

                if barrier:
                    barrier.wait()  # Synchronize before starting new repeat

                # Reset for each repeat
                rid = bs * repeat_idx
                batch_request_ids = []
                for i in range(bs):
                    request_id = f"{process_id}_{prompt_token_length}_{bs}_{repeat_idx}_{rid}"
                    prompt_token_ids = [randint(0, 8192)
                                        for _ in range(actual_input_len)]
                    token_prompt = TokensPrompt(
                        prompt_token_ids=prompt_token_ids)
                    llm_engine.add_request(
                        request_id=request_id,
                        prompt=token_prompt,
                        params=sampling_params,
                    )
                    draft_tokens[request_id] = [
                        randint(0, 8192) for _ in range(args.n_verify)]
                    batch_request_ids.append(request_id)
                    rid += 1

                # if barrier:
                #     barrier.wait()  # Synchronize after adding all requests

                window = args.window_size
                total_steps = 0
                batch_outputs = []

                ttft = None

                num_steps = 0
                start = time.perf_counter()
                while llm_engine.has_unfinished_requests():
                    # if barrier:
                    #     barrier.wait()  # Synchronize before step
                    # print(f"[Process {process_id}] Prefill step {num_steps + 1}...")
                    step_outputs = llm_engine.step()
                    end = time.perf_counter()
                    ttft = None
                    # req_prefilled_list = []
                    num_steps += 1
                    if len(step_outputs) == bs:
                        ttft = (end - start) * 1000  # in ms

                    # print(f"[Process {process_id}] Prefill step {num_steps} completed, with {len(step_outputs)} step outputs")
                    # for i, output in enumerate(step_outputs):
                    #     print(f"[Process {process_id}] Checking step output {i} for TTFT, {output.outputs}")

                    #     if hasattr(output, 'outputs') and output.outputs:
                    #         all_req_prefilled = True
                            
                    #         for j, req_output in enumerate(output.outputs):
                    #             print(f"[Process {process_id}] Step output {i} request output {j} has {len(req_output.token_ids)} tokens")
                    #             if len(req_output.token_ids) == 0:
                    #                 all_req_prefilled = False
                    #                 break
                    #         print(f"[Process {process_id}] Step output {i} has {len(output.outputs)} outputs, all_req_prefilled={all_req_prefilled}")
                    #         if all_req_prefilled:
                    #             ttft = (end - start) * 1000  # in ms
                    #         req_prefilled_list.append(output.request_id)

                    if ttft is not None:
                        print(
                            f"[Process {process_id}] TTFT for prompt_length {prompt_token_length}, batch size {bs}, repeat {repeat_idx + 1}: {ttft:.2f} ms")
                        break
                    # else:
                    #     print(
                    #         f"[Process {process_id}] Prefill step completed but no tokens generated yet for prompt_length {prompt_token_length}, batch size {bs}, repeat {repeat_idx + 1}")

                while llm_engine.has_unfinished_requests():
                    start = time.perf_counter()
                    num_steps = 0
                    for _ in range(window):
                        if llm_engine.has_unfinished_requests():
                            # if barrier:
                            #     barrier.wait()  # Synchronize before each step
                            step_outputs = llm_engine.step()
                            num_steps += 1
                        else:
                            break
                    current_time = time.perf_counter()
                    total_steps += num_steps
                    tpot_dur_window = (current_time - start) / \
                        num_steps if num_steps > 0 else 0
                    print(
                        f"[Process {process_id}] {total_steps=}, prompt_length={prompt_token_length}, {bs=}, repeat={repeat_idx+1}, {tpot_dur_window * 1000:.2f}, {llm_engine.get_num_unfinished_requests()=}")

                    total_length = actual_input_len + total_steps
                    # Write result immediately to CSV with repeat information
                    result_row = pd.DataFrame([{
                        "context": total_steps,
                        "prompt_length": prompt_token_length,
                        "bs": bs,
                        "repeat_idx": repeat_idx,
                        "actual_input_len": actual_input_len,
                        "tpot": tpot_dur_window * 1000,
                        "decode_time": (current_time - start),
                        "total_length": total_length,
                        "batched_tokens": total_length * bs,
                        "ttft": ttft,
                        "process_id": process_id
                    }])
                    if file_lock:
                        with file_lock:
                            result_row.to_csv(data_path, mode='a',
                                              header=False, index=False)
                    else:
                        result_row.to_csv(data_path, mode='a',
                                          header=False, index=False)

                # Add batch outputs to all outputs
                all_outputs.extend(batch_outputs)
                print(
                    f"[Process {process_id}] Prompt length {prompt_token_length}, batch {bs}, repeat {repeat_idx + 1} completed with {len(batch_outputs)} finished requests")


def decode_bench_combined_batch(args, barrier=None, file_lock=None, process_id=0):
    """Benchmark vLLM decode performance with combined batches.

    Combines multiple sub-batches with different batch_sizes, prompt_lengths, 
    and output_lengths into one large batch, similar to test_transformers_combined_batch.py.

    Args:
        args: Arguments object
        barrier: multiprocessing.Barrier for synchronization
        file_lock: multiprocessing.Lock for file writing
        process_id: ID of the current process
    """
    np.random.seed(42 + process_id)  # Different seed for each process

    os.makedirs(args.log_dir, exist_ok=True)

    # Validate array lengths
    num_configs = len(args.combined_batch_sizes)
    if len(args.combined_prompt_lengths) != num_configs:
        raise ValueError(
            f"combined_prompt_lengths length ({len(args.combined_prompt_lengths)}) must match "
            f"combined_batch_sizes length ({num_configs})"
        )

    # Use minimum output_length for all sub-batches
    output_length = min(args.combined_output_lengths)
    total_batch_size = sum(args.combined_batch_sizes)
    max_prompt_length = max(args.combined_prompt_lengths)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enforce_eager=False,  # enable CUDA graph
        max_num_seqs=1024,
        max_num_batched_tokens=args.batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
    )
    llm_engine = llm.llm_engine

    # Use process-specific log file if multiple processes
    if barrier is not None and barrier.parties > 1:
        base_name = os.path.splitext(args.log_path)[0]
        ext = os.path.splitext(args.log_path)[1]
        data_path = os.path.join(
            args.log_dir, f"{base_name}_proc{process_id}{ext}")
    else:
        data_path = os.path.join(args.log_dir, args.log_path)

    # Initialize CSV file with headers
    if not os.path.exists(data_path):
        df_header = pd.DataFrame(columns=[
            "repeat_idx", "sub_batch_idx", "batch_size", "prompt_length",
            "generation_steps", "ttft_ms", "tpot_ms", "total_time_ms",
            "finish_time", "process_id"
        ])
        if file_lock:
            with file_lock:
                if not os.path.exists(data_path):
                    df_header.to_csv(data_path, index=False)
        else:
            df_header.to_csv(data_path, index=False)

    print(f"\n[Process {process_id}] {'='*60}")
    print(f"[Process {process_id}] Combined Batch Configuration")
    print(f"[Process {process_id}] {'='*60}")
    print(f"[Process {process_id}] Total batch size: {total_batch_size}")
    print(f"[Process {process_id}] Max prompt length: {max_prompt_length}")
    print(f"[Process {process_id}] Output length (min): {output_length}")
    print(f"[Process {process_id}] Sub-batch configurations:")
    for i, (bs, pl, ol) in enumerate(zip(args.combined_batch_sizes,
                                         args.combined_prompt_lengths,
                                         args.combined_output_lengths)):
        print(f"[Process {process_id}]   Sub-batch {i}: batch_size={bs}, "
              f"prompt_length={pl}, output_length={ol}")
    print(f"[Process {process_id}] {'='*60}\n")

    # Warm up
    print(f"[Process {process_id}] Warming up...")
    warmup_sampling_params = SamplingParams(
        temperature=1,
        ignore_eos=True,
        max_tokens=output_length,
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    warmup_iters = args.warmup_iters
    warmup_bs = args.warmup_bs

    for _ in range(warmup_iters):
        for i in range(warmup_bs):
            prompt_token_ids = [randint(0, 8192)
                                for _ in range(max_prompt_length)]
            token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
            llm_engine.add_request(
                request_id=f"warmup_{process_id}_{i}",
                prompt=token_prompt,
                params=warmup_sampling_params,
            )
        while llm_engine.has_unfinished_requests():
            llm_engine.step()
    print(f"[Process {process_id}] Warmup complete")

    # Run benchmarks
    for repeat_idx in range(args.num_repeat):
        print(f"\n[Process {process_id}] {'='*60}")
        print(
            f"[Process {process_id}] Repeat {repeat_idx + 1}/{args.num_repeat}")
        print(f"[Process {process_id}] {'='*60}")

        # Clear previous requests if any
        while llm_engine.has_unfinished_requests():
            llm_engine.step()

        if barrier:
            barrier.wait()  # Synchronize before starting new repeat

        # Create combined batch with sub-batch configurations
        # (sub_idx, request_ids, batch_size, prompt_length)
        sub_batch_configs = []
        all_request_ids = []
        request_to_sub_batch = {}  # Map request_id -> sub_batch_idx

        current_rid = 0
        for sub_idx, (batch_size, prompt_length) in enumerate(
            zip(args.combined_batch_sizes, args.combined_prompt_lengths)
        ):
            sub_request_ids = []
            actual_input_len = prompt_length - output_length // 2

            sampling_params = SamplingParams(
                temperature=1,
                ignore_eos=True,
                max_tokens=output_length,
                output_kind=RequestOutputKind.CUMULATIVE,
            )

            for i in range(batch_size):
                request_id = (f"{process_id}_combined_{sub_idx}_"
                              f"{prompt_length}_{batch_size}_{repeat_idx}_{current_rid}")

                prompt_token_ids = [randint(0, 8192)
                                    for _ in range(actual_input_len)]
                token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

                llm_engine.add_request(
                    request_id=request_id,
                    prompt=token_prompt,
                    params=sampling_params,
                )

                sub_request_ids.append(request_id)
                all_request_ids.append(request_id)
                request_to_sub_batch[request_id] = sub_idx
                current_rid += 1

            sub_batch_configs.append(
                (sub_idx, sub_request_ids, batch_size, prompt_length))

        print(f"[Process {process_id}] Running inference on combined batch "
              f"(size={total_batch_size})...")

        # Track timing for each request
        request_ttft = {}  # request_id -> ttft time
        # request_id -> list of token generation times
        request_token_times = defaultdict(list)
        # request_id -> number of tokens generated
        request_token_counts = defaultdict(int)
        request_first_token_step = {}  # request_id -> step when first token generated

        gen_start = time.perf_counter()
        window = args.window_size
        total_steps = 0

        # Prefill phase - measure TTFT
        prefill_step = 0
        while llm_engine.has_unfinished_requests():
            step_start = time.perf_counter()
            step_outputs = llm_engine.step()
            step_end = time.perf_counter()
            prefill_step += 1

            # Check for first tokens generated
            for output in step_outputs:
                if hasattr(output, 'request_id') and hasattr(output, 'outputs') and output.outputs:
                    if len(output.outputs[0].token_ids) > 0:
                        request_id = output.request_id
                        if request_id not in request_ttft:
                            # First token generated
                            request_ttft[request_id] = (
                                step_end - step_start) * 1000  # ms
                            request_first_token_step[request_id] = prefill_step
                            # Record initial token count
                            request_token_counts[request_id] = len(
                                output.outputs[0].token_ids)

            # Check if all requests have started generating
            if all(req_id in request_ttft for req_id in all_request_ids):
                break

        # Decode phase - measure TPOT
        # Track previous token counts to detect new tokens
        request_prev_token_counts = {req_id: request_token_counts.get(req_id, 0)
                                     for req_id in all_request_ids}

        while llm_engine.has_unfinished_requests():
            window_start = time.perf_counter()
            num_steps = 0

            # Execute steps in window and track timing per step
            for step_in_window in range(window):
                if llm_engine.has_unfinished_requests():
                    step_start = time.perf_counter()
                    step_outputs = llm_engine.step()
                    step_end = time.perf_counter()
                    step_duration = (step_end - step_start) * 1000  # ms
                    num_steps += 1

                    # Process outputs to track token generation per request
                    for output in step_outputs:
                        if hasattr(output, 'request_id') and hasattr(output, 'outputs') and output.outputs:
                            request_id = output.request_id
                            if request_id in request_to_sub_batch:
                                current_token_count = len(
                                    output.outputs[0].token_ids)
                                prev_token_count = request_prev_token_counts.get(
                                    request_id, 0)

                                # If new tokens were generated, record timing
                                if current_token_count > prev_token_count:
                                    new_tokens = current_token_count - prev_token_count
                                    # Time per token for this step
                                    time_per_token = step_duration / new_tokens  # ms per token
                                    # Record time for each new token
                                    for _ in range(new_tokens):
                                        request_token_times[request_id].append(
                                            time_per_token)

                                    request_prev_token_counts[request_id] = current_token_count
                else:
                    break

            window_end = time.perf_counter()

            if num_steps > 0:
                total_steps += num_steps
                window_tpot = ((window_end - window_start) /
                               num_steps) * 1000  # avg ms per step
                print(f"[Process {process_id}] Total steps: {total_steps}, "
                      f"TPOT: {window_tpot:.2f}ms, "
                      f"unfinished requests: {llm_engine.get_num_unfinished_requests()}")

        gen_end = time.perf_counter()
        total_time_ms = (gen_end - gen_start) * 1000

        # Aggregate metrics per sub-batch
        for sub_idx, sub_request_ids, orig_batch_size, prompt_length in sub_batch_configs:
            # Collect metrics for this sub-batch
            sub_ttfts = []
            sub_tpots = []
            sub_generated_steps = []

            for request_id in sub_request_ids:
                if request_id in request_ttft:
                    sub_ttfts.append(request_ttft[request_id])

                if request_id in request_token_times:
                    token_times = request_token_times[request_id]
                    if token_times:
                        sub_tpots.extend(token_times)
                        sub_generated_steps.append(len(token_times))

            # Calculate averages
            ttft_ms = float(np.mean(sub_ttfts)) if sub_ttfts else 0.0
            tpot_ms = float(np.mean(sub_tpots)) if sub_tpots else 0.0
            generation_steps = int(
                np.mean(sub_generated_steps)) if sub_generated_steps else 0

            # Write result
            result_row = {
                "repeat_idx": repeat_idx,
                "sub_batch_idx": sub_idx,
                "batch_size": orig_batch_size,
                "prompt_length": prompt_length,
                "generation_steps": generation_steps,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "total_time_ms": total_time_ms,
                "finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "process_id": process_id
            }

            result_df = pd.DataFrame([result_row])
            if file_lock:
                with file_lock:
                    result_df.to_csv(data_path, mode='a',
                                     header=False, index=False)
            else:
                result_df.to_csv(data_path, mode='a',
                                 header=False, index=False)

            print(f"[Process {process_id}]   Sub-batch {sub_idx}: "
                  f"bs={orig_batch_size} pl={prompt_length} "
                  f"tpot={tpot_ms:.2f}ms ttft={ttft_ms:.2f}ms "
                  f"total={total_time_ms:.2f}ms steps={generation_steps}")

    print(f"\n[Process {process_id}] {'='*60}")
    print(f"[Process {process_id}] Combined batch benchmark complete. "
          f"Results saved to {data_path}")
    print(f"[Process {process_id}] {'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='vLLM Decode Benchmark with Full CLI Arguments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument('--model', type=str, default='/nfs/xjzhang/Qwen/Qwen3-4B',
                        help='Model path')
    parser.add_argument('--tp-size', type=int, default=1,
                        help='Tensor parallel size')
    parser.add_argument('--max-model-len', type=int, default=40960,
                        help='Max model length')

    # Batch and token configuration
    parser.add_argument('--batched-tokens', type=int, default=65537,
                        help='Max batched tokens per step')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32],
                        help='List of batch sizes to test')
    parser.add_argument('--prompt-length', type=int, nargs='+', default=[512, 1024, 2048, 4096],
                        help='Prompt token length(s) to test (can specify multiple values)')
    parser.add_argument('--output-len', type=int, default=128,
                        help='Output length')
    parser.add_argument('--window-size', type=int, default=128,
                        help='Window size for measuring TPOT')

    # Repeat and warmup configuration
    parser.add_argument('--num-repeat', type=int, default=3,
                        help='Number of repeats for each batch size')
    parser.add_argument('--warmup-iters', type=int, default=2,
                        help='Number of warmup iterations')
    parser.add_argument('--warmup-bs', type=int, default=2,
                        help='Batch size for warmup')

    # Verification tokens (currently unused but kept for compatibility)
    parser.add_argument('--n-verify', type=int, default=1,
                        help='Number of verification tokens')

    # GPU configuration
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='GPU memory utilization ratio')
    parser.add_argument('--distributed-executor-backend', type=str, default=None,
                        help='Distributed executor backend (e.g., "uni", "ray", "mp")')
    parser.add_argument('--kv-cache-memory-bytes', type=int, default=None,
                        help='KV cache memory bytes (optional, overrides default allocation)')

    # Output configuration
    parser.add_argument('--log-dir', type=str, default='profile_results',
                        help='Output directory for results')
    parser.add_argument('--log-path', type=str, default='decode_benchmark.csv',
                        help='Output CSV filename')

    # Multiprocessing configuration
    parser.add_argument('--num-processes', type=int, default=1,
                        help='Number of processes to run in parallel')

    # Combined batch mode configuration
    parser.add_argument('--combined-batch', action='store_true',
                        help='Enable combined batch mode (combines multiple sub-batches with different configs)')
    parser.add_argument('--combined-batch-sizes', type=int, nargs='+', default=None,
                        help='Batch sizes for each sub-batch in combined mode (array)')
    parser.add_argument('--combined-prompt-lengths', type=int, nargs='+', default=None,
                        help='Prompt lengths for each sub-batch in combined mode (array)')
    parser.add_argument('--combined-output-lengths', type=int, nargs='+', default=None,
                        help='Output lengths for each sub-batch in combined mode (array, min will be used)')

    args = parser.parse_args()

    # Validate combined batch mode arguments
    if args.combined_batch:
        if args.combined_batch_sizes is None or args.combined_prompt_lengths is None or args.combined_output_lengths is None:
            raise ValueError(
                "Combined batch mode requires --combined-batch-sizes, "
                "--combined-prompt-lengths, and --combined-output-lengths"
            )
        if len(args.combined_batch_sizes) != len(args.combined_prompt_lengths) or \
           len(args.combined_batch_sizes) != len(args.combined_output_lengths):
            raise ValueError(
                "combined_batch_sizes, combined_prompt_lengths, and "
                "combined_output_lengths must have the same length"
            )
        benchmark_func = decode_bench_combined_batch
        mode_name = "Combined Batch"
    else:
        benchmark_func = decode_bench
        mode_name = "Standard"

    print("="*60)
    print(f"vLLM Decode Benchmark - {mode_name} Mode")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tp_size}")
    print(f"Max Model Len: {args.max_model_len}")
    print(f"Batched Tokens: {args.batched_tokens}")

    if args.combined_batch:
        print(f"Combined Batch Sizes: {args.combined_batch_sizes}")
        print(f"Combined Prompt Lengths: {args.combined_prompt_lengths}")
        print(f"Combined Output Lengths: {args.combined_output_lengths}")
        print(f"Total Batch Size: {sum(args.combined_batch_sizes)}")
        print(f"Min Output Length: {min(args.combined_output_lengths)}")
    else:
        print(f"Batch Sizes: {args.batch_sizes}")
        prompt_lengths = args.prompt_length if isinstance(
            args.prompt_length, list) else [args.prompt_length]
        print(f"Prompt Length(s): {prompt_lengths}")
        print(
            f"Output Length: {args.output_len if args.output_len > 0 else 'auto'}")

    print(f"Num Repeats: {args.num_repeat}")
    print(f"GPU Memory Util: {args.gpu_memory_utilization}")
    if args.kv_cache_memory_bytes is not None:
        print(f"KV Cache Memory Bytes: {args.kv_cache_memory_bytes}")
    print(f"Window Size: {args.window_size}")
    print(f"Num Processes: {args.num_processes}")
    print("="*60)

    if args.num_processes > 1:
        # Create barrier and lock for multiprocessing
        barrier = Barrier(args.num_processes)
        file_lock = Lock()

        # Create and start processes
        processes = []
        for i in range(args.num_processes):
            p = mp.Process(target=benchmark_func, args=(
                args, barrier, file_lock, i))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print(f"\nAll {args.num_processes} processes completed.")
        print(f"Results saved to: {os.path.join(args.log_dir, args.log_path)}")
        if args.num_processes > 1:
            print(f"Note: Each process wrote to a separate file (proc0, proc1, ...)")
    else:
        # Single process mode
        benchmark_func(args, barrier=None, file_lock=None, process_id=0)
        print(
            f"\nResults saved to: {os.path.join(args.log_dir, args.log_path)}")


if __name__ == "__main__":
    main()
