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

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind


def decode_bench(args):
    """Benchmark vLLM decode performance."""
    np.random.seed(42)

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
    )
    llm_engine = llm.llm_engine
    batch_sizes = args.batch_sizes

    data_path = os.path.join(args.log_dir, args.log_path)

    # Initialize CSV file with headers
    df_header = pd.DataFrame(columns=[
        "context", "prompt_length", "bs", "repeat_idx", "actual_input_len", "tpot", "decode_time",
        "total_length", "batched_tokens", "ttft"
    ])
    df_header.to_csv(data_path, index=False)

    draft_tokens = {}
    all_outputs = []

    # Warm up (use first prompt length for warmup)
    print("Warming up...")
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
                request_id=f"warmup_{i}",
                prompt=token_prompt,
                params=sampling_params,
            )
        while llm_engine.has_unfinished_requests():
            llm_engine.step()

    # Loop over all prompt lengths
    for prompt_token_length in prompt_lengths:
        print(f"\n{'='*60}")
        print(f"Testing prompt length: {prompt_token_length}")
        print(f"{'='*60}")

        for bs in batch_sizes:
            output_len = args.output_len

            if (prompt_token_length + output_len // 2) * bs > args.batched_tokens:
                print(
                    f"Skipping batch size {bs} due to insufficient batched tokens, (prompt_token_length + output_len) * bs={prompt_token_length + output_len} * {bs} > {args.batched_tokens}")
                continue

            actual_input_len = prompt_token_length - output_len // 2

            sampling_params = SamplingParams(
                temperature=1,
                ignore_eos=True,
                max_tokens=output_len,
                output_kind=RequestOutputKind.CUMULATIVE,
            )

            for repeat_idx in range(args.num_repeat):
                print(
                    f"\n=== Running prompt_length {prompt_token_length}, batch size {bs}, repeat {repeat_idx + 1}/{args.num_repeat} output_len {output_len} ===")

                # Clear previous requests if any
                while llm_engine.has_unfinished_requests():
                    llm_engine.step()

                # Reset for each repeat
                rid = bs * repeat_idx
                batch_request_ids = []
                for i in range(bs):
                    request_id = f"{prompt_token_length}_{bs}_{repeat_idx}_{rid}"
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

                window = args.window_size
                total_steps = 0
                batch_outputs = []

                ttft = None

                # prefill the requests first and get the TTFT
                while llm_engine.has_unfinished_requests():
                    start = time.perf_counter()
                    step_outputs = llm_engine.step()
                    end = time.perf_counter()
                    ttft = None
                    for output in step_outputs:
                        if hasattr(output, 'outputs') and output.outputs:
                            if len(output.outputs[0].token_ids) > 0:
                                ttft = (end - start) * 1000  # in ms
                                break

                    if ttft is not None:
                        print(
                            f"TTFT for prompt_length {prompt_token_length}, batch size {bs}, repeat {repeat_idx + 1}: {ttft:.2f} ms")
                        break
                    else:
                        print(
                            f"Prefill step completed but no tokens generated yet for prompt_length {prompt_token_length}, batch size {bs}, repeat {repeat_idx + 1}")

                while llm_engine.has_unfinished_requests():
                    start = time.perf_counter()
                    num_steps = 0
                    for _ in range(window):
                        if llm_engine.has_unfinished_requests():
                            step_outputs = llm_engine.step()
                            num_steps += 1
                        else:
                            break
                    current_time = time.perf_counter()
                    total_steps += num_steps
                    tpot_dur_window = (current_time - start) / num_steps
                    print(
                        f"{total_steps=}, prompt_length={prompt_token_length}, {bs=}, repeat={repeat_idx+1}, {tpot_dur_window * 1000:.2f}, {llm_engine.get_num_unfinished_requests()=}")

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
                        "ttft": ttft
                    }])
                    result_row.to_csv(data_path, mode='a',
                                      header=False, index=False)

                # Add batch outputs to all outputs
                all_outputs.extend(batch_outputs)
                print(
                    f"Prompt length {prompt_token_length}, batch {bs}, repeat {repeat_idx + 1} completed with {len(batch_outputs)} finished requests")


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

    # Output configuration
    parser.add_argument('--log-dir', type=str, default='profile_results',
                        help='Output directory for results')
    parser.add_argument('--log-path', type=str, default='decode_benchmark.csv',
                        help='Output CSV filename')

    args = parser.parse_args()

    print("="*60)
    print("vLLM Decode Benchmark")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tp_size}")
    print(f"Max Model Len: {args.max_model_len}")
    print(f"Batched Tokens: {args.batched_tokens}")
    print(f"Batch Sizes: {args.batch_sizes}")
    prompt_lengths = args.prompt_length if isinstance(
        args.prompt_length, list) else [args.prompt_length]
    print(f"Prompt Length(s): {prompt_lengths}")
    print(
        f"Output Length: {args.output_len if args.output_len > 0 else 'auto'}")
    print(f"Num Repeats: {args.num_repeat}")
    print(f"GPU Memory Util: {args.gpu_memory_utilization}")
    print(f"Window Size: {args.window_size}")
    print("="*60)

    decode_bench(args)

    print(f"\nResults saved to: {os.path.join(args.log_dir, args.log_path)}")


if __name__ == "__main__":
    main()
