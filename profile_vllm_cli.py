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

    # only 1 token per prompt to show decode performance
    prompt_token_length = args.prompt_length
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enforce_eager=False,  # enable CUDA graph
        max_num_seqs=1024,
        max_num_batched_tokens=args.batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    llm_engine = llm.llm_engine
    batch_sizes = args.batch_sizes

    data_path = os.path.join(args.log_dir, args.log_path)
    
    # Initialize CSV file with headers
    df_header = pd.DataFrame(columns=[
        "context", "prompt_length", "bs", "repeat_idx", "tpot", "decode_time", 
        "total_length", "batched_tokens", "ttft"
    ])
    df_header.to_csv(data_path, index=False)
    
    draft_tokens = {}
    all_outputs = []

    # Warm up
    print("Warming up...")
    output_len = args.output_len if args.output_len > 0 else (args.batched_tokens // args.batch_sizes[0] - prompt_token_length)
    
    sampling_params = SamplingParams(
        temperature=1,
        ignore_eos=True,
        max_tokens=output_len,
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    warmup_iters = args.warmup_iters
    warmup_bs = args.warmup_bs

    for _ in range(warmup_iters):
        for i in range(warmup_bs):
            prompt_token_ids = [randint(0, 8192) for _ in range(prompt_token_length)]
            token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
            llm_engine.add_request(
                request_id=f"warmup_{i}",
                prompt=token_prompt,
                params=sampling_params,
            )
        while llm_engine.has_unfinished_requests():
            llm_engine.step()

    
    for bs in batch_sizes:
        if args.output_len > 0:
            output_len = args.output_len
        else:
            output_len = args.batched_tokens // bs - prompt_token_length

        if output_len <= 1:
            print(f"Skipping batch size {bs} due to insufficient output length, output_len={output_len}")
            continue

        sampling_params = SamplingParams(
            temperature=1,
            ignore_eos=True,
            max_tokens=output_len,
            output_kind=RequestOutputKind.CUMULATIVE,
        )

        for repeat_idx in range(args.num_repeat):
            print(f"\n=== Running batch size {bs}, repeat {repeat_idx + 1}/{args.num_repeat} output_len {output_len} ===")
            
            # Clear previous requests if any
            while llm_engine.has_unfinished_requests():
                llm_engine.step()
            
            # Reset for each repeat
            rid = bs * repeat_idx
            batch_request_ids = []
            for i in range(bs):
                request_id = f"{bs}_{repeat_idx}_{rid}"
                prompt_token_ids = [randint(0, 8192) for _ in range(prompt_token_length)]
                token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
                llm_engine.add_request(
                    request_id=request_id,
                    prompt=token_prompt,
                    params=sampling_params,
                )
                draft_tokens[request_id] = [randint(0, 8192) for _ in range(args.n_verify)]
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
                    print(f"TTFT for batch size {bs}, repeat {repeat_idx + 1}: {ttft:.2f} ms")
                    break
                else:
                    print(f"Prefill step completed but no tokens generated yet for batch size {bs}, repeat {repeat_idx + 1}")

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
                print(f"{total_steps=}, {bs=}, repeat={repeat_idx+1}, {tpot_dur_window * 1000:.2f}, {llm_engine.get_num_unfinished_requests()=}")

                total_length = prompt_token_length + total_steps
                # Write result immediately to CSV with repeat information
                result_row = pd.DataFrame([{
                    "context": total_steps,
                    "prompt_length": prompt_token_length,
                    "bs": bs,
                    "repeat_idx": repeat_idx,
                    "tpot": tpot_dur_window * 1000,
                    "decode_time": (current_time - start),
                    "total_length": total_length,
                    "batched_tokens": total_length * bs,
                    "ttft": ttft
                }])
                result_row.to_csv(data_path, mode='a', header=False, index=False)
            
            # Add batch outputs to all outputs
            all_outputs.extend(batch_outputs)
            print(f"Batch {bs}, repeat {repeat_idx + 1} completed with {len(batch_outputs)} finished requests")


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
    parser.add_argument('--prompt-length', type=int, default=512,
                        help='Prompt token length')
    parser.add_argument('--output-len', type=int, default=128,
                        help='Output length (0 means auto-calculate based on batched_tokens and batch_size)')
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
    print(f"Prompt Length: {args.prompt_length}")
    print(f"Output Length: {args.output_len if args.output_len > 0 else 'auto'}")
    print(f"Num Repeats: {args.num_repeat}")
    print(f"GPU Memory Util: {args.gpu_memory_utilization}")
    print(f"Window Size: {args.window_size}")
    print("="*60)
    
    decode_bench(args)
    
    print(f"\nResults saved to: {os.path.join(args.log_dir, args.log_path)}")


if __name__ == "__main__":
    main()
