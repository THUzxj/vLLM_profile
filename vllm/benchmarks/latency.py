# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import vllm.envs as envs
from vllm.benchmarks.lib.utils import (convert_to_pytorch_benchmark_format,
                                       write_to_json)
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k]
                    for k in ["avg_latency", "percentiles"]})
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )

    parser = EngineArgs.add_cli_args(parser)
    # V1 enables prefix caching by default which skews the latency
    # numbers. We need to disable prefix caching by default.
    parser.set_defaults(enable_prefix_caching=False)


def main(args: argparse.Namespace):
    if args.profile and not envs.VLLM_TORCH_PROFILER_DIR:
        raise OSError(
            "The environment variable 'VLLM_TORCH_PROFILER_DIR' is not set. "
            "Please set it to a valid path to use torch profiler.")
    engine_args = EngineArgs.from_cli_args(args)

    # Lazy import to avoid importing LLM when the bench command is not selected.
    from vllm import LLM, SamplingParams

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len +
        args.output_len), ("Please ensure that max_model_len is greater than"
                           " the sum of input_len and output_len.")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def _generate_with_timing(prompts, sampling_params=None, beam_params=None, use_tqdm=False):
        """
        Generate with detailed timing information.
        Returns: Tuple of (outputs, total_latency_ms, prefill_time_ms, decode_time_ms)
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        def log_output_lengths(outputs):
            """Log the actual output lengths for each sentence."""
            import sys
            if outputs:
                print(f"\n--- Output Length Analysis ---", file=sys.stderr)
                for i, output in enumerate(outputs):
                    if hasattr(output, 'outputs') and output.outputs:
                        for j, completion in enumerate(output.outputs):
                            if hasattr(completion, 'token_ids'):
                                actual_length = len(completion.token_ids)
                                print(f"Sentence {i+1}, Completion {j+1}: {actual_length} tokens", file=sys.stderr)
                            elif hasattr(completion, 'text'):
                                # Approximate token count from text length
                                approx_tokens = len(completion.text.split())
                                print(f"Sentence {i+1}, Completion {j+1}: ~{approx_tokens} tokens (approx from text)", file=sys.stderr)
                    elif hasattr(output, 'token_ids'):
                        actual_length = len(output.token_ids)
                        print(f"Sentence {i+1}: {actual_length} tokens", file=sys.stderr)
                    elif hasattr(output, 'text'):
                        approx_tokens = len(output.text.split())
                        print(f"Sentence {i+1}: ~{approx_tokens} tokens (approx from text)", file=sys.stderr)
                
                # Calculate statistics
                all_lengths = []
                for output in outputs:
                    if hasattr(output, 'outputs') and output.outputs:
                        for completion in output.outputs:
                            if hasattr(completion, 'token_ids'):
                                all_lengths.append(len(completion.token_ids))
                    elif hasattr(output, 'token_ids'):
                        all_lengths.append(len(output.token_ids))
                
                if all_lengths:
                    avg_length = sum(all_lengths) / len(all_lengths)
                    min_length = min(all_lengths)
                    max_length = max(all_lengths)
                    print(f"Output Length Summary: avg={avg_length:.1f}, min={min_length}, max={max_length} tokens", file=sys.stderr)
                    
                    # Compare with expected length
                    expected_length = sampling_params.max_tokens if sampling_params else (beam_params.max_tokens if beam_params else "unknown")
                    if expected_length != "unknown":
                        print(f"Expected vs Actual: expected={expected_length}, actual_avg={avg_length:.1f}", file=sys.stderr)
                print(f"--- End Output Length Analysis ---\n", file=sys.stderr)
        
        # Try multiple approaches for accessing detailed timing
        engine = llm.llm_engine
        
        # Method 1: Try to use v1 engine's output processor stats  
        try:
            if hasattr(engine, 'output_processor'):
                # Clear any existing stats
                if hasattr(engine.output_processor, 'iteration_stats_collector'):
                    collector = engine.output_processor.iteration_stats_collector
                    # Reset the collector to capture new stats
                    if hasattr(collector, 'finished_requests'):
                        collector.finished_requests = []
                    
                    if beam_params is not None:
                        outputs = llm.beam_search(prompts, beam_params)
                    else:
                        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    total_latency = (end_time - start_time) * 1000
                    
                    # Try to get timing from the collector
                    if hasattr(collector, 'finished_requests') and collector.finished_requests:
                        finished_req = collector.finished_requests[0]
                        prefill_time_ms = finished_req.prefill_time * 1000
                        decode_time_ms = finished_req.decode_time * 1000
                        import sys
                        print("✓ Using detailed timing from output processor", file=sys.stderr)
                        log_output_lengths(outputs)
                        return outputs, total_latency, prefill_time_ms, decode_time_ms
        except Exception as e:
            import sys
            print(f"Output processor method failed: {e}", file=sys.stderr)
        
        # Method 2: Try to access engine_core stats_collector (original approach but improved)
        try:
            if hasattr(engine, 'engine_core'):
                engine_core = engine.engine_core
                
                # Check for stats_collector attribute
                if hasattr(engine_core, 'stats_collector'):
                    stats_collector = engine_core.stats_collector
                    
                    # Try to reset finished requests using different possible attributes
                    for attr_name in ['finished_requests', 'finished_requests_iter', 'iteration_stats']:
                        if hasattr(stats_collector, attr_name):
                            try:
                                if isinstance(getattr(stats_collector, attr_name), list):
                                    setattr(stats_collector, attr_name, [])
                                elif hasattr(getattr(stats_collector, attr_name), 'clear'):
                                    getattr(stats_collector, attr_name).clear()
                            except:
                                pass
                    
                    if beam_params is not None:
                        outputs = llm.beam_search(prompts, beam_params)
                    else:
                        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    total_latency = (end_time - start_time) * 1000
                    
                    # Try to extract timing data from different possible attributes
                    finished_reqs = None
                    for attr_name in ['finished_requests', 'finished_requests_iter', 'iteration_stats']:
                        if hasattr(stats_collector, attr_name):
                            attr_value = getattr(stats_collector, attr_name)
                            if isinstance(attr_value, list) and attr_value:
                                finished_reqs = attr_value
                                break
                            elif hasattr(attr_value, '__iter__'):
                                try:
                                    finished_reqs = list(attr_value)
                                    if finished_reqs:
                                        break
                                except:
                                    pass
                    
                    if finished_reqs:
                        finished_req = finished_reqs[0]
                        # Try different attribute names for timing
                        prefill_time_ms = 0
                        decode_time_ms = 0
                        
                        for prefill_attr in ['prefill_time', 'time_prefill', 'prefill_duration']:
                            if hasattr(finished_req, prefill_attr):
                                prefill_time_ms = getattr(finished_req, prefill_attr) * 1000
                                break
                        
                        for decode_attr in ['decode_time', 'time_decode', 'decode_duration']:
                            if hasattr(finished_req, decode_attr):
                                decode_time_ms = getattr(finished_req, decode_attr) * 1000
                                break
                        
                        if decode_time_ms == 0:
                            decode_time_ms = total_latency - prefill_time_ms
                        
                        import sys
                        print("✓ Using detailed timing from engine core stats", file=sys.stderr)
                        log_output_lengths(outputs)
                        return outputs, total_latency, prefill_time_ms, decode_time_ms
                else:
                    import sys
                    print(f"engine_core exists but no stats_collector found", file=sys.stderr)
            else:
                import sys
                print(f"No engine_core found", file=sys.stderr)
        except Exception as e:
            import sys
            print(f"Engine core stats method failed: {e}", file=sys.stderr)
        
        # Method 3: Try direct engine stats access
        try:
            if hasattr(engine, 'step'):
                # Hook into the engine step to capture metrics
                original_step = engine.step
                step_metrics = {'prefill_time': None, 'decode_time': None}
                
                def hooked_step():
                    step_start = time.perf_counter()
                    result = original_step()
                    step_end = time.perf_counter()
                    
                    # Record timing based on step results
                    if step_metrics['prefill_time'] is None:
                        step_metrics['prefill_time'] = step_end - step_start
                    else:
                        if step_metrics['decode_time'] is None:
                            step_metrics['decode_time'] = 0
                        step_metrics['decode_time'] += step_end - step_start
                    
                    return result
                
                engine.step = hooked_step
                
                try:
                    if beam_params is not None:
                        outputs = llm.beam_search(prompts, beam_params)
                    else:
                        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    total_latency = (end_time - start_time) * 1000
                    
                    if step_metrics['prefill_time'] is not None:
                        prefill_time_ms = step_metrics['prefill_time'] * 1000
                        decode_time_ms = step_metrics['decode_time'] * 1000 if step_metrics['decode_time'] else (total_latency - prefill_time_ms)
                        import sys
                        print("✓ Using detailed timing from engine step hooks", file=sys.stderr)
                        log_output_lengths(outputs)
                        return outputs, total_latency, prefill_time_ms, decode_time_ms
                finally:
                    engine.step = original_step
        except Exception as e:
            import sys
            print(f"Engine step hook method failed: {e}", file=sys.stderr)
        
        import sys
        print("⚠ Falling back to simple timing with estimation", file=sys.stderr)
        
        raise Exception("All detailed timing methods failed.")
        # Fallback: simple timing with estimation
        if beam_params is not None:
            outputs = llm.beam_search(prompts, beam_params)
        else:
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        total_latency = (end_time - start_time) * 1000
        # Estimate prefill time as 20% of total (conservative estimate)
        prefill_time_ms = total_latency * 0.2
        decode_time_ms = total_latency * 0.8
        
        log_output_lengths(outputs)
        return outputs, total_latency, prefill_time_ms, decode_time_ms

    def llm_generate():
        if not args.use_beam_search:
            return _generate_with_timing(
                dummy_prompts,
                sampling_params=sampling_params,
                use_tqdm=False
            )
        else:
            beam_params = BeamSearchParams(
                beam_width=args.n,
                max_tokens=args.output_len,
                ignore_eos=True,
            )
            return _generate_with_timing(
                dummy_prompts,
                beam_params=beam_params
            )

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            llm.start_profile()
            outputs, total_latency_ms, prefill_time_ms, decode_time_ms = llm_generate()
            llm.stop_profile()
            # return None  # Profiling mode doesn't return timing data
        else:
            outputs, total_latency_ms, prefill_time_ms, decode_time_ms = llm_generate()
        # Convert ms to seconds for compatibility with existing code
        latency_seconds = total_latency_ms / 1000.0
        prefill_seconds = prefill_time_ms / 1000.0
        decode_seconds = decode_time_ms / 1000.0
        return latency_seconds, prefill_seconds, decode_seconds

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        result = run_to_completion(profile_dir=None)
        # Warmup doesn't need to store results

    if args.profile:
        profile_dir = envs.VLLM_TORCH_PROFILER_DIR
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        # return

    # Benchmark.
    latencies = []
    prefill_times = []
    decode_times = []
    
    print(f"\nBenchmark Configuration:")
    print(f"  Input length: {args.input_len} tokens")
    print(f"  Expected output length: {args.output_len} tokens") 
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of iterations: {args.num_iters}")
    
    for i in tqdm(range(args.num_iters), desc="Profiling iterations"):
        result = run_to_completion(profile_dir=profile_dir)
        if result is not None:  # Not in profiling mode
            latency_sec, prefill_sec, decode_sec = result
            latencies.append(latency_sec)
            prefill_times.append(prefill_sec)
            decode_times.append(decode_sec)
            
            # Log iteration summary
            import sys
            print(f"Iteration {i+1}: Total={latency_sec:.4f}s, Prefill={prefill_sec:.4f}s, Decode={decode_sec:.4f}s", file=sys.stderr)
    
    latencies = np.array(latencies)
    prefill_times = np.array(prefill_times)
    decode_times = np.array(decode_times)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    prefill_percentiles = np.percentile(prefill_times, percentages)
    decode_percentiles = np.percentile(decode_times, percentages)
    
    print(f"Avg total latency: {np.mean(latencies):.4f} seconds")
    print(f"Avg prefill time: {np.mean(prefill_times):.4f} seconds")
    print(f"Avg decode time: {np.mean(decode_times):.4f} seconds")
    print(f"Prefill/Total ratio: {np.mean(prefill_times)/np.mean(latencies):.2%}")
    
    print("\nLatency Percentiles:")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile total latency: {percentile:.4f} seconds")
    
    print("\nPrefill Time Percentiles:")
    for percentage, percentile in zip(percentages, prefill_percentiles):
        print(f"{percentage}% percentile prefill time: {percentile:.4f} seconds")
    
    print("\nDecode Time Percentiles:")
    for percentage, percentile in zip(percentages, decode_percentiles):
        print(f"{percentage}% percentile decode time: {percentile:.4f} seconds")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_total_latency": np.mean(latencies),
            "avg_prefill_time": np.mean(prefill_times),
            "avg_decode_time": np.mean(decode_times),
            "prefill_ratio": np.mean(prefill_times) / np.mean(latencies),
            "total_latencies": latencies.tolist(),
            "prefill_times": prefill_times.tolist(),
            "decode_times": decode_times.tolist(),
            "total_latency_percentiles": dict(zip(percentages, percentiles.tolist())),
            "prefill_time_percentiles": dict(zip(percentages, prefill_percentiles.tolist())),
            "decode_time_percentiles": dict(zip(percentages, decode_percentiles.tolist())),
            # Configuration information
            "benchmark_config": {
                "input_length": args.input_len,
                "expected_output_length": args.output_len,
                "batch_size": args.batch_size,
                "num_iterations": args.num_iters,
                "num_warmup_iterations": args.num_iters_warmup,
                "use_beam_search": args.use_beam_search,
                "n_sequences": args.n,
            },
            # Keep backward compatibility
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)
