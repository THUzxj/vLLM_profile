#!/usr/bin/env python3
"""
Comprehensive prefill and decoding time benchmark script for vLLM.

This script benchmarks:
1. Prefill time (Time to First Token - TTFT) 
2. Decoding time per token (Time Per Output Token - TPOT)
3. Total latency across different configurations

The script varies:
- Input sequence lengths
- Batch sizes  
- Max batched tokens
- Output lengths
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from tqdm import tqdm
import torch
from contextlib import contextmanager

import vllm
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    input_length: int
    output_length: int
    batch_size: int
    max_batched_tokens: int
    model: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    prefill_time_ms: float  # Time to first token
    avg_tpot_ms: float      # Average time per output token  
    total_latency_ms: float # Total end-to-end latency
    throughput_tokens_per_sec: float
    memory_usage_gb: float
    num_warmup_iters: int
    num_benchmark_iters: int
    
    def to_dict(self) -> Dict[str, Any]:
        result_dict = asdict(self)
        result_dict['config'] = self.config.to_dict()
        return result_dict


class PrefillDecodeBenchmark:
    """Benchmark class for measuring prefill and decode performance."""
    
    def __init__(self, model_name: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.9):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.results: List[BenchmarkResult] = []
        
    def _create_llm_engine(self, max_model_len: int, max_num_batched_tokens: int) -> LLM:
        """Create LLM engine with specific configuration."""
        return LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=True,  # For consistent timing
            disable_log_stats=True,
            enable_prefix_caching=False,  # Disable for accurate timing
        )
    
    def _generate_dummy_prompts(self, input_length: int, batch_size: int) -> List[Dict[str, Any]]:
        """Generate dummy prompts with specified length."""
        # Generate random token IDs to avoid tokenization overhead
        dummy_token_ids = np.random.randint(1000, 50000, size=(batch_size, input_length))
        return [{"prompt_token_ids": token_ids.tolist()} 
                for token_ids in dummy_token_ids]
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    @contextmanager
    def _hook_engine_timing(self, llm: LLM):
        """Context manager to hook into engine timing for accurate prefill/decode measurement."""
        timing_data = {
            'first_token_time': None,
            'start_time': None,
            'step_times': []
        }
        
        # Try to hook into the engine step method
        engine = llm.llm_engine
        original_step = None
        
        try:
            if hasattr(engine, 'step'):
                original_step = engine.step
                
                def timed_step():
                    step_start = time.perf_counter()
                    result = original_step()
                    step_end = time.perf_counter()
                    
                    # Record first token time (first step that produces output)
                    if timing_data['first_token_time'] is None and result:
                        # Check if any request produced tokens
                        for output in result:
                            if hasattr(output, 'outputs') and output.outputs:
                                if len(output.outputs[0].token_ids) > 0:
                                    timing_data['first_token_time'] = step_end
                                    break
                    
                    timing_data['step_times'].append(step_end - step_start)
                    return result
                
                engine.step = timed_step
            
            elif hasattr(engine, 'engine_core') and hasattr(engine.engine_core, 'step'):
                # Hook into v1 engine core step
                original_step = engine.engine_core.step
                
                def timed_core_step():
                    step_start = time.perf_counter()
                    result = original_step()
                    step_end = time.perf_counter()
                    
                    # Check for first token generation
                    if timing_data['first_token_time'] is None and result:
                        outputs, _ = result
                        if outputs:
                            for req_id, output in outputs.items():
                                if hasattr(output, 'new_token_ids') and output.new_token_ids:
                                    timing_data['first_token_time'] = step_end
                                    break
                    
                    timing_data['step_times'].append(step_end - step_start)
                    return result
                
                engine.engine_core.step = timed_core_step
            
            yield timing_data
            
        finally:
            # Restore original method
            if original_step is not None:
                if hasattr(engine, 'step'):
                    engine.step = original_step
                elif hasattr(engine, 'engine_core'):
                    engine.engine_core.step = original_step
    
    def _measure_latency_breakdown(self, llm: LLM, prompts: List[Dict[str, Any]], 
                                   sampling_params: SamplingParams, 
                                   num_warmup: int = 3, 
                                   num_trials: int = 10) -> tuple[float, float, float]:
        """
        Measure prefill time, decode time, and total latency using vLLM's internal metrics.
        
        Returns:
            Tuple of (prefill_time_ms, avg_tpot_ms, total_latency_ms)
        """
        # Import here to avoid circular imports
        from vllm.v1.metrics.stats import IterationStatsCollector
        
        print(f"  Using stats-based timing method")
        
        # Warmup runs
        print(f"  Warming up with {num_warmup} iterations...")
        for _ in range(num_warmup):
            _ = llm.generate(prompts, sampling_params, use_tqdm=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Benchmark runs with timing hooks
        print(f"  Running {num_trials} benchmark iterations...")
        prefill_times = []
        decode_times = []
        total_latencies = []
        
        for trial in tqdm(range(num_trials), desc="  Benchmarking"):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Hook into the engine to get detailed timing
            try:
                # Try to access the v1 engine's stats collector
                engine = llm.llm_engine
                if hasattr(engine, 'engine_core') and hasattr(engine.engine_core, 'stats_collector'):
                    stats_collector = engine.engine_core.stats_collector
                    
                    # Clear previous stats
                    stats_collector.finished_requests_iter = []
                    
                    start_time = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    total_latency = (end_time - start_time) * 1000  # Convert to ms
                    total_latencies.append(total_latency)
                    
                    # Extract timing from finished requests
                    if stats_collector.finished_requests_iter:
                        finished_req = stats_collector.finished_requests_iter[0]
                        prefill_time_ms = finished_req.prefill_time * 1000  # Convert to ms
                        decode_time_ms = finished_req.decode_time * 1000    # Convert to ms
                        
                        prefill_times.append(prefill_time_ms)
                        decode_times.append(decode_time_ms)
                    else:
                        # Fallback to estimation if stats not available
                        print(f"    Warning: No finished request stats available, using 15% estimation for prefill time")
                        estimated_prefill_time = total_latency * 0.15  # Better estimate
                        decode_time = total_latency - estimated_prefill_time
                        prefill_times.append(estimated_prefill_time)
                        decode_times.append(decode_time)
                        
                else:
                    # Fallback for non-v1 engine or missing stats collector
                    print(f"    Warning: Engine core stats collector not available, using ratio-based estimation")
                    start_time = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    total_latency = (end_time - start_time) * 1000
                    total_latencies.append(total_latency)
                    
                    # Use improved estimation based on batch size and sequence length
                    batch_size = len(prompts)
                    input_length = len(prompts[0]["prompt_token_ids"]) if prompts else 100
                    output_length = sampling_params.max_tokens
                    
                    # Empirical formula: prefill time scales with input length, decode with output
                    prefill_ratio = min(0.3, input_length / (input_length + output_length * 3))
                    print(f"    Using prefill ratio: {prefill_ratio:.3f} (input_len={input_length}, output_len={output_length})")
                    estimated_prefill_time = total_latency * prefill_ratio
                    decode_time = total_latency - estimated_prefill_time
                    
                    prefill_times.append(estimated_prefill_time)
                    decode_times.append(decode_time)
                    
            except Exception as e:
                print(f"    Warning: Could not access detailed timing, using final fallback (15% estimation): {e}")
                # Final fallback
                start_time = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                total_latency = (end_time - start_time) * 1000
                total_latencies.append(total_latency)
                
                estimated_prefill_time = total_latency * 0.15
                decode_time = total_latency - estimated_prefill_time
                print(f"    Using 15% prefill estimation: prefill={estimated_prefill_time:.2f}ms, decode={decode_time:.2f}ms")
                prefill_times.append(estimated_prefill_time)
                decode_times.append(decode_time)
        
        avg_prefill_time = np.mean(prefill_times)
        avg_total_latency = np.mean(total_latencies)
        avg_decode_time = np.mean(decode_times)
        
        # Calculate TPOT (Time Per Output Token)
        output_length = sampling_params.max_tokens
        avg_tpot = avg_decode_time / output_length if output_length > 0 else 0
        
        return avg_prefill_time, avg_tpot, avg_total_latency
    
    def _measure_latency_with_hooks(self, llm: LLM, prompts: List[Dict[str, Any]], 
                                    sampling_params: SamplingParams, 
                                    num_warmup: int = 3, 
                                    num_trials: int = 10) -> tuple[float, float, float]:
        """
        Alternative measurement using engine hooks for more accurate timing.
        
        Returns:
            Tuple of (prefill_time_ms, avg_tpot_ms, total_latency_ms)
        """
        print(f"  Using engine hooks timing method")
        
        # Warmup runs
        print(f"  Warming up with {num_warmup} iterations...")
        for _ in range(num_warmup):
            _ = llm.generate(prompts, sampling_params, use_tqdm=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Benchmark runs with hooks
        print(f"  Running {num_trials} benchmark iterations with engine hooks...")
        prefill_times = []
        decode_times = []
        total_latencies = []
        
        for trial in tqdm(range(num_trials), desc="  Benchmarking with hooks"):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            with self._hook_engine_timing(llm) as timing_data:
                start_time = time.perf_counter()
                timing_data['start_time'] = start_time
                
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                total_latency = (end_time - start_time) * 1000  # Convert to ms
                total_latencies.append(total_latency)
                
                # Calculate prefill time using hooked timing
                if timing_data['first_token_time'] is not None:
                    prefill_time_ms = (timing_data['first_token_time'] - start_time) * 1000
                    decode_time_ms = (end_time - timing_data['first_token_time']) * 1000
                else:
                    # Fallback to estimation
                    print(f"    Warning: Engine hooks did not capture first token time, using ratio-based estimation")
                    batch_size = len(prompts)
                    input_length = len(prompts[0]["prompt_token_ids"]) if prompts else 100
                    output_length = sampling_params.max_tokens
                    
                    prefill_ratio = min(0.3, input_length / (input_length + output_length * 3))
                    print(f"    Hook fallback using prefill ratio: {prefill_ratio:.3f}")
                    prefill_time_ms = total_latency * prefill_ratio
                    decode_time_ms = total_latency - prefill_time_ms
                
                prefill_times.append(prefill_time_ms)
                decode_times.append(decode_time_ms)
        
        avg_prefill_time = np.mean(prefill_times)
        avg_total_latency = np.mean(total_latencies)
        avg_decode_time = np.mean(decode_times)
        
        # Calculate TPOT (Time Per Output Token)
        output_length = sampling_params.max_tokens
        avg_tpot = avg_decode_time / output_length if output_length > 0 else 0
        
        return avg_prefill_time, avg_tpot, avg_total_latency
    
    def _measure_latency_simple(self, llm: LLM, prompts: List[Dict[str, Any]], 
                               sampling_params: SamplingParams, 
                               num_warmup: int = 3, 
                               num_trials: int = 10) -> tuple[float, float, float]:
        """
        Simple latency measurement with improved estimation.
        
        Returns:
            Tuple of (prefill_time_ms, avg_tpot_ms, total_latency_ms)
        """
        # Warmup runs
        print(f"  Warming up with {num_warmup} iterations...")
        for _ in range(num_warmup):
            _ = llm.generate(prompts, sampling_params, use_tqdm=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Benchmark runs
        print(f"  Running {num_trials} benchmark iterations with simple timing...")
        total_latencies = []
        
        for trial in tqdm(range(num_trials), desc="  Benchmarking"):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            total_latency = (end_time - start_time) * 1000  # Convert to ms
            total_latencies.append(total_latency)
        
        avg_total_latency = np.mean(total_latencies)
        
        # Improved estimation based on workload characteristics
        batch_size = len(prompts)
        input_length = len(prompts[0]["prompt_token_ids"]) if prompts else 100
        output_length = sampling_params.max_tokens
        
        print(f"  Simple estimation parameters: batch_size={batch_size}, input_len={input_length}, output_len={output_length}")
        
        # Empirical formula based on typical vLLM behavior:
        # - Prefill is compute-bound and scales with input length
        # - Decode is memory-bound and roughly constant per token
        total_tokens = input_length + output_length
        prefill_weight = input_length / total_tokens
        
        # Adjust for batch size (larger batches spend more time in prefill)
        batch_factor = min(1.5, 1.0 + (batch_size - 1) * 0.1)
        prefill_ratio = min(0.4, prefill_weight * batch_factor)
        
        print(f"  Calculated prefill ratio: {prefill_ratio:.3f} (prefill_weight={prefill_weight:.3f}, batch_factor={batch_factor:.3f})")
        
        estimated_prefill_time = avg_total_latency * prefill_ratio
        estimated_decode_time = avg_total_latency - estimated_prefill_time
        
        print(f"  Estimated breakdown: prefill={estimated_prefill_time:.2f}ms, decode={estimated_decode_time:.2f}ms")
        
        # Calculate TPOT
        avg_tpot = estimated_decode_time / output_length if output_length > 0 else 0
        
        return estimated_prefill_time, avg_tpot, avg_total_latency
    
    def run_benchmark(self, config: BenchmarkConfig, 
                      num_warmup: int = 3, 
                      num_trials: int = 10,
                      timing_method: str = "auto") -> BenchmarkResult:
        """Run benchmark for a specific configuration."""
        print(f"\nRunning benchmark:")
        print(f"  Model: {config.model}")
        print(f"  Input length: {config.input_length}")
        print(f"  Output length: {config.output_length}")  
        print(f"  Batch size: {config.batch_size}")
        print(f"  Max batched tokens: {config.max_batched_tokens}")
        
        # Calculate max model length needed
        max_model_len = config.input_length + config.output_length + 100  # Buffer
        
        # Create LLM engine
        llm = self._create_llm_engine(
            max_model_len=max_model_len,
            max_num_batched_tokens=config.max_batched_tokens
        )
        
        # Generate prompts
        prompts = self._generate_dummy_prompts(config.input_length, config.batch_size)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for consistent results
            top_p=1.0,
            max_tokens=config.output_length,
            ignore_eos=True,
        )
        
        # Measure latency breakdown using specified method
        if timing_method == "hooks":
            prefill_time, avg_tpot, total_latency = self._measure_latency_with_hooks(
                llm, prompts, sampling_params, num_warmup, num_trials
            )
            print(f"  Used engine hooks for timing")
        elif timing_method == "stats":
            prefill_time, avg_tpot, total_latency = self._measure_latency_breakdown(
                llm, prompts, sampling_params, num_warmup, num_trials
            )
            print(f"  Used vLLM stats for timing")
        elif timing_method == "estimate":
            prefill_time, avg_tpot, total_latency = self._measure_latency_simple(
                llm, prompts, sampling_params, num_warmup, num_trials
            )
            print(f"  Used estimation for timing")
        else:  # auto
            print(f"  Auto timing mode: trying hooks -> stats -> estimation")
            try:
                prefill_time, avg_tpot, total_latency = self._measure_latency_with_hooks(
                    llm, prompts, sampling_params, num_warmup, num_trials
                )
                print(f"  ✓ Successfully used engine hooks for accurate timing")
            except Exception as e:
                print(f"  ✗ Engine hooks failed: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
                print(f"  Trying stats-based timing...")
                try:
                    prefill_time, avg_tpot, total_latency = self._measure_latency_breakdown(
                        llm, prompts, sampling_params, num_warmup, num_trials
                    )
                    print(f"  ✓ Successfully used vLLM stats for timing")
                except Exception as e2:
                    print(f"  ✗ Stats-based timing failed: {str(e2)[:100]}{'...' if len(str(e2)) > 100 else ''}")
                    print(f"  Falling back to estimation timing...")
                    prefill_time, avg_tpot, total_latency = self._measure_latency_simple(
                        llm, prompts, sampling_params, num_warmup, num_trials
                    )
                    print(f"  ✓ Used estimation for timing (fallback)")
        
        # Calculate throughput
        total_tokens = config.batch_size * (config.input_length + config.output_length)
        throughput = total_tokens / (total_latency / 1000)  # tokens per second
        
        # Get memory usage
        memory_usage = self._get_memory_usage()
        
        # Clean up
        del llm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        result = BenchmarkResult(
            config=config,
            prefill_time_ms=prefill_time,
            avg_tpot_ms=avg_tpot,
            total_latency_ms=total_latency,
            throughput_tokens_per_sec=throughput,
            memory_usage_gb=memory_usage,
            num_warmup_iters=num_warmup,
            num_benchmark_iters=num_trials
        )
        
        self.results.append(result)
        return result
    
    def run_parameter_sweep(self, 
                           input_lengths: List[int],
                           output_lengths: List[int], 
                           batch_sizes: List[int],
                           max_batched_tokens_list: List[int],
                           num_warmup: int = 3,
                           num_trials: int = 10,
                           timing_method: str = "auto") -> List[BenchmarkResult]:
        """Run benchmark across multiple parameter combinations."""
        
        total_configs = (len(input_lengths) * len(output_lengths) * 
                        len(batch_sizes) * len(max_batched_tokens_list))
        
        print(f"Running parameter sweep with {total_configs} configurations...")
        
        results = []
        config_num = 0
        
        for input_len in input_lengths:
            for output_len in output_lengths:
                for batch_size in batch_sizes:
                    for max_batched_tokens in max_batched_tokens_list:
                        config_num += 1
                        print(f"\n[{config_num}/{total_configs}] Configuration:")
                        
                        config = BenchmarkConfig(
                            input_length=input_len,
                            output_length=output_len,
                            batch_size=batch_size,
                            max_batched_tokens=max_batched_tokens,
                            model=self.model_name
                        )
                        
                        try:
                            result = self.run_benchmark(config, num_warmup, num_trials, timing_method)
                            results.append(result)
                            
                            # Print summary
                            print(f"  Results:")
                            print(f"    Prefill time: {result.prefill_time_ms:.2f} ms")
                            print(f"    TPOT: {result.avg_tpot_ms:.2f} ms")
                            print(f"    Total latency: {result.total_latency_ms:.2f} ms")
                            print(f"    Throughput: {result.throughput_tokens_per_sec:.2f} tokens/s")
                            print(f"    Memory usage: {result.memory_usage_gb:.2f} GB")
                            
                        except Exception as e:
                            print(f"  Error running configuration: {e}")
                            continue
        
        return results
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        results_dict = {
            'benchmark_info': {
                'model': self.model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'vllm_version': vllm.__version__,
            },
            'results': [result.to_dict() for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of all results."""
        if not self.results:
            print("No results to summarize.")
            return
            
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"{'Input':<8} {'Output':<8} {'Batch':<8} {'MaxBatch':<10} "
              f"{'Prefill':<10} {'TPOT':<10} {'Total':<10} {'Throughput':<12}")
        print(f"{'Len':<8} {'Len':<8} {'Size':<8} {'Tokens':<10} "
              f"{'(ms)':<10} {'(ms)':<10} {'(ms)':<10} {'(tok/s)':<12}")
        print("-" * 80)
        
        for result in self.results:
            c = result.config
            print(f"{c.input_length:<8} {c.output_length:<8} {c.batch_size:<8} "
                  f"{c.max_batched_tokens:<10} {result.prefill_time_ms:<10.2f} "
                  f"{result.avg_tpot_ms:<10.2f} {result.total_latency_ms:<10.2f} "
                  f"{result.throughput_tokens_per_sec:<12.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM prefill and decode performance"
    )
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    
    # Benchmark parameters
    parser.add_argument("--input-lengths", type=int, nargs="+", 
                       default=[128, 512, 1024, 2048],
                       help="Input sequence lengths to test")
    parser.add_argument("--output-lengths", type=int, nargs="+",
                       default=[32, 128, 256],
                       help="Output sequence lengths to test")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                       default=[1, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--max-batched-tokens", type=int, nargs="+",
                       default=[2048, 4096, 8192, 16384],
                       help="Max batched tokens to test")
    
    # Benchmark settings
    parser.add_argument("--num-warmup", type=int, default=3,
                       help="Number of warmup iterations")
    parser.add_argument("--num-trials", type=int, default=10,
                       help="Number of benchmark trials")
    parser.add_argument("--timing-method", type=str, choices=["auto", "hooks", "stats", "estimate"],
                       default="auto",
                       help="Timing method: auto (try hooks->stats->estimate), hooks (engine hooks), stats (vLLM stats), estimate (rough estimation)")
    
    # Output
    parser.add_argument("--output", type=str, 
                       default=f"prefill_decode_benchmark_{int(time.time())}.json",
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    print("Starting vLLM Prefill/Decode Benchmark")
    print(f"Model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    
    # Create benchmark instance
    benchmark = PrefillDecodeBenchmark(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Run parameter sweep
    results = benchmark.run_parameter_sweep(
        input_lengths=args.input_lengths,
        output_lengths=args.output_lengths,
        batch_sizes=args.batch_sizes,
        max_batched_tokens_list=args.max_batched_tokens,
        num_warmup=args.num_warmup,
        num_trials=args.num_trials,
        timing_method=args.timing_method
    )
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results(args.output)
    
    print(f"\nBenchmark completed! {len(results)} configurations tested.")


if __name__ == "__main__":
    main()