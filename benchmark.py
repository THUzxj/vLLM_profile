#!/usr/bin/env python3
"""
Transformer Model Benchmarking Script
Measures prefill time, decode time per token, and total latency for different batch sizes and input lengths.
"""

import os
import time
import json
import torch
import psutil
from datetime import datetime
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import pandas as pd
import numpy as np
from config import BenchmarkConfig


class TransformerBenchmark:
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the benchmark with a transformer model.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ("auto", "cpu", "cuda", "cuda:0", etc.)
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        self.model.eval()
        
        # Warm up GPU if using CUDA
        if "cuda" in str(self.device):
            self._warmup()
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate the computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        if "cuda" in device and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def _warmup(self, warmup_steps: int = 3):
        """Warm up the model to ensure consistent timing measurements."""
        print("Warming up model...")
        dummy_text = "This is a warmup text for the model."
        
        for _ in range(warmup_steps):
            inputs = self.tokenizer(dummy_text, return_tensors="pt")
            if "cuda" in str(self.device):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            if "cuda" in str(self.device):
                torch.cuda.synchronize()
    
    def generate_input_tokens(self, batch_size: int, input_length: int) -> torch.Tensor:
        """
        Generate input token tensors with exact token length for benchmarking.
        
        Args:
            batch_size: Number of input sequences
            input_length: Exact number of tokens per input
            
        Returns:
            Tensor of shape (batch_size, input_length) containing token IDs
        """
        # Get vocabulary size
        vocab_size = self.tokenizer.vocab_size
        
        # Generate diverse token sequences for each batch item
        input_ids = []
        
        for i in range(batch_size):
            # Create a diverse token sequence for each batch item
            # Start with some common tokens to make it more realistic
            common_tokens = [
                1,    # Often a common token like <s> or similar
                262,  # Common word tokens 
                318,  
                257,
                286,
            ]
            
            # Create base sequence with some structure
            if i == 0:
                # First batch item: use a simple ascending pattern
                base_tokens = list(range(100, 100 + min(50, input_length)))
            else:
                # Other batch items: create variation based on batch index
                offset = (i * 1000) % (vocab_size // 2)  # Ensure we stay within vocab
                base_tokens = list(range(offset, offset + min(50, input_length)))
            
            # Ensure all tokens are within vocabulary
            base_tokens = [token % vocab_size for token in base_tokens]
            
            # Extend or truncate to exact length
            if len(base_tokens) >= input_length:
                # Truncate to exact length
                tokens = base_tokens[:input_length]
            else:
                # Extend to exact length with pattern
                tokens = base_tokens.copy()
                
                # Fill remaining positions with a repeating pattern
                pattern_offset = i * 100  # Different pattern for each batch item
                while len(tokens) < input_length:
                    # Create a repeating pattern based on position
                    pos = len(tokens)
                    next_token = ((pos + pattern_offset) % (vocab_size // 4)) + 1
                    tokens.append(next_token)
            
            # Ensure exactly the target length
            tokens = tokens[:input_length]
            
            # Add to batch
            input_ids.append(tokens)
        
        # Convert to tensor
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        
        # Move to appropriate device
        if "cuda" in str(self.device):
            input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def measure_inference(self, 
                         texts: List[str], 
                         max_new_tokens: int,
                         temperature: float = 1.0,
                         top_p: float = 0.9) -> Dict[str, Any]:
        """
        Measure inference performance for given inputs using accurate TTFT and per-token decode time.
        
        Args:
            texts: List of input texts
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary containing timing measurements and metadata
        """
        # For backward compatibility, check if texts are provided or if we need to generate tokens
        if isinstance(texts, torch.Tensor):
            # Direct token tensor input
            inputs = {
                "input_ids": texts,
                "attention_mask": torch.ones_like(texts)
            }
            batch_size = texts.shape[0]
            input_length = texts.shape[1]
        else:
            # Legacy text input (for backward compatibility)
            batch_size = len(texts)
            
            # Tokenize inputs
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=None
            )
            
            if "cuda" in str(self.device):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
        
        # Clear cache before measurement
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        
        # Measure TTFT (Time To First Token) - Prefill Time
        ttft_start = time.perf_counter()
        if "cuda" in str(self.device):
            torch.cuda.synchronize()
        
        with torch.no_grad():
            # Generate first token only to measure prefill time
            first_token_outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True
            )
        
        if "cuda" in str(self.device):
            torch.cuda.synchronize()
        
        ttft_end = time.perf_counter()
        prefill_time = ttft_end - ttft_start  # Actual prefill time (TTFT)
        
        # If we need more than 1 token, measure decode time
        if max_new_tokens > 1:
            # Use the output from first token generation as input for remaining tokens
            decode_start = time.perf_counter()
            if "cuda" in str(self.device):
                torch.cuda.synchronize()
            
            # Prepare inputs for remaining tokens (starting from the first generated token)
            decode_inputs = {
                "input_ids": first_token_outputs.sequences,
                "attention_mask": torch.ones_like(first_token_outputs.sequences)
            }
            
            with torch.no_grad():
                # Generate remaining tokens
                remaining_outputs = self.model.generate(
                    **decode_inputs,
                    max_new_tokens=max_new_tokens - 1,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True
                )
            
            if "cuda" in str(self.device):
                torch.cuda.synchronize()
            
            decode_end = time.perf_counter()
            decode_time = decode_end - decode_start
            
            # Combine sequences for final output
            final_sequences = remaining_outputs.sequences
            actual_new_tokens = final_sequences.shape[1] - input_length
            total_decode_tokens = actual_new_tokens - 1  # Exclude first token from decode time
            
        else:
            # Only generated 1 token
            decode_time = 0.0
            final_sequences = first_token_outputs.sequences
            actual_new_tokens = 1
            total_decode_tokens = 0
        
        # Calculate total time
        total_time = prefill_time + decode_time
        
        # Calculate per-token decode time
        decode_time_per_token = decode_time / max(total_decode_tokens, 1) if total_decode_tokens > 0 else 0.0
        
        # Memory usage
        memory_info = self._get_memory_usage()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": batch_size,
            "input_length": input_length,
            "target_output_length": max_new_tokens,
            "actual_output_length": actual_new_tokens,
            "total_latency": total_time,
            "prefill_time": prefill_time,  # Accurate TTFT measurement
            "decode_time": decode_time,    # Accurate decode time measurement
            "decode_time_per_token": decode_time_per_token,  # Accurate per-token decode time
            "decode_tokens_count": total_decode_tokens,  # Number of tokens used for decode timing
            "tokens_per_second": actual_new_tokens / total_time if total_time > 0 else 0,
            "memory_usage": memory_info,
            "temperature": temperature,
            "top_p": top_p
        }
        
        return result
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        if "cuda" in str(self.device) and torch.cuda.is_available():
            memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_info
    
    def run_benchmark(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """
        Run comprehensive benchmark with different configurations.
        
        Args:
            config: Benchmark configuration object
            
        Returns:
            List of benchmark results
        """
        results = []
        total_experiments = len(config.batch_sizes) * len(config.input_lengths)
        experiment_count = 0
        skipped_count = 0
        
        print(f"Starting benchmark with {total_experiments} configurations...")
        print(f"Model: {self.model_name}")
        print(f"Output length: {config.output_length}")
        print(f"Max batch*input product: {config.max_batch_input_product}")
        print(f"Batch sizes: {config.batch_sizes}")
        print(f"Input lengths: {config.input_lengths}")
        print("-" * 60)
        
        for batch_size in config.batch_sizes:
            for input_length in config.input_lengths:
                experiment_count += 1
                
                # Check if we should skip this experiment due to size constraints
                batch_input_product = batch_size * input_length
                if batch_input_product > config.max_batch_input_product:
                    skipped_count += 1
                    print(f"Experiment {experiment_count}/{total_experiments}: "
                          f"batch_size={batch_size}, input_length={input_length} "
                          f"SKIPPED (product={batch_input_product} > {config.max_batch_input_product})")
                    continue
                
                print(f"Experiment {experiment_count}/{total_experiments}: "
                      f"batch_size={batch_size}, input_length={input_length} "
                      f"(product={batch_input_product})")
                
                try:
                    # Generate input tokens directly
                    input_tokens = self.generate_input_tokens(batch_size, input_length)
                    
                    # Run multiple measurements for stability
                    run_results = []
                    for run in range(config.num_runs):
                        print(f"  Run {run + 1}/{config.num_runs}...", end=" ")
                        
                        result = self.measure_inference(
                            input_tokens, 
                            config.output_length,
                            temperature=config.temperature,
                            top_p=config.top_p
                        )
                        result["run_number"] = run + 1
                        run_results.append(result)
                        
                        print(f"Total: {result['total_latency']:.3f}s, "
                              f"TTFT: {result['prefill_time']:.3f}s, "
                              f"Per token: {result['decode_time_per_token']:.3f}s")
                    
                    results.extend(run_results)
                    
                    # Clear cache between experiments
                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"  Error in experiment: {e}")
                    continue
        
        print(f"\nBenchmark completed.")
        print(f"Total experiments: {total_experiments}")
        print(f"Skipped experiments: {skipped_count}")
        print(f"Completed experiments: {experiment_count - skipped_count}")
        print(f"Total results: {len(results)}")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save benchmark results to JSON and CSV files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # Save as JSON
        json_file = f"{filename}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_file}")
        
        # Save as CSV for easier analysis
        df = pd.DataFrame(results)
        csv_file = f"{filename}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
        
        return json_file, csv_file


def main():
    """Main function to run the benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transformer Model Benchmark")
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--config-file", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output filename prefix")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        config = BenchmarkConfig.from_file(args.config_file)
    else:
        config = BenchmarkConfig()  # Use default configuration
    
    # Initialize benchmark
    benchmark = TransformerBenchmark(args.model, args.device)
    
    # Run benchmark
    results = benchmark.run_benchmark(config)
    
    # Save results
    benchmark.save_results(results, args.output_file)
    
    print("\nBenchmark Summary:")
    df = pd.DataFrame(results)
    summary = df.groupby(['batch_size', 'input_length']).agg({
        'total_latency': ['mean', 'std'],
        'decode_time_per_token': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std']
    }).round(4)
    print(summary)


if __name__ == "__main__":
    main()