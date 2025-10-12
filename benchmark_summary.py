#!/usr/bin/env python3
"""
Benchmark Summary Script for vLLM Prefill/Decode Benchmarks

This script loads and summarizes benchmark results from both latency (offline) 
and serving (online) benchmarks, providing formatted tables with key metrics.
"""

import json
import os
import sys
from pathlib import Path
import argparse


def load_results(result_dir):
    """Load all benchmark results from the directory."""
    latency_results = {}
    serving_results = {}
    
    for file_path in Path(result_dir).glob("*.json"):
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            filename = file_path.name
            if filename.startswith("latency_"):
                # Parse filename: latency_in128_out32_bs1_mbt2048.json
                parts = filename.replace("latency_", "").replace(".json", "").split("_")
                config = {
                    'input_len': int(parts[0][2:]),  # Remove 'in' prefix
                    'output_len': int(parts[1][3:]), # Remove 'out' prefix  
                    'batch_size': int(parts[2][2:]), # Remove 'bs' prefix
                    'max_batched_tokens': int(parts[3][3:]) # Remove 'mbt' prefix
                }
                latency_results[filename] = {'config': config, 'data': data}
                
            elif filename.startswith("serving_"):
                # Parse filename: serving_in128_out32_mbt2048_np4.json
                parts = filename.replace("serving_", "").replace(".json", "").split("_")
                config = {
                    'input_len': int(parts[0][2:]),  # Remove 'in' prefix
                    'output_len': int(parts[1][3:]), # Remove 'out' prefix
                    'max_batched_tokens': int(parts[2][3:]), # Remove 'mbt' prefix
                    'num_prompts': int(parts[3][2:]) # Remove 'np' prefix
                }
                serving_results[filename] = {'config': config, 'data': data}
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return latency_results, serving_results


def print_summary(latency_results, serving_results, show_details=True):
    """Print a summary of benchmark results."""
    print("=" * 100)
    print("vLLM PREFILL AND DECODE BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Print statistics
    print(f"\nLoaded Results:")
    print(f"  Latency (offline) configurations: {len(latency_results)}")
    print(f"  Serving (online) configurations: {len(serving_results)}")
    
    # Latency results summary
    if latency_results and show_details:
        print("\nOFFLINE LATENCY RESULTS:")
        print("-" * 80)
        print(f"{'Input':<8} {'Output':<8} {'Batch':<8} {'MaxBatch':<10} {'Avg Latency':<12} {'P50':<8} {'P99':<8}")
        print(f"{'Len':<8} {'Len':<8} {'Size':<8} {'Tokens':<10} {'(s)':<12} {'(s)':<8} {'(s)':<8}")
        print("-" * 80)
        
        # Sort by input_len, output_len, batch_size, max_batched_tokens
        sorted_latency = sorted(latency_results.items(), 
                               key=lambda x: (x[1]['config']['input_len'],
                                            x[1]['config']['output_len'], 
                                            x[1]['config']['batch_size'],
                                            x[1]['config']['max_batched_tokens']))
        
        for filename, result in sorted_latency:
            config = result['config']
            data = result['data']
            
            if 'avg_latency' in data and 'percentiles' in data:
                avg_lat = data['avg_latency']
                p50 = data['percentiles'].get(50, 0)
                p99 = data['percentiles'].get(99, 0)
                
                print(f"{config['input_len']:<8} {config['output_len']:<8} "
                      f"{config['batch_size']:<8} {config['max_batched_tokens']:<10} "
                      f"{avg_lat:<12.3f} {p50:<8.3f} {p99:<8.3f}")
            else:
                print(f"{config['input_len']:<8} {config['output_len']:<8} "
                      f"{config['batch_size']:<8} {config['max_batched_tokens']:<10} "
                      f"{'N/A':<12} {'N/A':<8} {'N/A':<8}")
    
    # Serving results summary
    if serving_results and show_details:
        print("\nONLINE SERVING RESULTS (TTFT/TPOT):")
        print("-" * 105)
        print(f"{'Input':<8} {'Output':<8} {'MaxBatch':<10} {'NumReqs':<8} {'Mean TTFT':<12} {'Mean TPOT':<12} {'Throughput':<12}")
        print(f"{'Len':<8} {'Len':<8} {'Tokens':<10} {'(#)':<8} {'(ms)':<12} {'(ms)':<12} {'(req/s)':<12}")
        print("-" * 105)
        
        # Sort by input_len, output_len, max_batched_tokens, num_prompts
        sorted_serving = sorted(serving_results.items(),
                               key=lambda x: (x[1]['config']['input_len'],
                                            x[1]['config']['output_len'],
                                            x[1]['config']['max_batched_tokens'],
                                            x[1]['config']['num_prompts']))
        
        for filename, result in sorted_serving:
            config = result['config']
            data = result['data']
            
            ttft = data.get('mean_ttft_ms', 0)
            tpot = data.get('mean_tpot_ms', 0)  
            throughput = data.get('request_throughput', 0)
            
            print(f"{config['input_len']:<8} {config['output_len']:<8} "
                  f"{config['max_batched_tokens']:<10} {config['num_prompts']:<8} {ttft:<12.1f} "
                  f"{tpot:<12.1f} {throughput:<12.2f}")
    
    print("\n" + "=" * 100)


def export_csv(latency_results, serving_results, output_dir):
    """Export results to CSV files."""
    import csv
    
    # Export latency results
    if latency_results:
        latency_csv_path = Path(output_dir) / "latency_results.csv"
        with open(latency_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['input_len', 'output_len', 'batch_size', 'max_batched_tokens', 
                         'avg_latency', 'p50_latency', 'p99_latency', 'throughput']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort by configuration values in numerical order
            sorted_latency = sorted(latency_results.items(), 
                                   key=lambda x: (x[1]['config']['input_len'],
                                                x[1]['config']['output_len'], 
                                                x[1]['config']['batch_size'],
                                                x[1]['config']['max_batched_tokens']))
            
            for filename, result in sorted_latency:
                config = result['config']
                data = result['data']
                
                row = {
                    'input_len': config['input_len'],
                    'output_len': config['output_len'],
                    'batch_size': config['batch_size'],
                    'max_batched_tokens': config['max_batched_tokens'],
                    'avg_latency': data.get('avg_latency', 0),
                    'p50_latency': data.get('percentiles', {}).get(50, 0),
                    'p99_latency': data.get('percentiles', {}).get(99, 0),
                    'throughput': data.get('throughput', 0)
                }
                writer.writerow(row)
        
        print(f"Latency results exported to: {latency_csv_path}")
    
    # Export serving results
    if serving_results:
        serving_csv_path = Path(output_dir) / "serving_results.csv"
        with open(serving_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['input_len', 'output_len', 'max_batched_tokens', 'num_prompts',
                         'mean_ttft_ms', 'mean_tpot_ms', 'request_throughput']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort by configuration values in numerical order
            sorted_serving = sorted(serving_results.items(),
                                   key=lambda x: (x[1]['config']['input_len'],
                                                x[1]['config']['output_len'],
                                                x[1]['config']['max_batched_tokens'],
                                                x[1]['config']['num_prompts']))
            
            for filename, result in sorted_serving:
                config = result['config']
                data = result['data']
                
                row = {
                    'input_len': config['input_len'],
                    'output_len': config['output_len'],
                    'max_batched_tokens': config['max_batched_tokens'],
                    'num_prompts': config['num_prompts'],
                    'mean_ttft_ms': data.get('mean_ttft_ms', 0),
                    'mean_tpot_ms': data.get('mean_tpot_ms', 0),
                    'request_throughput': data.get('request_throughput', 0)
                }
                writer.writerow(row)
        
        print(f"Serving results exported to: {serving_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize vLLM benchmark results"
    )
    parser.add_argument("result_dir", 
                       help="Directory containing benchmark result files")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export results to CSV files")
    parser.add_argument("--summary-only", action="store_true",
                       help="Show only summary statistics, not detailed tables")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory '{args.result_dir}' does not exist")
        sys.exit(1)
    
    # Load results
    latency_results, serving_results = load_results(args.result_dir)
    
    if not latency_results and not serving_results:
        print(f"No benchmark result files found in '{args.result_dir}'")
        sys.exit(1)
    
    # Print summary
    print_summary(latency_results, serving_results, show_details=not args.summary_only)
    
    # Export CSV if requested
    if args.export_csv:
        export_csv(latency_results, serving_results, args.result_dir)


if __name__ == "__main__":
    main()