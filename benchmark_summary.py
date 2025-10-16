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
        print("-" * 140)
        print(f"{'Input':<8} {'Output':<8} {'Batch':<8} {'MaxBatch':<10} {'Total Lat':<10} {'Prefill':<10} {'Decode':<10} {'Prefill%':<10} {'P50 Total':<10} {'P99 Total':<10}")
        print(f"{'Len':<8} {'Len':<8} {'Size':<8} {'Tokens':<10} {'(s)':<10} {'(s)':<10} {'(s)':<10} {'(%)':<10} {'(s)':<10} {'(s)':<10}")
        print("-" * 140)
        
        # Sort by input_len, output_len, batch_size, max_batched_tokens
        sorted_latency = sorted(latency_results.items(), 
                               key=lambda x: (x[1]['config']['input_len'],
                                            x[1]['config']['output_len'], 
                                            x[1]['config']['batch_size'],
                                            x[1]['config']['max_batched_tokens']))
        
        for filename, result in sorted_latency:
            config = result['config']
            data = result['data']
            
            # Try new format first (with detailed timing), fallback to old format
            if 'avg_total_latency' in data:
                # New format with detailed timing
                total_lat = data['avg_total_latency']
                prefill_time = data.get('avg_prefill_time', 0)
                decode_time = data.get('avg_decode_time', 0)
                prefill_ratio = data.get('prefill_ratio', 0) * 100  # Convert to percentage
                p50 = data.get('total_latency_percentiles', {}).get('50', 0)
                p99 = data.get('total_latency_percentiles', {}).get('99', 0)
                
                print(f"{config['input_len']:<8} {config['output_len']:<8} "
                      f"{config['batch_size']:<8} {config['max_batched_tokens']:<10} "
                      f"{total_lat:<10.3f} {prefill_time:<10.3f} {decode_time:<10.3f} "
                      f"{prefill_ratio:<10.1f} {p50:<10.3f} {p99:<10.3f}")
            elif 'avg_latency' in data and 'percentiles' in data:
                # Old format (backward compatibility)
                avg_lat = data['avg_latency']
                p50 = data['percentiles'].get(50, 0)
                p99 = data['percentiles'].get(99, 0)
                
                print(f"{config['input_len']:<8} {config['output_len']:<8} "
                      f"{config['batch_size']:<8} {config['max_batched_tokens']:<10} "
                      f"{avg_lat:<10.3f} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                      f"{p50:<10.3f} {p99:<10.3f}")
            else:
                print(f"{config['input_len']:<8} {config['output_len']:<8} "
                      f"{config['batch_size']:<8} {config['max_batched_tokens']:<10} "
                      f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        
        # Additional detailed timing breakdown for configurations with detailed timing data
        detailed_configs = [(filename, result) for filename, result in sorted_latency 
                           if 'avg_total_latency' in result['data']]
        
        if detailed_configs:
            print("\nDETAILED TIMING BREAKDOWN (Percentiles):")
            print("-" * 120)
            print(f"{'Config':<25} {'Total P50/P99':<15} {'Prefill P50/P99':<17} {'Decode P50/P99':<16} {'Prefill %':<10}")
            print(f"{'(in/out/bs/mbt)':<25} {'(s)':<15} {'(s)':<17} {'(s)':<16} {'Avg':<10}")
            print("-" * 120)
            
            for filename, result in detailed_configs:
                config = result['config']
                data = result['data']
                
                config_str = f"{config['input_len']}/{config['output_len']}/{config['batch_size']}/{config['max_batched_tokens']}"
                
                total_p50 = data.get('total_latency_percentiles', {}).get('50', 0)
                total_p99 = data.get('total_latency_percentiles', {}).get('99', 0)
                prefill_p50 = data.get('prefill_time_percentiles', {}).get('50', 0)
                prefill_p99 = data.get('prefill_time_percentiles', {}).get('99', 0)
                decode_p50 = data.get('decode_time_percentiles', {}).get('50', 0)
                decode_p99 = data.get('decode_time_percentiles', {}).get('99', 0)
                prefill_ratio = data.get('prefill_ratio', 0) * 100
                
                print(f"{config_str:<25} {total_p50:.3f}/{total_p99:.3f}   "
                      f"{prefill_p50:.3f}/{prefill_p99:.3f}     "
                      f"{decode_p50:.3f}/{decode_p99:.3f}      {prefill_ratio:.1f}%")
    
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
                         'avg_total_latency', 'avg_prefill_time', 'avg_decode_time', 'prefill_ratio',
                         'p50_total_latency', 'p99_total_latency', 'p50_prefill_time', 'p99_prefill_time',
                         'p50_decode_time', 'p99_decode_time', 'throughput',
                         # Backward compatibility fields
                         'avg_latency', 'p50_latency', 'p99_latency']
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
                
                # Base configuration
                row = {
                    'input_len': config['input_len'],
                    'output_len': config['output_len'],
                    'batch_size': config['batch_size'],
                    'max_batched_tokens': config['max_batched_tokens'],
                }
                
                # New detailed timing format
                if 'avg_total_latency' in data:
                    row.update({
                        'avg_total_latency': data.get('avg_total_latency', 0),
                        'avg_prefill_time': data.get('avg_prefill_time', 0),
                        'avg_decode_time': data.get('avg_decode_time', 0),
                        'prefill_ratio': data.get('prefill_ratio', 0),
                        'p50_total_latency': data.get('total_latency_percentiles', {}).get('50', 0),
                        'p99_total_latency': data.get('total_latency_percentiles', {}).get('99', 0),
                        'p50_prefill_time': data.get('prefill_time_percentiles', {}).get('50', 0),
                        'p99_prefill_time': data.get('prefill_time_percentiles', {}).get('99', 0),
                        'p50_decode_time': data.get('decode_time_percentiles', {}).get('50', 0),
                        'p99_decode_time': data.get('decode_time_percentiles', {}).get('99', 0),
                        # Backward compatibility
                        'avg_latency': data.get('avg_total_latency', 0),
                        'p50_latency': data.get('total_latency_percentiles', {}).get('50', 0),
                        'p99_latency': data.get('total_latency_percentiles', {}).get('99', 0),
                    })
                else:
                    # Old format (backward compatibility)
                    row.update({
                        'avg_total_latency': data.get('avg_latency', 0),
                        'avg_prefill_time': 0,
                        'avg_decode_time': 0,
                        'prefill_ratio': 0,
                        'p50_total_latency': data.get('percentiles', {}).get('50', 0),
                        'p99_total_latency': data.get('percentiles', {}).get('99', 0),
                        'p50_prefill_time': 0,
                        'p99_prefill_time': 0,
                        'p50_decode_time': 0,
                        'p99_decode_time': 0,
                        'avg_latency': data.get('avg_latency', 0),
                        'p50_latency': data.get('percentiles', {}).get('50', 0),
                        'p99_latency': data.get('percentiles', {}).get('99', 0),
                    })
                
                row['throughput'] = data.get('throughput', 0)
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