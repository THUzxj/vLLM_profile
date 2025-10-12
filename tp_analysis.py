#!/usr/bin/env python3
"""
TP Benchmark Results Analysis Script

This script analyzes the results from tp_benchmark.sh and generates
detailed analysis of tensor parallel scaling performance.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics

def load_benchmark_results(results_dir: str) -> Dict:
    """Load all benchmark results from directory"""
    results_dir = Path(results_dir)
    
    throughput_results = []
    latency_results = []
    
    # Load throughput results
    for file_path in results_dir.glob("throughput_*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Parse filename for configuration
            filename = file_path.stem
            parts = filename.split('_')
            config = {}
            for part in parts[1:]:  # Skip 'throughput' prefix
                if part.startswith('tp'):
                    config['tp_size'] = int(part[2:])
                elif part.startswith('in'):
                    config['input_len'] = int(part[2:])
                elif part.startswith('out'):
                    config['output_len'] = int(part[3:])
                elif part.startswith('bs'):
                    config['batch_size'] = int(part[2:])
            
            # Add performance data
            result = {**config, **data}
            throughput_results.append(result)
            
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    # Load latency results
    for file_path in results_dir.glob("latency_*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Parse filename for configuration
            filename = file_path.stem
            parts = filename.split('_')
            config = {}
            for part in parts[1:]:  # Skip 'latency' prefix
                if part.startswith('tp'):
                    config['tp_size'] = int(part[2:])
                elif part.startswith('in'):
                    config['input_len'] = int(part[2:])
                elif part.startswith('out'):
                    config['output_len'] = int(part[3:])
                elif part.startswith('bs'):
                    config['batch_size'] = int(part[2:])
            
            # Add performance data
            result = {**config, **data}
            latency_results.append(result)
            
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return {
        'throughput': throughput_results,
        'latency': latency_results
    }

def analyze_tp_scaling(results: Dict) -> Dict:
    """Analyze TP scaling efficiency"""
    throughput_data = results['throughput']
    latency_data = results['latency']
    
    analysis = {
        'throughput_scaling': {},
        'latency_scaling': {},
        'efficiency_analysis': {},
        'optimal_configs': {}
    }
    
    # Group by configuration (excluding tp_size)
    throughput_groups = {}
    latency_groups = {}
    
    # Group throughput results
    for result in throughput_data:
        key = (result['input_len'], result['output_len'], result['batch_size'])
        if key not in throughput_groups:
            throughput_groups[key] = []
        throughput_groups[key].append(result)
    
    # Group latency results
    for result in latency_data:
        key = (result['input_len'], result['output_len'], result['batch_size'])
        if key not in latency_groups:
            latency_groups[key] = []
        latency_groups[key].append(result)
    
    # Analyze throughput scaling
    for config_key, group in throughput_groups.items():
        input_len, output_len, batch_size = config_key
        config_name = f"in{input_len}_out{output_len}_bs{batch_size}"
        
        # Sort by TP size
        group.sort(key=lambda x: x['tp_size'])
        
        scaling_data = []
        baseline_throughput = None
        
        for result in group:
            tp_size = result['tp_size']
            throughput = result.get('throughput_output_token', 0)
            
            if baseline_throughput is None:
                baseline_throughput = throughput
                efficiency = 1.0
            else:
                ideal_speedup = tp_size
                actual_speedup = throughput / baseline_throughput if baseline_throughput > 0 else 0
                efficiency = actual_speedup / ideal_speedup if ideal_speedup > 0 else 0
            
            scaling_data.append({
                'tp_size': tp_size,
                'throughput': throughput,
                'efficiency': efficiency,
                'speedup': throughput / baseline_throughput if baseline_throughput > 0 else 0
            })
        
        analysis['throughput_scaling'][config_name] = scaling_data
    
    # Analyze latency scaling
    for config_key, group in latency_groups.items():
        input_len, output_len, batch_size = config_key
        config_name = f"in{input_len}_out{output_len}_bs{batch_size}"
        
        # Sort by TP size
        group.sort(key=lambda x: x['tp_size'])
        
        scaling_data = []
        
        for result in group:
            tp_size = result['tp_size']
            ttft = result.get('ttft_s', {}).get('mean', 0) * 1000  # Convert to ms
            tpot = result.get('tpot_s', {}).get('mean', 0) * 1000  # Convert to ms
            e2e = result.get('end_to_end_latency_s', {}).get('mean', 0) * 1000  # Convert to ms
            
            scaling_data.append({
                'tp_size': tp_size,
                'ttft_ms': ttft,
                'tpot_ms': tpot,
                'e2e_latency_ms': e2e
            })
        
        analysis['latency_scaling'][config_name] = scaling_data
    
    return analysis

def print_throughput_analysis(analysis: Dict):
    """Print throughput scaling analysis"""
    print("=" * 80)
    print("THROUGHPUT SCALING ANALYSIS")
    print("=" * 80)
    
    for config_name, scaling_data in analysis['throughput_scaling'].items():
        print(f"\nConfiguration: {config_name}")
        print("-" * 60)
        print(f"{'TP Size':<8} {'Throughput':<12} {'Speedup':<10} {'Efficiency':<12}")
        print(f"{'':.<8} {'(tokens/s)':<12} {'(vs TP=1)':<10} {'(%)':<12}")
        print("-" * 60)
        
        for data in scaling_data:
            efficiency_pct = data['efficiency'] * 100
            print(f"{data['tp_size']:<8} {data['throughput']:<12.1f} "
                  f"{data['speedup']:<10.2f} {efficiency_pct:<12.1f}")
        
        # Calculate average efficiency for TP > 1
        if len(scaling_data) > 1:
            avg_efficiency = statistics.mean([d['efficiency'] for d in scaling_data[1:]])
            print(f"\nAverage scaling efficiency (TP>1): {avg_efficiency*100:.1f}%")

def print_latency_analysis(analysis: Dict):
    """Print latency analysis"""
    print("\n" + "=" * 80)
    print("LATENCY ANALYSIS")
    print("=" * 80)
    
    for config_name, scaling_data in analysis['latency_scaling'].items():
        print(f"\nConfiguration: {config_name}")
        print("-" * 70)
        print(f"{'TP Size':<8} {'TTFT (ms)':<12} {'TPOT (ms)':<12} {'E2E Latency (ms)':<16}")
        print("-" * 70)
        
        for data in scaling_data:
            print(f"{data['tp_size']:<8} {data['ttft_ms']:<12.1f} "
                  f"{data['tpot_ms']:<12.1f} {data['e2e_latency_ms']:<16.1f}")

def find_optimal_configs(analysis: Dict) -> Dict:
    """Find optimal TP configurations for different metrics"""
    optimal = {
        'best_throughput': {},
        'best_efficiency': {},
        'best_latency': {}
    }
    
    # Find best throughput
    for config_name, scaling_data in analysis['throughput_scaling'].items():
        best_throughput = max(scaling_data, key=lambda x: x['throughput'])
        optimal['best_throughput'][config_name] = best_throughput
    
    # Find best efficiency (for TP > 1)
    for config_name, scaling_data in analysis['throughput_scaling'].items():
        tp_gt_1 = [d for d in scaling_data if d['tp_size'] > 1]
        if tp_gt_1:
            best_efficiency = max(tp_gt_1, key=lambda x: x['efficiency'])
            optimal['best_efficiency'][config_name] = best_efficiency
    
    # Find best latency
    for config_name, scaling_data in analysis['latency_scaling'].items():
        best_latency = min(scaling_data, key=lambda x: x['e2e_latency_ms'])
        optimal['best_latency'][config_name] = best_latency
    
    return optimal

def print_optimal_configs(optimal: Dict):
    """Print optimal configuration recommendations"""
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nBest Throughput Configurations:")
    print("-" * 50)
    for config_name, best in optimal['best_throughput'].items():
        print(f"{config_name}: TP={best['tp_size']} "
              f"({best['throughput']:.1f} tokens/s)")
    
    print("\nBest Efficiency Configurations (TP > 1):")
    print("-" * 50)
    for config_name, best in optimal['best_efficiency'].items():
        print(f"{config_name}: TP={best['tp_size']} "
              f"({best['efficiency']*100:.1f}% efficiency)")
    
    print("\nBest Latency Configurations:")
    print("-" * 50)
    for config_name, best in optimal['best_latency'].items():
        print(f"{config_name}: TP={best['tp_size']} "
              f"({best['e2e_latency_ms']:.1f}ms E2E latency)")

def export_detailed_csv(results: Dict, analysis: Dict, output_dir: str):
    """Export detailed results to CSV files"""
    import csv
    
    output_dir = Path(output_dir)
    
    # Export throughput results
    throughput_csv = output_dir / "tp_throughput_analysis.csv"
    with open(throughput_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'tp_size', 'input_len', 'output_len', 'batch_size', 
                        'throughput_tokens_s', 'speedup', 'efficiency_pct', 'elapsed_time_s'])
        
        for config_name, scaling_data in analysis['throughput_scaling'].items():
            for data in scaling_data:
                parts = config_name.split('_')
                input_len = int(parts[0][2:])
                output_len = int(parts[1][3:])
                batch_size = int(parts[2][2:])
                
                # Find original result for elapsed time
                elapsed_time = 0
                for result in results['throughput']:
                    if (result['tp_size'] == data['tp_size'] and 
                        result['input_len'] == input_len and
                        result['output_len'] == output_len and
                        result['batch_size'] == batch_size):
                        elapsed_time = result.get('elapsed_time', 0)
                        break
                
                writer.writerow([config_name, data['tp_size'], input_len, output_len, batch_size,
                               data['throughput'], data['speedup'], data['efficiency']*100, elapsed_time])
    
    # Export latency results
    latency_csv = output_dir / "tp_latency_analysis.csv"
    with open(latency_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'tp_size', 'input_len', 'output_len', 'batch_size',
                        'ttft_ms', 'tpot_ms', 'e2e_latency_ms'])
        
        for config_name, scaling_data in analysis['latency_scaling'].items():
            for data in scaling_data:
                parts = config_name.split('_')
                input_len = int(parts[0][2:])
                output_len = int(parts[1][3:])
                batch_size = int(parts[2][2:])
                
                writer.writerow([config_name, data['tp_size'], input_len, output_len, batch_size,
                               data['ttft_ms'], data['tpot_ms'], data['e2e_latency_ms']])
    
    print(f"\nCSV files exported:")
    print(f"  Throughput: {throughput_csv}")
    print(f"  Latency: {latency_csv}")

def print_scaling_summary(analysis: Dict):
    """Print overall scaling summary"""
    print("\n" + "=" * 80)
    print("TENSOR PARALLEL SCALING SUMMARY")
    print("=" * 80)
    
    # Overall efficiency statistics
    all_efficiencies = []
    for scaling_data in analysis['throughput_scaling'].values():
        for data in scaling_data:
            if data['tp_size'] > 1:  # Exclude TP=1 baseline
                all_efficiencies.append(data['efficiency'])
    
    if all_efficiencies:
        avg_efficiency = statistics.mean(all_efficiencies)
        min_efficiency = min(all_efficiencies)
        max_efficiency = max(all_efficiencies)
        
        print(f"Overall TP Scaling Statistics:")
        print(f"  Average efficiency: {avg_efficiency*100:.1f}%")
        print(f"  Best efficiency: {max_efficiency*100:.1f}%")
        print(f"  Worst efficiency: {min_efficiency*100:.1f}%")
        
        # Efficiency by TP size
        tp_efficiencies = {}
        for scaling_data in analysis['throughput_scaling'].values():
            for data in scaling_data:
                if data['tp_size'] > 1:
                    tp_size = data['tp_size']
                    if tp_size not in tp_efficiencies:
                        tp_efficiencies[tp_size] = []
                    tp_efficiencies[tp_size].append(data['efficiency'])
        
        print(f"\nEfficiency by TP Size:")
        for tp_size in sorted(tp_efficiencies.keys()):
            avg_eff = statistics.mean(tp_efficiencies[tp_size])
            print(f"  TP={tp_size}: {avg_eff*100:.1f}% average efficiency")

def main():
    parser = argparse.ArgumentParser(description="Analyze TP benchmark results")
    parser.add_argument("results_dir", help="Directory containing benchmark results")
    parser.add_argument("--export-csv", action="store_true", 
                       help="Export detailed results to CSV files")
    parser.add_argument("--summary-only", action="store_true",
                       help="Show only summary statistics")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    print("Loading benchmark results...")
    results = load_benchmark_results(args.results_dir)
    
    if not results['throughput'] and not results['latency']:
        print("Error: No benchmark results found in directory")
        sys.exit(1)
    
    print(f"Loaded {len(results['throughput'])} throughput results and "
          f"{len(results['latency'])} latency results")
    
    print("Analyzing TP scaling performance...")
    analysis = analyze_tp_scaling(results)
    
    if not args.summary_only:
        print_throughput_analysis(analysis)
        print_latency_analysis(analysis)
    
    optimal = find_optimal_configs(analysis)
    print_optimal_configs(optimal)
    print_scaling_summary(analysis)
    
    if args.export_csv:
        export_detailed_csv(results, analysis, args.results_dir)
    
    print(f"\nAnalysis completed for results in: {args.results_dir}")

if __name__ == "__main__":
    main()