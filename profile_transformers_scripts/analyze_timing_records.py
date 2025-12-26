#!/usr/bin/env python3
"""
Analyze and visualize timing records from run_model_timing.py output.

This script reads JSON timing records and provides:
- Summary statistics
- Per-layer analysis
- Comparative analysis across different configurations
- Export to CSV for further analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import statistics

import numpy as np


class TimingAnalyzer:
    """Analyze timing records from model benchmarks."""
    
    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.records = self.load_records()
    
    def load_records(self) -> List[Dict[str, Any]]:
        """Load records from JSON file."""
        if not self.json_file.exists():
            print(f"Error: File not found: {self.json_file}")
            sys.exit(1)
        
        with open(self.json_file, 'r') as f:
            records = json.load(f)
        
        if not records:
            print("Warning: No records found in file")
        
        return records
    
    def print_overall_summary(self):
        """Print overall statistics."""
        if not self.records:
            return
        
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        print(f"Total records: {len(self.records)}")
        print(f"Date range: {self.records[0]['timestamp']} to {self.records[-1]['timestamp']}")
        
        # Collect all times
        total_times = [r['total_time'] for r in self.records]
        
        print(f"\nTotal Execution Time Statistics:")
        print(f"  Min:    {min(total_times):.4f}s")
        print(f"  Max:    {max(total_times):.4f}s")
        print(f"  Mean:   {statistics.mean(total_times):.4f}s")
        print(f"  Median: {statistics.median(total_times):.4f}s")
        if len(total_times) > 1:
            print(f"  StdDev: {statistics.stdev(total_times):.4f}s")
    
    def print_config_analysis(self):
        """Analyze performance by configuration."""
        if not self.records:
            return
        
        print("\n" + "="*80)
        print("CONFIGURATION ANALYSIS")
        print("="*80)
        
        # Group by batch size and input length
        config_groups = {}
        for record in self.records:
            key = (record['batch_size'], record['input_len'])
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(record['total_time'])
        
        # Print per configuration
        for (batch_size, input_len), times in sorted(config_groups.items()):
            print(f"\nBatch Size={batch_size}, Input Length={input_len}:")
            print(f"  Runs:   {len(times)}")
            print(f"  Mean:   {statistics.mean(times):.4f}s")
            if len(times) > 1:
                print(f"  StdDev: {statistics.stdev(times):.4f}s")
                print(f"  Min:    {min(times):.4f}s")
                print(f"  Max:    {max(times):.4f}s")
    
    def print_layer_analysis(self):
        """Analyze per-layer timing breakdown."""
        if not self.records:
            return
        
        print("\n" + "="*80)
        print("LAYER-LEVEL TIMING ANALYSIS")
        print("="*80)
        
        # Analyze first record's layer details
        if self.records and self.records[0].get('layer_details'):
            layer_details = self.records[0]['layer_details']
            
            attn_times = []
            ffn_times = []
            
            for i, layer in enumerate(layer_details):
                attn_times.append(layer.get('attn', 0))
                ffn_times.append(layer.get('ffn', 0))
            
            print(f"\nTotal Layers: {len(layer_details)}")
            
            print(f"\nAttention Layer Timing:")
            print(f"  Total:   {sum(attn_times):.4f}s")
            print(f"  Average: {statistics.mean(attn_times):.4f}s")
            print(f"  Min:     {min(attn_times):.4f}s")
            print(f"  Max:     {max(attn_times):.4f}s")
            
            print(f"\nFFN Layer Timing:")
            print(f"  Total:   {sum(ffn_times):.4f}s")
            print(f"  Average: {statistics.mean(ffn_times):.4f}s")
            print(f"  Min:     {min(ffn_times):.4f}s")
            print(f"  Max:     {max(ffn_times):.4f}s")
            
            print(f"\nAttention % of Total: {sum(attn_times)/(sum(attn_times)+sum(ffn_times))*100:.1f}%")
            print(f"FFN % of Total:       {sum(ffn_times)/(sum(attn_times)+sum(ffn_times))*100:.1f}%")
    
    def print_scaling_analysis(self):
        """Analyze performance scaling with batch size and sequence length."""
        if not self.records:
            return
        
        print("\n" + "="*80)
        print("SCALING ANALYSIS")
        print("="*80)
        
        # Group by batch size
        batch_groups = {}
        for record in self.records:
            batch_size = record['batch_size']
            if batch_size not in batch_groups:
                batch_groups[batch_size] = []
            batch_groups[batch_size].append(record)
        
        if len(batch_groups) > 1:
            print("\nScaling with Batch Size:")
            for batch_size in sorted(batch_groups.keys()):
                times = [r['total_time'] for r in batch_groups[batch_size]]
                avg_time = statistics.mean(times)
                print(f"  Batch={batch_size}: {avg_time:.4f}s (avg)")
        
        # Group by input length
        seq_groups = {}
        for record in self.records:
            input_len = record['input_len']
            if input_len not in seq_groups:
                seq_groups[input_len] = []
            seq_groups[input_len].append(record)
        
        if len(seq_groups) > 1:
            print("\nScaling with Input Length:")
            for input_len in sorted(seq_groups.keys()):
                times = [r['total_time'] for r in seq_groups[input_len]]
                avg_time = statistics.mean(times)
                print(f"  Length={input_len}: {avg_time:.4f}s (avg)")
    
    def export_to_csv(self, output_file: str = None):
        """Export timing records to CSV format."""
        if not self.records:
            print("No records to export")
            return
        
        if output_file is None:
            output_file = self.json_file.stem + "_summary.csv"
        
        try:
            import csv
        except ImportError:
            print("Error: csv module not available")
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['timestamp', 'batch_size', 'input_len', 'total_time']
            )
            writer.writeheader()
            
            for record in self.records:
                writer.writerow({
                    'timestamp': record['timestamp'],
                    'batch_size': record['batch_size'],
                    'input_len': record['input_len'],
                    'total_time': record['total_time'],
                })
        
        print(f"✓ Exported to CSV: {output_file}")
    
    def generate_report(self, output_file: str = None):
        """Generate a comprehensive text report."""
        if not self.records:
            print("No records to report")
            return
        
        if output_file is None:
            output_file = self.json_file.stem + "_report.txt"
        
        report_lines = []
        
        # Capture all analysis
        print_functions = [
            self.print_overall_summary,
            self.print_config_analysis,
            self.print_layer_analysis,
            self.print_scaling_analysis,
        ]
        
        # Run analysis and capture to string
        import io
        from contextlib import redirect_stdout
        
        old_stdout = sys.stdout
        
        for func in print_functions:
            buffer = io.StringIO()
            sys.stdout = buffer
            func()
            sys.stdout = old_stdout
            report_lines.append(buffer.getvalue())
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("\n".join(report_lines))
        
        print(f"✓ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze timing records from model benchmarks"
    )
    parser.add_argument(
        "input_file",
        help="Path to timing_records.json file"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export data to CSV format"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive text report"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file name (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Load and analyze
    analyzer = TimingAnalyzer(args.input_file)
    
    # Print analysis
    analyzer.print_overall_summary()
    analyzer.print_config_analysis()
    analyzer.print_layer_analysis()
    analyzer.print_scaling_analysis()
    
    # Export if requested
    if args.export_csv:
        analyzer.export_to_csv(args.output)
    
    if args.report:
        analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()
