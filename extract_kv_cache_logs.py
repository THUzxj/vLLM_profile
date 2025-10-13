#!/usr/bin/env python3
"""
KV Cache Usage Log Extractor and Analyzer

This script extracts [KV-Cache-Usage] logs from vLLM log files,
calculates relative timestamps, and saves the data to structured files.
"""

import re
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple


class KVCacheLogExtractor:
    """Extractor for KV cache usage logs"""
    
    def __init__(self):
        # Regex pattern to match KV-Cache-Usage logs
        self.kv_cache_pattern = re.compile(
            r'(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
            r'.*?\[KV-Cache-Usage\] Request (?P<request_id>\d+) - (?P<operation>\w+): '
            r'allocated_blocks=(?P<allocated_blocks>\d+), '
            r'cached_blocks=(?P<cached_blocks>\d+), '
            r'new_blocks=(?P<new_blocks>\d+), '
            r'memory_est=(?P<memory_est>[\d.]+)MB, '
            r'tokens=(?P<tokens>\d+), '
            r'block_size=(?P<block_size>\d+), '
            r'global_usage=(?P<global_usage>[\d.]+)%, '
            r'free_blocks=(?P<free_blocks>\d+)/(?P<total_blocks>\d+)'
        )
        
        # Pattern to match general timestamps for relative time calculation
        self.timestamp_pattern = re.compile(
            r'(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        )
    
    def parse_timestamp(self, timestamp_str: str, year: int = None) -> datetime:
        """Parse timestamp string to datetime object"""
        if year is None:
            year = datetime.now().year
        
        # Parse MM-DD HH:MM:SS format
        try:
            dt = datetime.strptime(f"{year}-{timestamp_str}", "%Y-%m-%d %H:%M:%S")
            return dt
        except ValueError as e:
            print(f"Error parsing timestamp {timestamp_str}: {e}")
            return None
    
    def extract_kv_cache_logs(self, log_file: str) -> List[Dict]:
        """Extract KV cache logs from a log file"""
        
        extracted_logs = []
        base_timestamp = None
        
        print(f"Extracting KV cache logs from: {log_file}")
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Find KV-Cache-Usage logs
                    match = self.kv_cache_pattern.search(line)
                    if match:
                        data = match.groupdict()
                        
                        # Parse timestamp
                        timestamp = self.parse_timestamp(data['timestamp'])
                        if timestamp is None:
                            continue
                        
                        # Set base timestamp from first log
                        if base_timestamp is None:
                            base_timestamp = timestamp
                        
                        # Calculate relative time in seconds
                        relative_time_seconds = (timestamp - base_timestamp).total_seconds()
                        
                        # Convert numeric fields
                        log_entry = {
                            'line_number': line_num,
                            'absolute_timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'relative_time_seconds': round(relative_time_seconds, 3),
                            'request_id': data['request_id'],
                            'operation': data['operation'],
                            'allocated_blocks': int(data['allocated_blocks']),
                            'cached_blocks': int(data['cached_blocks']),
                            'new_blocks': int(data['new_blocks']),
                            'memory_est_mb': float(data['memory_est']),
                            'tokens': int(data['tokens']),
                            'block_size': int(data['block_size']),
                            'global_usage_percent': float(data['global_usage']),
                            'free_blocks': int(data['free_blocks']),
                            'total_blocks': int(data['total_blocks']),
                            'used_blocks': int(data['total_blocks']) - int(data['free_blocks']),
                            'original_line': line.strip()
                        }
                        
                        extracted_logs.append(log_entry)
                        
        except FileNotFoundError:
            print(f"Error: Log file not found: {log_file}")
            return []
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
            return []
        
        print(f"Extracted {len(extracted_logs)} KV cache log entries")
        return extracted_logs
    
    def save_to_json(self, logs: List[Dict], output_file: str):
        """Save logs to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'extraction_info': {
                        'total_entries': len(logs),
                        'extraction_time': datetime.now().isoformat(),
                        'time_range_seconds': logs[-1]['relative_time_seconds'] if logs else 0
                    },
                    'kv_cache_logs': logs
                }, f, indent=2, ensure_ascii=False)
            print(f"JSON data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
    
    def save_to_csv(self, logs: List[Dict], output_file: str):
        """Save logs to CSV file"""
        if not logs:
            print("No logs to save to CSV")
            return
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if logs:
                    fieldnames = logs[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(logs)
            print(f"CSV data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
    
    def generate_summary(self, logs: List[Dict]) -> Dict:
        """Generate summary statistics"""
        if not logs:
            return {}
        
        # Calculate unique requests
        unique_requests = set(log['request_id'] for log in logs)
        
        # Calculate operations
        operations = {'allocate': 0, 'free': 0, 'allocate_failed': 0}
        for log in logs:
            op = log['operation']
            if op in operations:
                operations[op] += 1
        
        # Calculate memory statistics
        memory_values = [log['memory_est_mb'] for log in logs]
        allocated_blocks = [log['allocated_blocks'] for log in logs]
        global_usage = [log['global_usage_percent'] for log in logs]
        relative_times = [log['relative_time_seconds'] for log in logs]
        
        summary = {
            'total_entries': len(logs),
            'unique_requests': len(unique_requests),
            'operations': operations,
            'memory_stats': {
                'avg_memory_mb': round(sum(memory_values) / len(memory_values), 2) if memory_values else 0,
                'max_memory_mb': round(max(memory_values), 2) if memory_values else 0,
                'min_memory_mb': round(min(memory_values), 2) if memory_values else 0,
                'total_allocated_blocks': sum(allocated_blocks),
                'avg_global_usage_percent': round(sum(global_usage) / len(global_usage), 2) if global_usage else 0
            },
            'time_stats': {
                'duration_seconds': round(max(relative_times), 3) if relative_times else 0,
                'first_log_time': logs[0]['absolute_timestamp'] if logs else None,
                'last_log_time': logs[-1]['absolute_timestamp'] if logs else None
            },
            'request_stats': {}
        }
        
        # Per-request statistics
        for request_id in unique_requests:
            req_logs = [log for log in logs if log['request_id'] == request_id]
            req_memory = [log['memory_est_mb'] for log in req_logs]
            req_blocks = [log['allocated_blocks'] for log in req_logs]
            req_times = [log['relative_time_seconds'] for log in req_logs]
            
            summary['request_stats'][request_id] = {
                'operations': [log['operation'] for log in req_logs],
                'max_memory_mb': round(max(req_memory), 2) if req_memory else 0,
                'max_allocated_blocks': max(req_blocks) if req_blocks else 0,
                'duration_seconds': round(max(req_times) - min(req_times), 3) if len(req_times) > 1 else 0
            }
        
        return summary
    
    def save_summary(self, summary: Dict, output_file: str):
        """Save summary to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Summary saved to: {output_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")
    
    def print_summary(self, summary: Dict):
        """Print summary to console"""
        print("\n" + "="*60)
        print("KV CACHE USAGE SUMMARY")
        print("="*60)
        
        print(f"Total log entries: {summary.get('total_entries', 0)}")
        print(f"Unique requests: {summary.get('unique_requests', 0)}")
        print(f"Duration: {summary.get('time_stats', {}).get('duration_seconds', 0)} seconds")
        
        ops = summary.get('operations', {})
        print(f"\nOperations:")
        print(f"  - Allocate: {ops.get('allocate', 0)}")
        print(f"  - Free: {ops.get('free', 0)}")
        print(f"  - Allocate Failed: {ops.get('allocate_failed', 0)}")
        
        mem_stats = summary.get('memory_stats', {})
        print(f"\nMemory Statistics:")
        print(f"  - Average usage: {mem_stats.get('avg_memory_mb', 0)} MB")
        print(f"  - Peak usage: {mem_stats.get('max_memory_mb', 0)} MB")
        print(f"  - Average global usage: {mem_stats.get('avg_global_usage_percent', 0)}%")
        
        print(f"\nTop 5 requests by memory usage:")
        req_stats = summary.get('request_stats', {})
        sorted_reqs = sorted(req_stats.items(), 
                           key=lambda x: x[1].get('max_memory_mb', 0), 
                           reverse=True)[:5]
        
        for req_id, stats in sorted_reqs:
            print(f"  - {req_id}: {stats.get('max_memory_mb', 0)} MB, "
                  f"{stats.get('max_allocated_blocks', 0)} blocks")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze KV cache usage logs from vLLM log files"
    )
    parser.add_argument("input_file", 
                       help="Input log file path")
    parser.add_argument("--output-dir", "-o", 
                       help="Output directory (default: same as input file)")
    parser.add_argument("--output-prefix", "-p", 
                       help="Output file prefix (default: input filename)")
    parser.add_argument("--json-only", action="store_true",
                       help="Only generate JSON output")
    parser.add_argument("--csv-only", action="store_true", 
                       help="Only generate CSV output")
    parser.add_argument("--no-summary", action="store_true",
                       help="Skip generating summary")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress console output")
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return 1
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output prefix
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        prefix = input_path.stem + "_kv_cache"
    
    # Extract logs
    extractor = KVCacheLogExtractor()
    logs = extractor.extract_kv_cache_logs(str(input_path))
    
    if not logs:
        print("No KV cache logs found in the input file")
        return 1
    
    # Generate outputs
    if not args.csv_only:
        json_file = output_dir / f"{prefix}.json"
        extractor.save_to_json(logs, str(json_file))
    
    if not args.json_only:
        csv_file = output_dir / f"{prefix}.csv" 
        extractor.save_to_csv(logs, str(csv_file))
    
    # Generate summary
    if not args.no_summary:
        summary = extractor.generate_summary(logs)
        summary_file = output_dir / f"{prefix}_summary.json"
        extractor.save_summary(summary, str(summary_file))
        
        if not args.quiet:
            extractor.print_summary(summary)
    
    if not args.quiet:
        print(f"\n✓ Extraction completed!")
        print(f"✓ Output files saved in: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())