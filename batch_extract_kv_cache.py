#!/usr/bin/env python3
"""
Batch KV Cache Log Processor

This script processes all log files in a directory and extracts KV cache usage data.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def process_log_file(log_file: Path, output_dir: Path, extractor_script: str) -> dict:
    """Process a single log file"""
    try:
        result = {
            'log_file': str(log_file),
            'status': 'success',
            'output_files': [],
            'error': None
        }
        
        # Run the extraction script
        cmd = [
            'python3', extractor_script,
            str(log_file),
            '--output-dir', str(output_dir),
            '--quiet'
        ]
        
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if process.returncode == 0:
            # Check for generated files
            base_name = log_file.stem + "_kv_cache"
            expected_files = [
                output_dir / f"{base_name}.json",
                output_dir / f"{base_name}.csv", 
                output_dir / f"{base_name}_summary.json"
            ]
            
            result['output_files'] = [
                str(f) for f in expected_files if f.exists()
            ]
        else:
            result['status'] = 'error'
            result['error'] = f"Script failed (code {process.returncode}): {process.stderr}"
            if process.stdout:
                result['error'] += f" | stdout: {process.stdout}"
        
    except subprocess.TimeoutExpired:
        result = {
            'log_file': str(log_file),
            'status': 'timeout',
            'error': 'Processing timeout (60s)'
        }
    except Exception as e:
        result = {
            'log_file': str(log_file),
            'status': 'error', 
            'error': str(e)
        }
    
    return result


def find_log_files(directory: Path, pattern: str = "*.log") -> list[Path]:
    """Find all log files in directory"""
    log_files = []
    
    if directory.is_dir():
        # Recursively find log files
        log_files.extend(directory.rglob(pattern))
    elif directory.is_file() and directory.suffix == '.log':
        log_files.append(directory)
    
    return sorted(log_files)


def generate_batch_summary(results: list, output_file: Path):
    """Generate summary of batch processing"""
    
    total_files = len(results)
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'error'])
    timeout = len([r for r in results if r['status'] == 'timeout'])
    
    summary = {
        'batch_processing_summary': {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'timeout': timeout,
            'success_rate': round(successful / total_files * 100, 2) if total_files > 0 else 0
        },
        'detailed_results': results,
        'failed_files': [
            r['log_file'] for r in results if r['status'] != 'success'
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch process log files to extract KV cache usage data"
    )
    parser.add_argument("input_path",
                       help="Input directory or log file")
    parser.add_argument("--output-dir", "-o", 
                       help="Output directory (default: same as input)")
    parser.add_argument("--pattern", "-p", default="*.log",
                       help="Log file pattern (default: *.log)")
    parser.add_argument("--max-workers", "-w", type=int, default=4,
                       help="Maximum number of parallel workers (default: 4)")
    parser.add_argument("--extractor-script", 
                       default="extract_kv_cache_logs.py",
                       help="Path to extraction script")
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if input_path.is_dir():
            output_dir = input_path / "kv_cache_extracted"
        else:
            output_dir = input_path.parent / "kv_cache_extracted"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check extractor script
    extractor_script = Path(args.extractor_script)
    if not extractor_script.exists():
        # Try relative to current script
        script_dir = Path(__file__).parent
        extractor_script = script_dir / args.extractor_script
        if not extractor_script.exists():
            print(f"Error: Extractor script not found: {args.extractor_script}")
            return 1
    
    # Find log files
    log_files = find_log_files(input_path, args.pattern)
    if not log_files:
        print(f"No log files found in: {input_path}")
        return 1
    
    print(f"Found {len(log_files)} log files to process")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.max_workers} parallel workers")
    print()
    
    # Process files
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_log_file, log_file, output_dir, str(extractor_script)): log_file
            for log_file in log_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            log_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                if result['status'] == 'success':
                    print(f"✓ {log_file.name} -> {len(result['output_files'])} files")
                else:
                    print(f"✗ {log_file.name} -> {result['error']}")
                    
            except Exception as e:
                print(f"✗ {log_file.name} -> Exception: {e}")
                results.append({
                    'log_file': str(log_file),
                    'status': 'exception',
                    'error': str(e)
                })
    
    # Generate batch summary
    summary_file = output_dir / "batch_processing_summary.json"
    summary = generate_batch_summary(results, summary_file)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {summary['batch_processing_summary']['total_files']}")
    print(f"Successful: {summary['batch_processing_summary']['successful']}")
    print(f"Failed: {summary['batch_processing_summary']['failed']}")
    print(f"Timeout: {summary['batch_processing_summary']['timeout']}")
    print(f"Success rate: {summary['batch_processing_summary']['success_rate']}%")
    
    if summary['failed_files']:
        print(f"\nFailed files:")
        for failed_file in summary['failed_files']:
            print(f"  - {failed_file}")
    
    print(f"\nBatch summary saved to: {summary_file}")
    print(f"All output files saved in: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())