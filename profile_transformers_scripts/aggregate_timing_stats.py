#!/usr/bin/env python3
"""
Aggregate timing JSON files and compute statistics (mean, median, std) for
attn and ffn times across all layers and runs.

Usage:
  python aggregate_timing_stats.py \
    --root /data/xjzhang/vLLM_profile/profile_transformers_scripts/timing_results \
    --output timing_stats.csv
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_and_aggregate_stats(root: Path) -> List[Dict]:
    """Load all JSON files and aggregate statistics."""
    stats = []
    
    for path in root.rglob("*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            bs = data.get("batch_size")
            il = data.get("input_len")
            attn_implementation = data.get("attn_implementation")
            layers_runs = data.get("all_layer_details", [])
            
            if bs is None or il is None or not layers_runs:
                continue
            
            # Collect all attn and ffn values across all runs and all layers
            attn_vals = []
            ffn_vals = []
            
            for run_layers in layers_runs:
                if not run_layers:
                    continue
                for layer_data in run_layers:
                    attn_val = layer_data.get("attn")
                    ffn_val = layer_data.get("ffn")
                    if attn_val is not None:
                        attn_vals.append(attn_val)
                    if ffn_val is not None:
                        ffn_vals.append(ffn_val)
            
            if not attn_vals or not ffn_vals:
                continue
            
            # Compute statistics
            attn_mean = float(np.mean(attn_vals))
            attn_median = float(np.median(attn_vals))
            attn_std = float(np.std(attn_vals))
            
            ffn_mean = float(np.mean(ffn_vals))
            ffn_median = float(np.median(ffn_vals))
            ffn_std = float(np.std(ffn_vals))
            
            stat_record = {
                "batch_size": bs,
                "input_len": il,
                "attn_implementation": attn_implementation if attn_implementation else "default",
                "attn_mean": attn_mean,
                "attn_median": attn_median,
                "attn_std": attn_std,
                "ffn_mean": ffn_mean,
                "ffn_median": ffn_median,
                "ffn_std": ffn_std,
                "num_samples": len(attn_vals),  # Number of layer-run combinations
            }
            stats.append(stat_record)
            
        except Exception as e:
            print(f"Warning: Failed to process {path}: {e}")
            continue
    
    return stats


def write_csv(stats: List[Dict], output_path: Path):
    """Write statistics to CSV file."""
    if not stats:
        print("No statistics to write.")
        return
    
    # Sort by batch_size, then input_len
    stats_sorted = sorted(stats, key=lambda x: (x["batch_size"], x["input_len"]))
    
    fieldnames = [
        "batch_size",
        "input_len",
        "attn_implementation",
        "attn_mean",
        "attn_median",
        "attn_std",
        "ffn_mean",
        "ffn_median",
        "ffn_std",
        "num_samples",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_sorted)
    
    print(f"✓ Statistics written to {output_path}")
    print(f"  Total records: {len(stats_sorted)}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate timing statistics from JSON files"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent / "timing_results",
        help="Root directory containing timing json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "timing_stats.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()
    
    print(f"Loading timing files from {args.root}...")
    stats = load_and_aggregate_stats(args.root)
    
    if not stats:
        print("No valid timing records found.")
        return
    
    print(f"Found {len(stats)} timing records.")
    write_csv(stats, args.output)


if __name__ == "__main__":
    main()

