"""
Aggregate Analysis Results - Aggregate trace analysis results across ranks.

This script:
1. Reads analysis CSV files from multiple ranks
2. Takes the last N steps from each rank
3. For each (step, layer, stage), takes the MAX across all ranks
4. Finally computes mean and stddev across steps for each (layer, stage)

Usage:
    python aggregate_analysis.py \
        --input-dir <analysis_results_dir> \
        --output-dir <output_dir> \
        --last-n-steps 20
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics


def load_csv_file(filepath: str) -> List[Dict]:
    """Load a CSV file and return list of row dictionaries."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["tp_rank"] = int(row["tp_rank"])
            row["step_idx"] = int(row["step_idx"])
            row["layer_idx"] = int(row["layer_idx"])
            row["dur_ms"] = float(row["dur_ms"])
            row["dur_us"] = float(row["dur_us"])
            row["kernel_count"] = int(row["kernel_count"])
            rows.append(row)
    return rows


def get_last_n_steps(rows: List[Dict], n: int) -> List[Dict]:
    """
    Get rows from the last N steps.

    Args:
        rows: All rows from a rank
        n: Number of steps to take from the end

    Returns:
        Filtered rows from last N steps
    """
    # Get all unique step indices
    steps = sorted(set(r["step_idx"] for r in rows))

    if len(steps) <= n:
        return rows

    # Take last N steps
    last_steps = set(steps[-n:])
    return [r for r in rows if r["step_idx"] in last_steps]


def aggregate_by_step_layer_stage(
    all_rows: List[Dict]
) -> Dict[Tuple[int, int, str], List[float]]:
    """
    Group data by (step, layer, stage) and collect durations from all ranks.

    Args:
        all_rows: All rows from all ranks

    Returns:
        Dict mapping (step, layer, stage) -> list of durations from all ranks
    """
    grouped = defaultdict(list)

    for row in all_rows:
        key = (row["step_idx"], row["layer_idx"], row["stage"])
        grouped[key].append(row["dur_ms"])

    return dict(grouped)


def take_max_per_key(
    grouped: Dict[Tuple[int, int, str], List[float]]
) -> Dict[Tuple[int, int, str], float]:
    """
    For each (step, layer, stage), take the MAX duration across all ranks.

    Args:
        grouped: Dict mapping (step, layer, stage) -> list of durations

    Returns:
        Dict mapping (step, layer, stage) -> max duration
    """
    return {key: max(durations) for key, durations in grouped.items()}


def compute_layer_stage_stats(
    max_durations: Dict[Tuple[int, int, str], float]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute statistics for each (layer, stage) across all steps,
    AND compute averaged statistics across all layers for each stage.

    Args:
        max_durations: Dict mapping (step, layer, stage) -> max duration

    Returns:
        Tuple of (per_layer_results, averaged_results)
        - per_layer_results: List of statistics for each (layer, stage)
        - averaged_results: List of statistics averaged across layers for each stage
    """
    # Group by (layer, stage) - first aggregate by step
    grouped_by_layer_stage = defaultdict(list)

    for (step_idx, layer_idx, stage), duration in max_durations.items():
        key = (layer_idx, stage)
        grouped_by_layer_stage[key].append(duration)

    # Compute statistics for each (layer, stage) across steps
    per_layer_results = []

    for (layer_idx, stage), durations in sorted(grouped_by_layer_stage.items()):
        if not durations:
            continue

        count = len(durations)
        mean_dur = statistics.mean(durations)
        std_dur = statistics.stdev(durations) if count > 1 else 0.0
        min_dur = min(durations)
        max_dur = max(durations)

        # Calculate percentiles
        sorted_durations = sorted(durations)
        p50 = sorted_durations[len(sorted_durations) // 2] if sorted_durations else 0
        p90_idx = int(len(sorted_durations) * 0.9)
        p95_idx = int(len(sorted_durations) * 0.95)
        p90 = sorted_durations[min(p90_idx, len(sorted_durations) - 1)] if sorted_durations else 0
        p95 = sorted_durations[min(p95_idx, len(sorted_durations) - 1)] if sorted_durations else 0

        # Determine layer type
        layer_type = "dense" if layer_idx < 3 else "moe"

        per_layer_results.append({
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "stage": stage,
            "count": count,
            "dur_mean_ms": mean_dur,
            "dur_std_ms": std_dur,
            "dur_min_ms": min_dur,
            "dur_max_ms": max_dur,
            "dur_p50_ms": p50,
            "dur_p90_ms": p90,
            "dur_p95_ms": p95,
        })

    # Now aggregate across layers for each stage
    # Group by stage, collect all layer means
    stage_layer_means = defaultdict(list)
    stage_layer_stds = defaultdict(list)
    stage_layer_p50s = defaultdict(list)
    stage_layer_p95s = defaultdict(list)

    for row in per_layer_results:
        stage = row["stage"]
        stage_layer_means[stage].append(row["dur_mean_ms"])
        stage_layer_stds[stage].append(row["dur_std_ms"])
        stage_layer_p50s[stage].append(row["dur_p50_ms"])
        stage_layer_p95s[stage].append(row["dur_p95_ms"])

    # Compute averaged statistics across layers
    averaged_results = []

    for stage in sorted(stage_layer_means.keys()):
        means = stage_layer_means[stage]
        stds = stage_layer_stds[stage]
        p50s = stage_layer_p50s[stage]
        p95s = stage_layer_p95s[stage]

        if not means:
            continue

        # Average across layers
        avg_mean = statistics.mean(means)
        avg_std = statistics.mean(stds) if stds else 0.0
        avg_p50 = statistics.mean(p50s) if p50s else 0.0
        avg_p95 = statistics.mean(p95s) if p95s else 0.0

        # Also compute std across layers (variation between layers)
        std_across_layers = statistics.stdev(means) if len(means) > 1 else 0.0

        # Determine layer type based on stage
        layer_type = "dense" if stage == "dense" else "moe"

        averaged_results.append({
            "layer_idx": "avg",  # Special marker for averaged across layers
            "layer_type": layer_type,
            "stage": stage,
            "count": len(means),  # Number of layers
            "dur_mean_ms": avg_mean,
            "dur_std_ms": avg_std,
            "dur_min_ms": min(means) if means else 0,
            "dur_max_ms": max(means) if means else 0,
            "dur_p50_ms": avg_p50,
            "dur_p90_ms": 0.0,  # Not computed for averaged
            "dur_p95_ms": avg_p95,
            "std_across_layers_ms": std_across_layers,  # Extra: variation between layers
        })

    return per_layer_results, averaged_results


def write_aggregated_csv(results: List[Dict], output_path: str) -> None:
    """Write aggregated results to CSV."""
    if not results:
        print(f"  Warning: No results to write to {output_path}")
        return

    # Check if this is averaged results (has 'avg' marker)
    is_averaged = any(str(r.get("layer_idx", "")) == "avg" for r in results)

    fieldnames = [
        "layer_idx",
        "layer_type",
        "stage",
        "count",
        "dur_mean_ms",
        "dur_std_ms",
        "dur_min_ms",
        "dur_max_ms",
        "dur_p50_ms",
        "dur_p90_ms",
        "dur_p95_ms",
    ]

    if is_averaged:
        fieldnames.append("std_across_layers_ms")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Filter to only include fields in fieldnames
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)

    print(f"  Wrote {len(results)} rows to {output_path}")


def write_summary(per_layer_results: List[Dict], averaged_results: List[Dict],
                  output_path: str, last_n_steps: int) -> None:
    """Write a human-readable summary file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MOE Analysis Aggregation Summary\n")
        f.write(f"Based on last {last_n_steps} steps from each rank\n")
        f.write("=" * 80 + "\n\n")

        # Section 1: Per-layer statistics
        f.write("=" * 80 + "\n")
        f.write("PART 1: Per-Layer Statistics (averaged across steps)\n")
        f.write("=" * 80 + "\n\n")

        # Group by layer type and stage
        dense_rows = [r for r in per_layer_results if r["layer_type"] == "dense"]
        moe_rows = [r for r in per_layer_results if r["layer_type"] == "moe"]

        # Dense layers summary
        if dense_rows:
            f.write("Dense Layers (0-2)\n")
            f.write("-" * 40 + "\n")
            for row in dense_rows:
                f.write(f"  Layer {row['layer_idx']:2d} - {row['stage']:12s}: "
                       f"mean={row['dur_mean_ms']:8.3f}ms, "
                       f"std={row['dur_std_ms']:8.3f}ms, "
                       f"p50={row['dur_p50_ms']:8.3f}ms, "
                       f"p95={row['dur_p95_ms']:8.3f}ms\n")
            f.write("\n")

        # MoE layers summary
        if moe_rows:
            f.write("MoE Layers (3-60)\n")
            f.write("-" * 40 + "\n")

            # Group by stage
            stages = ["attention", "dispatch", "expert", "combine"]
            for stage in stages:
                stage_rows = [r for r in moe_rows if r["stage"] == stage]
                if not stage_rows:
                    continue

                f.write(f"\n  Stage: {stage.upper()}\n")
                f.write(f"  {'Layer':<8} {'Mean(ms)':<12} {'Std(ms)':<12} {'P50(ms)':<12} {'P95(ms)':<12}\n")
                f.write(f"  {'-'*56}\n")

                for row in stage_rows:
                    f.write(f"  {row['layer_idx']:<8} "
                           f"{row['dur_mean_ms']:<12.3f} "
                           f"{row['dur_std_ms']:<12.3f} "
                           f"{row['dur_p50_ms']:<12.3f} "
                           f"{row['dur_p95_ms']:<12.3f}\n")

        # Section 2: Averaged across layers
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("PART 2: Statistics Averaged Across All Layers\n")
        f.write("=" * 80 + "\n\n")

        f.write("  {'Stage':<12} {'AvgMean(ms)':<14} {'AvgStd(ms)':<14} {'AvgP50(ms)':<14} {'AvgP95(ms)':<14} {'LayerStd(ms)':<14}\n")
        f.write("  " + "-" * 80 + "\n")

        for row in averaged_results:
            stage = row["stage"]
            f.write(f"  {stage:<12} "
                   f"{row['dur_mean_ms']:<14.3f} "
                   f"{row['dur_std_ms']:<14.3f} "
                   f"{row['dur_p50_ms']:<14.3f} "
                   f"{row['dur_p95_ms']:<14.3f} "
                   f"{row.get('std_across_layers_ms', 0):<14.3f}\n")

        f.write("\n  Note:\n")
        f.write("    - AvgMean: Average of per-layer means (averaged across steps and layers)\n")
        f.write("    - AvgStd: Average of per-layer stds\n")
        f.write("    - LayerStd: Standard deviation of per-layer means (variation between layers)\n")

    print(f"  Wrote summary to {output_path}")


def aggregate_analysis(
    input_dir: str,
    output_dir: str,
    last_n_steps: int = 20
) -> None:
    """
    Main aggregation function.

    Args:
        input_dir: Directory containing analysis CSV files
        output_dir: Output directory for aggregated results
        last_n_steps: Number of last steps to use from each rank
    """
    # Find all CSV files
    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".analysis.csv")
    ]

    if not csv_files:
        print(f"Error: No .analysis.csv files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Load data from all ranks
    all_rows = []
    rank_info = {}

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"  Loading: {filename}")

        rows = load_csv_file(filepath)
        if not rows:
            continue

        # Get rank info
        rank = rows[0]["tp_rank"]
        total_steps = len(set(r["step_idx"] for r in rows))

        # Filter to last N steps
        filtered_rows = get_last_n_steps(rows, last_n_steps)
        used_steps = len(set(r["step_idx"] for r in filtered_rows))

        rank_info[rank] = {
            "file": filename,
            "total_steps": total_steps,
            "used_steps": used_steps,
            "total_rows": len(rows),
            "filtered_rows": len(filtered_rows)
        }

        all_rows.extend(filtered_rows)

    print(f"\nLoaded {len(all_rows)} rows from {len(csv_files)} ranks")
    print(f"\nRank details:")
    for rank, info in sorted(rank_info.items()):
        print(f"  Rank {rank:2d}: {info['used_steps']}/{info['total_steps']} steps, "
              f"{info['filtered_rows']} rows")

    # Step 1: Group by (step, layer, stage) and collect all durations
    print(f"\nAggregating by (step, layer, stage)...")
    grouped = aggregate_by_step_layer_stage(all_rows)
    print(f"  Found {len(grouped)} unique (step, layer, stage) combinations")

    # Step 2: For each key, take MAX across all ranks
    print(f"Taking MAX across ranks for each (step, layer, stage)...")
    max_durations = take_max_per_key(grouped)
    print(f"  Computed {len(max_durations)} max values")

    # Step 3: Compute statistics for each (layer, stage) across steps,
    # AND compute statistics averaged across layers
    print(f"Computing statistics for each (layer, stage) across steps...")
    print(f"  Also computing statistics averaged across layers...")
    per_layer_results, averaged_results = compute_layer_stage_stats(max_durations)
    print(f"  Generated {len(per_layer_results)} per-layer rows")
    print(f"  Generated {len(averaged_results)} averaged rows")

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    # Write per-layer CSV
    csv_path = os.path.join(output_dir, "aggregated_stats_per_layer.csv")
    write_aggregated_csv(per_layer_results, csv_path)

    # Write averaged CSV
    avg_csv_path = os.path.join(output_dir, "aggregated_stats_averaged.csv")
    write_aggregated_csv(averaged_results, avg_csv_path)

    # Write summary (includes both)
    summary_path = os.path.join(output_dir, "aggregation_summary.txt")
    write_summary(per_layer_results, averaged_results, summary_path, last_n_steps)

    print(f"\n{'='*60}")
    print(f"Aggregation complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate MOE analysis results across multiple ranks"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing analysis CSV files"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for aggregated results"
    )

    parser.add_argument(
        "--last-n-steps",
        type=int,
        default=20,
        help="Number of last steps to use from each rank (default: 20)"
    )

    args = parser.parse_args()

    aggregate_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        last_n_steps=args.last_n_steps
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
