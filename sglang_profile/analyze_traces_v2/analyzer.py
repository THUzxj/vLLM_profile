"""
Analyzer - Main analysis orchestrator for MOE trace analysis.
"""

import csv
import os
from typing import Dict, List, Tuple

from trace_loader import KernelEvent, list_trace_files, load_kernel_events_from_file, extract_tp_rank_from_filename
from step_segmenter import segment_steps_by_combine_intervals
from layer_segmenter import segment_trace, compute_stage_duration


def compute_analysis_rows(
    segmented: Dict[int, Dict[int, Dict[str, List[KernelEvent]]]],
    trace_file: str,
    tp_rank: int
) -> List[Dict]:
    """
    Compute analysis rows from segmented data.

    Args:
        segmented: {step_idx: {layer_idx: {stage: [kernels]}}}
        trace_file: Trace filename
        tp_rank: TP rank

    Returns:
        List of row dictionaries for CSV output
    """
    rows = []

    for step_idx, layers in sorted(segmented.items()):
        for layer_idx, stages in sorted(layers.items()):
            layer_type = "dense" if layer_idx < 3 else "moe"

            for stage, kernels in stages.items():
                if not kernels:
                    continue

                start_ts, end_ts, duration_us = compute_stage_duration(kernels)

                rows.append({
                    "trace_file": trace_file,
                    "tp_rank": tp_rank,
                    "step_idx": step_idx,
                    "layer_idx": layer_idx,
                    "layer_type": layer_type,
                    "stage": stage,
                    "start_us": start_ts,
                    "end_us": end_ts,
                    "dur_us": duration_us,
                    "dur_ms": duration_us / 1000.0,
                    "kernel_count": len(kernels),
                })

    return rows


def write_analysis_csv(rows: List[Dict], output_path: str) -> None:
    """
    Write analysis rows to CSV file.

    Args:
        rows: List of row dictionaries
        output_path: Output CSV file path
    """
    if not rows:
        print(f"  Warning: No data to write to {output_path}")
        return

    fieldnames = [
        "trace_file",
        "tp_rank",
        "step_idx",
        "layer_idx",
        "layer_type",
        "stage",
        "start_us",
        "end_us",
        "dur_us",
        "dur_ms",
        "kernel_count",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"  Wrote {len(rows)} rows to {output_path}")


def analyze_trace_file(
    trace_path: str,
    gap_threshold_us: float = 10000,
    verbose: bool = True
) -> Tuple[List[Dict], Dict[int, Dict[int, Dict[str, List[KernelEvent]]]]]:
    """
    Analyze a single trace file.

    Args:
        trace_path: Path to trace file
        gap_threshold_us: Step detection threshold
        verbose: Whether to print progress

    Returns:
        Tuple of (analysis_rows, segmented_data)
    """
    filename = os.path.basename(trace_path)
    tp_rank = extract_tp_rank_from_filename(filename)

    if verbose:
        print(f"\nProcessing: {filename}")
        print(f"  TP Rank: {tp_rank}")

    # Load kernels
    if verbose:
        print(f"  Loading kernels...")

    kernels = load_kernel_events_from_file(trace_path)

    if verbose:
        print(f"  Loaded {len(kernels)} kernels")

    if not kernels:
        print(f"  Warning: No kernels found in {filename}")
        return [], {}

    # Segment into steps
    if verbose:
        print(f"  Segmenting steps (threshold={gap_threshold_us}us)...")

    step_boundaries = segment_steps_by_combine_intervals(kernels, gap_threshold_us)

    if verbose:
        print(f"  Found {len(step_boundaries)} steps")

    if not step_boundaries:
        print(f"  Warning: No steps found in {filename}")
        return [], {}

    # Segment into layers
    if verbose:
        print(f"  Segmenting layers...")

    segmented = segment_trace(kernels, step_boundaries)

    if verbose:
        num_steps = len(segmented)
        num_layers = sum(len(layers) for layers in segmented.values())
        print(f"  Segmented into {num_steps} steps, {num_layers} layer instances")

    if not segmented:
        print(f"  Warning: No segmented data found in {filename}")
        return [], {}

    # Compute analysis rows
    rows = compute_analysis_rows(segmented, filename, tp_rank)

    if verbose:
        print(f"  Generated {len(rows)} analysis rows")

        # Show stage breakdown
        stage_counts = {}
        stage_times = {}
        for r in rows:
            stage = r["stage"]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            stage_times[stage] = stage_times.get(stage, 0) + r["dur_ms"]

        print(f"  Stage breakdown:")
        for stage in sorted(stage_counts.keys()):
            count = stage_counts[stage]
            total_ms = stage_times[stage]
            avg_ms = total_ms / count if count > 0 else 0
            print(f"    {stage:12s}: {count:4d} instances, total={total_ms:10.2f}ms, avg={avg_ms:7.3f}ms")

    return rows, segmented


def run_analysis(
    trace_dir: str,
    output_dir: str,
    gap_threshold_us: float = 10000,
    max_traces: int = 0
) -> None:
    """
    Run analysis on all trace files in directory.

    Args:
        trace_dir: Input directory containing trace files
        output_dir: Output directory for CSV files
        gap_threshold_us: Step detection threshold in microseconds
        max_traces: Maximum number of traces to process (0 = all)
    """
    # Find trace files
    trace_files = list_trace_files(trace_dir)

    if not trace_files:
        print(f"Error: No trace files found in {trace_dir}")
        return

    print(f"Found {len(trace_files)} trace files")

    if max_traces > 0:
        trace_files = trace_files[:max_traces]
        print(f"Limited to first {len(trace_files)} files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each trace file
    total_files = len(trace_files)

    for i, trace_path in enumerate(trace_files, 1):
        print(f"\n[{i}/{total_files}] ({i/total_files*100:.1f}%)")

        # Analyze file
        rows, _ = analyze_trace_file(trace_path, gap_threshold_us, verbose=True)

        if not rows:
            print(f"  Skipped (no data)")
            continue

        # Generate output filename
        input_filename = os.path.basename(trace_path)
        output_filename = f"{input_filename}.analysis.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Write output
        write_analysis_csv(rows, output_path)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
