"""
Export Layer Kernels - Export kernels for a specific MoE layer to CSV and JSON.

This tool extracts all kernels for a specific step and layer, showing the
boundary detection and stage classification for verification.

Usage:
    python export_layer_kernels.py \
        --trace-file <trace_file> \
        --step-idx <step_index> \
        --layer-idx <layer_index> \
        --output-dir <output_directory>
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

from trace_loader import load_kernel_events_from_file, extract_tp_rank_from_filename
from step_segmenter import segment_steps_by_combine_intervals
from layer_segmenter import segment_layers_in_step, find_combine_kernel_pairs, find_dispatch_kernel_pairs
from stage_classifier import classify_kernel


def export_layer_kernels_json(
    kernels: List,
    step_idx: int,
    layer_idx: int,
    stages: Dict[str, List],
    output_path: str
) -> None:
    """
    Export layer kernels to JSON format.

    Args:
        kernels: All kernels in the layer
        step_idx: Step index
        layer_idx: Layer index
        stages: Dict of stage -> kernels
        output_path: Output JSON file path
    """
    # Build export data
    export_data = {
        "metadata": {
            "step_idx": step_idx,
            "layer_idx": layer_idx,
            "layer_type": "dense" if layer_idx < 3 else "moe",
            "total_kernels": len(kernels),
            "stage_count": len(stages)
        },
        "stages": {}
    }

    # Sort kernels by timestamp for each stage
    for stage_name, stage_kernels in stages.items():
        sorted_kernels = sorted(stage_kernels, key=lambda k: k.ts)

        kernel_list = []
        for i, k in enumerate(sorted_kernels):
            kernel_info = {
                "idx": i,
                "name": k.name,
                "ts_us": k.ts,
                "dur_us": k.dur,
                "end_us": k.ts + k.dur,
                "pid": k.pid,
                "tid": k.tid,
                "cat": k.cat
            }
            kernel_list.append(kernel_info)

        export_data["stages"][stage_name] = {
            "kernel_count": len(sorted_kernels),
            "kernels": kernel_list
        }

    # Add boundary information
    export_data["boundaries"] = {}

    # Find dispatch and combine pairs for boundary info
    for stage_name in ["dispatch", "combine"]:
        if stage_name in stages and stages[stage_name]:
            stage_kernels = sorted(stages[stage_name], key=lambda k: k.ts)
            if len(stage_kernels) >= 2:
                export_data["boundaries"][stage_name] = {
                    "first_kernel_ts": stage_kernels[0].ts,
                    "first_kernel_dur": stage_kernels[0].dur,
                    "second_kernel_ts": stage_kernels[1].ts if len(stage_kernels) > 1 else None,
                    "second_kernel_dur": stage_kernels[1].dur if len(stage_kernels) > 1 else None,
                    "pair_end_ts": stage_kernels[1].ts + stage_kernels[1].dur if len(stage_kernels) > 1 else None
                }

    # Write JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"  JSON exported: {output_path}")


def export_layer_kernels_csv(
    kernels: List,
    step_idx: int,
    layer_idx: int,
    stages: Dict[str, List],
    output_path: str
) -> None:
    """
    Export layer kernels to CSV format.

    Args:
        kernels: All kernels in the layer
        step_idx: Step index
        layer_idx: Layer index
        stages: Dict of stage -> kernels
        output_path: Output CSV file path
    """
    fieldnames = [
        "step_idx",
        "layer_idx",
        "layer_type",
        "stage",
        "kernel_idx",
        "kernel_name",
        "ts_us",
        "dur_us",
        "end_us",
        "pid",
        "tid",
        "cat"
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write kernels for each stage
        layer_type = "dense" if layer_idx < 3 else "moe"

        for stage_name in ["attention", "dispatch", "expert", "combine", "dense"]:
            if stage_name not in stages:
                continue

            stage_kernels = sorted(stages[stage_name], key=lambda k: k.ts)

            for kernel_idx, k in enumerate(stage_kernels):
                # Truncate very long kernel names
                name = k.name
                if len(name) > 200:
                    name = name[:197] + "..."

                writer.writerow({
                    "step_idx": step_idx,
                    "layer_idx": layer_idx,
                    "layer_type": layer_type,
                    "stage": stage_name,
                    "kernel_idx": kernel_idx,
                    "kernel_name": name,
                    "ts_us": k.ts,
                    "dur_us": k.dur,
                    "end_us": k.ts + k.dur,
                    "pid": k.pid,
                    "tid": k.tid,
                    "cat": k.cat
                })

    print(f"  CSV exported: {output_path}")


def extract_layer_kernels(
    trace_path: str,
    step_idx: int,
    layer_idx: int,
    gap_threshold_us: float = 10000,
    verbose: bool = True
) -> Dict:
    """
    Extract kernels for a specific layer.

    Args:
        trace_path: Path to trace file
        step_idx: Step index to extract
        layer_idx: Layer index to extract
        gap_threshold_us: Step detection threshold
        verbose: Whether to print progress

    Returns:
        Dict with layer information and stages
    """
    if verbose:
        print(f"\nProcessing: {os.path.basename(trace_path)}")
        print(f"  Extracting Step {step_idx}, Layer {layer_idx}")

    # Load kernels
    if verbose:
        print(f"  Loading kernels...")

    kernels = load_kernel_events_from_file(trace_path)

    if verbose:
        print(f"  Loaded {len(kernels)} kernels")

    if not kernels:
        raise ValueError("No kernels found in trace file")

    # Segment into steps
    step_boundaries = segment_steps_by_combine_intervals(kernels, gap_threshold_us)

    if verbose:
        print(f"  Found {len(step_boundaries)} steps")

    if step_idx >= len(step_boundaries):
        raise ValueError(f"Step index {step_idx} out of range (found {len(step_boundaries)} steps)")

    step_start, step_end = step_boundaries[step_idx]

    if verbose:
        print(f"  Step {step_idx} range: [{step_start:.0f}, {step_end:.0f}) us")

    # Extract layer
    layers = segment_layers_in_step(kernels, step_start, step_end)

    if verbose:
        print(f"  Found {len(layers)} layers in step {step_idx}")

    if layer_idx not in layers:
        raise ValueError(f"Layer index {layer_idx} not found (available: {sorted(layers.keys())})")

    stages = layers[layer_idx]

    if verbose:
        print(f"\n  Layer {layer_idx} ({'dense' if layer_idx < 3 else 'moe'}) stages:")
        for stage_name, stage_kernels in stages.items():
            if stage_kernels:
                print(f"    {stage_name:12s}: {len(stage_kernels)} kernels")

    return {
        "step_idx": step_idx,
        "layer_idx": layer_idx,
        "step_start": step_start,
        "step_end": step_end,
        "stages": stages,
        "all_kernels": kernels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export kernels for a specific MoE layer to CSV and JSON for verification"
    )

    parser.add_argument(
        "--trace-file",
        required=True,
        help="Path to trace file (*.trace.json or *.trace.json.gz)"
    )

    parser.add_argument(
        "--step-idx",
        type=int,
        default=0,
        help="Step index to extract (default: 0)"
    )

    parser.add_argument(
        "--layer-idx",
        type=int,
        default=4,
        help="Layer index to extract (default: 4, first MoE layer)"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for CSV and JSON files"
    )

    parser.add_argument(
        "--gap-threshold-us",
        type=float,
        default=10000,
        help="Gap threshold in microseconds for step detection (default: 10000 = 10ms)"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.trace_file):
        print(f"Error: Trace file not found: {args.trace_file}")
        return 1

    # Extract layer kernels
    try:
        result = extract_layer_kernels(
            args.trace_file,
            args.step_idx,
            args.layer_idx,
            args.gap_threshold_us,
            verbose=True
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Generate output filenames
    trace_basename = os.path.basename(args.trace_file)
    base_name = f"{trace_basename}_step{args.step_idx}_layer{args.layer_idx}"

    json_path = os.path.join(args.output_dir, f"{base_name}.kernels.json")
    csv_path = os.path.join(args.output_dir, f"{base_name}.kernels.csv")

    # Export
    print(f"\nExporting layer kernels...")
    export_layer_kernels_json(
        result["all_kernels"],
        args.step_idx,
        args.layer_idx,
        result["stages"],
        json_path
    )
    export_layer_kernels_csv(
        result["all_kernels"],
        args.step_idx,
        args.layer_idx,
        result["stages"],
        csv_path
    )

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  Step: {args.step_idx}")
    print(f"  Layer: {args.layer_idx} ({'dense' if args.layer_idx < 3 else 'moe'})")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
