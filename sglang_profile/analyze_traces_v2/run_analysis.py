"""
Run Analysis - CLI entry point for MOE trace analysis v2.

Usage:
    python run_analysis.py \
        --trace-dir <input_directory> \
        --output-dir <output_directory> \
        [--gap-threshold-us 10000] \
        [--max-traces 0]
"""

import argparse
import sys

from analyzer import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MOE trace files v2 - Step and layer segmentation using combine kernel intervals"
    )

    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Directory containing *.trace.json or *.trace.json.gz files"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output CSV files"
    )

    parser.add_argument(
        "--gap-threshold-us",
        type=float,
        default=10000,
        help="Gap threshold in microseconds for step detection (default: 10000 = 10ms)"
    )

    parser.add_argument(
        "--max-traces",
        type=int,
        default=0,
        help="Maximum number of trace files to process (0 = all, default: 0)"
    )

    args = parser.parse_args()

    # Run analysis
    run_analysis(
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        gap_threshold_us=args.gap_threshold_us,
        max_traces=args.max_traces
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
