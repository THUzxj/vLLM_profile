#!/usr/bin/env python3
"""
Aggregate timing JSON files and plot first-layer attention/ffn times
against batch size, input length, and their product.

Usage:
  python plot_first_layer_times.py \
    --root /data/xjzhang/vLLM_profile/profile_transformers_scripts/timing_results \
    --output-dir ./timing_plots

The script expects timing JSONs produced by run_model_timing.py, either in the
root folder (timing_bs*_len*.json) or inside per-run subfolders (bs*/).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from collections import defaultdict


def load_records(root: Path) -> List[Dict]:
    records = []
    for path in root.rglob("*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            bs = data.get("batch_size")
            il = data.get("input_len")
            layers_runs = data.get("all_layer_details", [])
            if bs is None or il is None or not layers_runs:
                continue
            # average first-layer attn/ffn over runs
            attn_vals = []
            ffn_vals = []
            for run_layers in layers_runs:
                if not run_layers:
                    continue
                first_layer = run_layers[0]
                attn_vals.append(first_layer.get("attn"))
                ffn_vals.append(first_layer.get("ffn"))
            # filter None
            attn_vals = [v for v in attn_vals if v is not None]
            ffn_vals = [v for v in ffn_vals if v is not None]
            if not attn_vals or not ffn_vals:
                continue
            record = {
                "batch_size": bs,
                "input_len": il,
                "bs_tokens": bs * il,
                "attn": sum(attn_vals) / len(attn_vals),
                "ffn": sum(ffn_vals) / len(ffn_vals),
                "path": path,
            }
            records.append(record)
        except Exception:
            continue
    return records


def plot_grouped(
    records: List[Dict],
    x_key: str,
    group_key: str,
    output_dir: Path,
    title: str,
) -> None:
    """
    Create one plot per component (attn/ffn), grouping curves by group_key.
    x_key: x-axis field in record.
    group_key: record field to create multiple series.
    """
    if not records:
        return

    for component in ["attn", "ffn"]:
        grouped: Dict = defaultdict(list)
        for r in records:
            grouped[r[group_key]].append(r)

        plt.figure()
        for g_val, items in grouped.items():
            items_sorted = sorted(items, key=lambda r: r[x_key])
            xs = [r[x_key] for r in items_sorted]
            ys = [r[component] for r in items_sorted]
            plt.plot(xs, ys, marker="o",
                     label=f"{component}, {group_key}={g_val}")

        plt.xlabel(x_key)
        plt.ylabel(f"{component} time (s)")
        plt.title(f"{title} ({component})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)

        outfile = output_dir / \
            f"first_layer_{component}_{x_key}_by_{group_key}.png"
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {outfile}")


def plot_grouped_combined(
    records: List[Dict],
    x_key: str,
    group_key: str,
    output_dir: Path,
    title: str,
) -> None:
    """
    Create a plot with both attn and ffn curves on the same figure,
    grouped by group_key.
    """
    if not records:
        return

    grouped: Dict = defaultdict(list)
    for r in records:
        grouped[r[group_key]].append(r)

    plt.figure()
    for g_val, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: r[x_key])
        xs = [r[x_key] for r in items_sorted]
        attn = [r["attn"] for r in items_sorted]
        ffn = [r["ffn"] for r in items_sorted]
        plt.plot(xs, attn, marker="o", label=f"attn, {group_key}={g_val}")
        plt.plot(xs, ffn, marker="s", label=f"ffn, {group_key}={g_val}")

    plt.xlabel(x_key)
    plt.ylabel("time (s)")
    plt.title(f"{title} (attn & ffn)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)

    outfile = output_dir / f"first_layer_attn_ffn_{x_key}_by_{group_key}.png"
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent / "timing_results",
        help="Root directory containing timing json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "timing_plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.root)
    if not records:
        print("No valid timing records found.")
        return

    # input_len as x-axis; grouped by batch_size
    plot_grouped(
        records,
        x_key="input_len",
        group_key="batch_size",
        output_dir=args.output_dir,
        title="First layer time vs input_len grouped by batch_size",
    )
    plot_grouped_combined(
        records,
        x_key="input_len",
        group_key="batch_size",
        output_dir=args.output_dir,
        title="First layer time vs input_len grouped by batch_size",
    )

    # batch_size as x-axis; grouped by input_len
    plot_grouped(
        records,
        x_key="batch_size",
        group_key="input_len",
        output_dir=args.output_dir,
        title="First layer time vs batch_size grouped by input_len",
    )
    plot_grouped_combined(
        records,
        x_key="batch_size",
        group_key="input_len",
        output_dir=args.output_dir,
        title="First layer time vs batch_size grouped by input_len",
    )

    # bs_tokens as x-axis; grouped by batch_size
    plot_grouped(
        records,
        x_key="bs_tokens",
        group_key="batch_size",
        output_dir=args.output_dir,
        title="First layer time vs batch_size*input_len grouped by batch_size",
    )
    plot_grouped_combined(
        records,
        x_key="bs_tokens",
        group_key="batch_size",
        output_dir=args.output_dir,
        title="First layer time vs batch_size*input_len grouped by batch_size",
    )


if __name__ == "__main__":
    main()
