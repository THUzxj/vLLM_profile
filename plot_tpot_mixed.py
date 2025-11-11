#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_mean_tpot(csv_path: str, save_path: str = None):
    df = pd.read_csv(csv_path)

    if 'mixed_name' not in df.columns or 'tpot' not in df.columns:
        raise ValueError('CSV must contain columns "mixed_name" and "tpot"')

    # Aggregate: mean tpot per mixed_name
    agg = df.groupby('mixed_name', as_index=False)['tpot'].mean()
    agg_sorted = agg.sort_values('tpot', ascending=True)

    print('\nMean tpot per mixed_name (ascending):')
    print(agg_sorted.to_string(index=False))

    # Plot
    plt.figure(figsize=(max(6, 0.6 * len(agg_sorted)), 5))
    bars = plt.bar(agg_sorted['mixed_name'], agg_sorted['tpot'], color='C0')
    plt.xlabel('mixed requests')
    plt.ylabel('mean tpot (ms)')
    plt.title('Mean TPOT per mixed requests group')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Annotate values on bars
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.splitext(csv_path)[0] + '_tpot_bar_sorted.png'

    plt.savefig(save_path, dpi=150)
    print(f"\nSaved bar chart to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to summary CSV')
    parser.add_argument('--out', type=str, default=None, help='Path to save PNG (optional)')
    args = parser.parse_args()

    plot_mean_tpot(args.csv, args.out)
