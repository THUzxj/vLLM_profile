#!/usr/bin/env python3
"""
Compare TPOT values between two CSV benchmark files.

This script compares TPOT (Time Per Output Token) values from two CSV files
by matching rows with the same prompt_length and batch size (bs).
It calculates relative differences and provides summary statistics.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def compare_tpot(file1_path, file2_path, output_path=None):
    """
    Compare TPOT values between two CSV files.
    
    First averages TPOT values for each (prompt_length, bs) group in each file,
    then compares the averaged values.
    
    Args:
        file1_path: Path to first CSV file (baseline)
        file2_path: Path to second CSV file (comparison)
        output_path: Optional path to save comparison results
    """
    # Read CSV files
    print(f"Reading {file1_path}...")
    df1 = pd.read_csv(file1_path)
    print(f"  Found {len(df1)} rows")
    
    print(f"Reading {file2_path}...")
    df2 = pd.read_csv(file2_path)
    print(f"  Found {len(df2)} rows")
    
    # Group by prompt_length and bs, calculate mean tpot for each file
    print("\nCalculating averages for each (prompt_length, bs) group...")
    df1_avg = df1.groupby(['prompt_length', 'bs'], as_index=False).agg({
        'tpot': 'mean',
        'repeat_idx': 'count'  # Count number of repeats
    }).rename(columns={'tpot': 'tpot_mean', 'repeat_idx': 'num_repeats'})
    df1_avg.columns = ['prompt_length', 'bs', 'tpot_file1', 'num_repeats_file1']
    
    df2_avg = df2.groupby(['prompt_length', 'bs'], as_index=False).agg({
        'tpot': 'mean',
        'repeat_idx': 'count'  # Count number of repeats
    }).rename(columns={'tpot': 'tpot_mean', 'repeat_idx': 'num_repeats'})
    df2_avg.columns = ['prompt_length', 'bs', 'tpot_file2', 'num_repeats_file2']
    
    print(f"  File1: {len(df1_avg)} unique (prompt_length, bs) groups")
    print(f"  File2: {len(df2_avg)} unique (prompt_length, bs) groups")
    
    # Merge on prompt_length and bs
    merged = pd.merge(
        df1_avg,
        df2_avg,
        on=['prompt_length', 'bs'],
        how='inner'
    )
    
    if len(merged) == 0:
        print("\nError: No matching (prompt_length, bs) groups found between the two files.")
        return
    
    print(f"\nFound {len(merged)} matching (prompt_length, bs) groups")
    
    # Calculate relative difference
    # Relative difference = (file2 - file1) / file1 * 100
    merged['tpot_diff'] = merged['tpot_file2'] - merged['tpot_file1']
    merged['tpot_rel_diff_pct'] = (merged['tpot_diff'] / merged['tpot_file1']) * 100
    merged['tpot_ratio'] = merged['tpot_file2'] / merged['tpot_file1']
    
    # Convert to int for display
    merged['prompt_length'] = merged['prompt_length'].astype(int)
    merged['bs'] = merged['bs'].astype(int)
    
    # Sort by prompt_length, then by bs
    merged = merged.sort_values(['prompt_length', 'bs'])
    
    # Display results
    print("\n" + "="*110)
    print("TPOT Comparison Results (Averaged by prompt_length and bs)")
    print("="*110)
    print(f"{'Prompt':<10} {'BS':<6} {'Repeats1':<10} {'Repeats2':<10} {'TPOT1 (ms)':<12} {'TPOT2 (ms)':<12} "
          f"{'Diff (ms)':<12} {'Rel Diff %':<12} {'Ratio':<8}")
    print("-"*110)
    
    for _, row in merged.iterrows():
        print(f"{int(row['prompt_length']):<10} {int(row['bs']):<6} "
              f"{int(row['num_repeats_file1']):<10} {int(row['num_repeats_file2']):<10} "
              f"{row['tpot_file1']:<12.4f} {row['tpot_file2']:<12.4f} "
              f"{row['tpot_diff']:<12.4f} {row['tpot_rel_diff_pct']:<12.2f} "
              f"{row['tpot_ratio']:<8.3f}")
    
    # Summary statistics
    print("\n" + "="*100)
    print("Summary Statistics")
    print("="*100)
    
    # Overall statistics
    print(f"\nOverall Statistics (all matching rows):")
    print(f"  Mean relative difference: {merged['tpot_rel_diff_pct'].mean():.2f}%")
    print(f"  Median relative difference: {merged['tpot_rel_diff_pct'].median():.2f}%")
    print(f"  Std relative difference: {merged['tpot_rel_diff_pct'].std():.2f}%")
    print(f"  Min relative difference: {merged['tpot_rel_diff_pct'].min():.2f}%")
    print(f"  Max relative difference: {merged['tpot_rel_diff_pct'].max():.2f}%")
    print(f"  Mean TPOT ratio (file2/file1): {merged['tpot_ratio'].mean():.3f}")
    
    # Statistics by prompt_length
    print(f"\nStatistics by Prompt Length:")
    print(f"{'Prompt Length':<15} {'Count':<8} {'Mean Rel Diff %':<18} {'Mean Ratio':<12}")
    print("-"*60)
    for pl in sorted(merged['prompt_length'].unique()):
        pl_data = merged[merged['prompt_length'] == pl]
        print(f"{int(pl):<15} {len(pl_data):<8} {pl_data['tpot_rel_diff_pct'].mean():<18.2f} "
              f"{pl_data['tpot_ratio'].mean():<12.3f}")
    
    # Statistics by batch size
    print(f"\nStatistics by Batch Size:")
    print(f"{'Batch Size':<12} {'Count':<8} {'Mean Rel Diff %':<18} {'Mean Ratio':<12}")
    print("-"*60)
    for bs in sorted(merged['bs'].unique()):
        bs_data = merged[merged['bs'] == bs]
        print(f"{int(bs):<12} {len(bs_data):<8} {bs_data['tpot_rel_diff_pct'].mean():<18.2f} "
              f"{bs_data['tpot_ratio'].mean():<12.3f}")
    
    # Save results if output path specified
    if output_path:
        output_df = merged[[
            'prompt_length', 'bs', 'num_repeats_file1', 'num_repeats_file2',
            'tpot_file1', 'tpot_file2', 'tpot_diff', 'tpot_rel_diff_pct', 'tpot_ratio'
        ]]
        output_df.columns = [
            'prompt_length', 'bs', 'num_repeats_file1', 'num_repeats_file2',
            'tpot_file1', 'tpot_file2', 'tpot_diff_ms', 'tpot_rel_diff_pct', 'tpot_ratio'
        ]
        output_df.to_csv(output_path, index=False)
        print(f"\nComparison results saved to: {output_path}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Compare TPOT values between two CSV benchmark files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two files and display results
  python compare_tpot.py file1.csv file2.csv
  
  # Compare and save results to CSV
  python compare_tpot.py file1.csv file2.csv -o comparison_results.csv
        """
    )
    
    parser.add_argument('file1', type=str,
                        help='Path to first CSV file (baseline)')
    parser.add_argument('file2', type=str,
                        help='Path to second CSV file (comparison)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Optional output CSV file to save comparison results')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}")
        return
    
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}")
        return
    
    compare_tpot(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()

