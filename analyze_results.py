#!/usr/bin/env python3
"""
Data analysis script for transformer benchmarking results.
Processes and visualizes experimental data with statistical analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
from pathlib import Path


class BenchmarkAnalyzer:
    """Analyzer for transformer benchmark results."""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file.
        
        Args:
            results_file: Path to JSON or CSV results file
        """
        self.results_file = results_file
        self.df = self.load_results()
        self.prepare_data()
        
    def load_results(self) -> pd.DataFrame:
        """Load benchmark results from file."""
        if self.results_file.endswith('.json'):
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.results_file.endswith('.csv'):
            df = pd.read_csv(self.results_file)
        else:
            raise ValueError("Results file must be .json or .csv")
        
        return df
    
    def prepare_data(self):
        """Prepare and clean data for analysis."""
        # Convert timestamp to datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Calculate derived metrics
        self.df['throughput_tokens_per_sec'] = self.df['actual_output_length'] / self.df['total_latency']
        # Handle both old (estimated) and new (accurate) prefill time columns
        prefill_col = 'prefill_time' if 'prefill_time' in self.df.columns else 'estimated_prefill_time'
        self.df['prefill_ratio'] = self.df[prefill_col] / self.df['total_latency']
        self.df['decode_ratio'] = self.df['decode_time'] / self.df['total_latency']
        
        # Create categorical columns for easier grouping
        self.df['batch_size_cat'] = self.df['batch_size'].astype(str)
        self.df['input_length_cat'] = self.df['input_length'].astype(str)
        
        print(f"Loaded {len(self.df)} benchmark results")
        print(f"Models: {self.df['model_name'].unique().tolist()}")
        print(f"Batch sizes: {sorted(self.df['batch_size'].unique())}")
        print(f"Input lengths: {sorted(self.df['input_length'].unique())}")
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {}
        
        # Overall statistics
        summary['overall'] = {
            'total_experiments': len(self.df),
            'unique_configurations': len(self.df.groupby(['batch_size', 'input_length'])),
            'models_tested': self.df['model_name'].unique().tolist(),
            'total_runtime': self.df['total_latency'].sum(),
            'average_latency': self.df['total_latency'].mean(),
            'average_throughput': self.df['throughput_tokens_per_sec'].mean(),
        }
        
        # Aggregate by configuration
        prefill_col = 'prefill_time' if 'prefill_time' in self.df.columns else 'estimated_prefill_time'
        config_stats = self.df.groupby(['batch_size', 'input_length']).agg({
            'total_latency': ['mean', 'std', 'min', 'max'],
            'decode_time_per_token': ['mean', 'std', 'min', 'max'],
            'throughput_tokens_per_sec': ['mean', 'std', 'min', 'max'],
            prefill_col: ['mean', 'std'],
            'decode_time': ['mean', 'std']
        }).round(4)
        
        summary['by_configuration'] = config_stats.to_dict()
        
        # Best and worst configurations
        best_throughput_idx = self.df['throughput_tokens_per_sec'].idxmax()
        worst_latency_idx = self.df['total_latency'].idxmax()
        
        summary['best_throughput'] = {
            'batch_size': int(self.df.loc[best_throughput_idx, 'batch_size']),
            'input_length': int(self.df.loc[best_throughput_idx, 'input_length']),
            'throughput': float(self.df.loc[best_throughput_idx, 'throughput_tokens_per_sec']),
            'latency': float(self.df.loc[best_throughput_idx, 'total_latency'])
        }
        
        summary['worst_latency'] = {
            'batch_size': int(self.df.loc[worst_latency_idx, 'batch_size']),
            'input_length': int(self.df.loc[worst_latency_idx, 'input_length']),
            'throughput': float(self.df.loc[worst_latency_idx, 'throughput_tokens_per_sec']),
            'latency': float(self.df.loc[worst_latency_idx, 'total_latency'])
        }
        
        return summary
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create comprehensive visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Latency heatmap
        self.plot_latency_heatmap(output_dir)
        
        # 2. Throughput heatmap
        self.plot_throughput_heatmap(output_dir)
        
        # 3. Batch size scaling
        self.plot_batch_size_scaling(output_dir)
        
        # 4. Input length scaling
        self.plot_input_length_scaling(output_dir)
        
        # 5. Prefill vs Decode time breakdown
        self.plot_time_breakdown(output_dir)
        
        # 6. Performance comparison
        self.plot_performance_comparison(output_dir)
        
        # 7. Memory usage analysis
        if 'cpu_memory_mb' in self.df.columns:
            self.plot_memory_usage(output_dir)
        
        # 8. Prefill/Decode vs Batched Tokens analysis
        self.plot_time_vs_batched_tokens(output_dir)
        
        print(f"Visualizations saved to {output_dir}/")
    
    def plot_latency_heatmap(self, output_dir: str):
        """Create latency heatmap by batch size and input length."""
        plt.figure(figsize=(12, 8))
        
        # Aggregate data for heatmap
        pivot_data = self.df.groupby(['batch_size', 'input_length'])['total_latency'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='batch_size', columns='input_length', values='total_latency')
        
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Total Latency (seconds)'})
        plt.title('Total Latency Heatmap\n(Batch Size vs Input Length)', fontsize=14, fontweight='bold')
        plt.xlabel('Input Length (tokens)')
        plt.ylabel('Batch Size')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_throughput_heatmap(self, output_dir: str):
        """Create throughput heatmap by batch size and input length."""
        plt.figure(figsize=(12, 8))
        
        # Aggregate data for heatmap
        pivot_data = self.df.groupby(['batch_size', 'input_length'])['throughput_tokens_per_sec'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='batch_size', columns='input_length', values='throughput_tokens_per_sec')
        
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlGnBu', 
                   cbar_kws={'label': 'Throughput (tokens/second)'})
        plt.title('Throughput Heatmap\n(Batch Size vs Input Length)', fontsize=14, fontweight='bold')
        plt.xlabel('Input Length (tokens)')
        plt.ylabel('Batch Size')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_batch_size_scaling(self, output_dir: str):
        """Plot batch size scaling analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Latency vs batch size
        for input_len in sorted(self.df['input_length'].unique()):
            subset = self.df[self.df['input_length'] == input_len]
            grouped = subset.groupby('batch_size')['total_latency'].agg(['mean', 'std'])
            axes[0, 0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Input Length {input_len}', marker='o')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Total Latency (seconds)')
        axes[0, 0].set_title('Latency vs Batch Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Throughput vs batch size
        for input_len in sorted(self.df['input_length'].unique()):
            subset = self.df[self.df['input_length'] == input_len]
            grouped = subset.groupby('batch_size')['throughput_tokens_per_sec'].agg(['mean', 'std'])
            axes[0, 1].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Input Length {input_len}', marker='o')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Throughput (tokens/second)')
        axes[0, 1].set_title('Throughput vs Batch Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prefill time vs batch size
        prefill_col = 'prefill_time' if 'prefill_time' in self.df.columns else 'estimated_prefill_time'
        prefill_label = 'Prefill Time (seconds)' if 'prefill_time' in self.df.columns else 'Estimated Prefill Time (seconds)'
        
        for input_len in sorted(self.df['input_length'].unique()):
            subset = self.df[self.df['input_length'] == input_len]
            grouped = subset.groupby('batch_size')[prefill_col].agg(['mean', 'std'])
            axes[0, 2].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Input Length {input_len}', marker='o')
        axes[0, 2].set_xlabel('Batch Size')
        axes[0, 2].set_ylabel(prefill_label)
        axes[0, 2].set_title('Prefill Time vs Batch Size')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Per-token decode time vs batch size
        for input_len in sorted(self.df['input_length'].unique()):
            subset = self.df[self.df['input_length'] == input_len]
            grouped = subset.groupby('batch_size')['decode_time_per_token'].agg(['mean', 'std'])
            axes[1, 0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Input Length {input_len}', marker='o')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Decode Time per Token (seconds)')
        axes[1, 0].set_title('Decode Time per Token vs Batch Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Batch efficiency (throughput per batch item)
        for input_len in sorted(self.df['input_length'].unique()):
            subset = self.df[self.df['input_length'] == input_len]
            subset_copy = subset.copy()
            subset_copy['efficiency'] = subset_copy['throughput_tokens_per_sec'] / subset_copy['batch_size']
            grouped = subset_copy.groupby('batch_size')['efficiency'].agg(['mean', 'std'])
            axes[1, 1].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Input Length {input_len}', marker='o')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Throughput per Batch Item (tokens/second)')
        axes[1, 1].set_title('Batch Efficiency vs Batch Size')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Decode time vs batch size
        for input_len in sorted(self.df['input_length'].unique()):
            subset = self.df[self.df['input_length'] == input_len]
            grouped = subset.groupby('batch_size')['decode_time'].agg(['mean', 'std'])
            axes[1, 2].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Input Length {input_len}', marker='o')
        axes[1, 2].set_xlabel('Batch Size')
        axes[1, 2].set_ylabel('Decode Time (seconds)')
        axes[1, 2].set_title('Decode Time vs Batch Size')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/batch_size_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_input_length_scaling(self, output_dir: str):
        """Plot input length scaling analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency vs input length
        for batch_size in sorted(self.df['batch_size'].unique()):
            subset = self.df[self.df['batch_size'] == batch_size]
            grouped = subset.groupby('input_length')['total_latency'].agg(['mean', 'std'])
            axes[0, 0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Batch Size {batch_size}', marker='o')
        axes[0, 0].set_xlabel('Input Length (tokens)')
        axes[0, 0].set_ylabel('Total Latency (seconds)')
        axes[0, 0].set_title('Latency vs Input Length')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Prefill time vs input length
        prefill_col = 'prefill_time' if 'prefill_time' in self.df.columns else 'estimated_prefill_time'
        prefill_label = 'Prefill Time (seconds)' if 'prefill_time' in self.df.columns else 'Estimated Prefill Time (seconds)'
        
        for batch_size in sorted(self.df['batch_size'].unique()):
            subset = self.df[self.df['batch_size'] == batch_size]
            grouped = subset.groupby('input_length')[prefill_col].agg(['mean', 'std'])
            axes[0, 1].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Batch Size {batch_size}', marker='o')
        axes[0, 1].set_xlabel('Input Length (tokens)')
        axes[0, 1].set_ylabel(prefill_label)
        axes[0, 1].set_title('Prefill Time vs Input Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput vs input length
        for batch_size in sorted(self.df['batch_size'].unique()):
            subset = self.df[self.df['batch_size'] == batch_size]
            grouped = subset.groupby('input_length')['throughput_tokens_per_sec'].agg(['mean', 'std'])
            axes[1, 0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Batch Size {batch_size}', marker='o')
        axes[1, 0].set_xlabel('Input Length (tokens)')
        axes[1, 0].set_ylabel('Throughput (tokens/second)')
        axes[1, 0].set_title('Throughput vs Input Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Decode time per token vs input length
        for batch_size in sorted(self.df['batch_size'].unique()):
            subset = self.df[self.df['batch_size'] == batch_size]
            grouped = subset.groupby('input_length')['decode_time_per_token'].agg(['mean', 'std'])
            axes[1, 1].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                              label=f'Batch Size {batch_size}', marker='o')
        axes[1, 1].set_xlabel('Input Length (tokens)')
        axes[1, 1].set_ylabel('Decode Time per Token (seconds)')
        axes[1, 1].set_title('Decode Time per Token vs Input Length')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/input_length_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_breakdown(self, output_dir: str):
        """Plot prefill vs decode time breakdown."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stacked bar chart of time breakdown
        prefill_col = 'prefill_time' if 'prefill_time' in self.df.columns else 'estimated_prefill_time'
        grouped = self.df.groupby(['batch_size', 'input_length']).agg({
            prefill_col: 'mean',
            'decode_time': 'mean'
        })
        
        configs = [f"B{bs}_I{il}" for bs, il in grouped.index]
        prefill_times = grouped[prefill_col].values
        decode_times = grouped['decode_time'].values
        
        x_pos = np.arange(len(configs))
        
        axes[0].bar(x_pos, prefill_times, label='Prefill Time', alpha=0.8)
        axes[0].bar(x_pos, decode_times, bottom=prefill_times, label='Decode Time', alpha=0.8)
        axes[0].set_xlabel('Configuration (Batch Size_Input Length)')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Time Breakdown by Configuration')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(configs, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Prefill ratio analysis
        prefill_ratios = self.df.groupby(['batch_size', 'input_length'])['prefill_ratio'].mean()
        
        axes[1].bar(x_pos, prefill_ratios.values, alpha=0.8, color='orange')
        axes[1].set_xlabel('Configuration (Batch Size_Input Length)')
        axes[1].set_ylabel('Prefill Time Ratio')
        axes[1].set_title('Prefill Time as Fraction of Total Latency')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(configs, rotation=45)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self, output_dir: str):
        """Create comprehensive performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot of latency distribution
        self.df.boxplot(column='total_latency', by='batch_size', ax=axes[0, 0])
        axes[0, 0].set_title('Latency Distribution by Batch Size')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Total Latency (seconds)')
        
        # Box plot of throughput distribution
        self.df.boxplot(column='throughput_tokens_per_sec', by='input_length', ax=axes[0, 1])
        axes[0, 1].set_title('Throughput Distribution by Input Length')
        axes[0, 1].set_xlabel('Input Length (tokens)')
        axes[0, 1].set_ylabel('Throughput (tokens/second)')
        
        # Scatter plot: Latency vs Throughput
        scatter = axes[1, 0].scatter(self.df['total_latency'], self.df['throughput_tokens_per_sec'], 
                                   c=self.df['batch_size'], cmap='viridis', alpha=0.7)
        axes[1, 0].set_xlabel('Total Latency (seconds)')
        axes[1, 0].set_ylabel('Throughput (tokens/second)')
        axes[1, 0].set_title('Latency vs Throughput (colored by Batch Size)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Batch Size')
        
        # Performance efficiency plot
        self.df['efficiency_score'] = self.df['throughput_tokens_per_sec'] / (self.df['total_latency'] + 1e-6)
        efficiency_by_config = self.df.groupby(['batch_size', 'input_length'])['efficiency_score'].mean()
        
        configs = [f"B{bs}_I{il}" for bs, il in efficiency_by_config.index]
        axes[1, 1].bar(range(len(configs)), efficiency_by_config.values, alpha=0.8)
        axes[1, 1].set_xlabel('Configuration (Batch Size_Input Length)')
        axes[1, 1].set_ylabel('Efficiency Score (Throughput/Latency)')
        axes[1, 1].set_title('Performance Efficiency by Configuration')
        axes[1, 1].set_xticks(range(len(configs)))
        axes[1, 1].set_xticklabels(configs, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_usage(self, output_dir: str):
        """Plot memory usage analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # CPU Memory usage
        if 'cpu_memory_mb' in self.df.columns:
            memory_by_config = self.df.groupby(['batch_size', 'input_length'])['cpu_memory_mb'].mean()
            configs = [f"B{bs}_I{il}" for bs, il in memory_by_config.index]
            
            axes[0].bar(range(len(configs)), memory_by_config.values, alpha=0.8, color='skyblue')
            axes[0].set_xlabel('Configuration (Batch Size_Input Length)')
            axes[0].set_ylabel('CPU Memory Usage (MB)')
            axes[0].set_title('CPU Memory Usage by Configuration')
            axes[0].set_xticks(range(len(configs)))
            axes[0].set_xticklabels(configs, rotation=45)
            axes[0].grid(True, alpha=0.3)
        
        # GPU Memory usage
        if 'gpu_memory_allocated_mb' in self.df.columns:
            gpu_memory_by_config = self.df.groupby(['batch_size', 'input_length'])['gpu_memory_allocated_mb'].mean()
            configs = [f"B{bs}_I{il}" for bs, il in gpu_memory_by_config.index]
            
            axes[1].bar(range(len(configs)), gpu_memory_by_config.values, alpha=0.8, color='lightcoral')
            axes[1].set_xlabel('Configuration (Batch Size_Input Length)')
            axes[1].set_ylabel('GPU Memory Usage (MB)')
            axes[1].set_title('GPU Memory Usage by Configuration')
            axes[1].set_xticks(range(len(configs)))
            axes[1].set_xticklabels(configs, rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_vs_batched_tokens(self, output_dir: str):
        """Plot prefill and decode time vs batched tokens (batch_size * input_length)."""
        # Calculate batched tokens
        self.df['batched_tokens'] = self.df['batch_size'] * self.df['input_length']
        
        # Use appropriate prefill time column
        prefill_col = 'prefill_time' if 'prefill_time' in self.df.columns else 'estimated_prefill_time'
        prefill_label = 'Prefill Time (seconds)' if 'prefill_time' in self.df.columns else 'Estimated Prefill Time (seconds)'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prefill Time vs Batched Tokens
        scatter = axes[0, 0].scatter(self.df['batched_tokens'], self.df[prefill_col], 
                                   c=self.df['batch_size'], cmap='viridis', alpha=0.7, s=60)
        axes[0, 0].set_xlabel('Batched Tokens (batch_size × input_length)')
        axes[0, 0].set_ylabel(prefill_label)
        axes[0, 0].set_title('Prefill Time vs Batched Tokens')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Batch Size')
        
        # Add trend line for prefill time
        z = np.polyfit(self.df['batched_tokens'], self.df[prefill_col], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(sorted(self.df['batched_tokens']), p(sorted(self.df['batched_tokens'])), 
                       "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2e})')
        axes[0, 0].legend()
        
        # 2. Decode Time vs Batched Tokens
        scatter = axes[0, 1].scatter(self.df['batched_tokens'], self.df['decode_time'], 
                                   c=self.df['batch_size'], cmap='plasma', alpha=0.7, s=60)
        axes[0, 1].set_xlabel('Batched Tokens (batch_size × input_length)')
        axes[0, 1].set_ylabel('Decode Time (seconds)')
        axes[0, 1].set_title('Decode Time vs Batched Tokens')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Batch Size')
        
        # Add trend line for decode time
        z = np.polyfit(self.df['batched_tokens'], self.df['decode_time'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(sorted(self.df['batched_tokens']), p(sorted(self.df['batched_tokens'])), 
                       "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2e})')
        axes[0, 1].legend()
        
        # 3. Combined view: Both times vs Batched Tokens
        # Group by batched_tokens to show mean and std
        grouped_data = self.df.groupby('batched_tokens').agg({
            prefill_col: ['mean', 'std'],
            'decode_time': ['mean', 'std'],
            'batch_size': 'first'  # For consistent coloring
        }).reset_index()
        
        x = grouped_data['batched_tokens']
        prefill_mean = grouped_data[(prefill_col, 'mean')]
        prefill_std = grouped_data[(prefill_col, 'std')]
        decode_mean = grouped_data[('decode_time', 'mean')]
        decode_std = grouped_data[('decode_time', 'std')]
        
        axes[1, 0].errorbar(x, prefill_mean, yerr=prefill_std, 
                          label='Prefill Time', marker='o', capsize=5, alpha=0.8)
        axes[1, 0].errorbar(x, decode_mean, yerr=decode_std, 
                          label='Decode Time', marker='s', capsize=5, alpha=0.8)
        axes[1, 0].set_xlabel('Batched Tokens (batch_size × input_length)')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Prefill vs Decode Time by Batched Tokens')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Time ratio vs Batched Tokens
        scatter = axes[1, 1].scatter(self.df['batched_tokens'], self.df['prefill_ratio'], 
                                   c=self.df['input_length'], cmap='coolwarm', alpha=0.7, s=60)
        axes[1, 1].set_xlabel('Batched Tokens (batch_size × input_length)')
        axes[1, 1].set_ylabel('Prefill Time Ratio')
        axes[1, 1].set_title('Prefill Time Ratio vs Batched Tokens')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        plt.colorbar(scatter, ax=axes[1, 1], label='Input Length')
        
        # Add trend line for prefill ratio
        z = np.polyfit(self.df['batched_tokens'], self.df['prefill_ratio'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(sorted(self.df['batched_tokens']), p(sorted(self.df['batched_tokens'])), 
                       "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2e})')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_vs_batched_tokens.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive analysis report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_analysis_report_{timestamp}.md"
        
        summary = self.generate_summary_statistics()
        
        report = f"""# Transformer Benchmark Analysis Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Results file: {self.results_file}

## Executive Summary

- **Total Experiments**: {summary['overall']['total_experiments']}
- **Unique Configurations**: {summary['overall']['unique_configurations']}
- **Models Tested**: {', '.join(summary['overall']['models_tested'])}
- **Total Runtime**: {summary['overall']['total_runtime']:.2f} seconds
- **Average Latency**: {summary['overall']['average_latency']:.4f} seconds
- **Average Throughput**: {summary['overall']['average_throughput']:.2f} tokens/second

## Best Performance Configuration

**Highest Throughput**:
- Batch Size: {summary['best_throughput']['batch_size']}
- Input Length: {summary['best_throughput']['input_length']} tokens
- Throughput: {summary['best_throughput']['throughput']:.2f} tokens/second
- Latency: {summary['best_throughput']['latency']:.4f} seconds

**Worst Latency Configuration**:
- Batch Size: {summary['worst_latency']['batch_size']}
- Input Length: {summary['worst_latency']['input_length']} tokens  
- Throughput: {summary['worst_latency']['throughput']:.2f} tokens/second
- Latency: {summary['worst_latency']['latency']:.4f} seconds

## Detailed Performance Analysis

### Batch Size Impact
"""
        
        # Add batch size analysis
        for batch_size in sorted(self.df['batch_size'].unique()):
            subset = self.df[self.df['batch_size'] == batch_size]
            avg_latency = subset['total_latency'].mean()
            avg_throughput = subset['throughput_tokens_per_sec'].mean()
            report += f"\n- **Batch Size {batch_size}**: Avg Latency = {avg_latency:.4f}s, Avg Throughput = {avg_throughput:.2f} tokens/s"
        
        report += "\n\n### Input Length Impact\n"
        
        # Add input length analysis
        for input_length in sorted(self.df['input_length'].unique()):
            subset = self.df[self.df['input_length'] == input_length]
            avg_latency = subset['total_latency'].mean()
            avg_throughput = subset['throughput_tokens_per_sec'].mean()
            report += f"\n- **Input Length {input_length}**: Avg Latency = {avg_latency:.4f}s, Avg Throughput = {avg_throughput:.2f} tokens/s"
        
        # Add configuration recommendations
        report += "\n\n## Recommendations\n"
        
        best_config = self.df.loc[self.df['throughput_tokens_per_sec'].idxmax()]
        report += f"\n- **For Maximum Throughput**: Use batch size {int(best_config['batch_size'])} with input length {int(best_config['input_length'])} tokens"
        
        fastest_config = self.df.loc[self.df['total_latency'].idxmin()]
        report += f"\n- **For Minimum Latency**: Use batch size {int(fastest_config['batch_size'])} with input length {int(fastest_config['input_length'])} tokens"
        
        # Scaling observations
        report += "\n\n## Scaling Observations\n"
        
        # Batch size scaling
        batch_correlation = self.df[['batch_size', 'total_latency']].corr().iloc[0, 1]
        report += f"\n- **Batch Size vs Latency Correlation**: {batch_correlation:.3f}"
        
        # Input length scaling  
        input_correlation = self.df[['input_length', 'total_latency']].corr().iloc[0, 1]
        report += f"\n- **Input Length vs Latency Correlation**: {input_correlation:.3f}"
        
        report += "\n\n## Files Generated\n"
        report += "\n- `analysis_plots/`: Directory containing all visualization plots"
        report += "\n- `latency_heatmap.png`: Latency performance heatmap"
        report += "\n- `throughput_heatmap.png`: Throughput performance heatmap" 
        report += "\n- `batch_size_scaling.png`: Batch size scaling analysis"
        report += "\n- `input_length_scaling.png`: Input length scaling analysis"
        report += "\n- `time_breakdown.png`: Prefill vs decode time breakdown"
        report += "\n- `performance_comparison.png`: Comprehensive performance comparison"
        report += "\n- `time_vs_batched_tokens.png`: Prefill and decode time vs batched tokens analysis"
        if 'cpu_memory_mb' in self.df.columns or 'gpu_memory_allocated_mb' in self.df.columns:
            report += "\n- `memory_usage.png`: Memory usage analysis"
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Analysis report saved to {output_file}")
        return output_file


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Analyze transformer benchmark results")
    parser.add_argument("results_file", type=str, help="Path to benchmark results file (.json or .csv)")
    parser.add_argument("--output-dir", type=str, default="analysis_output", 
                       help="Directory to save analysis outputs")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    print(f"Loading and analyzing results from {args.results_file}...")
    analyzer = BenchmarkAnalyzer(args.results_file)
    
    # Generate visualizations
    if not args.no_plots and not args.report_only:
        plot_dir = os.path.join(args.output_dir, "plots")
        analyzer.create_visualizations(plot_dir)
    
    # Generate report
    report_file = os.path.join(args.output_dir, "analysis_report.md")
    analyzer.generate_report(report_file)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    summary = analyzer.generate_summary_statistics()
    print(f"Analyzed {summary['overall']['total_experiments']} experiments")
    print(f"Best throughput: {summary['best_throughput']['throughput']:.2f} tokens/s")
    print(f"Best latency: {analyzer.df['total_latency'].min():.4f}s")
    print(f"Output files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()