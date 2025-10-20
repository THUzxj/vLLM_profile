#!/usr/bin/env python3
"""
Configuration module for transformer benchmarking.
Defines benchmark parameters and settings.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class BenchmarkConfig:
    """Configuration class for benchmark parameters."""
    
    # Model generation parameters
    output_length: int = 50  # Fixed output length
    temperature: float = 1.0
    top_p: float = 0.9
    
    # Benchmark parameters
    batch_sizes: List[int] = None
    input_lengths: List[int] = None
    num_runs: int = 3  # Number of runs per configuration for averaging
    
    # System parameters
    device: str = "auto"
    max_batch_input_product: int = 65536 * 2  # Skip experiments if batch_size * input_length > this value
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16]
        
        if self.input_lengths is None:
            self.input_lengths = [32, 64, 128, 256, 512, 1024]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'BenchmarkConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        """Create configuration from dictionary."""
        return cls(**data)


class PresetConfigs:
    """Preset configurations for common benchmarking scenarios."""
    
    @staticmethod
    def small_scale() -> BenchmarkConfig:
        """Configuration for small-scale testing."""
        return BenchmarkConfig(
            batch_sizes=[1, 2],
            input_lengths=[32, 64, 128],
            output_length=30,
            num_runs=2
        )
    
    @staticmethod
    def medium_scale() -> BenchmarkConfig:
        """Configuration for medium-scale testing."""
        return BenchmarkConfig(
            batch_sizes=[1, 2, 4, 8, 16, 32],
            input_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            output_length=50,
            num_runs=3
        )
    
    @staticmethod
    def large_scale() -> BenchmarkConfig:
        """Configuration for large-scale testing."""
        return BenchmarkConfig(
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
            input_lengths=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
            output_length=100,
            num_runs=5
        )
    
    @staticmethod
    def batch_size_study() -> BenchmarkConfig:
        """Configuration focused on batch size scaling."""
        return BenchmarkConfig(
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
            input_lengths=[256],  # Fixed input length
            output_length=50,
            num_runs=5
        )
    
    @staticmethod
    def input_length_study() -> BenchmarkConfig:
        """Configuration focused on input length scaling."""
        return BenchmarkConfig(
            batch_sizes=[1],  # Fixed batch size
            input_lengths=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
            output_length=50,
            num_runs=5
        )
    
    @staticmethod
    def quick_test() -> BenchmarkConfig:
        """Configuration for quick testing."""
        return BenchmarkConfig(
            batch_sizes=[1, 2],
            input_lengths=[32, 64],
            output_length=20,
            num_runs=1
        )


def create_config_file(preset_name: str = "medium_scale", filepath: str = "benchmark_config.json"):
    """
    Create a configuration file with a preset configuration.
    
    Args:
        preset_name: Name of the preset configuration
        filepath: Path to save the configuration file
    """
    preset_configs = {
        "small_scale": PresetConfigs.small_scale,
        "medium_scale": PresetConfigs.medium_scale,
        "large_scale": PresetConfigs.large_scale,
        "batch_size_study": PresetConfigs.batch_size_study,
        "input_length_study": PresetConfigs.input_length_study,
        "quick_test": PresetConfigs.quick_test,
    }
    
    if preset_name not in preset_configs:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(preset_configs.keys())}")
    
    config = preset_configs[preset_name]()
    config.save_to_file(filepath)
    print(f"Configuration saved to {filepath}")
    return config


if __name__ == "__main__":
    """Command line interface for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Configuration Manager")
    parser.add_argument("--create", type=str, choices=[
        "small_scale", "medium_scale", "large_scale", 
        "batch_size_study", "input_length_study", "quick_test"
    ], help="Create a preset configuration file")
    parser.add_argument("--output", type=str, default="benchmark_config.json",
                       help="Output configuration file path")
    parser.add_argument("--show-presets", action="store_true",
                       help="Show available preset configurations")
    
    args = parser.parse_args()
    
    if args.show_presets:
        print("Available preset configurations:")
        print("\n1. small_scale: Small-scale testing")
        config = PresetConfigs.small_scale()
        print(f"   - Batch sizes: {config.batch_sizes}")
        print(f"   - Input lengths: {config.input_lengths}")
        print(f"   - Output length: {config.output_length}")
        print(f"   - Runs: {config.num_runs}")
        
        print("\n2. medium_scale: Medium-scale testing")
        config = PresetConfigs.medium_scale()
        print(f"   - Batch sizes: {config.batch_sizes}")
        print(f"   - Input lengths: {config.input_lengths}")
        print(f"   - Output length: {config.output_length}")
        print(f"   - Runs: {config.num_runs}")
        
        print("\n3. large_scale: Large-scale testing")
        config = PresetConfigs.large_scale()
        print(f"   - Batch sizes: {config.batch_sizes}")
        print(f"   - Input lengths: {config.input_lengths}")
        print(f"   - Output length: {config.output_length}")
        print(f"   - Runs: {config.num_runs}")
        
        print("\n4. batch_size_study: Focus on batch size scaling")
        config = PresetConfigs.batch_size_study()
        print(f"   - Batch sizes: {config.batch_sizes}")
        print(f"   - Input lengths: {config.input_lengths}")
        print(f"   - Output length: {config.output_length}")
        print(f"   - Runs: {config.num_runs}")
        
        print("\n5. input_length_study: Focus on input length scaling")
        config = PresetConfigs.input_length_study()
        print(f"   - Batch sizes: {config.batch_sizes}")
        print(f"   - Input lengths: {config.input_lengths}")
        print(f"   - Output length: {config.output_length}")
        print(f"   - Runs: {config.num_runs}")
        
        print("\n6. quick_test: Quick testing")
        config = PresetConfigs.quick_test()
        print(f"   - Batch sizes: {config.batch_sizes}")
        print(f"   - Input lengths: {config.input_lengths}")
        print(f"   - Output length: {config.output_length}")
        print(f"   - Runs: {config.num_runs}")
    
    elif args.create:
        create_config_file(args.create, args.output)
    
    else:
        parser.print_help()