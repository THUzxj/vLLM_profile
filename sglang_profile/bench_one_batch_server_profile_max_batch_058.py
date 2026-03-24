"""
Benchmark the latency of running a single batch with a server using metrics-triggered profiling.

This script launches a server and uses the HTTP interface.
It monitors running requests via /metrics endpoint and triggers profiling when
the number of running requests reaches a configurable threshold.

Usage:
# Basic profiling with metrics trigger
python3 bench_one_batch_server_profile_max_batch_058.py --model meta-llama/Meta-Llama-3.1-8B --batch-size 256 --input-len 1024 --output-len 8 --profile

# With external server
python3 bench_one_batch_server_profile_max_batch_058.py --model None --base-url http://localhost:30000 --batch-size 256 --input-len 1024 --output-len 8 --profile

# With custom trigger threshold and continuous sending
python3 bench_one_batch_server_profile_max_batch_058.py --model None --base-url http://localhost:30000 --batch-size 256 --input-len 1024 --output-len 8 --profile --profile-trigger-threshold 0.9 --send-interval 0.5 --total-rounds 0

# PD disaggregation mode
python3 bench_one_batch_server_profile_max_batch_058.py --model None --base-url http://prefill:30000 --decode-url http://decode:30001 --batch-size 256 --profile
"""

import argparse

from sglang.srt.server_args import ServerArgs
from bench_one_batch_server_internal_profile_max_batch_058 import (
    BenchArgs,
    run_benchmark_internal,
)
from sglang.test.nightly_bench_utils import save_results_as_pydantic_models


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    results, server_info = run_benchmark_internal(server_args, bench_args)

    # Save results as pydantic models in the JSON format
    if bench_args.pydantic_result_filename:
        save_results_as_pydantic_models(
            results,
            pydantic_result_filename=bench_args.pydantic_result_filename,
            model_path=server_args.model_path,
        )

    return results, server_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)