import argparse
import csv
import dataclasses
import itertools
import json
import multiprocessing
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
from pydantic import BaseModel
from tabulate import tabulate
from transformers import AutoProcessor, PreTrainedTokenizer

from sglang.bench_serving import (
    get_processor,
    get_tokenizer,
    sample_mmmu_requests,
    sample_random_requests,
)
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_blackwell, kill_process_tree
from sglang.test.test_utils import is_in_ci, write_github_step_summary

DEFAULT_TIMEOUT = 600
DEFAULT_PROFILE_STAGES = ["decode"]


def run_profile_with_stages(
    url: str,
    num_steps: int,
    activities: List[str],
    output_dir: Optional[str] = None,
    profile_by_stage: bool = False,
    merge_profiles: bool = False,
    profile_prefix: Optional[str] = None,
    profile_stages: Optional[List[str]] = None,
) -> str:
    if output_dir is None:
        output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")

    output_dir = Path(os.path.abspath(os.path.normpath(output_dir))) / str(time.time())
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Dump profiling traces to {output_dir}")
    print(
        f"Waiting for {num_steps} steps and the trace to be flushed.... ({profile_by_stage=}, {profile_stages=})"
    )

    file_path = Path(output_dir) / "server_args.json"
    if not file_path.exists():
        response = requests.get(url + "/get_server_info")
        response.raise_for_status()
        server_args_data = response.json()
        with open(file_path, "w") as file:
            file.write(json.dumps(server_args_data))

    json_data = {
        "output_dir": str(output_dir),
        "num_steps": str(num_steps),
        "activities": activities,
        "profile_by_stage": profile_by_stage,
        "merge_profiles": merge_profiles,
        "profile_prefix": profile_prefix,
    }
    if profile_stages is not None:
        json_data["profile_stages"] = profile_stages

    response = requests.post(url=url + "/start_profile", json=json_data)
    response.raise_for_status()
    return str(output_dir)


def get_cache_tokens_from_metrics(url: str) -> Optional[tuple]:
    """
    Get cached_tokens_total and prompt_tokens_total from Prometheus /metrics endpoint.
    Returns (cached_tokens_total, prompt_tokens_total) or None if metrics are not available.
    """
    try:
        response = requests.get(url + "/metrics", timeout=5)
        response.raise_for_status()

        # Parse Prometheus text format
        # Looking for: sglang:cached_tokens_total{...} <value>
        #              sglang:prompt_tokens_total{...} <value>
        cached_tokens_total = 0.0
        prompt_tokens_total = 0.0

        for line in response.text.split("\n"):
            if line.startswith("sglang:cached_tokens_total{"):
                match = re.search(
                    r"sglang:cached_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    cached_tokens_total += float(match.group(1))
            elif line.startswith("sglang:prompt_tokens_total{"):
                match = re.search(
                    r"sglang:prompt_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    prompt_tokens_total += float(match.group(1))

        return (cached_tokens_total, prompt_tokens_total)
    except Exception as e:
        print(f"Warning: Failed to get cache tokens from metrics: {e}")
        return None


def calculate_cache_hit_rate(
    before: Optional[tuple], after: Optional[tuple]
) -> Optional[float]:
    """
    Calculate cache hit rate from before/after metrics snapshots.
    Returns cached_tokens_delta / prompt_tokens_delta for the benchmark run.
    """
    if before is None or after is None:
        return None

    cached_delta = after[0] - before[0]
    prompt_delta = after[1] - before[1]

    if prompt_delta > 0:
        return cached_delta / prompt_delta
    return None


def _save_tbt_artifacts(
    tbt_samples: List[float],
    result_filename: str,
    run_name: str,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> tuple[Path, Optional[Path]]:
    if result_filename:
        artifact_dir = Path(result_filename).resolve().parent
    else:
        artifact_dir = Path.cwd() / "tbt_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name or "default")
    ts_ms = int(time.time() * 1000)
    base_name = (
        f"{safe_run_name}_bs{batch_size}_il{input_len}_ol{output_len}_{ts_ms}"
    )
    csv_path = artifact_dir / f"{base_name}_tbt.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_index", "tbt_s", "tbt_ms"])
        for i, tbt in enumerate(tbt_samples, start=1):
            writer.writerow([i, f"{tbt:.9f}", f"{tbt * 1000.0:.6f}"])

    plot_path: Optional[Path] = None
    # Keep full samples in CSV, but ignore the first point in plotting.
    tbt_samples_for_plot = tbt_samples[1:] if len(tbt_samples) > 1 else []
    if tbt_samples_for_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x = np.arange(2, len(tbt_samples) + 1)
            y_ms = np.asarray(tbt_samples_for_plot) * 1000.0

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, y_ms, color="#1f77b4", linewidth=1.2)
            ax.set_title(
                f"TBT Fluctuation (ignore first point, run={safe_run_name}, bs={batch_size}, il={input_len}, ol={output_len})"
            )
            ax.set_xlabel("Token Index")
            ax.set_ylabel("TBT (ms)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            plot_path = artifact_dir / f"{base_name}_tbt.png"
            fig.savefig(plot_path, dpi=160)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: failed to draw TBT plot: {e}")

    return csv_path, plot_path


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_steps: int = 5
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    profile_output_dir: Optional[str] = None
    dataset_path: str = ""
    dataset_name: str = "random"
    parallel_batch: bool = False
    result_filename: str = "result.jsonl"
    pydantic_result_filename: Optional[str] = None
    append_to_github_summary: bool = True
    seed: int = 42
    cache_hit_rate: float = 0.0
    cached_token_len: Optional[int] = None
    measure: bool = False
    measure_tbt: bool = False
    profile_activities: Optional[List[str]] = None
    use_nsys: bool = False
    profile_stages: Optional[List[str]] = None
    merge_profiles: bool = False
    decode_url: Optional[str] = None
    prefill_url: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--client-stream-interval",
            type=int,
            default=BenchArgs.client_stream_interval,
        )
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix",
            type=str,
            default=BenchArgs.profile_prefix,
        )
        parser.add_argument(
            "--profile-output-dir",
            type=str,
            default=BenchArgs.profile_output_dir,
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=["mmmu", "random", "dummy"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument("--parallel-batch", action="store_true")
        parser.add_argument(
            "--result-filename",
            type=str,
            default=BenchArgs.result_filename,
            help="Store the results line by line in the JSON Line format to this file.",
        )
        parser.add_argument(
            "--pydantic-result-filename",
            type=str,
            default=BenchArgs.pydantic_result_filename,
            help="Store the results as pydantic models in the JSON format to this file.",
        )
        parser.add_argument(
            "--no-append-to-github-summary",
            action="store_false",
            dest="append_to_github_summary",
            help="Disable appending the output of this run to github ci summary",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
        parser.add_argument(
            "--cache-hit-rate",
            type=float,
            default=BenchArgs.cache_hit_rate,
            help="Cache hit rate for benchmarking (0.0-1.0). "
            "0.0 means no cache hits (flush all), 0.4 means 40%% of input tokens are cached.",
        )
        parser.add_argument(
            "--cached-token-len",
            type=int,
            default=BenchArgs.cached_token_len,
            help="Number of tokens to cache. If specified, overrides cache-hit-rate calculation.",
        )
        parser.add_argument(
            "--measure",
            action="store_true",
            default=BenchArgs.measure,
            help="If set, run the benchmark measurement loop (batch_size x input_len x output_len).",
        )
        parser.add_argument(
            "--measure-tbt",
            action="store_true",
            default=BenchArgs.measure_tbt,
            help="If set, directly measure per-token time-between-tokens (TBT) "
            "from streaming chunks and report mean/min/max/std.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=BenchArgs.profile_activities,
            help="Profile activities to enable. Choices: CPU, GPU, CUDA_PROFILER, MEM, RPD. "
            "Default: ['CPU', 'GPU'] if not specified. If --use-nsys is set, this will be overridden to ['CUDA_PROFILER'] unless explicitly specified.",
        )
        parser.add_argument(
            "--use-nsys",
            action="store_true",
            default=BenchArgs.use_nsys,
            help="Use nsys profiling (CUDA_PROFILER). If set, profile_activities will be automatically set to ['CUDA_PROFILER'] unless --profile-activities is explicitly specified.",
        )
        parser.add_argument(
            "--profile-stages",
            type=str,
            nargs="+",
            default=BenchArgs.profile_stages,
            help="Profile stages for /start_profile, e.g. decode prefill. Default: decode only.",
        )
        parser.add_argument(
            "--merge-profiles",
            action="store_true",
            default=BenchArgs.merge_profiles,
            help="Merge profile traces from all ranks (TP/DP/PP/EP) into a single trace file.",
        )
        parser.add_argument(
            "--decode-url",
            type=str,
            default=BenchArgs.decode_url,
            help="URL for profiling decode workers in PD-separated mode. If not specified, uses --base-url for profiling.",
        )
        parser.add_argument(
            "--prefill-url",
            type=str,
            default=BenchArgs.prefill_url,
            help="URL for getting server info in PD-separated mode. If not specified, uses --base-url.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        instance = cls(**{attr: getattr(args, attr) for attr in attrs})
        
        # If use_nsys is True and profile_activities is not explicitly set, use CUDA_PROFILER
        if instance.use_nsys and instance.profile_activities is None:
            instance.profile_activities = ["CUDA_PROFILER"]

        # Default to decode-only profiling unless explicitly provided.
        if instance.profile_stages is None:
            instance.profile_stages = DEFAULT_PROFILE_STAGES.copy()
        
        return instance


class BenchOneCaseResult(BaseModel):
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_ttft: float
    tpot: float
    decode_latency: float
    last_gen_throughput: float
    acc_length: float
    cache_hit_rate: Optional[float] = None
    tbt_mean: Optional[float] = None
    tbt_median: Optional[float] = None
    tbt_min: Optional[float] = None
    tbt_max: Optional[float] = None
    tbt_std: Optional[float] = None
    tbt_est_total_time: Optional[float] = None
    profile_link: Optional[str] = None

    def dump_to_jsonl(self, result_filename: str):
        with open(result_filename, "a") as fout:
            res = {
                "run_name": self.run_name,
                "batch_size": self.batch_size,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "latency": round(self.latency, 4),
                "input_throughput": round(self.input_throughput, 2),
                "output_throughput": round(self.output_throughput, 2),
                "overall_throughput": round(self.overall_throughput, 2),
                "last_ttft": round(self.last_ttft, 4),
                "tpot": round(self.tpot, 4),
                "decode_latency": round(self.decode_latency, 4),
                "last_gen_throughput": round(self.last_gen_throughput, 2),
                "acc_length": round(self.acc_length, 2),
                "cache_hit_rate": (
                    round(self.cache_hit_rate, 4)
                    if self.cache_hit_rate is not None
                    else None
                ),
                "tbt_mean": round(self.tbt_mean, 6) if self.tbt_mean is not None else None,
                "tbt_median": round(self.tbt_median, 6) if self.tbt_median is not None else None,
                "tbt_min": round(self.tbt_min, 6) if self.tbt_min is not None else None,
                "tbt_max": round(self.tbt_max, 6) if self.tbt_max is not None else None,
                "tbt_std": round(self.tbt_std, 6) if self.tbt_std is not None else None,
                "tbt_est_total_time": (
                    round(self.tbt_est_total_time, 6)
                    if self.tbt_est_total_time is not None
                    else None
                ),
            }
            fout.write(json.dumps(res) + "\n")


def launch_server_internal(launch_server_func: Callable, server_args: ServerArgs):
    try:
        launch_server_func(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process(launch_server_func: Callable, server_args: ServerArgs):
    proc = multiprocessing.Process(
        target=launch_server_internal,
        args=(
            launch_server_func,
            server_args,
        ),
    )
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"

    start_time = time.time()
    while time.time() - start_time < DEFAULT_TIMEOUT:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(
                f"{base_url}/v1/models", headers=headers, timeout=DEFAULT_TIMEOUT
            )
            if response.status_code == 200:
                return proc, base_url
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


def _warmup_cache(
    url: str,
    input_ids: List[List[int]],
    input_len: int,
    cache_hit_rate: float,
    dataset_name: str = "random",
    image_data: Optional[List[str]] = None,
    cached_token_len: Optional[int] = None,
):
    """Warm up the cache by sending prefix tokens to populate the radix cache.

    Args:
        url: Server URL
        input_ids: List of input token id lists
        input_len: Length of input tokens
        cache_hit_rate: Fraction of input tokens to cache (0.0-1.0)
        dataset_name: Name of the dataset (used to determine if image data should be included)
        image_data: Optional image data for VLM models
        cached_token_len: Optional explicit number of tokens to cache (overrides cache_hit_rate)
    """
    if cached_token_len is None:
        cached_token_len = int(input_len * cache_hit_rate)

    if cached_token_len <= 0:
        return

    print(
        f"Warming up cache with {cache_hit_rate*100:.1f}% hit rate "
        f"({cached_token_len} tokens per request)"
    )
    # Create prefix input_ids for cache warming
    cache_warmup_input_ids = [ids[:cached_token_len] for ids in input_ids]
    cache_warmup_payload = {
        "input_ids": cache_warmup_input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,  # Minimal output, just to populate cache
            "ignore_eos": True,
        },
        "stream": False,
        "log_metrics": False,
    }
    if dataset_name == "mmmu" and image_data is not None:
        # include image data in cache warmup
        cache_warmup_payload["image_data"] = image_data

    warmup_response = requests.post(
        url + "/generate",
        json=cache_warmup_payload,
        timeout=DEFAULT_TIMEOUT,
    )
    warmup_response.raise_for_status()
    print("Cache warmup completed")


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    stream_interval: int,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer: PreTrainedTokenizer | AutoProcessor,
    profile: bool = False,
    profile_steps: int = BenchArgs.profile_steps,
    profile_by_stage: bool = False,
    profile_prefix: Optional[str] = BenchArgs.profile_prefix,
    profile_output_dir: Optional[str] = BenchArgs.profile_output_dir,
    dataset_name: str = BenchArgs.dataset_name,
    dataset_path: str = BenchArgs.dataset_path,
    parallel_batch: bool = False,
    cache_hit_rate: float = BenchArgs.cache_hit_rate,
    cached_token_len: Optional[int] = BenchArgs.cached_token_len,
    profile_activities: Optional[List[str]] = None,
    profile_stages: Optional[List[str]] = None,
    merge_profiles: bool = False,
    decode_url: Optional[str] = None,
    measure_tbt: bool = False,
    log_metrics: bool = True,
):
    response = requests.post(url + "/flush_cache", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    effective_decode_url = decode_url if decode_url else url

    # Load input token ids
    # TODO: reuse bench_serving.get_dataset ?
    if dataset_name == "mmmu":
        input_requests = sample_mmmu_requests(
            num_requests=batch_size,
            processor=tokenizer,
            fixed_output_len=output_len,
            random_sample=False,
        )
    elif dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=input_len,
            output_len=output_len,
            num_prompts=batch_size,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            random_sample=True,
            return_text=False,
        )
    elif dataset_name == "dummy":
        input_requests = sample_random_requests(
            input_len=input_len,
            output_len=output_len,
            num_prompts=batch_size,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            random_sample=False,
            return_text=False,
        )

    # Load sampling parameters
    use_structured_outputs = False
    if use_structured_outputs:
        texts = []
        for _ in range(batch_size):
            texts.append(
                "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                * 50
                + "Assistant:"
            )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    payload = {
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": output_len,
            "ignore_eos": True,
            "json_schema": json_schema,
            "stream_interval": stream_interval,
        },
        "return_logprob": return_logprob,
        "stream": True,
        "log_metrics": log_metrics,
        **({"parallel_batch": parallel_batch} if parallel_batch else {}),
    }
    if dataset_name == "mmmu":
        # vlm
        input_ids = []
        # for vlms, tokenizer is an instance of AutoProcessor
        tokenizer = tokenizer.tokenizer
        for input_req in input_requests:
            input_ids += [tokenizer.encode(input_req.prompt)]
        payload["image_data"] = [req.image_data for req in input_requests]

    else:
        input_ids = [req.prompt for req in input_requests]

    payload["input_ids"] = input_ids
    num_requests = len(input_ids)

    # Warm up cache if cache_hit_rate > 0.0
    if cache_hit_rate > 0.0 or cached_token_len:
        _warmup_cache(
            url=url,
            input_ids=input_ids,
            input_len=input_len,
            cache_hit_rate=cache_hit_rate,
            dataset_name=dataset_name,
            image_data=payload.get("image_data"),
            cached_token_len=cached_token_len,
        )
    else:
        print("No cache warmup, cache_hit_rate is set to 0.0 or cached_token_len is not specified.")

    # Turn on profiler
    profile_link = None
    if profile:
        # Use provided activities or default to ["CPU", "GPU"]
        activities = profile_activities if profile_activities is not None else ["CPU", "GPU"]
        # Use decode_url if provided (for PD-separated mode), otherwise use url
        effective_decode_url = decode_url if decode_url else url
        profile_link: str = run_profile_with_stages(
            url=effective_decode_url,
            num_steps=profile_steps,
            activities=activities,
            output_dir=profile_output_dir,
            profile_by_stage=profile_by_stage,
            profile_prefix=profile_prefix,
            profile_stages=profile_stages or DEFAULT_PROFILE_STAGES,
            merge_profiles=merge_profiles,
        )

    # Get metrics before the request (for cache hit rate calculation)
    metrics_before = get_cache_tokens_from_metrics(effective_decode_url)

    # Log request count before send and compare with bs
    print(
        f"[before send] request_count={num_requests}, bs={batch_size}, "
        f"match={num_requests == batch_size}"
    )

    # Run the request
    tic = time.perf_counter()
    response = requests.post(
        url + "/generate",
        json=payload,
        stream=True,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    # Get the TTFT of the last request in the batch
    last_ttft = 0.0
    last_completion_tokens = 0
    last_token_timestamp: Optional[float] = None
    tbt_samples: List[float] = []
    response_request_count: Optional[int] = None
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            if "error" in data:
                raise RuntimeError(f"Request has failed. {data}.")
            now = time.perf_counter()

            assert (
                data["meta_info"]["finish_reason"] is None
                or data["meta_info"]["finish_reason"]["type"] == "length"
            )
            completion_tokens = data["meta_info"]["completion_tokens"]
            if completion_tokens == 1:
                last_ttft = now - tic
            if (
                measure_tbt
                and isinstance(completion_tokens, int)
                and completion_tokens > last_completion_tokens
            ):
                new_token_count = completion_tokens - last_completion_tokens
                if last_completion_tokens > 0 and last_token_timestamp is not None:
                    elapsed = now - last_token_timestamp
                    per_token_tbt = elapsed / new_token_count
                    tbt_samples.extend([per_token_tbt] * new_token_count)
                    print(
                        "[TBT DEBUG] append samples: "
                        f"completion_tokens={completion_tokens}, "
                        f"new_token_count={new_token_count}, "
                        f"per_token_tbt_s={per_token_tbt:.9f}, "
                        f"total_samples={len(tbt_samples)}"
                    )
                last_completion_tokens = completion_tokens
                last_token_timestamp = now
            # Infer response request count from chunk (e.g. batched stream may have "text" list)
            if "text" in data and isinstance(data["text"], list):
                response_request_count = len(data["text"])

    # Log request count after response and compare with bs
    match_after = (
        (response_request_count == batch_size)
        if response_request_count is not None
        else None
    )
    print(
        f"[after response] request_count={response_request_count if response_request_count is not None else 'N/A'}, "
        f"bs={batch_size}, match={match_after if match_after is not None else 'N/A'}"
    )

    # Compute metrics
    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / last_ttft
    decode_latency = latency - last_ttft
    output_throughput = batch_size * output_len / decode_latency
    overall_throughput = batch_size * (input_len + output_len) / latency
    tbt_mean: Optional[float] = None
    tbt_median: Optional[float] = None
    tbt_min: Optional[float] = None
    tbt_max: Optional[float] = None
    tbt_std: Optional[float] = None
    tbt_est_total_time: Optional[float] = None
    if measure_tbt and tbt_samples:
        tbt_samples_for_stats = tbt_samples[1:] if len(tbt_samples) > 1 else []
        if tbt_samples_for_stats:
            tbt_mean = float(np.mean(tbt_samples_for_stats))
            tbt_median = float(np.median(tbt_samples_for_stats))
            tbt_min = float(np.min(tbt_samples_for_stats))
            tbt_max = float(np.max(tbt_samples_for_stats))
            tbt_std = float(np.std(tbt_samples_for_stats))
            decode_token_count_for_tbt = len(tbt_samples_for_stats)
            tbt_est_total_time = tbt_median * decode_token_count_for_tbt
            print(
                "[TBT DEBUG] stats exclude first sample: "
                f"raw_samples={len(tbt_samples)}, used_samples={len(tbt_samples_for_stats)}"
            )
        else:
            print(
                "[TBT DEBUG] stats unavailable after excluding first sample: "
                f"raw_samples={len(tbt_samples)}"
            )

    if measure_tbt:
        tbt_csv_path, tbt_plot_path = _save_tbt_artifacts(
            tbt_samples=tbt_samples,
            result_filename=result_filename,
            run_name=run_name,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
        )
        print(f"TBT samples saved to CSV: {tbt_csv_path}")
        if tbt_plot_path is not None:
            print(f"TBT fluctuation plot saved to: {tbt_plot_path}")
        else:
            print("TBT fluctuation plot not generated (no TBT samples).")

    tpot = (decode_latency / (output_len - 1) if output_len > 1 else 0.0)


    effective_decode_url = decode_url if decode_url else url
    response = requests.get(effective_decode_url + "/get_server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    server_info = response.json()
    internal_state = server_info.get("internal_states", [{}])
    last_gen_throughput = internal_state[0].get("last_gen_throughput", None) or -1
    acc_length = internal_state[0].get("avg_spec_accept_length", None) or -1

    # Calculate cache hit rate from before/after metrics delta
    metrics_after = get_cache_tokens_from_metrics(effective_decode_url)
    metrics_cache_hit_rate = calculate_cache_hit_rate(metrics_before, metrics_after)

    # Print results
    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")
    print(f"last_ttft: {last_ttft:.2f} s")
    print(f"decode_latency: {decode_latency:.2f} s")
    print(f"TPOT: {tpot:.4f} s")
    if measure_tbt:
        if tbt_mean is not None:
            print(
                "TBT (between-token, s): "
                f"mean={tbt_mean:.4f}, median={tbt_median:.4f}, min={tbt_min:.4f}, max={tbt_max:.4f}, std={tbt_std:.4f}"
            )
            print(
                "TBT estimated total decode time (s): "
                f"{tbt_est_total_time:.4f} (median * decode_token_count)"
            )
        else:
            print("TBT (between-token) stats: n/a (insufficient streamed token intervals).")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    if acc_length > 0:
        print(f"acc_length: {acc_length:.2f} ")
    if metrics_cache_hit_rate is not None:
        print(f"cache hit rate: {metrics_cache_hit_rate:.4f}")

    # Dump results
    result = BenchOneCaseResult(
        run_name=run_name,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        latency=latency,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        overall_throughput=overall_throughput,
        last_ttft=last_ttft,
        tpot=tpot,
        decode_latency=decode_latency,
        last_gen_throughput=last_gen_throughput,
        acc_length=acc_length,
        cache_hit_rate=metrics_cache_hit_rate,
        tbt_mean=tbt_mean,
        tbt_median=tbt_median,
        tbt_min=tbt_min,
        tbt_max=tbt_max,
        tbt_std=tbt_std,
        tbt_est_total_time=tbt_est_total_time,
        profile_link=profile_link,
    )

    # Save and return the results
    if result_filename:
        result.dump_to_jsonl(result_filename)

    return result


def should_skip_due_to_token_capacity(
    batch_size, input_len, output_len, skip_token_capacity_threshold, enable_dp_attention=False, dp_size=1
):
    if enable_dp_attention:
        if batch_size * (input_len + output_len) > skip_token_capacity_threshold * dp_size:
            print(
                "=" * 8
                + f"Skip benchmark {batch_size=} * ({input_len=} + {output_len=}) = {batch_size * (input_len + output_len)} > {skip_token_capacity_threshold=} * dp_size={dp_size} due to kv cache limit in DP attention mode."
                + "=" * 8
            )
            return True
    else:
        if batch_size * (input_len + output_len) > skip_token_capacity_threshold:
            print(
                "=" * 8
                + f"Skip benchmark {batch_size=} * ({input_len=} + {output_len=}) = {batch_size * (input_len + output_len)} > {skip_token_capacity_threshold=} due to kv cache limit."
                + "=" * 8
            )
            return True
    return False


def should_skip_due_to_max_running_requests(
    batch_size, skip_max_running_requests_threshold
):
    if batch_size > skip_max_running_requests_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} > {skip_max_running_requests_threshold=} due to max running requests limit."
            + "=" * 8
        )
        return True
    return False


def get_report_summary(
    results: List[BenchOneCaseResult], bench_args: BenchArgs, server_args: ServerArgs
):
    summary = (
        f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}."
    )
    if bench_args.cache_hit_rate > 0.0:
        summary += f" Cache hit rate: {bench_args.cache_hit_rate*100:.1f}%."
    if bench_args.measure_tbt:
        summary += " TBT direct measurement: enabled."
    summary += "\n"

    if is_blackwell():
        hourly_cost_per_gpu = 4  # $4/hour for one B200
    else:
        hourly_cost_per_gpu = 2  # $2/hour for one H100
    input_util = 0.7

    # sort result by input_len
    results.sort(key=lambda x: x.input_len)
    rows = []
    headers = [
        "batch size",
        "input len",
        "latency (s)",
        "input throughput (tok/s)",
        "output throughput (tok/s)",
        "acc length",
        "ITL (ms)",
        "input cost ($/1M)",
        "output cost ($/1M)",
        "cache hit rate",
    ]
    if bench_args.measure_tbt:
        headers.extend(
            [
                "TBT mean (ms)",
                "TBT median (ms)",
                "TBT min (ms)",
                "TBT max (ms)",
                "TBT std (ms)",
                "TBT est total (ms)",
            ]
        )
    if bench_args.profile:
        headers.append("profile")

    for res in results:
        hourly_cost = hourly_cost_per_gpu * server_args.tp_size
        accept_length = f"{res.acc_length:.2f}" if res.acc_length > 0 else "n/a"
        itl_ms = 1000 * res.batch_size / res.output_throughput
        input_cost = 1e6 / (res.input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / res.output_throughput / 3600 * hourly_cost
        cache_hit_rate = (
            f"{res.cache_hit_rate:.4f}" if res.cache_hit_rate is not None else "n/a"
        )

        row = [
            res.batch_size,
            res.input_len,
            f"{res.latency:.2f}",
            f"{res.input_throughput:.2f}",
            f"{res.output_throughput:.2f}",
            accept_length,
            f"{itl_ms:.2f}",
            f"{input_cost:.2f}",
            f"{output_cost:.2f}",
            cache_hit_rate,
        ]
        if bench_args.measure_tbt:
            row.extend(
                [
                    f"{res.tbt_mean * 1000:.2f}" if res.tbt_mean is not None else "n/a",
                    f"{res.tbt_median * 1000:.2f}" if res.tbt_median is not None else "n/a",
                    f"{res.tbt_min * 1000:.2f}" if res.tbt_min is not None else "n/a",
                    f"{res.tbt_max * 1000:.2f}" if res.tbt_max is not None else "n/a",
                    f"{res.tbt_std * 1000:.2f}" if res.tbt_std is not None else "n/a",
                    (
                        f"{res.tbt_est_total_time * 1000:.2f}"
                        if res.tbt_est_total_time is not None
                        else "n/a"
                    ),
                ]
            )
        if bench_args.profile:
            if res.profile_link:
                row.append(f"[Profile]({res.profile_link})")
            else:
                row.append("n/a")
        rows.append(row)

    summary += tabulate(rows, headers=headers, tablefmt="github")
    summary += "\n"

    return summary


def run_benchmark_internal(
    server_args: ServerArgs,
    bench_args: BenchArgs,
    launch_server_func: Callable = launch_server,
):
    print(f"Running customized benchmark.")
    # set random seed
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # launch a server or use the provided base_url
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(launch_server_func, server_args)

    # Get tokenizer
    # Use prefill_url for server info if provided (PD-separated mode)
    server_info_url = bench_args.prefill_url if bench_args.prefill_url else base_url
    response = requests.get(server_info_url + "/get_server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    server_info = response.json()
    if "tokenizer_path" in server_info:
        tokenizer_path = server_info["tokenizer_path"]
    elif "prefill" in server_info:
        tokenizer_path = server_info["prefill"][0]["tokenizer_path"]
    if bench_args.dataset_name == "mmmu":
        # mmmu implies this is a MLLM
        tokenizer = get_processor(tokenizer_path)
    else:
        tokenizer = get_tokenizer(tokenizer_path)

    # Get token capacity
    internal_state = server_info.get("internal_states", [{}])
    skip_token_capacity_threshold = (
        internal_state[0].get("memory_usage", {}).get("token_capacity", 1000000000)
    )

    # Get effective max running requests
    max_running_requests_per_dp = internal_state[0].get(
        "effective_max_running_requests_per_dp", -1
    )
    dp_size = server_info.get("dp_size", None) or 1
    assert (
        max_running_requests_per_dp > 0
    ), f"effective_max_running_requests_per_dp is not set, {max_running_requests_per_dp=}"
    skip_max_running_requests_threshold = max_running_requests_per_dp * dp_size

    # Warmup
    if not bench_args.skip_warmup:
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        print(f"Warmup with batch_size={bench_args.batch_size}")
        for bs in bench_args.batch_size:
            run_one_case(
                base_url,
                batch_size=bs,
                input_len=1024,
                output_len=16,
                temperature=bench_args.temperature,
                return_logprob=bench_args.return_logprob,
                stream_interval=bench_args.client_stream_interval,
                input_len_step_percentage=bench_args.input_len_step_percentage,
                run_name="",
                result_filename="",
                tokenizer=tokenizer,
                dataset_name=bench_args.dataset_name,
                dataset_path=bench_args.dataset_path,
                parallel_batch=bench_args.parallel_batch,
                log_metrics=False,
            )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    results = []
    profile_results = []
    try:
        # Benchmark all cases
        if bench_args.measure:
            for bs, il, ol in itertools.product(
                bench_args.batch_size, bench_args.input_len, bench_args.output_len
            ):
                if should_skip_due_to_max_running_requests(
                    bs, skip_max_running_requests_threshold
                ) or should_skip_due_to_token_capacity(
                    bs, il, ol, skip_token_capacity_threshold, server_args.enable_dp_attention, server_args.dp_size
                ):
                    continue
                results.append(
                    run_one_case(
                        base_url,
                        bs,
                        il,
                        ol,
                        temperature=bench_args.temperature,
                        return_logprob=bench_args.return_logprob,
                        stream_interval=bench_args.client_stream_interval,
                        input_len_step_percentage=bench_args.input_len_step_percentage,
                        run_name=bench_args.run_name,
                        result_filename=bench_args.result_filename,
                        tokenizer=tokenizer,
                        dataset_name=bench_args.dataset_name,
                        dataset_path=bench_args.dataset_path,
                        parallel_batch=bench_args.parallel_batch,
                        cache_hit_rate=bench_args.cache_hit_rate,
                        cached_token_len=bench_args.cached_token_len,
                        measure_tbt=bench_args.measure_tbt,
                    )
                )

        # Profile all cases
        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    if should_skip_due_to_max_running_requests(
                        bs, skip_max_running_requests_threshold
                    ) or should_skip_due_to_token_capacity(
                        bs, il, ol, skip_token_capacity_threshold, server_args.enable_dp_attention, server_args.dp_size
                    ):
                        continue
                    profile_prefix = (
                        bench_args.profile_prefix or ""
                    ) + f"bs-{bs}-il-{il}"
                    profile_results.append(
                        run_one_case(
                            base_url,
                            bs,
                            il,
                            ol,
                            temperature=bench_args.temperature,
                            return_logprob=bench_args.return_logprob,
                            stream_interval=bench_args.client_stream_interval,
                            input_len_step_percentage=bench_args.input_len_step_percentage,
                            run_name=bench_args.run_name,
                            result_filename=bench_args.result_filename,
                            tokenizer=tokenizer,
                            dataset_name=bench_args.dataset_name,
                            dataset_path=bench_args.dataset_path,
                            parallel_batch=bench_args.parallel_batch,
                            cache_hit_rate=bench_args.cache_hit_rate,
                            measure_tbt=bench_args.measure_tbt,
                            profile=bench_args.profile,
                            profile_steps=bench_args.profile_steps,
                            profile_by_stage=bench_args.profile_by_stage,
                            profile_prefix=profile_prefix,
                            profile_output_dir=bench_args.profile_output_dir,
                            profile_activities=bench_args.profile_activities,
                            profile_stages=bench_args.profile_stages,
                            merge_profiles=bench_args.merge_profiles,
                            decode_url=bench_args.decode_url,
                        )
                    )
            except Exception as e:
                print(f"Error profiling, some profile traces may not be dumped: {e}")

            # Replace the profile link for any successful profile results
            for res, profile_res in zip(results, profile_results, strict=False):
                res.profile_link = profile_res.profile_link
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return results, server_info

    # Print summary
    summary = get_report_summary(results, bench_args, server_args)
    print(summary)

    if is_in_ci() and bench_args.append_to_github_summary:
        write_github_step_summary(summary)

    return results, server_info
