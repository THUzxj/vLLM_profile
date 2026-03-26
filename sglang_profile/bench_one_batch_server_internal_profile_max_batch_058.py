import argparse
import concurrent.futures
import dataclasses
import itertools
import json
import multiprocessing
import os
import random
import re
import threading
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


def get_running_requests(url: str) -> Optional[int]:
    """
    Get the number of running requests from Prometheus /metrics endpoint.
    Returns the number of running requests or None if metrics are not available.
    """
    try:
        response = requests.get(url + "/metrics", timeout=5)
        response.raise_for_status()

        # Parse Prometheus text format
        # Looking for: sglang:num_running_reqs{...} <value>
        for line in response.text.split("\n"):
            if line.startswith("sglang:num_running_reqs"):
                match = re.search(
                    r"sglang:num_running_reqs(?:\{[^}]*\})?\s+([\d.eE+-]+)", line
                )
                if match:
                    return int(float(match.group(1)))
        return 0
    except Exception as e:
        print(f"Warning: Failed to get running requests from metrics: {e}")
        return None


def get_running_requests_by_rank(url: str, dp_rank: int = 0, timeout: int = 60) -> Optional[int]:
    """
    Get running requests for a specific DP rank from Prometheus /metrics endpoint.

    Args:
        url: Server URL
        dp_rank: DP rank to query (default: 0)
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Number of running requests for the specified rank, or None if not found.
    """
    try:
        start_time = time.time()
        response = requests.get(url + "/metrics", timeout=timeout)
        elapsed = time.time() - start_time
        print(f"[metrics] GET /metrics took {elapsed:.2f}s")
        response.raise_for_status()

        for line in response.text.split("\n"):
            if line.startswith("sglang:num_running_reqs"):
                # Parse: sglang:num_running_reqs{dp_rank="0",tp_rank="0",...} 128
                match = re.match(
                    r'sglang:num_running_reqs\{([^}]*)\}\s+([\d.eE+-]+)',
                    line
                )
                if match:
                    labels_str = match.group(1)
                    value = int(float(match.group(2)))

                    # Parse labels
                    labels = {}
                    for label in labels_str.split(','):
                        if '=' in label:
                            k, v = label.split('=')
                            labels[k.strip()] = v.strip('"')

                    # Check if dp_rank matches
                    if int(labels.get('dp_rank', 0)) == dp_rank:
                        return value, labels

        print(f"[metrics] No running requests found for dp_rank={dp_rank}")
        return None, None
    except Exception as e:
        print(f"[metrics] Failed to get running requests: {e}")
        return None, None


def get_all_running_requests_by_rank(url: str, timeout: int = 60) -> dict:
    """
    Get running requests for all DP ranks from Prometheus /metrics endpoint.

    Args:
        url: Server URL
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Dict mapping dp_rank to (value, labels) tuple
    """
    try:
        start_time = time.time()
        response = requests.get(url + "/metrics", timeout=timeout)
        elapsed = time.time() - start_time
        print(f"[metrics] GET /metrics took {elapsed:.2f}s")
        response.raise_for_status()

        results = {}
        for line in response.text.split("\n"):
            if line.startswith("sglang:num_running_reqs"):
                match = re.match(
                    r'sglang:num_running_reqs\{([^}]*)\}\s+([\d.eE+-]+)',
                    line
                )
                if match:
                    labels_str = match.group(1)
                    value = int(float(match.group(2)))

                    # Parse labels
                    labels = {}
                    for label in labels_str.split(','):
                        if '=' in label:
                            k, v = label.split('=')
                            labels[k.strip()] = v.strip('"')

                    dp_rank = int(labels.get('dp_rank', 0))
                    results[dp_rank] = (value, labels)

        if not results:
            print(f"[metrics] No running requests found in metrics")
        return results
    except Exception as e:
        print(f"[metrics] Failed to get running requests: {e}")
        return {}


def update_max_running_requests(url: str, new_value: int, max_retries: int = 30, retry_interval: float = 1.0) -> bool:
    """
    Update the server's max_running_requests at runtime.
    Requires no active requests in the server.

    Args:
        url: Server URL
        new_value: New max_running_requests value
        max_retries: Maximum number of retries if update fails
        retry_interval: Interval between retries in seconds

    Returns:
        True if update succeeded, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url + "/set_internal_state",
                json={"server_args": {"max_running_requests": new_value}},
                timeout=10,
            )
            result = response.json()

            # /set_internal_state returns List[bool] (one per DP rank)
            # All should be True for success
            if isinstance(result, list):
                success = all(result)
            elif isinstance(result, dict):
                success = result.get("updated", False)
            else:
                success = False

            if success:
                print(f"[update_max_running_requests] Successfully updated to {new_value}")
                return True
            else:
                if attempt < max_retries - 1:
                    print(f"[update_max_running_requests] Attempt {attempt + 1}/{max_retries} failed (result={result}), "
                          f"retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                else:
                    print(f"[update_max_running_requests] Failed after {max_retries} attempts (result={result})")
                    return False
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[update_max_running_requests] Error: {e}, retrying in {retry_interval}s...")
                time.sleep(retry_interval)
            else:
                print(f"[update_max_running_requests] Failed after {max_retries} attempts: {e}")
                return False

    return False


def get_max_running_requests_from_server(url: str) -> Optional[int]:
    """
    Get the current max_running_requests from server info.
    """
    try:
        response = requests.get(url + "/get_server_info", timeout=10)
        response.raise_for_status()
        server_info = response.json()
        internal_state = server_info.get("internal_states", [{}])
        if internal_state:
            return internal_state[0].get("effective_max_running_requests_per_dp")
        return None
    except Exception as e:
        print(f"Warning: Failed to get max_running_requests from server: {e}")
        return None


class ProfileTriggerThread(threading.Thread):
    """
    A background thread that monitors running requests and triggers profiling
    when the number of running requests reaches the target batch size.
    """

    def __init__(
        self,
        decode_url: str,
        target_batch_size: int,
        profile_steps: int,
        activities: List[str],
        output_dir: Optional[str] = None,
        profile_by_stage: bool = False,
        merge_profiles: bool = False,
        profile_prefix: Optional[str] = None,
        profile_stages: Optional[List[str]] = None,
        polling_interval: float = 0.1,
        trigger_threshold: float = 0.9,
        profile_delay_steps: int = 0,
        log_interval: float = 5.0,
        dp_size: int = 1,
        target_dp_rank: int = 0,
    ):
        super().__init__(daemon=True)
        self.decode_url = decode_url
        self.target_batch_size = target_batch_size
        self.profile_steps = profile_steps
        self.activities = activities
        self.output_dir = output_dir
        self.profile_by_stage = profile_by_stage
        self.merge_profiles = merge_profiles
        self.profile_prefix = profile_prefix
        self.profile_stages = profile_stages
        self.polling_interval = polling_interval
        self.trigger_threshold = trigger_threshold
        self.profile_delay_steps = profile_delay_steps
        self.log_interval = log_interval
        self.dp_size = dp_size
        self.target_dp_rank = target_dp_rank

        self._stop_event = threading.Event()
        self.profile_link: Optional[str] = None
        self.profile_started = False
        self.profile_error: Optional[str] = None
        self.running_requests_history: List[int] = []

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self):
        """Main loop: poll metrics and trigger profile when condition is met."""
        # Divide by dp_size because running_requests from metrics is per DP rank
        # Use half of batch_size as the actual max_running_requests
        max_running_per_dp = (self.target_batch_size // 2) // self.dp_size
        trigger_count = int(max_running_per_dp * self.trigger_threshold)
        print(f"[ProfileTriggerThread] Started monitoring. Target: {self.target_batch_size} "
              f"(max_running_per_dp: {max_running_per_dp}), Trigger threshold: {trigger_count} ({self.trigger_threshold*100:.0f}%), "
              f"DP size: {self.dp_size}, Target DP rank: {self.target_dp_rank}")

        consecutive_hits = 0
        required_consecutive = 3  # Require 3 consecutive hits to trigger

        while not self._stop_event.is_set():
            # Get running requests for target DP rank
            running_reqs, labels = get_running_requests_by_rank(self.decode_url, self.target_dp_rank)

            if running_reqs is not None:
                self.running_requests_history.append(running_reqs)

                # Log running requests on every poll
                label_str = ", ".join([f"{k}={v}" for k, v in (labels or {}).items()])
                print(f"[ProfileTriggerThread] Running requests: {running_reqs} "
                      f"(rank: {{{label_str}}}, threshold: {trigger_count})")

                if running_reqs >= trigger_count:
                    consecutive_hits += 1
                    if consecutive_hits >= required_consecutive and not self.profile_started:
                        label_str = ", ".join([f"{k}={v}" for k, v in (labels or {}).items()])
                        print(f"[ProfileTriggerThread] Running requests ({running_reqs}) >= "
                              f"threshold ({trigger_count}) for {consecutive_hits} times. "
                              f"Starting profile (rank: {{{label_str}}})...")

                        # Add optional delay before profiling
                        if self.profile_delay_steps > 0:
                            print(f"[ProfileTriggerThread] Waiting {self.profile_delay_steps} more polling cycles...")
                            delay_count = 0
                            while delay_count < self.profile_delay_steps and not self._stop_event.is_set():
                                time.sleep(self.polling_interval)
                                delay_count += 1

                        try:
                            self.profile_link = run_profile_with_stages(
                                url=self.decode_url,
                                num_steps=self.profile_steps,
                                activities=self.activities,
                                output_dir=self.output_dir,
                                profile_by_stage=self.profile_by_stage,
                                merge_profiles=self.merge_profiles,
                                profile_prefix=self.profile_prefix,
                                profile_stages=self.profile_stages,
                            )
                            self.profile_started = True
                            print(f"[ProfileTriggerThread] Profile started successfully. Output: {self.profile_link}")
                        except Exception as e:
                            self.profile_error = str(e)
                            print(f"[ProfileTriggerThread] Failed to start profile: {e}")
                else:
                    consecutive_hits = 0

            time.sleep(self.polling_interval)

    def get_profile_link(self) -> Optional[str]:
        """Get the profile output directory if profiling has started."""
        return self.profile_link

    def is_profile_started(self) -> bool:
        """Check if profiling has been triggered."""
        return self.profile_started

    def get_running_requests_history(self) -> List[int]:
        """Get the history of running requests counts."""
        return self.running_requests_history


class RequestSenderThread(threading.Thread):
    """
    A background thread that continuously sends requests with newly sampled payloads.
    This is decoupled from the profile trigger thread.
    Requests are sent asynchronously - doesn't wait for response before sending next.
    """

    def __init__(
        self,
        url: str,
        batch_size: int,
        input_len: int,
        output_len: int,
        temperature: float,
        return_logprob: bool,
        stream_interval: int,
        tokenizer: PreTrainedTokenizer | AutoProcessor,
        dataset_name: str = "random",
        dataset_path: str = "",
        parallel_batch: bool = False,
        send_interval: float = 5.0,
        total_rounds: int = 0,
        wait_for_profile: bool = True,
        profile_trigger_thread: Optional[ProfileTriggerThread] = None,
    ):
        super().__init__(daemon=True)
        self.url = url
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len
        self.temperature = temperature
        self.return_logprob = return_logprob
        self.stream_interval = stream_interval
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.parallel_batch = parallel_batch
        self.send_interval = send_interval
        self.total_rounds = total_rounds
        self.wait_for_profile = wait_for_profile
        self.profile_trigger_thread = profile_trigger_thread

        self._stop_event = threading.Event()
        self.rounds_completed = 0
        self.total_requests_sent = 0
        self.errors: List[str] = []
        self.latencies: List[float] = []
        self._pending_futures: List[concurrent.futures.Future] = []
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def _sample_payload(self) -> dict:
        """Sample a new payload for each round."""
        # Sample input requests based on dataset type
        if self.dataset_name == "mmmu":
            input_requests = sample_mmmu_requests(
                num_requests=self.batch_size,
                processor=self.tokenizer,
                fixed_output_len=self.output_len,
                random_sample=False,
            )
        elif self.dataset_name == "random":
            input_requests = sample_random_requests(
                input_len=self.input_len,
                output_len=self.output_len,
                num_prompts=self.batch_size,
                range_ratio=1.0,
                tokenizer=self.tokenizer,
                dataset_path=self.dataset_path,
                random_sample=True,
                return_text=False,
            )
        elif self.dataset_name == "dummy":
            input_requests = sample_random_requests(
                input_len=self.input_len,
                output_len=self.output_len,
                num_prompts=self.batch_size,
                range_ratio=1.0,
                tokenizer=self.tokenizer,
                dataset_path=self.dataset_path,
                random_sample=False,
                return_text=False,
            )
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        # Build payload
        payload = {
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.output_len,
                "ignore_eos": True,
                "stream_interval": self.stream_interval,
            },
            "return_logprob": self.return_logprob,
            "stream": True,
            **({"parallel_batch": self.parallel_batch} if self.parallel_batch else {}),
        }

        if self.dataset_name == "mmmu":
            input_ids = []
            actual_tokenizer = self.tokenizer.tokenizer
            for input_req in input_requests:
                input_ids += [actual_tokenizer.encode(input_req.prompt)]
            payload["image_data"] = [req.image_data for req in input_requests]
            payload["input_ids"] = input_ids
        else:
            payload["input_ids"] = [req.prompt for req in input_requests]

        return payload

    def _send_request(self, payload: dict) -> float:
        """Send a single request and return the latency."""
        tic = time.perf_counter()
        try:
            response = requests.post(
                self.url + "/generate",
                json=payload,
                stream=True,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()

            # Consume the response stream (fire-and-forget style, but still read it)
            for chunk in response.iter_lines(decode_unicode=False):
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    if chunk == "data: [DONE]":
                        break
                    data = json.loads(chunk[5:].strip("\n"))
                    if "error" in data:
                        error_msg = f"Request error: {data}"
                        self.errors.append(error_msg)
                        print(f"[RequestSenderThread] {error_msg}")

            latency = time.perf_counter() - tic
            return latency
        except Exception as e:
            error_msg = f"Request exception: {e}"
            self.errors.append(error_msg)
            print(f"[RequestSenderThread] {error_msg}")
            return -1.0

    def _on_request_complete(self, future: concurrent.futures.Future):
        """Callback when a request completes."""
        try:
            latency = future.result()
            if latency > 0:
                self.latencies.append(latency)
        except Exception as e:
            error_msg = f"Request callback error: {e}"
            self.errors.append(error_msg)
            print(f"[RequestSenderThread] {error_msg}")

    def run(self):
        """Main loop: send requests asynchronously at fixed intervals."""
        print(f"[RequestSenderThread] Started. batch_size={self.batch_size}, "
              f"send_interval={self.send_interval}s, total_rounds={self.total_rounds}")

        last_send_time = None
        # Track next scheduled send time for fixed interval
        next_send_time = time.time()

        while not self._stop_event.is_set():
            # Check if we've reached the total rounds (0 means infinite)
            if self.total_rounds > 0 and self.rounds_completed >= self.total_rounds:
                print(f"[RequestSenderThread] Reached total_rounds={self.total_rounds}, stopping.")
                break

            # Check if profile is done (if wait_for_profile is True and profile_trigger_thread exists)
            if self.wait_for_profile and self.profile_trigger_thread is not None:
                if self.profile_trigger_thread.is_profile_started():
                    print(f"[RequestSenderThread] Profile started, stopping after current round.")
                    # Send one more round then stop
                    if self.rounds_completed > 0:
                        break

            # Calculate and log interval since last send
            current_time = time.time()
            if last_send_time is not None:
                interval = current_time - last_send_time
                print(f"[RequestSenderThread] Interval since last send: {interval:.2f}s")
            last_send_time = current_time

            # Sample new payload and send request asynchronously
            try:
                payload = self._sample_payload()
                # Submit to thread pool - doesn't block
                future = self._executor.submit(self._send_request, payload)
                future.add_done_callback(self._on_request_complete)
                self._pending_futures.append(future)

                self.rounds_completed += 1
                self.total_requests_sent += self.batch_size
                print(f"[RequestSenderThread] Round {self.rounds_completed}: "
                      f"sent {self.batch_size} requests (async)")
            except Exception as e:
                self.errors.append(f"Error in round {self.rounds_completed}: {e}")
                print(f"[RequestSenderThread] Error in round {self.rounds_completed}: {e}")

            # Calculate exact wait time for fixed interval
            next_send_time += self.send_interval
            now = time.time()
            wait_time = next_send_time - now
            if wait_time > 0:
                time.sleep(wait_time)
            # If wait_time <= 0, we're behind schedule - send immediately

        # Wait for all pending requests to complete
        print(f"[RequestSenderThread] Waiting for {len(self._pending_futures)} pending requests...")
        concurrent.futures.wait(self._pending_futures, timeout=DEFAULT_TIMEOUT)

        print(f"[RequestSenderThread] Stopped. Total rounds: {self.rounds_completed}, "
              f"total requests: {self.total_requests_sent}, errors: {len(self.errors)}, "
              f"completed latencies: {len(self.latencies)}")

    def get_stats(self) -> dict:
        """Get statistics about the requests sent."""
        return {
            "rounds_completed": self.rounds_completed,
            "total_requests_sent": self.total_requests_sent,
            "errors": self.errors,
            "avg_latency": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            "latencies": self.latencies,
        }


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
    measure: bool = False
    profile_activities: Optional[List[str]] = None
    use_nsys: bool = False
    profile_stages: Optional[List[str]] = None
    merge_profiles: bool = False
    decode_url: Optional[str] = None
    prefill_url: Optional[str] = None
    profile_trigger_threshold: float = 0.9
    profile_polling_interval: float = 0.1
    profile_delay_steps: int = 0
    profile_log_interval: float = 5.0  # Interval for logging running requests
    # New parameters for continuous request sending
    send_interval: float = 5.0  # Interval between sending batches (seconds)
    total_rounds: int = 1  # Total number of rounds to send batches (0 = infinite until profile done)
    wait_for_profile: bool = True  # Whether to wait for profile to complete before stopping
    # Update max_running_requests before profiling
    update_max_running_reqs: bool = True
    max_running_reqs_update_retries: int = 30
    target_dp_rank: int = 0  # DP rank to monitor for running requests

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
            "--measure",
            action="store_true",
            default=BenchArgs.measure,
            help="If set, run the benchmark measurement loop (batch_size x input_len x output_len).",
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
        parser.add_argument(
            "--profile-trigger-threshold",
            type=float,
            default=BenchArgs.profile_trigger_threshold,
            help="Threshold (0.0-1.0) of batch size to trigger profiling. Default: 0.9 (90%%).",
        )
        parser.add_argument(
            "--profile-polling-interval",
            type=float,
            default=BenchArgs.profile_polling_interval,
            help="Interval in seconds between polling metrics. Default: 0.1.",
        )
        parser.add_argument(
            "--profile-delay-steps",
            type=int,
            default=BenchArgs.profile_delay_steps,
            help="Number of polling cycles to wait after trigger threshold before starting profile. Default: 0.",
        )
        parser.add_argument(
            "--profile-log-interval",
            type=float,
            default=BenchArgs.profile_log_interval,
            help="Interval in seconds between logging running requests. Default: 5.0.",
        )
        parser.add_argument(
            "--send-interval",
            type=float,
            default=BenchArgs.send_interval,
            help="Interval in seconds between sending batches. Default: 0.0 (send immediately after previous batch starts).",
        )
        parser.add_argument(
            "--total-rounds",
            type=int,
            default=BenchArgs.total_rounds,
            help="Total number of rounds to send batches. 0 means infinite until profile is done. Default: 1.",
        )
        parser.add_argument(
            "--wait-for-profile",
            action="store_true",
            default=BenchArgs.wait_for_profile,
            help="Wait for profile to complete before stopping request sender. Default: True.",
        )
        parser.add_argument(
            "--update-max-running-reqs",
            action="store_true",
            default=BenchArgs.update_max_running_reqs,
            help="Update max_running_requests to batch_size before profiling. Default: True.",
        )
        parser.add_argument(
            "--no-update-max-running-reqs",
            action="store_false",
            dest="update_max_running_reqs",
            help="Do not update max_running_requests before profiling.",
        )
        parser.add_argument(
            "--max-running-reqs-update-retries",
            type=int,
            default=BenchArgs.max_running_reqs_update_retries,
            help="Max retries for updating max_running_requests. Default: 30.",
        )
        parser.add_argument(
            "--target-dp-rank",
            type=int,
            default=BenchArgs.target_dp_rank,
            help="DP rank to monitor for running requests. Default: 0.",
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
    last_gen_throughput: float
    acc_length: float
    cache_hit_rate: Optional[float] = None
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
                "last_gen_throughput": round(self.last_gen_throughput, 2),
                "acc_length": round(self.acc_length, 2),
                "cache_hit_rate": (
                    round(self.cache_hit_rate, 4)
                    if self.cache_hit_rate is not None
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
    image_data: Optional[List] = None,
):
    """Warm up the cache by sending prefix tokens to populate the radix cache.

    Args:
        url: Server URL
        input_ids: List of input token id lists
        input_len: Length of input tokens
        cache_hit_rate: Fraction of input tokens to cache (0.0-1.0)
        dataset_name: Name of the dataset (used to determine if image data should be included)
        image_data: Optional image data for VLM models
    """
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
    profile_activities: Optional[List[str]] = None,
    profile_stages: Optional[List[str]] = None,
    merge_profiles: bool = False,
    decode_url: Optional[str] = None,
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
    if cache_hit_rate > 0.0:
        _warmup_cache(
            url=url,
            input_ids=input_ids,
            input_len=input_len,
            cache_hit_rate=cache_hit_rate,
            dataset_name=dataset_name,
            image_data=payload.get("image_data"),
        )

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
    response_request_count: Optional[int] = None
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            if "error" in data:
                raise RuntimeError(f"Request has failed. {data}.")

            assert (
                data["meta_info"]["finish_reason"] is None
                or data["meta_info"]["finish_reason"]["type"] == "length"
            )
            if data["meta_info"]["completion_tokens"] == 1:
                last_ttft = time.perf_counter() - tic
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
    output_throughput = batch_size * output_len / (latency - last_ttft)
    overall_throughput = batch_size * (input_len + output_len) / latency


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
        last_gen_throughput=last_gen_throughput,
        acc_length=acc_length,
        cache_hit_rate=metrics_cache_hit_rate,
        profile_link=profile_link,
    )

    # Save and return the results
    if result_filename:
        result.dump_to_jsonl(result_filename)

    return result


def run_one_case_with_metrics_trigger(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    stream_interval: int,
    run_name: str,
    result_filename: str,
    tokenizer: PreTrainedTokenizer | AutoProcessor,
    profile_steps: int = BenchArgs.profile_steps,
    profile_by_stage: bool = False,
    profile_prefix: Optional[str] = BenchArgs.profile_prefix,
    profile_output_dir: Optional[str] = BenchArgs.profile_output_dir,
    dataset_name: str = BenchArgs.dataset_name,
    dataset_path: str = BenchArgs.dataset_path,
    parallel_batch: bool = False,
    profile_activities: Optional[List[str]] = None,
    profile_stages: Optional[List[str]] = None,
    merge_profiles: bool = False,
    decode_url: Optional[str] = None,
    # Metrics-triggered profiling parameters
    profile_trigger_threshold: float = 0.9,
    profile_polling_interval: float = 0.1,
    profile_delay_steps: int = 0,
    profile_log_interval: float = 5.0,
    dp_size: int = 1,
    target_dp_rank: int = 0,
    # Continuous sending parameters
    send_interval: float = 5.0,
    total_rounds: int = 0,
    wait_for_profile: bool = True,
    # Update max_running_requests parameter
    update_max_running_reqs: bool = True,
    max_running_reqs_update_retries: int = 30,
    # Timeout for the entire operation
    timeout: float = 600.0,
):
    """
    Run benchmark with metrics-triggered profiling.

    This function uses two decoupled threads:
    1. RequestSenderThread: Continuously sends requests with newly sampled payloads
    2. ProfileTriggerThread: Monitors metrics and triggers profiling when running requests
       reach the threshold

    Args:
        url: Server URL
        batch_size: Number of requests per batch
        input_len: Input length per request
        output_len: Output length per request
        temperature: Sampling temperature
        return_logprob: Whether to return log probabilities
        stream_interval: Stream interval
        run_name: Name of the run
        result_filename: File to save results
        tokenizer: Tokenizer for sampling
        profile_steps: Number of steps to profile
        profile_by_stage: Whether to profile by stage
        profile_prefix: Prefix for profile output
        profile_output_dir: Output directory for profile
        dataset_name: Dataset name
        dataset_path: Path to dataset
        parallel_batch: Whether to use parallel batch
        profile_activities: Profile activities
        profile_stages: Profile stages
        merge_profiles: Whether to merge profiles
        decode_url: Decode URL for PD-separated mode
        profile_trigger_threshold: Threshold (0.0-1.0) of batch size to trigger profiling
        profile_polling_interval: Interval between polling metrics
        profile_delay_steps: Number of polling cycles to wait before profiling
        send_interval: Interval between sending batches
        total_rounds: Total rounds to send (0 = infinite until profile done)
        wait_for_profile: Whether to wait for profile to complete
        update_max_running_reqs: Whether to update max_running_requests to batch_size before profiling
        max_running_reqs_update_retries: Max retries for updating max_running_requests
        timeout: Timeout for the entire operation
    """
    response = requests.post(url + "/flush_cache", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    effective_decode_url = decode_url if decode_url else url
    activities = profile_activities if profile_activities is not None else ["CPU", "GPU"]

    print(f"[run_one_case_with_metrics_trigger] Starting with:")
    print(f"  batch_size={batch_size}, input_len={input_len}, output_len={output_len}")
    print(f"  profile_trigger_threshold={profile_trigger_threshold}")
    print(f"  send_interval={send_interval}s, total_rounds={total_rounds}")
    print(f"  dp_size={dp_size}, update_max_running_reqs={update_max_running_reqs}")

    # Update max_running_requests before starting if enabled
    if update_max_running_reqs:
        # Calculate per-DP max_running_requests (set to half of batch_size)
        new_max_running_reqs = (batch_size // 2) // dp_size
        print(f"[run_one_case_with_metrics_trigger] Updating max_running_requests to {new_max_running_reqs} (half of batch_size={batch_size}, per DP)")

        # Wait for server to be idle before updating
        print(f"[run_one_case_with_metrics_trigger] Waiting for server to be idle...")
        idle_wait_start = time.time()
        while time.time() - idle_wait_start < timeout:
            running_reqs, labels = get_running_requests_by_rank(effective_decode_url, target_dp_rank)
            if running_reqs is not None and running_reqs == 0:
                label_str = ", ".join([f"{k}={v}" for k, v in (labels or {}).items()])
                print(f"[run_one_case_with_metrics_trigger] Server is idle (running_reqs=0, rank: {{{label_str}}})")
                break
            time.sleep(0.5)
        else:
            print(f"[run_one_case_with_metrics_trigger] Warning: Timeout waiting for idle server")

        # Update max_running_requests
        success = update_max_running_requests(
            effective_decode_url,
            new_max_running_reqs,
            max_retries=max_running_reqs_update_retries,
            retry_interval=1.0,
        )
        if not success:
            print(f"[run_one_case_with_metrics_trigger] Warning: Failed to update max_running_requests, continuing anyway")
        else:
            # Verify the update
            current_max = get_max_running_requests_from_server(effective_decode_url)
            print(f"[run_one_case_with_metrics_trigger] Verified max_running_requests={current_max}")

    # Create profile trigger thread
    profile_trigger_thread = ProfileTriggerThread(
        decode_url=effective_decode_url,
        target_batch_size=batch_size,
        profile_steps=profile_steps,
        activities=activities,
        output_dir=profile_output_dir,
        profile_by_stage=profile_by_stage,
        merge_profiles=merge_profiles,
        profile_prefix=profile_prefix,
        profile_stages=profile_stages or DEFAULT_PROFILE_STAGES,
        polling_interval=profile_polling_interval,
        trigger_threshold=profile_trigger_threshold,
        profile_delay_steps=profile_delay_steps,
        log_interval=profile_log_interval,
        dp_size=dp_size,
        target_dp_rank=target_dp_rank,
    )

    # Create request sender thread
    request_sender_thread = RequestSenderThread(
        url=url,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        temperature=temperature,
        return_logprob=return_logprob,
        stream_interval=stream_interval,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        parallel_batch=parallel_batch,
        send_interval=send_interval,
        total_rounds=total_rounds,
        wait_for_profile=wait_for_profile,
        profile_trigger_thread=profile_trigger_thread,
    )

    # Start both threads
    profile_trigger_thread.start()
    request_sender_thread.start()

    # Wait for completion or timeout
    start_time = time.time()
    profile_started_time = None

    try:
        while time.time() - start_time < timeout:
            # Check if profile has started
            if profile_trigger_thread.is_profile_started():
                if profile_started_time is None:
                    profile_started_time = time.time()
                    print(f"[run_one_case_with_metrics_trigger] Profile started at {profile_started_time - start_time:.2f}s")

                # After profile starts, wait for request sender to finish
                # (it will stop after the current round if wait_for_profile is True)
                if not request_sender_thread.is_alive():
                    break

            # Check if request sender has finished
            if not request_sender_thread.is_alive():
                if total_rounds > 0 or not wait_for_profile:
                    break

            # Check for errors
            if profile_trigger_thread.profile_error:
                print(f"[run_one_case_with_metrics_trigger] Profile error: {profile_trigger_thread.profile_error}")
                break

            time.sleep(0.5)

    finally:
        # Stop both threads
        request_sender_thread.stop()
        profile_trigger_thread.stop()

        # Wait for threads to finish
        request_sender_thread.join(timeout=10)
        profile_trigger_thread.join(timeout=10)

    # Collect results
    request_stats = request_sender_thread.get_stats()
    profile_link = profile_trigger_thread.get_profile_link()
    running_requests_history = profile_trigger_thread.get_running_requests_history()

    # Get server info
    response = requests.get(effective_decode_url + "/get_server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    server_info = response.json()
    internal_state = server_info.get("internal_states", [{}])
    last_gen_throughput = internal_state[0].get("last_gen_throughput", None) or -1
    acc_length = internal_state[0].get("avg_spec_accept_length", None) or -1

    # Calculate metrics
    total_latency = time.time() - start_time
    avg_latency = request_stats["avg_latency"]
    total_requests = request_stats["total_requests_sent"]

    # Estimate throughput based on average latency
    if avg_latency > 0:
        input_throughput = batch_size * input_len / avg_latency
        output_throughput = batch_size * output_len / avg_latency
    else:
        input_throughput = 0
        output_throughput = 0

    # Print results
    print(f"\n[run_one_case_with_metrics_trigger] Results:")
    print(f"  Total rounds completed: {request_stats['rounds_completed']}")
    print(f"  Total requests sent: {total_requests}")
    print(f"  Total latency: {total_latency:.2f}s")
    print(f"  Average latency per batch: {avg_latency:.2f}s")
    print(f"  Input throughput: {input_throughput:.2f} tok/s")
    print(f"  Output throughput: {output_throughput:.2f} tok/s")
    print(f"  Last gen throughput: {last_gen_throughput:.2f} tok/s")
    if acc_length > 0:
        print(f"  Accept length: {acc_length:.2f}")
    if profile_link:
        print(f"  Profile output: {profile_link}")
    if running_requests_history:
        print(f"  Running requests history (last 10): {running_requests_history[-10:]}")
    if request_stats["errors"]:
        print(f"  Errors: {len(request_stats['errors'])} errors")
        for err in request_stats["errors"][:3]:
            print(f"    - {err}")

    # Create result object
    result = BenchOneCaseResult(
        run_name=run_name,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        latency=total_latency,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        overall_throughput=(input_throughput + output_throughput) if output_throughput > 0 else input_throughput,
        last_ttft=avg_latency,  # Use avg latency as approximation
        last_gen_throughput=last_gen_throughput,
        acc_length=acc_length,
        cache_hit_rate=None,  # Not calculated in metrics-triggered mode
        profile_link=profile_link,
    )

    # Save results
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
                        run_one_case_with_metrics_trigger(
                            url=base_url,
                            batch_size=bs,
                            input_len=il,
                            output_len=ol,
                            temperature=bench_args.temperature,
                            return_logprob=bench_args.return_logprob,
                            stream_interval=bench_args.client_stream_interval,
                            run_name=bench_args.run_name,
                            result_filename=bench_args.result_filename,
                            tokenizer=tokenizer,
                            profile_steps=bench_args.profile_steps,
                            profile_by_stage=bench_args.profile_by_stage,
                            profile_prefix=profile_prefix,
                            profile_output_dir=bench_args.profile_output_dir,
                            dataset_name=bench_args.dataset_name,
                            dataset_path=bench_args.dataset_path,
                            parallel_batch=bench_args.parallel_batch,
                            profile_activities=bench_args.profile_activities,
                            profile_stages=bench_args.profile_stages,
                            merge_profiles=bench_args.merge_profiles,
                            decode_url=bench_args.decode_url,
                            profile_trigger_threshold=bench_args.profile_trigger_threshold,
                            profile_polling_interval=bench_args.profile_polling_interval,
                            profile_delay_steps=bench_args.profile_delay_steps,
                            profile_log_interval=bench_args.profile_log_interval,
                            dp_size=server_args.dp_size,
                            target_dp_rank=bench_args.target_dp_rank,
                            send_interval=bench_args.send_interval,
                            total_rounds=bench_args.total_rounds,
                            wait_for_profile=bench_args.wait_for_profile,
                            update_max_running_reqs=bench_args.update_max_running_reqs,
                            max_running_reqs_update_retries=bench_args.max_running_reqs_update_retries,
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
