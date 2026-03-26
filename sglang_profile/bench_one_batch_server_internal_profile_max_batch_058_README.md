# Metrics-Triggered Profiling Benchmark

`bench_one_batch_server_internal_profile_max_batch_058.py` is a benchmarking tool that monitors server metrics and triggers profiling when running requests reach a configurable threshold. This is useful for profiling at steady-state load conditions.

## Features

- **Metrics-triggered profiling**: Polls Prometheus `/metrics` endpoint and starts profiling when running requests reach a specified threshold
- **Continuous request sending**: Sends batches of requests at configurable intervals with newly sampled payloads each round
- **Decoupled architecture**: Request sending and profile triggering run in separate threads
- **PD disaggregation support**: Separate URLs for decode and prefill nodes in disaggregated deployments

## Architecture

The benchmark uses two decoupled daemon threads:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Main Thread                                  │
│  - Coordinates both threads                                      │
│  - Waits for completion or timeout                               │
│  - Collects and reports results                                  │
└─────────────────────────────────────────────────────────────────┘
            │                                    │
            ▼                                    ▼
┌─────────────────────────┐      ┌─────────────────────────────────┐
│  RequestSenderThread    │      │  ProfileTriggerThread            │
│                         │      │                                  │
│  - Samples new payload  │      │  - Polls /metrics endpoint       │
│    each round           │      │  - Monitors running requests     │
│  - Sends requests at    │      │  - Triggers profile when         │
│    specified intervals  │      │    threshold reached             │
│  - Can stop when        │      │  - Supports delay before         │
│    profile starts       │      │    starting profile              │
└─────────────────────────┘      └─────────────────────────────────┘
```

### ProfileTriggerThread

Monitors the server's `/metrics` endpoint and triggers profiling when:
1. Running requests >= `batch_size * profile_trigger_threshold`
2. Condition holds for 3 consecutive polls (to avoid false triggers)

### RequestSenderThread

Continuously sends requests with newly sampled payloads:
- Each round re-samples input tokens from the dataset
- Configurable interval between batches
- Can run indefinitely or stop when profiling starts

## CLI Arguments

### Basic Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-url` | `""` | Server URL (required if not launching server) |
| `--batch-size` | `[1]` | Batch size(s) to benchmark |
| `--input-len` | `[1024]` | Input length(s) per request |
| `--output-len` | `[16]` | Output length(s) per request |
| `--temperature` | `0.0` | Sampling temperature |
| `--dataset-name` | `random` | Dataset: `random`, `dummy`, or `mmmu` |
| `--dataset-path` | `""` | Path to dataset file |
| `--result-filename` | `result.jsonl` | Output file for results |

### Profile Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile` | `False` | Enable profiling |
| `--profile-steps` | `5` | Number of steps to profile |
| `--profile-by-stage` | `False` | Profile by stage (prefill/decode) |
| `--profile-stages` | `["decode"]` | Stages to profile |
| `--profile-activities` | `["CPU", "GPU"]` | Profile activities |
| `--profile-output-dir` | `None` | Output directory for traces |
| `--profile-prefix` | `None` | Prefix for profile files |
| `--merge-profiles` | `False` | Merge profiles from all ranks |
| `--use-nsys` | `False` | Use NSYS profiling |

### Metrics-Triggered Profiling Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile-trigger-threshold` | `0.9` | Threshold (0.0-1.0) of batch size to trigger profiling |
| `--profile-polling-interval` | `0.1` | Seconds between polling metrics |
| `--profile-delay-steps` | `0` | Polling cycles to wait after trigger before profiling |

### Continuous Sending Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--send-interval` | `0.0` | Seconds between sending batches |
| `--total-rounds` | `1` | Number of rounds (0 = infinite until profile done) |
| `--wait-for-profile` | `True` | Stop sender when profile starts |

### PD Disaggregation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--decode-url` | `None` | URL for decode node (for profiling in PD mode) |
| `--prefill-url` | `None` | URL for prefill node (for server info in PD mode) |

## Usage Examples

### Basic Usage

Profile when running requests reach 90% of batch size:

```bash
python bench_one_batch_server_internal_profile_max_batch_058.py \
    --base-url http://127.0.0.1:30000 \
    --batch-size 256 \
    --input-len 32000 \
    --output-len 101 \
    --profile \
    --profile-steps 10
```

### Continuous Sending with Re-sampling

Send batches continuously, profile when load is high:

```bash
python bench_one_batch_server_internal_profile_max_batch_058.py \
    --base-url http://127.0.0.1:30000 \
    --batch-size 256 \
    --input-len 32000 \
    --output-len 101 \
    --profile \
    --profile-steps 10 \
    --profile-trigger-threshold 0.9 \
    --send-interval 0.5 \
    --total-rounds 0 \
    --wait-for-profile
```

### PD Disaggregation Mode

Profile the decode node while sending requests to prefill:

```bash
python bench_one_batch_server_internal_profile_max_batch_058.py \
    --base-url http://prefill-node:30000 \
    --decode-url http://decode-node:30001 \
    --batch-size 256 \
    --profile \
    --profile-trigger-threshold 0.95
```

### Using the Shell Script

```bash
# Set environment variables
BS=256 IL=32000 OL=101 PROFILE_STEPS=10 \
PROFILE_TRIGGER_THRESHOLD=0.9 \
SEND_INTERVAL=0.5 TOTAL_ROUNDS=0 \
DP=1 TP=8 EP=1 \
./scripts_sc/bench_one_batch_server_profile_metrics_trigger.sh
```

## Output

### Console Output

The benchmark prints detailed progress information:

```
[ProfileTriggerThread] Started monitoring. Target: 256, Trigger threshold: 230 (90%)
[RequestSenderThread] Started. batch_size=256, send_interval=0.5s, total_rounds=0
[RequestSenderThread] Round 1: sent 256 requests, latency=12.34s
[ProfileTriggerThread] Running requests (235) >= threshold (230) for 3 times. Starting profile...
[ProfileTriggerThread] Profile started successfully. Output: /path/to/profile
[run_one_case_with_metrics_trigger] Results:
  Total rounds completed: 2
  Total requests sent: 512
  ...
```

### Result File

Results are saved in JSONL format:

```json
{
  "run_name": "metrics_trigger",
  "batch_size": 256,
  "input_len": 32000,
  "output_len": 101,
  "latency": 25.67,
  "input_throughput": 320000.00,
  "output_throughput": 1010.00,
  "overall_throughput": 321010.00,
  "last_ttft": 12.34,
  "last_gen_throughput": 1050.00,
  "acc_length": -1,
  "cache_hit_rate": null,
  "profile_link": "/path/to/profile"
}
```

## How It Works

### Metrics Polling

The `get_running_requests()` function fetches the `sglang:num_running_reqs` metric from the Prometheus `/metrics` endpoint:

```python
def get_running_requests(url: str) -> Optional[int]:
    response = requests.get(url + "/metrics", timeout=5)
    for line in response.text.split("\n"):
        if line.startswith("sglang:num_running_reqs"):
            # Parse and return the value
            ...
```

### Trigger Logic

Profiling is triggered when:
1. `running_requests >= batch_size * profile_trigger_threshold`
2. This condition holds for 3 consecutive polls
3. Optional `profile_delay_steps` have passed

### Why Metrics-Triggered?

Traditional profiling starts immediately after sending requests, which may profile during the ramp-up phase rather than steady-state. Metrics-triggered profiling ensures:
- Profiling happens at actual target load
- More accurate performance measurements
- Better reproducibility across runs

## Troubleshooting

### Profile Not Triggering

- Check that `batch_size` is achievable (not exceeding `max_running_requests`)
- Lower `--profile-trigger-threshold` if needed
- Increase `--total-rounds` to allow more time for requests to accumulate

### Requests Timing Out

- Increase `DEFAULT_TIMEOUT` in the script for large batches
- Check server logs for errors
- Reduce `--batch-size` or `--input-len`

### Connection Errors

- Verify server is running and accessible
- Check firewall rules
- Ensure correct URL format (include `http://`)

## Related Files

- `scripts_sc/bench_one_batch_server_profile_metrics_trigger.sh` - Shell wrapper script
- `scripts_sc/bench_one_batch_server_profile.sh` - Original non-metrics-triggered version
- `bench_one_batch_server_058.py` - Base benchmark script