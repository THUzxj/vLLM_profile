#!/usr/bin/env bash
set -euo pipefail

# Parallel benchmark script: run bench against two vllm engines (ports 8315 and 8316)
# Both bench processes are started and wait on a shared file; touch the file to start them simultaneously.

model_id=/nfs/xjzhang/Qwen/Qwen3-0.6B/
model_name=Qwen3-0.6B
max_batched_tokens=100000
ports=(8315 8316)

for iter in 1 2 3 4 5; do
    for batch_size in 1 2 4 8 16 32 64 128; do
        num_prompts=$((batch_size * 10))
        input_len=1024
        output_len=128
        MPS_SIZE="50"

        # create a per-run start-file; processes wait for this file to appear
        START_FILE="/tmp/vllm_bench_
        start_${model_name}_bs${batch_size}_iter${iter}_$$"
        rm -f "$START_FILE"

        pids=()
        for port in "${ports[@]}"; do
            RESULT_DIR="bs_test_parallel_model${model_name}_mps${MPS_SIZE}_3/benchmark_results_parallel_port${port}_mbt${max_batched_tokens}_iter${iter}"
            mkdir -p "$RESULT_DIR"
            result_filename="serving_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs${batch_size}_port${port}.json"

            (
                echo "[bench-wait] port ${port} ready, waiting for start file: ${START_FILE}"
                # busy-wait small sleep loop until START_FILE exists
                while [ ! -f "$START_FILE" ]; do
                    sleep 0.01
                done
                echo "[bench-start] port ${port} starting bench"
                vllm bench serve --backend openai --base-url "http://localhost:${port}" --model "$model_id" \
                    --dataset-name random --random-input-len $input_len --random-output-len $output_len --num-prompts $num_prompts \
                    --request-rate inf --max-concurrency $batch_size --save-result --result-dir "$RESULT_DIR" \
                    --result-filename "$result_filename" \
                    --percentile-metrics ttft,tpot,e2el --num-warmups 5
            ) &
            pids+=("$!")
        done

        # give spawned processes a moment to reach the waiting loop
        sleep 0.2
        # release both processes at (nearly) the same time
        touch "$START_FILE"

        # wait for both benches to finish
        for pid in "${pids[@]}"; do
            wait "$pid"
        done

        rm -f "$START_FILE"
    done
done

echo "All parallel benches completed."
