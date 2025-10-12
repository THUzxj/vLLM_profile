#!/bin/bash
# filepath: serving_benchmark.sh

# vLLM Serving Benchmark Script for TTFT/TPOT measurements
# This script runs online serving benchmarks to measure Time to First Token (TTFT)
# and Tokens Per Output Token (TPOT) across different configurations

set -e

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DISABLE_LOG_STATS=False
export VLLM_LOG_KV_CACHE_USAGE=True

# Default parameters
MODEL="../ASearcher-Web-14B/"
OUTPUT_DIR="serving_results_$(date +%Y%m%d_%H%M%S)"
# INPUT_LENGTHS=(128 512 1024 2048 4096)
# OUTPUT_LENGTHS=(4 8)
# BATCH_SIZES=(1 4 8 16 32 64 128 256)
INPUT_LENGTHS=(128)
OUTPUT_LENGTHS=(4)
BATCH_SIZES=(4)
# MAX_BATCHED_TOKENS=(2048 4096 8192 16384)
MAX_BATCHED_TOKENS=(16384)
NUM_PROMPTS=100
REQUEST_RATE="inf"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --input-lengths)
            IFS=',' read -ra INPUT_LENGTHS <<< "$2"
            shift 2
            ;;
        --output-lengths)
            IFS=',' read -ra OUTPUT_LENGTHS <<< "$2"
            shift 2
            ;;
        --batch-sizes)
            IFS=',' read -ra BATCH_SIZES <<< "$2"
            shift 2
            ;;
        --max-batched-tokens)
            IFS=',' read -ra MAX_BATCHED_TOKENS <<< "$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --request-rate)
            REQUEST_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL                    Model name (default: $MODEL)"
            echo "  --output-dir DIR                 Output directory (default: auto-generated)"
            echo "  --input-lengths LENGTHS          Comma-separated input lengths (default: 128,512,1024,2048)"
            echo "  --output-lengths LENGTHS         Comma-separated output lengths (default: 32,128,256)"
            echo "  --batch-sizes SIZES              Comma-separated batch sizes (default: 1,4,8,16)"
            echo "  --max-batched-tokens TOKENS      Comma-separated max batched tokens (default: 2048,4096,8192,16384)"
            echo "  --num-prompts N                  Number of prompts per test (default: $NUM_PROMPTS)"
            echo "  --request-rate RATE              Request rate (default: $REQUEST_RATE)"
            echo "  --help                           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "Starting vLLM Serving Benchmark for TTFT/TPOT"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "Input lengths: ${INPUT_LENGTHS[*]}"
echo "Output lengths: ${OUTPUT_LENGTHS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Max batched tokens: ${MAX_BATCHED_TOKENS[*]}"
echo "Number of prompts: $NUM_PROMPTS"
echo "Request rate: $REQUEST_RATE"
echo ""

# Function to run serving benchmark for TTFT/TPOT measurements  
run_serving_benchmark() {
    local input_len=$1
    local output_len=$2
    local max_batched_tokens=$3
    local batch_size=$4
    
    local result_file="${OUTPUT_DIR}/serving_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs${batch_size}.json"
    local server_log_file="${LOG_DIR}/server_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs${batch_size}.log"
    local client_log_file="${LOG_DIR}/client_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs${batch_size}.log"
    local server_port=$((8000 + RANDOM % 1000))
    
    echo "Running serving benchmark: input=$input_len, output=$output_len, max_batched_tokens=$max_batched_tokens, batch_size=$batch_size"
    echo "  Server log: $server_log_file"
    echo "  Client log: $client_log_file"
    echo "  Server port: $server_port"
    
    # Start vLLM server in background
    vllm serve "$MODEL" \
        --port "$server_port" \
        --max-num-batched-tokens "$max_batched_tokens" \
        --disable-log-stats \
        --enforce-eager \
        --disable-custom-all-reduce \
        >"$server_log_file" 2>&1 &
    
    local server_pid=$!
    
    # Wait for server to start
    echo "Waiting for server to start on port $server_port..."
    timeout=120
    while ! curl -s "http://localhost:$server_port/health" > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            echo "Server failed to start on port $server_port"
            echo "Check server log: $server_log_file"
            kill $server_pid 2>/dev/null || true
            return 1
        fi
    done
    
    echo "Server started successfully on port $server_port"
    
    # Calculate actual num_prompts based on batch_size
    local actual_num_prompts=$((batch_size * (NUM_PROMPTS / batch_size)))
    if [ $actual_num_prompts -eq 0 ]; then
        actual_num_prompts=$batch_size
    fi
    
    # Run benchmark
    vllm bench serve \
        --backend openai \
        --base-url "http://localhost:$server_port" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --num-prompts "$actual_num_prompts" \
        --request-rate "$REQUEST_RATE" \
        --save-result \
        --result-dir "$OUTPUT_DIR" \
        --result-filename "serving_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs${batch_size}.json" \
        --percentile-metrics ttft,tpot,e2el >"$client_log_file" 2>&1 || {
            echo "Failed serving benchmark for input=$input_len, output=$output_len, max_batched_tokens=$max_batched_tokens, batch_size=$batch_size"
            echo "Check client log: $client_log_file"
        }
    
    # Stop server
    echo "Stopping server (PID: $server_pid)..."
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    
    # Wait a bit for cleanup
    sleep 3
    
    echo "Completed benchmark for input=$input_len, output=$output_len, max_batched_tokens=$max_batched_tokens, batch_size=$batch_size"
    echo ""
}

# Run serving benchmarks (online) for TTFT/TPOT
echo "=== Running Online Serving Benchmarks for TTFT/TPOT ==="
total_serving_configs=$((${#INPUT_LENGTHS[@]} * ${#OUTPUT_LENGTHS[@]} * ${#MAX_BATCHED_TOKENS[@]} * ${#BATCH_SIZES[@]}))
current_config=0

for input_len in "${INPUT_LENGTHS[@]}"; do
    for output_len in "${OUTPUT_LENGTHS[@]}"; do
        for max_batched_tokens in "${MAX_BATCHED_TOKENS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                current_config=$((current_config + 1))
                echo "[$current_config/$total_serving_configs] Serving benchmark"
                run_serving_benchmark "$input_len" "$output_len" "$max_batched_tokens" "$batch_size"
            done
        done
    done
done

# Generate summary report
echo ""
echo "=== Generating Summary Report ==="

# Check if external summary script exists
SUMMARY_SCRIPT="$(dirname "$0")/benchmark_summary.py"
if [ -f "$SUMMARY_SCRIPT" ]; then
    echo "Using external summary script: $SUMMARY_SCRIPT"
    python3 "$SUMMARY_SCRIPT" "$OUTPUT_DIR" --export-csv --serving-only
else
    echo "External summary script not found at $SUMMARY_SCRIPT"
    echo "Generating basic summary..."
    
    # Basic summary fallback
    echo "Results directory: $OUTPUT_DIR"
    echo "Serving result files:"
    find "$OUTPUT_DIR" -name "serving_*.json" -type f | wc -l | xargs echo "  Count:"
    
    # Show average TTFT and TPOT if files exist
    serving_files=$(find "$OUTPUT_DIR" -name "serving_*.json" -type f)
    if [ -n "$serving_files" ]; then
        echo ""
        echo "Quick Performance Summary:"
        echo "Configuration | TTFT (ms) | TPOT (ms) | Throughput (req/s)"
        echo "-------------|-----------|-----------|------------------"
        
        for file in $serving_files; do
            if [ -s "$file" ] && command -v jq >/dev/null 2>&1; then
                filename=$(basename "$file" .json)
                ttft=$(jq -r '.results.ttft_s.mean // "N/A"' "$file" 2>/dev/null | awk '{printf "%.1f", $1*1000}')
                tpot=$(jq -r '.results.tpot_s.mean // "N/A"' "$file" 2>/dev/null | awk '{printf "%.1f", $1*1000}')
                throughput=$(jq -r '.results.request_throughput // "N/A"' "$file" 2>/dev/null)
                echo "$filename | $ttft | $tpot | $throughput"
            fi
        done
    fi
fi

echo ""
echo "Serving benchmark completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Logs saved in: $LOG_DIR"
echo ""
echo "To view summary again:"
if [ -f "$SUMMARY_SCRIPT" ]; then
    echo "  python3 $SUMMARY_SCRIPT $OUTPUT_DIR --serving-only"
    echo "  python3 $SUMMARY_SCRIPT $OUTPUT_DIR --export-csv --serving-only"
else
    echo "  Summary script not available"
fi
echo ""
echo "Log files created:"
ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "No log files found"
echo ""
echo "Result files created:"
ls -la "$OUTPUT_DIR"/serving_*.json 2>/dev/null || echo "No result files found"