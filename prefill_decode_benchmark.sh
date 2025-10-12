#!/bin/bash
# filepath: prefill_decode_benchmark.sh

# vLLM Offline Latency Benchmark Script
# This script uses vLLM's built-in benchmarking tools to measure
# offline latency (prefill and decode) across different configurations
# For online serving benchmarks (TTFT/TPOT), use serving_benchmark.sh

set -e

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DISABLE_LOG_STATS=False
export VLLM_LOG_KV_CACHE_USAGE=True

# Default parameters
MODEL="../ASearcher-Web-14B/"
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
# INPUT_LENGTHS=(128 512 1024 2048 4096)
INPUT_LENGTHS=(256)
OUTPUT_LENGTHS=(4 8 16)
BATCH_SIZES=(1 4 8 16 32 64 128 256)
# INPUT_LENGTHS=(2048)
# OUTPUT_LENGTHS=(4)
# BATCH_SIZES=(256)

MAX_BATCHED_TOKENS=(2048 4096 8192 16384)
# MAX_BATCHED_TOKENS=(16384)
NUM_PROMPTS=100
REQUEST_RATE="inf"
RUN_SERVING=false

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
        --run-serving)
            RUN_SERVING=true
            shift
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
            echo "  --run-serving                    Also run serving benchmarks after latency benchmarks"
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

echo "Starting vLLM Offline Latency Benchmark"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "Input lengths: ${INPUT_LENGTHS[*]}"
echo "Output lengths: ${OUTPUT_LENGTHS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Max batched tokens: ${MAX_BATCHED_TOKENS[*]}"
echo "Number of prompts: $NUM_PROMPTS"
echo ""

# Function to run latency benchmark
run_latency_benchmark() {
    local input_len=$1
    local output_len=$2
    local batch_size=$3
    local max_batched_tokens=$4
    
    local result_file="${OUTPUT_DIR}/latency_in${input_len}_out${output_len}_bs${batch_size}_mbt${max_batched_tokens}.json"
    local log_file="${LOG_DIR}/latency_in${input_len}_out${output_len}_bs${batch_size}_mbt${max_batched_tokens}.log"
    
    echo "Running latency benchmark: input=$input_len, output=$output_len, batch=$batch_size, max_batched_tokens=$max_batched_tokens"
    echo "  Log file: $log_file"
    
    vllm bench latency \
        --model "$MODEL" \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --batch-size "$batch_size" \
        --num-iters 2 \
        --num-iters-warmup 1 \
        --max-num-batched-tokens "$max_batched_tokens" \
        --output-json "$result_file" \
        --disable-log-stats \
        --enforce-eager \
        --disable-custom-all-reduce >"$log_file" 2>&1 || {
            echo "Failed for configuration: input=$input_len, output=$output_len, batch=$batch_size, max_batched_tokens=$max_batched_tokens"
            echo "Check log file: $log_file"
        }
}

# Note: Serving benchmarks have been moved to serving_benchmark.sh
# This script now focuses on offline latency benchmarks only

# Run latency benchmarks (offline)
echo "=== Running Offline Latency Benchmarks ==="
total_latency_configs=$((${#INPUT_LENGTHS[@]} * ${#OUTPUT_LENGTHS[@]} * ${#BATCH_SIZES[@]} * ${#MAX_BATCHED_TOKENS[@]}))
current_config=0

for input_len in "${INPUT_LENGTHS[@]}"; do
    for output_len in "${OUTPUT_LENGTHS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for max_batched_tokens in "${MAX_BATCHED_TOKENS[@]}"; do
                current_config=$((current_config + 1))
                echo "[$current_config/$total_latency_configs] Latency benchmark"
                run_latency_benchmark "$input_len" "$output_len" "$batch_size" "$max_batched_tokens"
            done
        done
    done
done

# Serving benchmarks have been moved to serving_benchmark.sh
# To run serving benchmarks for TTFT/TPOT measurements, use:
echo ""
echo "=== Serving Benchmarks ==="
echo "Serving benchmarks (for TTFT/TPOT) have been moved to a separate script."

if [ "$RUN_SERVING" = true ]; then
    echo "Running serving benchmarks automatically..."
    SERVING_SCRIPT="$(dirname "$0")/serving_benchmark.sh"
    if [ -f "$SERVING_SCRIPT" ]; then
        "$SERVING_SCRIPT" \
            --model "$MODEL" \
            --output-dir "${OUTPUT_DIR}_serving" \
            --input-lengths "$(IFS=,; echo "${INPUT_LENGTHS[*]}")" \
            --output-lengths "$(IFS=,; echo "${OUTPUT_LENGTHS[*]}")" \
            --batch-sizes "$(IFS=,; echo "${BATCH_SIZES[*]}")" \
            --max-batched-tokens "$(IFS=,; echo "${MAX_BATCHED_TOKENS[*]}")" \
            --num-prompts "$NUM_PROMPTS"
    else
        echo "Error: serving_benchmark.sh not found at $SERVING_SCRIPT"
    fi
else
    echo "To run serving benchmarks manually, use:"
    echo "  ./serving_benchmark.sh --model \"$MODEL\" --output-dir \"${OUTPUT_DIR}_serving\""
fi
echo ""

# Generate summary report
echo ""
echo "=== Generating Summary Report ==="

# Check if external summary script exists
SUMMARY_SCRIPT="$(dirname "$0")/benchmark_summary.py"
if [ -f "$SUMMARY_SCRIPT" ]; then
    echo "Using external summary script: $SUMMARY_SCRIPT"
    python3 "$SUMMARY_SCRIPT" "$OUTPUT_DIR" --export-csv
else
    echo "External summary script not found at $SUMMARY_SCRIPT"
    echo "Generating basic summary..."
    
    # Basic summary fallback
    echo "Results directory: $OUTPUT_DIR"
    echo "Latency result files:"
    find "$OUTPUT_DIR" -name "latency_*.json" -type f | wc -l | xargs echo "  Count:"
fi

echo ""
echo "Benchmark completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Logs saved in: $LOG_DIR"
echo ""
echo "To view summary again:"
if [ -f "$SUMMARY_SCRIPT" ]; then
    echo "  python3 $SUMMARY_SCRIPT $OUTPUT_DIR"
    echo "  python3 $SUMMARY_SCRIPT $OUTPUT_DIR --export-csv  # Export to CSV"
    echo "  python3 $SUMMARY_SCRIPT $OUTPUT_DIR --summary-only  # Show only counts"
else
    echo "  Summary script not available"
fi
echo ""
echo "Log files created:"
ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "No log files found"