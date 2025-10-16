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

# Default torch profiler directory (can be overridden by command line or environment)
DEFAULT_PROFILER_DIR="./traces/"

# Default parameters
MODEL="../Qwen3-32B/"
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
INPUT_LENGTHS=(64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
# INPUT_LENGTHS=(32768 65536)
# INPUT_LENGTHS=(256)
OUTPUT_LENGTHS=(8)
# BATCH_SIZES=(4)
# BATCH_SIZES=(2)
# BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
BATCH_SIZES=(8)
# INPUT_LENGTHS=(2048)
# OUTPUT_LENGTHS=(4)
# BATCH_SIZES=(256)

MAX_BATCHED_TOKENS=(131072)
# MAX_BATCHED_TOKENS=(16384)
# MAX_BATCHED_TOKENS=(16384)
NUM_PROMPTS=100
REQUEST_RATE="inf"
RUN_SERVING=false

# Parse command line arguments
RESUME_MODE=false
FORCE_OVERWRITE=false
MAX_MODEL_LEN=""  # Empty means use vLLM default
ENABLE_PROFILER=false  # Default: disable profiler

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
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --force-overwrite)
            FORCE_OVERWRITE=true
            shift
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
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --enable-profiler)
            ENABLE_PROFILER=true
            if [ -n "$2" ] && [[ "$2" != --* ]]; then
                export VLLM_TORCH_PROFILER_DIR="$2"
                shift 2
            else
                export VLLM_TORCH_PROFILER_DIR="$DEFAULT_PROFILER_DIR"
                shift
            fi
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
            echo "  --resume                         Resume from existing directory, skip completed experiments"
            echo "  --force-overwrite               Overwrite existing result files"
            echo "  --input-lengths LENGTHS          Comma-separated input lengths (default: 128,512,1024,2048)"
            echo "  --output-lengths LENGTHS         Comma-separated output lengths (default: 32,128,256)"
            echo "  --batch-sizes SIZES              Comma-separated batch sizes (default: 1,4,8,16)"
            echo "  --max-batched-tokens TOKENS      Comma-separated max batched tokens (default: 2048,4096,8192,16384)"
            echo "  --max-model-len LENGTH           Maximum model length (default: use vLLM default)"
            echo "  --num-prompts N                  Number of prompts per test (default: $NUM_PROMPTS)"
            echo "  --enable-profiler [DIR]          Enable torch profiler (default dir: $DEFAULT_PROFILER_DIR)"
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

# Handle output directory creation and resume mode
if [ "$RESUME_MODE" = true ]; then
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Error: Resume mode specified but output directory '$OUTPUT_DIR' does not exist"
        echo "Either create the directory or run without --resume to create a new one"
        exit 1
    fi
    echo "Resume mode: Using existing directory '$OUTPUT_DIR'"
    echo "Will skip experiments with existing result files"
else
    # Check if directory exists and handle accordingly
    if [ -d "$OUTPUT_DIR" ] && [ "$FORCE_OVERWRITE" != true ]; then
        echo "Warning: Output directory '$OUTPUT_DIR' already exists"
        echo "Use --resume to continue from existing results"
        echo "Use --force-overwrite to overwrite existing results"
        echo "Or specify a different --output-dir"
        exit 1
    fi
fi

# Handle profiler configuration
# Check if VLLM_TORCH_PROFILER_DIR is already set in environment
if [ -n "$VLLM_TORCH_PROFILER_DIR" ] && [ "$ENABLE_PROFILER" != true ]; then
    echo "VLLM_TORCH_PROFILER_DIR is set in environment: $VLLM_TORCH_PROFILER_DIR"
    ENABLE_PROFILER=true
elif [ "$ENABLE_PROFILER" != true ]; then
    # Profiler is disabled, ensure environment variable is unset
    unset VLLM_TORCH_PROFILER_DIR
fi

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Setup profiler directory if profiler is enabled
if [ "$ENABLE_PROFILER" = true ]; then
    PROFILER_BASE_DIR="$OUTPUT_DIR/traces"
    mkdir -p "$PROFILER_BASE_DIR"
    # Create the base profiler directory
    export VLLM_TORCH_PROFILER_DIR="$PROFILER_BASE_DIR"
    echo "Torch profiler enabled: $VLLM_TORCH_PROFILER_DIR"
    echo "  Individual experiment traces will be saved to subdirectories"
else
    echo "Torch profiler disabled"
fi

echo "Starting vLLM Offline Latency Benchmark"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "Resume mode: $RESUME_MODE"
echo "Force overwrite: $FORCE_OVERWRITE"
echo "Torch profiler: $([ "$ENABLE_PROFILER" = true ] && echo "enabled ($VLLM_TORCH_PROFILER_DIR)" || echo "disabled")"
echo "Input lengths: ${INPUT_LENGTHS[*]}"
echo "Output lengths: ${OUTPUT_LENGTHS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Max batched tokens: ${MAX_BATCHED_TOKENS[*]}"
echo "Max model length: ${MAX_MODEL_LEN:-'(use vLLM default)'}"
echo "Number of prompts: $NUM_PROMPTS"
echo ""

# Function to run latency benchmark
run_latency_benchmark() {
    local input_len=$1
    local output_len=$2
    local batch_size=$3
    local max_batched_tokens=$4
    
    # Check if input_length * batch_size exceeds max_batched_tokens
    local total_input_tokens=$((input_len * batch_size))
    if [ $total_input_tokens -gt $max_batched_tokens ]; then
        echo "Skipping configuration: input=$input_len, batch=$batch_size (total_input_tokens=$total_input_tokens > max_batched_tokens=$max_batched_tokens)"
        return 0
    fi
    
    local result_file="${OUTPUT_DIR}/latency_in${input_len}_out${output_len}_bs${batch_size}_mbt${max_batched_tokens}.json"
    local log_file="${LOG_DIR}/latency_in${input_len}_out${output_len}_bs${batch_size}_mbt${max_batched_tokens}.log"
    
    # Check if result file already exists and handle accordingly
    if [ -f "$result_file" ]; then
        if [ "$FORCE_OVERWRITE" = true ]; then
            echo "Overwriting existing result: input=$input_len, output=$output_len, batch=$batch_size, max_batched_tokens=$max_batched_tokens"
        elif [ "$RESUME_MODE" = true ]; then
            echo "Skipping completed experiment: input=$input_len, output=$output_len, batch=$batch_size, max_batched_tokens=$max_batched_tokens (file exists: $(basename "$result_file"))"
            return 0
        else
            echo "Result file exists: $result_file"
            echo "Use --resume to skip or --force-overwrite to overwrite"
            return 1
        fi
    fi
    
    echo "Running latency benchmark: input=$input_len, output=$output_len, batch=$batch_size, max_batched_tokens=$max_batched_tokens (total_input_tokens=$total_input_tokens)"
    echo "  Result file: $result_file"
    echo "  Log file: $log_file"
    
    # Store original profiler directory if profiler is enabled
    local original_profiler_dir=""
    if [ "$ENABLE_PROFILER" = true ]; then
        original_profiler_dir="$VLLM_TORCH_PROFILER_DIR"
    fi
    
    # Build vLLM command with optional parameters
    vllm_cmd="vllm bench latency \
        --model \"$MODEL\" \
        --input-len \"$input_len\" \
        --output-len \"$output_len\" \
        --batch-size \"$batch_size\" \
        --num-iters 10 \
        --num-iters-warmup 3 \
        --max-num-batched-tokens \"$max_batched_tokens\" \
        --output-json \"$result_file\" \
        --disable-log-stats \
        --enforce-eager \
        --disable-custom-all-reduce"
    
    # Add profiler flag if enabled with config-specific directory
    if [ "$ENABLE_PROFILER" = true ]; then
        # Create config-specific profiler directory
        local config_profiler_dir="${VLLM_TORCH_PROFILER_DIR%/}/trace_in${input_len}_out${output_len}_bs${batch_size}_mbt${max_batched_tokens}"
        mkdir -p "$config_profiler_dir"
        
        # Set the profiler directory for this specific run
        export VLLM_TORCH_PROFILER_DIR="$config_profiler_dir"
        
        vllm_cmd="$vllm_cmd --profile"
        echo "  Profiler: enabled (traces will be saved to $config_profiler_dir)"
    else
        echo "  Profiler: disabled"
    fi
    
    # Add max-model-len parameter if specified
    if [ -n "$MAX_MODEL_LEN" ]; then
        vllm_cmd="$vllm_cmd --max-model-len \"$MAX_MODEL_LEN\""
        echo "  Max model length: $MAX_MODEL_LEN"
    else
        echo "  Max model length: (using vLLM default)"
    fi
    
    # Execute the command
    eval "$vllm_cmd" >"$log_file" 2>&1 || {
            echo "Failed for configuration: input=$input_len, output=$output_len, batch=$batch_size, max_batched_tokens=$max_batched_tokens"
            echo "Check log file: $log_file"
        }
    
    # Restore original profiler directory if it was changed
    if [ "$ENABLE_PROFILER" = true ] && [ -n "$original_profiler_dir" ]; then
        export VLLM_TORCH_PROFILER_DIR="$original_profiler_dir"
    fi
}

# Note: Serving benchmarks have been moved to serving_benchmark.sh
# This script now focuses on offline latency benchmarks only

# Run latency benchmarks (offline)
echo "=== Running Offline Latency Benchmarks ==="

# Calculate total configurations that need to be run
# (excluding those that exceed max_batched_tokens and already completed ones)
total_latency_configs=0
completed_configs=0
skipped_token_limit=0

for input_len in "${INPUT_LENGTHS[@]}"; do
    for output_len in "${OUTPUT_LENGTHS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for max_batched_tokens in "${MAX_BATCHED_TOKENS[@]}"; do
                total_input_tokens=$((input_len * batch_size))
                
                if [ $total_input_tokens -gt $max_batched_tokens ]; then
                    skipped_token_limit=$((skipped_token_limit + 1))
                    continue
                fi
                
                result_file="${OUTPUT_DIR}/latency_in${input_len}_out${output_len}_bs${batch_size}_mbt${max_batched_tokens}.json"
                
                if [ -f "$result_file" ] && [ "$FORCE_OVERWRITE" != true ]; then
                    completed_configs=$((completed_configs + 1))
                else
                    total_latency_configs=$((total_latency_configs + 1))
                fi
            done
        done
    done
done

echo "Configuration Summary:"
echo "  Configurations to run: $total_latency_configs"
echo "  Already completed: $completed_configs"
echo "  Skipped (token limit): $skipped_token_limit"
echo "  Total possible: $((total_latency_configs + completed_configs + skipped_token_limit))"
echo ""

if [ $total_latency_configs -eq 0 ]; then
    echo "No configurations need to be run!"
    if [ $completed_configs -gt 0 ]; then
        echo "All experiments are already completed. Use --force-overwrite to rerun them."
    fi
    echo "Proceeding to summary generation..."
else
    echo "Starting benchmark execution..."
fi

current_config=0

for input_len in "${INPUT_LENGTHS[@]}"; do
    for output_len in "${OUTPUT_LENGTHS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for max_batched_tokens in "${MAX_BATCHED_TOKENS[@]}"; do
                # Check if this configuration should be run
                total_input_tokens=$((input_len * batch_size))
                if [ $total_input_tokens -gt $max_batched_tokens ]; then
                    continue
                fi
                
                result_file="${OUTPUT_DIR}/latency_in${input_len}_out${output_len}_bs${batch_size}_mbt${max_batched_tokens}.json"
                
                # Skip if file exists and not in force overwrite mode
                if [ -f "$result_file" ] && [ "$FORCE_OVERWRITE" != true ]; then
                    continue
                fi
                
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
echo "=== Benchmark Completion Report ==="

# Count actual result files
actual_results=$(find "$OUTPUT_DIR" -name "latency_*.json" -type f 2>/dev/null | wc -l)
expected_total=$((total_latency_configs + completed_configs))

echo "Latency benchmark results:"
echo "  Completed result files: $actual_results"
echo "  Expected total: $expected_total"

if [ $actual_results -eq $expected_total ]; then
    echo "  Status: ✓ All experiments completed successfully"
elif [ $actual_results -gt 0 ]; then
    echo "  Status: ⚠ Partially completed ($actual_results/$expected_total)"
else
    echo "  Status: ✗ No results found"
fi

echo ""
echo "Results saved in: $OUTPUT_DIR"
echo "Logs saved in: $LOG_DIR"

# Report profiler traces if enabled
if [ "$ENABLE_PROFILER" = true ]; then
    echo "Profiler traces saved in: $VLLM_TORCH_PROFILER_DIR"
    
    # Count trace directories
    trace_dirs=$(find "$VLLM_TORCH_PROFILER_DIR" -name "trace_*" -type d 2>/dev/null | wc -l)
    if [ $trace_dirs -gt 0 ]; then
        echo "  Number of trace directories: $trace_dirs"
        echo "  Trace directories:"
        find "$VLLM_TORCH_PROFILER_DIR" -name "trace_*" -type d 2>/dev/null | sort | while read dir; do
            trace_files=$(find "$dir" -name "*.json" -o -name "*.pt" 2>/dev/null | wc -l)
            echo "    $(basename "$dir"): $trace_files files"
        done
    else
        echo "  No trace directories found"
    fi
fi

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