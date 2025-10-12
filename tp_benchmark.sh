#!/bin/bash
# filepath: tp_benchmark.sh

# vLLM Tensor Parallel (TP) Benchmark Script
# This script tests offline inference throughput and latency across different TP configurations

set -e

export VLLM_LOG_KV_CACHE_USAGE=True
export VLLM_DISABLE_LOG_STATS=False
export VLLM_LOGGING_LEVEL=DEBUG

# Default parameters
MODEL="meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR="/home/v-xingjzhang/xingjian/blob/tp_benchmark/tp_benchmark_results_$(date +%Y%m%d_%H%M%S)"

# Default TP configurations to test
# TP_SIZES=(1 2 4 8)
TP_SIZES=(1 2 4 8)

# Test configurations
INPUT_LENGTHS=(128 512 1024 2048)
OUTPUT_LENGTHS=(32)
BATCH_SIZES=(16 256)

# Benchmark parameters
NUM_PROMPTS=200
NUM_ITERS=3
NUM_WARMUP_ITERS=1

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
        --tp-sizes)
            IFS=',' read -ra TP_SIZES <<< "$2"
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
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --num-iters)
            NUM_ITERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL                    Model path (default: $MODEL)"
            echo "  --output-dir DIR                 Output directory (default: auto-generated)"
            echo "  --tp-sizes SIZES                 Comma-separated TP sizes (default: 1,2,4,8)"
            echo "  --input-lengths LENGTHS          Comma-separated input lengths (default: 128,512,1024,2048)"
            echo "  --output-lengths LENGTHS         Comma-separated output lengths (default: 32,128,256)"
            echo "  --batch-sizes SIZES              Comma-separated batch sizes (default: 1,4,16,32)"
            echo "  --num-prompts N                  Number of prompts per test (default: $NUM_PROMPTS)"
            echo "  --num-iters N                    Number of iterations per test (default: $NUM_ITERS)"
            echo "  --help                           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate TP sizes
for tp_size in "${TP_SIZES[@]}"; do
    if ! [[ "$tp_size" =~ ^[0-9]+$ ]] || [ "$tp_size" -lt 1 ]; then
        echo "Error: Invalid TP size: $tp_size"
        exit 1
    fi
done

# Check GPU availability
check_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo "Detected $gpu_count GPUs"
        
        max_tp_size=$(printf '%s\n' "${TP_SIZES[@]}" | sort -rn | head -1)
        if [ "$max_tp_size" -gt "$gpu_count" ]; then
            echo "Warning: Maximum TP size ($max_tp_size) exceeds available GPUs ($gpu_count)"
            echo "Some tests may fail or use CPU fallback"
        fi
    else
        echo "Warning: nvidia-smi not found, cannot check GPU count"
    fi
}

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "Starting vLLM Tensor Parallel Benchmark"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "TP sizes: ${TP_SIZES[*]}"
echo "Input lengths: ${INPUT_LENGTHS[*]}"
echo "Output lengths: ${OUTPUT_LENGTHS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Number of prompts per test: $NUM_PROMPTS"
echo "Number of iterations per test: $NUM_ITERS"
echo ""

check_gpu_count
echo ""

# Function to check if processes are using GPUs
cleanup_gpu_processes() {
    echo "Cleaning up any remaining GPU processes..."
    pkill -f "vllm" || true
    sleep 3
    
    # Additional cleanup if needed
    if command -v nvidia-smi &> /dev/null; then
        gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
        if [ -n "$gpu_processes" ]; then
            echo "Found GPU processes: $gpu_processes"
            for pid in $gpu_processes; do
                if [ -n "$pid" ] && [ "$pid" != "pid" ]; then
                    echo "Killing GPU process: $pid"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
        fi
    fi
    sleep 2
}

# Function to run throughput benchmark
run_throughput_benchmark() {
    local tp_size=$1
    local input_len=$2
    local output_len=$3
    local batch_size=$4
    
    local result_file="${OUTPUT_DIR}/throughput_tp${tp_size}_in${input_len}_out${output_len}_bs${batch_size}.json"
    local log_file="${LOG_DIR}/throughput_tp${tp_size}_in${input_len}_out${output_len}_bs${batch_size}.log"
    
    echo "  Running throughput benchmark: TP=$tp_size, input=$input_len, output=$output_len, batch=$batch_size"
    echo "    Log file: $log_file"
    
    # Clean up before running
    cleanup_gpu_processes
    
    # Set CUDA_VISIBLE_DEVICES for this TP size
    if [ "$tp_size" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ "$tp_size" -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [ "$tp_size" -eq 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ "$tp_size" -eq 8 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    else
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((tp_size-1)))
    fi
    
    echo "    Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Run throughput benchmark
    vllm bench throughput \
        --model "$MODEL" \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --batch-size "$batch_size" \
        --num-prompts "$NUM_PROMPTS" \
        --tensor-parallel-size "$tp_size" \
        --output-json "$result_file" \
        --enforce-eager \
        --disable-custom-all-reduce \
        --disable-log-stats >"$log_file" 2>&1 || {
            echo "    ✗ Failed throughput benchmark for TP=$tp_size, input=$input_len, output=$output_len, batch=$batch_size"
            echo "    Check log file: $log_file"
            return 1
        }
    
    echo "    ✓ Completed throughput benchmark"
    return 0
}

# Function to run latency benchmark
run_latency_benchmark() {
    local tp_size=$1
    local input_len=$2
    local output_len=$3
    local batch_size=$4
    
    local result_file="${OUTPUT_DIR}/latency_tp${tp_size}_in${input_len}_out${output_len}_bs${batch_size}.json"
    local log_file="${LOG_DIR}/latency_tp${tp_size}_in${input_len}_out${output_len}_bs${batch_size}.log"
    
    echo "  Running latency benchmark: TP=$tp_size, input=$input_len, output=$output_len, batch=$batch_size"
    echo "    Log file: $log_file"
    
    # Clean up before running
    cleanup_gpu_processes
    
    # Set CUDA_VISIBLE_DEVICES for this TP size
    if [ "$tp_size" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ "$tp_size" -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [ "$tp_size" -eq 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ "$tp_size" -eq 8 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    else
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((tp_size-1)))
    fi
    
    echo "    Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Run latency benchmark
    vllm bench latency \
        --model "$MODEL" \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --batch-size "$batch_size" \
        --num-iters "$NUM_ITERS" \
        --num-iters-warmup "$NUM_WARMUP_ITERS" \
        --tensor-parallel-size "$tp_size" \
        --output-json "$result_file" \
        --enforce-eager \
        --disable-custom-all-reduce \
        --disable-log-stats >"$log_file" 2>&1 || {
            echo "    ✗ Failed latency benchmark for TP=$tp_size, input=$input_len, output=$output_len, batch=$batch_size"
            echo "    Check log file: $log_file"
            return 1
        }
    
    echo "    ✓ Completed latency benchmark"
    return 0
}

# Function to show current system status
show_system_status() {
    echo "=== System Status ==="
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory Usage:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
            echo "  GPU $line"
        done
    fi
    echo ""
}

# Main benchmark execution
echo "=== Starting Benchmarks ==="
total_configs=$((${#TP_SIZES[@]} * ${#INPUT_LENGTHS[@]} * ${#OUTPUT_LENGTHS[@]} * ${#BATCH_SIZES[@]}))
current_config=0
failed_configs=0
successful_configs=0

for tp_size in "${TP_SIZES[@]}"; do
    echo ""
    echo "--- Testing TP Size: $tp_size ---"
    show_system_status
    
    for input_len in "${INPUT_LENGTHS[@]}"; do
        for output_len in "${OUTPUT_LENGTHS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                current_config=$((current_config + 1))
                echo ""
                echo "[$current_config/$total_configs] Configuration: TP=$tp_size, Input=$input_len, Output=$output_len, Batch=$batch_size"
                
                # Run throughput benchmark
                if run_throughput_benchmark "$tp_size" "$input_len" "$output_len" "$batch_size"; then
                    throughput_success=true
                else
                    throughput_success=false
                    failed_configs=$((failed_configs + 1))
                fi
                
                # Run latency benchmark
                if run_latency_benchmark "$tp_size" "$input_len" "$output_len" "$batch_size"; then
                    latency_success=true
                else
                    latency_success=false
                    failed_configs=$((failed_configs + 1))
                fi
                
                if [ "$throughput_success" = true ] && [ "$latency_success" = true ]; then
                    successful_configs=$((successful_configs + 1))
                    echo "  ✓ All benchmarks completed successfully"
                else
                    echo "  ⚠ Some benchmarks failed"
                fi
                
                # Brief pause between configurations
                sleep 1
            done
        done
    done
done

# Final cleanup
cleanup_gpu_processes

# Generate summary report
echo ""
echo "=== Generating Summary Report ==="

# Create a comprehensive summary
SUMMARY_FILE="${OUTPUT_DIR}/tp_benchmark_summary.txt"
cat > "$SUMMARY_FILE" << EOF
vLLM Tensor Parallel Benchmark Summary
======================================

Benchmark Details:
- Model: $MODEL
- TP Sizes Tested: ${TP_SIZES[*]}
- Input Lengths: ${INPUT_LENGTHS[*]}
- Output Lengths: ${OUTPUT_LENGTHS[*]}
- Batch Sizes: ${BATCH_SIZES[*]}
- Number of Prompts: $NUM_PROMPTS
- Number of Iterations: $NUM_ITERS

Results Summary:
- Total Configurations: $total_configs
- Successful Configurations: $successful_configs
- Failed Configurations: $failed_configs
- Success Rate: $(( successful_configs * 100 / total_configs ))%

File Locations:
- Results Directory: $OUTPUT_DIR
- Log Directory: $LOG_DIR
- Summary File: $SUMMARY_FILE

EOF

# Add detailed results if jq is available
if command -v jq >/dev/null 2>&1; then
    echo "" >> "$SUMMARY_FILE"
    echo "Detailed Performance Results:" >> "$SUMMARY_FILE"
    echo "=============================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Throughput results
    echo "Throughput Results (tokens/s):" >> "$SUMMARY_FILE"
    echo "TP_Size | Input | Output | Batch | Throughput | Total_Time" >> "$SUMMARY_FILE"
    echo "--------|-------|--------|-------|------------|----------" >> "$SUMMARY_FILE" 
    
    for file in "$OUTPUT_DIR"/throughput_*.json; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .json)
            tp_size=$(echo "$filename" | sed 's/.*tp\([0-9]*\)_.*/\1/')
            input_len=$(echo "$filename" | sed 's/.*in\([0-9]*\)_.*/\1/')
            output_len=$(echo "$filename" | sed 's/.*out\([0-9]*\)_.*/\1/')
            batch_size=$(echo "$filename" | sed 's/.*bs\([0-9]*\).*/\1/')
            
            throughput=$(jq -r '.throughput_output_token // "N/A"' "$file" 2>/dev/null)
            total_time=$(jq -r '.elapsed_time // "N/A"' "$file" 2>/dev/null)
            
            printf "%-7s | %-5s | %-6s | %-5s | %-10s | %-10s\n" "$tp_size" "$input_len" "$output_len" "$batch_size" "$throughput" "$total_time" >> "$SUMMARY_FILE"
        fi
    done
    
    echo "" >> "$SUMMARY_FILE"
    
    # Latency results
    echo "Latency Results:" >> "$SUMMARY_FILE"
    echo "TP_Size | Input | Output | Batch | TTFT(ms) | TPOT(ms) | E2E(ms)" >> "$SUMMARY_FILE"
    echo "--------|-------|--------|-------|----------|----------|--------" >> "$SUMMARY_FILE"
    
    for file in "$OUTPUT_DIR"/latency_*.json; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .json)
            tp_size=$(echo "$filename" | sed 's/.*tp\([0-9]*\)_.*/\1/')
            input_len=$(echo "$filename" | sed 's/.*in\([0-9]*\)_.*/\1/')
            output_len=$(echo "$filename" | sed 's/.*out\([0-9]*\)_.*/\1/')
            batch_size=$(echo "$filename" | sed 's/.*bs\([0-9]*\).*/\1/')
            
            ttft=$(jq -r '.ttft_s.mean // "N/A"' "$file" 2>/dev/null | awk '{printf "%.1f", $1*1000}')
            tpot=$(jq -r '.tpot_s.mean // "N/A"' "$file" 2>/dev/null | awk '{printf "%.1f", $1*1000}')
            e2e=$(jq -r '.end_to_end_latency_s.mean // "N/A"' "$file" 2>/dev/null | awk '{printf "%.1f", $1*1000}')
            
            printf "%-7s | %-5s | %-6s | %-5s | %-8s | %-8s | %-7s\n" "$tp_size" "$input_len" "$output_len" "$batch_size" "$ttft" "$tpot" "$e2e" >> "$SUMMARY_FILE"
        fi
    done
fi

echo "Summary report saved to: $SUMMARY_FILE"
echo ""

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "=== Benchmark Completed ==="
echo "Results saved in: $OUTPUT_DIR"
echo "Logs saved in: $LOG_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "Analysis commands:"
echo "  cat $SUMMARY_FILE"
echo "  ls -la $OUTPUT_DIR/*.json"
echo "  ls -la $LOG_DIR/*.log"
echo ""

if [ $failed_configs -gt 0 ]; then
    echo "⚠ Warning: $failed_configs configurations failed"
    echo "Check log files in $LOG_DIR for details"
else
    echo "✓ All configurations completed successfully!"
fi

echo ""
echo "Next steps:"
echo "1. Analyze the throughput vs TP size scaling"
echo "2. Compare latency improvements across different TP configurations"
echo "3. Check GPU utilization logs for efficiency analysis"
echo "4. Consider memory usage patterns in the logs"