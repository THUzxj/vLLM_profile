#!/bin/bash

# Sweep script for testing Qwen3MLP parallel execution with different parameters
# This script iterates over different combinations of batch_size, seq_len, and num_groups

export CUDA_VISIBLE_DEVICES=2

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="test_flash_attn_green_ctx.py"

MODEL_SIZE="32B"
MIN_COUNT=106
ID="_3"

OUTPUT_DIR="sweep_results_fa"
OUTPUT_CSV_SINGLE="${SCRIPT_DIR}/${OUTPUT_DIR}/sweep_results_model${MODEL_SIZE}_single.csv"
OUTPUT_CSV="${SCRIPT_DIR}/${OUTPUT_DIR}/sweep_results_model${MODEL_SIZE}_SM${MIN_COUNT}${ID}.csv"
DEVICE="cuda:0"
DTYPE="bfloat16"
WARMUP_ITERATIONS=10
BENCHMARK_ITERATIONS=100

# Parameter arrays to sweep
BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
KV_LENS=(512 1024 2048 4096 8192)
NUM_GROUPS=(1)


# BATCH_SIZES=(2)
# KV_LENS=(1024)
# NUM_GROUPS=(1)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --min-count)
            MIN_COUNT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --warmup-iterations)
            WARMUP_ITERATIONS="$2"
            shift 2
            ;;
        --benchmark-iterations)
            BENCHMARK_ITERATIONS="$2"
            shift 2
            ;;
        --batch-sizes)
            # Parse comma-separated values or space-separated values
            shift
            BATCH_SIZES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                BATCH_SIZES+=("$1")
                shift
            done
            ;;
        --seq-lens)
            shift
            KV_LENS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                KV_LENS+=("$1")
                shift
            done
            ;;
        --num-groups)
            shift
            NUM_GROUPS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                NUM_GROUPS+=("$1")
                shift
            done
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-csv PATH          Output CSV file path (default: sweep_results.csv)"
            echo "  --model-size SIZE          Model size: 0.6B, 4B, 14B, 32B (default: 4B)"
            echo "  --min-count COUNT          Minimum SM count per group (default: 16)"
            echo "  --device DEVICE            CUDA device (default: cuda:0)"
            echo "  --dtype DTYPE              Data type: bfloat16, float16, float32 (default: bfloat16)"
            echo "  --warmup-iterations N      Number of warmup iterations (default: 10)"
            echo "  --benchmark-iterations N   Number of benchmark iterations (default: 100)"
            echo "  --batch-sizes SIZE ...     Batch sizes to test (default: 1 2 4 8 16 32)"
            echo "  --seq-lens LEN ...        Sequence lengths to test (default: 1 128 256 512 1024)"
            echo "  --num-groups N ...        Number of groups to test (default: 1 2 3)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --output-csv results.csv --batch-sizes 1 2 4 --seq-lens 1 128 --num-groups 1 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if script exists
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Calculate total number of combinations
TOTAL_COMBINATIONS=$((${#BATCH_SIZES[@]} * ${#KV_LENS[@]} * ${#NUM_GROUPS[@]}))
CURRENT_COMBINATION=0

# Print configuration
echo "=========================================="
echo "Qwen3MLP Parallel Stream Sweep"
echo "=========================================="
echo "Script: $SCRIPT_PATH"
echo "Output CSV: $OUTPUT_CSV"
echo "Model size: $MODEL_SIZE"
echo "Min SM count: $MIN_COUNT"
echo "Device: $DEVICE"
echo "Data type: $DTYPE"
echo "Warmup iterations: $WARMUP_ITERATIONS"
echo "Benchmark iterations: $BENCHMARK_ITERATIONS"
echo ""
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Sequence lengths: ${KV_LENS[@]}"
echo "Number of groups: ${NUM_GROUPS[@]}"
echo ""
echo "Total combinations: $TOTAL_COMBINATIONS"
echo "=========================================="
echo ""

# Remove existing CSV file if it exists (to start fresh)
if [[ -f "$OUTPUT_CSV" ]]; then
    echo "Warning: Output CSV file exists: $OUTPUT_CSV"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$OUTPUT_CSV"
        echo "Removed existing CSV file."
    else
        echo "Appending to existing CSV file."
    fi
fi

# Start time
START_TIME=$(date +%s)

# Sweep over all parameter combinations
for batch_size in "${BATCH_SIZES[@]}"; do
    for kv_len in "${KV_LENS[@]}"; do
        for num_groups in "${NUM_GROUPS[@]}"; do
            CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
            
            echo "[$CURRENT_COMBINATION/$TOTAL_COMBINATIONS] Testing: "
            echo "  batch_size=$batch_size, kv_len=$kv_len, num_groups=$num_groups"
            
            # Run the test
            python "$SCRIPT_PATH" \
                --num-groups "$num_groups" \
                --min-count "$MIN_COUNT" \
                --batch-size "$batch_size" \
                --kv-len "$kv_len" \
                --model-size "$MODEL_SIZE" \
                --device "$DEVICE" \
                --dtype "$DTYPE" \
                --warmup-iterations "$WARMUP_ITERATIONS" \
                --benchmark-iterations "$BENCHMARK_ITERATIONS" \
                --output-csv "$OUTPUT_CSV"

            # python "$SCRIPT_PATH" \
            #     --num-groups "$num_groups" \
            #     --min-count "$MIN_COUNT" \
            #     --batch-size "$batch_size" \
            #     --seq-len "$seq_len" \
            #     --model-size "$MODEL_SIZE" \
            #     --device "$DEVICE" \
            #     --dtype "$DTYPE" \
            #     --warmup-iterations "$WARMUP_ITERATIONS" \
            #     --benchmark-iterations "$BENCHMARK_ITERATIONS" \
            #     --output-csv "$OUTPUT_CSV_SINGLE" \
            #     --single-stream
            
            # Check if the command succeeded
            if [[ $? -ne 0 ]]; then
                echo "ERROR: Test failed for batch_size=$batch_size, seq_len=$seq_len, num_groups=$num_groups"
                echo "Continuing with next combination..."
            else
                echo "  ✓ Completed successfully"
            fi
            
            echo ""
        done
    done
done

# End time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED_TIME / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "=========================================="
echo "Sweep completed!"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="

