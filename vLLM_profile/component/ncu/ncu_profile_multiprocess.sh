#!/bin/bash

# Nsight Compute (Ncu) Profile Script
# 用于分析 benchmark_qwen3_mlp.py 的 GPU 性能


echo "=== 使用 Nsight Compute (Ncu) Profile benchmark_qwen3_mlp.py ==="
echo ""

export CUDA_VISIBLE_DEVICES=1

# BATCH_SIZE=${1:-8}
# KV_LEN=${2:-2048}
SEQ_LEN=1
MODEL_SIZE=${2:-32B}
OUTPUT_DIR=${3:-ncu_profile_result_v4_${MODEL_SIZE}_multiprocess2_rep4}

mkdir -p ${OUTPUT_DIR}

# OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}"
SECTION="SpeedOfLight"

WARMUP_ITERATIONS=${4:-0}
BENCHMARK_ITERATIONS=${5:-1}

# BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
BATCH_SIZES=(2)
KV_LENS=(1024)

for BATCH_SIZE in ${BATCH_SIZES[@]}; do
    for KV_LEN in ${KV_LENS[@]}; do
        OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}_model${MODEL_SIZE}"
        echo "NCU Output file: ${OUTPUT_FILE}"
        ncu \
            --target-processes all \
            --mps client \
            python3 benchmark_flash_attn.py \
            --batch-sizes ${BATCH_SIZE} \
            --kv-lens ${KV_LEN} ${KV_LEN} \
            --output-dir ${OUTPUT_FILE}_run_output \
            --warmup-iterations ${WARMUP_ITERATIONS} \
            --benchmark-iterations ${BENCHMARK_ITERATIONS} \
            --model-size ${MODEL_SIZE} \
            --num-processes 2
    done
done