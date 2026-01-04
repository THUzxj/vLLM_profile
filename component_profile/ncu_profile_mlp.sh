#!/bin/bash

# Nsight Compute (Ncu) Profile Script
# 用于分析 benchmark_qwen3_mlp.py 的 GPU 性能

echo "=== 使用 Nsight Compute (Ncu) Profile benchmark_qwen3_mlp.py ==="
echo ""

export CUDA_VISIBLE_DEVICES=0

# BATCH_SIZE=${1:-8}
# KV_LEN=${2:-2048}
SEQ_LEN=1
MODEL_SIZE=${2:-4B}
OUTPUT_DIR=${3:-ncu_profile_result_${MODEL_SIZE}}

mkdir -p ${OUTPUT_DIR}

# OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}"
SECTION="SpeedOfLight"

WARMUP_ITERATIONS=${4:-1}
BENCHMARK_ITERATIONS=${5:-1}

BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
SEQ_LENS=(1)

for BATCH_SIZE in ${BATCH_SIZES[@]}; do
    for SEQ_LEN in ${SEQ_LENS[@]}; do
        OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_qwen3_mlp_bs${BATCH_SIZE}_seq${SEQ_LEN}_model${MODEL_SIZE}"
        ncu \
            --set default \
            --section "${SECTION}" \
            --section "MemoryWorkloadAnalysis" \
            --target-processes all \
            -o ${OUTPUT_FILE} \
            --force-overwrite \
            python3 benchmark_qwen3_mlp.py \
            --batch-sizes ${BATCH_SIZE} \
            --seq-lens ${SEQ_LEN} \
            --output-dir ${OUTPUT_DIR} \
            --warmup-iterations ${WARMUP_ITERATIONS} \
            --benchmark-iterations ${BENCHMARK_ITERATIONS} \
            --model-size ${MODEL_SIZE}
    done
done

# ncu \
#     --set default \
#     --section "${SECTION}" \
#     --section "MemoryWorkloadAnalysis" \
#     --target-processes all \
#     -o ${OUTPUT_FILE} \
#     --force-overwrite \
#     python3 benchmark_flash_attn.py \
#     --batch-sizes ${BATCH_SIZE} \
#     --kv-lens ${KV_LEN} \
#     --output-dir ${OUTPUT_DIR} \
#     --warmup-iterations ${WARMUP_ITERATIONS} \
#     --benchmark-iterations ${BENCHMARK_ITERATIONS}

