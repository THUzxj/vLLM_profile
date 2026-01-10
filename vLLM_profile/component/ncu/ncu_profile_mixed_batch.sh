#!/bin/bash

# Nsight Compute (Ncu) Profile Script
# 用于分析 benchmark_qwen3_mlp.py 的 GPU 性能

echo "=== 使用 Nsight Compute (Ncu) ==="
echo ""

export CUDA_VISIBLE_DEVICES=1

# BATCH_SIZE=${1:-8}
# KV_LEN=${2:-2048}
SEQ_LEN=1
MODEL_SIZE=${2:-4B}
OUTPUT_DIR=${3:-ncu_profile_result_v4_${MODEL_SIZE}_mixed_batch}

mkdir -p ${OUTPUT_DIR}

# OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}"
SECTION="SpeedOfLight"

WARMUP_ITERATIONS=${4:-0}
BENCHMARK_ITERATIONS=${5:-1}

BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
KV_LENS=(1024 2048 4096 8192)

BATCH1="8192:4"
BATCH2="512:64"


# replace : to _ in BATCH1 and BATCH2
BATCH1_NAME=$(echo ${BATCH1} | tr ':' '_')
BATCH2_NAME=$(echo ${BATCH2} | tr ':' '_')

# OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_${BATCH1_NAME}_${BATCH2_NAME}_model${MODEL_SIZE}"
# ncu \
#     --set full \
#     --section "${SECTION}" \
#     --section "MemoryWorkloadAnalysis" \
#     --section "ComputeWorkloadAnalysis" \
#     --section "PmSampling" \
#     --section "MemoryWorkloadAnalysis_Chart" \
#     --section "MemoryWorkloadAnalysis_Tables" \
#     --section "SourceCounters" \
#     --section "SchedulerStats" \
#     --target-processes all \
#     -o ${OUTPUT_FILE} \
#     --force-overwrite \
#     python3 benchmark_flash_attn_mixed_batch.py \
#     --batch-configs "${BATCH1} ${BATCH2}" \
#     --output-dir ${OUTPUT_FILE}_run_output \
#     --warmup-iterations ${WARMUP_ITERATIONS} \
#     --benchmark-iterations ${BENCHMARK_ITERATIONS} \
#     --model-size ${MODEL_SIZE}

OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_${BATCH1_NAME}_model${MODEL_SIZE}"
ncu \
    --set full \
    --section "${SECTION}" \
    --section "MemoryWorkloadAnalysis" \
    --section "ComputeWorkloadAnalysis" \
    --section "PmSampling" \
    --section "MemoryWorkloadAnalysis_Chart" \
    --section "MemoryWorkloadAnalysis_Tables" \
    --section "SourceCounters" \
    --section "SchedulerStats" \
    --target-processes all \
    -o ${OUTPUT_FILE} \
    --force-overwrite \
    python3 benchmark_flash_attn_mixed_batch.py \
    --batch-configs "${BATCH1}" \
    --output-dir ${OUTPUT_FILE}_run_output \
    --warmup-iterations ${WARMUP_ITERATIONS} \
    --benchmark-iterations ${BENCHMARK_ITERATIONS} \
    --model-size ${MODEL_SIZE}

OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_${BATCH2_NAME}_model${MODEL_SIZE}"
ncu \
    --set full \
    --section "${SECTION}" \
    --section "MemoryWorkloadAnalysis" \
    --section "ComputeWorkloadAnalysis" \
    --section "PmSampling" \
    --section "MemoryWorkloadAnalysis_Chart" \
    --section "MemoryWorkloadAnalysis_Tables" \
    --section "SourceCounters" \
    --section "SchedulerStats" \
    --target-processes all \
    -o ${OUTPUT_FILE} \
    --force-overwrite \
    python3 benchmark_flash_attn_mixed_batch.py \
    --batch-configs "${BATCH2}" \
    --output-dir ${OUTPUT_FILE}_run_output \
    --warmup-iterations ${WARMUP_ITERATIONS} \
    --benchmark-iterations ${BENCHMARK_ITERATIONS} \
    --model-size ${MODEL_SIZE}
