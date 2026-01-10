#!/bin/bash

# Nsight Compute (Ncu) Profile Script
# 用于分析 benchmark_qwen3_mlp.py 的 GPU 性能

echo "=== 使用 Nsight Compute (Ncu) Profile benchmark_qwen3_mlp.py ==="
echo ""
set -e
export CUDA_VISIBLE_DEVICES=1

# BATCH_SIZE=${1:-8}
# KV_LEN=${2:-2048}
SEQ_LEN=1
MODEL_SIZE=${1:-32B}
OUTPUT_DIR=${2:-ncu_profile_result/${MODEL_SIZE}_rep0}

mkdir -p ${OUTPUT_DIR}

# OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}"
SECTION="SpeedOfLight"

WARMUP_ITERATIONS=${4:-0}
BENCHMARK_ITERATIONS=${5:-1}

BATCH_SIZES=(1 2 4 8 16 32 64 128 256 512 768 1024)
# BATCH_SIZES=(512 768 1024)
KV_LENS=(1024 2048 4096 8192)

for BATCH_SIZE in ${BATCH_SIZES[@]}; do
    for KV_LEN in ${KV_LENS[@]}; do
        OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}_model${MODEL_SIZE}"
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
            --kernel-name "flash_fwd_splitkv_kernel" \
            --target-processes all \
            -o ${OUTPUT_FILE} \
            --force-overwrite \
            python3 benchmark/benchmark_flash_attn_v2.py \
            --batch-sizes ${BATCH_SIZE} \
            --kv-lens ${KV_LEN} \
            --output-dir ${OUTPUT_FILE}_run_output \
            --warmup-iterations ${WARMUP_ITERATIONS} \
            --benchmark-iterations ${BENCHMARK_ITERATIONS} \
            --model-size ${MODEL_SIZE} \
            --skip-plots
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

