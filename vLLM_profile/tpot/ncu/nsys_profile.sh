

#!/bin/bash

# Nsight Compute (Ncu) Profile Script
export CUDA_VISIBLE_DEVICES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL=/nfs/xjzhang/Qwen/Qwen3-4B-1layer
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`
set -e

# BATCH_SIZE=${1:-8}
# KV_LEN=${2:-2048}
SEQ_LEN=1
OUTPUT_DIR=${2:-ncu_profile_result_${MODEL_NAME}_rep0}

mkdir -p ${OUTPUT_DIR}

# OUTPUT_FILE="${OUTPUT_DIR}/ncu_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}"

WARMUP_ITERATIONS=${4:-0}
BENCHMARK_ITERATIONS=${5:-1}

batched_tokens=500000
kv_cache_usage=$((batched_tokens * 4 * 1024 + 64 * 1024))
echo "KV Cache Memory Bytes: ${kv_cache_usage}"


BATCH_SIZES=(1 2 4 8 16 32 64 128)
KV_LENS=(1024 2048 4096 8192)

for BATCH_SIZE in ${BATCH_SIZES[@]}; do
    for KV_LEN in ${KV_LENS[@]}; do
        OUTPUT_FILE="${OUTPUT_DIR}/nsys_report_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}_model${MODEL_NAME}"
        nsys profile \
            -o ${OUTPUT_FILE} \
            python benchmark/profile_vllm_cli.py \
                --model $MODEL \
                --batched-tokens $batched_tokens \
                --batch-sizes $BATCH_SIZE \
                --prompt-length $KV_LEN \
                --output-len 4 \
                --window-size 200 \
                --num-repeat 1 \
                --log-dir results_${trial_id}/vllm_step_${MODEL_NAME}_bs${BATCH_SIZE}_in${KV_LEN}_${DATE} \
                --kv-cache-memory-bytes $kv_cache_usage \
                --max-model-len 8210 \
                --max-num-seqs 128 \
                --custom-model \
                --enforce-eager
    done
done
