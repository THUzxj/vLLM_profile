#!/bin/bash

set -e

export MPSDIR=/data/xjzhang/vLLM_profile_v1/vLLM_profile/component/mps_files
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log
# MODE: mps nomps
MODE="mps"

if [ $MODE == "mps" ]; then
  export CUDA_VISIBLE_DEVICES=0
elif [ $MODE == "nomps" ]; then
  export CUDA_VISIBLE_DEVICES=1
else
    echo "Invalid mode: $MODE"
    exit 1
fi

# "0.6B" "4B"
for model_size in "32B"; do
    OUTPUT_DIR="sweep_profile_results_3/${model_size}_mode_${MODE}_rep4"
    mkdir -p ${OUTPUT_DIR}

    python benchmark/benchmark_flash_attn_v2.py --batch-sizes 1 2 4 8 16 32 64 128 256 --kv-lens 512 1024 2048 4096 8192 16384 32768 --output-dir ${OUTPUT_DIR} --model-size ${model_size}
    python benchmark/benchmark_qwen3_mlp.py --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 --seq-lens 1 --output-dir ${OUTPUT_DIR} --model-size ${model_size}
done
