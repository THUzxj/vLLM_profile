#!/usr/bin/env bash
set -euo pipefail

# Sweep run_model_timing.py across batch sizes and input lengths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Core settings (can be overridden via env)
MODEL_NAME="${MODEL_NAME:-/nfs/xjzhang/Qwen/Qwen3-4B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
NUM_RUNS="${NUM_RUNS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/timing_results_attn_${ATTN_IMPLEMENTATION}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Sweeps (space-separated lists, overridable via env)
DEFAULT_BATCH_SIZES="1 2 4 8 16 32 64 128"
DEFAULT_INPUT_LENS="128 256 512 1024 2048"

read -r -a BATCH_SIZES <<< "${BATCH_SIZES:-$DEFAULT_BATCH_SIZES}"
read -r -a INPUT_LENS <<< "${INPUT_LENS:-$DEFAULT_INPUT_LENS}"

mkdir -p "${OUTPUT_DIR}"

# for bs in "${BATCH_SIZES[@]}"; do
#   for il in "${INPUT_LENS[@]}"; do
#     echo "Running batch_size=${bs}, input_len=${il}"
#     "${PYTHON_BIN}" "${SCRIPT_DIR}/run_model_timing.py" \
#       --batch-size "${bs}" \
#       --input-len "${il}" \
#       --model-name "${MODEL_NAME}" \
#       --device "${DEVICE}" \
#       --dtype "${DTYPE}" \
#       --num-runs "${NUM_RUNS}" \
#       --warmup-runs "${WARMUP_RUNS}" \
#       --output-dir "${OUTPUT_DIR}/bs${bs}_il${il}"
#   done
# done

for bs in "${BATCH_SIZES[@]}"; do
  for il in "${INPUT_LENS[@]}"; do
    echo "Running batch_size=${bs}, input_len=${il}, attn_implementation=flash_attention_2"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_model_timing.py" \
      --batch-size "${bs}" \
      --input-len "${il}" \
      --model-name "${MODEL_NAME}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --num-runs "${NUM_RUNS}" \
      --warmup-runs "${WARMUP_RUNS}" \
      --output-dir "${OUTPUT_DIR}/bs${bs}_il${il}" \
      --attn-implementation "flash_attention_2"
  done
done

