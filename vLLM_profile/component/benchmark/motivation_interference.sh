#!/bin/bash

export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export CUDA_VISIBLE_DEVICES=2



for batch_size in 1 2 4; do

BATCH1_BATCH_SIZE=$batch_size
BATCH2_BATCH_SIZE=125
BATCH1_KV_LEN=16384
BATCH2_KV_LEN=16384
MODEL_SIZE=32B

OUTPUT_DIR="results/motivation_interference_results/len_${BATCH1_KV_LEN}_${BATCH2_KV_LEN}_bs_${BATCH1_BATCH_SIZE}_${BATCH2_BATCH_SIZE}/"


python benchmark/benchmark_flash_attn_mixed_batch.py \
 --batch-configs "${BATCH1_KV_LEN}:${BATCH1_BATCH_SIZE}" "${BATCH2_KV_LEN}:${BATCH2_BATCH_SIZE}"  "${BATCH1_KV_LEN}:${BATCH1_BATCH_SIZE} ${BATCH2_KV_LEN}:${BATCH2_BATCH_SIZE}" \
 --output-dir ${OUTPUT_DIR}


python3 benchmark/benchmark_qwen3_mlp.py --batch-sizes ${BATCH1_BATCH_SIZE} \
      --seq-lens 1 --output-dir ${OUTPUT_DIR} \
      --model-size ${MODEL_SIZE} --num-processes 1 --skip-plot

python3 benchmark/benchmark_qwen3_mlp.py --batch-sizes ${BATCH2_BATCH_SIZE} \
      --seq-lens 1 --output-dir ${OUTPUT_DIR} \
      --model-size ${MODEL_SIZE} --num-processes 1 --skip-plot

python3 benchmark/benchmark_qwen3_mlp.py --batch-sizes `expr ${BATCH1_BATCH_SIZE} + ${BATCH2_BATCH_SIZE}` \
      --seq-lens 1 --output-dir ${OUTPUT_DIR} \
      --model-size ${MODEL_SIZE} --num-processes 1 --skip-plot
done