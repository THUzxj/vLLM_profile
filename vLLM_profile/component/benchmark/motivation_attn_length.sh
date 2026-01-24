set -e

# NO MPS, MPS
MODE="mps"

# if [ $MODE == "mps" ]; then
#   export CUDA_VISIBLE_DEVICES=0
# elif [ $MODE == "nomps" ]; then
#   export CUDA_VISIBLE_DEVICES=1
# else
#     echo "Invalid mode: $MODE"
#     exit 1
# fi

MODEL_SIZE="32B"


for batch_size in 16; do
BATCH1_BATCH_SIZE=${batch_size}
BATCH2_BATCH_SIZE=${batch_size}

# BATCH1_KV_LEN=512
# BATCH2_KV_LEN=8192


# BATCH1_KV_LEN=4096
# BATCH2_KV_LEN=8192

# BATCH1_KV_LEN=8192
# BATCH2_KV_LEN=16384

BATCH1_KV_LEN=30720
BATCH2_KV_LEN=32768

# BATCH1_KV_LEN=20000
# BATCH2_KV_LEN=40000

OUTPUT_DIR="results/motivation_results_${MODE}_len_${BATCH1_KV_LEN}_${BATCH2_KV_LEN}_bs_${BATCH1_BATCH_SIZE}_${BATCH2_BATCH_SIZE}/"

YAML_CONFIG_PATH="./benchmark/flash_attn_v2_motivation_config_debug.yaml"


echo "Creating YAML config file at ${YAML_CONFIG_PATH}"
echo "
default:
  batch_sizes: [${BATCH1_BATCH_SIZE}, ${BATCH2_BATCH_SIZE}]
  kv_lens: [${BATCH1_KV_LEN}, ${BATCH2_KV_LEN}]
processes:
  - id: 0
    batch_sizes: [${BATCH1_BATCH_SIZE}]
    kv_lens: [${BATCH1_KV_LEN}]
  - id: 1
    batch_sizes: [${BATCH2_BATCH_SIZE}]
    kv_lens: [${BATCH2_KV_LEN}]
"> ${YAML_CONFIG_PATH}


export MPSDIR=/data/xjzhang/vLLM_profile_v1/vLLM_profile/component/mps_files_host
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log
export CUDA_VISIBLE_DEVICES=0
  

for proc1_gpu_percentage in 10 20 30 40 50 60 70 80 90 100; do
  proc2_gpu_percentage=$((100 - $proc1_gpu_percentage))


  if [ $proc1_gpu_percentage != 100 ]; then
    python3 ./benchmark/benchmark_flash_attn_v2.py \
    --num-processes 2 \
    --config-yaml ${YAML_CONFIG_PATH} \
    --output-dir ${OUTPUT_DIR}/parallel_${proc1_gpu_percentage}_${proc2_gpu_percentage}/  \
    --model-size ${MODEL_SIZE} \
    --process-gpu-percentage ${proc1_gpu_percentage} ${proc2_gpu_percentage} \
    --skip-plot
  fi

  python3 benchmark/benchmark_flash_attn_v2.py --batch-sizes ${BATCH1_BATCH_SIZE} \
      --kv-lens ${BATCH1_KV_LEN} --output-dir ${OUTPUT_DIR}/bs${BATCH1_BATCH_SIZE}_len${BATCH1_KV_LEN}_pct${proc1_gpu_percentage}/ \
      --model-size ${MODEL_SIZE} --num-processes 1 --process-gpu-percentage ${proc1_gpu_percentage} --skip-plot

  python3 benchmark/benchmark_flash_attn_v2.py --batch-sizes ${BATCH2_BATCH_SIZE} \
        --kv-lens ${BATCH2_KV_LEN} --output-dir ${OUTPUT_DIR}/bs${BATCH2_BATCH_SIZE}_len${BATCH2_KV_LEN}_pct${proc1_gpu_percentage}/ \
        --model-size ${MODEL_SIZE} --num-processes 1 --process-gpu-percentage ${proc1_gpu_percentage} --skip-plot
done

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100; python benchmark/benchmark_flash_attn_mixed_batch.py \
 --batch-configs "${BATCH1_KV_LEN}:${BATCH1_BATCH_SIZE}" "${BATCH2_KV_LEN}:${BATCH2_BATCH_SIZE}"  "${BATCH1_KV_LEN}:${BATCH1_BATCH_SIZE} ${BATCH2_KV_LEN}:${BATCH2_BATCH_SIZE}" \
 --output-dir ${OUTPUT_DIR}/combined_pct100/


export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export CUDA_VISIBLE_DEVICES=2

# for min_count in 10 20 30 40 50 60 70 80 90 100 106; do
#   export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${min_count}

#   python3 ./green_context/test_flash_attn_green_ctx.py \
#     --num-groups 1 \
#     --min-count "$min_count" \
#     --batch-size "$BATCH1_BATCH_SIZE" \
#     --kv-len "$BATCH1_KV_LEN" \
#     --model-size "$MODEL_SIZE" \
#     --output-csv ${OUTPUT_DIR}/green_context_bs${BATCH1_BATCH_SIZE}_len${BATCH1_KV_LEN}.csv
#   python3 ./green_context/test_flash_attn_green_ctx.py \
#     --num-groups 1 \
#     --min-count "$min_count" \
#     --batch-size "$BATCH2_BATCH_SIZE" \
#     --kv-len "$BATCH2_KV_LEN" \
#     --model-size "$MODEL_SIZE" \
#     --output-csv ${OUTPUT_DIR}/green_context_bs${BATCH2_BATCH_SIZE}_len${BATCH2_KV_LEN}.csv
# done


python3 benchmark/benchmark_flash_attn_v2.py --batch-sizes ${BATCH1_BATCH_SIZE} \
      --kv-lens ${BATCH1_KV_LEN} --output-dir ${OUTPUT_DIR}/bs${BATCH1_BATCH_SIZE}_len${BATCH1_KV_LEN}_nomps/ \
      --model-size ${MODEL_SIZE} --num-processes 1 --skip-plot

python3 benchmark/benchmark_flash_attn_v2.py --batch-sizes ${BATCH2_BATCH_SIZE} \
      --kv-lens ${BATCH2_KV_LEN} --output-dir ${OUTPUT_DIR}/bs${BATCH2_BATCH_SIZE}_len${BATCH2_KV_LEN}_nomps/ \
      --model-size ${MODEL_SIZE} --num-processes 1 --skip-plot
done

done
