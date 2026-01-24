
set -e
export MPSDIR=/xingjian/vLLM_profile_v1/vLLM_profile/component/mps_files_host
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log

# NO MPS, MPS
MODE="mps"

if [ $MODE == "mps" ]; then
  export CUDA_VISIBLE_DEVICES=0
elif [ $MODE == "nomps" ]; then
  export CUDA_VISIBLE_DEVICES=1
else
    echo "Invalid mode: $MODE"
    exit 1
fi

RUN_PROCESSES="2"

if [ $RUN_PROCESSES == "2" ] || [ $RUN_PROCESSES == "1 2" ]; then
  # "0.6B" "4B"
  for model_size in "32B"; do
      OUTPUT_DIR="sweep_profile_results_0124/${model_size}_num_processes_2_mode_${MODE}_rep7"
      mkdir -p ${OUTPUT_DIR}

      # python benchmark/benchmark_flash_attn_v2.py --batch-sizes 1 2 4 8 16 32 64 128 256 \
      # --kv-lens 512 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR} \
      # --model-size ${model_size} --num-processes 2 --process-gpu-percentage 50 50
      python3 benchmark/benchmark_qwen3_mlp.py --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
        --seq-lens 1 --output-dir ${OUTPUT_DIR} \
        --model-size ${model_size} --num-processes 2 --process-gpu-percentage 50 50
  done

elif [ $RUN_PROCESSES == "1" ] || [ $RUN_PROCESSES == "1 2" ]; then
  for model_size in "32B"; do


    for process_gpu_percentage in "50" "100"; do
      OUTPUT_DIR="sweep_profile_results_0124/${model_size}_num_processes_1_mode_${MODE}_process_gpu_percentage_${process_gpu_percentage}"
      mkdir -p ${OUTPUT_DIR}

      # python benchmark/benchmark_flash_attn_v2.py --batch-sizes 1 2 4 8 16 32 64 128 256 \
      # --kv-lens 512 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR} \
      # --model-size ${model_size} --num-processes 1 --process-gpu-percentage ${process_gpu_percentage}
      python3 benchmark/benchmark_qwen3_mlp.py --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
        --seq-lens 1 --output-dir ${OUTPUT_DIR} \
        --model-size ${model_size} --num-processes 1 --process-gpu-percentage ${process_gpu_percentage}
    done
  done
fi