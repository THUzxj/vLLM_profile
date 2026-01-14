
set -e
export MPSDIR=/data/xjzhang/vLLM_profile_v1/vLLM_profile/component/mps_files_2
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log

# NO MPS, MPS
MODE="mps"

# "0.6B" "4B"
for model_size in "32B"; do
    OUTPUT_DIR="sweep_profile_results_0112/${model_size}_num_processes_2_mode_${MODE}_rep2"
    mkdir -p ${OUTPUT_DIR}

    python benchmark/benchmark_flash_attn_v2.py --batch-sizes 1 2 4 8 16 32 64 128 256 \
     --kv-lens 512 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR} \
     --model-size ${model_size} --num-processes 2
    python benchmark/benchmark_qwen3_mlp.py --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
      --seq-lens 1 --output-dir ${OUTPUT_DIR} \
      --model-size ${model_size} --num-processes 2
    break
done

# for model_size in "4B" "32B"; do
#     OUTPUT_DIR="sweep_profile_results_0112/${model_size}_num_processes_1_mode_${MODE}_rep2"
#     mkdir -p ${OUTPUT_DIR}

#     python benchmark/benchmark_flash_attn_v2.py --batch-sizes 1 2 4 8 16 32 64 128 256 \
#      --kv-lens 512 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR} \
#      --model-size ${model_size} --num-processes 1
#     python benchmark/benchmark_qwen3_mlp.py --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
#       --seq-lens 1 --output-dir ${OUTPUT_DIR} \
#       --model-size ${model_size} --num-processes 1
# done
