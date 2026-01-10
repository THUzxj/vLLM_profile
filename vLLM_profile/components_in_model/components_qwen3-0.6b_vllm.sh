MODEL=/nfs/xjzhang/Qwen/Qwen3-0.6B
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`

export CUDA_VISIBLE_DEVICES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PROFILE_COMPONENT_MODEL=$MODEL_NAME
VERSION="v1_component"

# 256 * 1024 strange result

for batch_size in 1 2 4 8 16 32 64 128 256 512 1024; do
    for input_len in 128 256 512 1024 2048 4096 8192 16384; do
        export PROFILE_COMPONENT_BS=$batch_size
        export PROFILE_COMPONENT_IN=$input_len

        
        # skip if batched tokens exceed 100000
        BATCHED_TOKENS=$((PROFILE_COMPONENT_BS * PROFILE_COMPONENT_IN))
        if [ $BATCHED_TOKENS -gt 280000 ]; then
            echo "Skipping BS=${PROFILE_COMPONENT_BS}, IN=${PROFILE_COMPONENT_IN} as batched tokens ${BATCHED_TOKENS} exceed limit."
            continue
        fi

        # debug
        python3 profile_vllm_cli.py \
            --model $MODEL \
            --batched-tokens 280000 \
            --batch-sizes $PROFILE_COMPONENT_BS \
            --prompt-length $PROFILE_COMPONENT_IN \
            --output-len 16 \
            --window-size 200 \
            --num-repeat 1 \
            --log-dir vllm_step_${MODEL_NAME}_${VERSION}_bs${PROFILE_COMPONENT_BS}_in${PROFILE_COMPONENT_IN}_${DATE} \
            --gpu-memory-utilization 0.95 \
            --custom-model \
            --enforce-eager
    done
done

# python3 profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 100000 \
#     --batch-sizes 1 2 4 8 16 32 64 128 \
#     --prompt-length 512 1024 2048 4096 8192 16384 \
#     --output-len 128 \
#     --window-size 200 \
#     --num-repeat 3 \
#     --log-dir vllm_step_${MODEL_NAME}_v1_${DATE} \
#     --gpu-memory-utilization 0.95 \
#     --custom-model

# python3 profile_vllm_cli_v2.py \
#     --model $MODEL \
#     --batched-tokens 100000 \
#     --batch-sizes 1 4 \
#     --prompt-length 512 1024 2048 4096 8192 16384 \
#     --output-len 128 \
#     --window-size 200 \
#     --num-repeat 3 \
#     --log-dir vllm_step_${MODEL_NAME}_${DATE} \
#     --gpu-memory-utilization 0.95

# --batch-sizes 1 2 4 8 16 32 64 128 \

# python3 profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 33000 \
#     --batch-sizes 1 2 4 8 16 32 64 128 \
#     --prompt-length 512 1024 2048 4096 8192 16384 \
#     --output-len 128 \
#     --window-size 200 \
#     --num-repeat 1 \
#     --log-dir vllm_step_${MODEL_NAME}_1proc_${DATE} \
#     --kv-cache-memory-bytes 10737418240 \
#     --max-model-len 17000

#     --gpu-memory-utilization 0.45 \

# python3 profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 20000 \
#     --batch-sizes 1 2 4 8 16 32 64 128 \
#     --prompt-length 512 1024 2048 4096 8192 \
#     --output-len 128 \
#     --window-size 200 \
#     --num-repeat 1 \
#     --log-dir vllm_step_${MODEL_NAME}_2proc_${DATE} \
#     --kv-cache-memory-bytes 4294967296 \
#     --num-processes 2 \
#     --max-model-len 10000

    # --gpu-memory-utilization 0.48 \