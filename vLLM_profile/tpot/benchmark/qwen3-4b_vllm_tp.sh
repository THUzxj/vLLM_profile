MODEL=/nfs/xjzhang/Qwen/Qwen3-4B
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`

TP_SIZE=2
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# debug
python3 profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 150000 \
    --batch-sizes 1 2 4 8 16 32 64 128 \
    --prompt-length 8192 16384 \
    --output-len 128 \
    --window-size 200 \
    --num-repeat 3 \
    --log-dir vllm_step_${MODEL_NAME}_v1_tp${TP_SIZE}_${DATE} \
    --tp-size $TP_SIZE \
    --gpu-memory-utilization 0.95

# python3 profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 200000 \
#     --batch-sizes 1 2 4 8 16 32 64 128 \
#     --prompt-length 512 1024 2048 4096 8192 16384 \
#     --output-len 128 \
#     --window-size 200 \
#     --num-repeat 3 \
#     --log-dir vllm_step_${MODEL_NAME}_v1_tp${TP_SIZE}_${DATE} \
#     --tp-size $TP_SIZE \
#     --gpu-memory-utilization 0.95

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