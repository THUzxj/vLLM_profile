MODEL=/nfs/xjzhang/Qwen/Qwen3-0.6B
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`

python3 profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 280000 \
    --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 \
    --prompt-length 128 256 512 1024 2048 4096 8192 16384 \
    --output-len 64 \
    --window-size 128 \
    --num-repeat 3 \
    --log-dir vllm_step_${MODEL_NAME}_v1_${DATE} \
    --gpu-memory-utilization 0.95

# python3 profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 80000 \
#     --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 \
#     --prompt-length 128 256 512 1024 2048 4096 8192 16384 \
#     --output-len 64 \
#     --window-size 128 \
#     --num-repeat 3 \
#     --log-dir vllm_step_${MODEL_NAME}_1proc_${DATE} \
#     --kv-cache-memory-bytes 10737418240 \
#     --max-model-len 17000

# python3 profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 80000 \
#     --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024\
#     --prompt-length 128 256 512 1024 2048 4096 8192 16384 \
#     --output-len 64 \
#     --window-size 128 \
#     --num-repeat 3 \
#     --log-dir vllm_step_${MODEL_NAME}_2proc_${DATE} \
#     --kv-cache-memory-bytes 10737418240 \
#     --num-processes 2 \
#     --max-model-len 17000
