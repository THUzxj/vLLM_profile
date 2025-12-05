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
    --num-repeat 1 \
    --log-dir vllm_step_${MODEL_NAME}_${DATE} \
    --gpu-memory-utilization 0.95
