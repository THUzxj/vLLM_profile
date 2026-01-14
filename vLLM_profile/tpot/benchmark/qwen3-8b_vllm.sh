MODEL=/nfs/xjzhang/Qwen/Qwen3-32B-1layer
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`
export CUDA_VISIBLE_DEVICES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python3 benchmark/profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 200000 \
    --batch-sizes 1 2 4 8 16 32 64 128 \
    --prompt-length 512 1024 2048 4096 8192 16384 \
    --output-len 128 \
    --window-size 200 \
    --num-repeat 3 \
    --log-dir results/vllm_step_${MODEL_NAME}_v1_${DATE} \
    --gpu-memory-utilization 0.95 \
    --custom-model
