MODEL=/nfs/xjzhang/Qwen/Qwen3-32B-1layer
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=0
MODE="nomps"

python3 benchmark/profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 100000 \
    --batch-sizes 1 2 4 8 16 32 64 128 \
    --prompt-length 512 1024 2048 4096 8192 \
    --output-len 128 \
    --window-size 200 \
    --num-repeat 1 \
    --log-dir results/vllm_step_${MODEL_NAME}_${MODE}_1proc_${DATE} \
    --kv-cache-memory-bytes 1073741824 \
    --max-model-len 17000 \
    --custom-model \
    --max-num-seqs 128

python3 benchmark/profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 50000 \
    --batch-sizes 1 2 4 8 16 32 64 128 \
    --prompt-length 512 1024 2048 4096 8192 \
    --output-len 128 \
    --window-size 200 \
    --num-repeat 1 \
    --log-dir results/vllm_step_${MODEL_NAME}_${MODE}_2proc_${DATE} \
    --kv-cache-memory-bytes 429496729 \
    --num-processes 2 \
    --max-model-len 10000 \
    --custom-model \
    --max-num-seqs 128
