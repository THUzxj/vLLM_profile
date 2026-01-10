MODEL=/nfs/xjzhang/Qwen/Qwen3-4B
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`
export CUDA_VISIBLE_DEVICES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python3 benchmark/profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 33000 \
    --batch-sizes 1 2 4 8 16 32 64 128 \
    --prompt-length 512 1024 2048 4096 8192 16384 \
    --output-len 128 \
    --window-size 200 \
    --num-repeat 1 \
    --log-dir results/vllm_step_${MODEL_NAME}_1proc_${DATE} \
    --kv-cache-memory-bytes 10737418240 \
    --max-model-len 17000

    --gpu-memory-utilization 0.45 \

python3 benchmark/profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 20000 \
    --batch-sizes 1 2 4 8 16 32 64 128 \
    --prompt-length 512 1024 2048 4096 8192 \
    --output-len 128 \
    --window-size 200 \
    --num-repeat 1 \
    --log-dir results/vllm_step_${MODEL_NAME}_2proc_${DATE} \
    --kv-cache-memory-bytes 4294967296 \
    --num-processes 2 \
    --max-model-len 10000
