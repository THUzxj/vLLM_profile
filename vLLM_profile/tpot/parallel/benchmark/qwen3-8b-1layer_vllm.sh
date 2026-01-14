
set -e
MODEL=/nfs/xjzhang/Qwen/Qwen3-8B-1layer
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=0
MODE="mps"

# python3 benchmark/profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 350000 \
#     --batch-sizes 32 \
#     --prompt-length 8192 \
#     --output-len 128 \
#     --window-size 200 \
#     --num-repeat 1 \
#     --log-dir results/vllm_step_${MODEL_NAME}_${MODE}_1proc_${DATE}_debug \
#     --kv-cache-memory-bytes 1673741824 \
#     --max-model-len 17000 \
#     --custom-model \
#     --custom-model-path custom_models.qwen3_0_13_0:Qwen3ForCausalLM \
#     --max-num-seqs 128 \
#     --enforce-eager


# python3 benchmark/profile_vllm_cli.py \
#     --model $MODEL \
#     --batched-tokens 350000 \
#     --batch-sizes 1 2 4 8 16 32 48 64 96 128 \
#     --prompt-length 512 1024 2048 4096 8192 16384 32768  \
#     --output-len 128 \
#     --window-size 200 \
#     --num-repeat 1 \
#     --log-dir results/vllm_step_${MODEL_NAME}_${MODE}_1proc_${DATE} \
#     --kv-cache-memory-bytes 1673741824 \
#     --max-model-len 34000 \
#     --custom-model \
#     --custom-model-path custom_models.qwen3_0_13_0:Qwen3ForCausalLM \
#     --max-num-seqs 128 \
#     --enforce-eager


python3 benchmark/profile_vllm_cli.py \
    --model $MODEL \
    --batched-tokens 135000 \
    --batch-sizes 1 2 4 8 16 24 32 40 48 56 64 80 \
    --prompt-length 512 1024 2048 4096 8192 16384 32768 \
    --output-len 128 \
    --window-size 200 \
    --num-repeat 1 \
    --log-dir results/vllm_step_${MODEL_NAME}_${MODE}_2proc_${DATE} \
    --kv-cache-memory-bytes 829496729 \
    --num-processes 2 \
    --max-model-len 34000 \
    --custom-model \
    --custom-model-path custom_models.qwen3_0_13_0:Qwen3ForCausalLM \
    --max-num-seqs 128
