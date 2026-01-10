MODEL=/nfs/xjzhang/Qwen/Qwen3-0.6B
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Combined batch 模式示例
# python profile_vllm_cli.py \
#     --combined-batch \
#     --combined-batch-sizes 1 16 \
#     --combined-prompt-lengths 16384 1024 \
#     --combined-output-lengths 128 128 \
#     --num-repeat 3 \
#     --log-dir vllm_step_${MODEL_NAME}_combined_batch_${DATE} \
#     --model $MODEL \
#     --batched-tokens 280000 \
#     --max-model-len 40960


OUTDIR=vllm_step_${MODEL_NAME}_combined_batch_v1_${DATE}_comparison

# python profile_vllm_cli.py \
#     --combined-batch \
#     --combined-batch-sizes 1 16 \
#     --combined-prompt-lengths 16384 1024 \
#     --combined-output-lengths 128 128 \
#     --num-repeat 3 \
#     --log-dir $OUTDIR/1_16384_16_1024 \
#     --model $MODEL \
#     --batched-tokens 280000 \
#     --max-model-len 40960

# python profile_vllm_cli.py \
#     --combined-batch \
#     --combined-batch-sizes 1 \
#     --combined-prompt-lengths 16384 \
#     --combined-output-lengths 128 \
#     --num-repeat 3 \
#     --log-dir $OUTDIR/1_16384 \
#     --model $MODEL \
#     --batched-tokens 280000 \
#     --max-model-len 40960

# python profile_vllm_cli.py \
#     --combined-batch \
#     --combined-batch-sizes 16 \
#     --combined-prompt-lengths 1024 \
#     --combined-output-lengths 128 \
#     --num-repeat 3 \
#     --log-dir $OUTDIR/16_1024 \
#     --model $MODEL \
#     --batched-tokens 280000 \
#     --max-model-len 40960


# python profile_vllm_cli.py \
#     --combined-batch \
#     --combined-batch-sizes 1 8 \
#     --combined-prompt-lengths 8192 1024 \
#     --combined-output-lengths 128 128 \
#     --num-repeat 3 \
#     --log-dir $OUTDIR/1_8192_16_1024 \
#     --model $MODEL \
#     --batched-tokens 280000 \
#     --max-model-len 40960


# python profile_vllm_cli.py \
#     --combined-batch \
#     --combined-batch-sizes 1 \
#     --combined-prompt-lengths 8192 \
#     --combined-output-lengths 128 \
#     --num-repeat 3 \
#     --log-dir $OUTDIR/1_8192 \
#     --model $MODEL \
#     --batched-tokens 280000 \
#     --max-model-len 40960

# python profile_vllm_cli.py \
#     --combined-batch \
#     --combined-batch-sizes 8 \
#     --combined-prompt-lengths 1024 \
#     --combined-output-lengths 128 \
#     --num-repeat 3 \
#     --log-dir $OUTDIR/8_1024 \
#     --model $MODEL \
#     --batched-tokens 280000 \
#     --max-model-len 40960


python profile_vllm_cli.py \
    --combined-batch \
    --combined-batch-sizes 1 32 \
    --combined-prompt-lengths 16384 512 \
    --combined-output-lengths 128 128 \
    --num-repeat 3 \
    --log-dir $OUTDIR/1_16384_32_512 \
    --model $MODEL \
    --batched-tokens 280000 \
    --max-model-len 40960


python profile_vllm_cli.py \
    --combined-batch \
    --combined-batch-sizes 1 \
    --combined-prompt-lengths 16384 \
    --combined-output-lengths 128 \
    --num-repeat 3 \
    --log-dir $OUTDIR/1_16384 \
    --model $MODEL \
    --batched-tokens 280000 \
    --max-model-len 40960

python profile_vllm_cli.py \
    --combined-batch \
    --combined-batch-sizes 32 \
    --combined-prompt-lengths 512 \
    --combined-output-lengths 128 \
    --num-repeat 3 \
    --log-dir $OUTDIR/32_512 \
    --model $MODEL \
    --batched-tokens 280000 \
    --max-model-len 40960
