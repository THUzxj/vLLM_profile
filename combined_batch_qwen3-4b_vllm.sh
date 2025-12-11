MODEL=/nfs/xjzhang/Qwen/Qwen3-4B
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

BATCH1_BS=4
BATCH1_LEN=16384
BATCH2_BS=128
BATCH2_LEN=512

python profile_vllm_cli.py \
    --combined-batch \
    --combined-batch-sizes $BATCH1_BS $BATCH2_BS \
    --combined-prompt-lengths $BATCH1_LEN $BATCH2_LEN \
    --combined-output-lengths 128 128 \
    --num-repeat 3 \
    --log-dir $OUTDIR/${BATCH1_BS}_${BATCH1_LEN}_${BATCH2_BS}_${BATCH2_LEN} \
    --model $MODEL \
    --batched-tokens 100000 \
    --max-model-len 40960

python profile_vllm_cli.py \
    --combined-batch \
    --combined-batch-sizes ${BATCH1_BS} \
    --combined-prompt-lengths $BATCH1_LEN \
    --combined-output-lengths 128 \
    --num-repeat 3 \
    --log-dir $OUTDIR/${BATCH1_BS}_${BATCH1_LEN} \
    --model $MODEL \
    --batched-tokens 100000 \
    --max-model-len 40960

python profile_vllm_cli.py \
    --combined-batch \
    --combined-batch-sizes ${BATCH2_BS} \
    --combined-prompt-lengths $BATCH2_LEN \
    --combined-output-lengths 128 \
    --num-repeat 3 \
    --log-dir $OUTDIR/${BATCH2_BS}_${BATCH2_LEN} \
    --model $MODEL \
    --batched-tokens 100000 \
    --max-model-len 40960
