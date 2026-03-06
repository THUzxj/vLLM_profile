export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH="deepseek-ai/DeepSeek-V3"
export MODEL_NAME=${MODEL_PATH##*/}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}


NODE_RANK={$RANK:-0}
NNODES=${WORLD_SIZE:-1}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GOU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}


MULTI_NODE_ARGS=""
if [ "$NNODES" -gt 1 ]; then
    MULTI_NODE_ARGS="--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $MASTER_ADDR:$MASTER_PORT"
fi



RESULT_DIR="results_v2/sglang_deepseek-v3_1layer_new2_il512_bs64_official_bench"

DATA_SOURCE="sharegpt"
PROMPT_FILE_ARGS="--prompt-file sharegpt_text.txt"

LOG_ARGS="--log-level debug --show-time-cost --log-decode-step 1"

source "$(dirname "$0")/common_serve_args.sh"

python3 -m sglang.bench_one_batch --model $MODEL_PATH \
--batch-size 64 --input-len 512 --output-len 11 \
--dp $DP --ep $EP --tp $TP --enable-dp-attention \
--result-filename "$RESULT_DIR/result.jsonl" \
--mem-fraction-static 0.9 \
--load-format dummy \
"${LONG_CONTEXT_ARGS[@]}" \
--enable-expert-distribution-metrics --enable-eplb \
--expert-distribution-recorder-mode stat_approx \
--eplb-rebalance-num-iterations 1000 \
--moe-a2a-backend deepep \
--deepep-mode normal \
$PROMPT_FILE_ARGS \
$MULTI_NODE_ARGS \
$LOG_ARGS > "$RESULT_DIR/run.log" 2>&1
