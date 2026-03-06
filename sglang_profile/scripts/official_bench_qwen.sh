export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH="Qwen/Qwen3-235B-A22B"
export MODEL_NAME=${MODEL_PATH##*/}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}


RESULT_DIR="results_v2/sglang_Qwen3-235B-A22B_1layer_new2_il512_bs64_official_bench"

DATA_SOURCE="sharegpt"
PROMPT_FILE_ARGS="--prompt-file sharegpt_text.txt"

LOG_ARGS="--log-level debug --show-time-cost --log-decode-step 1"

python3 -m sglang.bench_one_batch --model $MODEL_PATH \
--batch-size 64 --input-len 512 --output-len 11 \
--dp $DP --ep $EP --tp $TP --enable-dp-attention \
--result-filename "$RESULT_DIR/result.jsonl" \
--mem-fraction-static 0.9 \
--json-model-override-args '{
    \"rope_scaling\": {
        \"rope_type\": \"yarn\",
        \"factor\": 4.0,
        \"original_max_position_embeddings\": 32768
    }
    }' \
--context-length 131072 \
--enable-expert-distribution-metrics --enable-eplb \
--expert-distribution-recorder-mode stat_approx \
--eplb-rebalance-num-iterations 1000 \
--moe-a2a-backend deepep \
--deepep-mode normal \
$PROMPT_FILE_ARGS \
$LOG_ARGS > "$RESULT_DIR/run.log" 2>&1
