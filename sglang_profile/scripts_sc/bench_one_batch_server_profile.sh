
BS=${BS:-"32"}
IL=${IL:-"32000"}
OL=${OL:-"101"}
PROFILE_STEPS=${PROFILE_STEPS:-10}

export DATE=`date +%Y%m%d_%H%M%S`
MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"}
export MODEL_NAME=${MODEL_PATH##*/}

# Allow RESULT_DIR to be passed from external script
# If not provided, use default pattern
# Usage: RESULT_DIR=/path/to/results ./bench_one_batch_server_profile.sh

DEPLOYMENT_TAG=${DEPLOYMENT_TAG:-"default"}

# Use external RESULT_DIR if provided, otherwise generate default
if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="results_v3/client/sglang_${MODEL_NAME}_il${IL}/${DEPLOYMENT_TAG}_${DATE}"
fi
mkdir -p "$RESULT_DIR"

export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR/torch_profile"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

python bench_one_batch_server_058.py \
 --base-url http://127.0.0.1:30000 \
 --model-path $MODEL_PATH \
 --batch-size $BS --input-len $IL --output-len $OL \
 --profile --profile-by-stage --profile-steps $PROFILE_STEPS \
 --result-filename $RESULT_DIR/result.jsonl \
 --dataset-path "ShareGPT_V3_sample_1pct.json" \
 --dp-size $DP --tp-size $TP --ep-size $EP --enable-dp-attention
