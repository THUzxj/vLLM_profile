
BS=${BS:-"32"}
IL=${IL:-"32000"}
OL=${OL:-"101"}

export DATE=`date +%Y%m%d_%H%M%S`
MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"}
export MODEL_NAME=${MODEL_PATH##*/}

DEPLOYMENT_TAG=${DEPLOYMENT_TAG:-"default"}

RESULT_DIR="results_v2/sglang_${MODEL_NAME}_il${IL}/${DEPLOYMENT_TAG}_${DATE}"
mkdir -p "$RESULT_DIR"

export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR/torch_profile"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

python -m sglang.bench_one_batch_server \
 --base-url http://127.0.0.1:30000 \
 --model-path $MODEL_PATH \
 --batch-size $BS --input-len $IL --output-len $OL \
 --profile --profile-by-stage \
 --result-filename $RESULT_DIR/result.jsonl
 