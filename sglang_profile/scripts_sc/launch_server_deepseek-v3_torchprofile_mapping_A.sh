
ARCHITECTURE="A"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=0

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/deepseek-ai/deepseek-v3-1layer-new-remapped"}
export MODEL_NAME=${MODEL_PATH##*/}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
DP=${DP:-4}
EP=${EP:-4}
TP=${TP:-4}

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"

# Setup output directories
RESULT_DIR="results/sglang_${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_${DATE}"
mkdir -p "$RESULT_DIR"

RESULT_FILENAME="$RESULT_DIR/result.log"

# Build the command with multi-node parameters if needed

set -x
python launch_server_058.py \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $METRICS_ARGS $TBO_ARGS 2>&1 | tee $RESULT_DIR/run.log
set +x

BENCH_EXIT_CODE=$?
if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark failed with exit code $BENCH_EXIT_CODE"
    exit $BENCH_EXIT_CODE
fi

echo "[INFO] Node $NODE_ID: All tasks completed"
