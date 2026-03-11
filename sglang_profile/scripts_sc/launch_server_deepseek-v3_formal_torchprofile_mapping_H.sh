
ARCHITECTURE="H"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=0
PROFILE_RANGES=${PROFILE_RANGES:-"0"}
USE_CUSTOM_MODEL=${USE_CUSTOM_MODEL:-0}

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-V3"}
export MODEL_NAME=${MODEL_PATH##*/}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}
MOE_DENSE_TP=${MOE_DENSE_TP:-1} # Only None or 1 is valid for now
MODULE_NAME=${MODULE_NAME:-"-m sglang.launch_server"}

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"

# Allow RESULT_DIR to be passed from external script
# If not provided, use default pattern
# Usage: RESULT_DIR=/path/to/results ./launch_server_deepseek-v3_formal_torchprofile_mapping_H.sh

# Setup output directories
if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="results_v3/server/sglang_${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_${DATE}"
fi
mkdir -p "$RESULT_DIR"

RESULT_FILENAME="$RESULT_DIR/result.log"

RUN_ARGS="
--load-format dummy \
--watchdog-timeout 3600 \
--dist-timeout 3600 \
"

# Build the command with multi-node parameters if needed
echo """
python $MODULE_NAME \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --moe-dense-tp-size $MOE_DENSE_TP --enable-dp-attention \
    $RUN_ARGS \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $METRICS_ARGS $TBO_ARGS $CUDA_GRAPH_ARGS
""" > $RESULT_DIR/command_node$NODE_RANK.log

set -x
python $MODULE_NAME \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --moe-dense-tp-size $MOE_DENSE_TP --enable-dp-attention \
    $RUN_ARGS \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $METRICS_ARGS $TBO_ARGS $CUDA_GRAPH_ARGS 2>&1 | tee $RESULT_DIR/run_node$NODE_RANK.log 
set +x

BENCH_EXIT_CODE=$?
if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark failed with exit code $BENCH_EXIT_CODE"
    exit $BENCH_EXIT_CODE
fi

echo "[INFO] Node $NODE_RANK: All tasks completed"
