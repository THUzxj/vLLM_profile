
ARCHITECTURE="H"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=0

# Multi-node configuration
# NODE_ID: 0 for the first node (master), 1, 2, ... for other nodes
# If not set, default to 0 (single node mode)
export NODE_ID=${NODE_ID:-0}
export NNODES=${NNODES:-1}
export DIST_INIT_ADDR=${DIST_INIT_ADDR:-"172.16.4.52:20000"}

# Check if this is the first node (master node)
IS_MASTER_NODE=0
if [ "$NODE_ID" = "0" ]; then
    IS_MASTER_NODE=1
    echo "[INFO] Running as master node (NODE_ID=0)"
else
    echo "[INFO] Running as worker node (NODE_ID=$NODE_ID)"
fi

# Validate multi-node configuration
if [ "$NNODES" -gt 1 ]; then
    if [ -z "$DIST_INIT_ADDR" ]; then
        echo "[ERROR] DIST_INIT_ADDR must be set for multi-node mode (e.g., export DIST_INIT_ADDR=192.168.0.1:5000)"
        exit 1
    fi
    echo "[INFO] Multi-node mode: NNODES=$NNODES, NODE_RANK=$NODE_RANK, DIST_INIT_ADDR=$DIST_INIT_ADDR"
else
    echo "[INFO] Single-node mode: NNODES=$NNODES, NODE_RANK=$NODE_RANK"
fi

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-V3"}
export MODEL_NAME=${MODEL_PATH##*/}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"

# Setup output directories
RESULT_DIR="results/sglang_${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_${DATE}"
mkdir -p "$RESULT_DIR"

RESULT_FILENAME="$RESULT_DIR/result.log"


# Add multi-node parameters if NNODES is set
MULTI_NODE_ARGS=""
if [ -n "$NNODES" ] && [ "$NNODES" -gt 1 ]; then
    MULTI_NODE_ARGS="--nnodes $NNODES --node-rank $NODE_ID"
    if [ -n "$DIST_INIT_ADDR" ]; then
        MULTI_NODE_ARGS="$MULTI_NODE_ARGS --dist-init-addr $DIST_INIT_ADDR"
    fi
    echo "[INFO] Multi-node mode: NNODES=$NNODES, NODE_RANK=$NODE_ID"
fi


# Build the command with multi-node parameters if needed

set -x
python bench_one_batch_058.py \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $TBO_ARGS > $RESULT_DIR/run.log 2>&1
set +x

BENCH_EXIT_CODE=$?
if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark failed with exit code $BENCH_EXIT_CODE"
    exit $BENCH_EXIT_CODE
fi

echo "[INFO] Node $NODE_ID: All tasks completed"
