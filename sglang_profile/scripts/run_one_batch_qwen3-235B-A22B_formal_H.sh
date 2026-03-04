
ARCHITECTURE="H"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=1

# For Ampere GPUs, disable DeepEP

# Multi-node configuration
# Get node rank from environment variables (support multiple common names)
# Priority: NODE_RANK > NODE_ID > SLURM_NODEID > default 0
export NODE_RANK=${NODE_RANK:-${NODE_ID:-${SLURM_NODEID:-0}}}

# Get number of nodes from environment variable (default to 1 for single node)
export NNODES=${NNODES:-1}
export DIST_INIT_ADDR=${DIST_INIT_ADDR:-""}

# Check if this is the master node (rank 0)
IS_MASTER_NODE=0
if [ "$NODE_RANK" = "0" ]; then
    IS_MASTER_NODE=1
    echo "[INFO] Running as master node (NODE_RANK=0)"
else
    echo "[INFO] Running as worker node (NODE_RANK=$NODE_RANK)"
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
export MODEL_PATH="Qwen/Qwen3-235B-A22B"
export MODEL_NAME=${MODEL_PATH##*/}

# Deployment Config
# These can be overridden by environment variables if needed
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}

# Input Config
BS="1 2 1 1 2 4 8 16 32 64"

WARMUP_STEPS=3
IL=${IL:-32000}
OL=11

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"


# Setup output directories
RESULT_DIR="results/sglang_${MODEL_NAME}_il${IL}/dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}"
mkdir -p "$RESULT_DIR"

RESULT_FILENAME="$RESULT_DIR/result.log"
MARK_FILENAME="$RESULT_DIR/forward_marks.json"

mkdir -p "$PROFILE_COMPONENT_OUTPUT_DIR"
mkdir -p "${RESULT_FILENAME%/*}"
mkdir -p "running_logs"

echo "=========================================="
echo "Running with $NNODES node(s): DP=$DP, EP=$EP, TP=$TP"
echo "NODE_RANK=$NODE_RANK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if [ "$NNODES" -gt 1 ]; then
    echo "DIST_INIT_ADDR=$DIST_INIT_ADDR"
fi
echo "=========================================="

# Run the benchmark (all nodes execute this)
echo "[INFO] Starting benchmark on node $NODE_RANK..."

# Build base command arguments
MULTI_NODE_ARGS=""
if [ "$NNODES" -gt 1 ]; then
    MULTI_NODE_ARGS="--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $DIST_INIT_ADDR"
fi
BENCH_CMD="python bench_one_batch_058.py \
    --model-path $MODEL_PATH \
    --batch $BS --input-len $IL --output-len $OL \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    --result-filename '$RESULT_FILENAME' \
    $MEM_ARGS \
    $PROMPT_FILE_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $TBO_ARGS > $RESULT_DIR/run.logs 2>&1"

echo "[INFO] Starting benchmark... with command: $BENCH_CMD"
eval $BENCH_CMD

BENCH_EXIT_CODE=$?

if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark failed with exit code $BENCH_EXIT_CODE on node $NODE_RANK"
    exit $BENCH_EXIT_CODE
fi

echo "[INFO] Node $NODE_RANK: All tasks completed"
