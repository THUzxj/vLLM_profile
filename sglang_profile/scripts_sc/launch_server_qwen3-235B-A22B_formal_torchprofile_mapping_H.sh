
ARCHITECTURE="H"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=1
PROFILE_RANGES=${PROFILE_RANGES:-"0"}

# For Ampere GPUs, disable DeepEP

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-235B-A22B"}
export MODEL_NAME=${MODEL_PATH##*/}

# Deployment Config
# These can be overridden by environment variables if needed
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}
MODULE_NAME=${MODULE_NAME:-"-m sglang.launch_server"}

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"

# Allow RESULT_DIR to be passed from external script
# If not provided, use default pattern
# Usage: RESULT_DIR=/path/to/results ./launch_server_qwen3-235B-A22B_formal_torchprofile_mapping_H.sh

# Setup output directories
if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="results_v3/server/sglang_${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_${DATE}"
fi
mkdir -p "$RESULT_DIR"

RESULT_FILENAME="$RESULT_DIR/result.log"

echo "=========================================="
echo "Running with $NNODES node(s): DP=$DP, EP=$EP, TP=$TP"
echo "RANK=$RANK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if [ "$NNODES" -gt 1 ]; then
    echo "DIST_INIT_ADDR=$DIST_INIT_ADDR"
fi
echo "=========================================="

# Run the benchmark (all nodes execute this)
echo "[INFO] Starting benchmark on node $RANK..."

echo """
python $MODULE_NAME \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $METRICS_ARGS $MULTI_NODE_ARGS $TBO_ARGS $CUDA_GRAPH_ARGS 2>&1 | tee "$RESULT_DIR/run_node$RANK.log"
""" > $RESULT_DIR/command_node$RANK.log

set -x
python $MODULE_NAME \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $METRICS_ARGS $MULTI_NODE_ARGS $TBO_ARGS $CUDA_GRAPH_ARGS 2>&1 | tee "$RESULT_DIR/run_node$RANK.log"
set +x

BENCH_EXIT_CODE=$?

if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark failed with exit code $BENCH_EXIT_CODE on node $RANK"
    exit $BENCH_EXIT_CODE
fi

echo "[INFO] Node $RANK: All tasks completed"
