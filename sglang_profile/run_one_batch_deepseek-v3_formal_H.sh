# Multi-node configuration
# NODE_ID: 0 for the first node (master), 1, 2, ... for other nodes
# If not set, default to 0 (single node mode)
export NODE_ID=${NODE_ID:-0}
export NNODES=${NNODES:-1}
export DIST_INIT_ADDR=${DIST_INIT_ADDR:-"172.16.4.52:20000"}

export ENABLE_TBO=${ENABLE_TBO:-0}
TBO_ARGS=""
if [ "$ENABLE_TBO" -eq 1 ]; then
    TBO_ARGS="--enable-two-batch-overlap"
fi

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
export MODEL_PATH="deepseek-ai/DeepSeek-V3"
export MODEL_NAME=${MODEL_PATH##*/}
export DISABLE_NVSHMEM=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}

# Input Config
BS="1 2 1 1 2 4 8 16 32 64"

WARMUP_STEPS=3
IL=${IL:-32000}
OL=4

# Profile Config
# export ENABLE_LOG_EXPERT=1
# DATA_SOURCE="random"
# PROMPT_FILE_ARGS=""

DATA_SOURCE="sharegpt"
PROMPT_FILE_ARGS="--prompt-file sharegpt_text.txt"

# Component profiling environment variables
export PROFILE_COMPONENT_OUTPUT_DIR="./results/component_times_output_${MODEL_NAME}_${DATE}"
RESULT_FILENAME="results/sglang_${MODEL_NAME}_il${IL}/dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}.log"


LOG_ARGS="--log-level debug --show-time-cost --log-decode-step 1"

mkdir -p "running_logs"

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
BENCH_CMD="python bench_one_batch_058.py \
    --model-path $MODEL_PATH \
    --batch $BS --input-len $IL --output-len $OL \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    --log-decode-step 1 \
    --result-filename $RESULT_FILENAME \
    --disable-cuda-graph \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.9 \
    --json-model-override-args '{
        \"rope_scaling\": {
          \"rope_type\": \"yarn\",
          \"factor\": 4.0,
          \"original_max_position_embeddings\": 32768
        }
      }' \
    --context-length 131072 \
    --ep-num-redundant-experts 32 \
    --enable-expert-distribution-metrics --enable-eplb \
    --expert-distribution-recorder-mode stat_approx \
    --eplb-rebalance-num-iterations 10 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    $PROMPT_FILE_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $TBO_ARGS > running_logs/run_one_batch_deepseek-v3_formal_H_dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}_${NODE_RANK}.log 2>&1"

# Run the benchmark (all nodes execute this)
echo "[INFO] Starting benchmark... with command: $BENCH_CMD"
eval $BENCH_CMD

BENCH_EXIT_CODE=$?
if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark failed with exit code $BENCH_EXIT_CODE"
    exit $BENCH_EXIT_CODE
fi

# Only the master node executes analysis and plotting
if [ $IS_MASTER_NODE -eq 1 ]; then
    echo "[INFO] Master node: Running analysis and plotting..."
    
    # Wait a bit to ensure all nodes have finished writing results
    sleep 2
    # Split input_len with space, in input lengths for analysis
    for input_len in $IL; do
        echo "[INFO] Analyzing results for input length $input_len..."
        python analyze_component_times.py "$PROFILE_COMPONENT_OUTPUT_DIR/il${input_len}/cuda/" --output-len $OL
        # python plot_mean_time_vs_batch.py $PROFILE_COMPONENT_OUTPUT_DIR/cuda/analysis
    done
    echo "[INFO] Master node: Analysis and plotting completed"
else
    echo "[INFO] Worker node: Skipping analysis and plotting (only master node executes)"
fi

echo "[INFO] Node $NODE_ID: All tasks completed"
