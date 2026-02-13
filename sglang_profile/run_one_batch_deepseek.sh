
export CUDA_VISIBLE_DEVICES=2,3

# Multi-node configuration
# NODE_ID: 0 for the first node (master), 1, 2, ... for other nodes
# If not set, default to 0 (single node mode)
export NODE_ID=${NODE_ID:-0}

# Check if this is the first node (master node)
IS_MASTER_NODE=0
if [ "$NODE_ID" = "0" ]; then
    IS_MASTER_NODE=1
    echo "[INFO] Running as master node (NODE_ID=0)"
else
    echo "[INFO] Running as worker node (NODE_ID=$NODE_ID)"
fi

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH="/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"
export MODEL_NAME=${MODEL_PATH##*/}

# Component profiling environment variables
export PROFILE_COMPONENT_OUTPUT_DIR="./results/component_times_output_${MODEL_NAME}_${DATE}"
RESULT_FILENAME="results/sglang_qwen3_4b_bs32_il256_dp2_ep2_tp2_enabledpattn_decode_step_${DATE}.log"

# Deployment Config
DP=2
EP=2
TP=2

# Multi-node Config (if needed, uncomment and set accordingly)
# NNODES=2  # Total number of nodes
# DIST_INIT_ADDR="172.16.4.52:20000"  # IP:PORT of the first node

# Input Config
BS="1 2 4 8 16 32 64 128"
IL=10
OL=4

# Build the command with multi-node parameters if needed
BENCH_CMD="python bench_one_batch_058.py \
    --model-path $MODEL_PATH \
    --batch $BS --input-len $IL --output-len $OL \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    --log-decode-step 1 \
    --result-filename $RESULT_FILENAME \
    --disable-cuda-graph \
    --prompt-file sharegpt_text.txt"

# Add multi-node parameters if NNODES is set
if [ -n "$NNODES" ] && [ "$NNODES" -gt 1 ]; then
    BENCH_CMD="$BENCH_CMD --nnodes $NNODES --node-rank $NODE_ID"
    if [ -n "$DIST_INIT_ADDR" ]; then
        BENCH_CMD="$BENCH_CMD --dist-init-addr $DIST_INIT_ADDR"
    fi
    echo "[INFO] Multi-node mode: NNODES=$NNODES, NODE_RANK=$NODE_ID"
fi

# Run the benchmark (all nodes execute this)
echo "[INFO] Starting benchmark..."
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
    
    python analyze_component_times.py $PROFILE_COMPONENT_OUTPUT_DIR/cuda
    
    python plot_mean_time_vs_batch.py $PROFILE_COMPONENT_OUTPUT_DIR/cuda/analysis
    
    echo "[INFO] Master node: Analysis and plotting completed"
else
    echo "[INFO] Worker node: Skipping analysis and plotting (only master node executes)"
fi

echo "[INFO] Node $NODE_ID: All tasks completed"
