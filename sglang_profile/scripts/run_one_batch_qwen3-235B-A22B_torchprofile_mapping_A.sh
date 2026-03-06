
ARCHITECTURE="A"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=0

# For Ampere GPUs, disable DeepEP

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH="/nfs/xjzhang/Qwen3-30B-A3B-2layer"
export MODEL_NAME=${MODEL_PATH##*/}

# Deployment Configs to traverse
# declare -a NODES=(1 2 4)
# declare -a CUDA_DEVICES=("2" "2,3" "0,1,2,3")
# declare -a DPS=(1 2 4)
# declare -a EPS=(1 2 4)
# declare -a TPS=(1 2 4)

# declare -a NODES=(2)
# declare -a CUDA_DEVICES=("2,3")
# declare -a DPS=(2)
# declare -a EPS=(2)
# declare -a TPS=(2)

declare -a NODES=(4)
declare -a CUDA_DEVICES=("0,1,2,3")
declare -a DPS=(4)
declare -a EPS=(4)
declare -a TPS=(4)

# declare -a NODES=(1)
# declare -a CUDA_DEVICES=("0")
# declare -a DPS=(1)
# declare -a EPS=(1)
# declare -a TPS=(1)

# Input Config

# BS="1 2 1 1 2 4 8 16 32 64 128 256 512 1024"
# BS="1 2 4 8 10 12 14 16 18 20 22 24 32 40 64 128 256 512 1024"

# BS="1 2 4 8 16 32 64 128"
# BS="1 2 4 8 16 32 64"
BS="4 16 64"
# BS="1 2 1 2 4 8 10 12 14 16 18 18 18 20 22 24 32 40 64 128 256 512 1024"
# BS="4"
WARMUP_STEPS=3
IL=${IL:-"32000"}
# IL=32000
OL=5

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"

# Main loop to traverse all node configurations
for i in "${!NODES[@]}"; do
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[$i]}"
    DP="${DPS[$i]}"
    EP="${EPS[$i]}"
    TP="${TPS[$i]}"
    
    RESULT_DIR="results_v2/sglang_${MODEL_NAME}_il${IL}/dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}"
    mkdir -p "$RESULT_DIR"

    echo "=========================================="
    echo "Running with ${NODES[$i]} node(s): DP=$DP, EP=$EP, TP=$TP"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Saving results to $RESULT_DIR"
    echo "=========================================="

    RESULT_FILENAME="$RESULT_DIR/result.jsonl"
    MARK_FILENAME="$RESULT_DIR/forward_marks.json"

    export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR/torch_profile"

    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

    if [ "$EP" = "1" ]; then
        CURRENT_EPLB_ARGS=""
    else
        CURRENT_EPLB_ARGS=$EPLB_ARGS
    fi
    set -x
    python bench_one_batch_058.py \
        $TORCH_PROFILER_ARGS \
        --model-path $MODEL_PATH \
        --batch $BS --input-len $IL --output-len $OL \
        --dp $DP --ep $EP --tp $TP --enable-dp-attention \
        --result-filename "$RESULT_FILENAME" \
        --mark-filename "$MARK_FILENAME" \
        $MEM_ARGS \
        $PROMPT_FILE_ARGS \
        "${LONG_CONTEXT_ARGS[@]}" \
        $CURRENT_EPLB_ARGS \
        $LOG_ARGS --disable-cuda-graph > "$RESULT_DIR/run.log" 2>&1
    
    set +x
done
