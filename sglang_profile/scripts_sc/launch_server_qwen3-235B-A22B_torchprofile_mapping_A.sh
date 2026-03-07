
ARCHITECTURE="A"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=0

# For Ampere GPUs, disable DeepEP

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"}
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

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"

# Main loop to traverse all node configurations
for i in "${!NODES[@]}"; do
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[$i]}"
    DP="${DPS[$i]}"
    EP="${EPS[$i]}"
    TP="${TPS[$i]}"
    
    RESULT_DIR="results_v3/server/sglang_${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_${DATE}"
    mkdir -p "$RESULT_DIR"

    echo "=========================================="
    echo "Running with ${NODES[$i]} node(s): DP=$DP, EP=$EP, TP=$TP"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Saving results to $RESULT_DIR"
    echo "=========================================="

    # export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR/torch_profile"

    # mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

    if [ "$EP" = "1" ]; then
        CURRENT_EPLB_ARGS=""
    else
        CURRENT_EPLB_ARGS=$EPLB_ARGS
    fi

    echo """
    python launch_server_058.py \
        --model-path $MODEL_PATH \
        --dp $DP --ep $EP --tp $TP --enable-dp-attention \
        $MEM_ARGS \
        "${LONG_CONTEXT_ARGS[@]}" \
        $CURRENT_EPLB_ARGS \
        $LOG_ARGS $METRICS_ARGS --disable-cuda-graph 2>&1 | tee "$RESULT_DIR/run.log"
    """ > $RESULT_DIR/command.log


    set -x
    python launch_server_058.py \
        --model-path $MODEL_PATH \
        --dp $DP --ep $EP --tp $TP --enable-dp-attention \
        $MEM_ARGS \
        "${LONG_CONTEXT_ARGS[@]}" \
        $CURRENT_EPLB_ARGS \
        $LOG_ARGS $METRICS_ARGS --disable-cuda-graph 2>&1 | tee "$RESULT_DIR/run.log"
    set +x
done
