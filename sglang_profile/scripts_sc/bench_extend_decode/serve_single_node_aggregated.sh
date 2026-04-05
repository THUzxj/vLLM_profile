
ARCHITECTURE="A"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=0
PROFILE_RANGES=${PROFILE_RANGES:-"0"}
ENABLE_DP_ATTENTION=1

# For Ampere GPUs, disable DeepEP

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Tongyi-DeepResearch-30B-A3B"}
export MODEL_NAME=${MODEL_PATH##*/}

# Allow RESULT_DIR to be passed from external script
# If not provided, use default pattern
# Usage: RESULT_DIR=/path/to/results ./launch_server.sh

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

declare -a NODES=(2)
declare -a CUDA_DEVICES=("0,1")
declare -a DPS=(2)
declare -a EPS=(2)
declare -a TPS=(2)
MODULE_NAME=${MODULE_NAME:-"-m sglang.launch_server"}

# Memory and performance settings
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS_DECODE=${MAX_RUNNING_REQUESTS_DECODE:-128}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}  # Increased for better prefill throughput (will be 8192 after DP division)


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

    # Use external RESULT_DIR if provided, otherwise generate default
    if [ -z "$RESULT_DIR" ]; then
        RESULT_DIR="results_v3/server/sglang_${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_${DATE}"
    fi
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
    python $MODULE_NAME \
        --model-path $MODEL_PATH \
        --dp $DP --ep $EP --tp $TP \
        $DP_ATTENTION_ARGS \
        $MEM_ARGS \
        "${LONG_CONTEXT_ARGS[@]}" \
        $CURRENT_EPLB_ARGS \
        $LOG_ARGS $METRICS_ARGS $CUDA_GRAPH_ARGS 2>&1 | tee "$RESULT_DIR/run.log"
    """ > $RESULT_DIR/command.log


    set -x
    python $MODULE_NAME \
        --model-path $MODEL_PATH \
        --dp $DP --ep $EP --tp $TP \
        $DP_ATTENTION_ARGS \
        $MEM_ARGS \
        "${LONG_CONTEXT_ARGS[@]}" \
        $CURRENT_EPLB_ARGS \
        $LOG_ARGS $METRICS_ARGS $CUDA_GRAPH_ARGS 2>&1 | tee "$RESULT_DIR/run.log"
    set +x
done
