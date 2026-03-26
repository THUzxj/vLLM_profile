
ARCHITECTURE="H"
ENABLE_EPLB=1
ENABLE_EXPERT_DISTRIBUTION_METRICS=${ENABLE_EXPERT_DISTRIBUTION_METRICS:-0}
PROFILE_RANGES=${PROFILE_RANGES:-"0"}
USE_CUSTOM_MODEL=${USE_CUSTOM_MODEL:-0}

# PD disaggregation configuration (set before sourcing common_serve_args.sh)
# ENABLE_PD_DISAGG=1 enables PD disaggregation mode
# PREFILL_NODES: number of prefill nodes (default: 1)
# DECODE_NODES: number of decode nodes (default: 1)
ENABLE_PD_DISAGG=${ENABLE_PD_DISAGG:-1}
PREFILL_NODES=${PREFILL_NODES:-1}
DECODE_NODES=${DECODE_NODES:-1}

# Enable Nsight Systems profiling with capture-range when set to 1
# Usage: ENABLE_NSYS_PROFILE=1 ./launch_server_deepseek-v3_formal_torchprofile_mapping_H.sh
ENABLE_NSYS_PROFILE=${ENABLE_NSYS_PROFILE:-0}

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-V3"}
export MODEL_NAME=${MODEL_PATH##*/}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
DP=${DP:-8}
EP=${EP:-8}
TP=${TP:-8}
MOE_DENSE_TP=${MOE_DENSE_TP:-1} # Only None or 1 is valid for now
MODULE_NAME=${MODULE_NAME:-"-m sglang.launch_server"}
MAX_RUNNING_REQUESTS_DECODE=${MAX_RUNNING_REQUESTS_DECODE:-256}

# Source common arguments
source "$(dirname "$0")/common_serve_args.sh"

# Router configuration (only started on rank 0)
ROUTER_PORT=${ROUTER_PORT:-9001}
ROUTER_POLICY=${ROUTER_POLICY:-"cache_aware"}

# Allow RESULT_DIR to be passed from external script
# If not provided, use default pattern
# Usage: RESULT_DIR=/path/to/results ./launch_server_deepseek-v3_formal_torchprofile_mapping_H.sh

# Setup output directories
if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="results_v3/server/sglang_${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_${DATE}"
fi
mkdir -p "$RESULT_DIR"

RESULT_FILENAME="$RESULT_DIR/result.log"

# Build router arguments and start router on rank 0 before server
if [ "$ENABLE_PD_DISAGG" -eq 1 ] && [ "$RANK" -eq 0 ]; then
    # Wait for IP files and get prefill node IPs
    PREFILL_IPS=()
    for i in $(seq 0 $((PREFILL_NODES - 1))); do
        if [ $i -eq 0 ]; then
            PREFILL_HOST="${DLC_JOB_ID}-master-0"
        else
            PREFILL_HOST="${DLC_JOB_ID}-worker-$((i - 1))"
        fi

        echo "[INFO] Waiting for prefill node IP: $PREFILL_HOST"
        if wait_for_node_ip "$PREFILL_HOST"; then
            PREFILL_IP=$(get_node_ip "$PREFILL_HOST")
            if [ -n "$PREFILL_IP" ]; then
                PREFILL_IPS+=("$PREFILL_IP")
                echo "[INFO] Got prefill node $i IP: $PREFILL_IP"
            else
                echo "[ERROR] Failed to get IP for $PREFILL_HOST"
                exit 1
            fi
        else
            echo "[ERROR] Timeout waiting for $PREFILL_HOST IP"
            exit 1
        fi
    done

    # Build prefill URLs using IPs
    ROUTER_PREFILL_ARGS="--prefill http://${PREFILL_IPS[0]}:30000 ${ROUTER_PORT}"
    for i in $(seq 1 $((PREFILL_NODES - 1))); do
        ROUTER_PREFILL_ARGS="$ROUTER_PREFILL_ARGS --prefill http://${PREFILL_IPS[$i]}:30000"
    done

    # Wait for IP files and get decode node IPs
    DECODE_IPS=()
    for i in $(seq $((PREFILL_NODES - 1)) $((TOTAL_PD_NODES - 2))); do
        DECODE_HOST="${DLC_JOB_ID}-worker-${i}"

        echo "[INFO] Waiting for decode node IP: $DECODE_HOST"
        if wait_for_node_ip "$DECODE_HOST"; then
            DECODE_IP=$(get_node_ip "$DECODE_HOST")
            if [ -n "$DECODE_IP" ]; then
                DECODE_IPS+=("$DECODE_IP")
                echo "[INFO] Got decode node IP: $DECODE_IP"
            else
                echo "[ERROR] Failed to get IP for $DECODE_HOST"
                exit 1
            fi
        else
            echo "[ERROR] Timeout waiting for $DECODE_HOST IP"
            exit 1
        fi
    done

    # Build decode URLs using IPs
    ROUTER_DECODE_ARGS=""
    for ip in "${DECODE_IPS[@]}"; do
        ROUTER_DECODE_ARGS="$ROUTER_DECODE_ARGS --decode http://${ip}:30000"
    done

    echo "[INFO] Starting router with: prefill_nodes=$PREFILL_NODES, decode_nodes=$DECODE_NODES"
    echo "[INFO] Prefill IPs: ${PREFILL_IPS[@]}"
    echo "[INFO] Decode IPs: ${DECODE_IPS[@]}"
    python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        $ROUTER_PREFILL_ARGS \
        $ROUTER_DECODE_ARGS \
        --policy $ROUTER_POLICY &
    ROUTER_PID=$!
    echo "[INFO] Router started with PID $ROUTER_PID on port $ROUTER_PORT"
fi

# Optional Nsight Systems profiling (capture-range mode)
PROFILE_PREFIX=""
if [ "$ENABLE_NSYS_PROFILE" -eq 1 ]; then
    # nsys output will be written under RESULT_DIR
    NSYS_OUT_BASENAME="$RESULT_DIR/nsys_profile"
    PROFILE_PREFIX="nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat -o $NSYS_OUT_BASENAME "
fi

# --load-format dummy \
RUN_ARGS="
--watchdog-timeout 3600 \
--dist-timeout 3600 \
"

# Build the command with multi-node parameters if needed
echo """
${PROFILE_PREFIX}python $MODULE_NAME \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --moe-dense-tp-size $MOE_DENSE_TP $DP_ATTENTION_ARGS \
    $RUN_ARGS \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $METRICS_ARGS $TBO_ARGS $CUDA_GRAPH_ARGS $ADD_ARGS
""" > $RESULT_DIR/command_node$RANK.log

set -x
${PROFILE_PREFIX}python $MODULE_NAME \
    --model-path $MODEL_PATH \
    --dp $DP --ep $EP --tp $TP --moe-dense-tp-size $MOE_DENSE_TP $DP_ATTENTION_ARGS \
    $RUN_ARGS \
    $MEM_ARGS \
    "${LONG_CONTEXT_ARGS[@]}" \
    $EPLB_ARGS $EXPERT_DISTRIBUTION_METRICS_ARGS \
    $LOG_ARGS $MULTI_NODE_ARGS $METRICS_ARGS $TBO_ARGS $CUDA_GRAPH_ARGS $ADD_ARGS 2>&1 | tee $RESULT_DIR/run_node$RANK.log 
set +x

BENCH_EXIT_CODE=$?
if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark failed with exit code $BENCH_EXIT_CODE"
    exit $BENCH_EXIT_CODE
fi

echo "[INFO] Node $RANK: All tasks completed"
