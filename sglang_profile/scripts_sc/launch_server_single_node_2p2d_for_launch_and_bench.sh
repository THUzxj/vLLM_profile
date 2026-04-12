#!/bin/bash
# Adapter launch script for launch_and_bench_server_pddisagg.sh in single-node PD mode.
# It starts prefill/decode workers + router via serve_single_node_2p2d.sh and then
# stays alive so launch_and_bench can monitor/stop this server process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVE_SCRIPT="${SCRIPT_DIR}/bench_extend_decode/serve_single_node_2p2d.sh"

if [ ! -f "$SERVE_SCRIPT" ]; then
    echo "[ERROR] Missing serve script: $SERVE_SCRIPT"
    exit 1
fi

DATE=${DATE:-$(date +%Y%m%d_%H%M%S)}
SERVE_RESULT_DIR=${SERVE_RESULT_DIR:-"results_single_node_pd/launch_and_bench_${DATE}"}
SERVER_MONITOR_INTERVAL=${SERVER_MONITOR_INTERVAL:-5}

WORKER_PID_FILE="${PROFILE_DIR}/${SERVE_RESULT_DIR}/worker_pids.txt"
ROUTER_PID_FILE="${PROFILE_DIR}/${SERVE_RESULT_DIR}/router_pid.txt"

cleanup() {
    # Graceful stop by PID files written by serve_single_node_2p2d.sh
    if [ -f "$WORKER_PID_FILE" ]; then
        while read -r pid; do
            [ -z "$pid" ] && continue
            kill -TERM "$pid" 2>/dev/null || true
        done < "$WORKER_PID_FILE"
    fi

    if [ -f "$ROUTER_PID_FILE" ]; then
        router_pid=$(cat "$ROUTER_PID_FILE" 2>/dev/null || true)
        if [ -n "${router_pid:-}" ]; then
            kill -TERM "$router_pid" 2>/dev/null || true
        fi
    fi

    # Fallback hard-stop for lingering processes.
    sleep 2
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[INFO] Launching single-node PD 2p2d service for launch_and_bench..."
(
    cd "$PROFILE_DIR"
    RESULT_DIR="$SERVE_RESULT_DIR" \
    MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3}" \
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}" \
    MAX_RUNNING_REQUESTS_DECODE="${MAX_RUNNING_REQUESTS_DECODE:-128}" \
    CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-${MEM_CHUNKED_PREFILL_SIZE:-16384}}" \
    DP="${DP:-2}" EP="${EP:-2}" TP="${TP:-2}" MOE_DENSE_TP="${MOE_DENSE_TP:-1}" \
    ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-1}" \
    ENABLE_EPLB="${ENABLE_EPLB:-1}" \
    ROUTER_PORT="${BASE_PORT:-8000}" \
    bash "$SERVE_SCRIPT"
)

if [ ! -f "$WORKER_PID_FILE" ]; then
    echo "[ERROR] Worker PID file not found: $WORKER_PID_FILE"
    exit 1
fi

echo "[INFO] Service started. Monitoring worker/router pids..."
while true; do
    while read -r pid; do
        [ -z "$pid" ] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] Worker process died unexpectedly: PID=$pid"
            exit 1
        fi
    done < "$WORKER_PID_FILE"

    if [ -f "$ROUTER_PID_FILE" ]; then
        router_pid=$(cat "$ROUTER_PID_FILE" 2>/dev/null || true)
        if [ -n "${router_pid:-}" ] && ! kill -0 "$router_pid" 2>/dev/null; then
            echo "[ERROR] Router process died unexpectedly: PID=$router_pid"
            exit 1
        fi
    fi

    sleep "$SERVER_MONITOR_INTERVAL"
done
