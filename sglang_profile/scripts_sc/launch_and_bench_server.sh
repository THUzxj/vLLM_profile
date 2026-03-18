#!/bin/bash
set -e

# Configuration
SERVER_READY_TIMEOUT=32000  # Maximum wait time in seconds
SERVER_READY_CHECK_INTERVAL=2  # Check interval in seconds
BASE_URL="http://127.0.0.1:30000"
MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"}
MODEL_NAME=${MODEL_PATH##*/}
NODE_RANK=${RANK:-0}
NNODES=${WORLD_SIZE:-1}
ARCHITECTURE=${ARCHITECTURE:-"H"}
ENABLE_TBO=${ENABLE_TBO:-0}

# Only allow Nsight Systems profiling on rank 0
# If ENABLE_NSYS_PROFILE is set for multi-node runs, non-zero ranks will have it disabled.
if [ -n "${ENABLE_NSYS_PROFILE:-}" ] && [ "${ENABLE_NSYS_PROFILE}" != "0" ]; then
    if [ -n "${NODE_RANK:-}" ] && [ "${NODE_RANK}" != "0" ]; then
        echo "[INFO] NODE_RANK=${NODE_RANK}: disabling Nsight Systems profiling (only rank 0 is profiled)"
        ENABLE_NSYS_PROFILE=0
    else
        echo "[INFO] NODE_RANK=${NODE_RANK}: Nsight Systems profiling enabled"
    fi
fi

# Function to check if server is ready by polling the health endpoint
check_server_ready() {
    local base_url="$1"
    local timeout="$2"
    local interval="$3"

    echo "[INFO] Waiting for server to be ready (max ${timeout}s)..."

    local start_time=$(date +%s)
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if curl -s -f "${base_url}/health" > /dev/null 2>&1; then
            echo "[INFO] Server health check passed!"
            return 0
        else
            echo "[ERROR] Server health check failed!"
        fi

        sleep $interval
        elapsed=$(( $(date +%s) - start_time ))

        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "[INFO] Waiting... (${elapsed}s elapsed)"
        fi
    done

    echo "[ERROR] Server did not become ready within ${timeout} seconds"
    return 1
}

# Function to launch server and wait for it to be ready
launch_and_wait_server() {
    local server_script="$1"
    local server_log="$2"
    local server_result_dir="$3"
    local timeout="$4"
    local interval="$5"

    echo "[INFO] Launching server: $server_script"
    echo "[INFO] Server result directory: $server_result_dir"
    echo "[INFO] Server log: $server_log"

    # Launch server in background with RESULT_DIR and DATE exported
    # Enable job control to get the correct process group ID


    if [ -z "$NODE_RANK" ] || [ "$NODE_RANK" = "0" ]; then
        export RESULT_DIR="$server_result_dir"
        {
            bash "$server_script" 2>&1 | tee "$server_log"
        } &
        SERVER_PID=$!
        echo "[INFO] Server process started with PID: $SERVER_PID"

        # Wait for server to be ready
        if ! check_server_ready "$BASE_URL" $timeout $interval; then
            echo "[ERROR] Failed to detect server ready state"
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            return 1
        fi

        return 0
    else
        # No background
        export RESULT_DIR="$server_result_dir"
        bash "$server_script" 2>&1 | tee "$server_log"
        return $?
    fi
}

# Main script
if [ $# -lt 2 ]; then
    echo "Usage: $0 <launch_server_script> <bench_script>"
    echo "Example: $0 ./launch_server_qwen3-235B-A22B_torchprofile_mapping_A.sh ./bench_one_batch_server_profile.sh"
    exit 1
fi

DATE=${DATE:-$(date +%Y%m%d_%H%M%S)}
export DATE

MOE_DENSE_TP=${MOE_DENSE_TP:-1} # Only None or 1 is valid for now

LAUNCH_SERVER_SCRIPT="$1"
BENCH_SCRIPT="$2"

# Multi-node RESULT_DIR synchronization via NFS shared file.
# Node 0 writes RESULT_DIR to a sync file; other nodes wait and read it.
# Skipped when RESULT_DIR is already set externally or in single-node mode.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_FILE="${SCRIPT_DIR}/.multi_node_result_dir_sync"
SYNC_TIMEOUT=${SYNC_TIMEOUT:-120}
SYNC_FILE_CREATED=0

cleanup_sync_file() {
    # Only node 0 cleans up the sync file, and only if this script created it.
    if [ "${NNODES:-1}" -gt 1 ] && [ "${NODE_RANK:-0}" = "0" ] && [ "${SYNC_FILE_CREATED:-0}" -eq 1 ]; then
        rm -f "$SYNC_FILE" 2>/dev/null || true
        echo "[INFO] Node 0: sync file removed"
    fi
}
trap cleanup_sync_file EXIT

# Optional hard override: if RESULT_DIR_FIXED is set, always use it and skip sync/default logic.
# This is useful when launching from different nodes without coordinating environment variables.
if [ -n "${RESULT_DIR_FIXED:-}" ]; then
    RESULT_DIR="$RESULT_DIR_FIXED"
    echo "[INFO] RESULT_DIR_FIXED is set, using RESULT_DIR=$RESULT_DIR (skip sync/default logic)"
fi

if [ -z "${RESULT_DIR:-}" ]; then
    if [ "$NNODES" -gt 1 ]; then
        if [ "$NODE_RANK" = "0" ]; then
            rm -f "$SYNC_FILE"
            RESULT_DIR="results_v4/${MODEL_NAME}/dp${DP}_TBO${ENABLE_TBO}_NORMAL${PROFILE_RANGES}_${DATE}"
            echo "$RESULT_DIR" > "${SYNC_FILE}.tmp"
            mv "${SYNC_FILE}.tmp" "$SYNC_FILE"
            SYNC_FILE_CREATED=1
            echo "[INFO] Node 0: RESULT_DIR written to sync file"
        else
            echo "[INFO] Node $NODE_RANK: waiting for RESULT_DIR sync from node 0..."
            _sync_elapsed=0
            while [ ! -f "$SYNC_FILE" ]; do
                sleep 1
                _sync_elapsed=$((_sync_elapsed + 1))
                if [ $_sync_elapsed -ge $SYNC_TIMEOUT ]; then
                    echo "[ERROR] Node $NODE_RANK: sync file not found after ${SYNC_TIMEOUT}s, aborting"
                    exit 1
                fi
                if [ $((_sync_elapsed % 10)) -eq 0 ]; then
                    echo "[INFO] Node $NODE_RANK: still waiting... (${_sync_elapsed}s)"
                fi
            done
            RESULT_DIR=$(cat "$SYNC_FILE")
            echo "[INFO] Node $NODE_RANK: synced RESULT_DIR=$RESULT_DIR"
        fi
    else
        RESULT_DIR="results_v4/${MODEL_NAME}/dp${DP}_TBO${ENABLE_TBO}_NORMAL${PROFILE_RANGES}_RANK${RANK}_${DATE}"
    fi
fi

SERVER_LOG=$RESULT_DIR/server_rank${RANK}.log
BENCH_LOG=$RESULT_DIR/bench_rank${RANK}.log

# Check if scripts exist
if [ ! -f "$LAUNCH_SERVER_SCRIPT" ]; then
    echo "[ERROR] Launch server script not found: $LAUNCH_SERVER_SCRIPT"
    exit 1
fi

if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "[ERROR] Benchmark script not found: $BENCH_SCRIPT"
    exit 1
fi

echo "=========================================="
echo "Starting server and benchmark workflow"
echo "=========================================="
echo "Server script: $LAUNCH_SERVER_SCRIPT"
echo "Benchmark script: $BENCH_SCRIPT"
echo "Result directory: $RESULT_DIR"
echo "Server log: $SERVER_LOG"
echo "Benchmark log: $BENCH_LOG"
echo "=========================================="

# Create result directory
mkdir -p "$RESULT_DIR"

# Determine server and bench result directories
SERVER_RESULT_DIR="$RESULT_DIR/server"
BENCH_RESULT_DIR="$RESULT_DIR/client"
mkdir -p "$SERVER_RESULT_DIR"

# Launch server and wait for it to be ready
if ! launch_and_wait_server "$LAUNCH_SERVER_SCRIPT" "$SERVER_LOG" "$SERVER_RESULT_DIR" $SERVER_READY_TIMEOUT $SERVER_READY_CHECK_INTERVAL; then
    echo "[ERROR] Server launch failed"
    exit 1
fi

# Server is ready, run benchmark with RESULT_DIR set
# Check if NODE_RANK is 0, only run benchmark on rank 0
if [ -z "$NODE_RANK" ] || [ "$NODE_RANK" = "0" ]; then
    echo "=============== START BENCHMARK ==============="
    mkdir -p "$BENCH_RESULT_DIR"
    export RESULT_DIR="$BENCH_RESULT_DIR"
    export SGLANG_TORCH_PROFILER_DIR="$BENCH_RESULT_DIR/torch_profile"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
    echo "[INFO] NODE_RANK is $NODE_RANK, running benchmark..."
    bash "$BENCH_SCRIPT" 2>&1 | tee "$BENCH_LOG"
    BENCH_EXIT_CODE=$?

    sleep 10

    # Kill server process
    echo "[INFO] Stopping server..."
    echo "[INFO] Server PID: $SERVER_PID"

    # Kill the process group (including child processes)
    if kill -0 -$SERVER_PID 2>/dev/null; then
        echo "[INFO] Killing process group..."
        kill -2 -$SERVER_PID 2>/dev/null || true
    fi

    # Wait for up to 10 seconds for graceful shutdown
    SERVER_STOPPED=0
    for i in {1..10}; do
        if ! ps -p $SERVER_PID > /dev/null 2>&1; then
            echo "[INFO] Server stopped gracefully"
            SERVER_STOPPED=1
            break
        fi
        sleep 1
    done

    # If still running after 10 seconds, force kill the process group
    if [ $SERVER_STOPPED -eq 0 ]; then
        echo "[WARNING] Server did not stop gracefully, forcing kill process group..."
        if kill -0 -$SERVER_PID 2>/dev/null; then
            kill -9 -- -$SERVER_PID 2>/dev/null || true
        fi
        # Also kill by name as fallback
        pkill -9 -f "python.*sglang" 2>/dev/null || true
    fi

    # Wait for the process to finish (non-blocking if already exited)
    # wait $SERVER_PID 2>/dev/null || true

    echo "Force exit"
    exit 1

    echo "[INFO] Server stop complete"

    if [ $BENCH_EXIT_CODE -eq 0 ]; then
        echo "[SUCCESS] Benchmark completed successfully"
    else
        echo "[ERROR] Benchmark failed with exit code: $BENCH_EXIT_CODE"
        # exit $BENCH_EXIT_CODE
        exit 1
    fi

    echo "=========================================="
    echo "Workflow completed!"
    echo "=========================================="

else
    echo "[INFO] NODE_RANK is $NODE_RANK, skipping benchmark (only rank 0 runs benchmark)"
    BENCH_EXIT_CODE=0
fi

