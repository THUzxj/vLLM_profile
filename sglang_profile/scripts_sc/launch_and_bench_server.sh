#!/bin/bash
set -e

# Configuration
SERVER_READY_TIMEOUT=120  # Maximum wait time in seconds
SERVER_READY_CHECK_INTERVAL=2  # Check interval in seconds
BASE_URL="http://127.0.0.1:30000"
MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"}
NODE_RANK=${RANK:-0}
NNODES=${WORLD_SIZE:-1}

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
        fi

        # Alternative: check if port is listening
        if nc -z 127.0.0.1 30000 2>/dev/null; then
            echo "[INFO] Server port is listening!"
            return 0
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
}

# Main script
if [ $# -lt 2 ]; then
    echo "Usage: $0 <launch_server_script> <bench_script>"
    echo "Example: $0 ./launch_server_qwen3-235B-A22B_torchprofile_mapping_A.sh ./bench_one_batch_server_profile.sh"
    exit 1
fi

DATE=$(date +%Y%m%d_%H%M%S)

MOE_DENSE_TP=${MOE_DENSE_TP:-1} # Only None or 1 is valid for now

LAUNCH_SERVER_SCRIPT="$1"
BENCH_SCRIPT="$2"
RESULT_DIR="results_v4/${MODEL_NAME}/dp${DP}_ep${EP}_tp${TP}_moedensetp${MOE_DENSE_TP}_${DATE}"
# RESULT_DIR="$3"
SERVER_LOG=$RESULT_DIR/server.log
BENCH_LOG=$RESULT_DIR/bench.log

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
    mkdir -p "$BENCH_RESULT_DIR"
    export RESULT_DIR="$BENCH_RESULT_DIR"
    export SGLANG_TORCH_PROFILER_DIR="$BENCH_RESULT_DIR/torch_profile"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
    echo "[INFO] NODE_RANK is $NODE_RANK, running benchmark..."
    bash "$BENCH_SCRIPT" 2>&1 | tee "$BENCH_LOG"
    BENCH_EXIT_CODE=$?


    # Kill server process
    echo "[INFO] Stopping server..."
    echo "[INFO] Server PID: $SERVER_PID"

    # Kill the process group (including child processes)
    if kill -0 -$SERVER_PID 2>/dev/null; then
        echo "[INFO] Killing process group..."
        kill -- -$SERVER_PID 2>/dev/null || true
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
    sleep infinity
    BENCH_EXIT_CODE=0
fi

