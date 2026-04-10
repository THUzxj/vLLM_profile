#!/bin/bash
set -e

# Register node IP before starting server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGISTER_SCRIPT="$SCRIPT_DIR/register_node_ip.sh"

if [ -f "$REGISTER_SCRIPT" ]; then
    echo "=========================================="
    echo "Registering node IP address..."
    echo "=========================================="

    export NODE_TYPE=${NODE_TYPE:-"worker"}
    if [ -n "${RANK:-}" ] && [ "$RANK" = "0" ] && [ "${WORLD_SIZE:-1}" -gt 1 ]; then
        export NODE_TYPE="master"
    fi
    export NODE_RANK=${RANK:-0}

    bash "$REGISTER_SCRIPT"
    echo "[INFO] Node IP registered, waiting 5 seconds for other nodes..."
    sleep 5
    echo "[INFO] Continuing with server launch..."
    echo ""
else
    echo "[WARN] register_node_ip.sh not found, skipping IP registration"
fi

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
EP_NUM_REDUNDANT_EXPERTS=${EP_NUM_REDUNDANT_EXPERTS:-0}
PROFILE_RANGES=${PROFILE_RANGES:-0}
ONLY_LAUNCH=${ONLY_LAUNCH:-0}
ENABLE_PD_DISAGG=${ENABLE_PD_DISAGG:-0}
PREFILL_NODES=${PREFILL_NODES:-1}
DECODE_NODES=${DECODE_NODES:-1}
TOTAL_PD_NODES=$((PREFILL_NODES + DECODE_NODES))
MAX_RUNNING_REQUESTS_DECODE=${MAX_RUNNING_REQUESTS_DECODE:-256}

# DeepGEMM pre-compilation configuration
ENABLE_DEEPGEMM=${ENABLE_DEEPGEMM:-1}       # Set to 0 to skip DeepGEMM pre-compilation
DEEPGEMM_COMPILE_RETRIES=${DEEPGEMM_COMPILE_RETRIES:-3}  # Max retry attempts
DEEPGEMM_COMPILE_TIMEOUT_SECONDS=${DEEPGEMM_COMPILE_TIMEOUT_SECONDS:-1800}  # 30 minutes

# Router configuration (only started on rank 0)
ROUTER_PORT=${ROUTER_PORT:-8998}
ROUTER_POLICY=${ROUTER_POLICY:-"cache_aware"}
ROUTER_PID=""


# NFS shared directory for node IP mapping
NFS_SHARED_DIR=${NFS_SHARED_DIR:-"/nfs/shared"}

# Function to get node IP from NFS file
# Usage: get_node_ip <node_name>
# Returns: IP address or empty string if not found
get_node_ip() {
    local node_name=$1
    local ip_file="$NFS_SHARED_DIR/${node_name}.ip"
    local ip=""

    if [ -f "$ip_file" ]; then
        ip=$(cat "$ip_file" | tr -d '[:space:]')
        if [ -z "$ip" ]; then
            echo "[WARN] get_node_ip: empty IP content for node $node_name (file: $ip_file)" >&2
        fi
    else
        echo "[WARN] get_node_ip: IP file not found for node $node_name (file: $ip_file)" >&2
    fi
    echo "$ip"
}

# Function to wait for node IP file to be available
# Usage: wait_for_node_ip <node_name> [timeout_seconds]
# Returns: 0 if IP found, 1 if timeout
wait_for_node_ip() {
    local node_name=$1
    local timeout=${2:-60}
    local ip_file="$NFS_SHARED_DIR/${node_name}.ip"
    local start_time=$(date +%s)

    while [ ! -f "$ip_file" ]; do
        local elapsed_time=$(( $(date +%s) - start_time ))
        if [ $elapsed_time -ge $timeout ]; then
            echo "[ERROR] Timeout waiting for IP file: $ip_file"
            return 1
        fi
        sleep 1
    done

    return 0
}

stop_router() {
    if [ -z "${ROUTER_PID:-}" ]; then
        return 0
    fi

    # Router is launched via a background pipeline; stop by process group when possible.
    if ps -p "$ROUTER_PID" > /dev/null 2>&1; then
        echo "[INFO] Stopping router..."
        if kill -0 -"$ROUTER_PID" 2>/dev/null; then
            kill -TERM -"$ROUTER_PID" 2>/dev/null || true
        else
            kill -TERM "$ROUTER_PID" 2>/dev/null || true
        fi

        ROUTER_STOPPED=0
        for i in {1..10}; do
            if ! ps -p "$ROUTER_PID" > /dev/null 2>&1; then
                ROUTER_STOPPED=1
                echo "[INFO] Router stopped gracefully"
                break
            fi
            sleep 1
        done

        if [ "$ROUTER_STOPPED" -eq 0 ]; then
            echo "[WARNING] Router did not stop gracefully, forcing kill..."
            if kill -0 -"$ROUTER_PID" 2>/dev/null; then
                kill -KILL -"$ROUTER_PID" 2>/dev/null || true
            else
                kill -KILL "$ROUTER_PID" 2>/dev/null || true
            fi
            pkill -9 -f "python3 -m sglang_router.launch_router.*--port ${ROUTER_PORT}" 2>/dev/null || true
        fi
    fi

    wait "$ROUTER_PID" 2>/dev/null || true
    ROUTER_PID=""
}

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

# Function to check if server is ready by polling the health endpoint.
# Also monitors the server PID (if SERVER_PID is set) so that an early
# crash is detected immediately instead of waiting for the full timeout.
check_server_ready() {
    local base_url="$1"
    local timeout="$2"
    local interval="$3"

    echo "[INFO] Waiting for server to be ready (max ${timeout}s)..."

    local start_time=$(date +%s)
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        # If the server process has already exited, abort immediately
        if [ -n "${SERVER_PID:-}" ] && ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
            echo "[ERROR] Server process (PID $SERVER_PID) has exited unexpectedly while waiting for readiness"
            return 1
        fi

        if curl -s -f "${base_url}/health" > /dev/null 2>&1; then
            echo "[INFO] Server health check passed!"
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

    # When ONLY_LAUNCH=1, all nodes (including rank 0) run server synchronously
    if [ "$ONLY_LAUNCH" = "1" ]; then
        echo "[INFO] ONLY_LAUNCH=1, running server synchronously on all nodes"
        export RESULT_DIR="$server_result_dir"
        bash "$server_script" 2>&1 | tee "$server_log"
        return $?
    fi

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

# Function to run DeepGEMM pre-compilation with retry logic.
# Only runs when ARCHITECTURE=H and ENABLE_DEEPGEMM=1.
# Runs $LAUNCH_SERVER_SCRIPT verbatim but with MODULE_NAME overridden to
# "-m sglang.compile_deep_gemm", so every arg (--dp/--ep/--tp/
# --moe-dense-tp-size/--enable-dp-attention/--context-length/
# --disaggregation-mode/--nnodes/--node-rank/--dist-init-addr …)
# is guaranteed to match the actual server launch exactly.
# Detects success by checking for "DeepGEMM Kernels compilation finished
# successfully." in the log produced by the launch script's tee.
run_deepgemm_precompile() {
    local log_dir="$1"
    local retries="${DEEPGEMM_COMPILE_RETRIES:-3}"
    local timeout_seconds="${DEEPGEMM_COMPILE_TIMEOUT_SECONDS:-1800}"
    local attempt=0
    local success=0
    # Use the same log naming convention as the launch script.
    local compile_log="$log_dir/run_node${RANK:-0}_BS${MAX_RUNNING_REQUESTS_DECODE}_${DATE}.log"

    echo "=========================================="
    echo "[DeepGEMM] Starting pre-compilation (ARCHITECTURE=H)"
    echo "[DeepGEMM] Launch script : $LAUNCH_SERVER_SCRIPT"
    echo "[DeepGEMM] MODULE_NAME   : -m sglang.compile_deep_gemm"
    echo "[DeepGEMM] Max retries   : $retries"
    echo "[DeepGEMM] Timeout (sec): $timeout_seconds"
    echo "[DeepGEMM] Compile log   : $compile_log"
    echo "=========================================="

    mkdir -p "$log_dir"

    while [ $attempt -lt $retries ]; do
        attempt=$((attempt + 1))
        echo "[DeepGEMM] Attempt $attempt / $retries ..."

        # Run the same launch script with MODULE_NAME replaced.
        # RESULT_DIR is redirected to log_dir so the tee log lands there.
        # ENABLE_NSYS_PROFILE=0 prevents nsys from wrapping the compile run.
        local exit_code=0
        if command -v timeout >/dev/null 2>&1; then
            MODULE_NAME="-m sglang.compile_deep_gemm" \
            RESULT_DIR="$log_dir" \
            LOG_FILENAME="$compile_log" \
            ENABLE_NSYS_PROFILE=0 \
            timeout --signal=TERM --kill-after=30s "${timeout_seconds}s" \
                bash "$LAUNCH_SERVER_SCRIPT" || exit_code=$?
        else
            echo "[WARN] 'timeout' command not found, running compile without hard timeout."
            MODULE_NAME="-m sglang.compile_deep_gemm" \
            RESULT_DIR="$log_dir" \
            LOG_FILENAME="$compile_log" \
            ENABLE_NSYS_PROFILE=0 \
            bash "$LAUNCH_SERVER_SCRIPT" || exit_code=$?
        fi

        # Check log for success marker written by compile_deep_gemm
        if grep -q "DeepGEMM Kernels compilation finished successfully\." "$compile_log" 2>/dev/null; then
            echo "[DeepGEMM] Pre-compilation succeeded on attempt $attempt."
            success=1
            break
        fi

        if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
            echo "[DeepGEMM] Compilation timed out after ${timeout_seconds}s on attempt $attempt."
        elif [ $exit_code -ne 0 ]; then
            echo "[DeepGEMM] Compilation process exited with code $exit_code on attempt $attempt."
        else
            echo "[DeepGEMM] Process exited 0 but success marker not found in log (attempt $attempt)."
        fi

        if [ $attempt -lt $retries ]; then
            echo "[DeepGEMM] Retrying in 10 seconds..."
            sleep 10
        fi
    done

    if [ $success -eq 0 ]; then
        echo "[ERROR] DeepGEMM pre-compilation failed after $retries attempt(s). Aborting."
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

cleanup_child_processes() {
    local self_pid="$$"
    echo "[INFO] Cleaning up child processes of script PID ${self_pid}..."
    pkill -TERM -P "$self_pid" 2>/dev/null || true
    sleep 5
    pkill -KILL -P "$self_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}

cleanup_on_exit() {
    stop_router
    cleanup_child_processes
    cleanup_sync_file
}
trap cleanup_on_exit EXIT

# Optional hard override: if RESULT_DIR_FIXED is set, always use it and skip sync/default logic.
# This is useful when launching from different nodes without coordinating environment variables.
if [ -n "${RESULT_DIR_FIXED:-}" ]; then
    RESULT_DIR="$RESULT_DIR_FIXED"
    echo "[INFO] RESULT_DIR_FIXED is set, using RESULT_DIR=$RESULT_DIR (skip sync/default logic)"
fi


RESULT_TAG="dp${DP}_TBO${ENABLE_TBO}_NORMAL${PROFILE_RANGES}_REDUNDANT${EP_NUM_REDUNDANT_EXPERTS}_RunningReqsDecode${MAX_RUNNING_REQUESTS_DECODE}"

if [ -z "${RESULT_DIR:-}" ]; then
    if [ "$NNODES" -gt 1 ]; then
        if [ "$NODE_RANK" = "0" ]; then
            rm -f "$SYNC_FILE"
            RESULT_DIR="results_v4/${MODEL_NAME}/${RESULT_TAG}_${DATE}"
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
        RESULT_DIR="results_v4/${MODEL_NAME}/${RESULT_TAG}_RANK${RANK}_${DATE}"
    fi
fi

SERVER_LOG=$RESULT_DIR/server_rank${RANK}.log
BENCH_LOG=$RESULT_DIR/bench_rank${RANK}.log
ROUTER_LOG=$RESULT_DIR/router_rank${RANK}.log

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

# DeepGEMM pre-compilation (only on Architecture H, before launching server)
if [ "$ARCHITECTURE" = "H" ] && [ "${ENABLE_DEEPGEMM:-1}" -eq 1 ]; then
    DEEPGEMM_LOG_DIR="$RESULT_DIR/deepgemm_compile_rank${NODE_RANK}"
    echo "[INFO] Architecture=H: running DeepGEMM pre-compilation on node rank ${NODE_RANK}..."
    if ! run_deepgemm_precompile "$DEEPGEMM_LOG_DIR"; then
        echo "[ERROR] DeepGEMM pre-compilation failed, cannot proceed to launch server."
        exit 1
    fi
    echo "[INFO] DeepGEMM pre-compilation complete, proceeding to server launch."
else
    if [ "$ARCHITECTURE" != "H" ]; then
        echo "[INFO] Architecture=$ARCHITECTURE: skipping DeepGEMM pre-compilation (only needed for H)."
    else
        echo "[INFO] ENABLE_DEEPGEMM=0: skipping DeepGEMM pre-compilation."
    fi
fi

# Determine server and bench result directories
SERVER_RESULT_DIR="$RESULT_DIR/server"
BENCH_RESULT_DIR="$RESULT_DIR/client"
mkdir -p "$SERVER_RESULT_DIR"

# Launch server and wait for it to be ready
if ! launch_and_wait_server "$LAUNCH_SERVER_SCRIPT" "$SERVER_LOG" "$SERVER_RESULT_DIR" $SERVER_READY_TIMEOUT $SERVER_READY_CHECK_INTERVAL; then
    echo "[ERROR] Server launch failed"
    exit 1
fi

# If ONLY_LAUNCH=1, skip benchmark (server runs synchronously and exits on its own)
if [ "$ONLY_LAUNCH" = "1" ]; then
    echo "[INFO] ONLY_LAUNCH=1, skipping benchmark"
    echo "=========================================="
    echo "Server-only mode completed!"
    echo "=========================================="
    exit 0
fi

# Server is ready, run benchmark with RESULT_DIR set
# Check if NODE_RANK is 0, only run benchmark on rank 0
if [ -z "$NODE_RANK" ] || [ "$NODE_RANK" = "0" ]; then
    echo "=============== START BENCHMARK ==============="
    mkdir -p "$BENCH_RESULT_DIR"
    export RESULT_DIR="$BENCH_RESULT_DIR"
    export SGLANG_TORCH_PROFILER_DIR="$BENCH_RESULT_DIR/torch_profile"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

    # Get decode and prefill URLs from NFS files (first IP of each group)
    # Use machine-index naming from register_node_ip.sh:
    # - first prefill node: ${DLC_JOB_ID}-master-0 (rank 0)
    # - first decode node:  ${DLC_JOB_ID}-worker-$((PREFILL_NODES - 1)) (rank PREFILL_NODES, worker_num = rank - 1)
    DLC_JOB_ID=${DLC_JOB_ID:-"test-job"}
    PREFILL_NODE_NAME="${DLC_JOB_ID}-master-0"
    DECODE_FIRST_RANK=$PREFILL_NODES
    DECODE_FIRST_WORKER_NUM=$((DECODE_FIRST_RANK - 1))
    DECODE_NODE_NAME="${DLC_JOB_ID}-worker-${DECODE_FIRST_WORKER_NUM}"

    DECODE_IP=$(get_node_ip "$DECODE_NODE_NAME")
    PREFILL_IP=$(get_node_ip "$PREFILL_NODE_NAME")

    if [ -n "$DECODE_IP" ]; then
        export DECODE_URL="http://${DECODE_IP}:30000"
        echo "[INFO] DECODE_URL set to: $DECODE_URL"
    else
        echo "[WARN] Could not find decode IP for node $DECODE_NODE_NAME"
    fi

    if [ -n "$PREFILL_IP" ]; then
        export PREFILL_URL="http://${PREFILL_IP}:30000"
        echo "[INFO] PREFILL_URL set to: $PREFILL_URL"
    else
        echo "[WARN] Could not find prefill IP for node $PREFILL_NODE_NAME"
    fi

    # Start router if PD disaggregation is enabled
    # if [ "${ENABLE_PD_DISAGG:-0}" -eq 1 ]; then
        echo "[INFO] PD disaggregation mode enabled, starting router..."

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
        for decode_rank in $(seq "$PREFILL_NODES" $((TOTAL_PD_NODES - 1))); do
            decode_worker_num=$((decode_rank - 1))
            DECODE_HOST="${DLC_JOB_ID}-worker-${decode_worker_num}"

            echo "[INFO] Waiting for decode node IP: $DECODE_HOST"
            if wait_for_node_ip "$DECODE_HOST"; then
                DECODE_IP=$(get_node_ip "$DECODE_HOST")
                if [ -n "$DECODE_IP" ]; then
                    DECODE_IPS+=("$DECODE_IP")
                    echo "[INFO] Got decode node rank ${decode_rank} (worker-${decode_worker_num}) IP: $DECODE_IP"
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

        ROUTER_STARTUP_DELAY=${ROUTER_STARTUP_DELAY:-10}
        echo "[INFO] Waiting ${ROUTER_STARTUP_DELAY}s for router to be ready..."
        sleep $ROUTER_STARTUP_DELAY

        ROUTER_CMD="python3 -m sglang_router.launch_router --port 8000 --pd-disaggregation $ROUTER_PREFILL_ARGS $ROUTER_DECODE_ARGS --policy $ROUTER_POLICY"
        echo "$ROUTER_CMD" > "$RESULT_DIR/router_cmd_rank${RANK}.txt"


        set -x
        python3 -m sglang_router.launch_router \
            --port 8000 \
            --pd-disaggregation \
            $ROUTER_PREFILL_ARGS \
            $ROUTER_DECODE_ARGS \
            --policy $ROUTER_POLICY 2>&1 | tee "$ROUTER_LOG" &
        set +x
        ROUTER_PID=$!
        echo "[INFO] Router started with PID $ROUTER_PID on port $ROUTER_PORT"

        # Wait for router to be ready
        echo "[INFO] Waiting ${ROUTER_STARTUP_DELAY}s for router to be ready..."
        sleep $ROUTER_STARTUP_DELAY
    # fi

    echo "[INFO] NODE_RANK is $NODE_RANK, running benchmark..."

    # Monitor server health in the background while bench is running.
    # If the server process dies during the benchmark, write a sentinel file
    # and kill the bench sub-processes so the main script does not hang.
    SERVER_DIED_FLAG="$RESULT_DIR/.server_died_rank${NODE_RANK}"
    rm -f "$SERVER_DIED_FLAG"
    (
        while true; do
            sleep 5
            if [ -n "${SERVER_PID:-}" ] && ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
                echo "[ERROR] Server process (PID $SERVER_PID) died during benchmark!" >&2
                touch "$SERVER_DIED_FLAG"
                # Interrupt children of this shell (bench + tee) so tee/bench exit
                pkill -TERM -P $$ 2>/dev/null || true
                break
            fi
        done
    ) &
    SERVER_MONITOR_PID=$!

    bash "$BENCH_SCRIPT" 2>&1 | tee "$BENCH_LOG"
    BENCH_EXIT_CODE=$?

    # Stop the monitor
    kill "$SERVER_MONITOR_PID" 2>/dev/null || true
    wait "$SERVER_MONITOR_PID" 2>/dev/null || true

    # If the server died during the benchmark, treat it as a failure
    if [ -f "$SERVER_DIED_FLAG" ]; then
        echo "[ERROR] Server exited abnormally during benchmark run."
        rm -f "$SERVER_DIED_FLAG"
        BENCH_EXIT_CODE=1
    fi

    sleep 10

    # Kill router process
    stop_router

    # Kill server process
    echo "[INFO] Stopping server..."
    echo "[INFO] Server PID: $SERVER_PID"

    # Collect server exit code in case it already exited on its own
    SERVER_EXIT_CODE=0
    if [ -n "${SERVER_PID:-}" ] && ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
        wait "$SERVER_PID" 2>/dev/null
        SERVER_EXIT_CODE=$?
        if [ $SERVER_EXIT_CODE -ne 0 ]; then
            echo "[ERROR] Server process (PID $SERVER_PID) had already exited with code $SERVER_EXIT_CODE"
        fi
    fi

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

    # Reap the server process
    wait $SERVER_PID 2>/dev/null || true

    echo "[INFO] Server stop complete"

    # Determine final exit code: fail if either bench or server failed
    FINAL_EXIT_CODE=0
    if [ $BENCH_EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Benchmark failed with exit code: $BENCH_EXIT_CODE"
        FINAL_EXIT_CODE=$BENCH_EXIT_CODE
    fi
    if [ ${SERVER_EXIT_CODE:-0} -ne 0 ]; then
        echo "[ERROR] Server exited with code: $SERVER_EXIT_CODE"
        FINAL_EXIT_CODE=$SERVER_EXIT_CODE
    fi

    if [ $FINAL_EXIT_CODE -eq 0 ]; then
        echo "[SUCCESS] Benchmark completed successfully"
    fi

    echo "=========================================="
    echo "Workflow completed!"
    echo "=========================================="

    # Fix to exit 1 to let all the nodes in the job stop
    exit 1
    # exit $FINAL_EXIT_CODE

else
    echo "[INFO] NODE_RANK is $NODE_RANK, skipping benchmark (only rank 0 runs benchmark)"
    BENCH_EXIT_CODE=0
fi
