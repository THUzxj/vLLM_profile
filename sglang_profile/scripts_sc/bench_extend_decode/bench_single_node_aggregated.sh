#!/bin/bash
# Benchmark script for unified (non-PD disaggregation) server
# Connects to a single SGLang server instance

set -e

# ============================================
# Configuration
# ============================================
MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Tongyi-DeepResearch-30B-A3B"}
SERVER_PORT=${SERVER_PORT:-30000}
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"

# Benchmark parameters
BATCH_SIZE=${BATCH_SIZE:-"64 32 16 8"} # 512 256 128 
CACHED_TOKEN_LENS=${CACHED_TOKEN_LENS:-"1000"}
OUTPUT_LEN=${OUTPUT_LEN:-"148"}
EXTEND_LENS=${EXTEND_LENS:-"608"} # + test 2000
DP=4

# Result directory
DATE=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results_single_node_unified/bench_${DATE}"
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "Benchmarking Unified SGLang Service"
echo "=========================================="
echo "Server URL: $SERVER_URL"
echo "Batch sizes: $BATCH_SIZE"
echo "Cached token lengths: $CACHED_TOKEN_LENS"
echo "Output lengths: $OUTPUT_LEN"
echo "Result directory: $RESULT_DIR"
echo "=========================================="

# Check if server is ready
echo "Checking server health..."
if ! curl -s -f "$SERVER_URL/health" > /dev/null 2>&1; then
    echo "[ERROR] Server is not responding at $SERVER_URL"
    echo "[ERROR] Please start the server first"
    exit 1
fi
echo "[OK] Server is ready"
echo ""

# Run benchmark for each cached token length
for CACHED_LEN in $CACHED_TOKEN_LENS; do
    for EXTEND_LEN in $EXTEND_LENS; do
        INPUT_LEN=$((CACHED_LEN + EXTEND_LEN))
        echo "Running benchmark: cached_len=$CACHED_LEN, input_len=$INPUT_LEN"

        python bench_one_batch_server_058.py \
            --model None \
            --base-url "$SERVER_URL" \
            --batch-size $BATCH_SIZE \
            --input-len $INPUT_LEN \
            --cached-token-len $CACHED_LEN \
            --output-len $OUTPUT_LEN \
            --result-filename "$RESULT_DIR/results_cached${CACHED_LEN}_extend${EXTEND_LEN}.jsonl" \
            --measure \
            --enable-dp-attention \
            --dp $DP \
            --measure-tbt \
            2>&1 | tee "$RESULT_DIR/benchmark_cached${CACHED_LEN}_extend${EXTEND_LEN}.log"

        echo ""
    done
done

python aggregate_pd_test_results.py $RESULT_DIR $RESULT_DIR/aggregated.csv


echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to: $RESULT_DIR"
echo "=========================================="
