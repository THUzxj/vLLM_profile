#!/bin/bash
# Quick single-node PD run config (launch_and_bench-based)
# Usage: edit MODEL_PATH and run: bash scripts_sc/1node_pddisag.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROFILE_DIR"

MAX_RUNNING_REQUESTS_DECODE=128 \
BASE_PORT=8000 \
PREFILL_NODES=1 \
DECODE_NODES=1 \
MEM_FRACTION_STATIC=0.9 \
DP=2 EP=2 TP=2 \
MODEL_PATH="/nfs/xjzhang/Tongyi-DeepResearch-30B-A3B" \
MEM_CHUNKED_PREFILL_SIZE=16384 \
PROFILE_TRIGGER_THRESHOLD=1.0 \
UPDATE_MAX_RUNNING_REQUESTS=1 \
TOTAL_ROUNDS=2 \
SKIP_WARMUP=1 \
PROFILE_POLLING_INTERVAL=0.5 \
SEND_INTERVAL=5 \
IL="1000" BS="4" OL="100" PROFILE_STEPS=50 \
INTERFACES="eth0" \
SKIP_INTERNAL_ROUTER=1 \
EXIT_WITH_ERROR_TO_STOP_JOB=0 \
ENABLE_DEEPGEMM=0 \
bash scripts_sc/single_node_pddisag_auto_bench.sh
