#!/bin/bash

# Benchmark script using metrics-triggered profiling
# This version uses bench_one_batch_server_internal_profile_max_batch_058.py
# which monitors running requests and triggers profiling when threshold is reached

BS=${BS:-"32"}
IL=${IL:-"32000"}
OL=${OL:-"101"}
PROFILE_STEPS=${PROFILE_STEPS:-10}

# Metrics-triggered profiling parameters
PROFILE_TRIGGER_THRESHOLD=${PROFILE_TRIGGER_THRESHOLD:-"0.9"}
PROFILE_POLLING_INTERVAL=${PROFILE_POLLING_INTERVAL:-"0.1"}
PROFILE_DELAY_STEPS=${PROFILE_DELAY_STEPS:-"0"}

# Continuous sending parameters
SEND_INTERVAL=${SEND_INTERVAL:-"0.0"}
TOTAL_ROUNDS=${TOTAL_ROUNDS:-"0"}  # 0 means infinite until profile done
WAIT_FOR_PROFILE=${WAIT_FOR_PROFILE:-"1"}

# PD disaggregation URLs (optional)
DECODE_URL=${DECODE_URL:-""}
PREFILL_URL=${PREFILL_URL:-""}

export DATE=`date +%Y%m%d_%H%M%S`
MODEL_PATH=${MODEL_PATH:-"/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"}
export MODEL_NAME=${MODEL_PATH##*/}

# Allow RESULT_DIR to be passed from external script
# If not provided, use default pattern
# Usage: RESULT_DIR=/path/to/results ./bench_one_batch_server_profile_metrics_trigger.sh

DEPLOYMENT_TAG=${DEPLOYMENT_TAG:-"metrics_trigger"}

if [ "$SKIP_WARMUP" = "1" ]; then
    SKIP_WARMUP_ARG="--skip-warmup"
else
    SKIP_WARMUP_ARG="--measure"
fi

if [ "$ENABLE_NSYS_PROFILE" = 1 ]; then
    NSYS_PROFILE_ARGS="--use-nsys"
else
    NSYS_PROFILE_ARGS=""
fi

# Build optional decode/prefill URL args
DECODE_URL_ARG=""
PREFILL_URL_ARG=""
if [ -n "$DECODE_URL" ]; then
    DECODE_URL_ARG="--decode-url $DECODE_URL"
fi
if [ -n "$PREFILL_URL" ]; then
    PREFILL_URL_ARG="--prefill-url $PREFILL_URL"
fi

# Use external RESULT_DIR if provided, otherwise generate default
if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="results_v3/client/sglang_${MODEL_NAME}_il${IL}/${DEPLOYMENT_TAG}_${DATE}"
fi
mkdir -p "$RESULT_DIR"

export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR/torch_profile"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

BASE_PORT=${BASE_PORT:-30000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/../bench_one_batch_server_internal_profile_max_batch_058.py"

echo "================================================================================
Metrics-triggered profiling benchmark
  BS=$BS, IL=$IL, OL=$OL
  PROFILE_STEPS=$PROFILE_STEPS
  PROFILE_TRIGGER_THRESHOLD=$PROFILE_TRIGGER_THRESHOLD
  PROFILE_POLLING_INTERVAL=$PROFILE_POLLING_INTERVAL
  PROFILE_DELAY_STEPS=$PROFILE_DELAY_STEPS
  SEND_INTERVAL=$SEND_INTERVAL
  TOTAL_ROUNDS=$TOTAL_ROUNDS
  WAIT_FOR_PROFILE=$WAIT_FOR_PROFILE
  DECODE_URL=$DECODE_URL
  PREFILL_URL=$PREFILL_URL
================================================================================"

echo "python $BENCH_SCRIPT \
    --base-url http://127.0.0.1:$BASE_PORT \
    --model-path $MODEL_PATH \
    --batch-size $BS --input-len $IL --output-len $OL \
    --profile --profile-by-stage --profile-steps $PROFILE_STEPS \
    --profile-trigger-threshold $PROFILE_TRIGGER_THRESHOLD \
    --profile-polling-interval $PROFILE_POLLING_INTERVAL \
    --profile-delay-steps $PROFILE_DELAY_STEPS \
    --send-interval $SEND_INTERVAL \
    --total-rounds $TOTAL_ROUNDS \
    $( [ "$WAIT_FOR_PROFILE" = "1" ] && echo "--wait-for-profile" ) \
    $DECODE_URL_ARG $PREFILL_URL_ARG \
    --result-filename $RESULT_DIR/result.jsonl \
    --dataset-path \"../sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json\" \
    --dp-size $DP --tp-size $TP --ep-size $EP --enable-dp-attention $SKIP_WARMUP_ARG $NSYS_PROFILE_ARGS
" > $RESULT_DIR/command.log

python $BENCH_SCRIPT \
    --base-url http://127.0.0.1:$BASE_PORT \
    --model-path $MODEL_PATH \
    --batch-size $BS --input-len $IL --output-len $OL \
    --profile --profile-by-stage --profile-steps $PROFILE_STEPS \
    --profile-trigger-threshold $PROFILE_TRIGGER_THRESHOLD \
    --profile-polling-interval $PROFILE_POLLING_INTERVAL \
    --profile-delay-steps $PROFILE_DELAY_STEPS \
    --send-interval $SEND_INTERVAL \
    --total-rounds $TOTAL_ROUNDS \
    $( [ "$WAIT_FOR_PROFILE" = "1" ] && echo "--wait-for-profile" ) \
    $DECODE_URL_ARG $PREFILL_URL_ARG \
    --result-filename $RESULT_DIR/result.jsonl \
    --dataset-path "../sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json" \
    --dp-size $DP --tp-size $TP --ep-size $EP --enable-dp-attention $SKIP_WARMUP_ARG $NSYS_PROFILE_ARGS 2>&1 | tee $RESULT_DIR/client.log