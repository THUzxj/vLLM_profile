ARCHITECTURE=${ARCHITECTURE:-"A"}

if [ "$ARCHITECTURE" = "A" ]; then
  echo "[INFO] Running with architecture A settings"
  export DISABLE_NVSHMEM=1
elif [ "$ARCHITECTURE" = "H" ]; then
  echo "[INFO] Running with architecture H settings"
else
    echo "Unsupported architecture: $ARCHITECTURE"
    exit 1
fi

DATA_SOURCE=${DATA_SOURCE:-"sharegpt"}
if [ "$DATA_SOURCE" = "sharegpt" ]; then
    PROMPT_FILE_ARGS="--prompt-file sharegpt_text.txt"
else
    PROMPT_FILE_ARGS=""
fi

LOG_ARGS="--decode-log-interval 1 --show-time-cost"


ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-1}

if [ "$ENABLE_DP_ATTENTION" = 1 ]; then
    DP_ATTENTION_ARGS="--enable-dp-attention"
else
    DP_ATTENTION_ARGS=""
fi

ATTN_BACKEND=${ATTN_BACKEND:-"fa3"}
DP_ATTENTION_ARGS+=" --attention-backend $ATTN_BACKEND"

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}


MEM_CHUNKED_PREFILL_SIZE=${MEM_CHUNKED_PREFILL_SIZE:-4096}
if [ "$ENABLE_DP_ATTENTION" = 1 ]; then
    MEM_CHUNKED_PREFILL_SIZE=$((MEM_CHUNKED_PREFILL_SIZE * DP))
    echo "[INFO] DP attention is enabled. The chunked prefill size is adjusted to $MEM_CHUNKED_PREFILL_SIZE"
fi

MEM_ARGS="
--chunked-prefill-size $MEM_CHUNKED_PREFILL_SIZE \
--mem-fraction-static $MEM_FRACTION_STATIC"

LONG_CONTEXT_ARGS=(
    "--context-length" "131072"
    "--json-model-override-args"
    '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}'
)


if [ "$ENABLE_EPLB" = 1 ]; then
    EPLB_ARGS="""
    --enable-eplb \
    --eplb-rebalance-num-iterations 1000 \
    --ep-num-redundant-experts ${EP_NUM_REDUNDANT_EXPERTS:-0} \
    """

    if [ "$ARCHITECTURE" = "H" ]; then
        if [ "$PROFILE_RANGES" = "0" ]; then
            export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
            EPLB_ARGS+="--moe-a2a-backend deepep --deepep-mode low_latency"
        else
            EPLB_ARGS+="--moe-a2a-backend deepep --deepep-mode normal"
        fi
    fi
else
    EPLB_ARGS=""
fi

if [ "$ENABLE_EXPERT_DISTRIBUTION_METRICS" = 1 ]; then
    EXPERT_DISTRIBUTION_METRICS_ARGS="--enable-expert-distribution-metrics"
else
    EXPERT_DISTRIBUTION_METRICS_ARGS=""
fi

ENABLE_TBO=${ENABLE_TBO:-0}
TBO_ARGS=""
if [ "$ENABLE_TBO" -eq 1 ]; then
    TBO_ARGS="--enable-two-batch-overlap"
fi


METRICS_ARGS="--enable-metrics"

if [ "$PROFILE_RANGES" = "1" ]; then
    CUDA_GRAPH_ARGS="--disable-cuda-graph"
else
    CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-256}
    CUDA_GRAPH_ARGS="--cuda-graph-max-bs $CUDA_GRAPH_MAX_BS"
fi

# export values
# export LOGGING_CHUNCKED_PREFILL=True

# distribution settings
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# PD disaggregation configuration
# Set ENABLE_PD_DISAGG=1 to enable PD disaggregation mode
# PREFILL_NODES: number of prefill nodes
# DECODE_NODES: number of decode nodes
ENABLE_PD_DISAGG=${ENABLE_PD_DISAGG:-0}
PREFILL_NODES=${PREFILL_NODES:-1}
DECODE_NODES=${DECODE_NODES:-1}

MULTI_NODE_ARGS=""
if [ "$WORLD_SIZE" -gt 1 ]; then
    # PD disaggregation mode assignment based on RANK
    if [ "$ENABLE_PD_DISAGG" -eq 1 ]; then
        TOTAL_PD_NODES=$((PREFILL_NODES + DECODE_NODES))
        if [ "$WORLD_SIZE" -ne "$TOTAL_PD_NODES" ]; then
            echo "[WARN] WORLD_SIZE=$WORLD_SIZE does not match PREFILL_NODES+DECODE_NODES=$TOTAL_PD_NODES"
        fi

        if [ "$RANK" -lt "$PREFILL_NODES" ]; then
            DISAGG_MODE="prefill"
            DISAGG_NNODES=$PREFILL_NODES
            DISAGG_RANK=$RANK
            DISAGG_DIST_ADDR=$MASTER_ADDR
            echo "[INFO] Node $RANK assigned as PREFILL node (internal rank=$DISAGG_RANK, nnodes=$DISAGG_NNODES)"
        elif [ "$RANK" -lt "$TOTAL_PD_NODES" ]; then
            DISAGG_MODE="decode"
            DISAGG_NNODES=$DECODE_NODES
            DISAGG_RANK=$((RANK - PREFILL_NODES))
            # Decode nodes connect to the last prefill worker
            PREFILL_WORKER_NUM=$((PREFILL_NODES - 1))
            DISAGG_DIST_ADDR="${DLC_JOB_ID}-worker-${PREFILL_WORKER_NUM}"
            echo "[INFO] Node $RANK assigned as DECODE node (internal rank=$DISAGG_RANK, nnodes=$DISAGG_NNODES)"
        else
            echo "[ERROR] RANK=$RANK exceeds total PD nodes ($TOTAL_PD_NODES)"
            exit 1
        fi
        MULTI_NODE_ARGS="--host 0.0.0.0 --nnodes $DISAGG_NNODES --node-rank $DISAGG_RANK --dist-init-addr $DISAGG_DIST_ADDR:$MASTER_PORT"
        MULTI_NODE_ARGS+=" --disaggregation-mode $DISAGG_MODE"
    else
        MULTI_NODE_ARGS="--host 0.0.0.0 --nnodes $WORLD_SIZE --node-rank $RANK --dist-init-addr $MASTER_ADDR:$MASTER_PORT"
        echo "[INFO] Multi-node mode: WORLD_SIZE=$WORLD_SIZE, RANK=$RANK, DIST_INIT_ADDR=$MASTER_ADDR:$MASTER_PORT"
    fi
else
    echo "[INFO] Single-node mode: WORLD_SIZE=$WORLD_SIZE, RANK=$RANK"
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1


if [ "$USE_CUSTOM_MODEL" = 1 ]; then
    export SGLANG_EXTERNAL_MODEL_PACKAGE="custom_models.torchprofile"
else
    export SGLANG_EXTERNAL_MODEL_PACKAGE=""
fi

export SGLANG_DG_CACHE_DIR="$PWD/dg_cache_nnode${WORLD_SIZE}_rank${RANK}"
export SGLANG_DEEPEP_STATS_DIR="$RESULT_DIR/deepep_stats"
echo "[INFO] SGLANG_DEEPEP_STATS_DIR: $SGLANG_DEEPEP_STATS_DIR"
