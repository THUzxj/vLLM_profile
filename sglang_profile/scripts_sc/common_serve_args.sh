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
NODE_RANK=${RANK:-0}
NNODES=${WORLD_SIZE:-1}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

MULTI_NODE_ARGS=""
if [ "$NNODES" -gt 1 ]; then
    MULTI_NODE_ARGS="--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $MASTER_ADDR:$MASTER_PORT"
    echo "[INFO] Multi-node mode: NNODES=$NNODES, NODE_RANK=$NODE_RANK, DIST_INIT_ADDR=$MASTER_ADDR:$MASTER_PORT"
else 
    echo "[INFO] Single-node mode: NNODES=$NNODES, NODE_RANK=$NODE_RANK"
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
