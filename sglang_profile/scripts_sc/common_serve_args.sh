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

LOG_ARGS="--log-level debug --show-time-cost"

MEM_ARGS="""
--chunked-prefill-size 4096 \
--mem-fraction-static 0.8"""

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
        EPLB_ARGS+="--moe-a2a-backend deepep --deepep-mode normal"
    fi
else
    EPLB_ARGS=""
fi

if [ "$ENABLE_EXPERT_DISTRIBUTION_METRICS" = 1 ]; then
    EXPERT_DISTRIBUTION_METRICS_ARGS="--enable-expert-distribution-metrics"
    if [ "$ARCHITECTURE" = "H" ]; then
        EXPERT_DISTRIBUTION_METRICS_ARGS+=" --expert-distribution-recorder-mode stat_approx"
    fi
else
    EXPERT_DISTRIBUTION_METRICS_ARGS=""
fi

ENABLE_TBO=${ENABLE_TBO:-0}
TBO_ARGS=""
if [ "$ENABLE_TBO" -eq 1 ]; then
    TBO_ARGS="--enable-two-batch-overlap"
fi


METRICS_ARGS="--enable-metrics"

TORCH_PROFILE_ARGS="--profile --custom-models-mode torchprofile"

# export values
# export LOGGING_CHUNCKED_PREFILL=True
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1


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

export SGLANG_EXTERNAL_MODEL_PACKAGE="custom_models.torchprofile"
export SGLANG_DG_CACHE_DIR="$PWD/dg_cache"
