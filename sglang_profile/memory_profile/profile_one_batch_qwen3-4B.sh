export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH="/nfs/xjzhang/Qwen/Qwen3-4B"
export MODEL_NAME=${MODEL_PATH##*/}

# Deployment Configs to traverse
declare -a NODES=(1)
declare -a CUDA_DEVICES=("2" "2,3" "0,1,2,3")
declare -a DPS=(1 2 4)
declare -a EPS=(1 2 4)
declare -a TPS=(1 2 4)


declare -a NODES=(2)
declare -a CUDA_DEVICES=("2,3")
declare -a DPS=(2)
declare -a EPS=(2)
declare -a TPS=(2)


# Input Config
# BS="1 2 4 8 10 12 14 16 18 20 22 24 32 40 64 128 256 512 1024"

BS="1 2 1 2 4 8 16 32 40 64 128 256 288 320 512 1024 1200 1300 1400 2048 4096 8192"
IL=100
OL=4

# Profile Config
# export 
DATA_SOURCE="random"
PROMPT_FILE_ARGS=""

MEMORY_FRACTION_STATIC=0.6 # token num 143574

# DATA_SOURCE="sharegpt"
# PROMPT_FILE_ARGS="--prompt-file sharegpt_text.txt"

# Component profiling environment variables
# Option 1: Use PROFILE_COMPONENT_OUTPUT_DIR to specify output directory directly
# export PROFILE_COMPONENT_OUTPUT_DIR="./results/component_times_output_${MODEL_NAME}_il${IL}_dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}"
# Option 2: Use PROFILE_COMPONENT_BS and PROFILE_COMPONENT_IN to auto-generate output directory
# export PROFILE_COMPONENT_BS=32
# export PROFILE_COMPONENT_IN=256
# export PROFILE_COMPONENT_MODEL="qwen3-moe"  # Optional, defaults to "qwen3-moe" or "deepseek-v2"

# RESULT_FILENAME="results/sglang_${MODEL_NAME}_il${IL}_dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}.log"

# Main loop to traverse all node configurations
for i in "${!NODES[@]}"; do
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[$i]}"
    DP="${DPS[$i]}"
    EP="${EPS[$i]}"
    TP="${TPS[$i]}"
    
    echo "=========================================="
    echo "Running with ${NODES[$i]} node(s): DP=$DP, EP=$EP, TP=$TP"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "=========================================="
    
    export PROFILE_COMPONENT_OUTPUT_DIR="./results/component_times_output_${MODEL_NAME}_il${IL}_memory${MEMORY_FRACTION_STATIC}/dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}"
    RESULT_FILENAME="results/sglang_${MODEL_NAME}_il${IL}_memory${MEMORY_FRACTION_STATIC}/dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}.log"
    
    mkdir -p $PROFILE_COMPONENT_OUTPUT_DIR
    mkdir -p ${RESULT_FILENAME%/*}
    
    export SGLANG_TORCH_PROFILER_DIR="$PWD/profile_log/profile_${MODEL_NAME}_il${IL}_memory${MEMORY_FRACTION_STATIC}_dp${DP}_ep${EP}_tp${TP}_${DATE}"


    python bench_one_batch_058.py \
        --model-path $MODEL_PATH \
        --batch $BS --input-len $IL --output-len $OL \
        --dp $DP --ep $EP --tp $TP --enable-dp-attention \
        --log-decode-step 1 \
        --result-filename $RESULT_FILENAME \
        --disable-cuda-graph $PROMPT_FILE_ARGS --profile \
        --mem-fraction-static $MEMORY_FRACTION_STATIC \
        --max-total-tokens 2000000 > logs/bench_${MODEL_NAME}_il${IL}_memory${MEMORY_FRACTION_STATIC}_dp${DP}_ep${EP}_tp${TP}_${DATA_SOURCE}_${DATE}.log 2>&1

    python analyze_component_times.py $PROFILE_COMPONENT_OUTPUT_DIR/cuda
    python plot_mean_time_vs_batch.py $PROFILE_COMPONENT_OUTPUT_DIR/cuda/analysis
done

