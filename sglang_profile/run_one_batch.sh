
export CUDA_VISIBLE_DEVICES=2,3

export DATE=`date +%Y%m%d_%H%M%S`
export MODEL_PATH="/nfs/xjzhang/Qwen/Qwen3-235B-A22B-1layer-new2"
export MODEL_NAME=${MODEL_PATH##*/}

# Component profiling environment variables
export PROFILE_COMPONENT_OUTPUT_DIR="./results/component_times_output_${MODEL_NAME}_${DATE}"
RESULT_FILENAME="results/sglang_qwen3_4b_bs32_il256_dp2_ep2_tp2_enabledpattn_decode_step_${DATE}.log"

# Deployment Config
DP=2
EP=2
TP=2

# Input Config
BS="1 2 4 8 16 32 64 128"
IL=10
OL=4

python bench_one_batch_058.py \
    --model-path $MODEL_PATH \
    --batch $BS --input-len $IL --output-len $OL \
    --dp $DP --ep $EP --tp $TP --enable-dp-attention \
    --log-decode-step 1 \
    --result-filename $RESULT_FILENAME \
    --disable-cuda-graph \
    --prompt-file sharegpt_text.txt

python analyze_component_times.py $PROFILE_COMPONENT_OUTPUT_DIR/cuda

python plot_mean_time_vs_batch.py $PROFILE_COMPONENT_OUTPUT_DIR/cuda/analysis
