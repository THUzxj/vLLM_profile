#!/bin/bash
set -e

MODEL=/nfs/xjzhang/Qwen/Qwen3-4B
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`

export CUDA_VISIBLE_DEVICES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

VERSION="v1_combined_batch_component"

# 创建主输出目录
MAIN_OUTDIR=vllm_step_${MODEL_NAME}_${VERSION}_${DATE}
mkdir -p $MAIN_OUTDIR

# 定义3种combined_batch配置
# 配置1: Small batch, Large input length + Large batch, Small input length
CONFIG1_BS1=4
CONFIG1_LEN1=8192
CONFIG1_BS2=64
CONFIG1_LEN2=512
CONFIG1_NAME="bs${CONFIG1_BS1}_in${CONFIG1_LEN1}_bs${CONFIG1_BS2}_in${CONFIG1_LEN2}"

# 配置2: Two medium batches
CONFIG2_BS1=16
CONFIG2_LEN1=2048
CONFIG2_BS2=32
CONFIG2_LEN2=1024
CONFIG2_NAME="bs${CONFIG2_BS1}_in${CONFIG2_LEN1}_bs${CONFIG2_BS2}_in${CONFIG2_LEN2}"

# 配置3: Large batch Small input + Small batch Large input
CONFIG3_BS1=128
CONFIG3_LEN1=512
CONFIG3_BS2=2
CONFIG3_LEN2=4096
CONFIG3_NAME="bs${CONFIG3_BS1}_in${CONFIG3_LEN1}_bs${CONFIG3_BS2}_in${CONFIG3_LEN2}"

echo "==================== Combined Batch Component Profiling ===================="
echo "Main output directory: $MAIN_OUTDIR"
echo ""

# 函数：执行combined batch profiling
run_combined_batch_profile() {
    local config_name=$1
    local bs1=$2
    local len1=$3
    local bs2=$4
    local len2=$5
    
    echo "========== Configuration: $config_name =========="
    echo "Batch 1: BS=$bs1, Input Length=$len1"
    echo "Batch 2: BS=$bs2, Input Length=$len2"
    
    local config_dir="$MAIN_OUTDIR/$config_name"
    mkdir -p $config_dir
    
    # 检查 batched tokens 是否超过限制
    local batched_tokens1=$((bs1 * len1))
    local batched_tokens2=$((bs2 * len2))
    
    if [ $batched_tokens1 -gt 100000 ] || [ $batched_tokens2 -gt 100000 ]; then
        echo "⚠️  Skipping configuration $config_name: batched tokens exceed 100000"
        echo "  Batch 1: $batched_tokens1, Batch 2: $batched_tokens2"
        return
    fi

    COMPONENT_OUTPUT_DIR="vllm_component_${MODEL_NAME}_${config_name}_${DATE}"

    export PROFILE_COMPONENT_OUTPUT_DIR="${COMPONENT_OUTPUT_DIR}/combined"

    OUTPUT_LEN=128
    
    # 执行profiling - Combined batch模式
    python3 profile_vllm_cli.py \
        --combined-batch \
        --combined-batch-sizes $bs1 $bs2 \
        --combined-prompt-lengths $len1 $len2 \
        --combined-output-lengths $OUTPUT_LEN $OUTPUT_LEN \
        --num-repeat 1 \
        --log-dir $config_dir/combined \
        --model $MODEL \
        --batched-tokens 100000 \
        --max-model-len 10000 \
        --window-size 200 \
        --custom-model \
        --enforce-eager

    export PROFILE_COMPONENT_OUTPUT_DIR="${COMPONENT_OUTPUT_DIR}/batch1_${bs1}_${len1}"
    
    # 执行profiling - 单独 batch 1
    python3 profile_vllm_cli.py \
        --model $MODEL \
        --batched-tokens 100000 \
        --combined-batch \
        --combined-batch-sizes $bs1 \
        --combined-prompt-lengths $len1 \
        --combined-output-len $OUTPUT_LEN \
        --window-size 200 \
        --num-repeat 1 \
        --log-dir $config_dir/batch1_${bs1}_${len1} \
        --gpu-memory-utilization 0.95 \
        --custom-model \
        --enforce-eager

    export PROFILE_COMPONENT_OUTPUT_DIR="${COMPONENT_OUTPUT_DIR}/batch2_${bs2}_${len2}"
    
    # 执行profiling - 单独 batch 2
    python3 profile_vllm_cli.py \
        --model $MODEL \
        --batched-tokens 100000 \
        --combined-batch \
        --combined-batch-sizes $bs1 \
        --combined-prompt-lengths $len1 \
        --combined-output-len $OUTPUT_LEN \
        --window-size 200 \
        --num-repeat 1 \
        --log-dir $config_dir/batch2_${bs2}_${len2} \
        --gpu-memory-utilization 0.95 \
        --custom-model \
        --enforce-eager
    
    echo "✓ Configuration $config_name completed"
    echo ""
}

# 执行三种配置
run_combined_batch_profile "$CONFIG1_NAME" $CONFIG1_BS1 $CONFIG1_LEN1 $CONFIG1_BS2 $CONFIG1_LEN2

# run_combined_batch_profile "$CONFIG2_NAME" $CONFIG2_BS1 $CONFIG2_LEN1 $CONFIG2_BS2 $CONFIG2_LEN2

# run_combined_batch_profile "$CONFIG3_NAME" $CONFIG3_BS1 $CONFIG3_LEN1 $CONFIG3_BS2 $CONFIG3_LEN2

echo "==================== All configurations completed ===================="
echo "Results saved to: $MAIN_OUTDIR"
