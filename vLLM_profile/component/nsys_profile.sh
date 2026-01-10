#!/bin/bash

# Nsight Systems (Nsys) Profile 脚本
# 用于分析三个 benchmark 的 GPU 性能

echo "=== 使用 Nsight Systems Profile 三个 Benchmark ==="
echo ""

export CUDA_VISIBLE_DEVICES=0

# 设置参数（只指定一个 batch_size 和 kv_len）
BATCH_SIZE=${1:-8}
KV_LEN=${2:-2048}
SEQ_LEN=1  # 对于 MLP benchmark，使用 seq_len
MODEL_SIZE=${4:-"4B"}
OUTPUT_DIR=${3:-"nsys_profile_result_${MODEL_SIZE}"}  # 输出目录，默认为 nsys_profile_result

RUN_OUTPUT_DIR="${OUTPUT_DIR}/run_output"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RUN_OUTPUT_DIR}"
echo "配置参数:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  KV length: ${KV_LEN}"
echo "  Sequence length: ${SEQ_LEN}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

# 检查 benchmark 文件是否存在
BENCHMARKS=(
    "benchmark_qwen3_mlp.py"
    "benchmark_flash_attn.py"
    "benchmark_attention_layer.py"
)

for benchmark in "${BENCHMARKS[@]}"; do
    if [ ! -f "${benchmark}" ]; then
        echo "错误: 找不到 ${benchmark}"
        exit 1
    fi
done

# 设置输出文件前缀
OUTPUT_PREFIX="$OUTPUT_DIR/nsys_profile"

# 1. Profile benchmark_qwen3_mlp.py
echo "=========================================="
echo "1. Profile benchmark_qwen3_mlp.py"
echo "=========================================="
OUTPUT_FILE="${OUTPUT_PREFIX}_qwen3_mlp_bs${BATCH_SIZE}_seq${SEQ_LEN}"
echo "输出文件: ${OUTPUT_FILE}.nsys-rep"
echo ""

nsys profile -o "${OUTPUT_FILE}" \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=20000 \
    --gpu-metrics-set=ga100 \
    python3 "${SCRIPT_DIR}/benchmark_qwen3_mlp.py" \
        --batch-sizes ${BATCH_SIZE} \
        --seq-lens 1 \
        --output-dir "${RUN_OUTPUT_DIR}"

if [ $? -eq 0 ]; then
    echo "✓ benchmark_qwen3_mlp.py profile 完成"
    echo "  查看报告: nsys-ui ${OUTPUT_FILE}.nsys-rep"
else
    echo "✗ benchmark_qwen3_mlp.py profile 失败"
fi
echo ""

# 2. Profile benchmark_flash_attn.py
echo "=========================================="
echo "2. Profile benchmark_flash_attn.py"
echo "=========================================="
OUTPUT_FILE="${OUTPUT_PREFIX}_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}"
echo "输出文件: ${OUTPUT_FILE}.nsys-rep"
echo ""

nsys profile -o "${OUTPUT_FILE}" \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=20000 \
    --gpu-metrics-set=ga100 \
    python3 "${SCRIPT_DIR}/benchmark_flash_attn.py" \
        --batch-sizes ${BATCH_SIZE} \
        --kv-lens ${KV_LEN} \
        --output-dir "${RUN_OUTPUT_DIR}"

if [ $? -eq 0 ]; then
    echo "✓ benchmark_flash_attn.py profile 完成"
    echo "  查看报告: nsys-ui ${OUTPUT_FILE}.nsys-rep"
else
    echo "✗ benchmark_flash_attn.py profile 失败"
fi
echo ""

# 3. Profile benchmark_attention_layer.py
echo "=========================================="
echo "3. Profile benchmark_attention_layer.py"
echo "=========================================="
OUTPUT_FILE="${OUTPUT_PREFIX}_attention_layer_bs${BATCH_SIZE}_kv${KV_LEN}"
echo "输出文件: ${OUTPUT_FILE}.nsys-rep"
echo ""

nsys profile -o "${OUTPUT_FILE}" \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=20000 \
    --gpu-metrics-set=ga100 \
    python3 "${SCRIPT_DIR}/benchmark_attention_layer.py" \
        --batch-sizes ${BATCH_SIZE} \
        --kv-lens ${KV_LEN} \
        --output-dir "${RUN_OUTPUT_DIR}"

if [ $? -eq 0 ]; then
    echo "✓ benchmark_attention_layer.py profile 完成"
    echo "  查看报告: nsys-ui ${OUTPUT_FILE}.nsys-rep"
else
    echo "✗ benchmark_attention_layer.py profile 失败"
fi
echo ""

echo "=========================================="
echo "=== 所有 Profile 完成 ==="
echo "=========================================="
echo ""
echo "生成的报告文件:"
echo "  ${OUTPUT_PREFIX}_qwen3_mlp_bs${BATCH_SIZE}_seq${SEQ_LEN}.nsys-rep"
echo "  ${OUTPUT_PREFIX}_flash_attn_bs${BATCH_SIZE}_kv${KV_LEN}.nsys-rep"
echo "  ${OUTPUT_PREFIX}_attention_layer_bs${BATCH_SIZE}_kv${KV_LEN}.nsys-rep"
echo ""
echo "Benchmark 结果文件保存在: ${OUTPUT_DIR}/"
echo "  - benchmark_qwen3_mlp_results.csv"
echo "  - benchmark_flash_attn_results.csv"
echo "  - benchmark_attention_layer_results.csv"
echo "  - 以及相应的图表文件 (.png)"
echo ""
echo "查看报告:"
echo "  nsys-ui <report_file>.nsys-rep"
echo ""
echo "导出统计信息:"
echo "  nsys stats --report cuda_gpu_trace --format csv <report_file>.nsys-rep"
echo ""

