#!/bin/bash
set -e
# 定义一个多行字符串，模拟元组列表（每行一个元组，元素用空格分隔）
tuples=$(cat <<EOF
2 50
EOF
)


BASE_DIR="component_debug"
SM_NUM=50
COMPONENTS=("layer_0_ffn")

for COMPONENT in "${COMPONENTS[@]}"; do
    # 循环遍历，每行解包到变量中
    while read -r STREAM_NUM SM; do
        # 在这里使用解包后的变量执行命令行操作
        echo "STREAM_NUM: $STREAM_NUM, SM: $SM"

        python3 test_transformers_component_benchmark.py   --model /nfs/xjzhang/Qwen/Qwen3-4B   --batch-sizes 2   --seq-lengths 1   --sm-partition-sizes $SM   --num-repeat 1   --warmup 1  --attention-impl flash_attention_2  --component-name $COMPONENT --log-dir $BASE_DIR/benchmark_sm${SM}_stream${STREAM_NUM}_component${COMPONENT} --debug-shapes
        # python3 test_transformers_parallel_components_green_ctx.py   --model /nfs/xjzhang/Qwen/Qwen3-4B --num-streams $STREAM_NUM --min-sm-per-stream $SM   --batch-sizes 1 2 4 8 16 32 64 128  --seq-lengths 1 --num-repeat 5 --warmup 5 --attention-impl flash_attention_2 --component-name $COMPONENT --no-leftover-mode --log-dir $BASE_DIR/benchmark_parallel_sm${SM}_stream${STREAM_NUM}_component${COMPONENT}
        # 示例：放到命令行里执行其他命令，比如 mv 或 cp
        # mv "${fruit}_old.txt" "${fruit}_${color}.txt"
    done <<< "$tuples"
done

python3 test_transformers_green_ctx.py \
  --model /nfs/xjzhang/Qwen/Qwen3-4B \
  --log-dir $BASE_DIR/single_sm${SM_NUM} \
  --sm-partition-sizes ${SM_NUM} \
  --batch-sizes 2 \
  --num-repeat 1 \
  --output-length 4 \
  --attention-impl flash_attention_2 \
  --debug-shapes
