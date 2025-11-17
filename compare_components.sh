#!/bin/bash
set -e
# 定义一个多行字符串，模拟元组列表（每行一个元组，元素用空格分隔）
tuples=$(cat <<EOF
1 106
2 50
3 32
4 26
5 20
EOF
)


BASE_DIR="model_reload_profile_2"
COMPONENTS=("layer_0_ffn" "layer_0_attn")


for COMPONENT in "${COMPONENTS[@]}"; do
    # 循环遍历，每行解包到变量中
    while read -r STREAM_NUM SM; do
        # 在这里使用解包后的变量执行命令行操作
        echo "STREAM_NUM: $STREAM_NUM, SM: $SM"

        python3 test_transformers_component_benchmark.py   --model /nfs/xjzhang/Qwen/Qwen3-4B   --batch-sizes 1 2 4 8 16 32 64 128   --seq-lengths 512   --sm-partition-sizes $SM   --num-repeat 5   --warmup 5   --attention-impl flash_attention_2  --component-name $COMPONENT --log-dir $BASE_DIR/benchmark_sm${SM}_stream${STREAM_NUM}_component${COMPONENT}
        python3 test_transformers_parallel_components_green_ctx.py   --model /nfs/xjzhang/Qwen/Qwen3-4B --num-streams $STREAM_NUM --min-sm-per-stream $SM   --batch-sizes 1 2 4 8 16 32 64 128  --seq-lengths 512 --num-repeat 5 --warmup 5 --attention-impl flash_attention_2 --component-name $COMPONENT --no-leftover-mode --log-dir $BASE_DIR/benchmark_parallel_sm${SM}_stream${STREAM_NUM}_component${COMPONENT}
        # 示例：放到命令行里执行其他命令，比如 mv 或 cp
        # mv "${fruit}_old.txt" "${fruit}_${color}.txt"
    done <<< "$tuples"
done
