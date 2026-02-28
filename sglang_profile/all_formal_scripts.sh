# For H200 GPUs
NNODE=1 DP=8 EP=8 TP=8 IL="32000 64000 128000" bash run_one_batch_qwen3-235B-A22B_formal_H.sh
# 分开跑的版本：
NNODE=1 DP=8 EP=8 TP=8 IL=32000 bash run_one_batch_qwen3-235B-A22B_formal_H.sh
NNODE=1 DP=8 EP=8 TP=8 IL=64000 bash run_one_batch_qwen3-235B-A22B_formal_H.sh
NNODE=1 DP=8 EP=8 TP=8 IL=128000 bash run_one_batch_qwen3-235B-A22B_formal_H.sh

NNODE=2 DP=16 EP=16 TP=16 IL="32000 64000 128000" bash run_one_batch_qwen3-235B-A22B_formal_H.sh
NNODE=4 DP=32 EP=32 TP=32 IL="32000 64000 128000" bash run_one_batch_qwen3-235B-A22B_formal_H.sh
# 或者分开input length跑
NNODE=1 DP=8 EP=8 TP=8    IL="32000 64000 128000" bash run_one_batch_deepseek-v3_formal_H.sh
NNODE=2 DP=16 EP=16 TP=16 IL="32000 64000 128000" bash run_one_batch_deepseek-v3_formal_H.sh
NNODE=4 DP=32 EP=32 TP=32 IL="32000 64000 128000" bash run_one_batch_deepseek-v3_formal_H.sh
