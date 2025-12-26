
BASE_DIR="transformers_whole_experiment_3"

python3 test_transformers_green_ctx.py \
  --model /nfs/xjzhang/Qwen/Qwen3-4B \
  --log-dir $BASE_DIR/profile_transformers_green_ctx_trace_sm32 \
  --sm-partition-sizes 50 \
  --batch-sizes 1 2 4 8 16 \
  --num-repeat 3 \
  --output-length 128


python3 test_transformers_parallel_green_ctx.py \
  --model /nfs/xjzhang/Qwen/Qwen3-4B \
  --num-streams 2 \
  --min-sm-per-stream 50 \
  --batch-sizes 1 2 4 8 16 \
  --num-repeat 3 \
  --no-leftover-mode \
  --log-dir $BASE_DIR/benchmark_parallel_sm50_stream2 \
  --attention-impl flash_attention_2
