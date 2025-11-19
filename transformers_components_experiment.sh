
BASE_DIR="transformers_components_experiment_5"

python3 test_transformers_green_ctx.py \
  --model /nfs/xjzhang/Qwen/Qwen3-4B \
  --log-dir $BASE_DIR/single_sm50 \
  --sm-partition-sizes 50 \
  --batch-sizes 1 2 4 8 16 \
  --num-repeat 3 \
  --output-length 128 \
  --attention-impl flash_attention_2


python3 test_transformers_parallel_green_ctx.py \
  --model /nfs/xjzhang/Qwen/Qwen3-4B \
  --num-streams 2 \
  --min-sm-per-stream 50 \
  --batch-sizes 1 2 4 8 16 \
  --num-repeat 3 \
  --no-leftover-mode \
  --log-dir $BASE_DIR/parallel_sm50_stream2 \
  --attention-impl flash_attention_2


# python3 aggregate_layer_times_json.py --input-dir $BASE_DIR/parallel_sm50_stream2/ --output $BASE_DIR/parallel_layer_times.csv
# python3 compare_green_ctx_csv.py --baseline $BASE_DIR/baseline_layer_times.csv --parallel $BASE_DIR/parallel_layer_times.csv --output $BASE_DIR/parallel_vs_baseline_attn.csv --metric attn_mean_ms
# python3 compare_green_ctx_csv.py --baseline $BASE_DIR/single_sm50/transformers_green_ctx_benchmark.csv --parallel $BASE_DIR/parallel_sm50_stream2/transformers_parallel_green_ctx_benchmark.csv --output $BASE_DIR/parallel_vs_baseline_tpot.csv --metric tpot_ms

