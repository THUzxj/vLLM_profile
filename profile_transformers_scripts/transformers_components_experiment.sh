
BASE_DIR="transformers_components_experiment_partition2_11243"
STREAM_NUM=1
SM_NUM=100

# python3 test_transformers_green_ctx.py \
#   --model /nfs/xjzhang/Qwen/Qwen3-4B \
#   --log-dir $BASE_DIR/single_sm${SM_NUM} \
#   --sm-partition-sizes ${SM_NUM} \
#   --batch-sizes 1 2 4 8 16 \
#   --num-repeat 3 \
#   --output-length 128 \
#   --attention-impl flash_attention_2


python3 test_transformers_parallel_green_ctx.py \
  --model /nfs/xjzhang/Qwen/Qwen3-4B \
  --num-streams ${STREAM_NUM} \
  --min-sm-per-stream ${SM_NUM} \
  --batch-sizes 1 2 4 8 16 \
  --num-repeat 3 \
  --no-leftover-mode \
  --log-dir $BASE_DIR/parallel_sm${SM_NUM}_stream${STREAM_NUM} \
  --attention-impl flash_attention_2


# python3 aggregate_layer_times_json.py --input-dir $BASE_DIR/parallel_sm50_stream2/ --output $BASE_DIR/parallel_layer_times.csv
# python3 compare_green_ctx_csv.py --baseline $BASE_DIR/baseline_layer_times.csv --parallel $BASE_DIR/parallel_layer_times.csv --output $BASE_DIR/parallel_vs_baseline_attn.csv --metric attn_mean_ms
# python3 compare_green_ctx_csv.py --baseline $BASE_DIR/single_sm50/transformers_green_ctx_benchmark.csv --parallel $BASE_DIR/parallel_sm50_stream2/transformers_parallel_green_ctx_benchmark.csv --output $BASE_DIR/parallel_vs_baseline_tpot.csv --metric tpot_ms

