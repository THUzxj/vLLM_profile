
BASE_DIR="transformers_components_experiment_5"

python3 aggregate_layer_times_json.py --input-dir $BASE_DIR/parallel_sm50_stream2/ --output $BASE_DIR/parallel_layer_times.csv
python3 aggregate_layer_times_json.py --input-dir $BASE_DIR/single_sm50/ --output $BASE_DIR/baseline_layer_times.csv
python3 compare_green_ctx_csv.py --baseline $BASE_DIR/baseline_layer_times.csv --parallel $BASE_DIR/parallel_layer_times.csv --output $BASE_DIR/parallel_vs_baseline_attn.csv --metric attn_mean_ms
python3 compare_green_ctx_csv.py --baseline $BASE_DIR/baseline_layer_times.csv --parallel $BASE_DIR/parallel_layer_times.csv --output $BASE_DIR/parallel_vs_baseline_ffn.csv --metric ffn_mean_ms
python3 compare_green_ctx_csv.py --baseline $BASE_DIR/single_sm50/transformers_green_ctx_benchmark.csv --parallel $BASE_DIR/parallel_sm50_stream2/transformers_parallel_green_ctx_benchmark.csv --output $BASE_DIR/parallel_vs_baseline_tpot.csv --metric tpot_ms