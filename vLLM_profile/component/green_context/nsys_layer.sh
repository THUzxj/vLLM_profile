
export CUDA_VISIBLE_DEVICES=1
# num_groups=2
MIN_COUNT=52
batch_size=2
seq_len=1
MODEL_SIZE="32B"
SCRIPT_PATH="./test_qwen3_decoder_layer_parallel.py"
DEVICE="cuda:0"
DTYPE="bfloat16"
WARMUP_ITERATIONS=10
BENCHMARK_ITERATIONS=101
OUTPUT_DIR="nsys_profile_result/layer_${MODEL_SIZE}_greencontext_SM${MIN_COUNT}_ngroups${num_groups}_bs${batch_size}_seqlen${seq_len}_iter${BENCHMARK_ITERATIONS}_v1"
OUTPUT_CSV="${OUTPUT_DIR}/sweep_results.csv"
mkdir -p ${OUTPUT_DIR}

for num_groups in 1 2; do
        nsys profile -o "${OUTPUT_DIR}/profile_output_layer_numgroups${num_groups}" \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --cudabacktrace=true \
        --cuda-graph-trace=node \
        --force-overwrite true \
        python "$SCRIPT_PATH" \
                --num-groups "$num_groups" \
                --min-count "$MIN_COUNT" \
                --batch-size "$batch_size" \
                --seq-len "$seq_len" \
                --model-size "$MODEL_SIZE" \
                --device "$DEVICE" \
                --dtype "$DTYPE" \
                --warmup-iterations "$WARMUP_ITERATIONS" \
                --benchmark-iterations "$BENCHMARK_ITERATIONS" \
                --output-csv "$OUTPUT_CSV"
done
