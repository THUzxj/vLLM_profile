set -x

# vllm serve /nfs/xjzhang/Qwen/Qwen3-4B/ --port 8315 --max-num-batched-tokens 131072 

model_id=/nfs/xjzhang/Qwen/Qwen3-8B/
model_name=Qwen3-8B
batch_size=4
output_len=128
max_batched_tokens=100000
for iter in 1 2 3 4 5; do
    for input_len in 512 1024 2048 4096 8192 16384; do
    # for batch_size in 1 2 4 8 16 32; do
        num_prompts=$((batch_size))
        RESULT_DIR="length_test_model${model_name}_bs${batch_size}_out${output_len}/benchmark_results_length_20251130_serving_0_11_2_no_mps_iter${iter}"
        vllm bench serve --backend openai --base-url http://localhost:8315 \
            --model $model_id --dataset-name random --random-input-len $input_len \
            --random-output-len $output_len --num-prompts $num_prompts --request-rate inf --save-result \
            --result-dir $RESULT_DIR --max-concurrency $batch_size \
            --result-filename serving_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs$batch_size.json \
            --percentile-metrics ttft,tpot,e2el --num-warmups 5
    done
done