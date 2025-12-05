set -x

# export CUDA_VISIBLE_DEVICES=0
# vllm serve /nfs/xjzhang/Qwen/Qwen3-4B/ --port 8315 --max-num-batched-tokens 131072
# with MPS:  vllm serve /nfs/xjzhang/Qwen/Qwen3-4B/ --port 8315 --max-num-batched-tokens 65536

model_id=/nfs/xjzhang/Qwen/Qwen3-0.6B/
model_name=Qwen3-0.6B
max_batched_tokens=100000
for iter in 1 2 3 4 5; do
    # for batch_size in 1 2 4 8 16 32; do
    # batch_size=8
    for batch_size in 1 2 4 8 16 32 64 128 256; do
        num_prompts=$((batch_size * 1))
        input_len=1024
        output_len=128
        MPS_SIZE="100"
        RESULT_DIR="bs_test_model${model_name}_mps${MPS_SIZE}/benchmark_results_20251130_serving_0_11_2_mps_${MPS_SIZE}_iter${iter}"
        vllm bench serve --backend openai --base-url http://localhost:8315 --model $model_id \
        --dataset-name random --random-input-len $input_len --random-output-len $output_len --num-prompts $num_prompts \
        --request-rate inf --max-concurrency $batch_size --save-result --result-dir $RESULT_DIR \
        --result-filename serving_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs$batch_size.json \
        --percentile-metrics ttft,tpot,e2el --num-warmups 5
    done
done
