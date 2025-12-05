

# export CUDA_VISIBLE_DEVICES=0
# vllm serve /nfs/xjzhang/Qwen/Qwen3-4B/ --port 8315 --max-num-batched-tokens 131072
# with MPS:  vllm serve /nfs/xjzhang/Qwen/Qwen3-4B/ --port 8315 --max-num-batched-tokens 65536

model_id=/nfs/xjzhang/Qwen/Qwen3-0.6B/
model_name=Qwen3-0.6B
max_batched_tokens=280000
output_len=64
MPS_SIZE="no"
for iter in 1 2 3 4 5; do
    # for batch_size in 1 2 4 8 16 32; do
    # batch_size=8

    for input_len in 128 256 512 1024 2048 4096 8192 16384; do
        for batch_size in 1 2 4 8 16 32 64 128 ; do #  256 512 1024 2048
            # skip if (input_len+output_len) * batch_size > max_batched_tokens
            if (( (input_len + output_len) * batch_size > max_batched_tokens )); then
                echo "skip: input_len=${input_len}, output_len=${output_len}, batch_size=${batch_size} -> (input_len+output_len)*batch_size=$(( (input_len + output_len) * batch_size )) > max_batched_tokens=${max_batched_tokens}"
                continue
            fi

            actual_input_len=$(($input_len - $output_len / 2))

            echo "actual input len: $actual_input_len"

            num_prompts=$((batch_size * 1))
            RESULT_DIR="bs_len_test_model${model_name}_mps${MPS_SIZE}_out${output_len}/benchmark_results_20251130_serving_0_11_2_mps_${MPS_SIZE}_iter${iter}"
            vllm bench serve --backend openai --base-url http://localhost:8315 --model $model_id \
            --dataset-name random --random-input-len $actual_input_len --random-output-len $output_len --num-prompts $num_prompts \
            --request-rate inf --max-concurrency $batch_size --save-result --result-dir $RESULT_DIR \
            --result-filename serving_in${input_len}_out${output_len}_mbt${max_batched_tokens}_bs$batch_size.json \
            --percentile-metrics ttft,tpot,e2el --num-warmups 5
        done
    done
done
