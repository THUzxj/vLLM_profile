set -x

# export CUDA_VISIBLE_DEVICES=0
# vllm serve /nfs/xjzhang/Qwen/Qwen3-4B/ --port 8315 --max-num-batched-tokens 131072
# with MPS:  vllm serve /nfs/xjzhang/Qwen/Qwen3-4B/ --port 8315 --max-num-batched-tokens 65536


for iter in 1 2 3 4 5; do
    for batch_size in 1 2 4 8 ; do
    # for batch_size in 1 2 4 8 16 32; do
        num_prompts=$((batch_size * 10))
        MPS_SIZE=100
        RESULT_DIR="mps_check/benchmark_results_20251130_serving_0_11_2_mps_${MPS_SIZE}_iter${iter}"
        vllm bench serve --backend openai --base-url http://localhost:8315 --model /nfs/xjzhang/Qwen/Qwen3-4B/ --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts $num_prompts --request-rate inf --save-result --result-dir $RESULT_DIR --max-concurrency $batch_size  --result-filename serving_in512_out128_mbt131072_bs$batch_size.json --percentile-metrics ttft,tpot,e2el --num-warmups 5
    done
done
