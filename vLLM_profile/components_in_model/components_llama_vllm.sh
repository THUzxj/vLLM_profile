MODEL=/nfs/xjzhang/meta-llama/Llama-2-7b-hf
MODEL_NAME=${MODEL##*/}
DATE=`date +%Y%m%d_%H%M%S`
set -e

export CUDA_VISIBLE_DEVICES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

VERSION="v1_component_debug"

batched_tokens=70000
# kv_cache_usage=$((batched_tokens * 144 * 1024 + 64 * 1024))
echo "KV Cache Memory Bytes: ${kv_cache_usage}"

trial_id=7

# for batch_size in 1 2 4 8 16 32 64 128; do
#     for input_len in 512 1024 2048 4096 8192; do
for batch_size in 1 2 4 8 16 32 64 80 128; do
    for input_len in 1024; do
        export PROFILE_COMPONENT_OUTPUT_DIR="components_results_${trial_id}/${MODEL_NAME}/in${input_len}_bs${batch_size}_${VERSION}_${DATE}"
        
        # skip if batched tokens exceed 100000
        BATCHED_TOKENS=$((batch_size * input_len))
        # if [ $BATCHED_TOKENS -gt $batched_tokens ] || [ $BATCHED_TOKENS -lt 100000 ]; then
        if [ $BATCHED_TOKENS -gt $batched_tokens ]; then
            echo "Skipping BS=${batch_size}, IN=${input_len} as batched tokens ${BATCHED_TOKENS} exceed limit."
            continue
        fi


        # debug
        python3 ../tpot/benchmark/profile_vllm_cli.py \
            --model $MODEL \
            --batched-tokens $batched_tokens \
            --batch-sizes $batch_size \
            --prompt-length $input_len \
            --output-len 16 \
            --window-size 200 \
            --num-repeat 1 \
            --log-dir results_${trial_id}/vllm_step_${MODEL_NAME}_${VERSION}_bs${batch_size}_in${input_len}_${DATE} \
            --gpu-memory-utilization 0.95 \
            --max-model-len 1050 \
            --custom-model \
            --enforce-eager

        # python3 ../tpot/benchmark/profile_vllm_cli.py \
        #     --model $MODEL \
        #     --batched-tokens 150000 \
        #     --batch-sizes $batch_size \
        #     --prompt-length $input_len \
        #     --output-len 16 \
        #     --window-size 200 \
        #     --num-repeat 1 \
        #     --log-dir results_3/vllm_step_${MODEL_NAME}_${VERSION}_bs${batch_size}_in${input_len}_${DATE} \
        #     --gpu-memory-utilization 0.95 \
        #     --custom-model \
        #     --enforce-eager
    done
done
