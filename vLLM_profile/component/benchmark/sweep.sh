
set -e
export CUDA_VISIBLE_DEVICES=2

# "0.6B" "4B"
for model_size in "4B" "32B"; do
    OUTPUT_DIR="sweep_profile_result_${model_size}_rep2"
    mkdir -p ${OUTPUT_DIR}

    python benchmark/benchmark_flash_attn_v2.py --batch-sizes 1 2 4 8 16 32 64 128 256 --kv-lens 512 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR} --model-size ${model_size}
    # python3 benchmark/benchmark_attention_layer.py --batch-sizes 1 2 4 8 16 32 64 128 256 --kv-lens 512 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR} --model-size ${model_size}
    # python benchmark/benchmark_qwen3_mlp.py --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 --seq-lens 1 --output-dir ${OUTPUT_DIR} --model-size ${model_size}
done
