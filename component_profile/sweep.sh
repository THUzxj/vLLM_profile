OUTPUT_DIR="sweep_profile_result"

mkdir -p ${OUTPUT_DIR}

# python benchmark_flash_attn.py --batch-sizes 1 2 4 8 16 32 64 128 256 --kv-lens 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR}
# python benchmark_attention_layer.py --batch-sizes 1 2 4 8 16 32 64 128 256 --kv-lens 1024 2048 4096 8192 --output-dir ${OUTPUT_DIR}
python benchmark_qwen3_mlp.py --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 --seq-lens 1 --output-dir ${OUTPUT_DIR}