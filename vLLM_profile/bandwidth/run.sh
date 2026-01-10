export CUDA_VISIBLE_DEVICES=3

for batch_size in 1 2 4 8 16 32 48 64 80 128; do
    for seq_len in 1024; do
        output_file="bandwidth_profile_llama2_7b_bs${batch_size}_seq${seq_len}.csv"
        python attention_bandwidth_profiler.py --model /nfs/xjzhang/meta-llama/Llama-2-7b-hf \
            --avg-seq-len ${seq_len} \
            --batch-size ${batch_size} \
            --output-file ${output_file}
    done
done
