


for port in 8315 8316; do
    vllm serve /nfs/xjzhang/Qwen/Qwen3-0.6B/ --port $port --max-num-batched-tokens 100000 --gpu-memory-utilization 0.48 &
done
