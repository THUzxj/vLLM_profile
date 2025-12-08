export CUDA_VISIBLE_DEVICES=0         # 这里以GPU0为例，其他卡类似
export CUDA_MPS_PIPE_DIRECTORY=/data/xjzhang/mps/nvidia-mps # Select a location that's accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=/data/xjzhang/mps/nvidia-log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

# ====== 启动 =========
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS  # 让GPU0变为独享模式。
nvidia-cuda-mps-control -d            # 开启mps服务 
# ====== 查看 =========
ps -ef | grep mps                     # 启动成功后能看到相应的进程


echo "set_default_active_thread_percentage 50" | nvidia-cuda-mps-control

echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control
# ====== 停止 =========
echo quit | nvidia-cuda-mps-control   # 关闭mps服务     
nvidia-smi -i 0 -c DEFAULT       # 让GPU恢复为默认模式。

export VLLM_LOGGING_LEVEL=DEBUG
python3 profile_vllm_cli.py --distributed-executor-backend uni --log-dir profile_results_uni_mps_50
