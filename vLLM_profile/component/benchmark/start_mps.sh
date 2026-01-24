
export CUDA_VISIBLE_DEVICES=1
export MPSDIR=$PWD/mps_files_host
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
mkdir -p ${MPSDIR}/nvidia-mps
mkdir -p ${MPSDIR}/nvidia-log
nvidia-cuda-mps-control -d

nvidia-smi -i 1 -c EXCLUSIVE_PROCESS

echo "quit" | nvidia-cuda-mps-control
