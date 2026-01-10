
export MPSDIR=/xingjian/vLLM_profile/component_profile/mps
mkdir -p $MPSDIR
export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
mkdir -p ${MPSDIR}/nvidia-mps
mkdir -p ${MPSDIR}/nvidia-log

nvidia-smi -i 1 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d

ncu --mps control \
    -o mps_profile \
    --set full