#!/bin/bash
# the following must be performed with root privilege
# >>> sudo sh stop_mps.sh
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mps_dir>"
    echo "bash stop_mps.sh /data/xjzhang/mps"
    exit 1
fi

MPSDIR=$1

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log

echo quit | nvidia-cuda-mps-control
# pkill -f nvidia-cuda-mps-control

# rm -rf ${MPSDIR}/nvidia-mps
# rm -rf ${MPSDIR}/nvidia-log
