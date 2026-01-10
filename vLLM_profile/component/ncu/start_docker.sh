docker run --gpus all -d \
 -v  /data/xjzhang:/xingjian \
 -v /nfs/xjzhang:/nfs/xjzhang \
 --privileged nvidia/cuda:12.9.0-devel-ubuntu22.04 sleep infinity 


