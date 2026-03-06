#!/bin/bash
# DeepEP 环境快速验证脚本
#
# 用途：快速验证 DeepEP 环境是否正常工作
#
# 使用方法：
#   1. 单机测试: ./verify_env.sh
#   2. 多节点测试:
#        - 在节点 0: ./verify_env.sh internode --rank 0 --world-size 2 --master-addr 10.0.0.1
#        - 在节点 1: ./verify_env.sh internode --rank 1 --world-size 2 --master-addr 10.0.0.1

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# cd "$SCRIPT_DIR"
SCRIPT_DIR="/sgl-workspace/DeepEP"


print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[i] $1${NC}"
}

# 检查 Python 环境
check_python_env() {
    print_header "检查 Python 环境"

    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "未找到 Python 环境"
        exit 1
    fi

    PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
    print_info "Python 路径: $PYTHON_CMD"
    print_info "Python 版本: $($PYTHON_CMD --version)"

    # 检查 PyTorch
    if ! $PYTHON_CMD -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        print_error "未安装 PyTorch"
        exit 1
    fi

    # 检查 CUDA
    CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
    print_info "CUDA 版本: $CUDA_VERSION"

    # 检查 deep_ep 模块
    if ! $PYTHON_CMD -c "import deep_ep; print(f'deep_ep version: {deep_ep.__version__}')" 2>/dev/null; then
        print_error "未安装 deep_ep 模块，请先执行: pip install -e ."
        exit 1
    fi

    print_success "Python 环境检查通过"
}

# 检查 GPU
check_gpus() {
    print_header "检查 GPU"

    GPU_COUNT=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

    if [ "$GPU_COUNT" -eq 0 ]; then
        print_error "未检测到 CUDA GPU"
        exit 1
    fi

    print_info "检测到 $GPU_COUNT 个 GPU"

    for ((i=0; i<$GPU_COUNT; i++)); do
        GPU_NAME=$($PYTHON_CMD <<EOF
import torch
torch.cuda.set_device($i)
print(torch.cuda.get_device_name($i))
EOF
)
        print_info "  GPU $i: $GPU_NAME"
    done

    print_success "GPU 检查通过"
}

# 运行节点内测试
run_intranode_test() {
    print_header "运行节点内通信测试 (test_intranode.py)"

    local num_processes=${1:-8}
    local num_gpus=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())")

    if [ "$num_processes" -gt "$num_gpus" ]; then
        print_warning "请求的进程数 ($num_processes) 大于可用 GPU 数 ($num_gpus)，调整为 $num_gpus"
        num_processes=$num_gpus
    fi

    print_info "使用 $num_processes 个 GPU 进行测试"

    $PYTHON_CMD tests/test_intranode.py \
        --num-processes "$num_processes" \
        --num-tokens 4096 \
        --hidden 7168 \
        --num-topk 8 \
        --num-experts 256
}

# 运行低延迟测试
run_low_latency_test() {
    print_header "运行低延迟通信测试 (test_low_latency.py)"

    local num_processes=${1:-8}
    local num_gpus=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())")

    if [ "$num_processes" -gt "$num_gpus" ]; then
        print_warning "请求的进程数 ($num_processes) 大于可用 GPU 数 ($num_gpus)，调整为 $num_gpus"
        num_processes=$num_gpus
    fi

    print_info "使用 $num_processes 个 GPU 进行测试"

    $PYTHON_CMD tests/test_low_latency.py \
        --num-processes "$num_processes" \
        --num-tokens 128 \
        --hidden 7168 \
        --num-topk 8 \
        --num-experts 288
}

# 运行节点间测试（多节点）
run_internode_test() {
    print_header "运行节点间通信测试 (test_internode.py)"

    local rank=$1
    local world_size=$2
    local master_addr=$3
    local master_port=${4:-8361}

    print_info "节点信息: rank=$rank, world_size=$world_size"
    print_info "主节点: $master_addr:$master_port"

    export WORLD_SIZE=$world_size
    export RANK=$rank
    export MASTER_ADDR=$master_addr
    export MASTER_PORT=$master_port

    $PYTHON_CMD tests/test_internode.py \
        --num-processes 8 \
        --num-tokens 4096 \
        --hidden 7168 \
        --num-topk 8 \
        --num-experts 256
}

# 显示帮助
show_help() {
    cat << EOF
DeepEP 环境快速验证脚本

用法:
  $0 [选项] [测试类型]

测试类型:
  intranode          运行节点内通信测试（默认）
  low-latency        运行低延迟通信测试
  internode          运行节点间通信测试（多节点）
  all                运行所有单机测试

选项:
  -h, --help         显示此帮助信息

节点间测试选项:
  --rank N           当前节点的 rank (0, 1, ...)
  --world-size N     总节点数
  --master-addr IP   master 节点地址
  --master-port N    master 节点端口 (默认: 8361)
  --num-processes N  使用的 GPU 数量 (默认: 8)

示例:
  # 单机测试（默认）
  $0

  # 运行低延迟测试
  $0 low-latency

  # 运行所有单机测试
  $0 all

  # 多节点测试（节点 0）
  $0 internode --rank 0 --world-size 2 --master-addr 10.0.0.1

  # 多节点测试（节点 1）
  $0 internode --rank 1 --world-size 2 --master-addr 10.0.0.1

EOF
}

# 解析参数
TEST_TYPE="intranode"
NUM_PROCESSES=8
RANK=""
WORLD_SIZE=""
MASTER_ADDR=""
MASTER_PORT="8361"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        intranode|intra)
            TEST_TYPE="intranode"
            shift
            ;;
        low-latency|low|ll)
            TEST_TYPE="low-latency"
            shift
            ;;
        internode|inter)
            TEST_TYPE="internode"
            shift
            ;;
        all)
            TEST_TYPE="all"
            shift
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --num-processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主执行逻辑
print_header "DeepEP 环境验证脚本"
print_info "脚本目录: $SCRIPT_DIR"

# 检查环境
check_python_env
check_gpus

# 运行测试
case $TEST_TYPE in
    intranode)
        print_info "运行节点内通信测试"
        run_intranode_test "$NUM_PROCESSES"
        print_success "节点内测试完成"
        ;;
    low-latency)
        print_info "运行低延迟通信测试"
        run_low_latency_test "$NUM_PROCESSES"
        print_success "低延迟测试完成"
        ;;
    internode)
        if [ -z "$RANK" ] || [ -z "$WORLD_SIZE" ] || [ -z "$MASTER_ADDR" ]; then
            print_error "节点间测试需要指定 --rank, --world-size 和 --master-addr"
            echo ""
            show_help
            exit 1
        fi
        run_internode_test "$RANK" "$WORLD_SIZE" "$MASTER_ADDR" "$MASTER_PORT"
        print_success "节点间测试完成"
        ;;
    all)
        print_info "运行所有单机测试"
        run_intranode_test "$NUM_PROCESSES"
        echo ""
        run_low_latency_test "$NUM_PROCESSES"
        print_success "所有单机测试完成"
        ;;
    *)
        print_error "未知测试类型: $TEST_TYPE"
        exit 1
        ;;
esac

print_header "✅ 环境验证完成"
print_success "DeepEP 环境正常工作！"
