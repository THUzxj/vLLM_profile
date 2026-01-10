#!/bin/bash

# 从ncu-rep文件中提取qwen3_mlp的3个kernel（act_and_mul_kernel和两个ampere或Kernel2开头的kernel）
# 记录每个kernel的完整指标：compute_throughput, memory_throughput, duration, device_memory_mb, device_memory_bandwidth_gbs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCU_REPORTS_DIR="${SCRIPT_DIR}/ncu_profile_result_v4_4B"

# 检查ncu命令是否存在
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu command not found. Please install Nsight Compute." >&2
    exit 1
fi

# 查找所有ncu-rep文件
ncu_files=($(find "${NCU_REPORTS_DIR}" -name "ncu_report_qwen3_mlp_*.ncu-rep" | sort))

if [ ${#ncu_files[@]} -eq 0 ]; then
    echo "Error: No ncu-rep files found in ${NCU_REPORTS_DIR}" >&2
    exit 1
fi

echo "Found ${#ncu_files[@]} ncu-rep files"
echo ""

# 创建输出CSV文件
OUTPUT_CSV="${NCU_REPORTS_DIR}/qwen3_mlp_kernel_metrics_3kernels.csv"

# 创建CSV文件并写入表头
echo "batch_size,seq_len,kernel_name,compute_throughput,memory_throughput,duration,device_memory_mb,device_memory_bandwidth_gbs" > "${OUTPUT_CSV}"

# 提取单个kernel的指标函数
extract_kernel_metrics() {
    local kernel_name="$1"
    local ncu_file="$2"
    
    local compute_throughput="N/A"
    local memory_throughput="N/A"
    local duration="N/A"
    local device_memory_mb="N/A"
    local device_memory_bandwidth_gbs="N/A"
    
    # 方法1: 使用--print-summary提取throughput和duration
    temp_output=$(mktemp)
    
    ncu --import "${ncu_file}" \
        --print-summary none \
        --kernel-name "${kernel_name}" \
        > "${temp_output}" 2>&1

    if [ $? -eq 0 ]; then
        # 在summary中查找throughput信息和duration
        while IFS= read -r line; do
            line_lower=$(echo "${line}" | tr '[:upper:]' '[:lower:]')
            if [[ ${line_lower} == *"throughput"* ]]; then
                # 提取数值（支持科学计数法）
                value=$(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?' | head -1)
                if [ -n "${value}" ]; then
                    if [[ ${line_lower} == *"compute"* ]] || [[ ${line_lower} == *"sm"* ]] || [[ ${line_lower} == *"alu"* ]] || [[ ${line_lower} == *"tensor"* ]]; then
                        if [ "${compute_throughput}" == "N/A" ]; then
                            compute_throughput="${value}"
                        fi
                    elif [[ ${line_lower} == *"memory"* ]] || [[ ${line_lower} == *"dram"* ]] || [[ ${line_lower} == *"l2"* ]] || [[ ${line_lower} == *"l1"* ]]; then
                        if [ "${memory_throughput}" == "N/A" ]; then
                            memory_throughput="${value}"
                        fi
                    fi
                fi
            fi
            # 查找duration相关信息
            if [[ ${line_lower} == *"duration"* ]] ||  [[ ${line_lower} == *"time"* ]]; then
                # 提取数值（支持科学计数法，可能包含单位如us, ms, s）
                value=$(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?' | head -1)

                if [ -n "${value}" ] && [ "${duration}" == "N/A" ]; then
                    # 检查单位并转换为毫秒（如果可能）
                    if [[ ${line_lower} == *"ms"* ]] || [[ ${line_lower} == *"millisecond"* ]]; then
                        # 如果是毫秒，保持不变
                        duration="${value}"
                    elif ([[ ${line_lower} == *"second"* ]] && [[ ${line_lower} != *"millisecond"* ]] && [[ ${line_lower} != *"microsecond"* ]] && [[ ${line_lower} != *"nanosecond"* ]]) || ([[ ${line_lower} == *"s"* ]] && [[ ${line_lower} != *"ms"* ]] && [[ ${line_lower} != *"us"* ]] && [[ ${line_lower} != *"ns"* ]] && [[ ${line_lower} != *"second"* ]]); then
                        # 如果是秒，转换为毫秒（乘以1000）
                        duration=$(awk "BEGIN {printf \"%.6f\", ${value} * 1000}" 2>/dev/null || echo "${value}")
                    elif [[ ${line_lower} == *"ns"* ]] || [[ ${line_lower} == *"nanosecond"* ]]; then
                        # 如果是纳秒，转换为毫秒（除以1000000）
                        duration=$(awk "BEGIN {printf \"%.6f\", ${value} / 1000000}" 2>/dev/null || echo "${value}")
                    elif [[ ${line_lower} == *"us"* ]] || [[ ${line_lower} == *"microsecond"* ]]; then
                        # 如果是微秒，转换为毫秒（除以1000）
                        duration=$(awk "BEGIN {printf \"%.6f\", ${value} / 1000}" 2>/dev/null || echo "${value}")
                    else
                        # 默认假设是微秒，转换为毫秒（除以1000）
                        duration=$(awk "BEGIN {printf \"%.6f\", ${value} / 1000}" 2>/dev/null || echo "${value}")
                    fi
                fi
            fi
        done < "${temp_output}"
    fi
    
    # 清理临时文件
    rm -f "${temp_output}"
   
    # 提取device memory读写量: dram__bytes_read.sum + dram__bytes_write.sum
    if [ "${device_memory_mb}" == "N/A" ]; then
        temp_details=$(mktemp)
        ncu --import "${ncu_file}" \
            --print-details all \
            --kernel-name "${kernel_name}" \
            --section "MemoryWorkloadAnalysis_Tables" \
            --print-units base \
            > "${temp_details}" 2>&1
        
        if [ $? -eq 0 ]; then
            bytes_read=""
            bytes_write=""
            bandwidth_read=""
            bandwidth_write=""
            
            # 从输出中提取数值
            while IFS= read -r line; do
                line_lower=$(echo "${line}" | tr '[:upper:]' '[:lower:]')
                # 匹配 dram__bytes_read.sum 行，提取数值
                if [[ ${line_lower} == *"dram__bytes_read.sum "* ]]; then
                    # 提取所有数值，第一个通常是总量，后面可能有带宽值
                    values=($(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?'))
                    if [ ${#values[@]} -ge 1 ]; then
                        bytes_read="${values[0]}"
                    fi
                    # 查找throughput或bandwidth相关的值（通常在后面的列）
                    if [[ ${line_lower} == *"throughput"* ]] || [[ ${line_lower} == *"bandwidth"* ]] || [[ ${line_lower} == *"/s"* ]]; then
                        # 尝试提取带宽值（可能是第二个或第三个数值）
                        if [ ${#values[@]} -ge 2 ]; then
                            bandwidth_read="${values[1]}"
                        elif [ ${#values[@]} -ge 1 ]; then
                            bandwidth_read="${values[0]}"
                        fi
                    fi
                # 匹配 dram__bytes_write.sum 行，提取数值
                elif [[ ${line_lower} == *"dram__bytes_write.sum "* ]]; then
                    values=($(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?'))
                    if [ ${#values[@]} -ge 1 ]; then
                        bytes_write="${values[0]}"
                    fi
                    # 查找throughput或bandwidth相关的值
                    if [[ ${line_lower} == *"throughput"* ]] || [[ ${line_lower} == *"bandwidth"* ]] || [[ ${line_lower} == *"/s"* ]]; then
                        if [ ${#values[@]} -ge 2 ]; then
                            bandwidth_write="${values[1]}"
                        elif [ ${#values[@]} -ge 1 ]; then
                            bandwidth_write="${values[0]}"
                        fi
                    fi
                # 单独查找带宽相关的行（可能在不同的格式中）
                elif [[ ${line_lower} == *"dram__bytes_read"* ]] && ([[ ${line_lower} == *"throughput"* ]] || [[ ${line_lower} == *"bandwidth"* ]] || [[ ${line_lower} == *"/s"* ]]); then
                    value=$(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?' | head -1)
                    if [ -n "${value}" ] && [ -z "${bandwidth_read}" ]; then
                        bandwidth_read="${value}"
                    fi
                elif [[ ${line_lower} == *"dram__bytes_write"* ]] && ([[ ${line_lower} == *"throughput"* ]] || [[ ${line_lower} == *"bandwidth"* ]] || [[ ${line_lower} == *"/s"* ]]); then
                    value=$(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?' | head -1)
                    if [ -n "${value}" ] && [ -z "${bandwidth_write}" ]; then
                        bandwidth_write="${value}"
                    fi
                fi
            done < "${temp_details}"
            
            # 如果找到了两个值，计算总和并转换为MB
            if [ -n "${bytes_read}" ] && [ -n "${bytes_write}" ]; then
                total_bytes=$(awk "BEGIN {printf \"%.6f\", ${bytes_read} + ${bytes_write}}" 2>/dev/null || echo "0")
                if [ -n "${total_bytes}" ] && [ "${total_bytes}" != "0" ]; then
                    device_memory_mb=$(awk "BEGIN {printf \"%.6f\", ${total_bytes} / 1048576}" 2>/dev/null || echo "N/A")
                fi
            elif [ -n "${bytes_read}" ]; then
                # 只有read值
                device_memory_mb=$(awk "BEGIN {printf \"%.6f\", ${bytes_read} / 1048576}" 2>/dev/null || echo "N/A")
            elif [ -n "${bytes_write}" ]; then
                # 只有write值
                device_memory_mb=$(awk "BEGIN {printf \"%.6f\", ${bytes_write} / 1048576}" 2>/dev/null || echo "N/A")
            fi
            
            # 计算带宽总和并转换为GB/s (除以10^9)
            if [ -n "${bandwidth_read}" ] && [ -n "${bandwidth_write}" ]; then
                total_bandwidth=$(awk "BEGIN {printf \"%.6f\", ${bandwidth_read} + ${bandwidth_write}}" 2>/dev/null)
                if [ -n "${total_bandwidth}" ]; then
                    device_memory_bandwidth_gbs=$(awk "BEGIN {printf \"%.6f\", ${total_bandwidth} / 1000000000}" 2>/dev/null || echo "N/A")
                fi
            elif [ -n "${bandwidth_read}" ]; then
                # 只有read带宽值
                device_memory_bandwidth_gbs=$(awk "BEGIN {printf \"%.6f\", ${bandwidth_read} / 1000000000}" 2>/dev/null || echo "N/A")
            elif [ -n "${bandwidth_write}" ]; then
                # 只有write带宽值
                device_memory_bandwidth_gbs=$(awk "BEGIN {printf \"%.6f\", ${bandwidth_write} / 1000000000}" 2>/dev/null || echo "N/A")
            fi
        fi
        rm -f "${temp_details}"
    fi
    
    # 使用全局变量返回结果（bash函数无法直接返回多个值）
    EXTRACTED_COMPUTE_THROUGHPUT="${compute_throughput}"
    EXTRACTED_MEMORY_THROUGHPUT="${memory_throughput}"
    EXTRACTED_DURATION="${duration}"
    EXTRACTED_DEVICE_MEMORY_MB="${device_memory_mb}"
    EXTRACTED_DEVICE_MEMORY_BANDWIDTH_GBS="${device_memory_bandwidth_gbs}"
}

# 处理每个文件
for ncu_file in "${ncu_files[@]}"; do
    filename=$(basename "${ncu_file}")
    echo "Processing ${filename}..."
    
    # 从文件名提取batch_size和seq_len
    if [[ ${filename} =~ bs([0-9]+)_seq([0-9]+) ]]; then
        batch_size=${BASH_REMATCH[1]}
        seq_len=${BASH_REMATCH[2]}
    else
        echo "Warning: Could not extract bs/seq from ${filename}" >&2
        continue
    fi
    
    # 1. 提取 act_and_mul_kernel
    echo "  Extracting act_and_mul_kernel..."
    extract_kernel_metrics "act_and_mul_kernel" "${ncu_file}"
    echo "${batch_size},${seq_len},act_and_mul_kernel,${EXTRACTED_COMPUTE_THROUGHPUT},${EXTRACTED_MEMORY_THROUGHPUT},${EXTRACTED_DURATION},${EXTRACTED_DEVICE_MEMORY_MB},${EXTRACTED_DEVICE_MEMORY_BANDWIDTH_GBS}" >> "${OUTPUT_CSV}"
    echo "    act_and_mul_kernel: compute=${EXTRACTED_COMPUTE_THROUGHPUT}, memory=${EXTRACTED_MEMORY_THROUGHPUT}, duration=${EXTRACTED_DURATION}, devmem=${EXTRACTED_DEVICE_MEMORY_MB}, bandwidth=${EXTRACTED_DEVICE_MEMORY_BANDWIDTH_GBS}"
    
    # 2. 查找并提取两个ampere或Kernel2开头的kernel
    echo "  Finding ampere or Kernel2 kernels..."
    temp_kernel_list=$(mktemp)
    ncu --import "${ncu_file}" 2>&1 > "${temp_kernel_list}"
    
    # 提取所有以 "ampere" 或 "Kernel2" 开头的kernel名称
    target_kernels=($(grep -oE '(ampere|Kernel)[a-zA-Z0-9_]+' "${temp_kernel_list}" | sort -u))
    rm -f "${temp_kernel_list}"

    echo "  Found ${#target_kernels[@]} ampere or Kernel2 kernels"
    echo "  ${target_kernels[@]}"
    
    # 只取前两个符合条件的kernel
    target_count=0
    for target_kernel in "${target_kernels[@]}"; do
        if [ ${target_count} -ge 2 ]; then
            break
        fi
        
        echo "  Extracting ${target_kernel}..."
        extract_kernel_metrics "${target_kernel}" "${ncu_file}"
        echo "${batch_size},${seq_len},${target_kernel},${EXTRACTED_COMPUTE_THROUGHPUT},${EXTRACTED_MEMORY_THROUGHPUT},${EXTRACTED_DURATION},${EXTRACTED_DEVICE_MEMORY_MB},${EXTRACTED_DEVICE_MEMORY_BANDWIDTH_GBS}" >> "${OUTPUT_CSV}"
        echo "    ${target_kernel}: compute=${EXTRACTED_COMPUTE_THROUGHPUT}, memory=${EXTRACTED_MEMORY_THROUGHPUT}, duration=${EXTRACTED_DURATION}, devmem=${EXTRACTED_DEVICE_MEMORY_MB}, bandwidth=${EXTRACTED_DEVICE_MEMORY_BANDWIDTH_GBS}"
        
        target_count=$((target_count + 1))
    done
    
    if [ ${target_count} -eq 0 ]; then
        echo "    Warning: No ampere or Kernel2 kernels found in ${filename}"
    elif [ ${target_count} -lt 2 ]; then
        echo "    Warning: Only ${target_count} ampere/Kernel2 kernel(s) found in ${filename}"
    fi
    
    echo ""
done

# 对CSV文件按batch_size和seq_len从小到大排序
echo "Sorting CSV file by batch_size and seq_len..."
temp_sorted=$(mktemp)

# 保留表头，对数据行进行排序
head -n 1 "${OUTPUT_CSV}" > "${temp_sorted}"
tail -n +2 "${OUTPUT_CSV}" | sort -t',' -k1,1n -k2,2n -k3,3 >> "${temp_sorted}"

# 替换原文件
mv "${temp_sorted}" "${OUTPUT_CSV}"

chmod 644 "${OUTPUT_CSV}"

echo "=========================================="
echo "All files processed successfully!"
echo "Results saved to ${OUTPUT_CSV}"
echo "Total records: $(($(wc -l < "${OUTPUT_CSV}") - 1))"
echo "=========================================="
