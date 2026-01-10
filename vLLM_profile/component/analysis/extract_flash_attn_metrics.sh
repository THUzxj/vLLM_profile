#!/bin/bash

# 从ncu-rep文件中提取flash_fwd_splitkv_kernel的compute throughput和memory throughput
# 并保存到CSV文件中

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCU_REPORTS_DIR=$1

if [ -z "${NCU_REPORTS_DIR}" ]; then
    echo "Usage: $0 <ncu_reports_directory>" >&2
    exit 1
fi

OUTPUT_CSV="${NCU_REPORTS_DIR}/flash_attn_kernel_metrics.csv"
KERNEL_NAME="flash_fwd_splitkv_kernel"

# 检查ncu命令是否存在
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu command not found. Please install Nsight Compute." >&2
    exit 1
fi

# 创建CSV文件并写入表头
echo "batch_size,kv_len,kernel_name,compute_throughput,memory_throughput,duration,device_memory_mb,device_memory_bandwidth_gbs" > "${OUTPUT_CSV}"

# 查找所有ncu-rep文件
ncu_files=($(find "${NCU_REPORTS_DIR}" -name "ncu_report_flash_attn_*.ncu-rep" | sort))

if [ ${#ncu_files[@]} -eq 0 ]; then
    echo "Error: No ncu-rep files found in ${NCU_REPORTS_DIR}" >&2
    exit 1
fi

echo "Found ${#ncu_files[@]} ncu-rep files"

# 处理每个文件
for ncu_file in "${ncu_files[@]}"; do
    filename=$(basename "${ncu_file}")
    echo "Processing ${filename}..."
    
    # 从文件名提取batch_size和kv_len
    if [[ ${filename} =~ bs([0-9]+)_kv([0-9]+) ]]; then
        batch_size=${BASH_REMATCH[1]}
        kv_len=${BASH_REMATCH[2]}
    else
        echo "Warning: Could not extract bs/kv from ${filename}" >&2
        continue
    fi
    
    # 尝试提取compute throughput和memory throughput
    compute_throughput="N/A"
    memory_throughput="N/A"
    duration="N/A"
    device_memory_mb="N/A"
    device_memory_bandwidth_gbs="N/A"
    
    # 方法1: 使用--print-summary（最可靠的方法）
    temp_output=$(mktemp)
    
    echo "execute: ncu --import ${ncu_file} --print-summary none --kernel-name ${KERNEL_NAME} > ${temp_output}"
    ncu --import "${ncu_file}" \
        --print-summary none \
        --kernel-name "${KERNEL_NAME}" \
        > "${temp_output}" 2>&1

    # cat "${temp_output}"

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

                # echo "value: ${value}"
                # echo "line_lower: ${line_lower}"
                # echo "line: ${line}"
                # exit 1
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
    
    # 提取device memory读写量: dram__bytes_read.sum + dram__bytes_write.sum
    if [ "${device_memory_mb}" == "N/A" ]; then
        temp_details=$(mktemp)
        echo "execute: ncu --import ${ncu_file} --print-details all --kernel-name ${KERNEL_NAME} --section MemoryWorkloadAnalysis_Tables --print-units base > ${temp_details}"
        ncu --import "${ncu_file}" \
            --print-details all \
            --kernel-name "${KERNEL_NAME}" \
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

            # echo "bytes_read: ${bytes_read}"
            # echo "bytes_write: ${bytes_write}"
            # echo "bandwidth_read: ${bandwidth_read}"
            # echo "bandwidth_write: ${bandwidth_write}"
            # exit 1
            
            # 如果找到了两个值，计算总和并转换为MB
            if [ -n "${bytes_read}" ] && [ -n "${bytes_write}" ]; then
                total_bytes=$(awk "BEGIN {printf \"%.6f\", ${bytes_read} + ${bytes_write}}" 2>/dev/null)
                if [ -n "${total_bytes}" ]; then
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
    
    # 如果方法1失败，尝试方法2: 使用--csv导出所有metrics
    # if [ "${compute_throughput}" == "N/A" ] || [ "${memory_throughput}" == "N/A" ]; then
    #     ncu --import "${ncu_file}" \
    #         --csv \
    #         --page raw \
    #         --kernel-name "${KERNEL_NAME}" \
    #         > "${temp_output}" 2>&1
        
    #     if [ $? -eq 0 ]; then
    #         # 查找throughput相关的metrics
    #         while IFS= read -r line; do
    #             line_lower=$(echo "${line}" | tr '[:upper:]' '[:lower:]')
    #             if [[ ${line_lower} == *"throughput"* ]]; then
    #                 value=$(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?' | head -1)
    #                 if [ -n "${value}" ]; then
    #                     if [[ ${line_lower} == *"sm__throughput"* ]] || ([[ ${line_lower} == *"compute"* ]] && [[ ${line_lower} == *"throughput"* ]]); then
    #                         if [ "${compute_throughput}" == "N/A" ]; then
    #                             compute_throughput="${value}"
    #                         fi
    #                     elif [[ ${line_lower} == *"dram__throughput"* ]] || ([[ ${line_lower} == *"memory"* ]] && [[ ${line_lower} == *"throughput"* ]]); then
    #                         if [ "${memory_throughput}" == "N/A" ]; then
    #                             memory_throughput="${value}"
    #                         fi
    #                     fi
    #                 fi
    #             fi
    #         done < "${temp_output}"
    #     fi
    # fi
    
    # # 如果还是失败，尝试方法3: 使用--query查询特定metrics
    # if [ "${compute_throughput}" == "N/A" ] || [ "${memory_throughput}" == "N/A" ]; then
    #     ncu --import "${ncu_file}" \
    #         --query "sm__throughput.avg.pct_of_peak_sustained_elapsed_time,dram__throughput.avg.pct_of_peak_sustained_elapsed_time" \
    #         --kernel-name "${KERNEL_NAME}" \
    #         > "${temp_output}" 2>&1
        
    #     if [ $? -eq 0 ]; then
    #         # 提取数值
    #         values=($(grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?' "${temp_output}"))
    #         if [ ${#values[@]} -ge 2 ]; then
    #             if [ "${compute_throughput}" == "N/A" ]; then
    #                 compute_throughput="${values[0]}"
    #             fi
    #             if [ "${memory_throughput}" == "N/A" ]; then
    #                 memory_throughput="${values[1]}"
    #             fi
    #         elif [ ${#values[@]} -eq 1 ]; then
    #             if [ "${compute_throughput}" == "N/A" ]; then
    #                 compute_throughput="${values[0]}"
    #             fi
    #         fi
    #     fi
    # fi
    
    # 清理临时文件
    rm -f "${temp_output}"
    
    # 写入CSV
    echo "${batch_size},${kv_len},${KERNEL_NAME},${compute_throughput},${memory_throughput},${duration},${device_memory_mb},${device_memory_bandwidth_gbs}" >> "${OUTPUT_CSV}"
    
    echo "  Extracted: compute_throughput=${compute_throughput}, memory_throughput=${memory_throughput}, duration=${duration}, device_memory_mb=${device_memory_mb}, device_memory_bandwidth_gbs=${device_memory_bandwidth_gbs}"
done

echo ""
echo "Results saved to ${OUTPUT_CSV}"
echo "Total records: $(($(wc -l < "${OUTPUT_CSV}") - 1))"

# 对CSV文件按batch_size和kv_len从小到大排序
echo "Sorting CSV file by batch_size and kv_len..."
temp_sorted=$(mktemp)

# 保留表头，对数据行进行排序
head -n 1 "${OUTPUT_CSV}" > "${temp_sorted}"
tail -n +2 "${OUTPUT_CSV}" | sort -t',' -k1,1n -k2,2n >> "${temp_sorted}"

# 替换原文件
mv "${temp_sorted}" "${OUTPUT_CSV}"

chmod 644 "${OUTPUT_CSV}"

echo "CSV file sorted successfully."

