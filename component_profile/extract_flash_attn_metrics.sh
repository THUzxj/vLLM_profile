#!/bin/bash

# 从ncu-rep文件中提取flash_fwd_splitkv_kernel的compute throughput和memory throughput
# 并保存到CSV文件中

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCU_REPORTS_DIR="${SCRIPT_DIR}/ncu_profile_result"
OUTPUT_CSV="${NCU_REPORTS_DIR}/flash_attn_kernel_metrics.csv"
KERNEL_NAME="flash_fwd_splitkv_kernel"

# 检查ncu命令是否存在
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu command not found. Please install Nsight Compute." >&2
    exit 1
fi

# 创建CSV文件并写入表头
echo "batch_size,kv_len,kernel_name,compute_throughput,memory_throughput,duration" > "${OUTPUT_CSV}"

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
    
    # 方法1: 使用--print-summary（最可靠的方法）
    temp_output=$(mktemp)
    
    ncu --import "${ncu_file}" \
        --print-summary none \
        --kernel-name "${KERNEL_NAME}" \
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
            if [[ ${line_lower} == *"duration"* ]] || [[ ${line_lower} == *"elapsed"* ]] || [[ ${line_lower} == *"time"* ]]; then
                # 提取数值（支持科学计数法，可能包含单位如us, ms, s）
                value=$(echo "${line}" | grep -oE '[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?' | head -1)
                if [ -n "${value}" ] && [ "${duration}" == "N/A" ]; then
                    # 检查单位并转换为微秒（如果可能）
                    if [[ ${line_lower} == *"ms"* ]] || [[ ${line_lower} == *"millisecond"* ]]; then
                        # 如果是毫秒，转换为微秒（乘以1000）
                        duration=$(awk "BEGIN {printf \"%.6f\", ${value} * 1000}" 2>/dev/null || echo "${value}")
                    elif ([[ ${line_lower} == *"second"* ]] && [[ ${line_lower} != *"millisecond"* ]] && [[ ${line_lower} != *"microsecond"* ]] && [[ ${line_lower} != *"nanosecond"* ]]) || ([[ ${line_lower} == *"s"* ]] && [[ ${line_lower} != *"ms"* ]] && [[ ${line_lower} != *"us"* ]] && [[ ${line_lower} != *"ns"* ]] && [[ ${line_lower} != *"second"* ]]); then
                        # 如果是秒，转换为微秒（乘以1000000）
                        duration=$(awk "BEGIN {printf \"%.6f\", ${value} * 1000000}" 2>/dev/null || echo "${value}")
                    elif [[ ${line_lower} == *"ns"* ]] || [[ ${line_lower} == *"nanosecond"* ]]; then
                        # 如果是纳秒，转换为微秒（除以1000）
                        duration=$(awk "BEGIN {printf \"%.6f\", ${value} / 1000}" 2>/dev/null || echo "${value}")
                    else
                        # 默认假设是微秒或已经是数值
                        duration="${value}"
                    fi
                fi
            fi
        done < "${temp_output}"
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
    echo "${batch_size},${kv_len},${KERNEL_NAME},${compute_throughput},${memory_throughput},${duration}" >> "${OUTPUT_CSV}"
    
    echo "  Extracted: compute_throughput=${compute_throughput}, memory_throughput=${memory_throughput}, duration=${duration}"
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

