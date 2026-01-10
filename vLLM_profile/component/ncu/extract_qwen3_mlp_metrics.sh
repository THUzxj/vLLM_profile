#!/bin/bash

# 从ncu-rep文件中提取qwen3_mlp kernel的compute throughput和memory throughput
# 并保存到CSV文件中
# 支持多个kernel名称，每个kernel保存到独立的CSV文件

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

# 扫描所有文件，找出所有以 "ampere_bf16_" 开头的 kernel
echo "Scanning all ncu-rep files for kernels starting with 'ampere_bf16_'..."
declare -A kernel_set  # 使用关联数组去重

for ncu_file in "${ncu_files[@]}"; do
    echo "  Scanning $(basename "${ncu_file}")..."
    temp_output=$(mktemp)
    ncu --import "${ncu_file}" 2>&1 > "${temp_output}"
    
    # 从 ncu 输出中提取所有以 "ampere_bf16_" 开头的 kernel 名称
    # ncu 输出中 kernel 名称可能出现在多种格式中
    # 使用 grep 直接提取所有匹配的 kernel 名称（更高效）
    while IFS= read -r kernel_name; do
        if [ -n "${kernel_name}" ]; then
            kernel_set["${kernel_name}"]=1
        fi
    done < <(grep -oE 'ampere_bf16_[a-zA-Z0-9_]+' "${temp_output}" | sort -u)
    
    rm -f "${temp_output}"
done

# 将关联数组的键转换为数组
KERNEL_NAMES=("act_and_mul_kernel")

for kernel_name in "${!kernel_set[@]}"; do
    KERNEL_NAMES+=("${kernel_name}")
done

# 对 kernel 名称排序
IFS=$'\n' KERNEL_NAMES=($(sort <<<"${KERNEL_NAMES[*]}"))
unset IFS

if [ ${#KERNEL_NAMES[@]} -eq 0 ]; then
    echo "Error: No kernels starting with 'ampere_bf16_' found in any ncu-rep files." >&2
    exit 1
fi

echo "Found ${#KERNEL_NAMES[@]} unique kernel(s) starting with 'ampere_bf16_':"
for kernel_name in "${KERNEL_NAMES[@]}"; do
    echo "  - ${kernel_name}"
done
echo ""

# 为每个kernel处理文件
for KERNEL_NAME in "${KERNEL_NAMES[@]}"; do
    # 为每个kernel创建独立的CSV文件
    # 将kernel名称中的特殊字符替换为下划线，用于文件名
    safe_kernel_name=$(echo "${KERNEL_NAME}" | sed 's/[^a-zA-Z0-9_]/_/g')
    OUTPUT_CSV="${NCU_REPORTS_DIR}/qwen3_mlp_kernel_metrics_${safe_kernel_name}.csv"
    
    echo "=========================================="
    echo "Processing kernel: ${KERNEL_NAME}"
    echo "Output CSV: ${OUTPUT_CSV}"
    echo "=========================================="
    
    # 创建CSV文件并写入表头
    echo "batch_size,seq_len,kernel_name,compute_throughput,memory_throughput,duration,device_memory_mb,device_memory_bandwidth_gbs" > "${OUTPUT_CSV}"
    
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
        
        # 尝试提取compute throughput和memory throughput
        compute_throughput="N/A"
        memory_throughput="N/A"
        duration="N/A"
        device_memory_mb="N/A"
        device_memory_bandwidth_gbs="N/A"
        
        # 方法1: 使用--print-summary（最可靠的方法）
        temp_output=$(mktemp)
        
        echo "  execute: ncu --import ${ncu_file} --print-summary none --kernel-name ${KERNEL_NAME} > ${temp_output}"
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
        
        # 清理临时文件
        rm -f "${temp_output}"
       
        # 提取device memory读写量: dram__bytes_read.sum + dram__bytes_write.sum
        if [ "${device_memory_mb}" == "N/A" ]; then
            temp_details=$(mktemp)
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
       
        # 写入CSV
        echo "${batch_size},${seq_len},${KERNEL_NAME},${compute_throughput},${memory_throughput},${duration},${device_memory_mb},${device_memory_bandwidth_gbs}" >> "${OUTPUT_CSV}"

        echo "  Extracted: compute_throughput=${compute_throughput}, memory_throughput=${memory_throughput}, duration=${duration}, device_memory_mb=${device_memory_mb}, device_memory_bandwidth_gbs=${device_memory_bandwidth_gbs}"
    done

    echo ""
    echo "Results saved to ${OUTPUT_CSV}"
    echo "Total records: $(($(wc -l < "${OUTPUT_CSV}") - 1))"

    # 对CSV文件按batch_size和seq_len从小到大排序
    echo "Sorting CSV file by batch_size and seq_len..."
    temp_sorted=$(mktemp)

    # 保留表头，对数据行进行排序
    head -n 1 "${OUTPUT_CSV}" > "${temp_sorted}"
    tail -n +2 "${OUTPUT_CSV}" | sort -t',' -k1,1n -k2,2n >> "${temp_sorted}"

    # 替换原文件
    mv "${temp_sorted}" "${OUTPUT_CSV}"

    chmod 644 "${OUTPUT_CSV}"

    echo "CSV file sorted successfully."
    echo ""
done

echo "=========================================="
echo "All kernels processed successfully!"
echo "=========================================="

# ---------------------------------------------------------------------------
# 将各个 kernel 的 CSV 进一步组合成一个汇总 CSV
# 对于每个 (batch_size, seq_len)，会有：
#   - act_and_mul_kernel 的 duration
#   - 最多两个 ampere_bf16_* kernel 的 duration
# 最终写入列：
#   batch_size,seq_len,act_kernel_name,act_kernel_value,
#   kernel1_name,kernel1_value,kernel2_name,kernel2_value
# 这里的 *_value 均为 duration（单位：毫秒）
# ---------------------------------------------------------------------------

COMBINED_CSV="${NCU_REPORTS_DIR}/qwen3_mlp_kernel_metrics_combined.csv"

echo "Combining per-kernel CSV files into: ${COMBINED_CSV}"

# 1) 读取 act_and_mul_kernel 的 duration
ACT_CSV="${NCU_REPORTS_DIR}/qwen3_mlp_kernel_metrics_act_and_mul_kernel.csv"

if [ ! -f "${ACT_CSV}" ]; then
    echo "Warning: ${ACT_CSV} not found, skip combining." >&2
    exit 0
fi

declare -A act_name_map
declare -A act_val_map

# 注意：不要用管道形式的 while，以免在子 shell 中修改不到关联数组
while IFS=',' read -r bs seq kname cthrough mthrough dur devmem bandwidth; do
    # 跳过空行
    [ -z "${bs}" ] && continue
    key="${bs},${seq}"
    act_name_map["${key}"]="${kname}"
    act_val_map["${key}"]="${dur}"
done < <(tail -n +2 "${ACT_CSV}")

# 2) 读取所有 ampere_bf16_* kernel，对每个 (bs, seq) 最多记录两个
declare -A ampere_name1_map
declare -A ampere_val1_map
declare -A ampere_name2_map
declare -A ampere_val2_map

for amp_csv in "${NCU_REPORTS_DIR}"/qwen3_mlp_kernel_metrics_ampere_bf16_*.csv; do
    [ -f "${amp_csv}" ] || continue

    # 同样使用重定向而不是管道，避免子 shell 问题
    while IFS=',' read -r bs seq kname cthrough mthrough dur devmem bandwidth; do
        [ -z "${bs}" ] && continue
        # 只保留有有效 duration 的行
        if [ "${dur}" = "N/A" ] || [ -z "${dur}" ]; then
            continue
        fi
        key="${bs},${seq}"
        if [ -z "${ampere_name1_map[${key}]+x}" ]; then
            ampere_name1_map["${key}"]="${kname}"
            ampere_val1_map["${key}"]="${dur}"
        elif [ -z "${ampere_name2_map[${key}]+x}" ]; then
            ampere_name2_map["${key}"]="${kname}"
            ampere_val2_map["${key}"]="${dur}"
        else
            # 已经有两个 ampere kernel，忽略更多的
            :
        fi
    done < <(tail -n +2 "${amp_csv}")
done

# 3) 写汇总 CSV
echo "batch_size,seq_len,act_kernel_name,act_kernel_value,kernel1_name,kernel1_value,kernel2_name,kernel2_value" > "${COMBINED_CSV}"

for key in "${!act_name_map[@]}"; do
    bs="${key%%,*}"
    seq="${key##*,}"
    act_name="${act_name_map[${key}]}"
    act_val="${act_val_map[${key}]}"

    k1_name="${ampere_name1_map[${key}]}"
    k1_val="${ampere_val1_map[${key}]}"
    k2_name="${ampere_name2_map[${key}]}"
    k2_val="${ampere_val2_map[${key}]}"

    echo "${bs},${seq},${act_name},${act_val},${k1_name},${k1_val},${k2_name},${k2_val}" >> "${COMBINED_CSV}"
done

chmod 644 "${COMBINED_CSV}"

# 按 batch_size、seq_len 排序汇总 CSV
tmp_combined_sorted=$(mktemp)
head -n 1 "${COMBINED_CSV}" > "${tmp_combined_sorted}"
tail -n +2 "${COMBINED_CSV}" | sort -t',' -k1,1n -k2,2n >> "${tmp_combined_sorted}"
mv "${tmp_combined_sorted}" "${COMBINED_CSV}"

echo "Combined CSV generated and sorted by batch_size,seq_len: ${COMBINED_CSV}"

