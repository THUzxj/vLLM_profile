#!/usr/bin/env python3
"""
递归搜索文件夹及其子文件夹中的所有JSON文件，汇总到CSV文件
用法: python json_to_csv.py <文件夹路径> [输出CSV路径]
"""

import json
import csv
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any


def extract_config_from_filename(filename: str) -> Dict[str, Any]:
    """从文件名中提取配置信息"""
    config = {}

    # 提取batch size (bs后面的数字)
    bs_match = re.search(r'bs(\d+)', filename)
    if bs_match:
        config['batch_size'] = int(bs_match.group(1))

    # 提取input tokens (in后面的数字)
    in_match = re.search(r'in(\d+)', filename)
    if in_match:
        config['input_tokens'] = int(in_match.group(1))

    # 提取output tokens (out后面的数字)
    out_match = re.search(r'out(\d+)', filename)
    if out_match:
        config['output_tokens'] = int(out_match.group(1))

    # 提取max batch tokens (mbt后面的数字)
    mbt_match = re.search(r'mbt(\d+)', filename)
    if mbt_match:
        config['max_batch_tokens'] = int(mbt_match.group(1))

    # 保存原始文件名
    config['filename'] = filename

    return config


def load_json_files(folder_path: str) -> List[Dict[str, Any]]:
    """递归加载文件夹及其子文件夹中的所有JSON文件"""
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")

    # 使用 rglob 递归搜索所有子文件夹中的 JSON 文件
    json_files = list(folder.rglob("*.json"))
    if not json_files:
        raise ValueError(f"文件夹及其子文件夹中没有找到JSON文件: {folder_path}")

    all_data = []

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 从文件名提取配置信息
            config = extract_config_from_filename(json_file.name)

            # 添加相对路径信息，以便区分来自不同文件夹的文件
            try:
                relative_path = json_file.relative_to(folder)
                config['relative_path'] = str(
                    relative_path.parent) if relative_path.parent != Path('.') else ''
            except ValueError:
                # 如果无法计算相对路径，使用绝对路径
                config['relative_path'] = str(json_file.parent)

            # 合并配置信息和JSON数据
            combined_data = {**config, **data}
            all_data.append(combined_data)

        except json.JSONDecodeError as e:
            print(f"警告: 无法解析 {json_file}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"警告: 读取 {json_file} 时出错: {e}", file=sys.stderr)

    return all_data


def write_to_csv(data: List[Dict[str, Any]], output_path: str):
    """将数据写入CSV文件"""
    if not data:
        raise ValueError("没有数据可写入")

    # 获取所有可能的字段名（合并所有记录的键）
    all_keys = set()
    for record in data:
        all_keys.update(record.keys())

    print("all_keys: ", all_keys)

    # 定义字段顺序：先配置文件相关字段，然后是JSON数据字段
    priority_fields = ['filename', 'relative_path', 'batch_size',
                       'input_tokens', 'output_tokens', 'max_batch_tokens']
    # other_fields = sorted([k for k in all_keys if k not in priority_fields])

    other_fields = ["mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "mean_tpot_ms",
                    "median_tpot_ms", "std_tpot_ms", "mean_e2el_ms", "median_e2el_ms", "std_e2el_ms"]
    fieldnames = [f for f in priority_fields if f in all_keys] + other_fields

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # filter fields that are not in fieldnames
        data = [{k: v for k, v in record.items() if k in fieldnames}
                for record in data]

        # sort data by batch_size and input_tokens
        data.sort(key=lambda x: (x['batch_size'], x['input_tokens']))
        writer.writerows(data)

    print(f"成功写入 {len(data)} 条记录到 {output_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python json_to_csv.py <文件夹路径> [输出CSV路径]")
        sys.exit(1)

    folder_path = sys.argv[1]

    # 如果没有指定输出路径，使用默认路径
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        folder_name = Path(folder_path).name
        output_path = os.path.join(folder_path, f"{folder_name}_summary.csv")

    try:
        # 递归加载所有JSON文件
        print(f"正在递归读取文件夹: {folder_path}")
        data = load_json_files(folder_path)
        print(f"成功加载 {len(data)} 个JSON文件（包括所有子文件夹）")

        # 写入CSV文件
        write_to_csv(data, output_path)

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
