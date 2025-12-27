import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_layer_times(json_file):
    """
    分析JSON文件中所有layer的时间组件统计信息
    """
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 提取所有layer的时间数据
    layer_times = data['layer_times']
    
    # 初始化数据集合
    total_layer_times = []
    self_attention_times = []
    mlp_times = []
    qkv_projection_times = []
    attention_forward_times = []
    output_projection_times = []
    
    # 遍历每个layer，提取时间数据
    for layer in layer_times:
        total_layer_times.append(layer['total_layer_time'])
        
        layer_details = layer['layer_details']
        self_attention_times.append(layer_details['self_attention'])
        mlp_times.append(layer_details['mlp'])
        
        attention_details = layer_details['attention_details']
        qkv_projection_times.append(attention_details['qkv_projection_and_rope'])
        attention_forward_times.append(attention_details['attention_forward'])
        output_projection_times.append(attention_details['output_projection'])
    
    # 计算统计信息
    def calculate_stats(data, name):
        return {
            'Component': name,
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Std Dev': np.std(data),
            'Min': np.min(data),
            'Max': np.max(data),
            'Count': len(data)
        }
    
    stats_list = [
        calculate_stats(total_layer_times, 'total_layer_time'),
        calculate_stats(self_attention_times, 'self_attention'),
        calculate_stats(mlp_times, 'mlp'),
        calculate_stats(qkv_projection_times, 'qkv_projection_and_rope'),
        calculate_stats(attention_forward_times, 'attention_forward'),
        calculate_stats(output_projection_times, 'output_projection'),
    ]
    
    # 如果存在model_time，则添加到统计列表
    if 'model_time' in data:
        model_time = data['model_time']
        stats_list.insert(0, calculate_stats([model_time], 'model_time'))
    
    # 创建DataFrame
    df = pd.DataFrame(stats_list)
    
    return df

def get_max_count_file(folder_path):
    """
    获取文件夹中count_后面数字最大的文件
    """
    json_files = list(folder_path.glob('count_*.json'))
    if not json_files:
        return None
    
    # 提取count数字并排序
    def extract_count(filename):
        try:
            # 从 "count_49_..." 中提取数字
            count_str = filename.name.split('_')[1]
            return int(count_str)
        except (IndexError, ValueError):
            return -1
    
    max_file = max(json_files, key=extract_count)
    return max_file

import sys
# 获取component_times文件夹
component_times_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./component_times/')

# 用于存储所有数据的列表
all_dfs = []

# 遍历每个子文件夹
for subfolder in sorted(component_times_folder.iterdir()):
    if not subfolder.is_dir():
        continue
    
    # 获取count数最大的JSON文件
    json_file = get_max_count_file(subfolder)
    
    if json_file is None:
        print(f"⚠️  {subfolder.name}: 未找到JSON文件")
        continue
    
    try:
        # 分析数据
        df = analyze_layer_times(json_file)
        
        # 添加文件夹名称列
        df.insert(0, 'Folder', subfolder.name)
        
        # 按照"子文件夹名称.csv"保存到当前目录
        output_csv = component_times_folder / f"{subfolder.name}.csv"
        df.to_csv(output_csv, index=False)

        print(f"✓ {subfolder.name} / {json_file.name}: 已保存到 {output_csv.name}")
        
        # 收集数据以便后续合并
        all_dfs.append(df)
    except Exception as e:
        print(f"✗ {subfolder.name}: 处理失败 - {str(e)}")

def parse_folder_name(folder_name):
    """
    从folder name解析出model_name, input_len, batch_size
    例如: qwen3-4b_in512_bs64 -> qwen3-4b, 512, 64
    或: 4b_512_20 -> 4b, 512, 20
    """
    model_name = None
    input_len = None
    batch_size = None
    
    # 尝试匹配格式 qwen3-4b_in512_bs64
    if 'in' in folder_name and 'bs' in folder_name:
        parts = folder_name.split('_')
        # 查找model_name (in前的部分)
        in_idx = -1
        for i, part in enumerate(parts):
            if part.startswith('in'):
                in_idx = i
                model_name = '_'.join(parts[:i])
                break
        
        if in_idx != -1:
            # 提取input_len
            in_part = parts[in_idx]
            input_len = int(in_part[2:])  # 去掉'in'前缀
            
            # 提取batch_size
            for part in parts[in_idx+1:]:
                if part.startswith('bs'):
                    batch_size = int(part[2:])  # 去掉'bs'前缀
                    break
    
    # 尝试匹配格式 4b_512_20
    elif '_' in folder_name:
        parts = folder_name.split('_')
        if len(parts) >= 3:
            model_name = parts[0]
            try:
                input_len = int(parts[1])
                batch_size = int(parts[2])
            except ValueError:
                pass
    
    return model_name, input_len, batch_size

# 合并所有数据到一个总的CSV文件
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 从folder name解析出model_name, input_len, batch_size
    combined_df[['model_name', 'input_len', 'batch_size']] = combined_df['Folder'].apply(
        lambda x: pd.Series(parse_folder_name(x))
    )
    
    # 重新排列列的顺序
    cols = combined_df.columns.tolist()
    cols.remove('model_name')
    cols.remove('input_len')
    cols.remove('batch_size')
    combined_df = combined_df[['Folder', 'model_name', 'input_len', 'batch_size'] + cols]
    
    combined_csv = component_times_folder / 'all_layer_times_statistics.csv'
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n✓ 所有数据已合并到: {combined_csv}")
    print(f"  总行数: {len(combined_df)}")
    print(f"\n示例数据:")
    print(combined_df.head(10).to_string())
