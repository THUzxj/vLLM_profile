#!/usr/bin/env python3
"""
验证total_layer_time是否等于各组件时间之和，并计算各组件占比
生成包含input_len、batch_size和各组件百分比的新CSV
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def verify_and_analyze_layer_times(csv_path, output_csv_path=None):
    """
    验证时间关系并生成百分比分析CSV
    
    Args:
        csv_path: 输入的CSV文件路径
        output_csv_path: 输出的CSV文件路径，如果为None则自动生成
    """
    # 读取CSV
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # 获取所有唯一的input_len和batch_size组合
    unique_configs = df[['input_len', 'batch_size']].drop_duplicates()
    
    results = []
    verification_results = []
    
    # 定义需要验证的组件
    components = ['mlp', 'qkv_projection_and_rope', 'attention_forward', 'output_projection']
    
    for _, config in unique_configs.iterrows():
        input_len = config['input_len']
        batch_size = config['batch_size']
        
        # 筛选当前配置的数据
        config_df = df[(df['input_len'] == input_len) & (df['batch_size'] == batch_size)]
        
        # 提取各组件的时间（使用Mean值）
        component_times = {}
        total_layer_time = None
        
        for component in components + ['total_layer_time']:
            component_row = config_df[config_df['Component'] == component]
            if not component_row.empty:
                component_times[component] = component_row['Mean'].values[0]
                if component == 'total_layer_time':
                    total_layer_time = component_row['Mean'].values[0]
        
        if total_layer_time is None or total_layer_time == 0:
            print(f"Warning: No total_layer_time found for input_len={input_len}, batch_size={batch_size}")
            continue
        
        # 计算各组件时间之和
        sum_components = sum(component_times.get(comp, 0) for comp in components)
        
        # 验证时间关系
        diff = abs(total_layer_time - sum_components)
        relative_error = (diff / total_layer_time * 100) if total_layer_time > 0 else 0
        
        # 检查是否有self_attention数据，可能total_layer_time包含其他内容
        self_attention_time = None
        self_attention_row = config_df[config_df['Component'] == 'self_attention']
        if not self_attention_row.empty:
            self_attention_time = self_attention_row['Mean'].values[0]
        
        verification_results.append({
            'input_len': input_len,
            'batch_size': batch_size,
            'total_layer_time': total_layer_time,
            'sum_components': sum_components,
            'difference': diff,
            'relative_error_percent': relative_error,
            'is_valid': relative_error < 5.0,  # 允许5%的误差（因为可能有其他未统计的组件）
            'self_attention_time': self_attention_time if self_attention_time is not None else 0
        })
        
        # 计算各组件占total_layer_time的百分比
        result_row = {
            'input_len': input_len,
            'batch_size': batch_size,
        }
        
        for component in components:
            component_time = component_times.get(component, 0)
            percentage = (component_time / total_layer_time * 100) if total_layer_time > 0 else 0
            # 使用更简洁的列名
            component_name_short = component.replace('_', '_')  # 保持原名
            result_row[f'{component}_percentage'] = percentage
            result_row[f'{component}_time'] = component_time  # 保留时间值以便验证
        
        # 添加验证信息（可选，用于调试）
        result_row['total_layer_time'] = total_layer_time
        result_row['sum_components'] = sum_components
        result_row['difference'] = diff
        result_row['relative_error_percent'] = relative_error
        
        results.append(result_row)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    verification_df = pd.DataFrame(verification_results)
    
    # 按input_len和batch_size从小到大排序
    results_df = results_df.sort_values(by=['input_len', 'batch_size'], ascending=[True, True]).reset_index(drop=True)
    verification_df = verification_df.sort_values(by=['input_len', 'batch_size'], ascending=[True, True]).reset_index(drop=True)
    
    # 打印验证结果摘要
    print("=" * 80)
    print("时间关系验证结果摘要")
    print("=" * 80)
    print(f"总配置数: {len(verification_df)}")
    print(f"验证通过数: {verification_df['is_valid'].sum()}")
    print(f"验证失败数: {(~verification_df['is_valid']).sum()}")
    print(f"\n平均相对误差: {verification_df['relative_error_percent'].mean():.4f}%")
    print(f"最大相对误差: {verification_df['relative_error_percent'].max():.4f}%")
    
    if not verification_df['is_valid'].all():
        print("\n验证失败的配置:")
        failed = verification_df[~verification_df['is_valid']]
        print(failed[['input_len', 'batch_size', 'relative_error_percent']].to_string(index=False))
    
    print("\n" + "=" * 80)
    
    # 保存结果到输入CSV所在的文件夹
    if output_csv_path is None:
        csv_dir = csv_path.parent
        default_output = csv_dir / 'component_percentage_analysis.csv'
        # 如果无写权限则回落到当前analysis目录
        if os.access(csv_dir, os.W_OK):
            output_csv_path = default_output
        else:
            fallback_dir = Path(__file__).parent
            output_csv_path = fallback_dir / 'component_percentage_analysis.csv'
            print(f"目标目录不可写，已回落到: {output_csv_path}")
    
    # 保存完整结果（包含时间值和百分比以及验证信息）
    results_df.to_csv(output_csv_path, index=False)
    print(f"\n完整结果已保存到: {output_csv_path}")
    
    # 生成简化版本：包含input_len, batch_size、四个组件的百分比、total_layer_time和sum_components
    simplified_df = results_df[['input_len', 'batch_size'] + [f'{comp}_percentage' for comp in components] + ['total_layer_time', 'sum_components']].copy()
    simplified_output_path = Path(output_csv_path).parent / 'component_percentage_summary.csv'
    simplified_df.to_csv(simplified_output_path, index=False)
    print(f"简化结果（百分比+时间总和）已保存到: {simplified_output_path}")
    
    # 保存验证结果
    verification_csv_path = Path(output_csv_path).parent / 'time_verification_results.csv'
    verification_df.to_csv(verification_csv_path, index=False)
    print(f"验证结果已保存到: {verification_csv_path}")
    
    return results_df, verification_df


if __name__ == '__main__':
    # CSV文件路径（相对于脚本所在目录）
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / 'components_results_5' / 'Qwen3-4B' / 'all_layer_times_statistics.csv'
    
    # 如果文件不存在，尝试相对路径
    if not csv_path.exists():
        csv_path = Path('components_results_4/Qwen3-4B/all_layer_times_statistics.csv')
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到CSV文件，请检查路径: {csv_path}")

    print(f"读取CSV文件: {csv_path}")
    
    # 执行分析和验证
    results_df, verification_df = verify_and_analyze_layer_times(csv_path)
    
    # 显示前几行结果
    print("\n前10行结果预览:")
    print(results_df.head(10).to_string(index=False))
