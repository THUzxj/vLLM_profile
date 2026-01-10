import pandas as pd
import numpy as np
from pathlib import Path
import sys

def compare_model_layer_time_diff(layers_csv_path, components_csv_path, output_csv_path=None):
    """
    比较两个CSV文件中对应项的model_time与total_layer_time的差值
    
    Args:
        layers_csv_path: layers_in_model的CSV文件路径
        components_csv_path: components_in_model的CSV文件路径
        output_csv_path: 输出CSV文件路径（可选）
    """
    # 读取两个CSV文件
    print(f"读取文件: {layers_csv_path}")
    layers_df = pd.read_csv(layers_csv_path)
    
    print(f"读取文件: {components_csv_path}")
    components_df = pd.read_csv(components_csv_path)
    
    # 提取model_time和total_layer_time的数据
    def extract_time_data(df, time_type):
        """提取指定类型的时间数据"""
        time_df = df[df['Component'] == time_type].copy()
        return time_df
    
    layers_model_time = extract_time_data(layers_df, 'model_time')
    layers_total_time = extract_time_data(layers_df, 'total_layer_time')
    
    components_model_time = extract_time_data(components_df, 'model_time')
    components_total_time = extract_time_data(components_df, 'total_layer_time')
    
    # 合并model_time和total_layer_time
    def merge_time_data(model_df, total_df):
        """合并model_time和total_layer_time数据"""
        merged = []
        for _, model_row in model_df.iterrows():
            # 找到对应的total_layer_time行
            total_row = total_df[
                (total_df['input_len'] == model_row['input_len']) &
                (total_df['batch_size'] == model_row['batch_size'])
            ]
            
            if len(total_row) > 0:
                total_row = total_row.iloc[0]
                merged.append({
                    'input_len': model_row['input_len'],
                    'batch_size': model_row['batch_size'],
                    'model_time_mean': model_row['Mean'],
                    'total_layer_time_mean': total_row['Mean'],
                    'model_time_median': model_row['Median'],
                    'total_layer_time_median': total_row['Median'],
                })
        return pd.DataFrame(merged)
    
    layers_merged = merge_time_data(layers_model_time, layers_total_time)
    components_merged = merge_time_data(components_model_time, components_total_time)
    
    # 根据input_len和batch_size合并两个数据集
    comparison_df = pd.merge(
        layers_merged,
        components_merged,
        on=['input_len', 'batch_size'],
        suffixes=('_layers', '_components'),
        how='outer'
    )
    
    # 计算差值
    comparison_df['model_time_diff_mean'] = (
        comparison_df['model_time_mean_layers'] - 
        comparison_df['model_time_mean_components']
    )
    comparison_df['model_time_diff_median'] = (
        comparison_df['model_time_median_layers'] - 
        comparison_df['model_time_median_components']
    )
    
    comparison_df['total_layer_time_diff_mean'] = (
        comparison_df['total_layer_time_mean_layers'] - 
        comparison_df['total_layer_time_mean_components']
    )
    comparison_df['total_layer_time_diff_median'] = (
        comparison_df['total_layer_time_median_layers'] - 
        comparison_df['total_layer_time_median_components']
    )
    
    # 计算model_time - total_layer_time的差值（对于每个数据集）
    comparison_df['layers_model_minus_total_mean'] = (
        comparison_df['model_time_mean_layers'] - 
        comparison_df['total_layer_time_mean_layers']
    )
    comparison_df['layers_model_minus_total_median'] = (
        comparison_df['model_time_median_layers'] - 
        comparison_df['total_layer_time_median_layers']
    )
    
    comparison_df['components_model_minus_total_mean'] = (
        comparison_df['model_time_mean_components'] - 
        comparison_df['total_layer_time_mean_components']
    )
    comparison_df['components_model_minus_total_median'] = (
        comparison_df['model_time_median_components'] - 
        comparison_df['total_layer_time_median_components']
    )
    
    # 重新排列列的顺序，使其更易读
    column_order = [
        'input_len', 'batch_size',
        'model_time_mean_layers', 'model_time_mean_components', 'model_time_diff_mean',
        'model_time_median_layers', 'model_time_median_components', 'model_time_diff_median',
        'total_layer_time_mean_layers', 'total_layer_time_mean_components', 'total_layer_time_diff_mean',
        'total_layer_time_median_layers', 'total_layer_time_median_components', 'total_layer_time_diff_median',
        'layers_model_minus_total_mean', 'layers_model_minus_total_median',
        'components_model_minus_total_mean', 'components_model_minus_total_median',
    ]
    
    # 只保留存在的列
    available_columns = [col for col in column_order if col in comparison_df.columns]
    comparison_df = comparison_df[available_columns]
    
    # 按input_len和batch_size排序
    comparison_df = comparison_df.sort_values(['input_len', 'batch_size'], ascending=[True, True])
    
    # 打印统计信息
    print("\n" + "="*80)
    print("比较结果统计:")
    print("="*80)
    print(f"总共有 {len(comparison_df)} 个配置")
    print(f"\nmodel_time差值统计 (layers - components):")
    print(f"  均值差值 - Mean: {comparison_df['model_time_diff_mean'].mean():.6f}, "
          f"Median: {comparison_df['model_time_diff_median'].median():.6f}")
    print(f"  均值差值 - Min: {comparison_df['model_time_diff_mean'].min():.6f}, "
          f"Max: {comparison_df['model_time_diff_mean'].max():.6f}")
    print(f"\ntotal_layer_time差值统计 (layers - components):")
    print(f"  均值差值 - Mean: {comparison_df['total_layer_time_diff_mean'].mean():.6f}, "
          f"Median: {comparison_df['total_layer_time_diff_median'].median():.6f}")
    print(f"  均值差值 - Min: {comparison_df['total_layer_time_diff_mean'].min():.6f}, "
          f"Max: {comparison_df['total_layer_time_diff_mean'].max():.6f}")
    print(f"\nlayers数据集: model_time - total_layer_time")
    print(f"  均值差值 - Mean: {comparison_df['layers_model_minus_total_mean'].mean():.6f}, "
          f"Median: {comparison_df['layers_model_minus_total_median'].median():.6f}")
    print(f"\ncomponents数据集: model_time - total_layer_time")
    print(f"  均值差值 - Mean: {comparison_df['components_model_minus_total_mean'].mean():.6f}, "
          f"Median: {comparison_df['components_model_minus_total_median'].median():.6f}")
    
    # 显示前几行数据
    print("\n" + "="*80)
    print("前10行数据预览:")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(comparison_df.head(10).to_string(index=False))
    
    # 保存到CSV文件
    if output_csv_path:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ 比较结果已保存到: {output_path}")
    else:
        # 默认保存到layers_in_model/analysis目录
        output_dir = Path(__file__).parent
        output_csv_path = output_dir / 'model_layer_time_comparison.csv'
        comparison_df.to_csv(output_csv_path, index=False)
        print(f"\n✓ 比较结果已保存到: {output_csv_path}")
    
    return comparison_df


if __name__ == '__main__':
    # 默认路径
    default_layers_csv = Path(__file__).parent.parent / 'layers_results_5' / 'Qwen3-4B' / 'all_layer_times_statistics.csv'
    default_components_csv = Path(__file__).parent.parent.parent / 'components_in_model' / 'components_results_5' / 'Qwen3-4B' / 'all_layer_times_statistics.csv'
    
    # 从命令行参数获取路径，或使用默认路径
    if len(sys.argv) >= 3:
        layers_csv_path = Path(sys.argv[1])
        components_csv_path = Path(sys.argv[2])
        output_csv_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else None
    else:
        layers_csv_path = default_layers_csv
        components_csv_path = default_components_csv
        output_csv_path = None
    
    # 检查文件是否存在
    if not layers_csv_path.exists():
        print(f"错误: 文件不存在: {layers_csv_path}")
        sys.exit(1)
    
    if not components_csv_path.exists():
        print(f"错误: 文件不存在: {components_csv_path}")
        sys.exit(1)
    
    # 执行比较
    comparison_df = compare_model_layer_time_diff(
        layers_csv_path,
        components_csv_path,
        output_csv_path
    )

