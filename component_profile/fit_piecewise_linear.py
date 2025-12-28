#!/usr/bin/env python3
"""
分段线性函数拟合脚本
使用两个分段一次函数拟合 mean_time_ms 与 batch_size 的关系
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os

def piecewise_linear(x, breakpoint, a1, b1, a2, b2):
    """
    分段线性函数
    第一段: y = a1 * x + b1 (x <= breakpoint)
    第二段: y = a2 * x + b2 (x > breakpoint)
    在 breakpoint 处需要连续
    """
    y = np.zeros_like(x)
    mask1 = x <= breakpoint
    mask2 = x > breakpoint
    
    y[mask1] = a1 * x[mask1] + b1
    # 确保在 breakpoint 处连续
    y_value_at_breakpoint = a1 * breakpoint + b1
    y[mask2] = a2 * (x[mask2] - breakpoint) + y_value_at_breakpoint
    
    return y

def objective(params, x, y):
    """优化目标函数：最小化残差平方和"""
    breakpoint, a1, b1, a2 = params
    # b2 由连续性条件确定
    b2 = (a1 - a2) * breakpoint + b1
    
    y_pred = piecewise_linear(x, breakpoint, a1, b1, a2, b2)
    return np.sum((y - y_pred) ** 2)

def fit_piecewise_linear(x, y, initial_breakpoint=None):
    """
    拟合分段线性函数
    返回: (breakpoint, a1, b1, a2, b2, r2, mse)
    """
    # 如果没有提供初始分段点，使用中位数
    if initial_breakpoint is None:
        initial_breakpoint = np.median(x)
    
    # 初始参数估计
    # 对前半部分和后半部分分别进行线性拟合
    mid_idx = len(x) // 2
    x1 = x[:mid_idx]
    y1 = y[:mid_idx]
    x2 = x[mid_idx:]
    y2 = y[mid_idx:]
    
    # 线性拟合第一段
    coeffs1 = np.polyfit(x1, y1, 1)
    a1_init, b1_init = coeffs1[0], coeffs1[1]
    
    # 线性拟合第二段
    coeffs2 = np.polyfit(x2, y2, 1)
    a2_init = coeffs2[0]
    
    # 初始参数
    initial_params = [initial_breakpoint, a1_init, b1_init, a2_init]
    
    # 约束条件：分段点必须在数据范围内
    bounds = [
        (x.min(), x.max()),  # breakpoint
        (None, None),  # a1
        (None, None),  # b1
        (None, None),  # a2
    ]
    
    # 优化
    result = minimize(
        objective,
        initial_params,
        args=(x, y),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    breakpoint, a1, b1, a2 = result.x
    b2 = (a1 - a2) * breakpoint + b1
    
    # 计算拟合质量
    y_pred = piecewise_linear(x, breakpoint, a1, b1, a2, b2)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return breakpoint, a1, b1, a2, b2, r2, mse

def main():
    # 读取 CSV 文件
    csv_path = 'sweep_profile_result/benchmark_qwen3_mlp_results.csv'
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # 提取数据
    batch_size = df['batch_size'].values
    mean_time_ms = df['mean_time_ms'].values
    
    # 过滤掉空值
    mask = ~(np.isnan(batch_size) | np.isnan(mean_time_ms))
    batch_size = batch_size[mask]
    mean_time_ms = mean_time_ms[mask]
    
    print(f"数据点数量: {len(batch_size)}")
    print(f"Batch size 范围: [{batch_size.min()}, {batch_size.max()}]")
    print(f"Mean time 范围: [{mean_time_ms.min():.4f}, {mean_time_ms.max():.4f}] ms")
    
    # 拟合分段线性函数
    print("\n开始拟合分段线性函数...")
    breakpoint, a1, b1, a2, b2, r2, mse = fit_piecewise_linear(batch_size, mean_time_ms)
    
    print(f"\n拟合结果:")
    print(f"分段点 (breakpoint): {breakpoint:.2f}")
    print(f"第一段: y = {a1:.6f} * x + {b1:.6f}  (x <= {breakpoint:.2f})")
    print(f"第二段: y = {a2:.6f} * x + {b2:.6f}  (x > {breakpoint:.2f})")
    print(f"R² 分数: {r2:.6f}")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {np.sqrt(mse):.6f} ms")
    
    # 生成拟合曲线
    x_fit = np.linspace(batch_size.min(), batch_size.max(), 1000)
    y_fit = piecewise_linear(x_fit, breakpoint, a1, b1, a2, b2)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    # 原始数据点
    plt.scatter(batch_size, mean_time_ms, color='blue', s=100, alpha=0.7, label='原始数据', zorder=3)
    
    # 拟合曲线
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='分段线性拟合', zorder=2)
    
    # 标记分段点
    y_breakpoint = piecewise_linear(np.array([breakpoint]), breakpoint, a1, b1, a2, b2)[0]
    plt.axvline(x=breakpoint, color='green', linestyle='--', linewidth=1.5, 
                label=f'分段点 (x={breakpoint:.2f})', zorder=1)
    plt.plot(breakpoint, y_breakpoint, 'go', markersize=10, zorder=4)
    
    # 分别绘制两段拟合线（用于可视化）
    x1_fit = x_fit[x_fit <= breakpoint]
    y1_fit = a1 * x1_fit + b1
    x2_fit = x_fit[x_fit > breakpoint]
    y2_fit = a2 * (x2_fit - breakpoint) + (a1 * breakpoint + b1)
    
    plt.plot(x1_fit, y1_fit, 'orange', linestyle=':', linewidth=1.5, alpha=0.7, label='第一段', zorder=1)
    plt.plot(x2_fit, y2_fit, 'purple', linestyle=':', linewidth=1.5, alpha=0.7, label='第二段', zorder=1)
    
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Mean Time (ms)', fontsize=12)
    plt.title('分段线性函数拟合: Mean Time vs Batch Size\n' + 
              f'R² = {r2:.4f}, RMSE = {np.sqrt(mse):.4f} ms', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    output_path = 'sweep_profile_result/benchmark_qwen3_mlp_piecewise_linear_fit.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 保存拟合参数到文件
    params_path = 'sweep_profile_result/benchmark_qwen3_mlp_piecewise_linear_params.txt'
    with open(params_path, 'w') as f:
        f.write("分段线性函数拟合参数\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"分段点 (breakpoint): {breakpoint:.6f}\n\n")
        f.write(f"第一段 (x <= {breakpoint:.6f}):\n")
        f.write(f"  y = {a1:.10f} * x + {b1:.10f}\n")
        f.write(f"  斜率: {a1:.10f}\n")
        f.write(f"  截距: {b1:.10f}\n\n")
        f.write(f"第二段 (x > {breakpoint:.6f}):\n")
        f.write(f"  y = {a2:.10f} * x + {b2:.10f}\n")
        f.write(f"  斜率: {a2:.10f}\n")
        f.write(f"  截距: {b2:.10f}\n\n")
        f.write(f"拟合质量:\n")
        f.write(f"  R² 分数: {r2:.10f}\n")
        f.write(f"  均方误差 (MSE): {mse:.10f}\n")
        f.write(f"  均方根误差 (RMSE): {np.sqrt(mse):.10f} ms\n")
    
    print(f"拟合参数已保存到: {params_path}")
    
    plt.show()

if __name__ == '__main__':
    main()

