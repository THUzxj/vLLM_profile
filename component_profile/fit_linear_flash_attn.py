#!/usr/bin/env python3
"""
线性函数拟合脚本
使用一次函数拟合 mean_time_ms 与 batch_size * kv_len 的关系
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import sys
import os
import argparse


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='线性函数拟合脚本')
    parser.add_argument('--result_dir', type=str, default='sweep_profile_result',
                        help='结果文件夹路径 (默认: sweep_profile_result)')
    args = parser.parse_args()

    result_dir = args.result_dir

    # 读取 CSV 文件
    csv_path = os.path.join(result_dir, 'benchmark_flash_attn_results.csv')
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # 提取数据
    batch_size = df['batch_size'].values
    kv_len = df['kv_len'].values
    mean_time_ms = df['mean_time_ms'].values

    # 过滤掉空值
    mask = ~(np.isnan(batch_size) | np.isnan(kv_len) | np.isnan(mean_time_ms))
    batch_size = batch_size[mask]
    kv_len = kv_len[mask]
    mean_time_ms = mean_time_ms[mask]

    # 计算 batch_size * kv_len
    batch_kv_product = batch_size * kv_len

    print(f"数据点数量: {len(batch_kv_product)}")
    print(
        f"Batch size * KV len 范围: [{batch_kv_product.min()}, {batch_kv_product.max()}]")
    print(
        f"Mean time 范围: [{mean_time_ms.min():.4f}, {mean_time_ms.max():.4f}] ms")

    # 准备数据用于拟合
    X = batch_kv_product.reshape(-1, 1)
    y = mean_time_ms

    # 线性拟合
    print("\n开始拟合线性函数...")
    regressor = LinearRegression()
    regressor.fit(X, y)

    # 获取拟合参数
    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    # 计算拟合质量
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n拟合结果:")
    print(f"线性函数: y = {slope:.10f} * x + {intercept:.10f}")
    print(f"其中 x = batch_size * kv_len")
    print(f"斜率: {slope:.10f}")
    print(f"截距: {intercept:.10f}")
    print(f"R² 分数: {r2:.6f}")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f} ms")

    # 计算相关系数
    correlation = np.corrcoef(batch_kv_product, mean_time_ms)[0, 1]
    print(f"相关系数: {correlation:.6f}")

    # 生成拟合曲线
    x_fit = np.linspace(batch_kv_product.min(), batch_kv_product.max(), 1000)
    y_fit = slope * x_fit + intercept

    # 绘制结果
    plt.figure(figsize=(12, 8))

    # 原始数据点
    plt.scatter(batch_kv_product, mean_time_ms, color='blue', s=100, alpha=0.7,
                label='原始数据', zorder=3)

    # 拟合曲线
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='线性拟合', zorder=2)

    plt.xlabel('Batch Size × KV Length', fontsize=12)
    plt.ylabel('Mean Time (ms)', fontsize=12)
    plt.title('Linear Fit: Mean Time vs Batch Size × KV Length\n' +
              f'y = {slope:.6f} × x + {intercept:.6f}, R² = {r2:.4f}, RMSE = {rmse:.4f} ms',
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(
        result_dir, 'benchmark_flash_attn_linear_fit.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")

    # 保存拟合参数到文件
    params_path = os.path.join(
        result_dir, 'benchmark_flash_attn_linear_params.txt')
    with open(params_path, 'w') as f:
        f.write("线性函数拟合参数\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"拟合公式:\n")
        f.write(
            f"  mean_time_ms = {slope:.10f} × (batch_size × kv_len) + {intercept:.10f}\n\n")
        f.write(f"参数:\n")
        f.write(f"  斜率 (slope): {slope:.10f}\n")
        f.write(f"  截距 (intercept): {intercept:.10f}\n\n")
        f.write(f"拟合质量:\n")
        f.write(f"  R² 分数: {r2:.10f}\n")
        f.write(f"  均方误差 (MSE): {mse:.10f}\n")
        f.write(f"  均方根误差 (RMSE): {rmse:.10f} ms\n")
        f.write(f"  相关系数: {correlation:.10f}\n\n")
        f.write(f"数据统计:\n")
        f.write(f"  数据点数量: {len(batch_kv_product)}\n")
        f.write(
            f"  Batch size × KV len 范围: [{batch_kv_product.min()}, {batch_kv_product.max()}]\n")
        f.write(
            f"  Mean time 范围: [{mean_time_ms.min():.6f}, {mean_time_ms.max():.6f}] ms\n")

    print(f"拟合参数已保存到: {params_path}")

    # 可选：显示残差图
    residuals = y - y_pred
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(batch_kv_product, residuals, color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Batch Size × KV Length', fontsize=12)
    plt.ylabel('Residuals (ms)', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Mean Time (ms)', fontsize=12)
    plt.ylabel('Residuals (ms)', fontsize=12)
    plt.title('Residuals vs Predicted', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    residual_path = os.path.join(
        result_dir, 'benchmark_flash_attn_linear_residuals.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    print(f"残差图已保存到: {residual_path}")

    plt.show()


if __name__ == '__main__':
    main()
