# Qwen3 Model Timing Benchmark Script

## 概述

`run_model_timing.py` 是一个脚本，用于调用 Qwen3 模型，以指定的 batch size 和输入长度构造输入，并将模型记录的时间保存到文件中。

## 文件说明

- `run_model_timing.py` - 主要脚本
- `benchmark_configs.json` - 预定义的基准配置（可选）
- `timing_results/` - 输出目录（自动创建）
- `timing_results/timing_records.json` - 输出的时间记录

## 使用方法

### 基础用法

单个配置运行：
```bash
python run_model_timing.py --batch-size 1 --input-len 128
```

### 常用参数

```
--batch-size         输入的 batch size（默认：1）
--input-len          输入序列长度（默认：128）
--model-name         模型名称或路径（默认：Qwen/Qwen3-0.5B）
--device             计算设备（默认：cuda）
--num-runs           测量运行次数（默认：3）
--warmup-runs        预热运行次数（默认：1）
--output-dir         输出目录（默认：./timing_results）
--output-file        输出文件名（默认：timing_records.json）
--dtype              计算数据类型：float32 | float16 | bfloat16（默认：bfloat16）
--configs            JSON 配置文件路径（可选）
```

### 示例

**示例 1：基础运行**
```bash
python run_model_timing.py --batch-size 1 --input-len 128
```

**示例 2：多个不同配置（使用配置文件）**
```bash
python run_model_timing.py --configs benchmark_configs.json
```

**示例 3：指定输出目录和文件**
```bash
python run_model_timing.py \
  --batch-size 2 \
  --input-len 256 \
  --output-dir ./my_results \
  --output-file my_benchmark.json
```

**示例 4：更多测量次数和预热**
```bash
python run_model_timing.py \
  --batch-size 4 \
  --input-len 512 \
  --num-runs 5 \
  --warmup-runs 2 \
  --dtype bfloat16
```

**示例 5：使用较小的模型**
```bash
python run_model_timing.py \
  --batch-size 1 \
  --input-len 256 \
  --model-name Qwen/Qwen3-0.5B
```

## 输出格式

输出 JSON 文件包含以下信息：

```json
[
  {
    "timestamp": "2025-12-16T10:30:45.123456",
    "batch_size": 1,
    "input_len": 128,
    "total_time": 0.5234,
    "layer_details": [
      {
        "attn": 0.0123,
        "attn_cuda": 0.0120,
        "ffn": 0.0234,
        "ffn_cuda": 0.0230,
        "attn_details": {
          "attn_forward": 0.0120,
          "attn_forward_cuda": 0.0118
        }
      },
      ...
    ]
  }
]
```

## 时间记录说明

每个运行的结果包含：

- **timestamp**: 运行时间戳
- **batch_size**: 批次大小
- **input_len**: 输入长度
- **total_time**: 总执行时间（秒）
- **layer_details**: 各层的时间明细
  - **attn**: Attention 层执行时间（CPU时间）
  - **attn_cuda**: Attention 层执行时间（GPU时间）
  - **ffn**: FFN 层执行时间（CPU时间）
  - **ffn_cuda**: FFN 层执行时间（GPU时间）
  - **attn_details**: Attention 内部细节

## 配置文件格式

`benchmark_configs.json` 示例：

```json
{
  "configs": [
    {"batch_size": 1, "input_len": 128},
    {"batch_size": 1, "input_len": 256},
    {"batch_size": 2, "input_len": 128}
  ]
}
```

## 主要特性

✓ 自动加载 Qwen3 模型和配置
✓ 灵活的 batch size 和输入长度设置
✓ GPU 预热机制
✓ 详细的时间记录（CPU 和 GPU 时间分别记录）
✓ 支持多种数据类型（float32, float16, bfloat16）
✓ 支持批量配置运行
✓ 自动生成汇总统计信息
✓ JSON 格式输出便于后续分析

## 故障排除

### 模型加载失败
- 确保有网络连接以下载模型
- 检查 HuggingFace 访问权限
- 可以使用本地模型路径替代

### GPU 显存不足
- 减少 batch size
- 减少 input_len
- 切换到更小的模型版本

### 时间测量异常
- 确保 CUDA 设备可用
- 检查 GPU 驱动程序版本
- 增加 warmup_runs 数量

## 后续分析

可以使用生成的 JSON 文件进行：
- 性能对比分析
- 时间分布统计
- 瓶颈识别
- 扩展性评估
