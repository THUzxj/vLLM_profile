# Transformer Model Benchmarking Suite

一个全面的 Transformer 大模型推理性能测试套件，支持测量 prefill time、decode time（每个 token）和 total latency，可以测试不同 batch size 和 input length 组合下的性能表现。

## 功能特性

- 🚀 **精确的性能测量**: 使用TTFT（Time To First Token）和per token decode time准确测量 prefill time 和 decode time
- 🎯 **直接Token生成**: 跳过tokenizer编码/解码过程，直接生成指定长度的token序列，消除文本处理开销
- 📊 **多维度测试**: 支持不同 batch size 和 input length 的组合测试
- 📈 **详细的数据分析**: 自动生成可视化图表和统计分析报告
- 🔧 **灵活配置**: 支持多种预设配置和自定义参数
- 💾 **多格式输出**: 支持 JSON 和 CSV 格式的结果保存
- 🖥️ **跨平台支持**: 支持 CPU 和 GPU（CUDA）推理
- ⚡ **智能跳过**: 自动跳过超大配置以避免内存问题

## 文件结构

```
transformers_profile/
├── benchmark.py          # 主要的性能测试脚本
├── config.py            # 配置管理模块
├── analyze_results.py   # 数据分析和可视化脚本
├── requirements.txt     # 依赖包列表
└── README.md           # 使用说明（本文件）
```

## 快速开始

### 1. 环境安装

```bash
# 克隆或下载项目到本地
cd transformers_profile

# 安装依赖
pip install -r requirements.txt
```

### 2. 基本使用

#### 运行基准测试

```bash
# 使用默认配置测试 GPT-2 模型
python benchmark.py --model gpt2 --device auto

# 指定 GPU 设备
python benchmark.py --model gpt2 --device cuda:0

# 使用自定义配置文件
python benchmark.py --model gpt2 --config-file custom_config.json

# 指定输出文件名前缀
python benchmark.py --model gpt2 --output-file my_benchmark
```

#### 生成配置文件

```bash
# 查看可用的预设配置
python config.py --show-presets

# 生成快速测试配置
python config.py --create quick_test --output quick_test_config.json

# 生成中等规模测试配置
python config.py --create medium_scale --output medium_config.json
```

#### 分析测试结果

```bash
# 分析基准测试结果
python analyze_results.py benchmark_results_20241017_143022.json

# 指定输出目录
python analyze_results.py results.json --output-dir analysis_output

# 只生成报告，不生成图表
python analyze_results.py results.json --report-only
```

## 详细使用指南

### 配置参数说明

#### 基本测试参数

- **batch_sizes**: 批处理大小列表，如 `[1, 2, 4, 8, 16, 32, 64, 128, 256]`
- **input_lengths**: 输入序列长度列表，如 `[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]`
- **output_length**: 固定输出长度（所有测试使用相同值）
- **num_runs**: 每个配置重复测试的次数（用于计算平均值和标准差）
- **max_batch_input_product**: 最大批处理×输入长度乘积（默认131072），超过此值的实验将被跳过以避免内存问题

#### 生成参数

- **temperature**: 采样温度（0 表示贪心解码，>0 表示随机采样）
- **top_p**: Nucleus 采样参数

### 预设配置

| 配置名称 | 批处理大小 | 输入长度 | 输出长度 | 运行次数 | 适用场景 |
|---------|-----------|----------|----------|----------|----------|
| quick_test | [1, 2] | [32, 64] | 20 | 1 | 快速验证 |
| small_scale | [1, 2] | [32, 64, 128] | 30 | 2 | 小规模测试 |
| medium_scale | [1, 2, 4, 8, 16, 32] | [32, 64, 128, 256, 512, 1024, 2048] | 50 | 3 | 标准测试 |
| large_scale | [1, 2, 4, 8, 16, 32, 64, 128, 256] | [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] | 100 | 5 | 大规模测试 |
| batch_size_study | [1, 2, 4, 8, 16, 32, 64, 128, 256] | [256] | 50 | 5 | 批处理规模研究 |
| input_length_study | [1] | [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] | 50 | 5 | 输入长度研究 |

## 输出结果说明

### 性能指标

- **total_latency**: 总延迟时间（秒）
- **prefill_time**: 准确的 prefill 时间（TTFT - Time To First Token）（秒）
- **decode_time**: 准确的解码时间（剩余tokens的生成时间）（秒）
- **decode_time_per_token**: 每个 token 的解码时间（秒）
- **decode_tokens_count**: 用于计算decode时间的token数量
- **tokens_per_second**: 吞吐量（tokens/秒）
- **memory_usage**: 内存使用情况（CPU 和 GPU）

### 输出文件

1. **JSON 格式** (`benchmark_results_YYYYMMDD_HHMMSS.json`)
   - 完整的原始测试数据
   - 包含所有配置和运行信息

2. **CSV 格式** (`benchmark_results_YYYYMMDD_HHMMSS.csv`)
   - 表格形式的结果数据
   - 便于 Excel 等工具分析

3. **分析报告** (`analysis_report.md`)
   - Markdown 格式的详细分析报告
   - 包含性能总结和建议

4. **可视化图表** (`analysis_plots/` 目录)
   - `latency_heatmap.png`: 延迟热力图
   - `throughput_heatmap.png`: 吞吐量热力图
   - `batch_size_scaling.png`: 批处理大小扩展性分析
   - `input_length_scaling.png`: 输入长度扩展性分析
   - `time_breakdown.png`: 时间分解图
   - `performance_comparison.png`: 性能对比图
   - `memory_usage.png`: 内存使用分析（如果有数据）

## 高级用法

### 自定义配置

创建自己的配置文件 `my_config.json`：

```json
{
  "batch_sizes": [1, 4, 16, 64],
  "input_lengths": [128, 512, 2048, 8192],
  "output_length": 100,
  "num_runs": 3,
  "temperature": 0.7,
  "top_p": 0.95,
  "max_batch_input_product": 32768
}
```

使用自定义配置：

```bash
python benchmark.py --model your-model --config-file my_config.json
```

### 测试多个模型

```bash
# 顺序测试多个模型
for model in "gpt2" "distilgpt2" "microsoft/DialoGPT-small"
do
    python benchmark.py --model $model --output-file ${model//\//_}_results
done
```

### 批量分析结果

```bash
# 分析多个结果文件
for result_file in *_results.json
do
    echo "Analyzing $result_file..."
    python analyze_results.py $result_file --output-dir "analysis_$(basename $result_file .json)"
done
```

## 性能优化建议

### GPU 优化

1. **使用适当的数据类型**: 脚本自动使用 `float16` 以减少内存使用
2. **批处理优化**: 较大的 batch size 通常有更好的 GPU 利用率
3. **内存管理**: 测试会自动清理 GPU 缓存以避免 OOM

### 直接Token生成

本测试套件采用 **直接token生成** 的先进方法：

- **零开销生成**: 直接生成token ID，无需tokenizer编码/解码过程
- **精确长度控制**: 每个输入序列都有**严格**的指定token数量
- **批次多样性**: 每个batch项目使用不同的token模式以确保多样性
- **词汇表兼容**: 自动确保所有生成的token都在模型词汇表范围内

```python
# 示例：直接生成token tensor
input_tokens = benchmark.generate_input_tokens(batch_size=4, input_length=256)
print(f"Shape: {input_tokens.shape}")  # torch.Size([4, 256])
print(f"Data type: {input_tokens.dtype}")  # torch.int64
# 每个序列都恰好是256个tokens，可直接输入模型
```

**与传统方法对比：**

| 特性 | 传统文本方法 | 直接Token方法 |
|------|-------------|---------------|
| 生成速度 | 慢（需要编码/解码） | **快（直接生成）** |
| 长度精度 | 近似（依赖tokenizer） | **100%精确** |
| 处理开销 | 高（多次转换） | **零开销** |
| 跨模型一致性 | 依赖tokenizer差异 | **完全一致** |
| 大长度支持 | 受限于文本处理 | **无限制支持** |

### 测试最佳实践

1. **热身**: 脚本自动进行模型热身以确保稳定的时间测量
2. **多次运行**: 使用 `num_runs > 1` 获得更可靠的平均值
3. **系统监控**: 监控 CPU 和 GPU 使用情况以确保无其他进程干扰
4. **内存限制**: 系统会自动跳过 `batch_size × input_length` 超过阈值的实验以避免内存溢出
5. **直接Token输入**: 使用直接生成的token序列，消除tokenizer处理的变异性和开销

## 常见问题

### Q: 如何测试本地模型？
```bash
python benchmark.py --model /path/to/your/local/model --device auto
```

### Q: 内存不足怎么办？
- 减小 batch_sizes 和 input_lengths
- 使用 `--device cpu` 进行 CPU 测试
- 使用 `quick_test` 或 `small_scale` 配置

### Q: 如何只测试特定配置？
创建自定义配置文件，只包含需要的参数组合。

### Q: 结果不稳定怎么办？
- 增加 `num_runs` 值
- 确保系统负载较低
- 检查是否有其他 GPU 进程运行

## 示例脚本

### 完整测试流程

```bash
#!/bin/bash

# 1. 创建配置文件
python config.py --create medium_scale --output test_config.json

# 2. 运行基准测试
python benchmark.py --model gpt2 --config-file test_config.json --output-file gpt2_benchmark

# 3. 分析结果
python analyze_results.py gpt2_benchmark.json --output-dir gpt2_analysis

echo "测试完成！查看 gpt2_analysis/ 目录获取详细分析结果。"
```

### 对比测试脚本

```bash
#!/bin/bash

models=("gpt2" "distilgpt2")
config="medium_scale"

# 创建配置
python config.py --create $config --output ${config}_config.json

# 测试每个模型
for model in "${models[@]}"; do
    echo "Testing $model..."
    python benchmark.py --model $model --config-file ${config}_config.json --output-file ${model//\//_}_${config}
    
    # 分析结果
    python analyze_results.py ${model//\//_}_${config}.json --output-dir ${model//\//_}_analysis
done

echo "所有模型测试完成！"
```

## 贡献和反馈

如果您在使用过程中遇到问题或有改进建议，欢迎：

1. 提交 Issue 描述问题
2. 提交 Pull Request 贡献代码
3. 分享您的测试结果和经验

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。