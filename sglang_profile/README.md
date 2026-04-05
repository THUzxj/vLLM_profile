# sglang_profile

SGLang 分布式推理性能分析与基准测试工具集，支持 DeepSeek-V3、Qwen3-235B-A22B 等 MoE 模型。

支持三种部署模式：
- **单节点聚合**：所有 GPU 运行统一服务
- **单节点 PD 分离**：同一节点内 prefill/decode GPU 分组
- **多节点 PD 分离**：跨节点 prefill/decode 分离部署

---

## scripts_sc/ 目录

### 配置文件

#### `common_serve_args.sh`
所有 `launch_server_*.sh` 脚本 source 的公共参数配置。

主要配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ARCHITECTURE` | 硬件架构，`A`=Ampere，`H`=Hopper | 需手动设置 |
| `ENABLE_DP_ATTENTION` | 启用 DP Attention 优化 | 1 |
| `MEM_FRACTION_STATIC` | 静态显存占比 | 0.8 |
| `MEM_CHUNKED_PREFILL_SIZE` | Chunked prefill 大小（按 DP 自动调整） | 4096 |
| `ENABLE_EPLB` | 启用 Expert Parallel Load Balancing | 1 |
| `ENABLE_TBO` | 启用 Two-Batch Overlap | 0 |
| `CUDA_GRAPH_MAX_BS` | CUDA Graph 最大 batch size | 256 |
| `NFS_SHARED_DIR` | 多节点 IP 共享目录 | `/nfs/shared` |

架构差异：
- Hopper：启用 DeepEP（`normal` 或 `low_latency` 模式）
- Ampere：禁用 NVSHMEM（`DISABLE_NVSHMEM=1`）

PD 分离模式下，脚本根据 `RANK` 自动分配 `--disaggregation-mode prefill/decode`。

---

### 服务器启动脚本 (`launch_server_*.sh`)

每个脚本对应一种模型+架构组合，source `common_serve_args.sh` 后启动 `python -m sglang.launch_server`。

| 脚本 | 模型 | 架构 | 默认 DP/EP/TP |
|------|------|------|--------------|
| `launch_server_qwen3-235B-A22B_formal_torchprofile_mapping_H.sh` | Qwen3-235B-A22B | H | 8/8/8 |
| `launch_server_qwen3-235B-A22B_torchprofile_mapping_A.sh` | Qwen3-235B-A22B (1-layer remapped) | A | 4/4/4 |
| `launch_server_deepseek-v3_formal_torchprofile_mapping_H.sh` | DeepSeek-V3 | H | 8/8/8 |
| `launch_server_deepseek-v3_torchprofile_mapping_A.sh` | DeepSeek-V3 (1-layer remapped) | A | 4/4/4 |
| `launch_server_deepseek-v3_formal_torchprofile_pddisagg_H.sh` | DeepSeek-V3 PD 分离 | H | 8/8/8 |

通过环境变量覆盖默认配置：
```bash
DP=4 EP=4 TP=4 bash launch_server_qwen3-235B-A22B_formal_torchprofile_mapping_H.sh
```

结果输出到 `results_v3/server/<model>/dp{DP}_ep{EP}_tp{TP}_{DATE}/`。

---

### 基准测试脚本 (`bench_*.sh`)

#### `bench_one_batch_server_profile.sh`
基础 profiling 基准，调用 `bench_one_batch_server_058.py`。

```bash
BS=32 IL=1000 OL=101 PROFILE_STEPS=10 bash bench_one_batch_server_profile.sh
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `BS` | Batch size | 32 |
| `IL` | Input length | 32000 |
| `OL` | Output length | 101 |
| `PROFILE_STEPS` | Profiling 步数 | 10 |
| `ENABLE_NSYS_PROFILE` | 启用 Nsight Systems | 0 |
| `SKIP_WARMUP` | 跳过 warmup | 0 |

输出：`results_v3/client/<model>_il{IL}/{tag}_{DATE}/result.jsonl`

#### `bench_one_batch_server_profile_full.sh`
与上同，但 `PROFILE_STEPS` 默认 10000，使用 ShareGPT 数据集做完整 profiling。

#### `bench_one_batch_server_profile_metrics_trigger.sh`
指标触发式 profiling，调用 `bench_one_batch_server_profile_max_batch_058.py`。

持续发送请求，当服务器 running requests 达到阈值时自动触发 profiling（测试PD分离server时使用）。

```bash
BS=256 IL=1000 OL=100 \
  PROFILE_TRIGGER_THRESHOLD=0.9 \
  PROFILE_POLLING_INTERVAL=0.1 \
  SEND_INTERVAL=5.0 \
  bash bench_one_batch_server_profile_metrics_trigger.sh
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `PROFILE_TRIGGER_THRESHOLD` | 触发阈值（running/max 比例） | 0.9 |
| `PROFILE_POLLING_INTERVAL` | 轮询间隔（秒） | 0.1 |
| `PROFILE_DELAY_STEPS` | 触发后延迟步数 | 0 |
| `SEND_INTERVAL` | 请求发送间隔（秒） | 5.0 |
| `TOTAL_ROUNDS` | 总轮数，0=无限 | 0 |
| `UPDATE_MAX_RUNNING_REQS` | profiling 前更新 max_running_requests | 1 |
| `TARGET_DP_RANK` | 监控的 DP rank | 0 |
| `DECODE_URL` / `PREFILL_URL` | PD 分离时的 URL | 空 |

---

### `bench_extend_decode/` 子目录

用于测试 extend（有 KV cache 命中）场景下的 decode 性能，扫描不同 cached token length，比较extend与decode的时间比例。

#### 服务器启动脚本

| 脚本 | 说明 | 默认 GPU | 默认 DP/EP/TP |
|------|------|---------|--------------|
| `serve_single_node_aggregated.sh` | 单节点聚合模式，遍历配置数组启动 | 0,1 | 2/2/2 |
| `serve_single_node_aggregated_4rank.sh` | 单节点聚合模式，4 GPU | 0,1,2,3 | 4/4/4 |
| `serve_single_node_2p2d.sh` | 单节点 PD 分离，2 prefill + 2 decode | 0,1 / 2,3 | 2/2/2 |

均 source `common_serve_args.sh`，默认模型 `Tongyi-DeepResearch-30B-A3B`，结果输出到 `results_v3/server/`。

`serve_single_node_2p2d.sh` 额外启动 router（端口 8000，`cache_aware` 策略），prefill dist-init 127.0.0.1:29500，decode dist-init 127.0.0.1:29600。

#### 基准测试脚本

| 脚本 | 连接目标 | 结果目录 |
|------|---------|---------|
| `bench_single_node_aggregated.sh` | 单节点聚合服务器（端口 30000） | `results_single_node_unified/` |
| `bench_single_node_2p2d.sh` | PD 分离 router（端口 8000） | `results_single_node_pd/` |

两个脚本均调用 `bench_one_batch_server_058.py`，对 `CACHED_TOKEN_LENS` 列表中每个值循环测试：

```bash
# 聚合模式
BATCH_SIZE=4 OUTPUT_LEN=148 EXTEND_LEN=608 \
  CACHED_TOKEN_LENS="1000 2000 4000 8000 16000 32000" \
  bash bench_extend_decode/bench_single_node_aggregated.sh

# PD 分离模式（需先启动 serve_single_node_2p2d.sh）
BATCH_SIZE=4 bash bench_extend_decode/bench_single_node_2p2d.sh
```

每次测试的 `input_len = cached_len + EXTEND_LEN`，通过 `--cached-token-len` 传入 bench 程序触发 KV cache 预热。`bench_single_node_2p2d.sh` 额外传入 `--prefill-url` 和 `--decode-url` 用于 PD 分离模式的 server info 获取。

---

### 编排脚本

#### `launch_and_bench_server.sh`
聚合模式的主编排脚本，负责：
1. 调用 `register_node_ip.sh` 注册本节点 IP 到 NFS
2. 后台启动服务器（rank 0）或前台等待（其他 rank）
3. 轮询 `/health` 等待服务就绪（最长 32000s）
4. 通过 NFS 同步 `RESULT_DIR` 到所有节点
5. 仅 rank 0 执行 bench 脚本
6. bench 完成后优雅关闭服务器

```bash
# 典型调用方式（由 1node_test.sh 或 Xnodes_pddisag.sh 调用）
NODE_RANK=0 WORLD_SIZE=1 \
  bash launch_and_bench_server.sh \
    launch_server_qwen3-235B-A22B_formal_torchprofile_mapping_H.sh \
    bench_one_batch_server_profile.sh
```

#### `launch_and_bench_server_pddisagg.sh`
PD 分离模式编排脚本，在 `launch_and_bench_server.sh` 基础上额外：
- rank 0 启动 router（端口 8000，`cache_aware` 策略）
- 从 NFS 收集所有 prefill/decode 节点 IP，构建 router 命令
- bench 脚本通过 router URL 发送请求

#### `register_node_ip.sh`
将本节点 IP 写入 `${NFS_SHARED_DIR}/${DLC_JOB_ID}/master-0.ip` 或 `worker-N.ip`，供其他节点读取（因为集群无法通过DNS解析正常访问其他节点）。

---

### 单节点PD聚合启动

#### `single_node_aggregated.sh` / `single_node_aggregated_4rank.sh`
直接在本机启动聚合模式服务器，无需多节点编排。遍历配置数组（DP/EP/TP 组合）依次测试。

```bash
# 修改脚本内的配置数组后直接运行
bash single_node_aggregated.sh
```

#### `single_node_pd_2p2d.sh`
单机 2 prefill + 2 decode 的 PD 分离部署：
- Prefill workers：GPU 0,1，端口 30000，dist-init 127.0.0.1:29500
- Decode workers：GPU 2,3，端口 30001，dist-init 127.0.0.1:29600
- Router：端口 8000，`cache_aware` 策略

```bash
MODEL_PATH=/path/to/model bash single_node_pd_2p2d.sh
```

#### `1node_test.sh`
单节点快速冒烟测试，固定参数（DP=4, EP=4, TP=4, BS=32, IL=1000, OL=10）。

---

### 多节点 PD 分离

`2nodes_pddisag.sh` / `4nodes_pddisag.sh` / `8nodes_pddisag.sh` 是多节点 PD 分离的配置模板，在每个节点上分别执行同一脚本，通过 `NODE_RANK` 区分角色。

| 脚本 | Prefill 节点 | Decode 节点 | 总 DP/EP/TP |
|------|------------|------------|------------|
| `2nodes_pddisag.sh` | 1 | 1 | 8/8/8 |
| `4nodes_pddisag.sh` | 2 | 2 | 16/16/16 |
| `8nodes_pddisag.sh` | 4 | 4 | 32/32/32 |

使用方式（以 2 节点为例）：
```bash
# 节点 0（prefill）
NODE_RANK=0 WORLD_SIZE=2 bash 2nodes_pddisag.sh

# 节点 1（decode）
NODE_RANK=1 WORLD_SIZE=2 bash 2nodes_pddisag.sh
```

bench 脚本固定为 `bench_one_batch_server_profile_metrics_trigger.sh`（指标触发式）。

---

### 工具脚本

| 脚本 | 功能 |
|------|------|
| `stop_pd_workers.sh` | 停止所有 PD 分离 worker 进程 |
| `test_pd_setup.sh` | 验证 PD 分离环境是否就绪 |
| `compile_deep_gemm.sh` | 预编译 DeepGEMM 模块（调用 launch_server 的 compile 模式） |

---

## Bench 程序实现

### 调用链

```
bench_*.sh
  └── bench_one_batch_server_058.py          # 入口，解析 ServerArgs + BenchArgs
        └── bench_one_batch_server_internal_058.py  # 核心逻辑

bench_one_batch_server_profile_metrics_trigger.sh
  └── bench_one_batch_server_profile_max_batch_058.py   # 入口
        └── bench_one_batch_server_internal_profile_max_batch_058.py
```

---

### `bench_one_batch_server_internal_058.py`（核心）

#### BenchArgs 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch-size` | 并发请求数 | 1 |
| `--input-len` | 输入 token 数 | 1024 |
| `--output-len` | 输出 token 数 | 16 |
| `--run-name` | 结果标识名 | `""` |
| `--base-url` | 服务器地址 | `http://127.0.0.1:30000` |
| `--skip-warmup` | 跳过 warmup | False |
| `--measure` | 执行测量循环 | False |
| `--profile` | 启用 torch profiler | False |
| `--profile-steps` | profiling 步数 | 5 |
| `--profile-by-stage` | 按 prefill/decode 阶段分别 profile | False |
| `--profile-activities` | 活动类型：`CPU,GPU,CUDA_PROFILER,MEM,RPD` | `CPU,GPU` |
| `--use-nsys` | 使用 Nsight Systems | False |
| `--dataset-path` | 数据集路径（ShareGPT JSON） | `""` |
| `--cached-token-len` | 预热缓存的 token 数 | 0 |
| `--decode-url` / `--prefill-url` | PD 分离时分别指定 URL | `""` |
| `--result-filename` | 结果输出文件（jsonl） | `""` |

#### 主要函数

**`run_one_case()`**
执行单次基准测试：发送 `batch_size` 个请求，等待全部完成，统计 TTFT、TPOT、吞吐量。支持 torch profiler 和 nsys profiling。

**`run_one_case_with_metrics_trigger()`**
指标触发式测试：
- `RequestSenderThread`：后台线程按 `send_interval` 持续发送新请求
- `ProfileTriggerThread`：轮询 `/metrics` 端点，当 running requests ≥ `threshold × max_running_requests` 时调用 `/start_profile` 触发 profiling

**`run_profile_with_stages()`**
通过 HTTP 调用服务器的 `/start_profile` 端点，支持按 prefill/decode 阶段分别触发。

---

### `bench_one_batch_server_internal_profile_max_batch_058.py`

专为"满载触发"场景设计，核心目标是：**在服务器 running requests 达到目标 batch size 时自动触发 profiling**，从而捕获真实满载状态下的性能数据。

#### 整体执行流程

```
run_one_case_with_metrics_trigger()
  │
  ├─ 1. flush_cache（清空 KV cache）
  │
  ├─ 2. [可选] update_max_running_requests()
  │      等待服务器空闲（running_reqs == 0）
  │      POST /set_internal_state → 将 max_running_requests 设为 batch_size / dp_size
  │      验证：GET /get_server_info → effective_max_running_requests_per_dp
  │
  ├─ 3. 启动两个并发线程
  │      ├─ RequestSenderThread（发请求）
  │      └─ ProfileTriggerThread（监控指标 + 触发 profiling）
  │
  ├─ 4. 主线程等待，直到 profile 完成或超时（默认 600s）
  │
  └─ 5. 停止线程，收集结果，写入 result.jsonl
```

#### `RequestSenderThread` — 持续发送请求

按固定间隔（`send_interval`）循环发送请求，每轮发送 `batch_size × requests_per_batch_multiplier` 个请求。

- 使用 `ThreadPoolExecutor`（16 workers）异步提交，不等待响应即发下一轮
- 每轮重新采样 payload（random/dummy/mmmu 数据集）
- 停止条件：
  - `total_rounds > 0` 且已完成指定轮数
  - `wait_for_profile=True` 且 `ProfileTriggerThread` 已触发 profiling
- 停止后等待所有 pending 请求完成（最长 600s）

#### `ProfileTriggerThread` — 监控指标并触发 profiling

按 `polling_interval`（默认 0.1s）轮询 `/metrics`，解析 `sglang:num_running_reqs{dp_rank="N"}` 指标。

触发条件：
```
running_reqs >= trigger_count  （连续 3 次）
其中 trigger_count = (batch_size / dp_size) × trigger_threshold
```

触发后执行流程：
1. 可选等待 `profile_delay_steps` 个轮询周期
2. 调用 `run_profile_with_stages()` → POST `/start_profile`
3. 设置 `profile_started = True`，通知 `RequestSenderThread` 停止

#### 关键函数

**`get_running_requests(url)`**
从 `/metrics` 解析 `sglang:num_running_reqs` 总量（不区分 DP rank）。

**`get_running_requests_by_rank(url, dp_rank)`**
解析 `sglang:num_running_reqs{dp_rank="N", tp_rank="0", ...}` 标签，返回指定 DP rank 的 running requests 及完整 label dict。用于 `ProfileTriggerThread` 的精确监控。

**`get_all_running_requests_by_rank(url)`**
返回所有 DP rank 的 `{dp_rank: (value, labels)}` 字典，用于调试多 DP 场景。

**`update_max_running_requests(url, new_value, max_retries=30)`**
POST `/set_internal_state` → `{"server_args": {"max_running_requests": new_value}}`

返回值为 `List[bool]`（每个 DP rank 一个），全部为 True 才算成功。失败时每隔 1s 重试，最多 30 次。**要求服务器当前无活跃请求**，否则更新会失败。

**`get_max_running_requests_from_server(url)`**
GET `/get_server_info` → `internal_states[0].effective_max_running_requests_per_dp`，用于验证更新是否生效。

**`run_profile_with_stages(url, num_steps, activities, ...)`**
POST `/start_profile`，参数包括：
- `output_dir`：profiling 输出目录（`$SGLANG_TORCH_PROFILER_DIR/<timestamp>/`）
- `num_steps`：采集步数
- `activities`：`["CPU", "GPU"]` 或 `["CUDA_PROFILER"]`（nsys 模式）
- `profile_by_stage`：是否按 prefill/decode 阶段分别采集
- `profile_stages`：默认 `["decode"]`
- `merge_profiles`：是否合并所有 rank 的 trace

同时将 `/get_server_info` 的结果保存为 `server_args.json` 到同一目录。

#### `BenchArgs` 关键参数（metrics-trigger 专用）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--profile-trigger-threshold` | 触发阈值（running/max 比例） | 0.9 |
| `--profile-polling-interval` | 轮询间隔（秒） | 0.1 |
| `--profile-delay-steps` | 触发后延迟轮询周期数 | 0 |
| `--send-interval` | 请求发送间隔（秒） | 5.0 |
| `--total-rounds` | 总发送轮数，0=无限 | 1 |
| `--wait-for-profile` | profile 完成后才停止发送 | True |
| `--update-max-running-reqs` | profiling 前更新 max_running_requests | True |
| `--requests-per-batch-multiplier` | 每轮实际发送 = batch_size × 该值 | 1 |
| `--target-dp-rank` | 监控的 DP rank | 0 |
| `--decode-url` | PD 分离时 decode worker URL | `""` |

#### `BenchOneCaseResult` 输出字段

| 字段 | 说明 |
|------|------|
| `latency` | 整个 run 的总耗时（秒） |
| `input_throughput` | 输入吞吐（tok/s），基于平均 batch 延迟估算 |
| `output_throughput` | 输出吞吐（tok/s） |
| `last_ttft` | 用平均 batch 延迟近似（metrics-trigger 模式无精确 TTFT） |
| `last_gen_throughput` | 从 `/get_server_info` 读取的服务端 gen throughput |
| `profile_link` | profiling 输出目录路径 |

---

## 关键环境变量

| 变量 | 说明 |
|------|------|
| `NODE_RANK` | 当前节点编号（0=master） |
| `WORLD_SIZE` | 总节点数 |
| `ARCHITECTURE` | `A`（Ampere）或 `H`（Hopper） |
| `MODEL_PATH` | 模型路径 |
| `RESULT_DIR` | 结果输出目录 |
| `NFS_SHARED_DIR` | NFS 共享目录（多节点 IP 交换） |
| `DLC_JOB_ID` | 作业 ID，用于 NFS 目录命名 |
| `ENABLE_NSYS_PROFILE` | 启用 Nsight Systems profiling |
| `SGLANG_TORCH_PROFILER_DIR` | Torch profiler 输出目录 |
| `ENABLE_EPLB` | 启用 EPLB |
| `ENABLE_TBO` | 启用 Two-Batch Overlap |
| `MAX_RUNNING_REQUESTS_DECODE` | Decode worker 最大并发请求数 |
