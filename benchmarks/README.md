# Benchmarks

DaseR 性能基准测试套件，用于对比 DaseR 与 LMCache 在 KV Cache 场景下的冷/热加速表现。

## 文件概览

| 文件 | 用途 |
|------|------|
| `bench_e2e_daser_vs_lmcache.py` | IMDB 数据集端到端基准测试，对比 DaseR 与 LMCache 的 cold/warm 吞吐 |
| `bench_e2e_longbench.py` | LongBench 数据集端到端基准测试，支持多数据集切换 |
| `bench_e2e_scbench.py` | SCBench 长上下文多轮对话基准测试，支持 Parquet/JSONL 格式 |
| `profiler.py` | 独立 Profiling 模块，提供 GPU 级计时、NVTX 标注、Chrome Trace 导出 |

## 路径约定

数据根目录通过当前用户名动态解析：

```python
_USER = os.environ.get("USER") or os.environ.get("LOGNAME") or os.getlogin()
_DATA_DIR = f"/data/{_USER}"
```

`--store-dir`、`--dataset-dir` 等参数默认值基于 `_DATA_DIR` 自动拼合。模型路径 (`--model`) 和 IMDB 路径 (`--imdb`) 为硬编码默认值，可通过命令行参数覆盖。

## 通用参数

以下参数在所有三个基准测试中均可用：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-prompts` | 200 | 测试 prompt 数量 |
| `--model` | `/data/zwt/model/models/Qwen/Qwen3-8B` | 模型路径 |
| `--store-dir` | `{_DATA_DIR}/daser_test` | KV 存储目录（默认基于当前用户） |
| `--max-input-tokens` | 1792 | 单条 prompt 最大 token 数 |
| `--gpu-util` | 0.4 | GPU 显存利用率 |
| `--max-num-seqs` | 64 | 最大并发序列数 |
| `--skip-daser` | - | 跳过 DaseR 测试 |
| `--skip-lmcache` | - | 跳过 LMCache 测试 |
| `--profile` | - | 开启 GPU 级 Profiling |
| `--trace` | - | 导出 Chrome Trace（隐含 --profile） |
| `--out` | - | JSON 结果输出路径 |

---

## IMDB 基准测试

对比 DaseR 与 LMCache 在 IMDB 影评数据集上的 KV Cache 性能。

```bash
# 基础用法（200 条 prompt）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --num-prompts 200
```

### IMDB 额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--imdb` | `/data/zwt/imdb.csv` | IMDB CSV 文件路径 |

---

## LongBench 基准测试

使用 LongBench 数据集进行多任务性能对比。

```bash
# 默认数据集 (narrativeqa)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_longbench.py \
    --num-prompts 200

# 指定数据集
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_longbench.py \
    --num-prompts 200 --dataset qasper

# 开启 Profiling
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_longbench.py \
    --num-prompts 50 --profile
```

### LongBench 额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `narrativeqa` | 数据集名称（不含 `.jsonl` 后缀） |
| `--dataset-dir` | `data/ld/longbench_data/data` | LongBench JSONL 数据目录 |

---

## SCBench 基准测试

基于 SCBench 数据集的长上下文多轮对话 KV Cache 性能对比。每个 (example, turn) 对生成一条 prompt。支持 Parquet 和 JSONL 两种格式，自动检测。

```bash
# 默认数据集 (scbench_choice_eng)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_scbench.py \
    --num-prompts 200

# 指定数据集和目录
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_scbench.py \
    --num-prompts 200 --dataset scbench_qa_eng

# 开启 Profiling
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_scbench.py \
    --num-prompts 200 --profile
```

### SCBench 额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `scbench_choice_eng` | 数据集名称（对应 `<dataset-dir>/<dataset>/` 目录） |
| `--dataset-dir` | `/data/ld/SCBench` | SCBench 根目录 |

### 数据格式

SCBench 脚本按以下优先级加载数据：

1. **Parquet** — 扫描 `<dataset-dir>/<dataset>/test-*.parquet`
2. **JSONL** — 读取 `<dataset-dir>/data/<dataset>.jsonl`
3. **Fallback** — 少量合成 prompt，用于快速语法验证

每条数据必须包含 `context`（长文本）和 `multi_turns`（list of `{input, answer, options?}`）。选项型任务（如 `scbench_choice_eng`）会自动在 prompt 中渲染选项标签。

---

## Profiling 模块 (`profiler.py`)

独立的性能分析模块，**零侵入** —— 不对 `daser/connector/` 代码做任何修改。

### 三层数据采集

```
ProfilerContext (benchmark 级)
├── torch.cuda.Event       → GPU-stream 精确计时 (ms)
├── torch.cuda.nvtx        → NVTX 标注 (chrome://tracing 可视)
├── torch.profiler.profile → Chrome Trace 导出 (CPU + CUDA)
└── DASER_PERF_LOG=1       → 连接器内部直方图 (p50/p95/p99)
```

### 报告结构

运行 `--profile` 后输出三部分：

1. **Phase Timing** — 各阶段 wall-clock 和 GPU-stream 耗时对比
   - `cold_warmup` / `cold_generate` / `warm_warmup` / `warm_generate`
2. **DaseR Connector Latency Profile** — 连接器内部各操作的 p50/p95/p99
   - 冷写路径: `gds_write`, `ipc_sync_alloc_chunk`, `ipc_async_commit_chunk`
   - 热读路径: `gds_read`, `load_gds_wait`, `load_gpu_copy`, `ipc_sync_match_and_alloc`
3. **Cold vs Warm Summary** — 冷/热加速比、token/s 对比

### 直方图指标说明

| 操作 | 含义 | 出现阶段 |
|------|------|----------|
| `ipc_sync_match_and_alloc` | 连接器向 DaseR 服务端发起缓存查找+分配请求的同步 IPC 延迟 | 冷+热 |
| `ipc_sync_alloc_chunk` | 为未命中 chunk 分配存储槽位的同步 IPC 延迟 | 冷 |
| `ipc_async_commit_chunk` | 写入完成后提交 chunk 元数据的异步 IPC 延迟 | 冷 |
| `load_gds_wait` | 等待 GDS 读取完成的等待时间 | 热 |
| `load_gpu_copy` | 从暂存 buffer 拷贝 KV 到 GPU 缓存的耗时 | 热 |
| `gds_read` | cuFile/io_uring 从 NVMe 读取 KV 数据的 I/O 耗时 | 热 |
| `gds_write` | cuFile/io_uring 向 NVMe 写入 KV 数据的 I/O 耗时 | 冷 |
| `save_gpu_copy` | 从 GPU 缓存拷贝 KV 到暂存 buffer 的耗时 | 冷 |
| `save_gds_write` | 暂存 buffer 写入 NVMe 的异步提交耗时 | 冷 |
| `save_ipc_commit` | 写入完成后提交 chunk 的同步 IPC 延迟 | 冷 |

### 核心类

- **`ProfilerContext(name, trace_dir)`** — 封装单个系统（DaseR 或 LMCache）的性能数据采集。通过 `measure_generate(label, llm, prompts, params)` 方法记录每次 `llm.generate()` 调用的 wall-clock 和 GPU-stream 耗时。
- **`PhaseResult`** — 单个测试阶段的结果数据类，包含阶段名称、wall-clock 时间、GPU 时间、直方图数据。
- **`ProfilerReport`** — 比较报告生成器，接收两个 `ProfilerContext` 实例，输出上述三层报告。
- **`export_trace_json(phases, path)`** — 将阶段计时数据导出为轻量级 JSON trace 文件，可在 `chrome://tracing` 或 Perfetto 中加载。

### Chrome Trace 导出

使用 `--trace` 参数时，`torch.profiler.profile()` 会记录 CPU 调用栈和 CUDA kernel 时间线，导出到 `<store_dir>/traces/` 目录。每个阶段生成一个独立的 Chrome Trace JSON 文件，可直接拖入 `chrome://tracing` 查看 GPU 调度细节。

## 输出示例

```
================================================================================
Phase Timing (wall-clock)
--------------------------------------------------------------------------------
Phase                           DaseR(s)      LMCache(s)       D/L Ratio
--------------------------------------------------------------------------------
cold_warmup                        0.12s           0.15s           0.80x
cold_generate                     12.34s          14.56s           0.85x
warm_warmup                        0.10s           0.13s           0.77x
warm_generate                      6.78s           8.90s           0.76x

────────────────────────────────────────────────────────────────────────────────
DaseR Connector Latency Profile (Cold Pass)
────────────────────────────────────────────────────────────────────────────────
Operation                            p50       p95       p99   count
--------------------------------------------------------------------------------
ipc_sync_match_and_alloc         0.12ms     0.45ms     1.23ms       200
gds_write                        2.12ms     4.98ms     7.45ms       200
ipc_async_commit_chunk           0.08ms     0.21ms     0.45ms       200

────────────────────────────────────────────────────────────────────────────────
Cold vs Warm Summary
────────────────────────────────────────────────────────────────────────────────
Metric                    DaseR Cold    DaseR Warm  LMCache Cold  LMCache Warm
--------------------------------------------------------------------------------
Elapsed                       12.34s         6.78s        14.56s         8.90s
tok/s                         28,901        52,603        24,500        40,090
Warm/Cold Speedup                  —         1.82x             —         1.64x
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `DASER_PERF_LOG=1` | 开启连接器内部性能直方图采集 |
| `DASER_PERF_HISTOGRAM_PATH` | 直方图 JSON 导出路径（连接器 shutdown 时写入） |
| `DASER_PERF_CACHE_PATH` | 缓存命中统计 JSON 导出路径 |
| `PYTHONHASHSEED=0` | 确保跨 LLM 重建时 token hashing 稳定（测试脚本自动设置） |
| `USER` | 影响 `--store-dir`、`--dataset-dir` 等默认路径的数据根前缀 |

## 测试原理

### Cold / Warm 流程

```
Cold Pass:
  1. 启动 DaseR/LMCache 服务端
  2. 构建 LLM → 预热 → 执行 generate (prompts 首次写入 KV cache)  ← 计时
  3. 销毁 LLM → 触发 connector shutdown → 导出直方图

Warm Pass:
  4. 重建 LLM → 预热 → 执行 generate (从存储层加载已缓存的 KV) ← 计时
  5. 销毁 LLM → 导出直方图
```

- **DaseR**: LLM 重建后依赖 `IPCServer` 持久化的 chunk 状态来命中缓存
- **LMCache**: LLM 在冷/热之间不重建（其 `LocalDiskBackend` 的 chunk 索引存在内存中，重建会导致数据丢失）

