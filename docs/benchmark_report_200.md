# DaseR vs LMCache 性能对比基准测试报告

> 测试日期：2026-05-20 | GPU：NVIDIA H800 PCIe 80GB | 模型：Qwen3-8B | 驱动：vLLM 0.19.0

## 测试配置

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA H800 PCIe 80GB (CUDA 12.x, SM 90) |
| 模型 | Qwen3-8B (`/data/zwt/model/models/Qwen/Qwen3-8B`) |
| vLLM 版本 | 0.19.0 |
| DaseR | 当前分支 `perf/benchmark-profiling` |
| LMCache | 0.4.5 (LocalDisk Backend, O_DIRECT=off) |
| 采样参数 | temperature=0, max_tokens=1 |
| Block 大小 | 16 tokens/block |
| Slot 大小 | 2,359,296 bytes (36 层 × 2 × 16 × 8 × 128 × bf16) |

### 各数据集配置

| 数据集 | max_input_tokens | gpu_memory_utilization | max_num_seqs | max_model_len |
|--------|-----------------|----------------------|-------------|---------------|
| IMDB | 1,792 | 0.3 | 64 | 2,048 |
| LongBench | **32,768** | 0.6 | 8 | 33,024 |
| SCBench | **8,192** | 0.5 | 32 | 8,448 |

> LongBench 和 SCBench 使用更大的 `max_input_tokens` 以测试完整长文本场景。由于 GPU 显存限制（H800 80GB），需相应降低并发数（max_num_seqs），并适当提高 `gpu_memory_utilization`。

## 数据集说明

| 数据集 | 类型 | 200条 Prompt 总 Token | 200条 KV 数据量 | 中位数 Token/Prompt | 完整度 |
|--------|------|----------------------|-----------------|---------------------|--------|
| IMDB | 电影评论 | 60,613 | ~9.1 GB | 303 | 100% 完整 |
| LongBench (narrativeqa) | 长文档问答 | 4,867,924 | ~718 GB | 31,313 | ~50% 完整（≤32K），其余截断 |
| SCBench (scbench_choice_eng) | 多轮对话选择题 | 1,638,400 | ~242 GB | 148,582 | 全部截断（最短 70K → 8K） |

> LongBench 原始文本 8K～65K tokens，32K 截断可覆盖约 50% prompt 的完整内容。SCBench 原始文本 70K～745K tokens，8K 截断较激进。

---

## 一、端到端性能总览

### 1.1 IMDB — 短文本场景

> max_input_tokens=1,792 | gpu_util=0.3 | max_num_seqs=64

| 指标 | DaseR | LMCache | DaseR 优势 |
|------|-------|---------|------------|
| Cold 耗时 | 5.38 s | **4.22 s** | 0.79x（慢 27%） |
| Warm 耗时 | **0.88 s** | 2.37 s | **2.71x** |
| Cold 吞吐 (tok/s) | 11,272 | **14,356** | 0.79x |
| Warm 吞吐 (tok/s) | **69,268** | 25,576 | **2.71x** |
| Cold→Warm 加速比 | **6.14x** | 1.78x | — |

### 1.2 LongBench

> max_input_tokens=32,768 | gpu_util=0.6 | max_num_seqs=8

| 指标 | DaseR | LMCache | DaseR 优势 |
|------|-------|---------|------------|
| Cold 耗时 | **173.57 s** | 209.23 s | **1.21x** |
| Warm 耗时 | **32.69 s** | 182.77 s | **5.59x** |
| Cold 吞吐 (tok/s) | **28,046** | 23,266 | **1.21x** |
| Warm 吞吐 (tok/s) | **148,896** | 26,634 | **5.59x** |
| Cold→Warm 加速比 | **5.31x** | 1.14x | — |

### 1.3 SCBench

> max_input_tokens=8,192 | gpu_util=0.5 | max_num_seqs=32

| 指标 | DaseR | LMCache | DaseR 优势 |
|------|-------|---------|------------|
| Cold 耗时 | 127.86 s | **90.70 s** | 0.71x（慢 41%） |
| Warm 耗时 | **12.36 s** | 60.30 s | **4.88x** |
| Cold 吞吐 (tok/s) | 12,814 | **18,064** | 0.71x |
| Warm 吞吐 (tok/s) | **132,546** | 27,172 | **4.88x** |
| Cold→Warm 加速比 | **10.34x** | 1.50x | — |

---

## 二、Cold→Warm 加速比分析

这是衡量 KV Cache 效率的核心指标——加速比越高，说明缓存命中后带来的收益越大。

```
               DaseR     LMCache
IMDB           6.14x      1.78x
LongBench      5.31x      1.14x
SCBench       10.34x      1.50x
```

- **DaseR**：加速比在 5.3x～10.3x 之间，缓存命中后性能显著提升
- **LMCache**：加速比仅 1.14x～1.78x，磁盘读取几乎没有比重新计算快多少

**LMCache 磁盘缓存的低效**是其 LocalDisk Backend 的根本问题：读路径涉及 CPU 内存拷贝 + Python 层反序列化 + GPU 上传，无法与 GDS DMA 直接写入 GPU 显存的路径竞争。即使在长文本场景下依然如此（LongBench 32K：LMCache warm 182.77s vs 重新计算 209.23s）。

---

## 三、Warm 路径 DaseR/LMCache 吞吐比

| 数据集 | max_input_tokens | KV 数据量 | D/L Warm 吞吐比 | Cold 路径 DaseR 胜出？ |
|--------|-----------------|-----------|-----------------|----------------------|
| IMDB | 1,792 | ~9 GB | 2.71x | 否（慢 27%） |
| SCBench | 8,192 | ~242 GB | **4.88x** | 否（慢 41%） |
| LongBench | 32,768 | ~718 GB | **5.59x** | **是（快 1.21x）** |

**核心发现**：DaseR 的 warm 路径优势随数据规模增大而扩大。从 9 GB（2.71x）到 718 GB（5.59x），GDS DMA 路径的可扩展性碾压 LMCache 的 CPU 缓冲池路径。

DaseR cold 路径在小数据量和中等数据量下慢于 LMCache（GDS 写入 + IPC 同步开销），但在大数据量（718 GB LongBench）下反超——此时 LMCache 的 CPU 缓冲池成为瓶颈。

---

## 四、max_input_tokens 对性能的影响

以下对比展示提升截断上限前后，各数据集 DaseR/LMCache 的性能变化。

### 4.1 LongBench：1792 → 32768 tokens

| 指标 | 1792 tokens | 32K tokens | 变化 |
|------|------------|-----------|------|
| Tokens 总量 | 358,400 | 4,867,924 | **13.6x** |
| DaseR Cold | 7.66 s | 173.57 s | 22.7x |
| DaseR Warm | 3.13 s | 32.69 s | 10.4x |
| LMCache Cold | 14.80 s | 209.23 s | 14.1x |
| LMCache Warm | 12.91 s | 182.77 s | 14.2x |
| D/L Warm 比 | 4.13x | **5.59x** | ↑35% |

> LMCache cold/warm 均接近线性增长（~14x），说明其瓶颈在 IO 吞吐本身。DaseR warm 只增长 10.4x（亚线性），因为 GDS 读带宽未被饱和。D/L warm 比从 4.13x 提升到 5.59x，**长文本场景 DaseR 优势更大**。

### 4.2 SCBench：1792 → 8192 tokens

| 指标 | 1792 tokens | 8K tokens | 变化 |
|------|------------|----------|------|
| Tokens 总量 | 358,400 | 1,638,400 | **4.6x** |
| DaseR Cold | 23.93 s | 127.86 s | 5.3x |
| DaseR Warm | 2.96 s | 12.36 s | 4.2x |
| LMCache Cold | 20.85 s | 90.70 s | 4.3x |
| LMCache Warm | 11.70 s | 60.30 s | 5.2x |
| D/L Warm 比 | 3.96x | **4.88x** | ↑23% |

> 趋势与 LongBench 一致：数据量增大 4.6x，DaseR warm 仅增 4.2x（亚线性），LMCache warm 增 5.2x（线性+）。D/L warm 比从 3.96x 提升到 4.88x。

### 4.3 1792-token 下的去重效应（历史分析）

1792-token 截断造成严重的 prompt 重复，导致 cold pass 数据失真：

- LongBench 1792：20 篇文章 × 10 问题 → 截断后仅 **20 种唯一 prompt**（10% 不重复）
- SCBench 1792：51 个 context × ~4 turn → 截断后仅 **51 种唯一 prompt**（25.5% 不重复）

DaseR 的 `_alloc_or_get_chunk()` 在 `generate()` 内部自动去重：相同 token 序列的 chunk 只写入一次。因此 LongBench 1792 实际冷写仅占 10%，cold pass 速度被显著"美化"。**提升 max_input_tokens 后去重效应减弱，结果更接近真实冷写性能。**

后续考虑缓存文章+提问的方式测试，相比冷热测试更贴近实际场景。  

---

## 附录：复现命令

```bash
# IMDB 基准测试 (200 prompts, 短文本)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 DASER_PERF_LOG=1 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --num-prompts 200 --gpu-util 0.3 --max-num-seqs 64 \
    --out /data/ld/daser_test/imdb_200_profile.json

# LongBench 基准测试 (200 prompts, 32K 长文本)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 DASER_PERF_LOG=1 \
python benchmarks/bench_e2e_longbench.py \
    --num-prompts 200 --max-input-tokens 32768 --gpu-util 0.6 --max-num-seqs 8 \
    --out /data/ld/daser_test/longbench_200_full.json

# SCBench 基准测试 (200 prompts, 8K 长文本)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 DASER_PERF_LOG=1 \
python benchmarks/bench_e2e_scbench.py \
    --num-prompts 200 --max-input-tokens 8192 --gpu-util 0.5 --max-num-seqs 32 \
    --out /data/ld/daser_test/scbench_200_long.json
```
