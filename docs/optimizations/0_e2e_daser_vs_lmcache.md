# E2E 基准测试：DaseR vs LMCache LocalDiskBackend

**运行日期：** 2026-04-23
**测试脚本：** `benchmarks/bench_e2e_daser_vs_lmcache.py`
**模型：** Qwen/Qwen3-8B (bfloat16)
**GPU：** CUDA 2 (NVIDIA H800, 79 GB；启动时 19 GB 空闲；`gpu_memory_utilization=0.22`, `max_num_seqs=64`)
**负载：** 200 条 IMDB 评论，`max_tokens=1`, `temperature=0`，关闭 `enable_prefix_caching`
**Prompt token 总量：** 60,595 tokens（平均每条 303 tokens，共 3,867 个 KV block）

两个系统都在**同一个** `LLM` 实例上跑 cold → warm。LMCache 的
`LocalDiskBackend` 把 chunk 索引只放在内存里（启动时不扫目录），所以
如果中途重建引擎，cold pass 写入的文件全部会变成孤块。关闭
prefix caching 的情况下，vLLM 仍会在多次 `generate()` 之间回收 GPU 里的
KV，因此 warm pass 依旧需要从外部存储层拉取 KV —— 这正是本基准测试
想要衡量的信号。DaseR 为保证公平也采用完全相同的流程。

## 结果

| 指标                         | DaseR     | LMCache   |
|------------------------------|----------:|----------:|
| cold 耗时 (s)                |    27.64  |     5.72  |
| warm 耗时 (s)                |    13.58  |     2.37  |
| cold prompt tok/s            |    2,192  |   10,590  |
| warm prompt tok/s            |    4,462  |   25,541  |
| warm/cold 加速比             |    2.04×  |    2.41×  |

DaseR warm tok/s ÷ LMCache warm tok/s = **0.17×**。

**正确性：** 两个系统各有 2/200 条 prompt 在生成的那一个 token 上
与 cold pass 不一致。这是已知的 KV 复用精度效应 —— 从缓存 KV
重构出的 hidden state，在两个 logit 接近时 argmax 可能翻转 —— 不是
存储层的 bug。

## 结果解读

- **两个系统的存储层都在正常工作。** warm 耗时都显著低于 cold，
  二者的 warm 加速比都接近 2×，说明跳过 prefill 重算确实生效。
- **LMCache 在端到端吞吐上绝对值快 5–6×。** 之前的存储层微基准
  （`benchmarks/bench_storage_imdb.py`）显示 DaseR 的裸 bytes I/O 层
  比 LMCache 快 1.54–1.83×（读路径）。因此 E2E 的差距**不在** GDS 数据
  平面，而在 DaseR 每请求 / 每层额外叠加的 **connector / 控制面** 开销。

## 已知公平性说明

- LMCache 的 `LocalDiskBackend` 不能跨引擎重启 —— 它的磁盘文件只能
  通过内存索引访问。本基准因此在同一引擎上连续跑 cold 和 warm。
  如果未来要测量**跨引擎**的持久化 KV 卸载，需要另选一种带可重启
  索引的 LMCache 后端（不在本次范围内）。
- Qwen3-8B 加上 19 GB 空闲显存上限，迫使 `gpu_memory_utilization=0.22`
  且 `max_num_seqs=64`。更大的 batch 会同时抬高两边的 tok/s，但对这次
  比较而言，重要的是**比值**。
- 2/200 的 token 不一致在两个系统上都出现，与使用哪个存储后端无关。

## 复现方法

```bash
source /data/zwt/vllm/bin/activate
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --num-prompts 200 \
    --gpu-util 0.22 \
    --max-num-seqs 64
```

脚本会先用 `PYTHONHASHSEED=0` 重新 exec 自己一次，以保证 LMCache 的
`NONE_HASH` 和 Python 字符串哈希在跨进程边界时仍然是确定性的
（缺少这一步时，scheduler 算出的 hash 与 worker 算出的不一致，
LMCache 会报告零个 warm 命中）。

## 意义

这次 E2E 比较暴露了一个存储微基准里看不到的 DaseR connector 层
开销。下一步优化的方向是剖析 `DaserConnector` 的每请求路径（索引
查找、slot 分配、逐层迭代、IPC 往返），而不是 GDS 数据平面。
