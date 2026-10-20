# CPU 编码可行性实验

本实验比较完整 slot 的离线 CPU 编码与 `master` 的 GPU 编码。
不接入服务写入端，不处理部分有效 token，不测量 TTFT 或 GPU 争用收益。
结果见 [实验报告](../../docs/optimizations/cpu_codec_feasibility.md)。

## 编码接口

显式编译共享库，不在安装或首次调用时编译，不依赖 Torch/pybind11 的构建系统。
需要 C++17 编译器，以及已有的 Python/NumPy 环境：

```bash
make -C daser/compression
```

默认载入 `daser/compression/libdaser_cpu_codec.so`。也可通过 Makefile 的
`OUTPUT` 参数将库写到外部目录，再设置 `DASER_CPU_CODEC_LIBRARY` 指向它。
安装包包含 C++ 源码和 Makefile，不包含预编译二进制。

```python
from daser.compression.cpu import CpuSlotEncoder
from daser.compression.codec import default_online_codebooks

encoder = CpuSlotEncoder(geometry, default_online_codebooks(geometry))
output = bytearray(geometry.slot_size)
try:
    result = encoder.encode_into(raw, output, slot_id=0)
    stored = memoryview(output)[: result.stored_length]
    print(result.mode, result.stored_length, encoder.workspace_bytes)
finally:
    encoder.close()
```

`geometry` 是 `CompressedStoreGeometry`，必须指定 `tile_scalars=256`。
`raw` 是完整、连续的 canonical `[layer, K/V, token, head, dim]` BF16 字节；
只读输入也可用。输出可写，至少容纳完整 RAW slot，与输入的地址范围不得重叠。
返回 `CpuEncodedSlot(mode, stored_length)`，未使用的输出尾部不属于记录。
只支持 master 的单页头部（最多 41 层）和小于 4 GiB 的 slot。

第一遍计数决定紧密排列的平面记录长度；第二遍直接生成两级 3-bit 位流、
低字节、逐 tile 前缀表。整槽记录向上对齐到 4 KiB；放不下时复制 canonical
RAW 数据。一个平面的转义流很大不会单独触发 CPU 回退，而 GPU 的逐平面
容量限制可能使同一输入回退。两种情况均通过现有 GPU 解码器恢复。

每个工作线程独占一个实例。C 调用释放 GIL，无内部线程池或逐 slot 原生堆分配；
实例关闭和并发调用互斥。`workspace_bytes` 包含原生对象、查找表和计数数组，
不含 Python 对象、分配器元数据、调用者输入输出或布局转换缓冲。
这是同步计算接口，只能用于离线实验或调用者自己的 executor。

内容哈希字段沿用 GPU 的 `01` 加 31 个零字节占位值，码表 SHA-256 正常保存。
参考 `decode_slot` 的 `verify_hash=True` 默认值未改；验证本实验输出时显式设置
`verify_hash=False`，随后逐字节比较原始数据。不得把占位值当作内容完整性校验。

## 采集与执行

使用仓库支持的 CUDA 环境：Torch、Transformers、TileLang、NumPy、pytest，
以及本地 BF16 Llama-3.1-8B 权重。采集通过 Transformers 的 `LlamaModel`
获得 post-RoPE K 和 V，然后构造与 vLLM `BLNHC` 相同的物理布局；
没有修改或启动 vLLM/LMCache 服务。

先用 `nvidia-smi`、`nvidia-smi topo -m` 和 `numactl --hardware` 核对空闲 GPU
及其 NUMA 节点。以下变量由操作者按当前环境设置，scratch 必须位于仓库外：

```bash
export MODEL_DIR=/path/to/Meta-Llama-3.1-8B
export DASER_TEST_SCRATCH_ROOT=/path/to/approved-scratch
export GPU_ID=0
export NUMA_NODE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
RUN_DIR=$(mktemp -d "$DASER_TEST_SCRATCH_ROOT/cpu-codec.XXXXXXXX")
export DASER_CPU_CODEC_LIBRARY="$RUN_DIR/libdaser_cpu_codec.so"

make -C daser/compression OUTPUT="$DASER_CPU_CODEC_LIBRARY"

numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
  python -m benchmarks.compression.capture_cpu_codec \
  --model "$MODEL_DIR" --output "$RUN_DIR/capture" \
  --slots 64 --block-tokens 128

pytest -q tests/compression/test_cpu_codec.py

numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
  python -m benchmarks.compression.bench_cpu_codec \
  --capture "$RUN_DIR/capture" --output "$RUN_DIR/results.json" \
  --threads 1 2 4 8 --batch-sizes 1 4 16 \
  --warmups 5 --iterations 30 --rounds 3
```

采集命令要求输出目录尚不存在，避免覆盖已有数据。`manifest.json` 保存固定文本、
实际 token IDs、种子、模型配置、软件版本和全数据／逐 slot 摘要。默认采集
8192 个 token，64 个完整 slot，每个 16 MiB，总计 1 GiB。
同一数据用于 CPU 和 GPU，并且使用未经校准的默认码表。

benchmark 先验证全部真实 slot：

- CPU 编码 → 参考解码，显式跳过占位内容哈希后逐字节比较。
- CPU 编码 → 现有 GPU 解码，比较 BF16 原始位模式。
- 现有 GPU 编码 → 现有 GPU 解码，比较相同输入。
- GPU 物理布局 → D2H → CPU canonical 布局，与采集数据比较。

参考解码使用最多 4 个 `spawn` 子进程加速离线校验；全部子进程结束后才开始
性能测量。编码测量只使用前述固定线程池。

边界测试另外覆盖 primary、secondary、entry 14/unmapped 转义、
NaN/Inf/有符号零/次正规数、全部 65536 种 BF16 位模式、tile 尾块、
RAW 回退、不同逐平面码表、混合批次、非连续目的 block 和输出缓冲复用。

## 计时和内存口径

每组每轮先预热 5 次，再测量 30 次，共 3 轮。CPU 的 12 个线程数／批大小组合
分别测两条路径，GPU 测 3 个批大小，共 27 组、2430 个稳态样本。
按批大小轮转读取全部 64 个 slot。线程池固定，实际工作线程数为
`min(requested_threads, batch_size)`，不创建嵌套线程池。

| 路径 | 计时范围 |
|---|---|
| `cpu_encode` | 已驻留主机内存的 canonical 输入 → 固定线程池分派 → CPU 编码 → 主机输出 |
| `cpu_full` | GPU 物理 KV → pinned D2H 和同步 → 每线程 CPU 布局转换 → CPU 编码 → 主机输出 |
| `gpu_full` | GPU 物理 KV → 现有 GPU 编码／紧密排列／元数据同步 → 仅有效记录的 pinned D2H → 同步完成 |

计时使用 `perf_counter_ns`。每批返回后检查 mode/length；每轮最后一批在计时外
核对有效记录 SHA-256。GPU 未定义的尾部对齐填充不参与摘要。
编译、模型加载、采集、初始 H2D、工作区分配、校验和结果写盘均在计时外。
没有跨批流水重叠，也没有并发推理负载。

JSON 包含全部 90 个样本及逐轮、聚合 median/p95，按原始字节计算的十进制 GB/s，
原始字节／存储字节压缩率、RAW 回退比例，以及各路径缓冲容量、原生工作区、
进程 RSS/峰值 RSS、Torch GPU allocated/reserved。1 GiB 主机语料和
1 GiB GPU 语料作为共享基线单列。RSS 包含解释器、编译器库、缓存等，不能代替
逐缓冲容量的归因；GPU allocated/reserved 也不包含所有驱动上下文开销。

所有命令结束后，先将 `results.json`、`capture/manifest.json` 和需要的测试结果
保存到持久结果目录，确认进程已退出，再直接删除本次 `RUN_DIR`。模型权重和其他
任务的数据不属于清理范围，不保留 `slots.npy` 或临时 store 文件。
