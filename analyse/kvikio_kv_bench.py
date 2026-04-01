# SPDX-License-Identifier: Apache-2.0
"""Qwen3-8B KV Cache 从 SSD 加载到 GPU 的吞吐量基准测试。

使用 kvikio 的 CuFile 接口（GPU Direct Storage / compat 模式）测试
将 KV cache 张量从 /data/zwt/fake_cache 加载到 GPU 的理论极限吞吐量。

测试流程：
  1. 生成阶段：按 Qwen3-8B 的 KV cache 格式，生成模拟 .bin 文件到 fake_cache。
  2. 单文件顺序 read：测量单个大文件的连续读取吞吐。
  3. 多文件并发 pread（异步）：提交多个 CuFile.pread() 后统一等待，
     模拟 LMCache 加载多个 chunk 的真实场景。
  4. 正确性验证：确保读回的张量数据与写入时完全一致。

用法：
  python analyse/kvikio_kv_bench.py [--gpu 0] [--num-chunks 10]
                                    [--chunk-size 256] [--no-gen]

Qwen3-8B KV cache 参数（bfloat16）：
  - 层数 (num_layers)   : 36
  - KV head 数          : 8
  - head_dim            : 128
  - 每个 chunk 每层大小 : 2 * chunk_size * 8 * 128 * 2 bytes
    → chunk_size=256  → 1 MB / 层，36 MB / chunk（全部层合一文件）
"""

# Standard
from __future__ import annotations

import argparse
import math
import os
import time
import unicodedata
from pathlib import Path
from typing import List, Tuple

# Third Party
import torch

# ─────────────────────────── 表格对齐工具 ───────────────────────────────────


def _display_width(s: str) -> int:
    """返回字符串的终端显示宽度（中文等全角字符占2列）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _pad_row(content: str, total_width: int) -> str:
    """在 content 右侧补空格，使显示宽度恰好达到 total_width。"""
    return content + " " * (total_width - _display_width(content))


# ─────────────────────────────── 常量 ───────────────────────────────────────

MODEL_NAME = "Qwen3-8B"
# Qwen3-8B 架构参数
NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16

FAKE_CACHE_DIR = Path("/data/zwt/fake_cache")

# ─────────────────────────── KV cache 尺寸计算 ───────────────────────────────


def kv_shape(chunk_size: int) -> Tuple[int, ...]:
    """返回单层、单 chunk 的 KV tensor shape。

    Shape 定义遵循 LMCache 约定：
      [2, num_kv_heads, chunk_size, head_dim]
      其中 2 = K / V 两个矩阵。

    Args:
        chunk_size: 每个 chunk 包含的 token 数量。

    Returns:
        (2, num_kv_heads, chunk_size, head_dim) 的整数元组。
    """
    return (2, NUM_KV_HEADS, chunk_size, HEAD_DIM)


def bytes_per_layer(chunk_size: int) -> int:
    """每层 KV cache 的字节数。

    Args:
        chunk_size: 每个 chunk 包含的 token 数量。

    Returns:
        字节数（bfloat16 = 2 字节/元素）。
    """
    return math.prod(kv_shape(chunk_size)) * 2  # bfloat16 = 2 bytes


def bytes_per_chunk(chunk_size: int) -> int:
    """一个 chunk 所有层 KV cache 的总字节数（拼合为单文件）。

    Args:
        chunk_size: 每个 chunk 包含的 token 数量。

    Returns:
        总字节数。
    """
    return bytes_per_layer(chunk_size) * NUM_LAYERS


# ─────────────────────────────── 生成阶段 ────────────────────────────────────


def generate_fake_cache(
    num_chunks: int,
    chunk_size: int,
    cache_dir: Path,
) -> List[Path]:
    """生成模拟的 KV cache 二进制文件。

    每个文件代表一个 chunk 的全部 36 层 KV cache，数据为随机 bfloat16 张量。

    Args:
        num_chunks: 生成的 chunk 数量。
        chunk_size: 每个 chunk 包含的 token 数量。
        cache_dir: 缓存文件保存目录。

    Returns:
        生成的 .bin 文件路径列表。
    """
    # 清空旧文件，避免 chunk_size 不同导致的大小不匹配
    if cache_dir.exists():
        import shutil  # noqa: PLC0415
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = bytes_per_chunk(chunk_size)
    file_paths: List[Path] = []

    print(f"\n[生成] 正在生成 {num_chunks} 个 chunk 文件...")
    print(f"  每 chunk 大小: {total_bytes / 1024 / 1024:.2f} MB  ({NUM_LAYERS} 层 × {bytes_per_layer(chunk_size)/1024:.0f} KB)")

    torch.manual_seed(42)
    for i in range(num_chunks):
        bin_path = cache_dir / f"chunk_{i:04d}.bin"

        # bfloat16 不支持 numpy，先 view 成 uint8 再转
        flat = torch.cat([
            torch.randn(kv_shape(chunk_size), dtype=DTYPE).flatten()
            for _ in range(NUM_LAYERS)
        ])
        bin_path.write_bytes(flat.view(torch.uint8).numpy().tobytes())

        file_paths.append(bin_path)
        if (i + 1) % max(1, num_chunks // 5) == 0 or i == num_chunks - 1:
            print(f"  已生成 {i + 1}/{num_chunks} 个 chunk")

    print(f"[生成] 完成，文件保存至 {cache_dir}")
    return file_paths


# ─────────────────────────── 单文件顺序读测试 ────────────────────────────────


def bench_sequential_read(
    file_paths: List[Path],
    chunk_size: int,
    gpu_device: torch.device,
    warmup_rounds: int = 2,
    bench_rounds: int = 5,
) -> None:
    """用 kvikio CuFile 逐文件顺序读取，测量吞吐量。

    每轮读取所有文件，对 bench_rounds 轮取平均。

    Args:
        file_paths: 待读取的 .bin 文件列表。
        chunk_size: 每个 chunk 包含的 token 数量。
        gpu_device: 目标 GPU 设备。
        warmup_rounds: 预热轮数（不计入统计）。
        bench_rounds: 正式测量轮数。
    """
    import kvikio  # noqa: PLC0415

    total_bytes = bytes_per_chunk(chunk_size)
    n_elements = total_bytes // 2  # bfloat16 元素数

    print(f"\n[顺序读] 单文件逐一读取 — {len(file_paths)} 个 chunk")
    print(f"  预热 {warmup_rounds} 轮，正式 {bench_rounds} 轮")

    # 预分配 GPU buffer（每个文件复用同一块，bfloat16 直接分配）
    gpu_buf = torch.empty(n_elements, dtype=torch.bfloat16, device=gpu_device)

    def _run_one_round() -> float:
        """读取所有文件，返回耗时（秒）。"""
        t0 = time.perf_counter()
        for fp in file_paths:
            with kvikio.CuFile(str(fp), "r") as cf:
                cf.read(gpu_buf)
        return time.perf_counter() - t0

    # 预热
    for _ in range(warmup_rounds):
        _run_one_round()

    # 正式测量
    times: List[float] = []
    for r in range(bench_rounds):
        elapsed = _run_one_round()
        times.append(elapsed)
        print(f"  Round {r + 1}: {elapsed:.4f}s")

    _report("顺序读（CuFile.read）", times, file_paths, total_bytes)


# ──────────────────────────── 多文件并发 pread 测试 ──────────────────────────


def bench_concurrent_pread(
    file_paths: List[Path],
    chunk_size: int,
    gpu_device: torch.device,
    warmup_rounds: int = 2,
    bench_rounds: int = 5,
) -> None:
    """用 kvikio CuFile.pread 并发提交所有文件的读请求，测量吞吐量。

    提交所有文件的异步 pread，再统一等待 future.get()，模拟
    LMCache async_load 场景下的极限 I/O 并发度。

    Args:
        file_paths: 待读取的 .bin 文件列表。
        chunk_size: 每个 chunk 包含的 token 数量。
        gpu_device: 目标 GPU 设备。
        warmup_rounds: 预热轮数（不计入统计）。
        bench_rounds: 正式测量轮数。
    """
    import kvikio  # noqa: PLC0415

    total_bytes = bytes_per_chunk(chunk_size)
    n_elements = total_bytes // 2  # bfloat16 元素数

    print(f"\n[并发 pread] 全部文件并发异步读取 — {len(file_paths)} 个 chunk")
    print(f"  预热 {warmup_rounds} 轮，正式 {bench_rounds} 轮")

    # 为每个文件预分配独立 GPU buffer（bfloat16）
    gpu_bufs = [
        torch.empty(n_elements, dtype=torch.bfloat16, device=gpu_device)
        for _ in file_paths
    ]

    def _run_one_round() -> float:
        """并发提交所有 pread，等待完成，返回耗时（秒）。"""
        handles = []
        t0 = time.perf_counter()
        for fp, buf in zip(file_paths, gpu_bufs):
            cf = kvikio.CuFile(str(fp), "r")
            future = cf.pread(buf)
            handles.append((cf, future))

        for cf, future in handles:
            future.get()
            cf.close()

        return time.perf_counter() - t0

    # 预热
    for _ in range(warmup_rounds):
        _run_one_round()

    # 正式测量
    times: List[float] = []
    for r in range(bench_rounds):
        elapsed = _run_one_round()
        times.append(elapsed)
        print(f"  Round {r + 1}: {elapsed:.4f}s")

    _report("并发 pread（CuFile.pread）", times, file_paths, total_bytes)


# ────────────────────────────── 正确性验证 ───────────────────────────────────


def verify_correctness(
    file_paths: List[Path],
    chunk_size: int,
    gpu_device: torch.device,
    num_verify: int = 3,
) -> None:
    """验证从 SSD 加载到 GPU 的数据与原始数据一致。

    逐元素比较 GPU 张量与 .ref 文件中的参考数据，确保 load 过程无损。

    Args:
        file_paths: .bin 文件路径列表。
        chunk_size: 每个 chunk 包含的 token 数量。
        gpu_device: 目标 GPU 设备。
        num_verify: 验证的文件数量（从列表头部取）。

    Raises:
        AssertionError: 当任一文件数据不一致时抛出。
    """
    import kvikio  # noqa: PLC0415

    total_bytes = bytes_per_chunk(chunk_size)
    n_elements = total_bytes // 2  # bfloat16

    sample_paths = file_paths[:num_verify]
    print(f"\n[正确性验证] 验证前 {len(sample_paths)} 个 chunk...")

    all_passed = True
    for fp in sample_paths:
        # 从 GPU 读回（直接 bfloat16 buffer）
        gpu_buf = torch.empty(n_elements, dtype=torch.bfloat16, device=gpu_device)
        with kvikio.CuFile(str(fp), "r") as cf:
            cf.read(gpu_buf)

        gpu_tensor = gpu_buf.cpu()
        # bytearray 可写，避免 torch.frombuffer 的只读警告
        ref_tensor = torch.frombuffer(bytearray(fp.read_bytes()), dtype=torch.bfloat16)

        if not torch.equal(gpu_tensor, ref_tensor):
            diff = (gpu_tensor != ref_tensor).sum().item()
            print(f"  ✗ {fp.name}: {diff}/{ref_tensor.numel()} 个元素不一致")
            all_passed = False
        else:
            print(f"  ✓ {fp.name}: 数据完全一致 ({ref_tensor.numel()} 元素, {total_bytes/1024/1024:.2f} MB)")

    if all_passed:
        print("[正确性验证] ✅ 全部通过！")
    else:
        print("[正确性验证] ❌ 存在错误！")
        raise AssertionError("KV cache 数据正确性验证失败，请检查 load 流程。")


# ─────────────────────────────── 报告输出 ────────────────────────────────────


def _report(
    label: str,
    times: List[float],
    file_paths: List[Path],
    bytes_per_file: int,
) -> None:
    """打印吞吐量统计报告。

    每个 times[i] 是第 i 轮读完所有 chunk 的耗时，
    bytes_per_file * len(file_paths) 是一轮读取的总数据量。

    Args:
        label: 测试名称。
        times: 每轮耗时（秒）列表，每轮均读取全部 chunk。
        file_paths: 被读取的文件列表。
        bytes_per_file: 单个文件的字节数。
    """
    # 一轮读取的总数据量
    round_bytes = bytes_per_file * len(file_paths)
    total_gib = round_bytes / (1024 ** 3)

    avg_t = sum(times) / len(times)
    best_t = min(times)   # 耗时最短的轮次
    worst_t = max(times)  # 耗时最长的轮次
    avg_tp = total_gib / avg_t
    max_tp = total_gib / best_t   # 吞吐 = 数据量 / 时间，时间越短吞吐越高
    min_tp = total_gib / worst_t

    # 表格内容宽度（不含左侧 "  │" 和右侧 "│"）
    INNER = 54

    rows = [
        f"  文件数量     : {len(file_paths):>8} 个",
        f"  单轮数据量   : {total_gib:>8.3f} GiB",
        f"  平均耗时     : {avg_t:>8.4f} s",
        f"  平均吞吐     : {avg_tp:>8.2f} GiB/s",
        f"  最大吞吐     : {max_tp:>8.2f} GiB/s",
        f"  最小吞吐     : {min_tp:>8.2f} GiB/s",
    ]

    sep = "─" * INNER
    print(f"\n  ┌{sep}┐")
    print(f"  │{_pad_row(' ' + label, INNER)}│")
    print(f"  ├{sep}┤")
    for row in rows:
        print(f"  │{_pad_row(row, INNER)}│")
    print(f"  └{sep}┘")


# ─────────────────────────────── 系统信息 ────────────────────────────────────


def print_system_info(gpu_device: torch.device, chunk_size: int, num_chunks: int) -> None:
    """打印 GPU、kvikio 及测试参数信息。

    Args:
        gpu_device: 使用的 GPU 设备。
        chunk_size: 每个 chunk 包含的 token 数量。
        num_chunks: 测试文件数量。
    """
    import kvikio  # noqa: PLC0415
    import kvikio.defaults  # noqa: PLC0415

    gpu_name = torch.cuda.get_device_name(gpu_device)
    total_mem = torch.cuda.get_device_properties(gpu_device).total_memory

    try:
        compat = kvikio.defaults.get("compat_mode")
    except Exception:
        compat = "unknown"

    chunk_bytes = bytes_per_chunk(chunk_size)
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Qwen3-8B KV Cache SSD→GPU 吞吐量基准测试")
    print(sep)
    print(f"  GPU           : {gpu_name}")
    print(f"  GPU 显存      : {total_mem / 1024**3:.1f} GiB")
    print(f"  kvikio 版本   : {kvikio.__version__}")
    print(f"  kvikio compat : {compat}")
    print(f"  模型          : {MODEL_NAME}")
    print(f"  层数          : {NUM_LAYERS}")
    print(f"  KV heads      : {NUM_KV_HEADS}")
    print(f"  head_dim      : {HEAD_DIM}")
    print(f"  数据类型      : {DTYPE}")
    print(f"  chunk_size    : {chunk_size} tokens")
    print(f"  每 chunk 大小 : {chunk_bytes/1024/1024:.2f} MB")
    print(f"  chunk 数量    : {num_chunks}")
    print(f"  总数据量      : {chunk_bytes*num_chunks/1024/1024:.2f} MB")
    print(f"  缓存目录      : {FAKE_CACHE_DIR}")
    print(sep)


# ──────────────────────────────── 入口 ──────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Qwen3-8B KV Cache SSD→GPU 吞吐量基准测试（kvikio）"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="使用的 GPU 编号（CUDA 设备索引，默认 0）"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=20,
        help="测试用 chunk 数量（默认 20）"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256,
        help="每个 chunk 的 token 数（默认 256，与 LMCache 一致）"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="预热轮数（默认 2）"
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="正式测量轮数（默认 5）"
    )
    parser.add_argument(
        "--no-gen", action="store_true",
        help="跳过生成阶段（文件已存在时使用）"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="仅运行正确性验证，跳过吞吐测试"
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口：生成 → 顺序读基准 → 并发 pread 基准 → 正确性验证。"""
    args = parse_args()

    # 设置 GPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_device = torch.device(f"cuda:{args.gpu}")

    print_system_info(gpu_device, args.chunk_size, args.num_chunks)

    # ── 1. 生成 fake cache 文件 ────────────────────────────────────────────
    if not args.no_gen:
        file_paths = generate_fake_cache(
            num_chunks=args.num_chunks,
            chunk_size=args.chunk_size,
            cache_dir=FAKE_CACHE_DIR,
        )
    else:
        file_paths = sorted(FAKE_CACHE_DIR.glob("chunk_*.bin"))
        print(f"\n[跳过生成] 找到 {len(file_paths)} 个已有 chunk 文件")

    if not file_paths:
        print("错误：没有找到任何 chunk 文件，请先运行不带 --no-gen 的命令生成文件。")
        return

    # ── 2. 正确性验证 ────────────────────────────────────────────────────
    verify_correctness(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        gpu_device=gpu_device,
        num_verify=min(3, len(file_paths)),
    )

    if args.verify_only:
        return

    # ── 3. 顺序读基准测试 ────────────────────────────────────────────────
    bench_sequential_read(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        gpu_device=gpu_device,
        warmup_rounds=args.warmup,
        bench_rounds=args.rounds,
    )

    # ── 4. 并发 pread 基准测试 ───────────────────────────────────────────
    bench_concurrent_pread(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        gpu_device=gpu_device,
        warmup_rounds=args.warmup,
        bench_rounds=args.rounds,
    )

    print("\n✅ 所有测试完成！")


if __name__ == "__main__":
    main()
