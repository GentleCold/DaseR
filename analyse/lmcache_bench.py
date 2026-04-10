# SPDX-License-Identifier: Apache-2.0
"""LMCache vs 原生 vLLM 推理性能对比基准测试。

工作流：
  Cold 阶段 ── 使用 LMCacheConnectorV1 推理纯 Data 部分，将 KV Cache 写入 SSD。
  Warm 阶段 ── 使用同一 LLM 推理 Data+Task，从 SSD 加载 Data 前缀 KV Cache。
  Baseline  ── 不使用 LMCache，直接推理 Data+Task。

用法示例：
  python analyse/lmcache_bench.py --num-samples 20 --gpu 0
  python analyse/lmcache_bench.py --num-samples 20 --gpu 0 --output-json analyse/results.json
"""

# Standard
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# 必须在 import torch / vllm 之前设置，保证 Cold/Warm 两阶段 block hash 一致
os.environ.setdefault("PYTHONHASHSEED", "0")

# Third Party
import pandas as pd
import torch


# ─────────────────────────── 常量 ────────────────────────────

MODEL = "/data/zwt/model/models/Qwen/Qwen3-8B"
DATA_PATH = "/data/zwt/imdb.csv"
KV_DIR = "/data/zwt/lmcache_kv/"
TASK = (
    'Given the above film review, answer whether the sentiment is "positive" or "negative". '
    'Respond ONLY with "positive" or "negative", in all lower case.\n'
)


# ─────────────────────────── 数据类 ──────────────────────────

@dataclass
class PhaseResult:
    """单个测试阶段的结果。"""

    name: str           # "cold" / "warm" / "baseline"
    elapsed_sec: float  # 推理总时间（秒）
    output_tokens: int  # 所有请求的输出 token 总数
    num_requests: int   # 请求数量
    throughput: float   # output_tokens / elapsed_sec


# ─────────────────────────── 环境配置 ────────────────────────

def setup_lmcache_env() -> None:
    """设置 LMCache 所需的环境变量（local_disk 模式，不开启 local_cpu）。"""
    os.environ["LMCACHE_LOCAL_CPU"] = "False"
    os.environ["LMCACHE_LOCAL_DISK"] = f"file://{KV_DIR}"
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "100"   # GB
    os.environ["LMCACHE_CHUNK_SIZE"] = "128"
    # os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "True"
    # os.environ["LMCACHE_PRE_CACHING_HASH_ALGORITHM"] = "sha256_cbor"


# ─────────────────────────── 数据加载 ────────────────────────

def load_reviews(path: str, num_samples: int) -> List[str]:
    """从 CSV 读取前 num_samples 条 review 文本。"""
    df = pd.read_csv(path, nrows=num_samples)
    return df["review"].tolist()


def build_prompts(texts: List[str], task: str) -> List[str]:
    """将 Data 文本与 Task 拼接为完整 prompt。"""
    return [f"{text}\n\n{task}" for text in texts]


# ─────────────────────────── 推理工具 ────────────────────────

def _run_generate(
    llm,
    prompts: List[str],
    max_tokens: int,
    phase_name: str,
) -> PhaseResult:
    """执行一次批量推理并计时，返回 PhaseResult。"""
    from vllm import SamplingParams  # 延迟导入，避免污染进程级 env

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_output_tokens / elapsed if elapsed > 0 else 0.0

    return PhaseResult(
        name=phase_name,
        elapsed_sec=elapsed,
        output_tokens=total_output_tokens,
        num_requests=len(prompts),
        throughput=throughput,
    )


def _cleanup_llm(llm) -> None:
    """销毁 LLM 并释放 GPU 显存。"""
    del llm
    gc.collect()
    torch.cuda.empty_cache()

# ─────────────────────────── 基准测试主体 ────────────────────

def run_lmcache_bench(
    data_texts: List[str],
    task_prompts: List[str],
    gpu_util: float,
    max_tokens: int,
    max_model_len: Optional[int] = None,
) -> tuple[PhaseResult, PhaseResult]:
    """运行 LMCache Cold + Warm 两个阶段。

    Cold 和 Warm 使用独立的 LLM 实例：Cold 结束后彻底销毁引擎并清空 GPU
    显存，再重建 LLM 进行 Warm，保证 Warm 阶段的 KV Cache 完全来自 SSD，
    而非 GPU 上的残留。

    Returns:
        (cold_result, warm_result)
    """
    from vllm import LLM
    from vllm.config import KVTransferConfig

    def _build_lmcache_llm() -> LLM:
        ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
        return LLM(
            model=MODEL,
            kv_transfer_config=ktc,
            gpu_memory_utilization=gpu_util,
            enforce_eager=True,
            max_model_len=max_model_len,
            enable_prefix_caching=False,  # Disable vLLM's internal prefix cache
        )

    # ── Cold 阶段 ──────────────────────────────────────────────
    print("  [Cold] 创建 LLM...")
    llm_cold = _build_lmcache_llm()
    cold_result = _run_generate(llm_cold, data_texts, max_tokens=1, phase_name="cold")

    # ── Warm 阶段 ──────────────────────────────────────────────
    print("  [Warm] 从 SSD 加载 KV Cache...")
    warm_result = _run_generate(llm_cold, task_prompts, max_tokens=max_tokens, phase_name="warm")

    _cleanup_llm(llm_cold)

    return cold_result, warm_result


def run_baseline_bench(
    task_prompts: List[str],
    gpu_util: float,
    max_tokens: int,
) -> PhaseResult:
    """运行原生 vLLM 推理（不使用 LMCache）。

    Returns:
        baseline_result
    """
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True,
    )

    baseline_result = _run_generate(
        llm, task_prompts, max_tokens=max_tokens, phase_name="baseline"
    )

    _cleanup_llm(llm)

    return baseline_result


# ─────────────────────────── 报告输出 ────────────────────────

def print_report(
    cold: PhaseResult,
    warm: PhaseResult,
    baseline: PhaseResult,
    num_samples: int,
) -> None:
    """打印对比结果表格。"""
    sep = "=" * 60

    def row(r: PhaseResult, label: str) -> str:
        return (
            f"  {label:<22}  {r.elapsed_sec:>10.2f}s  "
            f"{r.output_tokens:>8}  {r.throughput:>12.1f}"
        )

    time_speedup = baseline.elapsed_sec / warm.elapsed_sec if warm.elapsed_sec > 0 else float("inf")
    tput_speedup = warm.throughput / baseline.throughput if baseline.throughput > 0 else float("inf")
    total_lmcache = cold.elapsed_sec + warm.elapsed_sec
    total_speedup = baseline.elapsed_sec / total_lmcache if total_lmcache > 0 else float("inf")

    print(f"\n{sep}")
    print("  LMCache Benchmark 结果")
    print(sep)
    print(f"  数据规模 : {num_samples} 条 IMDB review")
    print(f"  模型     : {Path(MODEL).name}")
    print()
    print(f"  {'阶段':<22}  {'推理总时间':>10}  {'输出Token':>8}  {'吞吐(tok/s)':>12}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*12}")
    print(row(cold, "LMCache Cold 阶段"))
    print(row(warm, "LMCache Warm 阶段"))
    print(row(baseline, "Baseline（原生推理）"))
    print()
    print("  Warm vs Baseline 加速比：")
    print(f"    总时间加速比 : {time_speedup:.2f}x")
    print(f"    吞吐加速比   : {tput_speedup:.2f}x")
    print()
    print(f"  LMCache 总耗时（Cold+Warm）: {total_lmcache:.2f}s")
    print(f"  Baseline 总耗时             : {baseline.elapsed_sec:.2f}s")
    print(f"  总体加速比                   : {total_speedup:.2f}x")
    print(sep)


def save_json(
    cold: PhaseResult,
    warm: PhaseResult,
    baseline: PhaseResult,
    output_path: str,
) -> None:
    """将结果序列化为 JSON 文件。"""
    data = {
        "cold": asdict(cold),
        "warm": asdict(warm),
        "baseline": asdict(baseline),
        "speedup": {
            "warm_vs_baseline_time": (
                baseline.elapsed_sec / warm.elapsed_sec if warm.elapsed_sec > 0 else None
            ),
            "warm_vs_baseline_throughput": (
                warm.throughput / baseline.throughput if baseline.throughput > 0 else None
            ),
            "total_lmcache_vs_baseline_time": (
                baseline.elapsed_sec / (cold.elapsed_sec + warm.elapsed_sec)
                if (cold.elapsed_sec + warm.elapsed_sec) > 0
                else None
            ),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入 {output_path}")


# ─────────────────────────── 入口 ────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LMCache vs 原生 vLLM 性能对比基准测试"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20, help="取前 N 条 IMDB review（默认 20）"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="使用的 GPU 编号，对应 CUDA_VISIBLE_DEVICES（默认 0）"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1, help="Warm/Baseline 最大生成 token 数（默认 1）"
    )
    parser.add_argument(
        "--gpu-util", type=float, default=0.85, help="vLLM 显存利用率（默认 0.85）"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None, help="可选：覆盖模型最大序列长度（适用于显存受限场景）"
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="可选：将结果写入 JSON 文件"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 必须在 import vllm 之前设置，保证环境变量对子进程可见
    setup_lmcache_env()

    # 清空 KV 缓存目录，确保本次测试不使用上次残留数据
    kv_dir = Path(KV_DIR)
    if kv_dir.exists():
        shutil.rmtree(kv_dir)
        print(f"已清空 KV 缓存目录：{KV_DIR}")
    kv_dir.mkdir(parents=True, exist_ok=True)

    texts = load_reviews(DATA_PATH, args.num_samples)
    task_prompts = build_prompts(texts, TASK)
    print(f"已加载 {len(texts)} 条 review，开始基准测试...")

    print("\n[1/2] 运行 LMCache Cold + Warm 阶段...")
    cold_result, warm_result = run_lmcache_bench(
        data_texts=texts,
        task_prompts=task_prompts,
        gpu_util=args.gpu_util,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
    )

    print("\n[2/2] 运行 Baseline（原生推理）...")
    baseline_result = run_baseline_bench(
        task_prompts=task_prompts,
        gpu_util=args.gpu_util,
        max_tokens=args.max_tokens,
    )

    print_report(cold_result, warm_result, baseline_result, len(texts))

    if args.output_json:
        save_json(cold_result, warm_result, baseline_result, args.output_json)


if __name__ == "__main__":
    main()
