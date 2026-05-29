# SPDX-License-Identifier: Apache-2.0
"""Longbench dataset benchmark: DaseR vs LMCache.

Iterates over Longbench JSONL datasets, running cold/warm inference passes
with each KV connector. ``max_model_len`` is auto-calculated from available
GPU VRAM so every GPU gets the longest context it can fit.

Usage:
    python benchmarks/bench_longbench.py \\
        --model /path/to/model \\
        --store-dir /path/to/benchmark-scratch \\
        --longbench-dir /data/ld/longbench_data/data \\
        [--datasets multi_news] \\
        [--num-prompts 10] \\
        [--out results.json]
"""

# ruff: noqa: E402

# Future
from __future__ import annotations

# Standard
import argparse
import gc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import uuid

# ---------------------------------------------------------------------------
# Deterministic hashing — same pattern as bench_e2e_daser_vs_lmcache.py.
# ---------------------------------------------------------------------------
BENCHMARK_SEED_ENV = "42"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != BENCHMARK_SEED_ENV:
    os.environ["PYTHONHASHSEED"] = BENCHMARK_SEED_ENV
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

# Early GPU selection — must happen before torch/vLLM imports.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu-id", default="auto")
_gpu_args, _ = _gpu_parser.parse_known_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.utils import (
    BYTES_PER_GIB,
    COMPARISON_GDS,
    COMPARISON_IOURING_MEM,
    apply_gpu_selection,
    calculate_max_model_len,
    derive_benchmark_sizing,
    derive_capacity_limits,
    load_longbench_prompts,
    set_global_seed,
    tokenise_and_truncate,
)

SELECTED_GPU_ID = (
    apply_gpu_selection(_gpu_args.gpu_id)
    if __name__ == "__main__"
    else os.environ.get("CUDA_VISIBLE_DEVICES")
)

# vLLM V1 forks EngineCore subprocesses — CUDA requires 'spawn' on Linux.
if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

# Third Party
import torch

from benchmarks.bench_e2e_daser_vs_lmcache import (
    BENCHMARK_SEED,
    BLOCK_TOKENS,
    DaserHarness,
    LMCacheHarness,
    MAX_MODEL_LEN,
    SLOT_SIZE,
    build_summary,
    print_report,
    run_daser_correctness,
    run_lmcache_correctness,
    run_system,
)

from daser.logging import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GPU_MEM_UTIL_DEFAULT: float = 0.9
MAX_NUM_SEQS_DEFAULT: int = 64
COMPARISON_MODE_DEFAULT: str = COMPARISON_IOURING_MEM
DEFAULT_DATASET: str = "multi_news"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_datasets(
    longbench_dir: str, dataset_filter: str | None
) -> dict[str, str]:
    """Return ``{name: path}`` for matching Longbench JSONL files.

    Args:
        longbench_dir: Path to the Longbench data directory.
        dataset_filter: Comma-separated dataset names (without ``.jsonl``),
            or ``"all"`` for every ``*.jsonl`` in the directory.

    Returns:
        Mapping of dataset name to absolute JSONL path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If a requested dataset name is not found.
    """
    root = Path(longbench_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Longbench dir not found: {longbench_dir}")

    available: dict[str, str] = {}
    for fp in sorted(root.glob("*.jsonl")):
        available[fp.stem] = str(fp)

    if dataset_filter is None or dataset_filter == "all":
        return available

    requested = {d.strip() for d in dataset_filter.split(",") if d.strip()}
    missing = requested - set(available.keys())
    if missing:
        raise ValueError(
            f"Datasets not found in {longbench_dir}: {', '.join(sorted(missing))}"
        )
    return {k: available[k] for k in requested if k in available}


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------


def print_aggregate_report(
    all_results: dict[str, dict[str, Any]],
) -> None:
    """Print a summary table across all completed datasets.

    Args:
        all_results: ``{dataset_name: result_dict}`` as accumulated by the
            per-dataset loop.
    """
    if not all_results:
        return

    print("\n" + "=" * 90)
    print("LONGBENCH AGGREGATE — DaseR vs LMCache")
    print("=" * 90)
    header = (
        f"{'Dataset':<26} {'D Cold t/s':>11} {'L Cold t/s':>11} "
        f"{'D Warm t/s':>11} {'L Warm t/s':>11} {'Ratio':>7} {'Parity':>7}"
    )
    print(header)
    print("-" * 90)

    parity_ok = 0
    parity_total = 0
    ratios: list[float] = []

    for name in sorted(all_results):
        r = all_results[name]
        s = r.get("summary", {})

        def _tps(system: str, metric: str) -> str:
            system_dict = s.get(system) or {}
            val = system_dict.get(metric)
            if val is None:
                return "       N/A"
            return f"{val:>11,.0f}"

        ratio = s.get("warm_tps_ratio_daser_over_lmcache")
        ratio_str = f"{ratio:.2f}" if ratio is not None else "   N/A"
        if ratio is not None and ratio > 0:
            ratios.append(ratio)

        parity = s.get("correctness_parity_ok")
        parity_str = "OK" if parity is True else ("FAIL" if parity is False else "N/A")
        if parity is not None:
            parity_total += 1
            if parity:
                parity_ok += 1

        print(
            f"{name:<26} {_tps('daser', 'cold_tok_per_s')} "
            f"{_tps('lmcache', 'cold_tok_per_s')} {_tps('daser', 'warm_tok_per_s')} "
            f"{_tps('lmcache', 'warm_tok_per_s')} {ratio_str:>7} {parity_str:>7}"
        )

    print("-" * 90)
    if ratios:
        geom = _geometric_mean(ratios)
        print(f"Overall warm tps ratio (geometric mean): {geom:.2f}x")
    if parity_total > 0:
        print(f"Overall correctness parity: {parity_ok}/{parity_total} OK")
    print("=" * 90)


def _geometric_mean(values: list[float]) -> float:
    """Compute geometric mean of positive floats.

    Args:
        values: Non-empty list of positive numbers.

    Returns:
        Geometric mean as a float.
    """
    import math

    return math.exp(sum(math.log(v) for v in values) / len(values))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — argparse + per-dataset orchestration
    """Entry point."""
    set_global_seed(BENCHMARK_SEED)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/data/zwt/model/models/Qwen/Qwen3-8B",
    )
    _default_user = os.environ.get("USER", "ld")
    parser.add_argument(
        "--store-dir",
        default=f"/data/{_default_user}/daser_test",
    )
    parser.add_argument(
        "--longbench-dir",
        default=f"/data/{_default_user}/longbench_data/data",
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASET)
    parser.add_argument("--num-prompts", type=int, default=0)
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=GPU_MEM_UTIL_DEFAULT,
    )
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=MAX_NUM_SEQS_DEFAULT,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="Override auto-calculated max_model_len (0 = auto).",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help="Per-prompt token ceiling (0 = max_model_len - 256).",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=(COMPARISON_GDS, COMPARISON_IOURING_MEM),
        default=COMPARISON_MODE_DEFAULT,
    )
    parser.add_argument("--skip-daser", action="store_true")
    parser.add_argument("--skip-lmcache", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--evict", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")

    # ---- discover datasets ----
    datasets = discover_datasets(args.longbench_dir, args.datasets)
    logger.info("selected %d dataset(s): %s", len(datasets), ", ".join(datasets))

    # ---- clamp max_model_len to model capability (must happen first) ----
    from transformers import AutoConfig  # Third Party

    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_max = getattr(model_config, "max_position_embeddings", None)

    # ---- VRAM-based max_model_len ----
    if args.max_model_len > 0:
        max_model_len = args.max_model_len
        logger.info("max_model_len override: %d", max_model_len)
    else:
        max_model_len = calculate_max_model_len(
            gpu_id=SELECTED_GPU_ID if SELECTED_GPU_ID is not None else None,
            gpu_memory_utilization=args.gpu_util,
            block_tokens=BLOCK_TOKENS,
        )
        logger.info("auto max_model_len from VRAM: %d", max_model_len)

    if model_max is not None and max_model_len > model_max:
        logger.info(
            "clamping max_model_len %d -> %d (model max_position_embeddings)",
            max_model_len,
            model_max,
        )
        max_model_len = model_max

    max_input_tokens = (
        args.max_input_tokens
        if args.max_input_tokens > 0
        else max(0, max_model_len - 256)
    )

    transfer_mode = (
        "iouring" if args.comparison_mode == COMPARISON_IOURING_MEM else "gds"
    )

    # ---- load tokenizer ----
    from transformers import AutoTokenizer  # Third Party

    logger.info("loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- store root ----
    store_root = os.path.join(args.store_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(store_root, exist_ok=False)
    logger.info("benchmark scratch root: %s", store_root)
    logger.info(
        "selected GPU: %s (CUDA_VISIBLE_DEVICES=%s)",
        SELECTED_GPU_ID if SELECTED_GPU_ID is not None else "current",
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    all_results: dict[str, dict[str, Any]] = {}

    # ---- per-dataset loop ----
    for dataset_name, dataset_path in datasets.items():
        logger.info("=== Dataset: %s ===", dataset_name)

        # -- load & tokenise --
        raw_prompts = load_longbench_prompts(dataset_path, args.num_prompts)
        if not raw_prompts:
            logger.warning("no prompts loaded for %s, skipping", dataset_name)
            continue

        prompts = tokenise_and_truncate(
            raw_prompts, tokenizer, max_input_tokens, BLOCK_TOKENS
        )
        max_num_seqs = args.max_num_seqs
        token_counts = [len(ids) for ids in prompts]
        prompt_tokens_total = sum(token_counts)
        total_blocks = sum(c // BLOCK_TOKENS for c in token_counts)
        max_prompt_blocks = max(
            (c // BLOCK_TOKENS for c in token_counts), default=1
        )
        logger.info(
            "%s: %d prompts, %d tokens, %d blocks "
            "(avg %.1f, max %d blocks/prompt)",
            dataset_name,
            len(prompts),
            prompt_tokens_total,
            total_blocks,
            total_blocks / max(1, len(prompts)),
            max_prompt_blocks,
        )

        # -- sizing --
        total_bytes = total_blocks * SLOT_SIZE
        capacity_limits = derive_capacity_limits(store_root, SELECTED_GPU_ID)
        sizing = derive_benchmark_sizing(
            total_blocks=total_blocks,
            max_prompt_blocks=max_prompt_blocks,
            slot_size=SLOT_SIZE,
            mode=args.comparison_mode,
            evict=args.evict,
            capacity_limits=capacity_limits,
        )
        logger.info(
            "sizing: workload=%.2f GiB, daser_l2=%d slots, daser_l1=%.2f GiB, "
            "evict=%s, capped=%s",
            total_bytes / BYTES_PER_GIB,
            sizing.daser_slots,
            sizing.daser_l1_bytes / BYTES_PER_GIB,
            args.evict,
            sizing.capacity_capped,
        )

        config = {
            "dataset": dataset_name,
            "comparison_mode": args.comparison_mode,
            "evict": args.evict,
            "num_prompts": len(prompts),
            "seed": BENCHMARK_SEED,
            "model": args.model,
            "block_tokens": BLOCK_TOKENS,
            "slot_bytes": SLOT_SIZE,
            "max_model_len": max_model_len,
            "max_input_tokens": max_input_tokens,
            "max_num_seqs": max_num_seqs,
            "total_blocks": total_blocks,
            "max_prompt_blocks": max_prompt_blocks,
            "total_bytes": total_bytes,
            "daser_transfer_mode": transfer_mode,
            "daser_slots": sizing.daser_slots,
            "daser_l2_bytes": sizing.daser_l2_bytes,
            "daser_l1_bytes": sizing.daser_l1_bytes,
            "lmcache_disk_gb": sizing.lmcache_disk_gb,
            "lmcache_cpu_gb": sizing.lmcache_cpu_gb,
            "selected_gpu_id": SELECTED_GPU_ID,
            "gpu_util": args.gpu_util,
            "capacity_limits": {
                "max_l1_bytes": capacity_limits.max_l1_bytes,
                "max_l2_bytes": capacity_limits.max_l2_bytes,
                "memory_available_bytes": capacity_limits.memory_available_bytes,
                "disk_available_bytes": capacity_limits.disk_available_bytes,
                "capacity_capped": sizing.capacity_capped,
            },
            "daser_warm_skip_save": True,
            "correctness_metric": "exact_generated_token_ids_and_text",
        }
        correctness_prompts = prompts

        # -- LMCache run --
        lmcache_result: dict[str, Any] | None = None
        if args.skip_lmcache:
            lmcache_result = {"skipped": True, "reason": "--skip-lmcache"}
        else:
            try:
                import lmcache  # noqa: F401
            except ImportError as exc:
                lmcache_result = {"skipped": True, "reason": f"import failed: {exc}"}
            if lmcache_result is None:
                lmcache_dir = tempfile.mkdtemp(
                    prefix="lmcache_bench_", dir=store_root
                )
                h_lm = LMCacheHarness(
                    lmcache_dir,
                    total_bytes,
                    args.model,
                    args.gpu_util,
                    max_num_seqs,
                    args.comparison_mode == COMPARISON_IOURING_MEM,
                    sizing.lmcache_disk_gb,
                    sizing.lmcache_cpu_gb,
                    max_model_len=max_model_len,
                )
                try:
                    h_lm.start()
                    r = run_system(
                        "LMCache",
                        h_lm.build_llm,
                        prompts,
                        after_cold_fn=h_lm.wait_for_disk_quiescence,
                    )
                    r["backend"] = "lmcache"
                    r["storage_tier"] = (
                        "local-ssd-mem"
                        if args.comparison_mode == COMPARISON_IOURING_MEM
                        else "local-ssd"
                    )
                    r["warm_skip_save"] = False
                    r["disk_limit_gb"] = sizing.lmcache_disk_gb
                    r["cpu_limit_gb"] = sizing.lmcache_cpu_gb
                    lmcache_result = r
                finally:
                    h_lm.stop()
                if lmcache_result is not None and not args.skip_correctness:
                    lmcache_result["correctness"] = run_lmcache_correctness(
                        store_root,
                        total_bytes,
                        args.model,
                        args.gpu_util,
                        max_num_seqs,
                        args.comparison_mode == COMPARISON_IOURING_MEM,
                        sizing.lmcache_disk_gb,
                        sizing.lmcache_cpu_gb,
                        correctness_prompts,
                        max_model_len=max_model_len,
                    )

        # -- DaseR run --
        daser_result: dict[str, Any] | None = None
        if args.skip_daser:
            daser_result = {"skipped": True, "reason": "--skip-daser"}
        else:
            daser_dir = tempfile.mkdtemp(prefix="daser_bench_", dir=store_root)
            socket_dir = tempfile.mkdtemp(prefix="daser_bench_ipc_")
            h = DaserHarness(
                daser_dir,
                socket_dir,
                sizing.daser_slots,
                args.model,
                args.gpu_util,
                max_num_seqs,
                transfer_mode,
                sizing.daser_l1_bytes,
                max_model_len=max_model_len,
            )
            try:
                h.start()
                r = run_system(
                    "DaseR",
                    h.build_llm,
                    prompts,
                    warm_skip_save=True,
                    after_cold_fn=lambda: h.wait_until_committed(
                        prompts,
                        BLOCK_TOKENS,
                        require_all_commits=not args.evict,
                        require_l2_drain=(
                            args.evict
                            or args.comparison_mode == COMPARISON_IOURING_MEM
                        ),
                        timeout_s=120.0,
                    ),
                )
                r["backend"] = transfer_mode
                r["storage_tier"] = (
                    "local-ssd-mem"
                    if args.comparison_mode == COMPARISON_IOURING_MEM
                    else "local-ssd"
                )
                r["warm_skip_save"] = True
                r["l2_bytes"] = sizing.daser_l2_bytes
                r["l1_bytes"] = sizing.daser_l1_bytes
                daser_result = r
            finally:
                h.stop()
            if daser_result is not None and not args.skip_correctness:
                correctness_require_l2_drain = (
                    args.evict
                    or args.comparison_mode == COMPARISON_IOURING_MEM
                )
                daser_result["correctness"] = run_daser_correctness(
                    store_root,
                    args.model,
                    args.gpu_util,
                    max_num_seqs,
                    transfer_mode,
                    sizing.daser_l1_bytes,
                    sizing.daser_slots,
                    correctness_prompts,
                    require_all_commits=not args.evict,
                    require_l2_drain=correctness_require_l2_drain,
                    max_model_len=max_model_len,
                )
                daser_result["visible_prompt_count"] = int(
                    daser_result["correctness"].get("visible_total", 0)
                )

        # -- per-dataset summary --
        summary = build_summary(
            daser_result,
            lmcache_result,
            prompt_tokens_total,
            args.comparison_mode,
        )
        print(f"\n{'=' * 72}")
        print(f"Longbench Dataset: {dataset_name} ({len(prompts)} prompts)")
        print_report(config, summary)

        all_results[dataset_name] = {
            "config": config,
            "summary": summary,
            "daser": daser_result,
            "lmcache": lmcache_result,
        }

        # Clean up between datasets.
        gc.collect()
        torch.cuda.empty_cache()

    # ---- aggregate report ----
    print_aggregate_report(all_results)

    if args.out:
        Path(args.out).write_text(
            json.dumps(all_results, indent=2, default=str)
        )
        print(f"\nJSON results written to {args.out}")


if __name__ == "__main__":
    main()
