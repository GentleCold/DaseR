# SPDX-License-Identifier: Apache-2.0
"""End-to-end inference benchmark: DaseR transfer modes vs LMCache.

Runs the same IMDB-review prompt batch through vLLM twice, once with each
KV connector, measuring cold-pass and warm-pass elapsed time and prompt-token
throughput. Prefix cache is disabled so the NVMe storage tier is the only
source of cross-run speedup.

Usage:
    python benchmarks/bench_imdb.py \\
        --model /path/to/model \\
        --store-dir /path/to/benchmark-scratch \\
        --imdb /path/to/imdb.csv \\
        [--num-prompts 200] \\
        [--out results.json]

    All three of --model, --store-dir, and --imdb are required.
"""

# ruff: noqa: E402

# Future
from __future__ import annotations

# Standard
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import uuid

# ---------------------------------------------------------------------------
# Deterministic hashing — re-exec with PYTHONHASHSEED set so both LMCache
# scheduler-side token hashing and vLLM's NONE_HASH seed are stable across
# cold/warm LLM rebuilds. Must happen before *any* import that touches
# Python string hashing or vLLM internals.
# ---------------------------------------------------------------------------
BENCHMARK_SEED_ENV = "42"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != BENCHMARK_SEED_ENV:
    os.environ["PYTHONHASHSEED"] = BENCHMARK_SEED_ENV
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

# Select the benchmark GPU before importing torch or vLLM. The regular
# argparse parser is built later; this minimal parser intentionally ignores
# all other options.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu-id", default="auto")
_gpu_args, _ = _gpu_parser.parse_known_args()

# First Party — add project root for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.bench_common import (  # noqa: E402
    BENCHMARK_SEED,
    BLOCK_TOKENS,
    BYTES_PER_GIB,
    COMPARISON_GDS,
    COMPARISON_IOURING_MEM,
    SLOT_SIZE,
    DaserHarness,
    LMCacheHarness,
    apply_gpu_selection,
    build_summary,
    derive_benchmark_sizing,
    derive_capacity_limits,
    load_prompts,
    print_report,
    run_daser_correctness,
    run_lmcache_correctness,
    run_system,
    set_global_seed,
    tokenise_and_truncate,
    wait_gpu_memory,
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
from daser.logging import init_logger  # noqa: E402

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_INPUT_TOKENS_DEFAULT: int = 1792
GPU_MEM_UTIL_DEFAULT: float = 0.9
MAX_NUM_SEQS_DEFAULT: int = 64


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — argparse + orchestration
    """Entry point."""
    set_global_seed(BENCHMARK_SEED)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--model", required=True)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--imdb", required=True)
    parser.add_argument(
        "--max-input-tokens", type=int, default=MAX_INPUT_TOKENS_DEFAULT
    )
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=GPU_MEM_UTIL_DEFAULT,
        help="vLLM gpu_memory_utilization (default: 0.9)",
    )
    parser.add_argument(
        "--gpu-id",
        default="auto",
        help=(
            "GPU ID to expose through CUDA_VISIBLE_DEVICES. Use 'auto' to pick "
            "the GPU with most free memory, or 'current' to keep the current env."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=MAX_NUM_SEQS_DEFAULT,
        help="vLLM max_num_seqs (default: 64).",
    )
    parser.add_argument("--skip-daser", action="store_true")
    parser.add_argument("--skip-lmcache", action="store_true")
    parser.add_argument(
        "--comparison-mode",
        choices=(COMPARISON_GDS, COMPARISON_IOURING_MEM),
        default=COMPARISON_GDS,
    )
    parser.add_argument(
        "--evict",
        action="store_true",
        help="Choose DaseR L2/L1 sizes that force eviction during the workload.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")
    store_root = os.path.join(args.store_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(store_root, exist_ok=False)
    logger.info("benchmark scratch root: %s", store_root)
    logger.info(
        "selected GPU: %s (CUDA_VISIBLE_DEVICES=%s)",
        SELECTED_GPU_ID if SELECTED_GPU_ID is not None else "current",
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    # ---- tokenise prompts ----
    logger.info("loading prompts from %s", args.imdb)
    raw_prompts = load_prompts(args.imdb, args.num_prompts)
    if len(raw_prompts) < args.num_prompts:
        logger.warning(
            "got %d prompts, requested %d — continuing with what we have",
            len(raw_prompts),
            args.num_prompts,
        )

    from transformers import AutoTokenizer  # Third Party

    logger.info("loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = tokenise_and_truncate(
        raw_prompts, tokenizer, args.max_input_tokens, BLOCK_TOKENS
    )
    max_num_seqs = args.max_num_seqs
    token_counts = [len(ids) for ids in prompts]
    prompt_tokens_total = sum(token_counts)
    total_blocks = sum(c // BLOCK_TOKENS for c in token_counts)
    max_prompt_blocks = max((c // BLOCK_TOKENS for c in token_counts), default=1)
    logger.info(
        "tokenised %d prompts, %d tokens, %d blocks (avg %.1f, max %d blocks/prompt)",
        len(prompts),
        prompt_tokens_total,
        total_blocks,
        total_blocks / max(1, len(prompts)),
        max_prompt_blocks,
    )

    # ---- sizes ----
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
    transfer_mode = (
        "iouring" if args.comparison_mode == COMPARISON_IOURING_MEM else "gds"
    )
    logger.info(
        "cache sizing: workload=%.2fGiB, daser_l2_slots=%d, "
        "daser_l1=%.2fGiB, lmcache_cpu=%.2fGiB, evict=%s, capped=%s, "
        "max_l1=%.2fGiB, max_l2=%.2fGiB",
        total_bytes / BYTES_PER_GIB,
        sizing.daser_slots,
        sizing.daser_l1_bytes / BYTES_PER_GIB,
        sizing.lmcache_cpu_gb,
        args.evict,
        sizing.capacity_capped,
        capacity_limits.max_l1_bytes / BYTES_PER_GIB,
        capacity_limits.max_l2_bytes / BYTES_PER_GIB,
    )

    config = {
        "comparison_mode": args.comparison_mode,
        "evict": args.evict,
        "num_prompts": len(prompts),
        "correctness_num_prompts": len(prompts),
        "seed": BENCHMARK_SEED,
        "model": args.model,
        "block_tokens": BLOCK_TOKENS,
        "slot_bytes": SLOT_SIZE,
        "max_input_tokens": args.max_input_tokens,
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

    # ---- LMCache run ----
    # Run LMCache before DaseR. The DaseR server opens CUDA IPC buffers in the
    # benchmark parent process, and forking another vLLM EngineCore after that
    # can fail CUDA initialization.
    lmcache_result: dict[str, Any] | None = None
    if args.skip_lmcache:
        lmcache_result = {"skipped": True, "reason": "--skip-lmcache"}
    else:
        try:
            import lmcache  # noqa: F401 — import probe
        except ImportError as exc:
            lmcache_result = {"skipped": True, "reason": f"import failed: {exc}"}
        if lmcache_result is None:
            lmcache_dir = tempfile.mkdtemp(prefix="lmcache_bench_", dir=store_root)
            h_lm = LMCacheHarness(
                lmcache_dir,
                total_bytes,
                args.model,
                args.gpu_util,
                max_num_seqs,
                args.comparison_mode == COMPARISON_IOURING_MEM,
                sizing.lmcache_disk_gb,
                sizing.lmcache_cpu_gb,
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
            if lmcache_result is not None:
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
                )

    # ---- DaseR run ----
    daser_result: dict[str, Any] | None = None
    if args.skip_daser:
        daser_result = {"skipped": True, "reason": "--skip-daser"}
    else:
        wait_gpu_memory(args.gpu_util)
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
                        args.evict or args.comparison_mode == COMPARISON_IOURING_MEM
                    ),
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
        if daser_result is not None:
            correctness_require_l2_drain = (
                args.evict or args.comparison_mode == COMPARISON_IOURING_MEM
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
            )
            daser_result["visible_prompt_count"] = int(
                daser_result["correctness"].get("visible_total", 0)
            )

    # ---- report ----
    summary = build_summary(
        daser_result,
        lmcache_result,
        prompt_tokens_total,
        args.comparison_mode,
    )
    print_report(config, summary)

    if args.out:
        out_obj = {
            "config": config,
            "summary": summary,
            "daser": daser_result,
            "lmcache": lmcache_result,
        }
        Path(args.out).write_text(json.dumps(out_obj, indent=2))
        print(f"\nJSON results written to {args.out}")


if __name__ == "__main__":
    main()
