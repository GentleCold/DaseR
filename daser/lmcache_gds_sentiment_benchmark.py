from __future__ import annotations

"""LMCache(GDS) KV Connector benchmark for IMDb sentiment classification.

This script demonstrates the requested two workflows integrated in DaseR:
  1) KV generation workflow: run text-only prompts to populate LMCache on disk.
  2) Online inference workflow: run text+task prompts; KV is prefetched to GPU
     via vLLM's LMCacheConnectorV1.

It also compares token throughput between:
  - Baseline (no kv connector)
  - LMCache(GDS) enabled

Notes:
  - Parameters are global constants (no argparse).
  - We run each benchmark in a separate process to avoid GPU memory leakage
    between vLLM engine instances.
"""

import csv
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from daser.config import CudaConfig
from daser.env import configure_cuda
from daser.kvcache.lmcache_gds_manager import LMCacheGDSConfig, LMCacheGDSKVCacheManager
from daser.kvcache.manager import SSDKVCacheManager


# ==========================
# Global constants (tweak me)
# ==========================

CUDA = CudaConfig(cuda_visible_devices="1")

MODEL = "/data/zwt/model/models/Qwen/Qwen3-8B"

DATASET_PATH = "/data/zwt/imdb.csv"
DATASET_TEXT_COL = "review"
LIMIT = 1000
MAX_TOKENS = 1
GPU_MEMORY_UTILIZATION = 0.7

# Warmup batching
# LMCache stores KV via a pinned-CPU staging buffer before writing to disk.
# Warming up with too many prompts at once can schedule many concurrent stores
# and exhaust the staging pool, causing repeated "local cpu memory under
# pressure" warnings. Keep this small for stability.
WARMUP_BATCH_SIZE = 32

# Profiling switch
# - True: run ONLY a profiled generate (and report metrics for that run)
# - False: run ONLY an unprofiled generate (and report metrics for that run)
ENABLE_PROFILING = False

# LMCache(GDS) storage
LMCACHE_DIR = "/data/zwt/lmcache_gds_imdb"
LMCACHE_CONFIG_FILE = f"{LMCACHE_DIR}/lmcache.yaml"
# IMPORTANT for prefix reuse:
# LMCache matches at chunk boundaries. With chunk_size=256 many reviews are
# shorter than one full chunk, so warmup(text-only) -> inference(text+task)
# often yields 0 hit tokens. Using a smaller chunk_size enables partial prefix
# reuse in multiples of chunk_size.
LMCACHE_CHUNK_SIZE = 64
# LMCache expects `cufile_buffer_size` in MiB (it multiplies by 1024**2).
LMCACHE_CUFILE_BUFFER_SIZE_MIB = 256

# SSD offloading (Scheme B) storage
SSD_CACHE_DIR = "/data/zwt/vllm_kv_ssd_imdb_bench"
SSD_MAX_BLOCKS = 200_000

# Torch profiler outputs (Chrome trace JSON)
# NOTE: vLLM's torch profiler writes traces to a directory (not a single JSON).
# Using vLLM's built-in profiler is required to capture worker-side CUDA work.
PROFILE_ROOT_DIR = "/home/zwt/DaseR/daser/profile/kv_prefetch_bench"
PROFILE_PROMPT_LIMIT = 200
PROFILE_FLUSH_SLEEP_S = 10

# Sentiment instruction
SENTIMENT_INSTRUCTION = (
    '\nGiven the above film review, answer whether the sentiment is "positive" '
    'or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
)


def _load_texts_csv(path: str, text_col: str, limit: int) -> list[str]:
    texts: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(texts) >= limit:
                break
            t = (row.get(text_col) or "").strip()
            if t:
                texts.append(t)
    return texts


def _build_task_prompts(texts: list[str]) -> list[str]:
    return [f"{t}{SENTIMENT_INSTRUCTION}" for t in texts]


def _count_tokens(tokenizer: Any, texts: list[str]) -> int:
    # Tokenizer API differs slightly across versions; try encode first.
    n = 0
    for t in texts:
        try:
            n += len(tokenizer.encode(t))
        except Exception:
            n += len(tokenizer(t).input_ids)
    return n


def _reset_dir(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _vllm_profiler_config(mode: str) -> dict[str, Any]:
    # vLLM requires torch_profiler_dir to be an absolute path.
    out_dir = Path(PROFILE_ROOT_DIR).expanduser().resolve() / mode
    _reset_dir(str(out_dir))
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(out_dir),
        # Do not collect stack traces.
        "torch_profiler_with_stack": False,
        # Make traces easier to inspect (avoid tiny *.json.gz files).
        "torch_profiler_use_gzip": False,
        # Optional: enable these if you want richer traces (more overhead).
        # "torch_profiler_record_shapes": True,
        # "torch_profiler_with_memory": True,
    }


def _run_generate(
    llm,
    prompts: list[str],
    *,
    enable_profiling: bool,
    sampling_extra_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_args=sampling_extra_args,
    )
    tokenizer = llm.get_tokenizer()

    prompt_tokens = _count_tokens(tokenizer, prompts)

    # IMPORTANT:
    # - vLLM runs most CUDA work in worker processes.
    # - torch.profiler in this (driver) process won't see those kernels.
    # - Use vLLM's built-in profiler hooks instead.
    if enable_profiling and hasattr(llm, "start_profile"):
        llm.start_profile()

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    dt = time.perf_counter() - t0

    if enable_profiling and hasattr(llm, "stop_profile"):
        llm.stop_profile()
        # Give background processes time to flush traces.
        time.sleep(PROFILE_FLUSH_SLEEP_S)

    gen_texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
    gen_tokens = _count_tokens(tokenizer, gen_texts)

    total_tokens = prompt_tokens + gen_tokens
    tok_s = total_tokens / dt if dt > 0 else float("inf")

    return {
        "requests": len(prompts),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "total_tokens": total_tokens,
        "elapsed_s": dt,
        "tok_per_s": tok_s,
    }


def _yield_batches(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _warmup_prefixes_batched(llm, texts: list[str], *, batch_size: int) -> None:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    for batch in _yield_batches(texts, batch_size):
        if not batch:
            continue
        _ = llm.generate(batch, params)


def _bench_baseline(conn):
    # Import vLLM inside the child process.
    from vllm import LLM

    texts = _load_texts_csv(DATASET_PATH, DATASET_TEXT_COL, LIMIT)
    prompts = _build_task_prompts(texts)
    run_prompts = prompts[:PROFILE_PROMPT_LIMIT] if ENABLE_PROFILING else prompts

    llm_kwargs: dict[str, Any] = {}
    if ENABLE_PROFILING:
        llm_kwargs["profiler_config"] = _vllm_profiler_config("baseline")

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        **llm_kwargs,
    )

    metrics = _run_generate(llm, run_prompts, enable_profiling=ENABLE_PROFILING)

    result: dict[str, Any] = {"mode": "baseline", "metrics": metrics}
    if ENABLE_PROFILING:
        result["profile"] = {
            "dir": str(Path(PROFILE_ROOT_DIR).expanduser().resolve() / "baseline"),
            "requests": int(min(PROFILE_PROMPT_LIMIT, len(prompts))),
        }
    conn.send(result)


def _bench_lmcache_gds(conn):
    texts = _load_texts_csv(DATASET_PATH, DATASET_TEXT_COL, LIMIT)
    prompts = _build_task_prompts(texts)
    run_prompts = prompts[:PROFILE_PROMPT_LIMIT] if ENABLE_PROFILING else prompts

    # Reset LMCache directory to ensure a clean warmup.
    cache_dir = Path(LMCACHE_DIR)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    lmcache_cfg = LMCacheGDSConfig(
        gds_path=str(cache_dir),
        chunk_size=LMCACHE_CHUNK_SIZE,
        cufile_buffer_size=LMCACHE_CUFILE_BUFFER_SIZE_MIB,
        # Keep CPU hot cache OFF; we want disk persistence, not keeping KV in RAM.
        local_cpu=False,
        # Pinned CPU staging buffer size (GB). If too small, LMCache may drop
        # stores ("store only 0 total chunks") and you'll see no disk files.
        max_local_cpu_size=8.0,
        extra_config={
            # Let LMCache decide; override here if you need to force:
            # "use_cufile": True,
            # Critical for warmup->inference correctness: make disk puts
            # synchronous so lookup can observe keys immediately.
            "sync_put": True,
            # Critical for warmup completeness: do not drop stores when the
            # pinned CPU staging pool is temporarily exhausted.
            # "force_store_wait": True,
        },
    )

    mgr = LMCacheGDSKVCacheManager(
        model=MODEL,
        lmcache_cfg=lmcache_cfg,
        lmcache_cfg_file=LMCACHE_CONFIG_FILE,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        kv_role="kv_both",
        kv_connector_extra_config={},
        **({"profiler_config": _vllm_profiler_config("lmcache_gds")} if ENABLE_PROFILING else {}),
    )

    # Workflow 1: KV generation (warmup text-only prefixes)
    warm_t0 = time.perf_counter()
    _warmup_prefixes_batched(mgr.llm, texts, batch_size=WARMUP_BATCH_SIZE)
    warm_dt = time.perf_counter() - warm_t0

    # IMPORTANT: Keep the same engine instance for inference.
    # LMCache LocalDisk backend maintains an in-memory index and does not
    # rebuild it from disk on startup, so recreating the engine can lead to
    # 0-hit lookups even if files exist.

    # Debug: estimate expected hit tokens at chunk boundaries.
    try:
        tok = mgr.llm.get_tokenizer()
        # Compare text-only vs text+task token lengths.
        sample_n = min(5, len(texts))
        lens_text = [len(tok.encode(t)) for t in texts[:sample_n]]
        lens_full = [len(tok.encode(p)) for p in prompts[:sample_n]]
        chunk = int(LMCACHE_CHUNK_SIZE)
        expected = [min(lt, lf) // chunk * chunk for lt, lf in zip(lens_text, lens_full)]
        print(
            f"[LMCache] chunk_size={chunk} sample_expected_hit_tokens={expected} "
            f"(text_lens={lens_text}, full_lens={lens_full})",
            flush=True,
        )
    except Exception:
        pass

    # Workflow 2: Online inference (KV prefetch to GPU happens here)
    metrics = _run_generate(
        mgr.llm,
        run_prompts,
        enable_profiling=ENABLE_PROFILING,
        sampling_extra_args={
            "kv_transfer_params": {
                # Enforce inference as load-only (no further saves) while
                # keeping the same engine instance.
                "lmcache.skip_save": True,
            }
        },
    )

    mgr.close()

    result: dict[str, Any] = {
        "mode": "lmcache_gds",
        "lmcache_cfg": asdict(lmcache_cfg),
        "warmup_elapsed_s": warm_dt,
        "metrics": metrics,
    }
    if ENABLE_PROFILING:
        result["profile"] = {
            "dir": str(Path(PROFILE_ROOT_DIR).expanduser().resolve() / "lmcache_gds"),
            "requests": int(min(PROFILE_PROMPT_LIMIT, len(prompts))),
        }
    conn.send(result)


def _bench_ssd_offload(conn):
    texts = _load_texts_csv(DATASET_PATH, DATASET_TEXT_COL, LIMIT)
    prompts = _build_task_prompts(texts)
    run_prompts = prompts[:PROFILE_PROMPT_LIMIT] if ENABLE_PROFILING else prompts

    # Reset SSD cache directory to ensure a clean warmup.
    cache_dir = Path(SSD_CACHE_DIR)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mgr = SSDKVCacheManager(
        model=MODEL,
        ssd_cache_dir=cache_dir,
        max_blocks=SSD_MAX_BLOCKS,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        pythonhashseed="0",
        # Ensure warmup stores are not deferred past generate() return.
        # sync_store_during_warmup=True,
        **({"profiler_config": _vllm_profiler_config("ssd_offload")} if ENABLE_PROFILING else {}),
    )

    # Workflow 1: KV generation (warmup text-only prefixes)
    warm_t0 = time.perf_counter()
    _warmup_prefixes_batched(mgr.llm, texts, batch_size=WARMUP_BATCH_SIZE)
    warm_dt = time.perf_counter() - warm_t0

    # Workflow 2: Online inference (should reuse prefix KV from SSD)
    metrics = _run_generate(mgr.llm, run_prompts, enable_profiling=ENABLE_PROFILING)

    result: dict[str, Any] = {
        "mode": "ssd_offload",
        "ssd_cache_dir": str(cache_dir),
        "ssd_max_blocks": int(SSD_MAX_BLOCKS),
        "warmup_elapsed_s": warm_dt,
        "metrics": metrics,
    }
    if ENABLE_PROFILING:
        result["profile"] = {
            "dir": str(Path(PROFILE_ROOT_DIR).expanduser().resolve() / "ssd_offload"),
            "requests": int(min(PROFILE_PROMPT_LIMIT, len(prompts))),
        }
    conn.send(result)


def main() -> None:
    # LMCache may use Python's builtin hash depending on config/version.
    # This MUST be fixed before spawning child processes, otherwise cache keys
    # can differ across processes and cause 0-hit lookups.
    os.environ.setdefault("PYTHONHASHSEED", "0")

    # Must be set before vLLM import in child processes.
    configure_cuda(CUDA)

    ctx = mp.get_context("spawn")

    def run_child(fn):
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=fn, args=(child,))
        p.start()
        result = parent.recv()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Child process failed: {fn.__name__}, exit={p.exitcode}")
        return result

    baseline = run_child(_bench_baseline)
    lmcache = run_child(_bench_lmcache_gds)
    ssd = run_child(_bench_ssd_offload)

    b = baseline["metrics"]
    c = lmcache["metrics"]
    s = ssd["metrics"]

    print("\n=== Baseline (no kv connector) ===")
    print({"profile": baseline.get("profile")})
    print(b)

    print("\n=== LMCache(GDS) enabled ===")
    print({k: lmcache[k] for k in lmcache.keys() if k != "metrics"})
    print(c)

    print("\n=== SSD Offloading (Scheme B) enabled ===")
    print({k: ssd[k] for k in ssd.keys() if k != "metrics"})
    print(s)

    speedup_lmcache = (c["tok_per_s"] / b["tok_per_s"]) if b["tok_per_s"] > 0 else float("inf")
    speedup_ssd = (s["tok_per_s"] / b["tok_per_s"]) if b["tok_per_s"] > 0 else float("inf")
    print("\n=== Summary ===")
    print(f"baseline tok/s: {b['tok_per_s']:.2f}")
    print(f"lmcache  tok/s: {c['tok_per_s']:.2f}")
    print(f"ssd      tok/s: {s['tok_per_s']:.2f}")
    print(f"lmcache speedup: {speedup_lmcache:.3f}x")
    print(f"ssd     speedup: {speedup_ssd:.3f}x")


if __name__ == "__main__":
    main()
