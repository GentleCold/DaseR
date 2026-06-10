# SPDX-License-Identifier: Apache-2.0
"""End-to-end inference benchmark: DaseR transfer modes vs LMCache.

Runs the same IMDB-review prompt batch through vLLM twice, once with each
KV connector, measuring cold-pass and warm-pass elapsed time and prompt-token
throughput. Prefix cache is disabled so the NVMe storage tier is the only
source of cross-run speedup.

Usage:
    python benchmarks/bench_e2e_daser_vs_lmcache.py \\
        --model /path/to/model \\
        --store-dir /path/to/benchmark-scratch \\
        --imdb /path/to/imdb.csv \\
        [--num-prompts 200] \\
        [--out results.json]
"""

# ruff: noqa: E402

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any
import uuid

# Third Party
import torch

from daser.connector.helpers import hash_tokens
from daser.connector.ipc_client import IPCClientSync
from daser.logging import init_logger
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BYTES_PER_GIB: int = 1024**3
EVICT_L2_FRACTION: float = 0.95
EVICT_L1_FRACTION: float = 0.9
LMCACHE_LOCAL_SSD_STAGING_GB: float = 0.5
COMPARISON_GDS = "gds-vs-lmcache-local-ssd"
COMPARISON_IOURING_MEM = "iouring-mem-vs-lmcache-local-ssd-mem"


def write_json_results(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write benchmark results to JSON with fallback object stringification.

    Args:
        path: Destination JSON path.
        payload: Benchmark result object. It may contain vLLM ``RequestOutput``
            objects retained for in-process correctness checks.

    Async/thread-safety:
        Synchronous file write helper for benchmark process shutdown.
    """
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of one GPU's memory state.

    Args:
        index: Physical GPU index reported by nvidia-smi.
        total_mb: Total memory in MiB.
        used_mb: Used memory in MiB.
        free_mb: Free memory in MiB.

    Thread-safety:
        Immutable value object; safe to share between threads.
    """

    index: int
    total_mb: int
    used_mb: int
    free_mb: int


@dataclass(frozen=True)
class BenchmarkCapacityLimits:
    """Machine-derived benchmark capacity ceilings.

    Args:
        max_l1_bytes: Maximum DaseR L1 bytes to use.
        max_l2_bytes: Maximum DaseR L2 bytes to use.
        memory_available_bytes: Observed host memory available for pinned L1.
        disk_available_bytes: Observed store directory disk free bytes.

    Thread-safety:
        Immutable value object; safe to share between threads.
    """

    max_l1_bytes: int
    max_l2_bytes: int
    memory_available_bytes: int
    disk_available_bytes: int


@dataclass(frozen=True)
class BenchmarkSizing:
    """Derived transfer and cache capacities for one benchmark run.

    Args:
        daser_slots: Number of DaseR L2 slots.
        daser_l2_bytes: DaseR L2 bytes.
        daser_l1_bytes: DaseR L1 bytes.
        lmcache_disk_gb: LMCache local disk limit in GiB units.
        lmcache_cpu_gb: LMCache local CPU limit in GiB units.
        capacity_capped: Whether machine limits capped the requested sizes.

    Thread-safety:
        Immutable value object; safe to share between threads.
    """

    daser_slots: int
    daser_l2_bytes: int
    daser_l1_bytes: int
    lmcache_disk_gb: float
    lmcache_cpu_gb: float
    capacity_capped: bool


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy when available, and torch RNGs for benchmark runs.

    Args:
        seed: Deterministic seed value to apply.

    Returns:
        None.

    Thread-safety:
        Mutates process-global RNG state and should be called during startup.
    """
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch as _torch
    except ImportError:
        _torch = None
    if _torch is not None:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)


def load_prompts(imdb_path: str, n: int) -> list[str]:
    """Load IMDB reviews as raw prompt strings.

    Args:
        imdb_path: Path to imdb.csv with a ``review`` column.
        n: Number of prompts to return.

    Returns:
        List of raw review strings.

    Thread-safety:
        Performs only local file reads and has no shared mutable state.
    """
    if not os.path.exists(imdb_path):
        raise FileNotFoundError(f"IMDB CSV not found: {imdb_path}")

    out: list[str] = []
    with open(imdb_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(out) >= n:
                break
            review = row.get("review", "").strip()
            if review:
                out.append(review)
    return out


def load_longbench_prompts(jsonl_path: str, n: int = 0) -> list[str]:
    """Load Longbench JSONL prompts as raw strings.

    Each JSON line must contain ``context`` and may contain ``input`` keys.
    The prompt string is ``context + "\\n\\n" + input``, or just ``context``
    when ``input`` is empty.

    Args:
        jsonl_path: Path to a Longbench .jsonl file.
        n: Maximum prompts to load (0 = load all).

    Returns:
        List of raw prompt strings.

    Thread-safety:
        Performs only local file reads and has no shared mutable state.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Longbench JSONL not found: {jsonl_path}")

    out: list[str] = []
    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            if n > 0 and len(out) >= n:
                break
            rec = json.loads(line)
            context = rec.get("context", "")
            inp = rec.get("input", "")
            prompt = f"{context}\n\n{inp}" if inp else context
            out.append(prompt)
    return out


def tokenise_and_truncate(
    prompts: list[str], tokenizer: Any, max_tokens: int, block_tokens: int
) -> list[list[int]]:
    """Tokenise and truncate prompts to a token ceiling.

    Args:
        prompts: Raw prompt strings.
        tokenizer: Hugging Face tokenizer.
        max_tokens: Per-prompt token ceiling.
        block_tokens: KV block size in tokens.

    Returns:
        Token-ID lists suitable for vLLM ``TokensPrompt``.

    Thread-safety:
        Depends on tokenizer implementation; this helper keeps no state.
    """
    out: list[list[int]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        if len(ids) < block_tokens + 1:
            pad = tokenizer.encode(" ", add_special_tokens=False)
            if pad:
                while len(ids) < block_tokens + 1:
                    ids = ids + pad
                ids = ids[: block_tokens + 1]
        out.append(ids)
    return out


def query_gpus() -> list[GPUInfo]:
    """Return GPU memory snapshots from nvidia-smi.

    Args:
        None.

    Returns:
        GPUInfo entries. Returns an empty list when nvidia-smi is unavailable
        or returns an unexpected payload.

    Thread-safety:
        Spawns a read-only subprocess and keeps no shared mutable state.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index, total_mb, used_mb, free_mb = (int(part) for part in parts)
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                index=index,
                total_mb=total_mb,
                used_mb=used_mb,
                free_mb=free_mb,
            )
        )
    return gpus


def choose_gpu_id(
    gpus: list[GPUInfo], requested: str, current_visible: str | None
) -> str | None:
    """Choose which GPU ID the benchmark should expose to vLLM.

    Args:
        gpus: GPU memory snapshots.
        requested: ``auto``, ``current``, or a concrete CUDA device index.
        current_visible: Existing ``CUDA_VISIBLE_DEVICES`` value.

    Returns:
        CUDA device ID string to use, or None when the current environment
        should be left unchanged.

    Thread-safety:
        Pure function.
    """
    if requested == "current":
        return current_visible
    if requested != "auto":
        return requested
    if not gpus:
        return current_visible
    return str(max(gpus, key=lambda gpu: (gpu.free_mb, gpu.total_mb)).index)


def apply_gpu_selection(requested: str) -> str | None:
    """Apply benchmark GPU selection before CUDA libraries initialize.

    Args:
        requested: ``auto``, ``current``, or a concrete CUDA device index.

    Returns:
        Selected CUDA device ID, or None when unchanged.

    Thread-safety:
        Mutates ``os.environ`` and should run during process startup.
    """
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = choose_gpu_id(query_gpus(), requested, current)
    if selected is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


def derive_capacity_limits(
    store_dir: str,
    gpu_id: str | None,
    disk_fraction: float = 0.8,
    host_mem_fraction: float = 0.25,
    max_l1_gib: float = 64.0,
    max_l2_gib: float = 512.0,
) -> BenchmarkCapacityLimits:
    """Derive benchmark capacity ceilings from current machine state.

    Args:
        store_dir: Benchmark store directory.
        gpu_id: Selected physical GPU ID, if known.
        disk_fraction: Fraction of free disk space allowed for L2.
        host_mem_fraction: Fraction of available host memory allowed for L1.
        max_l1_gib: Absolute L1 ceiling.
        max_l2_gib: Absolute L2 ceiling.

    Returns:
        BenchmarkCapacityLimits for DaseR sizing.

    Thread-safety:
        Reads system state and keeps no shared mutable state.
    """
    os.makedirs(store_dir, exist_ok=True)
    disk_free = shutil.disk_usage(store_dir).free
    host_available = _host_available_bytes()
    max_l1 = min(
        int(max_l1_gib * BYTES_PER_GIB),
        int(host_available * host_mem_fraction),
    )
    max_l2 = min(
        int(max_l2_gib * BYTES_PER_GIB),
        int(disk_free * disk_fraction),
    )
    return BenchmarkCapacityLimits(
        max_l1_bytes=max(0, max_l1),
        max_l2_bytes=max(0, max_l2),
        memory_available_bytes=host_available,
        disk_available_bytes=disk_free,
    )


def derive_benchmark_sizing(
    total_blocks: int,
    max_prompt_blocks: int,
    slot_size: int,
    mode: str,
    evict: bool,
    capacity_limits: BenchmarkCapacityLimits,
) -> BenchmarkSizing:
    """Derive L1/L2 sizes for benchmark scenarios with machine caps.

    Args:
        total_blocks: KV blocks in the workload.
        max_prompt_blocks: Largest single-prompt aligned KV block count.
        slot_size: Bytes per KV block slot.
        mode: Comparison mode.
        evict: When True, choose capacities that force L2 eviction.
        capacity_limits: Machine-derived maximum capacities.

    Returns:
        BenchmarkSizing with aligned DaseR and LMCache capacities.

    Thread-safety:
        Pure function.
    """
    if total_blocks <= 0:
        raise ValueError("total_blocks must be positive")
    required_l2_bytes = max(1, max_prompt_blocks) * slot_size
    if capacity_limits.max_l2_bytes < required_l2_bytes:
        raise ValueError(
            "benchmark L2 capacity cap cannot fit the largest prompt "
            f"({capacity_limits.max_l2_bytes} < {required_l2_bytes} bytes)"
        )

    if evict:
        desired_l2_blocks = max(1, math.floor(total_blocks * EVICT_L2_FRACTION))
        if desired_l2_blocks >= total_blocks:
            desired_l2_blocks = max(1, total_blocks - 1)
    else:
        desired_l2_blocks = max(1, math.ceil(total_blocks * 1.5))

    max_l2_blocks = max(1, capacity_limits.max_l2_bytes // slot_size)
    l2_blocks = min(desired_l2_blocks, max_l2_blocks)
    if not evict and l2_blocks < total_blocks:
        l2_blocks = max(max_prompt_blocks, l2_blocks)

    desired_l2_bytes = desired_l2_blocks * slot_size
    daser_l2_bytes = l2_blocks * slot_size
    capacity_capped = daser_l2_bytes < desired_l2_bytes

    if mode == COMPARISON_IOURING_MEM:
        required_l1_bytes = max(1, max_prompt_blocks) * slot_size
        if capacity_limits.max_l1_bytes < required_l1_bytes:
            raise ValueError(
                "benchmark L1 capacity cap cannot fit the largest prompt "
                f"({capacity_limits.max_l1_bytes} < {required_l1_bytes} bytes)"
            )
        cpu_gib_ceiling = 96.0
        workload_bytes = total_blocks * slot_size
        daser_l1_bytes = min(
            workload_bytes,
            int(cpu_gib_ceiling * BYTES_PER_GIB),
            capacity_limits.max_l1_bytes,
        )
        capacity_capped = capacity_capped or daser_l1_bytes < workload_bytes
    else:
        daser_l1_bytes = 0

    lmcache_cpu_gb = bytes_to_lmcache_gb(daser_l1_bytes)

    return BenchmarkSizing(
        daser_slots=l2_blocks,
        daser_l2_bytes=daser_l2_bytes,
        daser_l1_bytes=daser_l1_bytes,
        lmcache_disk_gb=bytes_to_lmcache_gb(daser_l2_bytes),
        lmcache_cpu_gb=lmcache_cpu_gb,
        capacity_capped=capacity_capped,
    )


def bytes_to_lmcache_gb(nbytes: int) -> float:
    """Convert bytes to LMCache's GiB-based config value.

    Args:
        nbytes: Capacity in bytes.

    Returns:
        Size value for LMCache GB config knobs.

    Thread-safety:
        Pure function.
    """
    return nbytes / BYTES_PER_GIB


def _host_available_bytes() -> int:
    """Return currently available host memory bytes.

    Args:
        None.

    Returns:
        Available host memory bytes, or a conservative fallback.

    Thread-safety:
        Reads procfs and keeps no shared mutable state.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 * BYTES_PER_GIB


# KV cache bytes per token for Qwen3-8B bf16.
# 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
# 2 * 36 * 8 * 128 * 2 = 147,456
KV_BYTES_PER_TOKEN: int = 147456

# Estimated model weight footprint for Qwen3-8B bf16 (~8B params).
MODEL_WEIGHTS_GIB: float = 16.0

# Estimated vLLM runtime overhead (CUDA context, workspace, etc.).
VLLM_OVERHEAD_GIB: float = 2.0


def calculate_max_model_len(
    gpu_id: str | None,
    gpu_memory_utilization: float,
    block_tokens: int = 16,
    model_weights_gib: float = MODEL_WEIGHTS_GIB,
    vllm_overhead_gib: float = VLLM_OVERHEAD_GIB,
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN,
) -> int:
    """Compute ``max_model_len`` from available GPU VRAM.

    Queries GPU total memory via ``query_gpus()`` (falling back to
    ``torch.cuda.get_device_properties()`` if nvidia-smi is unavailable),
    then subtracts estimated model weights and vLLM runtime overhead.

    Args:
        gpu_id: CUDA device index string (e.g. ``"0"``), or None.
        gpu_memory_utilization: vLLM ``gpu_memory_utilization`` (0.0–1.0).
        block_tokens: KV block size in tokens (used for alignment).
        model_weights_gib: Estimated model weight footprint in GiB.
        vllm_overhead_gib: Estimated vLLM runtime overhead in GiB.
        kv_bytes_per_token: KV cache bytes consumed per token.

    Returns:
        Maximum model length in tokens, block-aligned.

    Raises:
        RuntimeError: If GPU memory cannot be determined or is insufficient.

    Thread-safety:
        Reads system/GPU state and keeps no shared mutable state.
    """
    gpus = query_gpus()
    total_mb = 0
    if gpus:
        if gpu_id is not None:
            for gpu in gpus:
                if str(gpu.index) == str(gpu_id):
                    total_mb = gpu.total_mb
                    break
        if total_mb == 0 and gpus:
            selected = gpus[0]
            total_mb = selected.total_mb

    if total_mb == 0:
        try:
            import torch as _torch
        except ImportError:
            _torch = None
        if _torch is not None and _torch.cuda.is_available():
            props = _torch.cuda.get_device_properties(0)
            total_mb = int(props.total_memory / (1024 * 1024))

    if total_mb == 0:
        raise RuntimeError(
            "Cannot determine GPU memory; pass --max-model-len explicitly"
        )

    total_gib = total_mb / 1024.0
    available_for_kv_gib = (
        total_gib * gpu_memory_utilization - model_weights_gib - vllm_overhead_gib
    )

    if available_for_kv_gib <= 0:
        raise RuntimeError(
            f"Not enough VRAM for KV cache: total={total_gib:.1f} GiB, "
            f"gpu_util={gpu_memory_utilization}, "
            f"model={model_weights_gib:.1f} GiB, "
            f"overhead={vllm_overhead_gib:.1f} GiB"
        )

    max_tokens = int(available_for_kv_gib * BYTES_PER_GIB / kv_bytes_per_token)
    return (max_tokens // block_tokens) * block_tokens


# ---------------------------------------------------------------------------
# Constants — Qwen3-8B KV geometry (matches tests/integration/conftest.py)
# ---------------------------------------------------------------------------
NUM_KV_HEADS: int = 8
HEAD_DIM: int = 128
NUM_LAYERS: int = 36
BLOCK_TOKENS: int = 16
DTYPE_BYTES: int = 2  # bfloat16
SLOT_SIZE: int = NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS * BLOCK_TOKENS * DTYPE_BYTES
# 8 * 128 * 2 * 36 * 16 * 2 = 2,359,296 bytes

MAX_MODEL_LEN: int = 2048
BENCHMARK_SEED: int = 42
MAX_INPUT_TOKENS_DEFAULT: int = MAX_MODEL_LEN
GPU_MEM_UTIL_DEFAULT: float = 0.9
MAX_NUM_SEQS_DEFAULT: int = 64


# ---------------------------------------------------------------------------
# LLM build/destroy helpers
# ---------------------------------------------------------------------------


def _destroy_llm(llm: Any) -> None:
    """Shut down a vLLM LLM and free GPU memory."""
    try:
        try:
            llm.llm_engine.engine_core.shutdown(timeout=30.0)
        except TypeError:
            llm.llm_engine.engine_core.shutdown()
    except Exception as exc:
        logger.warning("engine_core.shutdown raised: %s", exc)
    finally:
        del llm
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Second pass: CUDA driver may release IPC memory after first
        # synchronize, so collect again for LMCache→DaseR handover.
        gc.collect()
        torch.cuda.empty_cache()


def wait_gpu_memory(
    gpu_util: float,
    timeout_s: float = 60.0,
    poll_s: float = 1.0,
) -> None:
    """Block until the selected GPU has enough free memory for a new LLM.

    vLLM V1 EngineCore subprocesses may hold GPU memory briefly after
    shutdown.  This polls ``torch.cuda.mem_get_info`` until free memory
    reaches ``total * gpu_util``.

    Args:
        gpu_util: vLLM ``gpu_memory_utilization`` for the next LLM.
        timeout_s: Maximum wait time in seconds.
        poll_s: Interval between polls in seconds.

    Raises:
        RuntimeError: If free memory never reaches the required threshold
            within *timeout_s*.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        free, total = torch.cuda.mem_get_info()
        free_gib = free / (1024**3)
        total_gib = total / (1024**3)
        needed = total_gib * gpu_util
        if free_gib >= needed:
            logger.info(
                "[GPU] %.2f GiB free >= %.2f GiB needed — proceeding",
                free_gib,
                needed,
            )
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU memory not freed after {timeout_s:.0f}s: "
                f"{free_gib:.2f}/{total_gib:.2f} GiB free, "
                f"need {needed:.2f} GiB"
            )
        logger.debug(
            "[GPU] waiting for memory: %.2f/%.2f GiB free, need %.2f GiB",
            free_gib,
            total_gib,
            needed,
        )
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# DaseR harness
# ---------------------------------------------------------------------------


class DaserHarness:
    """Owns a DaseR IPCServer and optional store file for one benchmark run."""

    def __init__(
        self,
        store_dir: str,
        socket_dir: str,
        total_slots: int,
        model_path: str,
        gpu_util: float,
        max_num_seqs: int,
        transfer_mode: str,
        l1_bytes: int,
        max_model_len: int = MAX_MODEL_LEN,
        enable_prefix_caching: bool = False,
        cache_reuse_mode: str = "prefix",
        skip_l2: bool = False,
    ) -> None:
        """Initialise paths and store file.

        Args:
            store_dir: Directory to hold DaseR store files.
            socket_dir: Short directory to hold the IPC socket.
            total_slots: Pre-allocated slot count for the store.
            model_path: HF model path for vLLM.
            gpu_util: vLLM ``gpu_memory_utilization``.
            max_num_seqs: vLLM ``max_num_seqs``.
            transfer_mode: DaseR transfer backend selected for the run.
            l1_bytes: L1 byte capacity for tiered transfer mode.
            max_model_len: vLLM ``max_model_len`` override.
            enable_prefix_caching: Enable vLLM prefix caching.
            cache_reuse_mode: ``"prefix"`` or ``"chunk"``.
            skip_l2: Use volatile L1 memory only and do not create a store file.
        """
        self.store_dir = store_dir
        self.socket_dir = socket_dir
        self.socket_path = os.path.join(socket_dir, "d.sock")
        self.store_path = os.path.join(store_dir, "daser.store")
        self.model_path = model_path
        self.total_slots = total_slots
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.transfer_mode = transfer_mode
        self.l1_bytes = l1_bytes
        self.max_model_len = max_model_len
        self.enable_prefix_caching = enable_prefix_caching
        self.cache_reuse_mode = cache_reuse_mode
        self.skip_l2 = skip_l2
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: IPCServer | None = None

    def start(self) -> None:
        """Pre-allocate optional store and start IPCServer in a daemon thread."""
        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.socket_dir, exist_ok=True)
        size = self.total_slots * SLOT_SIZE
        if not self.skip_l2:
            with open(self.store_path, "wb") as f:
                f.truncate(size)

        metadata = MetadataStore(total_slots=self.total_slots)
        registry = DocRegistry()
        cm = ChunkManager(
            total_slots=self.total_slots,
            metadata_store=metadata,
            doc_registry=registry,
        )
        if self.cache_reuse_mode == "chunk":
            retrieval_index = ChunkReuseIndex(block_tokens=BLOCK_TOKENS)
            position_encoder = ChunkPositionEncoder(initial_offset=0)
        else:
            retrieval_index = PrefixHashIndex(block_tokens=BLOCK_TOKENS)
            position_encoder = FixedOffsetEncoder(fixed_offset=0)

        core = ServerCore(
            chunk_manager=cm,
            retrieval_index=retrieval_index,
            position_encoder=position_encoder,
            slot_size=SLOT_SIZE,
            block_tokens=BLOCK_TOKENS,
        )
        server = IPCServer(
            socket_path=self.socket_path,
            core=core,
            runtime_config={
                "socket_path": self.socket_path,
                "store_path": "" if self.skip_l2 else self.store_path,
                "slot_size": SLOT_SIZE,
                "block_tokens": BLOCK_TOKENS,
                "model_id": "qwen3-8b",
                "transfer_mode": self.transfer_mode,
                "l1_size_bytes": self.l1_bytes,
                "l2_size_bytes": size,
                "total_slots": self.total_slots,
                "total_store_bytes": size,
                "cache_reuse_mode": self.cache_reuse_mode,
                "skip_l2": self.skip_l2,
            },
        )

        loop = asyncio.new_event_loop()
        started = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.start())
            started.set()
            loop.run_forever()

        thread = threading.Thread(target=_run, daemon=True, name="daser-bench-server")
        thread.start()
        assert started.wait(timeout=10.0), "DaseR IPCServer failed to start in 10s"
        self._loop = loop
        self._thread = thread
        self._server = server
        if self.skip_l2:
            logger.info(
                "[DaseR] server up — skip_l2 L1-only (logical %.1f GiB, %d slots)",
                size / BYTES_PER_GIB,
                self.total_slots,
            )
        else:
            logger.info(
                "[DaseR] server up — store=%s (%.1f GiB, %d slots)",
                self.store_path,
                size / BYTES_PER_GIB,
                self.total_slots,
            )

    def build_llm(self) -> Any:
        """Construct a vLLM LLM wired to DaserConnector.

        Returns:
            Configured vLLM ``LLM`` instance.
        """
        from vllm import LLM  # Third Party

        kv_transfer_config = {
            "kv_connector": "DaserConnector",
            "kv_connector_module_path": "daser.connector.daser_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "socket_path": self.socket_path,
                "cache_reuse_mode": self.cache_reuse_mode,
            },
        }
        return LLM(
            model=self.model_path,
            kv_transfer_config=kv_transfer_config,
            gpu_memory_utilization=self.gpu_util,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            seed=BENCHMARK_SEED,
            enable_prefix_caching=self.enable_prefix_caching,
            disable_hybrid_kv_cache_manager=True,
        )

    def wait_until_committed(
        self,
        prompts: list[list[int]],
        block_tokens: int,
        require_all_commits: bool,
        require_l2_drain: bool,
        timeout_s: float = 120.0,
    ) -> None:
        """Wait until cold-pass transfer writes are visible in DaseR.

        Args:
            prompts: Tokenized benchmark prompts.
            block_tokens: block size used by the index.
            require_all_commits: when True, require every prompt chunk's store
                commit to complete. Full lookup visibility is measured after
                timing by ``visible_prompt_mask``.
            require_l2_drain: when True, wait for async L2 persistence after
                commit visibility so background cold writes do not interfere
                with warm-load timing.
            timeout_s: Maximum wait time.
        """
        client = IPCClientSync(self.socket_path)
        deadline = time.monotonic() + timeout_s
        model_id = "qwen3-8b"

        # ---- compute unique chunk keys (diagnostic only) ----
        chunk_keys: set[str] = set()
        for tokens in prompts:
            aligned = (len(tokens) // block_tokens) * block_tokens
            if aligned > 0:
                chunk_keys.add(hash_tokens(tokens[:aligned]))
        if len(chunk_keys) < len(prompts):
            logger.info(
                "[DaseR] unique aligned chunks: %d/%d prompts (%d duplicate prefixes)",
                len(chunk_keys),
                len(prompts),
                len(prompts) - len(chunk_keys),
            )

        # ---- build prompt prefix list for visibility check ----
        prompt_prefixes: list[list[int]] = []
        for tokens in prompts:
            aligned = (len(tokens) // block_tokens) * block_tokens
            prompt_prefixes.append(tokens[:aligned] if aligned > 0 else [])

        try:
            while True:
                if not require_all_commits:
                    stats = client.commit_stats()
                    committed = int(stats.get("commit_requests", 0))
                    if committed > 0:
                        if require_l2_drain:
                            client.transfer_drain()
                        late = int(stats.get("late_evicted_commits", 0))
                        logger.info(
                            "[DaseR] cold transfer writes drained "
                            "(commits=%d, late_evicted=%d, lookups=%d/%d)",
                            committed,
                            late,
                            int(stats.get("lookup_hits", 0)),
                            int(stats.get("lookup_requests", 0)),
                        )
                        return
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "timed out waiting for any DaseR commit request"
                        )
                    time.sleep(0.05)
                    continue

                # require_all_commits: check every prompt prefix is visible.
                # We cannot rely on commit_requests count because intra-run
                # cache hits mean some unique chunk keys are never committed
                # (an earlier request with a matching shorter prefix already
                # committed that chunk).
                missing = 0
                for prefix in prompt_prefixes:
                    if not prefix:
                        continue
                    try:
                        chunks = client.lookup(prefix, model_id)
                    except Exception:
                        chunks = []
                    if not chunks:
                        missing += 1
                if missing == 0:
                    if require_l2_drain:
                        client.transfer_drain()
                    stats = client.commit_stats()
                    logger.info(
                        "[DaseR] all %d prompts visible (commits=%d, lookups=%d/%d)",
                        len(prompts),
                        int(stats.get("commit_requests", 0)),
                        int(stats.get("lookup_hits", 0)),
                        int(stats.get("lookup_requests", 0)),
                    )
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for DaseR commits "
                        f"({len(prompts) - missing}/{len(prompts)} prompts visible)"
                    )
                time.sleep(0.05)
        finally:
            client.close()

    def visible_prompt_count(
        self,
        prompts: list[list[int]],
        model_id: str,
        block_tokens: int,
    ) -> int:
        """Return how many prompt prefixes are currently visible in DaseR.

        Args:
            prompts: Tokenized benchmark prompts.
            model_id: DaseR model ID used by the harness.
            block_tokens: block size used by the index.

        Returns:
            Number of prompts whose aligned prefix can be looked up.
        """
        return sum(self.visible_prompt_mask(prompts, model_id, block_tokens))

    def visible_prompt_mask(
        self,
        prompts: list[list[int]],
        model_id: str,
        block_tokens: int,
    ) -> list[bool]:
        """Return per-prompt DaseR lookup visibility before a warm pass.

        Args:
            prompts: Tokenized benchmark prompts.
            model_id: DaseR model ID used by the harness.
            block_tokens: block size used by the index.

        Returns:
            Boolean list aligned with ``prompts``.
        """
        client = IPCClientSync(self.socket_path)
        visible: list[bool] = []
        try:
            for tokens in prompts:
                aligned = (len(tokens) // block_tokens) * block_tokens
                if aligned <= 0:
                    visible.append(False)
                    continue
                chunks = client.lookup(tokens[:aligned], model_id)
                visible.append(
                    bool(chunks) and int(chunks[0].get("token_count", 0)) == aligned
                )
        finally:
            client.close()
        return visible

    def stop(self) -> None:
        """Stop the IPCServer cleanly."""
        if self._server is not None and self._loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._server.stop(), self._loop)
                fut.result(timeout=10.0)
            except Exception as exc:
                logger.warning("[DaseR] server stop raised: %s", exc)
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=10.0)
        try:
            os.rmdir(self.socket_dir)
        except OSError:
            pass
        logger.info("[DaseR] server stopped")


# ---------------------------------------------------------------------------
# LMCache harness
# ---------------------------------------------------------------------------


class LMCacheHarness:
    """Configures LMCache via env vars and builds an LMCacheConnectorV1 LLM."""

    def __init__(
        self,
        tmpdir: str,
        total_bytes: int,
        model_path: str,
        gpu_util: float,
        max_num_seqs: int,
        local_cpu: bool,
        disk_limit_gb: float,
        cpu_limit_gb: float,
        max_model_len: int = MAX_MODEL_LEN,
        enable_prefix_caching: bool = False,
    ) -> None:
        """Initialise paths.

        Args:
            tmpdir: Directory used as LMCache's local_disk.
            total_bytes: Expected bytes-on-disk (drives max_local_disk_size).
            model_path: HF model path for vLLM.
            gpu_util: vLLM ``gpu_memory_utilization``.
            max_num_seqs: vLLM ``max_num_seqs``.
            local_cpu: Whether LMCache L1 CPU tier is enabled.
            disk_limit_gb: LMCache local-disk limit in GiB units.
            cpu_limit_gb: LMCache local-CPU limit in GiB units.
            max_model_len: vLLM ``max_model_len`` override.
            enable_prefix_caching: Enable vLLM prefix caching.
        """
        self.tmpdir = tmpdir
        self.model_path = model_path
        self.total_bytes = total_bytes
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.local_cpu = local_cpu
        self.disk_limit_gb = disk_limit_gb
        self.cpu_limit_gb = cpu_limit_gb
        self.max_model_len = max_model_len
        self.enable_prefix_caching = enable_prefix_caching
        digest = hashlib.sha1(tmpdir.encode("utf-8")).hexdigest()[:12]
        self.instance_id = f"daser_vs_lmcache_{digest}"
        self._saved_env: dict[str, str | None] = {}

    def start(self) -> None:
        """Apply LMCache env configuration before LLM init."""
        env = {
            "LMCACHE_CHUNK_SIZE": str(BLOCK_TOKENS),
            "LMCACHE_LOCAL_CPU": "True" if self.local_cpu else "False",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": f"{self.cpu_limit_gb:.6f}",
            "LMCACHE_LOCAL_DISK": f"file://{self.tmpdir}/",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": f"{self.disk_limit_gb:.6f}",
            "LMCACHE_USE_LAYERWISE": "False",
            "LMCACHE_LMCACHE_INSTANCE_ID": self.instance_id,
            # The BENCHMARK_SEED_ENV value ("42") is imported by callers.
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "42"),
        }
        for k, v in env.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v
        logger.info(
            "[LMCache] env configured — local_disk=%s (%s GB-config ceiling)",
            self.tmpdir,
            env["LMCACHE_MAX_LOCAL_DISK_SIZE"],
        )

    def build_llm(self) -> Any:
        """Construct a vLLM LLM wired to LMCacheConnectorV1.

        Returns:
            Configured vLLM ``LLM`` instance.
        """
        from vllm import LLM  # Third Party

        kv_transfer_config = {
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_both",
        }
        return LLM(
            model=self.model_path,
            kv_transfer_config=kv_transfer_config,
            gpu_memory_utilization=self.gpu_util,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            seed=BENCHMARK_SEED,
            enable_prefix_caching=self.enable_prefix_caching,
        )

    def wait_for_disk_quiescence(
        self,
        timeout_s: float = 120.0,
        stable_for_s: float = 1.0,
        poll_s: float = 0.1,
    ) -> None:
        """Wait until LMCache local-disk files stop changing.

        LMCache's LocalDiskBackend submits SSD writes to a background worker and
        adds a key to the lookup index only after that key's write completes.
        The benchmark cannot call a public LMCache drain API, so it waits for
        the observable local-disk tier to become quiescent before the warm pass.

        Args:
            timeout_s: Maximum wait time in seconds.
            stable_for_s: Required duration with unchanged file count and bytes.
            poll_s: Poll interval in seconds.

        Raises:
            TimeoutError: If the local-disk snapshot does not become stable.
        """
        from pathlib import Path

        root = Path(self.tmpdir)
        deadline = time.monotonic() + timeout_s
        last_snapshot: tuple[int, int] | None = None
        stable_since: float | None = None
        while time.monotonic() < deadline:
            snapshot = self._disk_snapshot(root)
            if snapshot == last_snapshot and snapshot[0] > 0:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= stable_for_s:
                    logger.info(
                        "[LMCache] local disk quiescent: files=%d bytes=%d",
                        snapshot[0],
                        snapshot[1],
                    )
                    return
            else:
                last_snapshot = snapshot
                stable_since = None
            time.sleep(poll_s)
        raise TimeoutError(
            "LMCache local-disk writes did not become quiescent within "
            f"{timeout_s:.1f}s under {root}"
        )

    def stop(self) -> None:
        """Restore previous env values."""
        for k, saved in self._saved_env.items():
            if saved is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved
        self._saved_env.clear()
        logger.info("[LMCache] env restored")

    @staticmethod
    def _disk_snapshot(root: Any) -> tuple[int, int]:
        """Return current LMCache local-disk file count and total bytes.

        Args:
            root: LMCache local-disk root.

        Returns:
            Tuple of ``(file_count, total_bytes)`` for stored ``.pt`` files.
        """
        count = 0
        total = 0
        for path in root.rglob("*.pt"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            count += 1
            total += stat.st_size
        return count, total


# ---------------------------------------------------------------------------
# Timed runner
# ---------------------------------------------------------------------------


def run_system(
    name: str,
    build_llm_fn: Any,
    prompts: list[list[int]],
    warm_skip_save: bool = False,
    after_cold_fn: Any | None = None,
) -> dict[str, Any]:
    """Run cold + warm timed passes for one system.

    Args:
        name: System label, used only for logging.
        build_llm_fn: Callable returning a fresh LLM instance.
        prompts: Prompt list to pass to generate().
        warm_skip_save: when True, use ``daser_skip_save`` on warm pass.
        after_cold_fn: Optional callback run after cold generation and before
            stopping the cold timer. DaseR uses this to include save
            commit/drain cost in cold elapsed time.

    Returns:
        Dict with cold_elapsed_s, warm_elapsed_s, cold_outputs,
        warm_outputs. ``cold_outputs`` and ``warm_outputs`` are the raw
        vLLM ``RequestOutput`` lists and can be passed directly to
        ``correctness_check`` or ``correctness_check_with_visibility``.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=BENCHMARK_SEED,
    )
    warm_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        if warm_skip_save
        else params
    )

    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]

    # NOTE: we intentionally do NOT destroy and rebuild the LLM between cold
    # and warm passes. LMCache's LocalDiskBackend keeps its chunk index in an
    # in-memory dict and does not scan the directory on startup, so rebuilding
    # the engine would orphan every chunk it just wrote. vLLM's in-GPU KV is
    # recycled between generate() calls with enable_prefix_caching=False, so
    # the warm pass still has to fetch from the external storage tier — which
    # is exactly the signal this benchmark measures.
    logger.info("[%s] building LLM", name)
    llm = build_llm_fn()

    logger.info("[%s] cold: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    cold_outputs = llm.generate(tp_prompts, params)
    if after_cold_fn is not None:
        logger.info("[%s] cold: waiting for save completion", name)
        after_cold_fn()
    cold_elapsed = time.perf_counter() - t0
    logger.info("[%s] cold elapsed: %.2fs", name, cold_elapsed)

    logger.info("[%s] warm: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    warm_outputs = llm.generate(tp_prompts, warm_params)
    warm_elapsed = time.perf_counter() - t0
    logger.info("[%s] warm elapsed: %.2fs", name, warm_elapsed)

    logger.info("[%s] destroying LLM", name)
    _destroy_llm(llm)

    return {
        "cold_elapsed_s": cold_elapsed,
        "warm_elapsed_s": warm_elapsed,
        "cold_outputs": cold_outputs,
        "warm_outputs": warm_outputs,
    }


def run_correctness_system(
    name: str,
    build_llm_fn: Any,
    prompts: list[list[int]],
    max_num_seqs: int,
    warm_skip_save: bool = False,
    after_cold_fn: Any | None = None,
    visible_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """Run an untimed cold/warm exact correctness pass.

    Args:
        name: System label, used only for logging.
        build_llm_fn: Callable returning an LLM instance.
        prompts: Prompt list to pass to generate().
        max_num_seqs: vLLM admission limit used for diagnostics.
        warm_skip_save: when True, skip DaseR duplicate warm stores.
        after_cold_fn: Optional callback run after cold correctness generation
            and before the warm correctness generation. DaseR uses this to
            make store commits visible before warm loads.
        visible_mask: Optional DaseR visible-hit mask for per-hit diagnostics.

    Returns:
        Correctness dictionary from ``correctness_check``.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=BENCHMARK_SEED,
    )
    warm_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        if warm_skip_save
        else params
    )
    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]

    logger.info("[%s] correctness: building exact-check LLM", name)
    llm = build_llm_fn()
    try:
        logger.info("[%s] correctness: cold generate(N=%d)", name, len(tp_prompts))
        cold_outputs = llm.generate(tp_prompts, params)
        if after_cold_fn is not None:
            logger.info("[%s] correctness: waiting for save completion", name)
            after_cold_fn()
        logger.info("[%s] correctness: warm generate(N=%d)", name, len(tp_prompts))
        warm_outputs = llm.generate(tp_prompts, warm_params)
        if visible_mask is None:
            return correctness_check(
                name,
                cold_outputs,
                warm_outputs,
                prompts,
                max_num_seqs,
            )
        return correctness_check_with_visibility(
            name,
            cold_outputs,
            warm_outputs,
            prompts,
            max_num_seqs,
            visible_mask,
        )
    finally:
        logger.info("[%s] correctness: destroying LLM", name)
        _destroy_llm(llm)


def run_lmcache_correctness(
    store_dir: str,
    total_bytes: int,
    model_path: str,
    gpu_util: float,
    max_num_seqs: int,
    local_cpu: bool,
    disk_limit_gb: float,
    cpu_limit_gb: float,
    prompts: list[list[int]],
) -> dict[str, Any]:
    """Run LMCache exact correctness in an isolated scratch store.

    Args:
        store_dir: Base directory for LMCache benchmark scratch files.
        total_bytes: Workload byte size used for LMCache sizing.
        model_path: HF model path for vLLM.
        gpu_util: vLLM GPU memory utilization.
        max_num_seqs: vLLM max_num_seqs.
        local_cpu: Whether LMCache L1 CPU tier is enabled.
        disk_limit_gb: LMCache local-disk limit in GiB units.
        cpu_limit_gb: LMCache local-CPU limit in GiB units.
        prompts: Tokenized prompts for correctness.

    Returns:
        Exact correctness result dictionary.
    """
    lmcache_dir = tempfile.mkdtemp(prefix="lmcache_correctness_", dir=store_dir)
    h_lm = LMCacheHarness(
        lmcache_dir,
        total_bytes,
        model_path,
        gpu_util,
        max_num_seqs,
        local_cpu,
        disk_limit_gb,
        cpu_limit_gb,
    )
    try:
        h_lm.start()
        return run_correctness_system(
            name="LMCache",
            build_llm_fn=h_lm.build_llm,
            prompts=prompts,
            max_num_seqs=max_num_seqs,
            after_cold_fn=h_lm.wait_for_disk_quiescence,
        )
    finally:
        h_lm.stop()


def run_daser_correctness(
    store_dir: str,
    model_path: str,
    gpu_util: float,
    max_num_seqs: int,
    transfer_mode: str,
    l1_bytes: int,
    total_slots: int,
    prompts: list[list[int]],
    require_all_commits: bool,
    require_l2_drain: bool,
    skip_l2: bool = False,
) -> dict[str, Any]:
    """Run DaseR exact correctness in an isolated server/store.

    Args:
        store_dir: Base directory for DaseR benchmark scratch files.
        model_path: HF model path for vLLM.
        gpu_util: vLLM GPU memory utilization.
        max_num_seqs: vLLM max_num_seqs.
        transfer_mode: DaseR transfer backend.
        l1_bytes: DaseR L1 byte capacity.
        total_slots: DaseR L2 slots.
        prompts: Tokenized prompts for correctness.
        require_all_commits: Whether all chunks must commit before warm.
        require_l2_drain: Whether tiered transfer must drain L2 before warm.
        skip_l2: Use volatile L1 memory only and do not create a store file.

    Returns:
        Exact correctness result dictionary with visible-hit counters.
    """
    daser_dir = tempfile.mkdtemp(prefix="daser_correctness_", dir=store_dir)
    socket_dir = tempfile.mkdtemp(prefix="daser_correctness_ipc_")
    h = DaserHarness(
        daser_dir,
        socket_dir,
        total_slots,
        model_path,
        gpu_util,
        max_num_seqs,
        transfer_mode,
        l1_bytes,
        skip_l2=skip_l2,
    )
    try:
        h.start()
        from vllm import SamplingParams  # Third Party
        from vllm.inputs import TokensPrompt  # Third Party

        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
        )
        warm_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]

        logger.info("[DaseR] correctness: building isolated exact-check LLM")
        llm = h.build_llm()
        try:
            logger.info("[DaseR] correctness: cold generate(N=%d)", len(tp_prompts))
            cold_outputs = llm.generate(tp_prompts, params)
            logger.info("[DaseR] correctness: waiting for save completion")
            h.wait_until_committed(
                prompts,
                BLOCK_TOKENS,
                require_all_commits=require_all_commits,
                require_l2_drain=require_l2_drain,
            )
            visible_mask = h.visible_prompt_mask(prompts, "qwen3-8b", BLOCK_TOKENS)
            logger.info("[DaseR] correctness: warm generate(N=%d)", len(tp_prompts))
            warm_outputs = llm.generate(tp_prompts, warm_params)
            return correctness_check_with_visibility(
                "DaseR",
                cold_outputs,
                warm_outputs,
                prompts,
                max_num_seqs,
                visible_mask,
            )
        finally:
            logger.info("[DaseR] correctness: destroying LLM")
            _destroy_llm(llm)
    finally:
        h.stop()


def correctness_check(
    name: str,
    cold_outputs: list,
    warm_outputs: list,
    prompts: list[list[int]],
    max_num_seqs: int,
) -> dict[str, Any]:
    """Compare cold vs warm generated output exactly.

    Args:
        name: System label used in diagnostics.
        cold_outputs: Outputs from the cold timed pass.
        warm_outputs: Outputs from the warm timed pass.
        prompts: Tokenized benchmark inputs in output order.
        max_num_seqs: vLLM admission limit used for mismatch diagnostics.

    Returns:
        Correctness counters. Only exact generated text and token-ID matches
        are accepted.
    """
    mismatches = 0
    mismatch_indices: list[int] = []
    mismatch_details: list[dict[str, Any]] = []
    prompt_alignment_mismatches = 0
    total = len(cold_outputs)
    for i, (c, w) in enumerate(zip(cold_outputs, warm_outputs, strict=False)):
        cold_prompt = list(getattr(c, "prompt_token_ids", prompts[i]))
        warm_prompt = list(getattr(w, "prompt_token_ids", prompts[i]))
        if cold_prompt != warm_prompt or cold_prompt != list(prompts[i]):
            prompt_alignment_mismatches += 1
            if prompt_alignment_mismatches <= 3:
                logger.warning(
                    "[%s] prompt %d alignment differs: input=%d cold=%d warm=%d",
                    name,
                    i,
                    len(prompts[i]),
                    len(cold_prompt),
                    len(warm_prompt),
                )
        if _generated_token_ids(c) == _generated_token_ids(w) and _output_text(
            c
        ) == _output_text(w):
            continue

        mismatches += 1
        mismatch_indices.append(i)
        detail = {
            "index": i,
            "wave": i // max(1, max_num_seqs),
            "position": i % max(1, max_num_seqs),
            "prompt_tokens": len(prompts[i]),
            "cold_token_ids": _generated_token_ids(c),
            "warm_token_ids": _generated_token_ids(w),
            "cold_text": _output_text(c),
            "warm_text": _output_text(w),
        }
        mismatch_details.append(detail)
        if mismatches <= 3:
            logger.warning(
                "[%s] prompt %d (wave=%d pos=%d len=%d): text mismatch cold=%s warm=%s",
                name,
                i,
                detail["wave"],
                detail["position"],
                detail["prompt_tokens"],
                detail["cold_token_ids"],
                detail["warm_token_ids"],
            )
    if mismatches:
        logger.warning(
            "[%s] exact text/token mismatches=%d/%d",
            name,
            mismatches,
            total,
        )
        logger.warning("[%s] mismatch indices: %s", name, mismatch_indices)
    else:
        logger.info(
            "[%s] exact text/token correctness OK (%d requests)",
            name,
            total,
        )
    if prompt_alignment_mismatches:
        logger.warning(
            "[%s] %d/%d prompt alignments mismatched",
            name,
            prompt_alignment_mismatches,
            total,
        )
    return {
        "mismatches": mismatches,
        "total": total,
        "indices": mismatch_indices,
        "mismatch_details": mismatch_details,
        "prompt_alignment_mismatches": prompt_alignment_mismatches,
    }


def _generated_token_ids(output: Any) -> list[int]:
    """Return generated token IDs from a vLLM RequestOutput.

    Args:
        output: vLLM request output.

    Returns:
        Generated token IDs, or an empty list when unavailable.
    """
    if not getattr(output, "outputs", None):
        return []
    return [int(token_id) for token_id in getattr(output.outputs[0], "token_ids", [])]


def _output_text(output: Any) -> str:
    """Return generated text from a vLLM RequestOutput.

    Args:
        output: vLLM request output.

    Returns:
        Generated text, or an empty string when unavailable.
    """
    if not getattr(output, "outputs", None):
        return ""
    return str(getattr(output.outputs[0], "text", ""))


def correctness_check_with_visibility(
    name: str,
    cold_outputs: list,
    warm_outputs: list,
    prompts: list[list[int]],
    max_num_seqs: int,
    visible_mask: list[bool],
) -> dict[str, Any]:
    """Compare cold/warm outputs and split exact mismatches by visible hits.

    Args:
        name: System label, used only for logging.
        cold_outputs: Outputs from the cold timed pass.
        warm_outputs: Outputs from the warm timed pass.
        prompts: Tokenized benchmark inputs in output order.
        max_num_seqs: vLLM admission limit, used only for diagnostics.
        visible_mask: Per-prompt boolean indicating whether the aligned DaseR
            prefix was visible before the warm pass.

    Returns:
        Dict with total and visible-hit mismatch counters.
    """
    result = correctness_check(name, cold_outputs, warm_outputs, prompts, max_num_seqs)
    visible_total = 0
    visible_mismatches = 0
    for cold, warm, visible in zip(
        cold_outputs, warm_outputs, visible_mask, strict=False
    ):
        if not visible:
            continue
        visible_total += 1
        if _generated_token_ids(cold) == _generated_token_ids(warm) and _output_text(
            cold
        ) == _output_text(warm):
            continue
        visible_mismatches += 1
    result["visible_mismatches"] = visible_mismatches
    result["visible_total"] = visible_total
    if visible_total:
        logger.info(
            "[%s] visible-hit correctness: mismatches=%d (%d requests)",
            name,
            visible_mismatches,
            visible_total,
        )
    return result


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _fmt_elapsed(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f} s"


def _fmt_tps(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{v:,.0f}"


def _fmt_count(v: Any) -> str:
    """Format an integer counter metric.

    Args:
        v: Numeric counter value or None.

    Returns:
        Human-readable counter string.
    """
    if v is None:
        return "N/A"
    return str(v)


def build_summary(
    daser: dict[str, Any] | None,
    lmcache: dict[str, Any] | None,
    prompt_tokens: int,
    comparison_mode: str,
) -> dict[str, Any]:
    """Derive tok/s and speedups for the report."""
    summary: dict[str, Any] = {
        "comparison_mode": comparison_mode,
        "prompt_tokens_total": prompt_tokens,
    }
    for key, r in (("daser", daser), ("lmcache", lmcache)):
        if r is None or r.get("skipped"):
            summary[key] = {"skipped": True, "reason": (r or {}).get("reason")}
            continue
        cold = r["cold_elapsed_s"]
        warm = r["warm_elapsed_s"]
        summary[key] = {
            "cold_elapsed_s": cold,
            "warm_elapsed_s": warm,
            "cold_tok_per_s": prompt_tokens / cold if cold > 0 else None,
            "warm_tok_per_s": prompt_tokens / warm if warm > 0 else None,
            "warm_cold_speedup": cold / warm if warm > 0 else None,
            "correctness": r.get("correctness"),
            "backend": r.get("backend"),
            "storage_tier": r.get("storage_tier"),
            "warm_skip_save": r.get("warm_skip_save", False),
            "visible_prompt_count": r.get("visible_prompt_count"),
        }
    d = summary.get("daser", {})
    lm = summary.get("lmcache", {})
    if not d.get("skipped") and not lm.get("skipped"):
        dw = d.get("warm_tok_per_s") or 0.0
        lw = lm.get("warm_tok_per_s") or 0.0
        dc = d.get("cold_tok_per_s") or 0.0
        lc = lm.get("cold_tok_per_s") or 0.0
        summary["warm_tps_ratio_daser_over_lmcache"] = dw / lw if lw > 0 else None
        summary["cold_tps_ratio_daser_over_lmcache"] = dc / lc if lc > 0 else None
        daser_correctness = d.get("correctness") or {}
        lmcache_correctness = lm.get("correctness") or {}
        daser_mismatches = daser_correctness.get("mismatches")
        lmcache_mismatches = lmcache_correctness.get("mismatches")
        if daser_mismatches is not None and lmcache_mismatches is not None:
            delta = int(daser_mismatches) - int(lmcache_mismatches)
            summary["correctness_mismatch_delta_daser_minus_lmcache"] = delta
            summary["correctness_parity_ok"] = delta <= 1
    return summary


def print_report(config: dict[str, Any], summary: dict[str, Any]) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 72)
    print("E2E vLLM Benchmark — DaseR vs LMCache")
    print("=" * 72)
    print(f"Model            : {config['model']}")
    print(f"Comparison mode  : {config['comparison_mode']}")
    label = config.get("dataset", "IMDB reviews")
    print(f"Prompts          : {config['num_prompts']} ({label})")
    print(f"Seed             : {config['seed']}")
    print(f"Prompt tokens    : {summary['prompt_tokens_total']:,}")
    print("Sampling         : temperature=0, max_tokens=1")
    print("Correctness src  : exact generated token IDs and output text")
    print("Correctness      : cold/warm outputs must match exactly")
    print("Correctness rule : DaseR mismatches <= LMCache mismatches + 1")
    prefix_cache = "enabled" if config.get("enable_prefix_caching") else "disabled"
    print(f"Prefix cache     : {prefix_cache}")
    print("-" * 72)
    print(f"{'Metric':<28}{'DaseR':>20}{'LMCache':>20}")
    print("-" * 72)

    d = summary.get("daser", {}) or {}
    lm = summary.get("lmcache", {}) or {}

    def _show(label: str, k: str, fmt: Any) -> None:
        dv = None if d.get("skipped") else d.get(k)
        lv = None if lm.get("skipped") else lm.get(k)
        print(f"{label:<28}{fmt(dv):>20}{fmt(lv):>20}")

    _show("cold elapsed", "cold_elapsed_s", _fmt_elapsed)
    _show("warm elapsed", "warm_elapsed_s", _fmt_elapsed)
    _show("cold tok/s (prompt)", "cold_tok_per_s", _fmt_tps)
    _show("warm tok/s (prompt)", "warm_tok_per_s", _fmt_tps)

    def _correctness_value(system: dict[str, Any], key: str) -> Any:
        correctness = system.get("correctness") or {}
        return correctness.get(key)

    print(
        f"{'exact mismatches':<28}"
        f"{_fmt_count(_correctness_value(d, 'mismatches')):>20}"
        f"{_fmt_count(_correctness_value(lm, 'mismatches')):>20}"
    )
    parity = summary.get("correctness_parity_ok")
    if parity is not None:
        delta = summary.get("correctness_mismatch_delta_daser_minus_lmcache")
        print(f"{'mismatch delta':<28}{_fmt_count(delta):>20}{'limit <= 1':>20}")
        print(f"{'correctness parity':<28}{str(bool(parity)):>20}{'':>20}")

    def _speedup(v: Any) -> str:
        return f"{v:.2f}×" if v is not None else "N/A"

    dv = None if d.get("skipped") else d.get("warm_cold_speedup")
    lv = None if lm.get("skipped") else lm.get("warm_cold_speedup")
    print(f"{'warm/cold speedup':<28}{_speedup(dv):>20}{_speedup(lv):>20}")

    ratio = summary.get("warm_tps_ratio_daser_over_lmcache")
    print("-" * 72)
    if ratio is not None:
        print(f"DaseR warm tok/s / LMCache warm tok/s = {ratio:.2f}×")
    print("=" * 72)


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
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable vLLM prefix caching (off by default).",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")
    selected_gpu_id = apply_gpu_selection(args.gpu_id)
    store_root = os.path.join(args.store_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(store_root, exist_ok=False)
    logger.info("benchmark scratch root: %s", store_root)
    logger.info(
        "selected GPU: %s (CUDA_VISIBLE_DEVICES=%s)",
        selected_gpu_id if selected_gpu_id is not None else "current",
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
    capacity_limits = derive_capacity_limits(store_root, selected_gpu_id)
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
        "daser_l1=%.2fGiB, evict=%s, capped=%s, "
        "max_l1=%.2fGiB, max_l2=%.2fGiB",
        total_bytes / BYTES_PER_GIB,
        sizing.daser_slots,
        sizing.daser_l1_bytes / BYTES_PER_GIB,
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
        "selected_gpu_id": selected_gpu_id,
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
        "enable_prefix_caching": args.enable_prefix_caching,
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
                enable_prefix_caching=args.enable_prefix_caching,
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
            enable_prefix_caching=args.enable_prefix_caching,
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
        write_json_results(args.out, out_obj)
        print(f"\nJSON results written to {args.out}")


if __name__ == "__main__":
    main()
