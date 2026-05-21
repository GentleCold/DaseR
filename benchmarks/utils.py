# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for DaseR benchmark scripts."""

# Standard
import csv
from dataclasses import dataclass
import math
import os
import random
import shutil
import subprocess
from typing import Any

BYTES_PER_GIB: int = 1024**3
EVICT_L2_FRACTION: float = 0.95
EVICT_L1_FRACTION: float = 0.9
LMCACHE_LOCAL_SSD_STAGING_GB: float = 0.5
COMPARISON_GDS = "gds-vs-lmcache-local-ssd"
COMPARISON_IOURING_MEM = "iouring-mem-vs-lmcache-local-ssd-mem"


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
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


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
        desired_l1_blocks = max(1, math.floor(desired_l2_blocks * EVICT_L1_FRACTION))
    else:
        desired_l2_blocks = max(1, math.ceil(total_blocks * 1.5))
        desired_l1_blocks = max(1, math.ceil(total_blocks * 1.25))

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
        max_l1_blocks = max(1, capacity_limits.max_l1_bytes // slot_size)
        l1_blocks = min(desired_l1_blocks, max_l1_blocks)
        desired_l1_bytes = desired_l1_blocks * slot_size
        daser_l1_bytes = l1_blocks * slot_size
        capacity_capped = capacity_capped or daser_l1_bytes < desired_l1_bytes
    else:
        daser_l1_bytes = 0

    return BenchmarkSizing(
        daser_slots=l2_blocks,
        daser_l2_bytes=daser_l2_bytes,
        daser_l1_bytes=daser_l1_bytes,
        lmcache_disk_gb=bytes_to_lmcache_gb(daser_l2_bytes),
        lmcache_cpu_gb=(
            bytes_to_lmcache_gb(daser_l1_bytes)
            if mode == COMPARISON_IOURING_MEM
            else LMCACHE_LOCAL_SSD_STAGING_GB
        ),
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
