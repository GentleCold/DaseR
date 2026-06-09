# SPDX-License-Identifier: Apache-2.0
"""Benchmark cache sizing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil

from benchmarks.utils.constants import (
    BYTES_PER_GIB,
    COMPARISON_IOURING_MEM,
)

EVICT_L2_FRACTION: float = 0.95
EVICT_L1_FRACTION: float = 0.9


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
        lmcache_disk_gb: LMCache local disk limit in integer GiB units when
            supported by the selected adapter. ``None`` means the adapter has
            no explicit capacity knob.
        lmcache_cpu_gb: LMCache local CPU limit in integer GiB units.
        capacity_capped: Whether machine limits capped the requested sizes.

    Thread-safety:
        Immutable value object; safe to share between threads.
    """

    daser_slots: int
    daser_l2_bytes: int
    daser_l1_bytes: int
    lmcache_disk_gb: int | None
    lmcache_cpu_gb: int
    capacity_capped: bool


def derive_capacity_limits(
    store_dir: str | Path,
    disk_fraction: float = 0.8,
    host_mem_fraction: float = 0.25,
    max_l1_gib: float = 256.0,
    max_l2_gib: float = 512.0,
) -> BenchmarkCapacityLimits:
    """Derive benchmark capacity ceilings from current machine state.

    Args:
        store_dir: Benchmark store directory.
        disk_fraction: Fraction of free disk space allowed for L2.
        host_mem_fraction: Fraction of available host memory allowed for L1.
        max_l1_gib: Absolute L1 ceiling.
        max_l2_gib: Absolute L2 ceiling.

    Returns:
        Capacity limits for sizing.

    Thread-safety:
        Reads system state and keeps no shared mutable state.
    """
    path = Path(store_dir)
    path.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(path).free
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
        evict: When True, choose capacities that force eviction.
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

    max_l2_bytes = align_down_gib(
        capacity_limits.max_l2_bytes,
        required_bytes=required_l2_bytes,
    )
    max_l2_blocks = max(1, max_l2_bytes // slot_size)
    l2_blocks = min(desired_l2_blocks, max_l2_blocks)
    if not evict and l2_blocks < total_blocks:
        l2_blocks = max(max_prompt_blocks, l2_blocks)

    desired_l2_bytes = desired_l2_blocks * slot_size
    requested_l2_bytes = align_down_gib(
        l2_blocks * slot_size,
        required_bytes=required_l2_bytes,
    )
    daser_l2_bytes = (requested_l2_bytes // slot_size) * slot_size
    l2_blocks = max(1, daser_l2_bytes // slot_size)
    capacity_capped = (
        max_l2_blocks < desired_l2_blocks and daser_l2_bytes < desired_l2_bytes
    )

    if mode == COMPARISON_IOURING_MEM:
        required_l1_bytes = max(1, max_prompt_blocks) * slot_size
        if capacity_limits.max_l1_bytes < required_l1_bytes:
            raise ValueError(
                "benchmark L1 capacity cap cannot fit the largest prompt "
                f"({capacity_limits.max_l1_bytes} < {required_l1_bytes} bytes)"
            )
        workload_bytes = total_blocks * slot_size
        if evict:
            desired_l1_bytes = max(
                required_l1_bytes,
                math.floor(workload_bytes * EVICT_L1_FRACTION),
            )
        else:
            desired_l1_bytes = workload_bytes
        requested_l1_bytes = align_down_gib(
            min(desired_l1_bytes, capacity_limits.max_l1_bytes),
            required_bytes=required_l1_bytes,
        )
        daser_l1_bytes = min(
            (requested_l1_bytes // slot_size) * slot_size,
            daser_l2_bytes,
        )
        capacity_capped = capacity_capped or (
            capacity_limits.max_l1_bytes < desired_l1_bytes
            and daser_l1_bytes < desired_l1_bytes
        )
    else:
        daser_l1_bytes = 0

    return BenchmarkSizing(
        daser_slots=l2_blocks,
        daser_l2_bytes=daser_l2_bytes,
        daser_l1_bytes=daser_l1_bytes,
        lmcache_disk_gb=None,
        lmcache_cpu_gb=bytes_to_lmcache_gb(daser_l1_bytes),
        capacity_capped=capacity_capped,
    )


def align_down_gib(nbytes: int, required_bytes: int = 0) -> int:
    """Align a capacity to integer GiB without dropping below a requirement.

    Args:
        nbytes: Desired capacity.
        required_bytes: Minimum capacity that must still fit.

    Returns:
        GiB-aligned capacity in bytes.

    Thread-safety:
        Pure function.
    """
    if nbytes <= 0:
        return 0
    aligned = (nbytes // BYTES_PER_GIB) * BYTES_PER_GIB
    if aligned >= required_bytes and aligned > 0:
        return aligned
    return math.ceil(max(required_bytes, 1) / BYTES_PER_GIB) * BYTES_PER_GIB


def bytes_to_lmcache_gb(nbytes: int) -> int:
    """Convert bytes to LMCache's GiB-based config value.

    Args:
        nbytes: Capacity in bytes.

    Returns:
        Size value for LMCache GB config knobs.

    Thread-safety:
        Pure function.
    """
    if nbytes <= 0:
        return 0
    return math.ceil(nbytes / BYTES_PER_GIB)


def parse_size_bytes(value: str) -> int:
    """Parse a human-readable byte size such as ``300gib``.

    Args:
        value: Size string.

    Returns:
        Parsed byte count.

    Thread-safety:
        Pure function.
    """
    units = {
        "": 1,
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    stripped = value.strip().lower()
    digits = ""
    unit = ""
    for ch in stripped:
        if ch.isdigit():
            if unit:
                raise ValueError(f"invalid size: {value}")
            digits += ch
        else:
            unit += ch
    if not digits or unit not in units:
        raise ValueError(f"invalid size: {value}")
    return int(digits) * units[unit]


def _host_available_bytes() -> int:
    """Return currently available host memory bytes.

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
    return int(os.environ.get("DASER_BENCH_HOST_MEM_FALLBACK", 8 * BYTES_PER_GIB))
