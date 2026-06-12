# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark utility helpers."""

# Third Party
import pytest

# First Party
from benchmarks.utils.sizing import (
    BenchmarkCapacityLimits,
    align_down_gib,
    bytes_to_lmcache_gb,
    bytes_to_lmcache_gb_for_effective_l1,
    derive_benchmark_sizing,
    derive_capacity_limits,
    format_capacity,
)
from benchmarks.utils.system import GPUInfo, choose_gpu_id


def test_choose_gpu_id_auto_selects_largest_free_memory() -> None:
    """Auto GPU selection picks the device with the most free memory."""
    gpus = [
        GPUInfo(index=0, total_mb=80_000, used_mb=60_000, free_mb=20_000),
        GPUInfo(index=1, total_mb=24_000, used_mb=1_000, free_mb=23_000),
        GPUInfo(index=2, total_mb=80_000, used_mb=10_000, free_mb=70_000),
    ]

    assert choose_gpu_id(gpus, requested="auto", current_visible="0") == "2"


def test_choose_gpu_id_current_preserves_environment() -> None:
    """The current mode keeps an explicit CUDA_VISIBLE_DEVICES value."""
    gpus = [
        GPUInfo(index=0, total_mb=80_000, used_mb=60_000, free_mb=20_000),
        GPUInfo(index=1, total_mb=24_000, used_mb=1_000, free_mb=23_000),
    ]

    assert choose_gpu_id(gpus, requested="current", current_visible="1") == "1"


def test_derive_benchmark_sizing_caps_noevict_capacity() -> None:
    """No-evict sizing is capped and aligned to integer GiB."""
    sizing = derive_benchmark_sizing(
        total_blocks=1000,
        max_prompt_blocks=8,
        slot_size=1024 * 1024,
        mode="iouring-mem-vs-lmcache-local-ssd-mem",
        evict=False,
        capacity_limits=BenchmarkCapacityLimits(
            max_l1_bytes=80 * 1024**3 + 123,
            max_l2_bytes=120 * 1024**3 + 123,
            memory_available_bytes=1_000_000_000_000,
            disk_available_bytes=1_000_000_000_000,
        ),
    )

    assert sizing.daser_l2_bytes == 2 * 1024**3
    assert sizing.daser_l1_bytes == 2 * 1024**3
    assert sizing.lmcache_disk_gb is None
    assert sizing.lmcache_cpu_gb == 2
    assert not sizing.capacity_capped


def test_derive_benchmark_sizing_keeps_l1_within_slot_aligned_l2() -> None:
    """Small no-evict runs keep DaseR L1 no larger than aligned L2."""
    gib = 1024**3
    slot_size = 2_359_296

    sizing = derive_benchmark_sizing(
        total_blocks=298,
        max_prompt_blocks=35,
        slot_size=slot_size,
        mode="iouring-mem-vs-lmcache-local-ssd-mem",
        evict=False,
        capacity_limits=BenchmarkCapacityLimits(
            max_l1_bytes=80 * gib,
            max_l2_bytes=120 * gib,
            memory_available_bytes=1_000_000_000_000,
            disk_available_bytes=1_000_000_000_000,
        ),
    )

    assert sizing.daser_l2_bytes == (gib // slot_size) * slot_size
    assert sizing.daser_l1_bytes <= sizing.daser_l2_bytes
    assert sizing.lmcache_cpu_gb == 1


def test_derive_benchmark_sizing_adds_noevict_l1_headroom() -> None:
    """No-evict L1 sizing keeps headroom above the workload."""
    gib = 1024**3

    sizing = derive_benchmark_sizing(
        total_blocks=2048,
        max_prompt_blocks=8,
        slot_size=1 * 1024**2,
        mode="iouring-mem-vs-lmcache-local-ssd-mem",
        evict=False,
        capacity_limits=BenchmarkCapacityLimits(
            max_l1_bytes=80 * gib,
            max_l2_bytes=120 * gib,
            memory_available_bytes=1_000_000_000_000,
            disk_available_bytes=1_000_000_000_000,
        ),
    )

    assert sizing.daser_l1_bytes == 3 * gib
    assert sizing.daser_l2_bytes == 3 * gib
    assert sizing.lmcache_cpu_gb == 3


def test_derive_benchmark_sizing_rejects_capped_noevict_l1() -> None:
    """No-evict runs fail fast when L1 cannot hold the workload headroom."""
    gib = 1024**3

    with pytest.raises(ValueError, match="no-evict L1 capacity"):
        derive_benchmark_sizing(
            total_blocks=4096,
            max_prompt_blocks=8,
            slot_size=1 * 1024**2,
            mode="iouring-mem-vs-lmcache-local-ssd-mem",
            evict=False,
            capacity_limits=BenchmarkCapacityLimits(
                max_l1_bytes=4 * gib,
                max_l2_bytes=16 * gib,
                memory_available_bytes=1_000_000_000_000,
                disk_available_bytes=1_000_000_000_000,
            ),
        )


def test_derive_capacity_limits_uses_host_memory_not_80_gib_cap(
    tmp_path, monkeypatch
) -> None:
    """Benchmark auto sizing derives L1 from system memory, not an 80 GiB cap."""
    monkeypatch.setattr(
        "benchmarks.utils.sizing._host_available_bytes",
        lambda: 1024 * 1024**3,
    )

    limits = derive_capacity_limits(tmp_path)

    assert limits.max_l1_bytes == 256 * 1024**3


def test_derive_benchmark_sizing_evict_keeps_l2_full_and_l1_partial() -> None:
    """Evict runs keep the full workload in L2 while forcing L1 eviction."""
    slot_size = 2_359_296
    total_blocks = 322
    max_prompt_blocks = 36

    sizing = derive_benchmark_sizing(
        total_blocks=total_blocks,
        max_prompt_blocks=max_prompt_blocks,
        slot_size=slot_size,
        mode="iouring-mem-vs-lmcache-local-ssd-mem",
        evict=True,
        capacity_limits=BenchmarkCapacityLimits(
            max_l1_bytes=80 * 1024**3,
            max_l2_bytes=120 * 1024**3,
            memory_available_bytes=1_000_000_000_000,
            disk_available_bytes=1_000_000_000_000,
        ),
    )

    assert sizing.daser_l2_bytes // slot_size >= total_blocks
    assert max_prompt_blocks <= sizing.daser_l1_bytes // slot_size < total_blocks
    assert sizing.daser_l1_bytes < sizing.daser_l2_bytes


def test_align_down_gib_preserves_required_capacity() -> None:
    """GiB alignment never rounds below the largest required object."""
    gib = 1024**3

    assert align_down_gib(3 * gib + 123, required_bytes=2 * gib + 1) == 3 * gib
    assert align_down_gib(3 * gib - 1, required_bytes=3 * gib - 1) == 3 * gib


def test_format_capacity_uses_mib_below_one_gib() -> None:
    """Human-readable capacities use MiB for sub-GiB values."""
    assert format_capacity(512 * 1024**2) == "512.00 MiB"
    assert format_capacity(2 * 1024**3) == "2.00 GiB"


def test_bytes_to_lmcache_gb_rounds_nonzero_capacity_up() -> None:
    """LMCache integer-GiB CLI values do not floor sub-GiB runs to zero."""
    assert bytes_to_lmcache_gb(0) == 0
    assert bytes_to_lmcache_gb(1) == 1
    assert bytes_to_lmcache_gb(1_073_479_680) == 1


def test_bytes_to_lmcache_gb_for_effective_l1_accounts_for_watermark() -> None:
    """Evict runs configure LMCache so its 80% watermark matches DaseR L1."""
    gib = 1024**3

    assert bytes_to_lmcache_gb_for_effective_l1(0) == 0
    assert bytes_to_lmcache_gb_for_effective_l1(1) == 1
    assert bytes_to_lmcache_gb_for_effective_l1(2 * gib) == 3


def test_derive_benchmark_sizing_rejects_impossible_capacity() -> None:
    """Sizing fails clearly when one prompt cannot fit in the capped store."""
    with pytest.raises(ValueError, match="largest prompt"):
        derive_benchmark_sizing(
            total_blocks=100,
            max_prompt_blocks=8,
            slot_size=1024,
            mode="gds-vs-lmcache-local-ssd",
            evict=False,
            capacity_limits=BenchmarkCapacityLimits(
                max_l1_bytes=0,
                max_l2_bytes=7 * 1024,
                memory_available_bytes=1_000_000,
                disk_available_bytes=1_000_000,
            ),
        )
