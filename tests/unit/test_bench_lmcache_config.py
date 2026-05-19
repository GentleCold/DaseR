# SPDX-License-Identifier: Apache-2.0

# First Party
import pytest

from benchmarks.bench_e2e_daser_vs_lmcache import (
    PRESSURE_STORE_TO_L1_RATIO,
    SLOT_SIZE,
    LMCacheHarness,
    _resolve_pressure_sizes,
)


def test_lmcache_harness_configures_ssd_and_local_cpu(monkeypatch, tmp_path) -> None:
    """The LMCache comparison can enable SSD plus a CPU tier."""
    monkeypatch.delenv("LMCACHE_LOCAL_CPU", raising=False)
    monkeypatch.delenv("LMCACHE_LOCAL_DISK", raising=False)
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_DISK_SIZE", raising=False)
    harness = LMCacheHarness(
        tmpdir=str(tmp_path),
        total_bytes=123,
        max_local_cpu_bytes=16_000_000_000,
        max_local_disk_bytes=927_000_000,
        model_path="/model",
        gpu_util=0.4,
        max_num_seqs=16,
        local_cpu=True,
    )

    harness.start()
    try:
        import os

        assert os.environ["LMCACHE_LOCAL_CPU"] == "True"
        assert os.environ["LMCACHE_LOCAL_DISK"] == f"file://{tmp_path}/"
        assert os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "16.000"
        assert os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "0.927"
    finally:
        harness.stop()
    assert "LMCACHE_LOCAL_CPU" not in os.environ
    assert "LMCACHE_LOCAL_DISK" not in os.environ
    assert "LMCACHE_MAX_LOCAL_CPU_SIZE" not in os.environ
    assert "LMCACHE_MAX_LOCAL_DISK_SIZE" not in os.environ


def test_lmcache_harness_configures_single_ssd(monkeypatch, tmp_path) -> None:
    """The GDS comparison disables LMCache CPU but still enables SSD."""
    monkeypatch.delenv("LMCACHE_LOCAL_CPU", raising=False)
    monkeypatch.delenv("LMCACHE_LOCAL_DISK", raising=False)
    harness = LMCacheHarness(
        tmpdir=str(tmp_path),
        total_bytes=123,
        max_local_cpu_bytes=0,
        max_local_disk_bytes=927_000_000,
        model_path="/model",
        gpu_util=0.4,
        max_num_seqs=16,
        local_cpu=False,
    )

    harness.start()
    try:
        import os

        assert os.environ["LMCACHE_LOCAL_CPU"] == "False"
        assert os.environ["LMCACHE_LOCAL_DISK"] == f"file://{tmp_path}/"
        assert os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "0.000"
        assert os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "0.927"
    finally:
        harness.stop()


def test_pressure_sizing_requires_kv_larger_than_l1() -> None:
    """Pressure mode rejects workloads that cannot evict the memory tier."""
    with pytest.raises(ValueError, match="total KV bytes to exceed L1"):
        _resolve_pressure_sizes(
            total_blocks=2,
            l1_cache_size=10 * SLOT_SIZE,
            requested_store_size=None,
            requested_lmcache_disk_size=None,
            pressure_eviction=True,
        )


def test_pressure_sizing_makes_store_larger_than_l1() -> None:
    """Pressure mode aligns LMCache disk to a store between L1 and total KV."""
    l1_cache_size = 4 * SLOT_SIZE
    slots_needed, store_bytes, lmcache_disk_size, kv_to_l1, store_to_l1 = (
        _resolve_pressure_sizes(
            total_blocks=9,
            l1_cache_size=l1_cache_size,
            requested_store_size=None,
            requested_lmcache_disk_size=None,
            pressure_eviction=True,
        )
    )

    assert slots_needed * SLOT_SIZE == store_bytes
    assert store_bytes <= PRESSURE_STORE_TO_L1_RATIO * l1_cache_size
    assert store_bytes > l1_cache_size
    assert 9 * SLOT_SIZE > store_bytes
    assert lmcache_disk_size == store_bytes
    assert kv_to_l1 is not None and kv_to_l1 > 1.0
    assert store_to_l1 is not None and store_to_l1 > 1.0


def test_pressure_sizing_rejects_disk_smaller_than_l1() -> None:
    """Pressure mode requires SSD capacity to exceed memory capacity."""
    with pytest.raises(ValueError, match="SSD capacity to exceed L1"):
        _resolve_pressure_sizes(
            total_blocks=9,
            l1_cache_size=4 * SLOT_SIZE,
            requested_store_size=None,
            requested_lmcache_disk_size=2 * SLOT_SIZE,
            pressure_eviction=True,
        )


def test_pressure_sizing_rejects_store_larger_than_kv() -> None:
    """Pressure mode requires KV volume to exceed SSD/store capacity."""
    with pytest.raises(ValueError, match="total KV bytes to exceed SSD/store"):
        _resolve_pressure_sizes(
            total_blocks=9,
            l1_cache_size=4 * SLOT_SIZE,
            requested_store_size=12 * SLOT_SIZE,
            requested_lmcache_disk_size=None,
            pressure_eviction=True,
        )


def test_pressure_sizing_rejects_lmcache_disk_larger_than_kv() -> None:
    """Pressure mode keeps the LMCache SSD tier under eviction pressure too."""
    with pytest.raises(ValueError, match="total KV bytes to exceed LMCache SSD"):
        _resolve_pressure_sizes(
            total_blocks=9,
            l1_cache_size=4 * SLOT_SIZE,
            requested_store_size=6 * SLOT_SIZE,
            requested_lmcache_disk_size=12 * SLOT_SIZE,
            pressure_eviction=True,
        )
