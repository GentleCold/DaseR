# SPDX-License-Identifier: Apache-2.0

from benchmarks.bench_e2e_daser_vs_lmcache import (
    SLOT_SIZE,
    LMCacheHarness,
    _resolve_cache_sizes,
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
    """The GDS comparison disables CPU tier but keeps LMCache disk buffers."""
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
        assert os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] != "0.000"
        assert os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "0.927"
    finally:
        harness.stop()


def test_gds_sizing_uses_l2_only_and_lmcache_single_ssd() -> None:
    """GDS comparison automatically sizes L2 and disables memory tiers."""
    sizes = _resolve_cache_sizes(
        total_blocks=8,
        transfer_backend="gds",
        evict=False,
    )

    assert sizes.l1_bytes == 0
    assert sizes.l2_bytes > 8 * SLOT_SIZE
    assert sizes.slots_needed * SLOT_SIZE == sizes.l2_bytes
    assert sizes.lmcache_cpu_bytes == 0
    assert sizes.lmcache_disk_bytes == sizes.l2_bytes
    assert sizes.lmcache_mode == "local-disk"


def test_iouring_sizing_without_evict_keeps_all_kv_in_l2() -> None:
    """Non-evict iouring mode sizes L2 > total KV bytes > L1."""
    total_bytes = 8 * SLOT_SIZE
    sizes = _resolve_cache_sizes(
        total_blocks=8,
        transfer_backend="iouring-mem",
        evict=False,
    )

    assert sizes.l2_bytes > total_bytes > sizes.l1_bytes
    assert sizes.lmcache_cpu_bytes == sizes.l1_bytes
    assert sizes.lmcache_disk_bytes == sizes.l2_bytes
    assert sizes.lmcache_mode == "local-cpu-disk"
    assert sizes.kv_to_l1_ratio is not None and sizes.kv_to_l1_ratio > 1.0
    assert sizes.store_to_l1_ratio is not None and sizes.store_to_l1_ratio > 1.0


def test_iouring_evict_sizing_forces_l1_and_l2_eviction() -> None:
    """Evict iouring mode sizes total KV bytes > L2 > L1."""
    total_bytes = 32 * SLOT_SIZE
    sizes = _resolve_cache_sizes(
        total_blocks=32,
        transfer_backend="iouring-mem",
        evict=True,
    )

    assert total_bytes > sizes.l2_bytes > sizes.l1_bytes > 0
    assert sizes.lmcache_cpu_bytes == sizes.l1_bytes
    assert sizes.lmcache_disk_bytes == sizes.l2_bytes
    assert sizes.kv_to_l1_ratio is not None and sizes.kv_to_l1_ratio > 1.0
    assert sizes.store_to_l1_ratio is not None and sizes.store_to_l1_ratio > 1.0
