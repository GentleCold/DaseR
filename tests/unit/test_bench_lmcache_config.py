# SPDX-License-Identifier: Apache-2.0

# First Party
from benchmarks.bench_e2e_daser_vs_lmcache import LMCacheHarness


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
