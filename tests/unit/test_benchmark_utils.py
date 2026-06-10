# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark utility helpers."""

# Third Party
import pytest

# First Party
from benchmarks.bench_common import (
    BenchmarkCapacityLimits,
    DaserHarness,
    GPUInfo,
    choose_gpu_id,
    derive_benchmark_sizing,
    write_json_results,
)
from daser.connector.ipc_client import IPCClientSync


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
    """No-evict sizing is capped by machine-derived L1/L2 limits."""
    sizing = derive_benchmark_sizing(
        total_blocks=100,
        max_prompt_blocks=8,
        slot_size=1024,
        mode="iouring-mem-vs-lmcache-local-ssd-mem",
        evict=False,
        capacity_limits=BenchmarkCapacityLimits(
            max_l1_bytes=80 * 1024,
            max_l2_bytes=120 * 1024,
            memory_available_bytes=1_000_000,
            disk_available_bytes=1_000_000,
        ),
    )

    assert sizing.daser_slots == 120
    assert sizing.daser_l2_bytes == 120 * 1024
    assert sizing.daser_l1_bytes == 80 * 1024
    assert sizing.capacity_capped


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


def test_daser_harness_skip_l2_does_not_create_store_file(tmp_path) -> None:
    """Benchmark harness should propagate skip_l2 to the in-process server."""
    harness = DaserHarness(
        store_dir=str(tmp_path / "store"),
        socket_dir=str(tmp_path / "socket"),
        total_slots=2,
        model_path="/model",
        gpu_util=0.1,
        max_num_seqs=1,
        transfer_mode="iouring",
        l1_bytes=4096,
        skip_l2=True,
    )
    try:
        harness.start()
        client = IPCClientSync(harness.socket_path)
        try:
            runtime_config = client.get_runtime_config()
        finally:
            client.close()
        assert not (tmp_path / "store" / "daser.store").exists()
        assert runtime_config["skip_l2"] is True
        assert runtime_config["store_path"] == ""
    finally:
        harness.stop()


def test_write_json_results_stringifies_non_json_objects(tmp_path) -> None:
    """Benchmark JSON output should handle retained RequestOutput-like objects."""

    class NonJson:
        def __str__(self) -> str:
            return "request-output"

    out = tmp_path / "result.json"

    write_json_results(out, {"outputs": [NonJson()]})

    assert out.read_text(encoding="utf-8") == (
        '{\n  "outputs": [\n    "request-output"\n  ]\n}'
    )
