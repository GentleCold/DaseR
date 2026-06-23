# SPDX-License-Identifier: Apache-2.0
"""Unit checks for standalone bandwidth microbench artifacts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BANDWIDTH_DIR = REPO_ROOT / "benchmarks" / "bandwidth"
SOURCE = BANDWIDTH_DIR / "kv_cache_copy_bandwidth.cu"
MAKEFILE = BANDWIDTH_DIR / "Makefile"
README = BANDWIDTH_DIR / "README.md"


def test_kv_cache_bandwidth_microbench_files_exist() -> None:
    """Bandwidth microbench should provide source, build, and usage docs."""
    assert SOURCE.exists()
    assert MAKEFILE.exists()
    assert README.exists()


def test_kv_cache_bandwidth_microbench_compares_expected_paths() -> None:
    """Microbench source should compare direct scatter and staging restore paths."""
    source = SOURCE.read_text(encoding="utf-8")

    assert "--layers" in source
    assert "--blocks" in source
    assert "--block-stride" in source
    assert "--pipeline-chunks" in source
    assert "--requests" in source
    assert "direct_h2d_scatter" in source
    assert "staging_h2d_then_d2d_scatter" in source
    assert "staging_h2d_then_kernel_scatter" in source
    assert "pipelined_staging_kernel_scatter" in source
    assert "cross_request_pipelined_staging_kernel_scatter" in source
    assert "h2d_to_staging_only" in source
    assert "kernel_scatter_only" in source
    assert "mapped_host_kernel_scatter" in source
    assert "cudaHostAllocMapped" in source
    assert "__global__" in source
    assert "cudaMemcpyHostToDevice" in source
    assert "cudaMemcpyDeviceToDevice" in source


def test_kv_cache_bandwidth_makefile_builds_cuda_source() -> None:
    """Makefile should compile the standalone CUDA C++ source with nvcc."""
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert "nvcc" in makefile
    assert "kv_cache_copy_bandwidth.cu" in makefile
    assert "kv_cache_copy_bandwidth" in makefile
