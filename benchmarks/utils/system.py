# SPDX-License-Identifier: Apache-2.0
"""System inspection helpers for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess


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


def query_gpus() -> list[GPUInfo]:
    """Return GPU memory snapshots from nvidia-smi.

    Returns:
        List of GPUInfo entries, or an empty list when unavailable.

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
        CUDA device ID string to use, or None when the environment should be
        left unchanged.

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
        Mutates process environment and should run during startup.
    """
    selected = choose_gpu_id(
        query_gpus(), requested, os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if selected is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected
