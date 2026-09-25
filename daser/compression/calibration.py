# SPDX-License-Identifier: Apache-2.0

"""Offline codebook calibration for the strict-lossless BF16 codec."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from daser.compression.format import (
    CODEBOOK_ENTRIES,
    CompressedStoreGeometry,
    digest_bytes,
)


@dataclass(frozen=True)
class CalibrationArtifact:
    """Model-bound plane-major codebooks produced by offline calibration.

    Args:
        model_id: Stable model identity supplied by the caller.
        geometry: KV geometry used to interpret every input slot.
        codebooks: Plane-major 15-entry codebook bytes.
        sample_slots: Number of raw slots used to fit the tables.
        source_sha256: Optional digest of the input activation file.

    Thread-safety:
        Immutable after construction; safe to share between threads.
    """

    model_id: str
    geometry: CompressedStoreGeometry
    codebooks: bytes
    sample_slots: int
    source_sha256: str | None = None

    def metadata(self) -> dict[str, object]:
        """Return reproducibility metadata for the calibrated codebooks.

        Returns:
            JSON-serializable model identity, geometry, layout, sample count,
            and codebook digest metadata.

        Thread-safety:
            Pure read-only serialization; safe to call concurrently.
        """
        return {
            "format_version": 1,
            "model_id": self.model_id,
            "geometry": {
                "num_slots": self.geometry.num_slots,
                "slot_size": self.geometry.slot_size,
                "block_tokens": self.geometry.block_tokens,
                "num_layers": self.geometry.num_layers,
                "num_kv_heads": self.geometry.num_kv_heads,
                "head_dim": self.geometry.head_dim,
                "dtype_bytes": self.geometry.dtype_bytes,
                "tile_scalars": self.geometry.tile_scalars,
            },
            "plane_count": self.geometry.plane_count,
            "codebook_entries": CODEBOOK_ENTRIES,
            "layout": "; ".join(
                (
                    "plane-major",
                    "plane=layer*2+kv",
                    "K=kv0,V=kv1",
                    "entry14=raw-sentinel",
                )
            ),
            "sample_slots": self.sample_slots,
            "codebook_sha256": digest_bytes(self.codebooks).hex(),
            "source_sha256": self.source_sha256,
        }


def calibrate_codebooks(
    raw_slots: Iterable[bytes | bytearray | memoryview],
    *,
    model_id: str,
    geometry: CompressedStoreGeometry,
    source_sha256: str | None = None,
) -> CalibrationArtifact:
    """Fit one deterministic high-byte palette for every KV plane.

    Args:
        raw_slots: Iterable of contiguous slot-major BF16 bytes laid out as
            ``[layer, K/V, token, head, dim]``. Each item must equal
            ``geometry.slot_size`` bytes.
        model_id: Stable model identity bound into the output metadata.
        geometry: Exact KV geometry for the input slots.
        source_sha256: Optional digest of the activation source file.

    Returns:
        CalibrationArtifact containing plane-major 15-entry codebooks.

    Raises:
        ValueError: If the model identity, input slot size, or sample count is
            invalid.

    Thread-safety:
        CPU-bound synchronous fitting over caller-owned input; do not share a
        mutable input iterator between threads.
    """
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    counts = np.zeros((geometry.plane_count, 256), dtype=np.int64)
    sample_slots = 0
    for raw_slot in raw_slots:
        payload = bytes(raw_slot)
        if len(payload) != geometry.slot_size:
            raise ValueError("raw slot length does not match geometry")
        values = np.frombuffer(payload, dtype=np.uint8).reshape(
            geometry.plane_count, geometry.plane_scalars, 2
        )[:, :, 1]
        for plane in range(geometry.plane_count):
            counts[plane] += np.bincount(values[plane], minlength=256)
        sample_slots += 1
    if sample_slots == 0:
        raise ValueError("at least one raw slot is required for calibration")

    tables = np.empty((geometry.plane_count, CODEBOOK_ENTRIES), dtype=np.uint8)
    for plane in range(geometry.plane_count):
        order = sorted(
            range(256),
            key=lambda value: (-int(counts[plane, value]), value),
        )
        primary = [value for value in order if counts[plane, value] > 0][
            : CODEBOOK_ENTRIES - 1
        ]
        primary.extend(value for value in order if value not in primary)
        primary = primary[: CODEBOOK_ENTRIES - 1]
        sentinel = next(value for value in range(256) if value not in primary)
        tables[plane] = np.asarray(
            primary + [sentinel],
            dtype=np.uint8,
        )
    return CalibrationArtifact(
        model_id=model_id,
        geometry=geometry,
        codebooks=tables.tobytes(),
        sample_slots=sample_slots,
        source_sha256=source_sha256,
    )


def write_calibration_artifact(
    artifact: CalibrationArtifact,
    output_dir: str | Path,
) -> None:
    """Write a calibration artifact without retaining raw activation bytes.

    Args:
        artifact: Model-bound codebooks and reproducibility metadata.
        output_dir: Directory receiving ``metadata.json`` and
            ``codebooks.bin``.

    Thread-safety:
        Synchronous filesystem writes; callers must serialize concurrent
        writes to the same output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "codebooks.bin").write_bytes(artifact.codebooks)
    (output_path / "metadata.json").write_text(
        json.dumps(artifact.metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CalibrationArtifact",
    "calibrate_codebooks",
    "write_calibration_artifact",
]
