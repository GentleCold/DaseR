# SPDX-License-Identifier: Apache-2.0

"""CLI for producing model-bound offline codec calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator

from daser.compression import CompressedStoreGeometry
from daser.compression.calibration import (
    calibrate_codebooks,
    write_calibration_artifact,
)


def _sha256_file(path: Path) -> str:
    """Hash an input file without loading it into memory.

    Args:
        path: File whose bytes are hashed.

    Returns:
        Lowercase SHA-256 digest.

    Thread-safety:
        Synchronous read-only filesystem operation.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slot_stream(path: Path, slot_size: int) -> Iterator[bytes]:
    """Yield complete raw slots from a concatenated activation file.

    Args:
        path: Concatenated slot-major BF16 input file.
        slot_size: Bytes per slot from the model geometry.

    Yields:
        Complete raw slot payloads.

    Raises:
        ValueError: If the input ends part-way through a slot.

    Thread-safety:
        Synchronous single-consumer file iterator.
    """
    with path.open("rb") as source:
        while True:
            payload = source.read(slot_size)
            if not payload:
                return
            if len(payload) != slot_size:
                raise ValueError("input file ends with an incomplete raw slot")
            yield payload


def _parse_args() -> argparse.Namespace:
    """Parse calibration CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fit model-bound DaseR codec codebooks from raw BF16 slots"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--num-slots", type=int, default=1)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--block-tokens", type=int, default=128)
    parser.add_argument("--tile-scalars", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    """Fit and write one offline calibration artifact."""
    args = _parse_args()
    slot_size = (
        args.num_layers * 2 * args.block_tokens * args.num_kv_heads * args.head_dim * 2
    )
    geometry = CompressedStoreGeometry(
        num_slots=args.num_slots,
        slot_size=slot_size,
        block_tokens=args.block_tokens,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        tile_scalars=args.tile_scalars,
    )
    artifact = calibrate_codebooks(
        _slot_stream(args.input, geometry.slot_size),
        model_id=args.model_id,
        geometry=geometry,
        source_sha256=_sha256_file(args.input),
    )
    write_calibration_artifact(artifact, args.output_dir)


if __name__ == "__main__":
    main()
