# SPDX-License-Identifier: Apache-2.0

"""Offline builder for immutable read-only compressed DaseR stores."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from daser.compression.codec import build_compressed_store, calibrate_codebooks
from daser.compression.format import CompressedStoreGeometry, digest_bytes
from daser.config import model_geometry_from_path
from daser.logging import init_logger

logger = init_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse offline compressed-store builder arguments.

    Args:
        argv: Optional argument vector excluding the program name.

    Returns:
        Parsed conversion arguments.

    Async/thread-safety:
        Startup-only parser with no shared mutable state.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-store", required=True, type=Path)
    parser.add_argument("--raw-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--block-tokens", required=True, type=int)
    parser.add_argument(
        "--calibration-slot",
        required=True,
        action="append",
        type=int,
        dest="calibration_slots",
        help="Logical raw slot used for codebook calibration; repeat per slot.",
    )
    return parser.parse_args(argv)


def _iter_slots(
    store_path: Path,
    slot_size: int,
    slot_ids: Iterable[int],
) -> Iterable[bytes]:
    """Yield selected fixed-envelope slots from an immutable raw store.

    Args:
        store_path: Raw production DaseR store snapshot.
        slot_size: Exact bytes in one logical KV slot.
        slot_ids: Logical slot IDs selected only for calibration.

    Yields:
        Complete raw slot byte payloads in the requested order.

    Raises:
        ValueError: If a selected slot is truncated.

    Async/thread-safety:
        Synchronous offline file reader. The source snapshot must not change
        while iteration is in progress.
    """
    with store_path.open("rb") as handle:
        for slot_id in slot_ids:
            handle.seek(slot_id * slot_size)
            payload = handle.read(slot_size)
            if len(payload) != slot_size:
                raise ValueError(f"calibration slot {slot_id} is truncated")
            yield payload


def _copy_control_index(source: Path, destination: Path) -> None:
    """Atomically copy the persisted DaseR control index.

    Args:
        source: Immutable source ``daser.index`` snapshot.
        destination: Output ``daser.index`` path.

    Async/thread-safety:
        Synchronous offline copy. Callers must serialize writers.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_store(args: argparse.Namespace) -> None:
    """Build one compressed-read-only store from a raw production snapshot.

    Args:
        args: Parsed arguments returned by :func:`_parse_args`.

    Raises:
        ValueError: If paths, calibration slots, model geometry, or source
            sizes violate the immutable format contract.

    Async/thread-safety:
        CPU- and IO-bound offline operation. It must not run on the DaseR
        server event loop or while the source snapshot is being written.
    """
    raw_store = args.raw_store.resolve()
    raw_index = args.raw_index.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    if not raw_store.is_file() or not raw_index.is_file():
        raise ValueError("raw store and control index must both exist")
    if args.block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    model = model_geometry_from_path(str(model_path))
    if model.dtype_name != "bfloat16":
        raise ValueError("compressed format version 1 requires BF16 model KV")
    slot_size = model.slot_size_for_block_tokens(args.block_tokens)
    raw_bytes = raw_store.stat().st_size
    if raw_bytes == 0 or raw_bytes % slot_size:
        raise ValueError("raw store size is not a whole number of model KV slots")
    num_slots = raw_bytes // slot_size
    calibration_slots = list(dict.fromkeys(args.calibration_slots))
    if not calibration_slots or any(
        slot_id < 0 or slot_id >= num_slots for slot_id in calibration_slots
    ):
        raise ValueError("calibration slot is outside the raw store")
    geometry = CompressedStoreGeometry(
        num_slots=num_slots,
        slot_size=slot_size,
        block_tokens=args.block_tokens,
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype_bytes=model.dtype_bytes,
    )
    codebooks = calibrate_codebooks(
        _iter_slots(raw_store, slot_size, calibration_slots),
        geometry,
    )
    config_bytes = (model_path / "config.json").read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    compressed_store = output_dir / "daser.store"
    compressed_index = output_dir / "daser.compressed.index"
    control_index = output_dir / "daser.index"
    output_files = (compressed_store, compressed_index, control_index)
    if any(path.exists() for path in output_files):
        raise ValueError("output directory already contains a DaseR store file")
    index = build_compressed_store(
        raw_store,
        compressed_store,
        compressed_index,
        geometry=geometry,
        model_hash=digest_bytes(config_bytes),
        codebooks=codebooks,
    )
    _copy_control_index(raw_index, control_index)
    stored_bytes = sum(ref.stored_length for ref in index.resolve_slots(0, num_slots))
    logger.info(
        "[INDEX] built compressed-read-only store slots=%d raw_bytes=%d "
        "transfer_bytes=%d transfer_ratio=%.6f calibration_slots=%d",
        num_slots,
        raw_bytes,
        stored_bytes,
        stored_bytes / raw_bytes,
        len(calibration_slots),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the offline compressed-store builder.

    Args:
        argv: Optional CLI arguments excluding the program name.

    Returns:
        Process exit code zero after a complete conversion.

    Async/thread-safety:
        Synchronous command entry point intended for a dedicated process.
    """
    build_store(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
