# SPDX-License-Identifier: Apache-2.0
"""Compare reference-codec wire bytes and CPU latency with a local git revision."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)


def _load_reference(revision: str) -> ModuleType:
    """Load only the reference codec from a trusted local revision into memory."""
    source = subprocess.check_output(
        ["git", "show", f"{revision}:daser/compression/codec.py"], text=True
    )
    module = ModuleType("_daser_audit_reference_codec")
    sys.modules[module.__name__] = module
    exec(compile(source, "reference_codec.py", "exec"), module.__dict__)
    return module


def exercise(reference: ModuleType, cases: int, seed: int) -> dict[str, Any]:
    """Compare complete encoded bytes and decoded output over random geometry.

    Args:
        reference: In-memory codec module from a trusted local revision.
        cases: Number of random slots (int, at least two).
        seed: Initial deterministic NumPy seed (int).

    Returns:
        Counts and paired CPU latency medians, excluding RAW decode timings.

    Async/thread-safety:
        Synchronous CPU benchmark with local arrays; no GPU or store files.
    """
    encode_before, encode_after, decode_before, decode_after = [], [], [], []
    modes: dict[str, int] = {}
    verified_bytes = 0
    for case in range(cases):
        rng = np.random.default_rng(seed + case)
        layers = int(rng.choice([1, 2, 8, 42]))
        dim = int(rng.choice([32, 64, 128]))
        geometry = CompressedStoreGeometry(
            num_slots=1,
            slot_size=layers * 2 * 128 * dim * 2,
            block_tokens=128,
            num_layers=layers,
            num_kv_heads=1,
            head_dim=dim,
            tile_scalars=int(rng.choice([128, 256, 1024])),
        )
        raw = rng.integers(0, 256, geometry.slot_size, dtype=np.uint8)
        if case % 3:
            raw[1::2] = rng.choice(
                np.array([63, 191, 62, 190, 64, 192, 61, 189], dtype=np.uint8),
                geometry.slot_size // 2,
            )
            raw[1::1994] = 0x7E
        source = raw.tobytes()
        codebooks = default_online_codebooks(geometry)
        kwargs = {"slot_id": 0, "geometry": geometry, "codebooks": codebooks}
        started = time.perf_counter_ns()
        old = reference.encode_slot(source, **kwargs)
        encode_before.append((time.perf_counter_ns() - started) / 1000)
        started = time.perf_counter_ns()
        new = encode_slot(source, **kwargs)
        encode_after.append((time.perf_counter_ns() - started) / 1000)
        assert (new.mode, new.payload) == (old.mode, old.payload), (seed, case)
        # Decode the original bytes with the new decoder and vice versa;
        # paired packing mistakes cannot hide behind successful round trips.
        started = time.perf_counter_ns()
        old_raw = reference.decode_slot(new.payload, mode=new.mode, **kwargs)
        old_us = (time.perf_counter_ns() - started) / 1000
        started = time.perf_counter_ns()
        new_raw = decode_slot(old.payload, mode=old.mode, **kwargs)
        new_us = (time.perf_counter_ns() - started) / 1000
        assert old_raw == new_raw == source, (seed, case)
        if new.mode is SlotMode.COMPRESSED:
            decode_before.append(old_us)
            decode_after.append(new_us)
        modes[new.mode.name] = modes.get(new.mode.name, 0) + 1
        verified_bytes += len(source)
    before_encode = statistics.median(encode_before)
    after_encode = statistics.median(encode_after)
    before_decode = statistics.median(decode_before)
    after_decode = statistics.median(decode_after)
    return {
        "seed": seed,
        "cases": cases,
        "verified_raw_bytes": verified_bytes,
        "modes": modes,
        "encode_before_median_us": before_encode,
        "encode_after_median_us": after_encode,
        "encode_speedup": before_encode / after_encode,
        "compressed_decode_before_median_us": before_decode,
        "compressed_decode_after_median_us": after_decode,
        "compressed_decode_speedup": before_decode / after_decode,
    }


def main() -> None:
    """Run a bounded local comparison and write portable JSON evidence.

    Returns:
        None; any wire-format or decode mismatch raises AssertionError.

    Async/thread-safety:
        Startup-only subprocess and report IO surround the CPU benchmark.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-rev", required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3019)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cases < 2:
        parser.error("--cases must be at least two")
    reference = _load_reference(args.reference_rev)
    try:
        result = exercise(reference, args.cases, args.seed)
    finally:
        sys.modules.pop(reference.__name__, None)
    result["reference_revision"] = args.reference_rev
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
