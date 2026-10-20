# SPDX-License-Identifier: Apache-2.0

"""Validate and measure CPU encoding against the unchanged online GPU codec."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import threading
import time
from typing import Any

import numpy as np
import torch

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    decode_slot,
    default_online_codebooks,
)
from daser.compression.cpu import CpuEncodedSlot, CpuSlotEncoder
from daser.ops.compressed_kv import FusedCompressedKVDecoder, FusedOnlineKVPacker
from daser.ops.kv_layout import allocate_cross_layer_kv_cache, physical_kv_view


def _cache(geometry: CompressedStoreGeometry, blocks: int) -> torch.Tensor:
    return allocate_cross_layer_kv_cache(
        blocks,
        geometry.num_layers,
        geometry.block_tokens,
        geometry.num_kv_heads,
        geometry.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )


def _digest(data: Any) -> str:
    return hashlib.sha256(data).hexdigest()


def _payload_digest(data: Any, mode: SlotMode) -> str:
    # The GPU writer does not define trailing alignment bytes. Compare all
    # header and plane-record bytes, but exclude only that unused padding.
    view = memoryview(data).cast("B")
    if mode == SlotMode.COMPRESSED:
        planes = struct.unpack_from("<I", view, 36)[0]
        offset, length = struct.unpack_from("<II", view, 120 + (planes - 1) * 48 + 12)
        view = view[: offset + length]
    return _digest(view)


def _process_rss_bytes() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )


def _distribution(samples: list[float], raw_bytes: int) -> dict[str, float]:
    median = float(np.median(samples))
    return {
        "median_ms": median,
        "p95_ms": float(np.percentile(samples, 95)),
        "raw_GB_per_s": raw_bytes / (median * 1e6),
    }


def _verify_reference(
    payload: bytes,
    raw: bytes,
    geometry: CompressedStoreGeometry,
    books: bytes,
    slot_id: int,
    mode: SlotMode,
) -> None:
    # The unchanged Python oracle is deliberately slow. Independent spawned
    # processes amortize validation without forking an initialized CUDA context.
    # These processes exit before any steady-state measurement starts.
    restored = decode_slot(
        payload,
        mode=mode,
        slot_id=slot_id,
        geometry=geometry,
        codebooks=books,
        verify_hash=False,
    )
    if restored != raw:
        raise AssertionError(f"CPU/reference mismatch in slot {slot_id}")


def _validate(
    corpus: np.ndarray,
    source: torch.Tensor,
    geometry: CompressedStoreGeometry,
    books: bytes,
) -> dict[str, Any]:
    """Validate every captured slot using both existing decoder implementations."""
    batch = min(16, len(corpus))
    stream = torch.cuda.current_stream()
    encoder = CpuSlotEncoder(geometry, books)
    destination = _cache(geometry, batch)
    staging = torch.empty(batch * geometry.slot_size, dtype=torch.uint8, device="cuda")
    decoder = FusedCompressedKVDecoder(
        kv_cache=destination,
        codebooks=books,
        tile_scalars=256,
        ring_depth=1,
        max_slots_per_buffer=batch,
    )
    packer = FusedOnlineKVPacker(
        kv_cache=source,
        codebooks=books,
        tile_scalars=256,
        max_slots_per_buffer=batch,
    )
    host = torch.empty(batch * geometry.slot_size, dtype=torch.uint8, pin_memory=True)
    host_array = host.numpy()
    records: dict[str, list[dict[str, Any]]] = {"cpu": [], "gpu": []}
    references = ProcessPoolExecutor(
        max_workers=min(4, batch), mp_context=multiprocessing.get_context("spawn")
    )
    try:
        for start in range(0, len(corpus), batch):
            count = min(batch, len(corpus) - start)
            offsets, modes = [], []
            checks = []
            cursor = 0
            for i in range(count):
                slot_id = start + i
                result = encoder.encode_into(
                    corpus[slot_id],
                    host_array[cursor : cursor + geometry.slot_size],
                    slot_id=slot_id,
                )
                payload = host_array[cursor : cursor + result.stored_length]
                checks.append(
                    references.submit(
                        _verify_reference,
                        payload.tobytes(),
                        corpus[slot_id].tobytes(),
                        geometry,
                        books,
                        slot_id,
                        result.mode,
                    )
                )
                offsets.append(cursor)
                modes.append(int(result.mode))
                records["cpu"].append(
                    {
                        "mode": int(result.mode),
                        "stored_length": result.stored_length,
                        "payload_sha256": _payload_digest(payload, result.mode),
                    }
                )
                cursor += result.stored_length
            staging[:cursor].copy_(host[:cursor], non_blocking=True)
            decoder.decode(
                staging=staging,
                staging_offsets=offsets,
                block_ids=list(range(count)),
                modes=modes,
                buffer_index=0,
                stream=stream,
            )
            stream.synchronize()
            restored = destination[:count].contiguous().view(torch.uint16).cpu().numpy()
            if not np.array_equal(restored, corpus[start : start + count]):
                raise AssertionError("CPU/GPU decoder mismatch")
            packed = packer.pack_into(
                staging=staging,
                block_ids=list(range(start, start + count)),
                logical_slots=list(range(start, start + count)),
                slot_stride=geometry.slot_size,
                stream=stream,
            )
            decoder.decode(
                staging=staging,
                staging_offsets=[p.source_offset for p in packed],
                block_ids=list(range(count)),
                modes=[int(p.mode) for p in packed],
                buffer_index=0,
                stream=stream,
            )
            stream.synchronize()
            restored = destination[:count].contiguous().view(torch.uint16).cpu().numpy()
            if not np.array_equal(restored, corpus[start : start + count]):
                raise AssertionError("GPU encoder/GPU decoder mismatch")
            used = sum(p.stored_length for p in packed)
            host[:used].copy_(staging[:used], non_blocking=True)
            stream.synchronize()
            records["gpu"].extend(
                {
                    "mode": int(p.mode),
                    "stored_length": p.stored_length,
                    "payload_sha256": _payload_digest(
                        host_array[p.source_offset : p.source_offset + p.stored_length],
                        p.mode,
                    ),
                }
                for p in packed
            )
            # Verify the full CPU path's layout contract, outside steady state.
            physical = physical_kv_view(source)[start : start + count]
            copied = physical.view(torch.uint16).cpu().numpy()
            canonical = copied.transpose(0, 1, 4, 2, 3, 5).copy()
            if not np.array_equal(canonical, corpus[start : start + count]):
                raise AssertionError("physical-to-canonical CPU layout mismatch")
            print(
                f"both GPU decoders verified slots {start}:{start + count}", flush=True
            )
            for i, check in enumerate(checks):
                check.result()
                print(
                    f"reference verified slot {start + i + 1}/{len(corpus)}", flush=True
                )
    finally:
        references.shutdown(wait=True, cancel_futures=True)
        encoder.close()
    return {"all_byte_comparisons_passed": True, "slot_count": len(corpus), **records}


class CpuBatch:
    """Own a fixed pool; each worker owns one encoder and optional layout buffer."""

    def __init__(
        self,
        corpus: np.ndarray,
        source: torch.Tensor,
        geometry: CompressedStoreGeometry,
        books: bytes,
        batch: int,
        threads: int,
        full_path: bool,
    ) -> None:
        """Allocate a fixed pool and buffers; call before measurement.

        Args:
            corpus (np.ndarray): Resident canonical BF16 bits.
            source (torch.Tensor): GPU logical view of the same slots.
            geometry (CompressedStoreGeometry): Slot geometry.
            books (bytes): Shared immutable codebooks.
            batch (int): Number of slots per call.
            threads (int): Requested maximum parallelism.
            full_path (bool): Include D2H and CPU layout conversion.

        Returns:
            None.

        Async/thread-safety:
            Synchronous startup; run and close belong to the calling thread.
        """
        self.corpus, self.source, self.geometry = corpus, source, geometry
        self.batch, self.active_threads = batch, min(batch, threads)
        self.full_path = full_path
        self.local = threading.local()
        self.encoders: list[CpuSlotEncoder] = []
        self.outputs = [bytearray(geometry.slot_size) for _ in range(batch)]
        self.host = None
        self.host_array = None
        if full_path:
            self.host = torch.empty_like(
                physical_kv_view(source)[:batch],
                device="cpu",
                pin_memory=True,
            )
            self.host_array = self.host.view(torch.uint16).numpy()
        self.stream = torch.cuda.current_stream()

        def initialize() -> None:
            self.local.encoder = CpuSlotEncoder(geometry, books)
            self.encoders.append(self.local.encoder)
            if full_path:
                self.local.canonical = np.empty_like(corpus[0])

        self.pool = ThreadPoolExecutor(
            max_workers=self.active_threads,
            initializer=initialize,
        )

    def run(self, start: int) -> list[CpuEncodedSlot]:
        """Encode batch at int start, returning metadata; caller serializes runs."""
        if self.full_path:
            self.host.copy_(
                physical_kv_view(self.source)[start : start + self.batch],
                non_blocking=True,
            )
            self.stream.synchronize()
        return list(self.pool.map(self._slot, ((start, i) for i in range(self.batch))))

    def memory(self) -> dict[str, int]:
        """Return accounted host capacities in bytes; read after pool warmup."""
        return {
            "host_output_capacity_bytes": self.batch * self.geometry.slot_size,
            "host_pinned_input_bytes": (
                self.batch * self.geometry.slot_size if self.full_path else 0
            ),
            "host_canonical_workspace_bytes": (
                self.active_threads * self.geometry.slot_size if self.full_path else 0
            ),
            "native_workspace_per_instance_bytes": self.encoders[0].workspace_bytes,
            "native_workspace_total_bytes": sum(
                e.workspace_bytes for e in self.encoders
            ),
            "encoder_instances": len(self.encoders),
        }

    def close(self) -> None:
        """Join all pool threads and release encoders; synchronous teardown only."""
        self.pool.shutdown(wait=True)
        # The initializer captures this instance; break the executor reference
        # cycle now so per-row buffers do not wait for cyclic GC to be freed.
        del self.pool
        for encoder in self.encoders:
            encoder.close()

    def _slot(self, item: tuple[int, int]) -> CpuEncodedSlot:
        start, index = item
        if self.full_path:
            # [layer, token, head, K/V, dim] -> [layer, K/V, token, head, dim].
            # The preallocated destination is worker-owned; no nested pools.
            np.copyto(
                self.local.canonical, self.host_array[index].transpose(0, 3, 1, 2, 4)
            )
            raw = self.local.canonical
        else:
            raw = self.corpus[start + index]
        return self.local.encoder.encode_into(
            raw, self.outputs[index], slot_id=start + index
        )


class GpuBatch:
    """Existing GPU packing plus a single contiguous, valid-length pinned D2H."""

    def __init__(
        self,
        source: torch.Tensor,
        geometry: CompressedStoreGeometry,
        books: bytes,
        batch: int,
    ) -> None:
        """Allocate the existing GPU packer and transfer buffers.

        Args:
            source (torch.Tensor): GPU logical KV view.
            geometry (CompressedStoreGeometry): Full-slot geometry.
            books (bytes): Immutable default codebooks.
            batch (int): Maximum slots per call.

        Returns:
            None.

        Async/thread-safety:
            Synchronous startup; all runs use the current CUDA stream serially.
        """
        self.batch, self.geometry = batch, geometry
        self.stream = torch.cuda.current_stream()
        before = torch.cuda.memory_allocated()
        self.packer = FusedOnlineKVPacker(
            kv_cache=source,
            codebooks=books,
            tile_scalars=256,
            max_slots_per_buffer=batch,
        )
        self.packer_device_bytes = torch.cuda.memory_allocated() - before
        self.staging = torch.empty(
            batch * geometry.slot_size, dtype=torch.uint8, device="cuda"
        )
        self.host = torch.empty(
            batch * geometry.slot_size, dtype=torch.uint8, pin_memory=True
        )

    def run(self, start: int) -> list[Any]:
        """Pack int start batch and return spans after D2H; serialized by caller."""
        ids = list(range(start, start + self.batch))
        plans = self.packer.pack_into(
            staging=self.staging,
            block_ids=ids,
            logical_slots=ids,
            slot_stride=self.geometry.slot_size,
            stream=self.stream,
        )
        # Master already compacts records. Transfer only the used prefix; this
        # includes Python span planning, packer's metadata synchronization, and
        # completion of the final D2H copy in the measured wall-clock interval.
        length = sum(p.stored_length for p in plans)
        self.host[:length].copy_(self.staging[:length], non_blocking=True)
        self.stream.synchronize()
        return plans

    def memory(self) -> dict[str, int]:
        """Return host/device capacities in bytes; no mutation or synchronization."""
        return {
            "host_pinned_output_bytes": self.host.nbytes,
            "device_output_capacity_bytes": self.staging.nbytes,
            "device_packer_workspace_bytes": self.packer_device_bytes,
        }


def _measure(
    runner: CpuBatch | GpuBatch,
    *,
    args: argparse.Namespace,
    geometry: CompressedStoreGeometry,
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    rounds, all_samples = [], []
    for round_id in range(args.rounds):
        for warmup in range(args.warmups):
            runner.run((warmup * runner.batch) % geometry.num_slots)
        samples = []
        for iteration in range(args.iterations):
            start = (
                (round_id * args.iterations + iteration) * runner.batch
            ) % geometry.num_slots
            begin = time.perf_counter_ns()
            result = runner.run(start)
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            samples.append(elapsed)
            # Validation stays outside timed intervals. Every measured batch
            # must preserve the mode/length observed during byte verification.
            for i, record in enumerate(result):
                if (
                    int(record.mode) != expected[start + i]["mode"]
                    or record.stored_length != expected[start + i]["stored_length"]
                ):
                    raise AssertionError("measured record differs from verified input")
        # Hash only the final completed batch of each round, after timing.
        # This also validates the actual CPU full-path conversion and the GPU
        # transfer destination, not just the standalone decoder helpers.
        for i, record in enumerate(result):
            if isinstance(runner, CpuBatch):
                payload = memoryview(runner.outputs[i])[: record.stored_length]
            else:
                begin = record.source_offset
                payload = runner.host[begin : begin + record.stored_length].numpy()
            if (
                _payload_digest(payload, record.mode)
                != expected[start + i]["payload_sha256"]
            ):
                raise AssertionError("timed path output differs from verified payload")
        rounds.append(
            {
                "round": round_id + 1,
                "samples_ms": samples,
                **_distribution(samples, runner.batch * geometry.slot_size),
            }
        )
        all_samples.extend(samples)
        print(f"round {round_id + 1}: median {np.median(samples):.3f} ms", flush=True)
    stored = sum(record["stored_length"] for record in expected)
    raw = len(expected) * geometry.slot_size
    return {
        **_distribution(all_samples, runner.batch * geometry.slot_size),
        "rounds": rounds,
        "memory": runner.memory(),
        "compression_ratio_raw_over_stored": raw / stored,
        "stored_fraction": stored / raw,
        "raw_fallback_fraction": sum(p["mode"] == 0 for p in expected) / len(expected),
        "process_rss_bytes": _process_rss_bytes(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "device_allocated_bytes": torch.cuda.memory_allocated(),
        "device_reserved_bytes": torch.cuda.memory_reserved(),
    }


def main() -> None:
    """Run the offline validation/matrix and write JSON; synchronous CLI only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            *args.threads, *args.batch_sizes, args.warmups, args.iterations, args.rounds
        )
        < 1
    ):
        parser.error("all matrix counts must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    # Fully resident copy prevents mmap faults and storage IO during timing.
    corpus = np.array(np.load(args.capture / "slots.npy", mmap_mode="r"), copy=True)
    if _digest(corpus) != manifest["data_sha256"]:
        raise ValueError("capture digest mismatch")
    if any(len(corpus) % batch for batch in args.batch_sizes):
        parser.error("batch sizes must divide the number of captured slots")
    geometry = CompressedStoreGeometry(
        num_slots=len(corpus),
        slot_size=corpus[0].nbytes,
        num_layers=corpus.shape[1],
        block_tokens=corpus.shape[3],
        num_kv_heads=corpus.shape[4],
        head_dim=corpus.shape[5],
        tile_scalars=256,
    )
    books = default_online_codebooks(geometry)
    source = _cache(geometry, len(corpus))
    source.copy_(torch.from_numpy(corpus).view(torch.bfloat16))
    torch.cuda.synchronize()
    root = Path(__file__).resolve().parents[2]
    source_files = [
        "daser/compression/cpu_codec.cpp",
        "daser/compression/cpu.py",
        "daser/compression/Makefile",
        "daser/ops/compressed_kv.py",
        "daser/ops/tilelang_compressed_kv.py",
        "benchmarks/compression/bench_cpu_codec.py",
        "benchmarks/compression/capture_cpu_codec.py",
    ]
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "source_sha256": {p: _digest((root / p).read_bytes()) for p in source_files},
        "geometry": asdict(geometry),
        "capture_sha256": manifest["data_sha256"],
        "codebooks_hex": books.hex(),
        "codebooks_sha256": _digest(books),
        "protocol": {
            "warmups_per_round": args.warmups,
            "iterations_per_round": args.iterations,
            "rounds": args.rounds,
            "threads": args.threads,
            "batches": args.batch_sizes,
            "timer": "perf_counter_ns, includes pool dispatch and required CUDA sync",
            "batch_schedule": "(round * iterations + iteration) * batch modulo slots",
            "thread_pool": (
                "fixed min(requested_threads, batch) workers; no nested pool"
            ),
            "validation_hashes_timed": False,
            "reference_validation_processes": 4,
            "payload_digest_scope": (
                "RAW all bytes; COMPRESSED header and plane records, "
                "excluding trailing alignment padding"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "numa_policy": subprocess.check_output(["numactl", "--show"], text=True),
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_version": torch.version.cuda,
            "gpu_inventory": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.free,utilization.gpu",
                    "--format=csv",
                ],
                text=True,
            ),
            "topology": subprocess.check_output(
                ["nvidia-smi", "topo", "-m"], text=True
            ),
            "compiler": subprocess.check_output(
                ["c++", "--version"], text=True
            ).splitlines()[0],
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "tilelang", "numpy", "transformers")
            },
        },
        "shared_resident_corpus_bytes": {
            "host": corpus.nbytes,
            "device": source.nbytes,
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    report["validation"] = _validate(corpus, source, geometry, books)
    save()
    # Round-level samples retain variation. Each batch has one GPU baseline;
    # GPU CPU-thread counts are irrelevant and must not multiply its trials.
    for batch in args.batch_sizes:
        for full_path in (False, True):
            for threads in args.threads:
                path = "cpu_full" if full_path else "cpu_encode"
                print(f"{path} threads={threads} batch={batch}", flush=True)
                runner = CpuBatch(
                    corpus, source, geometry, books, batch, threads, full_path
                )
                try:
                    row = _measure(
                        runner,
                        args=args,
                        geometry=geometry,
                        expected=report["validation"]["cpu"],
                    )
                    report["rows"].append(
                        {
                            "path": path,
                            "threads": threads,
                            "active_threads": runner.active_threads,
                            "batch": batch,
                            **row,
                        }
                    )
                finally:
                    runner.close()
                del runner
                save()
        print(f"gpu_full batch={batch}", flush=True)
        gpu = GpuBatch(source, geometry, books, batch)
        row = _measure(
            gpu,
            args=args,
            geometry=geometry,
            expected=report["validation"]["gpu"],
        )
        report["rows"].append(
            {"path": "gpu_full", "threads": None, "batch": batch, **row}
        )
        del gpu
        save()
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"completed {len(report['rows'])} matrix rows", flush=True)


if __name__ == "__main__":
    main()
