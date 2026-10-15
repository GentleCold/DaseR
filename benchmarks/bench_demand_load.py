# SPDX-License-Identifier: Apache-2.0
"""Measure real demand reads through NVMe, pinned L1 and CUDA destinations.

Run each repository variant in a separate process. Setup, L1 priming and
byte-for-byte validation are outside the timed region; every cold iteration
opens a fresh transfer layer over the same fully initialized O_DIRECT store.
No model server, synthetic IO completion, or global page-cache flush is used.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

ALIGNMENT = 4096
CASES = ("same", "overlap", "disjoint", "mixed", "warm", "small-l1")


def parse_size(value: str) -> int:
    """Parse ``value`` (str) into aligned positive bytes (int).

    Pure argument validation; safe to call from any thread.
    """
    match = re.fullmatch(r"(\d+)\s*(b|kib|mib|gib|k|m|g)?", value.strip().lower())
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}")
    units = {
        "b": 1,
        "k": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gib": 1024**3,
    }
    size = int(match[1]) * units.get(match[2], 1)
    if size <= 0 or size % ALIGNMENT:
        raise argparse.ArgumentTypeError("sizes must be positive multiples of 4096")
    return size


def parse_args() -> argparse.Namespace:
    """Return validated CLI arguments as argparse.Namespace.

    Reads process arguments on the main thread without starting IO workers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--expect-commit", default=None)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA-visible ordinal; select hardware with CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--sizes", default="4KiB,256KiB,4MiB,16MiB")
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--cases", default=",".join(CASES[:-1]))
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l1-bytes", type=parse_size, default=None)
    parser.add_argument("--max-device-bytes", type=parse_size, default=2 << 30)
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum seconds per concurrent request wave",
    )
    parser.add_argument(
        "--native-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count real NativeIOUring completions through public methods",
    )
    args = parser.parse_args()
    try:
        args.sizes = [parse_size(item) for item in args.sizes.split(",")]
        args.concurrency = [int(item) for item in args.concurrency.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    args.cases = args.cases.split(",")
    if not args.cases or any(case not in CASES for case in args.cases):
        parser.error(f"--cases must select from {','.join(CASES)}")
    if min(args.concurrency) <= 0 or args.repeat <= 0 or args.warmup < 0:
        parser.error("concurrency/repeat must be positive and warmup nonnegative")
    if args.io_workers <= 0 or args.timeout <= 0:
        parser.error("io-workers and timeout must be positive")
    if max(args.sizes) >= 1 << 30:
        parser.error("each request must be smaller than 1 GiB")
    if max(args.sizes) * max(args.concurrency) > args.max_device_bytes:
        parser.error("largest wave exceeds --max-device-bytes")
    if args.l1_bytes is not None and args.l1_bytes < max(args.sizes):
        parser.error("--l1-bytes must accommodate at least one largest request")
    return args


def source_info(repo: Path, expected_commit: str | None) -> tuple[Any, dict[str, Any]]:
    """Import the requested repository and return its class and identity.

    Args:
        repo (Path): checkout containing the production daser package.
        expected_commit (str | None): optional expected Git commit prefix.
    Returns:
        Production transfer class and JSON-compatible source metadata.
    Async/thread-safety:
        Call once on the main thread before constructing transfer objects.
    """
    repo = repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    module = importlib.import_module("daser.transfer.iouring.layer")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(repo / "daser"):
        raise RuntimeError(f"wrong production source imported: {module_path}")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected_commit is not None and not commit.startswith(expected_commit):
        raise RuntimeError(f"expected commit {expected_commit}, imported {commit}")
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--", "daser/transfer/"]
    )
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--", "daser/"]
    )
    return module.TieredIOUringTransferLayer, {
        "commit": commit,
        "dirty": bool(dirty),
        "module": str(module_path.relative_to(repo)),
        "transfer_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "layer_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def io_read_bytes() -> int | None:
    """Return this process's storage-byte count (int), or None if unavailable.

    Reads kernel counters synchronously without changing IO state.
    """
    try:
        fields = dict(
            line.split(":", 1)
            for line in Path("/proc/self/io").read_text().splitlines()
        )
        return int(fields["read_bytes"])
    except (OSError, KeyError, ValueError):
        return None


def block_read_bytes(stat_path: Path) -> int | None:
    """Read ``stat_path`` (Path), returning device-read bytes (int | None).

    This synchronous observation includes other users' IO and mutates no state.
    """
    try:
        return int(stat_path.read_text().split()[2]) * 512
    except (OSError, IndexError, ValueError):
        return None


def counter_delta(after: int | None, before: int | None) -> int | None:
    """Subtract ``before`` from ``after`` (both int | None).

    Returns a nonnegative int or None. Pure and safe for concurrent callers.
    """
    return None if after is None or before is None else max(0, after - before)


def write_result(path: Path, result: dict[str, Any]) -> None:
    """Write ``result`` (dict) atomically to ``path`` (Path); return None.

    Synchronous main-thread IO, always outside the measured request interval.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(samples: list[dict[str, Any]], numpy: Any) -> dict[str, Any]:
    """Summarize ``samples`` (list[dict]) using the ``numpy`` module (Any).

    Returns a dict of latency and throughput metrics. Pure CPU work on the
    main thread; callers retain the unmodified raw samples separately.
    """
    latencies = [
        latency for sample in samples for latency in sample["request_latency_ms"]
    ]
    elapsed_s = sum(sample["elapsed_ms"] for sample in samples) / 1000
    return {
        "sample_count": len(samples),
        "request_count": len(latencies),
        "p50_ms": float(numpy.percentile(latencies, 50)),
        "p95_ms": float(numpy.percentile(latencies, 95)),
        "wave_p50_ms": float(
            numpy.percentile([sample["elapsed_ms"] for sample in samples], 50)
        ),
        "wave_p95_ms": float(
            numpy.percentile([sample["elapsed_ms"] for sample in samples], 95)
        ),
        "throughput_gib_s": sum(sample["requested_bytes"] for sample in samples)
        / (1 << 30)
        / elapsed_s,
        "l2_reads_total": sum(sample["l2_reads"] for sample in samples),
        "native_read_extents_total": sum(
            sample["native_read_extents"] for sample in samples
        ),
        "native_read_bytes_total": sum(
            sample["native_read_bytes"] for sample in samples
        ),
        "process_read_bytes_total": sum(
            sample["process_read_bytes"] or 0 for sample in samples
        ),
        "correctness": all(sample["correctness"] for sample in samples),
    }


class NativeReadCounter:
    """Observe real public io_uring read calls without replacing their work.

    Args:
        enabled (bool): whether to install lightweight counting wrappers.
    Async/thread-safety:
        Call install/restore with no active IO. Completed reads update counters
        under a thread lock; the kernel methods and callbacks run unchanged.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.calls = 0
        self.extents = 0
        self.nbytes = 0
        self.lock = threading.Lock()
        self.originals: list[tuple[Any, str, Any]] = []

    def install(self) -> None:
        """Install wrappers before IO; returns None and requires exclusive setup."""
        if not self.enabled:
            return
        from daser.transfer.iouring.native import NativeIOUring

        original_single = NativeIOUring.read_into
        original_batch = NativeIOUring.read_batch_into
        counter = self

        def counted_single(
            ring: Any, fd: int, file_offset: int, dst: memoryview
        ) -> int:
            count = original_single(ring, fd, file_offset, dst)
            with counter.lock:
                counter.calls += 1
                counter.extents += 1
                counter.nbytes += count
            return count

        def counted_batch(
            ring: Any,
            fd: int,
            reads: Sequence[tuple[int, memoryview]],
            on_complete: Callable[[int, int], None] | None = None,
        ) -> int:
            count = original_batch(ring, fd, reads, on_complete)
            with counter.lock:
                counter.calls += 1
                counter.extents += len(reads)
                counter.nbytes += count
            return count

        self.originals = [
            (NativeIOUring, "read_into", original_single),
            (NativeIOUring, "read_batch_into", original_batch),
        ]
        NativeIOUring.read_into = counted_single
        NativeIOUring.read_batch_into = counted_batch

    def snapshot(self) -> tuple[int, int, int]:
        """Return (calls, extents, bytes) as tuple[int, int, int] under the lock."""
        with self.lock:
            return self.calls, self.extents, self.nbytes

    def restore(self) -> None:
        """Restore original methods after all IO has drained; returns None."""
        for owner, name, method in self.originals:
            setattr(owner, name, method)
        self.originals.clear()


class CudaDestination:
    """Own a real CUDA array and copy stream, matching transfer destination APIs.

    Args:
        cupy (Any): imported CuPy module using the selected visible device.
        nbytes (int): destination byte capacity.
    Async/thread-safety:
        One request owns this array and stream until its load and verification
        finish. No tensor or CUDA IPC behavior is simulated.
    """

    def __init__(self, cupy: Any, nbytes: int) -> None:
        self.array = cupy.empty(nbytes, dtype=cupy.uint8)
        self.stream = cupy.cuda.Stream(non_blocking=True)

    @property
    def copy_stream_ptr(self) -> int:
        """Return the real stream pointer (int), immutable during requests."""
        return int(self.stream.ptr)

    def __getitem__(self, item: Any) -> Any:
        """Return the real CuPy view (Any) for ``item`` (Any), without syncing."""
        return self.array[item]


async def run_wave(
    layer_type: Any,
    cupy: Any,
    numpy: Any,
    path: Path,
    payload: Any,
    case: str,
    size: int,
    concurrency: int,
    l1_bytes: int,
    args: argparse.Namespace,
    counter: NativeReadCounter,
    device_stat: Path,
) -> dict[str, Any]:
    """Measure one real concurrent load wave with a fresh L1 cache.

    Args:
        layer_type/cupy/numpy (Any): actual production class and array modules.
        path (Path): initialized store; payload (Any): matching CPU uint8 array.
        case (str), size/concurrency/l1_bytes (int): workload geometry.
        args (Namespace): IO worker count and per-wave timeout configuration.
        counter (NativeReadCounter): observer for real kernel read completions.
        device_stat (Path): Linux block-device statistics for the scratch disk.
    Returns:
        JSON-compatible timings, read counters and byte-exact correctness.
    Async/thread-safety:
        Runs on one event loop. IO executor threads and CUDA streams finish
        before arrays, pinned memory, and the store descriptor are released.
    """
    if case == "overlap":
        offsets = [index * (size // 2) for index in range(concurrency)]
    elif case in ("disjoint", "small-l1"):
        offsets = [index * size for index in range(concurrency)]
    else:
        offsets = [0] * concurrency
    unique_bytes = max(offsets) + size
    layer = layer_type(
        str(path),
        l1_bytes=l1_bytes,
        l2_bytes=int(payload.nbytes),
        io_workers=args.io_workers,
        bip_enabled=False,
    )
    destinations: list[CudaDestination] = []
    tasks: list[asyncio.Task[float]] = []
    try:
        destinations = [CudaDestination(cupy, size) for _ in offsets]
        if case == "mixed":
            await layer.load_bytes(bytearray(size // 2), 0, size // 2)
        elif case == "warm":
            await layer.load_bytes(bytearray(size), 0, size)
        await layer.drain()
        # Initialize destinations before timing so stale values cannot make a
        # missing copy appear correct. Each request has its own real stream.
        for destination in destinations:
            with destination.stream:
                destination.array.fill(0xA5)
            destination.stream.synchronize()
        gate = asyncio.Event()

        async def load_one(destination: CudaDestination, offset: int) -> float:
            await gate.wait()
            started = time.perf_counter_ns()
            loaded = await layer.load_bytes_grouped(
                destination,
                [{"target_offset": 0, "file_offset": offset, "nbytes": size}],
            )
            # The public load contract includes the H2D completion event.
            # Synchronizing the same stream also detects an accidental early
            # return in a variant and includes that wait in measured latency.
            destination.stream.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            if loaded != size:
                raise AssertionError(f"short load: {loaded} != {size}")
            return elapsed_ms

        tasks = [
            asyncio.create_task(load_one(destination, offset))
            for destination, offset in zip(destinations, offsets, strict=True)
        ]
        await asyncio.sleep(0)
        stats_before = asdict(layer.stats)
        native_before = counter.snapshot()
        process_before = io_read_bytes()
        device_before = block_read_bytes(device_stat)
        started = time.perf_counter_ns()
        gate.set()
        latencies = await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.timeout)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        process_after = io_read_bytes()
        device_after = block_read_bytes(device_stat)
        await layer.drain()
        native_after = counter.snapshot()
        stats_after = asdict(layer.stats)
        for destination, offset in zip(destinations, offsets, strict=True):
            if not numpy.array_equal(
                destination.array.get(), payload[offset : offset + size]
            ):
                raise AssertionError(
                    f"destination mismatch: case={case} offset={offset}"
                )
        stats = {key: stats_after[key] - stats_before[key] for key in stats_before}
        if case == "warm" and stats["l2_reads"]:
            raise AssertionError("warm load unexpectedly reached L2")
        if case == "mixed" and (stats["l1_hits"] == 0 or stats["l2_reads"] == 0):
            raise AssertionError("mixed case did not exercise both tiers")
        native = tuple(
            after - before
            for after, before in zip(native_after, native_before, strict=True)
        )
        return {
            "elapsed_ms": elapsed_ms,
            "request_latency_ms": latencies,
            "requested_bytes": size * concurrency,
            "unique_requested_bytes": unique_bytes,
            "unique_cold_bytes": 0
            if case == "warm"
            else unique_bytes - (size // 2 if case == "mixed" else 0),
            **stats,
            "native_read_calls": native[0],
            "native_read_extents": native[1],
            "native_read_bytes": native[2],
            "process_read_bytes": counter_delta(process_after, process_before),
            "nvme_read_bytes": counter_delta(device_after, device_before),
            "correctness": True,
        }
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(layer.drain(), timeout=args.timeout)
        finally:
            layer.close()
        destinations.clear()


async def benchmark(args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Run every requested cell sequentially and retain per-wave raw metrics.

    Args:
        args (Namespace): validated workload and output configuration.
        result (dict): progress record updated after every cell.
    Returns:
        None. Results are written atomically and ephemeral stores are deleted.
    Async/thread-safety:
        Uses one main event loop. Each wave drains before the next starts;
        variants must be scheduled by an external sequential A/B runner.
    """
    import cupy
    import numpy

    layer_type, source = source_info(args.repo, args.expect_commit)
    result["source"] = source
    cupy.cuda.Device(args.device).use()
    properties = cupy.cuda.runtime.getDeviceProperties(args.device)
    result["device"] = {
        "visible_ordinal": args.device,
        "name": properties["name"].decode(),
        "pci_bus_id": cupy.cuda.runtime.deviceGetPCIBusId(args.device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_version": cupy.__version__,
    }
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    device = args.scratch_root.stat().st_dev
    device_stat = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}/stat")
    result["storage"] = {
        "major": os.major(device),
        "minor": os.minor(device),
        "direct_io": True,
        "device_counter_includes_other_processes": True,
    }
    l2_bytes = max(max(args.sizes) * max(args.concurrency), args.l1_bytes or 0)
    payload = numpy.random.default_rng(args.seed).integers(
        0, 256, size=l2_bytes, dtype=numpy.uint8
    )
    counter = NativeReadCounter(args.native_counts)
    counter.install()
    try:
        with tempfile.TemporaryDirectory(
            prefix="demand-load-", dir=args.scratch_root
        ) as scratch:
            path = Path(scratch) / "daser.store"
            with path.open("wb") as store:
                store.write(memoryview(payload))
                store.flush()
                os.fsync(store.fileno())
            for size in args.sizes:
                for concurrency in args.concurrency:
                    for case in args.cases:
                        l1_bytes = args.l1_bytes or size * concurrency
                        if case == "small-l1" and args.l1_bytes is None:
                            l1_bytes = size * min(concurrency, 2)
                        row: dict[str, Any] = {
                            "case": case,
                            "size_bytes": size,
                            "request_bytes": size,
                            "concurrency": concurrency,
                            "l1_bytes": l1_bytes,
                            "l2_bytes": l2_bytes,
                            "status": "running",
                            "samples": [],
                        }
                        result["results"].append(row)
                        write_result(args.output, result)
                        if case in ("overlap", "mixed") and (size // 2) % ALIGNMENT:
                            row.update(
                                status="skipped",
                                reason=(
                                    "half-range is not 4096-byte aligned; "
                                    "request size is unchanged"
                                ),
                            )
                            write_result(args.output, result)
                            continue
                        try:
                            for iteration in range(-args.warmup, args.repeat):
                                sample = await run_wave(
                                    layer_type,
                                    cupy,
                                    numpy,
                                    path,
                                    payload,
                                    case,
                                    size,
                                    concurrency,
                                    l1_bytes,
                                    args,
                                    counter,
                                    device_stat,
                                )
                                if iteration >= 0:
                                    sample["iteration"] = iteration
                                    row["samples"].append(sample)
                            row["summary"] = summarize(row["samples"], numpy)
                            row["status"] = "ok"
                        except asyncio.TimeoutError as error:
                            row.update(
                                status="timeout",
                                error=str(error) or "request wave exceeded timeout",
                            )
                        except Exception as error:
                            row.update(
                                status="error", error=f"{type(error).__name__}: {error}"
                            )
                            raise
                        finally:
                            write_result(args.output, result)
                        print(
                            json.dumps(
                                {
                                    key: row[key]
                                    for key in (
                                        "case",
                                        "size_bytes",
                                        "concurrency",
                                        "status",
                                    )
                                }
                            ),
                            flush=True,
                        )
    finally:
        counter.restore()
    result["complete"] = all(
        row["status"] in ("ok", "skipped") for row in result["results"]
    )


def main() -> int:
    """Run the CLI with one event loop and return its exit status (int).

    Returns zero only for a complete matrix. All executor and CUDA work is
    drained by the wave lifecycle before the event loop closes.
    """
    args = parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "variant": args.variant,
        "complete": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": [],
        "measurement": (
            "real O_DIRECT io_uring + pinned L1 + CUDA H2D; validation excluded"
        ),
    }
    started = time.time()
    try:
        asyncio.run(benchmark(args, result))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        result["total_elapsed_s"] = time.time() - started
        write_result(args.output, result)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
