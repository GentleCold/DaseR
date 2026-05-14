# SPDX-License-Identifier: Apache-2.0
"""Lightweight latency measurement for hot-path performance diagnosis.

Provides a thread-safe LatencyHistogram and a timed context manager
that integrate with the existing PerfLogger. Enable via DASER_PERF_LOG=1.

Global registry
---------------
LatencyHistogram instances are automatically registered in a module-level
registry so that benchmark harnesses can collect and report them without
touching internal component state. Call :func:`collect_histograms` after
a benchmark run to get all measurements.
"""

# Standard
from contextlib import contextmanager
import json
import os
import threading
import time
from typing import Optional

# First Party
from daser.logging import init_perf_logger

perf = init_perf_logger(__name__)

# Module-level registry of all histogram instances.
_registry: dict[str, "LatencyHistogram"] = {}
_registry_lock = threading.Lock()


class LatencyHistogram:
    """Thread-safe histogram for latency measurements.

    Records raw latencies and computes percentiles on demand. Designed
    for low overhead on the hot path — record() is O(1).

    Args:
        name: metric name, e.g. "ipc_lookup_latency_s".
        unit: display unit string, e.g. "ms".
        scale: multiplier to convert stored values to the display unit
            (e.g. 1000 for s→ms). Default 1 (seconds, no conversion).

    Example::

        hist = LatencyHistogram("gds_read", unit="ms", scale=1000)
        with timed(hist):
            await gds.read_into_async(buf, offset)

        # After the benchmark run:
        print(hist.summary())  # p50=5.12 p95=8.34 p99=12.45
    """

    def __init__(
        self,
        name: str,
        unit: str = "s",
        scale: float = 1.0,
    ) -> None:
        self.name = name
        self.unit = unit
        self.scale = scale
        self._values: list[float] = []
        self._lock = threading.Lock()
        self._count = 0
        self._sum = 0.0
        # Auto-register in global registry for cross-component collection.
        with _registry_lock:
            _registry[name] = self

    def record(self, value_s: float) -> None:
        """Record a latency value (in seconds).

        Args:
            value_s: latency in fractional seconds.
        """
        with self._lock:
            self._values.append(value_s)
            self._count += 1
            self._sum += value_s

    @property
    def count(self) -> int:
        """Total number of recorded values."""
        return self._count

    @property
    def mean(self) -> float:
        """Arithmetic mean in display units."""
        if self._count == 0:
            return 0.0
        return (self._sum / self._count) * self.scale

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0–100) in display units.

        Args:
            p: percentile, e.g. 50, 95, 99.

        Returns:
            The p-th percentile value, or 0.0 if no data.
        """
        with self._lock:
            if not self._values:
                return 0.0
            s = sorted(self._values)
        idx = (len(s) - 1) * p / 100.0
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        frac = idx - lo
        return (s[lo] + (s[hi] - s[lo]) * frac) * self.scale

    def summary(self) -> str:
        """Return a compact one-line summary string.

        Returns:
            Formatted string like "n=1000 p50=1.23 p95=2.34 p99=5.67ms".
        """
        return (
            f"n={self._count} "
            f"mean={self.mean:.3f}{self.unit} "
            f"p50={self.percentile(50):.3f}{self.unit} "
            f"p95={self.percentile(95):.3f}{self.unit} "
            f"p99={self.percentile(99):.3f}{self.unit}"
        )

    def to_dict(self) -> dict:
        """Return a dict with count, mean, and key percentiles in display units."""
        return {
            "name": self.name,
            "unit": self.unit,
            "count": self._count,
            "mean": round(self.mean, 4),
            "p50": round(self.percentile(50), 4),
            "p95": round(self.percentile(95), 4),
            "p99": round(self.percentile(99), 4),
        }

    def log(self) -> None:
        """Write a one-line summary to the PerfLogger."""
        if self._count > 0:
            perf.record(
                self.name,
                self.percentile(50),
                f"{self.unit}_p50_count={self._count}",
            )
            perf.record(
                f"{self.name}_p99",
                self.percentile(99),
                f"{self.unit}_count={self._count}",
            )


@contextmanager
def timed(
    hist: Optional[LatencyHistogram] = None,
    label: str = "",
    log_direct: bool = False,
):
    """Context manager that records elapsed time to a histogram.

    When ``hist`` is None this is a no-op (zero overhead on the hot path
    when perf logging is disabled).

    Args:
        hist: LatencyHistogram to record into. If None, no measurement.
        label: human-readable label logged when log_direct is True.
        log_direct: if True, also log to PerfLogger immediately.

    Example::

        with timed(hist_ipc, "match_and_alloc", log_direct=True):
            resp = self._ipc_sync.match_and_alloc(tokens, key, model)
    """
    if hist is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        hist.record(elapsed)
        if log_direct and elapsed > 0.010:  # only log > 10ms to reduce noise
            perf.record(f"{hist.name}.{label}", elapsed * 1000, "ms")


def collect_histograms() -> dict[str, "LatencyHistogram"]:
    """Return a snapshot of all registered latency histograms.

    Returns:
        Dict mapping metric name → LatencyHistogram.
    """
    with _registry_lock:
        return dict(_registry)


def print_report() -> None:
    """Print a formatted latency distribution report to stdout.

    Collects all registered histograms and outputs a table with
    p50/p95/p99 for each metric. Intended for use at the end of
    a benchmark run when DASER_PERF_LOG=1.
    """
    histograms = collect_histograms()
    if not histograms:
        return

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("DaseR Latency Distribution Report")
    lines.append("=" * 90)
    header = f"{'Metric':<30} {'count':>7} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}"
    lines.append(header)
    lines.append("-" * 90)

    for name in sorted(histograms.keys()):
        h = histograms[name]
        if h.count == 0:
            continue
        lines.append(
            f"{name:<30} {h.count:>7} "
            f"{h.mean:>7.3f}{h.unit} "
            f"{h.percentile(50):>7.3f}{h.unit} "
            f"{h.percentile(95):>7.3f}{h.unit} "
            f"{h.percentile(99):>7.3f}{h.unit}"
        )

    lines.append("=" * 90)
    print("\n".join(lines))


def print_report_from_dict(data: dict) -> None:
    """Print a formatted latency distribution report from a loaded dict.

    Reads the same format produced by :func:`LatencyHistogram.to_dict`
    and :func:`export_histograms_json`. Useful when histogram data has
    been exported from a subprocess and loaded back via
    :func:`load_histograms_json`.

    Args:
        data: Dict mapping metric name → dict with count/mean/p50/p95/...
    """
    if not data:
        return

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("DaseR Latency Distribution Report")
    lines.append("=" * 90)
    header = f"{'Metric':<30} {'count':>7} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}"
    lines.append(header)
    lines.append("-" * 90)

    for name in sorted(data.keys()):
        h = data[name]
        if h.get("count", 0) == 0:
            continue
        unit = h.get("unit", "")
        lines.append(
            f"{name:<30} {h['count']:>7} "
            f"{h.get('mean', 0):>7.3f}{unit} "
            f"{h.get('p50', 0):>7.3f}{unit} "
            f"{h.get('p95', 0):>7.3f}{unit} "
            f"{h.get('p99', 0):>7.3f}{unit}"
        )

    lines.append("=" * 90)
    print("\n".join(lines))


def export_histograms_json(histograms: dict, path: str) -> None:
    """Write histogram data to a JSON file for cross-process collection.

    Designed for the EngineCore subprocess to persist its histogram
    measurements so the benchmark main process can read them back.

    Args:
        histograms: Dict mapping metric name → LatencyHistogram.
        path: File path to write JSON to.
    """
    data = {name: h.to_dict() for name, h in histograms.items() if h.count > 0}
    if not data:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_histograms_json(path: str) -> dict:
    """Read histogram data previously written by :func:`export_histograms_json`.

    Args:
        path: File path to read JSON from.

    Returns:
        Dict mapping metric name → dict with count/mean/p50/p95/p99.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
