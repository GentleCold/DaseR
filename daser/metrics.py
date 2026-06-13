# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
import math
import threading
from typing import Iterable

Labels = dict[str, str]
LabelKey = tuple[tuple[str, str], ...]


def _label_key(labels: Labels | None) -> LabelKey:
    """Return a stable, sorted label key.

    Args:
        labels: Optional label mapping.

    Returns:
        Tuple key sorted by label name.
    """
    return tuple(sorted((labels or {}).items()))


def _format_float(value: float) -> str:
    """Format a float for Prometheus text exposition.

    Args:
        value: Numeric sample value.

    Returns:
        Stable decimal representation accepted by Prometheus.
    """
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    return str(float(value))


def _escape_label_value(value: str) -> str:
    """Escape one Prometheus label value.

    Args:
        value: Raw label value.

    Returns:
        Escaped label value.
    """
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: LabelKey) -> str:
    """Format labels for one Prometheus sample.

    Args:
        labels: Stable label key.

    Returns:
        Empty string or a ``{k="v"}`` label block.
    """
    if not labels:
        return ""
    body = ",".join(f'{key}="{_escape_label_value(value)}"' for key, value in labels)
    return f"{{{body}}}"


@dataclass
class Counter:
    """Monotonic Prometheus counter.

    Args:
        name: Metric name.
        description: HELP text.

    Async/thread-safety:
        Mutations and rendering are protected by an internal lock.
    """

    name: str
    description: str
    _values: dict[LabelKey, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0, labels: Labels | None = None) -> None:
        """Increase a labeled counter series.

        Args:
            amount: Positive increment.
            labels: Optional low-cardinality labels.
        """
        if amount < 0:
            raise ValueError("counter increment must be non-negative")
        key = _label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        """Render this counter in Prometheus text format.

        Returns:
            Text lines without trailing newlines.
        """
        with self._lock:
            samples = sorted(self._values.items())
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        for labels, value in samples:
            lines.append(f"{self.name}{_format_labels(labels)} {_format_float(value)}")
        return lines


@dataclass
class Gauge:
    """Prometheus gauge.

    Args:
        name: Metric name.
        description: HELP text.

    Async/thread-safety:
        Mutations and rendering are protected by an internal lock.
    """

    name: str
    description: str
    _values: dict[LabelKey, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, labels: Labels | None = None) -> None:
        """Set a labeled gauge series.

        Args:
            value: Gauge value.
            labels: Optional low-cardinality labels.
        """
        key = _label_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: Labels | None = None) -> None:
        """Increase a labeled gauge series.

        Args:
            amount: Increment amount.
            labels: Optional low-cardinality labels.
        """
        key = _label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        """Render this gauge in Prometheus text format.

        Returns:
            Text lines without trailing newlines.
        """
        with self._lock:
            samples = sorted(self._values.items())
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        for labels, value in samples:
            lines.append(f"{self.name}{_format_labels(labels)} {_format_float(value)}")
        return lines


@dataclass
class _HistogramSeries:
    """Mutable histogram state for one label set."""

    buckets: list[int]
    count: int = 0
    total: float = 0.0


@dataclass
class Histogram:
    """Prometheus histogram with fixed buckets.

    Args:
        name: Metric name.
        description: HELP text.
        buckets: Finite bucket upper bounds.

    Async/thread-safety:
        Mutations and rendering are protected by an internal lock.
    """

    name: str
    description: str
    buckets: tuple[float, ...]
    _values: dict[LabelKey, _HistogramSeries] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, labels: Labels | None = None) -> None:
        """Record one histogram observation.

        Args:
            value: Observed value.
            labels: Optional low-cardinality labels.
        """
        key = _label_key(labels)
        with self._lock:
            series = self._values.get(key)
            if series is None:
                series = _HistogramSeries(buckets=[0] * len(self.buckets))
                self._values[key] = series
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    series.buckets[index] += 1
            series.count += 1
            series.total += value

    def render(self) -> list[str]:
        """Render this histogram in Prometheus text format.

        Returns:
            Text lines without trailing newlines.
        """
        with self._lock:
            samples = {
                labels: _HistogramSeries(
                    buckets=list(series.buckets),
                    count=series.count,
                    total=series.total,
                )
                for labels, series in self._values.items()
            }
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        for labels, series in sorted(samples.items()):
            for bound, count in zip(self.buckets, series.buckets, strict=True):
                bucket_labels = (*labels, ("le", _format_float(bound)))
                lines.append(
                    f"{self.name}_bucket{_format_labels(bucket_labels)} {count}"
                )
            inf_labels = (*labels, ("le", "+Inf"))
            lines.append(
                f"{self.name}_bucket{_format_labels(inf_labels)} {series.count}"
            )
            lines.append(f"{self.name}_count{_format_labels(labels)} {series.count}")
            lines.append(
                f"{self.name}_sum{_format_labels(labels)} {_format_float(series.total)}"
            )
        return lines


class MetricsRegistry:
    """In-process Prometheus metric registry.

    Async/thread-safety:
        Registry mutation is protected by a lock. Individual metric objects
        also protect their sample state for concurrent HTTP and IPC paths.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str) -> Counter:
        """Return or create a counter metric.

        Args:
            name: Metric name.
            description: HELP text.

        Returns:
            Counter instance.
        """
        return self._metric(name, Counter(name, description), Counter)

    def gauge(self, name: str, description: str) -> Gauge:
        """Return or create a gauge metric.

        Args:
            name: Metric name.
            description: HELP text.

        Returns:
            Gauge instance.
        """
        return self._metric(name, Gauge(name, description), Gauge)

    def histogram(
        self,
        name: str,
        description: str,
        buckets: Iterable[float],
    ) -> Histogram:
        """Return or create a histogram metric.

        Args:
            name: Metric name.
            description: HELP text.
            buckets: Finite bucket upper bounds.

        Returns:
            Histogram instance.
        """
        bucket_tuple = tuple(float(bucket) for bucket in buckets)
        return self._metric(name, Histogram(name, description, bucket_tuple), Histogram)

    def render_prometheus(self) -> str:
        """Render all registered metrics in Prometheus text format.

        Returns:
            Prometheus exposition text ending with a newline.
        """
        with self._lock:
            metrics = [self._metrics[name] for name in sorted(self._metrics)]
        lines: list[str] = []
        for metric in metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"

    def _metric(
        self,
        name: str,
        created: Counter | Gauge | Histogram,
        expected_type: type[Counter] | type[Gauge] | type[Histogram],
    ) -> Counter | Gauge | Histogram:
        """Return a metric, validating name/type consistency."""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is None:
                self._metrics[name] = created
                return created
            if not isinstance(existing, expected_type):
                raise TypeError(f"metric {name} already registered with another type")
            return existing


REGISTRY = MetricsRegistry()
