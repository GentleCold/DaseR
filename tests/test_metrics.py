# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.metrics import MetricsRegistry


def test_registry_renders_prometheus_counters_and_gauges() -> None:
    """MetricsRegistry should render low-cardinality counters and gauges."""
    registry = MetricsRegistry()

    registry.counter("daser_test_requests_total", "Test requests").inc(
        labels={"route": "/infer", "status": "ok"}
    )
    registry.gauge("daser_test_inflight", "In-flight requests").set(
        3,
        labels={"route": "/infer"},
    )

    rendered = registry.render_prometheus()

    assert "# HELP daser_test_requests_total Test requests" in rendered
    assert "# TYPE daser_test_requests_total counter" in rendered
    assert 'daser_test_requests_total{route="/infer",status="ok"} 1.0' in rendered
    assert "# HELP daser_test_inflight In-flight requests" in rendered
    assert "# TYPE daser_test_inflight gauge" in rendered
    assert 'daser_test_inflight{route="/infer"} 3.0' in rendered


def test_registry_renders_histogram_buckets() -> None:
    """MetricsRegistry should render cumulative Prometheus histogram buckets."""
    registry = MetricsRegistry()
    histogram = registry.histogram(
        "daser_test_latency_seconds",
        "Test latency",
        buckets=(0.1, 0.5, 1.0),
    )

    histogram.observe(0.2, labels={"op": "lookup"})
    histogram.observe(2.0, labels={"op": "lookup"})

    rendered = registry.render_prometheus()

    assert "# TYPE daser_test_latency_seconds histogram" in rendered
    assert 'daser_test_latency_seconds_bucket{op="lookup",le="0.1"} 0' in rendered
    assert 'daser_test_latency_seconds_bucket{op="lookup",le="0.5"} 1' in rendered
    assert 'daser_test_latency_seconds_bucket{op="lookup",le="1.0"} 1' in rendered
    assert 'daser_test_latency_seconds_bucket{op="lookup",le="+Inf"} 2' in rendered
    assert 'daser_test_latency_seconds_count{op="lookup"} 2' in rendered
    assert 'daser_test_latency_seconds_sum{op="lookup"} 2.2' in rendered
