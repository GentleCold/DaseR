# SPDX-License-Identifier: Apache-2.0

# Third Party
from fastapi.testclient import TestClient

# First Party
from daser.metrics import MetricsRegistry
from daser.server.http import HTTPServerConfig, build_http_app
from tests.server.test_http_server import FakeTokenizer, FakeVLLMClient, make_core


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    """The HTTP server should expose a Prometheus-compatible /metrics endpoint."""
    registry = MetricsRegistry()
    registry.counter("daser_test_total", "Test counter").inc()
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
        ),
        make_core(),
        tokenizer=FakeTokenizer(),
        vllm=FakeVLLMClient(),
        metrics_registry=registry,
    )
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "# HELP daser_test_total Test counter" in response.text
    assert "# TYPE daser_test_total counter" in response.text
    assert "daser_test_total 1.0" in response.text


def test_http_routes_record_request_metrics() -> None:
    """HTTP server should expose daser_up and daser_info gauges."""
    registry = MetricsRegistry()
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
            cache_reuse_mode="chunk",
            transfer_mode="iouring",
        ),
        make_core(),
        tokenizer=FakeTokenizer(),
        vllm=FakeVLLMClient(),
        metrics_registry=registry,
    )
    client = TestClient(app)

    rendered = client.get("/metrics").text

    assert "daser_up 1.0" in rendered
    assert 'daser_info{mode="chunk",transfer="iouring"} 1.0' in rendered


def test_http_routes_record_vllm_health_gauge() -> None:
    """The health route should publish vLLM liveness as a gauge."""
    registry = MetricsRegistry()
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
        ),
        make_core(),
        tokenizer=FakeTokenizer(),
        vllm=FakeVLLMClient(),
        metrics_registry=registry,
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert "daser_vllm_health_up 1.0" in client.get("/metrics").text
