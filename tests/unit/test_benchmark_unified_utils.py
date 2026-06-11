# SPDX-License-Identifier: Apache-2.0
"""Unit tests for unified benchmark dataset, prompt, and manifest utilities."""

# Standard
import asyncio
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.bench_load import (
    _add_phase_comparison,
    _backend_server_hit_rate,
    _common_config_for_run,
    _generation_params,
    _serialise_phase,
    _should_dedup_context,
)

# First Party
from benchmarks.utils.constants import BYTES_PER_GIB, COMPARISON_IOURING_MEM
from benchmarks.utils.datasets import BenchmarkSample, ImdbDataset, LongBenchDataset
from benchmarks.utils.loadgen import (
    PhaseResult,
    RequestResult,
    _metric_hit_ratios,
    lmcache_metrics_url,
    run_daser_chunk,
    run_daser_prefix,
    run_lmcache,
    summarise_results,
    vllm_completion_stream,
)
from benchmarks.utils.metrics import (
    compute_metric_delta,
    extract_lmcache_status_metrics,
    extract_prometheus_counters,
    first_available_hit_ratio,
    hit_ratio_from_metrics,
)
from benchmarks.utils.prompts import (
    DOCS_MARKER,
    build_chunk_aligned_prompt_ids,
    build_document_prompt,
    build_full_prompt,
)
from benchmarks.utils.servers import (
    REPO_ROOT,
    BenchmarkManifest,
    ServerManager,
    ServiceEndpoint,
)
from benchmarks.utils.sizing import (
    BenchmarkCapacityLimits,
    derive_benchmark_sizing,
    format_capacity,
    parse_size_bytes,
)


class _Tokenizer:
    """Small tokenizer stub for prompt rendering tests."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return (
            f"system: {messages[0]['content']}\n"
            f"user: {messages[1]['content']}\n"
            "assistant: "
        )


def test_longbench_dataset_loads_samples_with_answers(tmp_path: Path) -> None:
    """LongBench loader preserves context, question, and answers."""
    data = tmp_path / "2wikimqa.jsonl"
    data.write_text(
        json.dumps(
            {
                "context": "doc text",
                "input": "question?",
                "answers": ["answer"],
            }
        )
        + "\n"
    )

    samples = LongBenchDataset(tmp_path, ["2wikimqa"], max_samples=1).load()

    assert len(samples) == 1
    assert samples[0].dataset == "2wikimqa"
    assert samples[0].context == "doc text"
    assert samples[0].question == "question?"
    assert samples[0].answers == ["answer"]


def test_imdb_dataset_uses_review_as_context(tmp_path: Path) -> None:
    """IMDB loader maps review text into the shared benchmark sample shape."""
    imdb = tmp_path / "imdb.csv"
    imdb.write_text('review\n"a good movie"\n')

    samples = ImdbDataset(imdb, max_samples=1).load()

    assert len(samples) == 1
    assert samples[0].dataset == "imdb"
    assert samples[0].context == "a good movie"
    assert samples[0].question == "Summarize the sentiment of this review."


def test_build_full_prompt_replaces_single_documents_marker() -> None:
    """Prompt builder injects context into the chat-template document slot."""
    prompt = build_full_prompt(_Tokenizer(), "doc body", "answer?")

    assert DOCS_MARKER not in prompt
    assert "Documents:\ndoc body\n\nTask: answer?" in prompt
    assert prompt.endswith("assistant: ")


def test_document_prompt_matches_full_prompt_for_single_doc() -> None:
    """DaseR chunk prompts match baseline/LMCache full prompts."""
    full_prompt = build_full_prompt(_Tokenizer(), "doc body", "answer?")
    document_prompt = build_document_prompt(_Tokenizer(), ["doc body"], "answer?")

    assert document_prompt == full_prompt


def test_chunk_aligned_prompt_ids_pad_like_daser_chunk_infer() -> None:
    """Baseline/LMCache can use the same padded token layout as DaseR chunk."""

    class Tokenizer(_Tokenizer):
        pad_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
            assert add_special_tokens is False
            return {"input_ids": self.encode(text, add_special_tokens=False)}

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            if text == "doc":
                return [10, 11, 12]
            return [1] * len(text)

    padded = build_chunk_aligned_prompt_ids(
        Tokenizer(),
        context="doc",
        question="q",
        block_tokens=4,
    )
    unpadded = Tokenizer()(
        build_full_prompt(Tokenizer(), "doc", "q"), add_special_tokens=False
    )["input_ids"]

    assert len(padded) % 4 != 0
    assert len(padded) > len(unpadded)


def test_prometheus_external_prefix_delta_hit_ratio() -> None:
    """Prometheus counter deltas expose vLLM external prefix hit ratio."""
    before = extract_prometheus_counters(
        """
        vllm:external_prefix_cache_queries_total{model="m"} 100
        vllm:external_prefix_cache_hits_total{model="m"} 25
        """
    )
    after = extract_prometheus_counters(
        """
        vllm:external_prefix_cache_queries_total{model="m"} 180
        vllm:external_prefix_cache_hits_total{model="m"} 105
        """
    )

    delta = compute_metric_delta(before, after)
    ratio = hit_ratio_from_metrics(
        delta,
        hits_key="vllm:external_prefix_cache_hits_total",
        queries_key="vllm:external_prefix_cache_queries_total",
    )

    assert delta["vllm:external_prefix_cache_queries_total"] == 80
    assert delta["vllm:external_prefix_cache_hits_total"] == 80
    assert ratio == 1.0


def test_prometheus_counters_preserve_labeled_series() -> None:
    """Prometheus parsing keeps both aggregate and labeled counter samples."""
    metrics = extract_prometheus_counters(
        """
        daser_cache_lookup_total{result="miss"} 3
        daser_cache_lookup_total{result="hit"} 7
        """
    )

    assert metrics["daser_cache_lookup_total"] == 10
    assert metrics['daser_cache_lookup_total{result="miss"}'] == 3
    assert metrics['daser_cache_lookup_total{result="hit"}'] == 7


def test_daser_prometheus_token_hit_ratio() -> None:
    """DaseR Prometheus token counters expose backend server hit ratio."""
    metrics = extract_prometheus_counters(
        """
        daser_cache_requested_tokens_total 4096
        daser_cache_matched_tokens_total 3072
        """
    )

    assert (
        hit_ratio_from_metrics(
            metrics,
            hits_key="daser_cache_matched_tokens_total",
            queries_key="daser_cache_requested_tokens_total",
        )
        == 0.75
    )


def test_daser_summary_hit_rate_uses_vllm_external_prefix() -> None:
    """DaseR summary hit rate uses DaseR's vLLM-equivalent internal counters."""
    assert (
        _backend_server_hit_rate(
            {
                "daser_external_prefix": 0.93,
                "daser_prometheus_tokens": 1.0,
                "daser_prometheus_requests": 1.0,
            }
        )
        == 0.93
    )


def test_daser_summary_hit_rate_ignores_control_plane_lookup_ratio() -> None:
    """DaseR lookup counters are diagnostics, not external-prefix hit ratio."""
    assert (
        _backend_server_hit_rate(
            {
                "daser_prometheus_tokens": 1.0,
                "daser_prometheus_requests": 1.0,
            }
        )
        is None
    )


def test_daser_external_prefix_hit_ratio_from_internal_metrics() -> None:
    """DaseR internal metrics expose the vLLM-equivalent external hit ratio."""
    metrics = extract_prometheus_counters(
        """
        daser_external_prefix_cache_queries_total 1000
        daser_external_prefix_cache_hits_total 930
        """
    )

    assert (
        first_available_hit_ratio(
            metrics,
            (
                (
                    "daser_external_prefix_cache_hits_total",
                    "daser_external_prefix_cache_queries_total",
                ),
            ),
        )
        == 0.93
    )


def test_lmcache_status_metrics_extract_prefetch_hit_ratio() -> None:
    """LMCache status metrics can be reduced to a token hit ratio."""
    status = {
        "storage_manager": {
            "prefetch_controller": {
                "requested_tokens": 1000,
                "hit_tokens": 750,
            }
        }
    }

    metrics = extract_lmcache_status_metrics(status)

    assert metrics["lmcache_prefetch_requested_tokens"] == 1000
    assert metrics["lmcache_prefetch_hit_tokens"] == 750
    assert (
        hit_ratio_from_metrics(
            metrics,
            hits_key="lmcache_prefetch_hit_tokens",
            queries_key="lmcache_prefetch_requested_tokens",
        )
        == 0.75
    )


def test_load_config_uses_prepared_sizing_over_runtime_limits(tmp_path: Path) -> None:
    """Load results reuse prepare-time sizing instead of runtime capacity state."""
    prepared = {
        "config": {
            "dataset": "longbench",
            "num_samples": 86,
            "max_inflight": 8,
            "gen_params": {"max_tokens": 1, "temperature": 0.0},
            "total_prompt_tokens": 956512,
            "total_blocks": 59747,
            "max_prompt_blocks": 2314,
            "derived_l1_size_bytes": 211525042176,
            "derived_l1_size": "197.00 GiB",
            "derived_l2_size_bytes": 211525042176,
            "derived_l2_size": "197.00 GiB",
            "lmcache_l1_gb": 197,
            "lmcache_l2_gb": None,
            "capacity_capped": False,
            "evict": False,
            "planned_skip_l2": True,
        }
    }
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(prepared))
    manifest = BenchmarkManifest(
        run_id="run1",
        backend="daser",
        reuse_mode="chunk",
        model="model",
        store_dir="/store",
        l1_size_bytes=211525042176,
        l2_size_bytes=211525042176,
        skip_l2=True,
        endpoints={"vllm": ServiceEndpoint("http://127.0.0.1:8001")},
        log_dir="/logs",
        pid_file="/pids.json",
    )

    config = _common_config_for_run(
        prepared_config_path=str(path),
        prepare_only=False,
        manifest=manifest,
        dataset="longbench",
        num_samples=86,
        max_inflight=8,
        gen_params={"max_tokens": 1, "temperature": 0.0},
        total_prompt_tokens=956512,
        total_blocks=59747,
        max_prompt_blocks=2314,
        evict=False,
        sizing=None,
    )

    assert config["derived_l1_size"] == "197.00 GiB"
    assert config["derived_l1_size_bytes"] == manifest.l1_size_bytes
    assert config["capacity_capped"] is False


def test_load_config_falls_back_to_manifest_sizing_without_prepare() -> None:
    """Direct load runs report manifest capacities, not a fresh size inference."""
    manifest = BenchmarkManifest(
        run_id="run1",
        backend="daser",
        reuse_mode="chunk",
        model="model",
        store_dir="/store",
        l1_size_bytes=211525042176,
        l2_size_bytes=211525042176,
        skip_l2=True,
        endpoints={"vllm": ServiceEndpoint("http://127.0.0.1:8001")},
        log_dir="/logs",
        pid_file="/pids.json",
    )

    config = _common_config_for_run(
        prepared_config_path=None,
        prepare_only=False,
        manifest=manifest,
        dataset="longbench",
        num_samples=86,
        max_inflight=8,
        gen_params={"max_tokens": 1, "temperature": 0.0},
        total_prompt_tokens=956512,
        total_blocks=59747,
        max_prompt_blocks=2314,
        evict=False,
        sizing=None,
    )

    assert config["derived_l1_size_bytes"] == manifest.l1_size_bytes
    assert config["derived_l1_size"] == format_capacity(manifest.l1_size_bytes)
    assert config["derived_l2_size_bytes"] == manifest.l2_size_bytes
    assert config["capacity_capped"] is None


def test_prefix_and_chunk_default_context_dedup_match() -> None:
    """Prefix and chunk modes use the same default workload dedup policy."""

    class Args:
        no_dedup_context = False

    assert _should_dedup_context(Args()) is True


def test_no_dedup_context_disables_workload_dedup() -> None:
    """The explicit opt-out flag disables context dedup for every mode."""

    class Args:
        no_dedup_context = True

    assert _should_dedup_context(Args()) is False


def test_lmcache_prometheus_lookup_hit_ratio_uses_mp_names() -> None:
    """LMCache MP Prometheus counters expose L1+L2 token hit ratio."""
    metrics = extract_prometheus_counters(
        """
        lmcache_mp_lookup_requested_total{model_name="m"} 2048
        lmcache_mp_lookup_hit_total{model_name="m"} 1536
        """
    )

    assert (
        first_available_hit_ratio(
            metrics,
            (
                (
                    "lmcache_mp_lookup_hit_total",
                    "lmcache_mp_lookup_requested_total",
                ),
            ),
        )
        == 0.75
    )


def test_lmcache_hit_ratio_prefers_token_counters_over_request_counters() -> None:
    """LMCache backend comparison should use token hit ratio when available."""
    metrics = extract_prometheus_counters(
        """
        lmcache_mp_lookup_requested_total{model_name="m"} 10
        lmcache_mp_lookup_hit_total{model_name="m"} 9
        lmcache_mp_lookup_requested_tokens_total{model_name="m"} 1000
        lmcache_mp_lookup_hit_tokens_total{model_name="m"} 750
        """
    )

    assert (
        first_available_hit_ratio(
            metrics,
            (
                (
                    "lmcache_mp_lookup_hit_tokens_total",
                    "lmcache_mp_lookup_requested_tokens_total",
                ),
                (
                    "lmcache_mp_lookup_hit_total",
                    "lmcache_mp_lookup_requested_total",
                ),
            ),
        )
        == 0.75
    )


def test_lmcache_metric_hit_ratios_prefer_token_counters() -> None:
    """Benchmark summary should prefer LMCache token ratio over request ratio."""
    metrics = extract_prometheus_counters(
        """
        lmcache_mp_lookup_requested_total{model_name="m"} 10
        lmcache_mp_lookup_hit_total{model_name="m"} 9
        lmcache_mp_lookup_requested_tokens_total{model_name="m"} 1000
        lmcache_mp_lookup_hit_tokens_total{model_name="m"} 750
        """
    )

    hit_ratios = _metric_hit_ratios(
        {
            "vllm_prometheus": {},
            "backend_prometheus": metrics,
            "backend_status": {},
        }
    )

    assert hit_ratios["lmcache_prometheus_lookup"] == 0.75


def test_manifest_round_trip(tmp_path: Path) -> None:
    """Benchmark manifest JSON preserves service endpoints and sizing fields."""
    manifest = BenchmarkManifest(
        run_id="run1",
        backend="daser",
        reuse_mode="chunk",
        model="/models/qwen",
        store_dir=str(tmp_path),
        l1_size_bytes=1024,
        l2_size_bytes=2048,
        skip_l2=True,
        endpoints={
            "vllm": ServiceEndpoint(url="http://127.0.0.1:8001"),
            "daser": ServiceEndpoint(url="http://127.0.0.1:2026"),
        },
        log_dir=str(tmp_path / "logs"),
        pid_file=str(tmp_path / "pids.json"),
    )
    path = tmp_path / "manifest.json"

    manifest.write(path)
    loaded = BenchmarkManifest.read(path)

    assert loaded == manifest


def test_daser_noevict_start_uses_l1_only_mode(tmp_path: Path) -> None:
    """DaseR no-evict starts without L2 sizing arguments."""
    manager = ServerManager(
        run_id="run1",
        backend="daser",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
        skip_l2=True,
    )

    cmd = manager._daser_server_command()  # noqa: SLF001
    manifest = manager.manifest()

    assert "--skip-l2" in cmd
    assert "--l1-size" in cmd
    assert cmd[cmd.index("--l1-size") + 1] == "1024"
    assert "--l2-size" not in cmd
    assert manifest.skip_l2 is True


def test_daser_evict_start_keeps_l2_enabled(tmp_path: Path) -> None:
    """DaseR evict runs keep the persistent L2 tier enabled."""
    manager = ServerManager(
        run_id="run1",
        backend="daser",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
        skip_l2=False,
    )

    cmd = manager._daser_server_command()  # noqa: SLF001
    manifest = manager.manifest()

    assert "--skip-l2" not in cmd
    assert manifest.skip_l2 is False


async def test_daser_benchmark_start_does_not_force_debug_logging(
    tmp_path: Path,
) -> None:
    """Benchmark DaseR startup keeps hot-path DEBUG logging disabled by default."""
    manager = ServerManager(
        run_id="run1",
        backend="daser",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
        skip_l2=True,
    )
    observed: dict[str, str | None] = {}

    def fake_start(
        cmd: list[str],
        log_name: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        observed["log_name"] = log_name
        observed["level"] = (
            None if extra_env is None else extra_env.get("DASER_LOG_LEVEL")
        )
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    async def fake_wait_healthy(
        base_url: str,
        path: str,
        timeout: float,
        proc: subprocess.Popen[bytes],
    ) -> None:
        proc.terminate()
        proc.wait(timeout=5)

    manager._start = fake_start  # type: ignore[method-assign]  # noqa: SLF001
    manager._wait_healthy = fake_wait_healthy  # type: ignore[method-assign]  # noqa: SLF001

    await manager.start_daser_server()

    assert observed == {"log_name": "daser.log", "level": None}


def test_lmcache_noevict_start_disables_l2_adapter(tmp_path: Path) -> None:
    """LMCache no-evict starts without a disk L2 adapter."""
    manager = ServerManager(
        run_id="run1",
        backend="lmcache",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024**3,
        l2_size_bytes=2 * 1024**3,
        skip_l2=True,
    )

    cmd = manager._lmcache_mp_server_command()  # noqa: SLF001
    manifest = manager.manifest()

    assert "--l2-adapter" not in cmd
    assert "lmcache_mp_disk" not in " ".join(cmd)
    assert manifest.skip_l2 is True


def test_lmcache_evict_start_keeps_l2_adapter(tmp_path: Path) -> None:
    """LMCache evict runs keep the filesystem L2 adapter."""
    manager = ServerManager(
        run_id="run1",
        backend="lmcache",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024**3,
        l2_size_bytes=2 * 1024**3,
        skip_l2=False,
    )

    cmd = manager._lmcache_mp_server_command()  # noqa: SLF001
    l2_spec = json.loads(cmd[cmd.index("--l2-adapter") + 1])

    assert l2_spec["type"] == "fs"
    assert l2_spec["base_path"].endswith("lmcache_mp_disk")


def test_summarise_results_reports_hit_rate() -> None:
    """Loadgen summary includes aggregate latency and cache-hit rate."""
    summary = summarise_results(
        [
            RequestResult(
                sample_id=1,
                dataset="imdb",
                generated_text="yes",
                ttft_ms=10.0,
                latency_ms=20.0,
                prompt_tokens=100,
                completion_tokens=1,
                cache_hits=2,
                cache_chunks_total=4,
                queue_ms=5.0,
            )
        ]
    )

    assert summary["num_requests"] == 1
    assert summary["ttft_ms_mean"] == 10.0
    assert summary["queue_ms_mean"] == 5.0
    assert summary["cache_hit_rate"] == 0.5


def test_generation_params_are_deterministic_by_default() -> None:
    """Benchmark generation defaults include deterministic sampling controls."""
    params = _generation_params(max_tokens=128, temperature=0.0, seed=42)

    assert params == {
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }


def test_serialise_phase_reports_elapsed_throughput() -> None:
    """Phase summaries include all-request elapsed time and prompt throughput."""
    phase = PhaseResult(
        requests=[
            RequestResult(
                sample_id=1,
                dataset="imdb",
                generated_text="yes",
                ttft_ms=10.0,
                latency_ms=20.0,
                prompt_tokens=100,
                completion_tokens=1,
            )
        ],
        metrics={},
        elapsed_ms=250.0,
    )

    result = _serialise_phase(phase, {})

    assert result["summary"]["all_requests_elapsed_ms"] == 250.0
    assert result["summary"]["phase_elapsed_ms"] == 250.0
    assert result["summary"]["phase_prompt_tok_per_s"] == 400.0


async def test_vllm_stream_timing_excludes_semaphore_wait() -> None:
    """Request TTFT excludes local client-side concurrency wait."""

    class _Response:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self) -> object:
            yield 'data: {"choices":[{"text":"x"}]}'
            yield (
                'data: {"usage":{"prompt_tokens":3,"completion_tokens":1},"choices":[]}'
            )
            yield "data: [DONE]"

        async def __aenter__(self) -> "_Response":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class _Client:
        def stream(self, *args: object, **kwargs: object) -> _Response:
            return _Response()

    sem = asyncio.Semaphore(1)
    await sem.acquire()
    sample = BenchmarkSample(
        sample_id=1,
        dataset="imdb",
        context="review",
        question="question",
        answers=[],
    )
    task = asyncio.create_task(
        vllm_completion_stream(
            _Client(),
            "http://127.0.0.1:8001",
            sample,
            "prompt",
            {"max_tokens": 1},
            sem,
            10.0,
        )
    )
    await asyncio.sleep(0.01)
    sem.release()

    result = await task

    assert result.queue_ms > 0.0
    assert result.ttft_ms < result.queue_ms


async def test_daser_chunk_warm_phase_records_elapsed_ms(monkeypatch) -> None:
    """DaseR chunk warm phase includes wall-clock elapsed time."""

    async def fake_collect_phase_metrics(*_args, **_kwargs):
        return {}

    async def fake_upload_doc(*_args, **_kwargs):
        return {"doc_id": "doc-1"}

    async def fake_infer(*_args, **_kwargs):
        await asyncio.sleep(0)
        return RequestResult(
            sample_id=1,
            dataset="longbench",
            generated_text="x",
            ttft_ms=1.0,
            latency_ms=2.0,
            prompt_tokens=3,
            completion_tokens=1,
        )

    import benchmarks.utils.loadgen as loadgen

    monkeypatch.setattr(loadgen, "collect_phase_metrics", fake_collect_phase_metrics)
    monkeypatch.setattr(loadgen, "_daser_upload_doc", fake_upload_doc)
    monkeypatch.setattr(loadgen, "_daser_infer", fake_infer)
    manifest = BenchmarkManifest(
        run_id="test",
        backend="daser",
        reuse_mode="chunk",
        model="model",
        store_dir="/store",
        l1_size_bytes=1,
        l2_size_bytes=1,
        skip_l2=True,
        endpoints={"daser": ServiceEndpoint("http://127.0.0.1:2026")},
        log_dir="/logs",
        pid_file="/pids.json",
    )

    result = await run_daser_chunk(
        manifest=manifest,
        samples=[
            BenchmarkSample(
                sample_id=1,
                dataset="longbench",
                context="ctx",
                question="q",
                answers=[],
            )
        ],
        max_inflight=1,
        gen_params={"max_tokens": 1},
        timeout=1.0,
    )

    assert result["warm"].elapsed_ms > 0


async def test_lmcache_reuses_identical_prompt_payloads_for_cold_and_warm(
    monkeypatch,
) -> None:
    """LMCache cold and warm phases use the same constructed prompt payloads."""
    seen_prompts: list[list[str | list[int]]] = []

    class Tokenizer(_Tokenizer):
        pad_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
            assert add_special_tokens is False
            return {"input_ids": self.encode(text, add_special_tokens=False)}

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(char) % 97 for char in text]

    async def fake_collect_phase_metrics(*_args, **_kwargs):
        return {}

    async def fake_wait_lmcache_quiescent(*_args, **_kwargs):
        return None

    async def fake_vllm_completion_stream(
        _client,
        _url,
        sample,
        prompt,
        _gen_params,
        sem,
        _timeout,
    ):
        async with sem:
            seen_prompts[-1].append(prompt)
        return RequestResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            generated_text="x",
            ttft_ms=1.0,
            latency_ms=2.0,
            prompt_tokens=len(prompt),
            completion_tokens=1,
        )

    import benchmarks.utils.loadgen as loadgen

    original_build = loadgen.build_prompt_payloads

    def tracking_build(*args, **kwargs):
        prompts = original_build(*args, **kwargs)
        seen_prompts.append([])
        return prompts

    monkeypatch.setattr(loadgen, "collect_phase_metrics", fake_collect_phase_metrics)
    monkeypatch.setattr(loadgen, "_wait_lmcache_quiescent", fake_wait_lmcache_quiescent)
    monkeypatch.setattr(loadgen, "vllm_completion_stream", fake_vllm_completion_stream)
    monkeypatch.setattr(loadgen, "build_prompt_payloads", tracking_build)

    manifest = BenchmarkManifest(
        run_id="test",
        backend="lmcache",
        reuse_mode="chunk",
        model="model",
        store_dir="/store",
        l1_size_bytes=1,
        l2_size_bytes=1,
        skip_l2=True,
        endpoints={"vllm": ServiceEndpoint("http://127.0.0.1:8001")},
        log_dir="/logs",
        pid_file="/pids.json",
    )

    await run_lmcache(
        manifest=manifest,
        samples=[
            BenchmarkSample(
                sample_id=1,
                dataset="longbench",
                context="ctx",
                question="q",
                answers=[],
            )
        ],
        tokenizer=Tokenizer(),
        max_inflight=1,
        gen_params={"max_tokens": 1},
        timeout=1.0,
        settle_seconds=0.0,
        chunk_aligned_prompts=True,
    )

    assert len(seen_prompts) == 1
    assert seen_prompts[0][0] == seen_prompts[0][1]


async def test_daser_prefix_uses_chunk_aligned_prompt_payloads(monkeypatch) -> None:
    """DaseR prefix uses the same padded prompt profile as chunk comparisons."""
    chunk_aligned_values: list[bool] = []

    async def fake_collect_phase_metrics(*_args, **_kwargs):
        return {}

    async def fake_daser_drain(*_args, **_kwargs):
        return None

    async def fake_run_vllm_phase_requests(
        _manifest,
        _samples,
        _tokenizer,
        _max_inflight,
        _gen_params,
        _timeout,
        chunk_aligned_prompts=False,
        prompts=None,
    ):
        del prompts
        chunk_aligned_values.append(bool(chunk_aligned_prompts))
        return (
            [
                RequestResult(
                    sample_id=1,
                    dataset="longbench",
                    generated_text="x",
                    ttft_ms=1.0,
                    latency_ms=2.0,
                    prompt_tokens=8,
                    completion_tokens=1,
                )
            ],
            1.0,
        )

    import benchmarks.utils.loadgen as loadgen

    monkeypatch.setattr(loadgen, "collect_phase_metrics", fake_collect_phase_metrics)
    monkeypatch.setattr(loadgen, "_wait_daser_drained", fake_daser_drain)
    monkeypatch.setattr(
        loadgen, "_run_vllm_phase_requests", fake_run_vllm_phase_requests
    )

    manifest = BenchmarkManifest(
        run_id="test",
        backend="daser",
        reuse_mode="prefix",
        model="model",
        store_dir="/store",
        l1_size_bytes=1,
        l2_size_bytes=1,
        skip_l2=True,
        endpoints={
            "vllm": ServiceEndpoint("http://127.0.0.1:8001"),
            "daser": ServiceEndpoint("http://127.0.0.1:2026"),
        },
        log_dir="/logs",
        pid_file="/pids.json",
    )

    await run_daser_prefix(
        manifest=manifest,
        samples=[
            BenchmarkSample(
                sample_id=1,
                dataset="longbench",
                context="ctx",
                question="q",
                answers=[],
            )
        ],
        tokenizer=_Tokenizer(),
        max_inflight=1,
        gen_params={"max_tokens": 1},
        timeout=1.0,
    )

    assert chunk_aligned_values == [True, True]


async def test_daser_prefix_waits_for_drain_without_sleep(monkeypatch) -> None:
    """DaseR prefix transitions to warm immediately after backend drain."""
    calls: list[str] = []

    async def fake_collect_phase_metrics(*_args, **_kwargs):
        return {}

    async def fake_daser_drain(*_args, **_kwargs):
        calls.append("drain")

    async def fake_sleep(_seconds):
        calls.append("sleep")

    async def fake_run_vllm_phase_requests(*_args, **_kwargs):
        calls.append("phase")
        return (
            [
                RequestResult(
                    sample_id=1,
                    dataset="longbench",
                    generated_text="x",
                    ttft_ms=1.0,
                    latency_ms=2.0,
                    prompt_tokens=8,
                    completion_tokens=1,
                )
            ],
            1.0,
        )

    import benchmarks.utils.loadgen as loadgen

    monkeypatch.setattr(loadgen, "collect_phase_metrics", fake_collect_phase_metrics)
    monkeypatch.setattr(loadgen, "_wait_daser_drained", fake_daser_drain)
    monkeypatch.setattr(
        loadgen, "_run_vllm_phase_requests", fake_run_vllm_phase_requests
    )
    monkeypatch.setattr(loadgen.asyncio, "sleep", fake_sleep)

    manifest = BenchmarkManifest(
        run_id="test",
        backend="daser",
        reuse_mode="prefix",
        model="model",
        store_dir="/store",
        l1_size_bytes=1,
        l2_size_bytes=1,
        skip_l2=True,
        endpoints={
            "vllm": ServiceEndpoint("http://127.0.0.1:8001"),
            "daser": ServiceEndpoint("http://127.0.0.1:2026"),
        },
        log_dir="/logs",
        pid_file="/pids.json",
    )

    await run_daser_prefix(
        manifest=manifest,
        samples=[
            BenchmarkSample(
                sample_id=1,
                dataset="longbench",
                context="ctx",
                question="q",
                answers=[],
            )
        ],
        tokenizer=_Tokenizer(),
        max_inflight=1,
        gen_params={"max_tokens": 1},
        timeout=1.0,
    )

    assert calls == ["phase", "drain", "phase"]


async def test_lmcache_waits_for_quiescence_without_extra_sleep(monkeypatch) -> None:
    """LMCache warm starts after quiescence without an additional fixed delay."""
    calls: list[str] = []

    class Tokenizer(_Tokenizer):
        pad_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
            assert add_special_tokens is False
            return {"input_ids": self.encode(text, add_special_tokens=False)}

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(char) % 97 for char in text]

    async def fake_collect_phase_metrics(*_args, **_kwargs):
        return {}

    async def fake_wait_lmcache_quiescent(*_args, **_kwargs):
        calls.append("drain")

    async def fake_sleep(_seconds):
        calls.append("sleep")

    async def fake_vllm_completion_stream(
        _client,
        _url,
        sample,
        prompt,
        _gen_params,
        sem,
        _timeout,
    ):
        del prompt
        async with sem:
            calls.append("request")
        return RequestResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            generated_text="x",
            ttft_ms=1.0,
            latency_ms=2.0,
            prompt_tokens=8,
            completion_tokens=1,
        )

    import benchmarks.utils.loadgen as loadgen

    monkeypatch.setattr(loadgen, "collect_phase_metrics", fake_collect_phase_metrics)
    monkeypatch.setattr(loadgen, "_wait_lmcache_quiescent", fake_wait_lmcache_quiescent)
    monkeypatch.setattr(loadgen, "vllm_completion_stream", fake_vllm_completion_stream)
    monkeypatch.setattr(loadgen.asyncio, "sleep", fake_sleep)

    manifest = BenchmarkManifest(
        run_id="test",
        backend="lmcache",
        reuse_mode="prefix",
        model="model",
        store_dir="/store",
        l1_size_bytes=1,
        l2_size_bytes=1,
        skip_l2=True,
        endpoints={"vllm": ServiceEndpoint("http://127.0.0.1:8001")},
        log_dir="/logs",
        pid_file="/pids.json",
    )

    await run_lmcache(
        manifest=manifest,
        samples=[
            BenchmarkSample(
                sample_id=1,
                dataset="longbench",
                context="ctx",
                question="q",
                answers=[],
            )
        ],
        tokenizer=Tokenizer(),
        max_inflight=1,
        gen_params={"max_tokens": 1},
        timeout=1.0,
        settle_seconds=10.0,
        chunk_aligned_prompts=True,
    )

    assert calls == ["request", "drain", "request"]


def test_add_phase_comparison_records_cold_warm_correctness() -> None:
    """IMDB-style service results include cold/warm exact-match correctness."""
    result = {
        "cold": {
            "requests": [
                {
                    "sample_id": 1,
                    "dataset": "imdb",
                    "generated_text": "positive",
                    "ttft_ms": 1.0,
                    "latency_ms": 2.0,
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "error": None,
                    "cache_hits": 0,
                    "cache_chunks_total": 0,
                }
            ]
        },
        "warm": {
            "requests": [
                {
                    "sample_id": 1,
                    "dataset": "imdb",
                    "generated_text": "positive",
                    "ttft_ms": 1.0,
                    "latency_ms": 2.0,
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "error": None,
                    "cache_hits": 1,
                    "cache_chunks_total": 1,
                }
            ]
        },
    }

    _add_phase_comparison(result)

    assert result["correctness"]["cold_warm_exact_match"]["accuracy"] == 1.0


def test_parse_size_bytes_accepts_plain_bytes() -> None:
    """Size parser accepts byte strings emitted by the shell prepare step."""
    assert parse_size_bytes("2048") == 2048
    assert parse_size_bytes("2gib") == 2 * 1024**3


def test_no_evict_l1_slot_capacity_covers_workload() -> None:
    """No-evict sizing keeps DaseR L1 slot capacity above the workload."""
    total_blocks = 56363
    slot_size = 2359296
    sizing = derive_benchmark_sizing(
        total_blocks=total_blocks,
        max_prompt_blocks=1358,
        slot_size=slot_size,
        mode=COMPARISON_IOURING_MEM,
        evict=False,
        capacity_limits=BenchmarkCapacityLimits(
            max_l1_bytes=256 * BYTES_PER_GIB,
            max_l2_bytes=512 * BYTES_PER_GIB,
            memory_available_bytes=512 * BYTES_PER_GIB,
            disk_available_bytes=1024 * BYTES_PER_GIB,
        ),
    )

    assert sizing.daser_l1_bytes == sizing.daser_l2_bytes
    assert sizing.daser_l1_bytes // slot_size >= int(total_blocks * 1.5)
    assert sizing.lmcache_cpu_gb == 186


def test_run_bench_entrypoint_hides_manual_cache_size_flags() -> None:
    """The e2e benchmark entrypoint derives cache sizes from workload and evict."""
    script = (REPO_ROOT / "benchmarks" / "run_bench.sh").read_text()

    assert "--max-l1-size" not in script
    assert "--max-l2-size" not in script
    assert "--l1-size)" not in script
    assert "--l2-size)" not in script


def test_start_process_records_cuda_visible_devices(tmp_path: Path) -> None:
    """Server process metadata records the selected CUDA device."""
    manager = ServerManager(
        run_id="run1",
        backend="vllm",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
    )

    proc = manager._start(  # noqa: SLF001
        [sys.executable, "-c", "import time; time.sleep(0.1)"],
        "test.log",
    )
    proc.wait(timeout=5)
    manager._write_pids()  # noqa: SLF001

    payload = json.loads((tmp_path / "pids.json").read_text())
    assert payload[0]["cuda_visible_devices"] == "2"


def test_vllm_start_uses_vllm_generation_config(tmp_path: Path) -> None:
    """Benchmark vLLM servers ignore model sampling defaults."""
    manager = ServerManager(
        run_id="run1",
        backend="vllm",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
    )

    command = manager.vllm_command(None)
    assert command[command.index("--generation-config") + 1] == "vllm"


def test_vllm_start_can_override_max_num_batched_tokens(tmp_path: Path) -> None:
    """Benchmark vLLM servers can override the scheduler token budget."""
    manager = ServerManager(
        run_id="run1",
        backend="vllm",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
    )

    command = manager.vllm_command(None)
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"


def test_lmcache_metrics_use_http_server_endpoint() -> None:
    """LMCache MP metrics are exposed by the HTTP server, not port 9090."""
    manifest = BenchmarkManifest(
        run_id="run1",
        backend="lmcache",
        reuse_mode="none",
        model="/models/qwen",
        store_dir="/bench",
        l1_size_bytes=1024,
        l2_size_bytes=2048,
        skip_l2=False,
        endpoints={"vllm": ServiceEndpoint(url="http://127.0.0.1:8001")},
        log_dir="/bench/logs",
        pid_file="/bench/pids.json",
    )

    assert lmcache_metrics_url(manifest) == "http://127.0.0.1:8080"


def test_non_daser_manifest_preserves_requested_reuse_mode(tmp_path: Path) -> None:
    """All backends need the benchmark reuse mode for identical load shaping."""
    manager = ServerManager(
        run_id="run1",
        backend="lmcache",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
        reuse_mode="prefix",
    )

    assert manager.manifest().reuse_mode == "prefix"


def test_start_process_prefers_current_repo_on_pythonpath(tmp_path: Path) -> None:
    """Benchmark child services import DaseR from the current checkout first."""
    manager = ServerManager(
        run_id="run1",
        backend="vllm",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
    )

    proc = manager._start(  # noqa: SLF001
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['PYTHONPATH'].split(os.pathsep)[0])",
        ],
        "pythonpath.log",
    )
    proc.wait(timeout=5)

    assert (tmp_path / "logs" / "pythonpath.log").read_text().strip() == str(REPO_ROOT)


async def test_wait_healthy_fails_when_startup_process_exits(
    tmp_path: Path,
) -> None:
    """Health waiting fails if the process exits before becoming healthy."""
    manager = ServerManager(
        run_id="run1",
        backend="vllm",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024,
        l2_size_bytes=2048,
    )
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])

    try:
        try:
            await manager._wait_healthy(  # noqa: SLF001
                "http://127.0.0.1:9", "/health", 5.0, proc
            )
        except RuntimeError as exc:
            assert "startup process exited" in str(exc)
        else:
            raise AssertionError("startup exit was not detected")
    finally:
        proc.wait(timeout=5)
