# SPDX-License-Identifier: Apache-2.0
"""Unit tests for unified benchmark dataset, prompt, and manifest utilities."""

# Standard
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from benchmarks.bench_load import (
    _add_phase_comparison,
    _backend_server_hit_rate,
    _common_config_for_run,
    _effective_max_context_tokens,
    _generation_params,
    _serialise_phase,
    _should_dedup_context,
)
from benchmarks.run_bench import (
    BackendRun,
    RunBenchArgs,
    _bench_prepare_config,
    _collect_vllm_bench_phase_metrics,
    _compare_vllm_bench_outputs,
    _expand_backend_runs,
    _normalise_vllm_bench_result,
    _probe_daser_metrics,
    _run_command,
    _should_probe_daser_metrics,
    _stage_title,
    _validate_backend_runs,
    _vllm_bench_command,
    parse_args,
    run_benchmark,
)

# First Party
from benchmarks.utils.constants import BYTES_PER_GIB, COMPARISON_IOURING_MEM
from benchmarks.utils.datasets import BenchmarkSample, ImdbDataset, LongBenchDataset
from benchmarks.utils.loadgen import (
    PhaseResult,
    RequestResult,
    _metric_hit_ratios,
    _wait_lmcache_quiescent,
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


def test_default_context_limit_uses_model_length_minus_generation_budget() -> None:
    """Unspecified context limits avoid prompts that vLLM would reject."""

    class Tokenizer:
        model_max_length = 40960

    assert (
        _effective_max_context_tokens(
            explicit_max_context_tokens=0,
            gen_max_tokens=128,
            tokenizer=Tokenizer(),
        )
        == 40832
    )


def test_default_context_limit_prefers_model_config_position_limit() -> None:
    """Model config limits override larger tokenizer sentinels."""

    class Tokenizer:
        model_max_length = 131072

    class ModelConfig:
        max_position_embeddings = 40960

    assert (
        _effective_max_context_tokens(
            explicit_max_context_tokens=0,
            gen_max_tokens=1,
            tokenizer=Tokenizer(),
            model_config=ModelConfig(),
        )
        == 40959
    )


def test_explicit_context_limit_overrides_model_length() -> None:
    """User-supplied context limits remain authoritative."""

    class Tokenizer:
        model_max_length = 40960

    assert (
        _effective_max_context_tokens(
            explicit_max_context_tokens=2048,
            gen_max_tokens=128,
            tokenizer=Tokenizer(),
        )
        == 2048
    )


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


def test_daser_summary_hit_rate_uses_token_counters() -> None:
    """DaseR backend comparison should use token-level cache hit ratio."""
    assert (
        _backend_server_hit_rate(
            {
                "daser_external_prefix": 0.93,
                "daser_prometheus_tokens": 0.75,
                "daser_prometheus_requests": 1.0,
            }
        )
        == 0.75
    )


def test_daser_summary_hit_rate_ignores_request_level_ratio() -> None:
    """DaseR request counters are diagnostics, not token-level hit ratio."""
    assert (
        _backend_server_hit_rate(
            {
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
        block_size=16,
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
        block_size=16,
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
        block_size=128,
    )
    path = tmp_path / "manifest.json"

    manifest.write(path)
    loaded = BenchmarkManifest.read(path)

    assert loaded == manifest


def test_manifest_read_defaults_legacy_block_size(tmp_path: Path) -> None:
    """Older benchmark manifests remain readable after block-size tracking."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "run1",
                "backend": "daser",
                "reuse_mode": "chunk",
                "model": "/models/qwen",
                "store_dir": str(tmp_path),
                "l1_size_bytes": 1024,
                "l2_size_bytes": 2048,
                "skip_l2": True,
                "endpoints": {
                    "vllm": {"url": "http://127.0.0.1:8001"},
                    "daser": {"url": "http://127.0.0.1:2026"},
                },
                "log_dir": str(tmp_path / "logs"),
                "pid_file": str(tmp_path / "pids.json"),
            }
        )
    )

    loaded = BenchmarkManifest.read(path)

    assert loaded.block_size == 16


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

    assert cmd[cmd.index("--l1-size-gb") + 1] == "1"
    assert cmd[cmd.index("--eviction-trigger-watermark") + 1] == "0.8"
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
        block_tokens=16,
    ):
        del prompts
        chunk_aligned_values.append(bool(chunk_aligned_prompts))
        assert block_tokens == 128
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
        block_size=128,
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


async def test_lmcache_quiescence_wait_can_exceed_old_fixed_sleep(
    monkeypatch,
) -> None:
    """LMCache warm-up waits for quiescence instead of a fixed 10s sleep."""
    polls = 0
    sleeps: list[float] = []

    busy_status = {
        "storage_manager": {
            "store_controller": {
                "pending_keys_count": 1,
                "in_flight_task_count": 0,
            },
            "prefetch_controller": {
                "submission_queue_size": 0,
                "pending_queue_size": 0,
                "in_flight_request_count": 0,
                "lookup_phase_count": 0,
                "load_phase_count": 0,
            },
        }
    }
    quiescent_status = {
        "storage_manager": {
            "store_controller": {
                "pending_keys_count": 0,
                "in_flight_task_count": 0,
            },
            "prefetch_controller": {
                "submission_queue_size": 0,
                "pending_queue_size": 0,
                "in_flight_request_count": 0,
                "lookup_phase_count": 0,
                "load_phase_count": 0,
            },
        }
    }

    async def fake_get_json(_client, _url):
        nonlocal polls
        polls += 1
        if polls <= 12:
            return busy_status
        return quiescent_status

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    now = -1.0

    def fake_monotonic() -> float:
        nonlocal now
        now += 1.0
        return now

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

    import benchmarks.utils.loadgen as loadgen

    monkeypatch.setattr(loadgen, "_get_json", fake_get_json)
    monkeypatch.setattr(loadgen.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(loadgen.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(loadgen.time, "monotonic", fake_monotonic)

    await _wait_lmcache_quiescent(
        BenchmarkManifest(
            run_id="test",
            backend="lmcache",
            reuse_mode="none",
            model="model",
            store_dir="/store",
            l1_size_bytes=1,
            l2_size_bytes=1,
            skip_l2=True,
            endpoints={"vllm": ServiceEndpoint("http://127.0.0.1:8001")},
            log_dir="/logs",
            pid_file="/pids.json",
        ),
        settle_seconds=10.0,
    )

    assert polls == 15
    assert len(sleeps) == 14


async def test_lmcache_quiescence_wait_times_out(monkeypatch) -> None:
    """LMCache warm-up should fail clearly instead of waiting forever."""
    sleeps: list[float] = []
    busy_status = {
        "storage_manager": {
            "store_controller": {
                "pending_keys_count": 1,
                "in_flight_task_count": 0,
            },
            "prefetch_controller": {
                "submission_queue_size": 0,
                "pending_queue_size": 0,
                "in_flight_request_count": 0,
                "lookup_phase_count": 0,
                "load_phase_count": 0,
            },
        }
    }

    async def fake_get_json(_client, _url):
        return busy_status

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    now = -1.0

    def fake_monotonic() -> float:
        nonlocal now
        now += 1.0
        return now

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

    import benchmarks.utils.loadgen as loadgen

    monkeypatch.setattr(loadgen, "_get_json", fake_get_json)
    monkeypatch.setattr(loadgen.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(loadgen.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(loadgen.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(loadgen, "_LMCACHE_QUIESCENCE_TIMEOUT_SECONDS", 3.0)

    with pytest.raises(TimeoutError, match="LMCache did not become quiescent"):
        await _wait_lmcache_quiescent(
            BenchmarkManifest(
                run_id="test",
                backend="lmcache",
                reuse_mode="none",
                model="model",
                store_dir="/store",
                l1_size_bytes=1,
                l2_size_bytes=1,
                skip_l2=True,
                endpoints={"vllm": ServiceEndpoint("http://127.0.0.1:8001")},
                log_dir="/logs",
                pid_file="/pids.json",
            ),
            settle_seconds=0.0,
        )

    assert len(sleeps) == 2


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
            memory_free_bytes=512 * BYTES_PER_GIB,
            disk_available_bytes=1024 * BYTES_PER_GIB,
        ),
    )

    assert sizing.daser_l1_bytes == sizing.daser_l2_bytes
    assert sizing.daser_l1_bytes // slot_size >= int(total_blocks * 1.5)
    assert sizing.lmcache_cpu_gb == 186


def test_run_bench_entrypoint_hides_manual_cache_size_flags() -> None:
    """The e2e benchmark entrypoint derives cache sizes from workload and evict."""
    script = (REPO_ROOT / "benchmarks" / "run_bench.py").read_text()

    assert "--max-l1-size" not in script
    assert "--max-l2-size" not in script
    assert 'parser.add_argument("--l1-size"' not in script
    assert 'parser.add_argument("--l2-size"' not in script


def test_run_bench_entrypoint_names_backend_matrix() -> None:
    """The e2e benchmark entrypoint exposes the full comparison matrix."""
    runs = _expand_backend_runs("all")

    assert [run.label for run in runs] == [
        "baseline",
        "lmcache",
        "daser-chunk",
        "daser-prefix",
    ]
    assert runs == [
        BackendRun("baseline", "vllm", "none"),
        BackendRun("lmcache", "lmcache", "none"),
        BackendRun("daser-chunk", "daser", "chunk"),
        BackendRun("daser-prefix", "daser", "prefix"),
    ]


def test_run_bench_parses_comma_separated_backend_rows() -> None:
    """The runner accepts a comma-separated subset of benchmark rows."""
    args = parse_args(
        [
            "--backend",
            "baseline,lmcache,daser-prefix",
            "--model",
            "/models/qwen",
            "--store-dir",
            "/data/zwt/daser_test/bench",
        ]
    )

    assert args.backend == "baseline,lmcache,daser-prefix"
    assert _expand_backend_runs(args.backend) == [
        BackendRun("baseline", "vllm", "none"),
        BackendRun("lmcache", "lmcache", "none"),
        BackendRun("daser-prefix", "daser", "prefix"),
    ]


def test_run_bench_backend_parser_rejects_legacy_aliases() -> None:
    """Backend choices stay limited to canonical benchmark row names."""
    for backend in ("vllm", "daser", "all-openai"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--backend",
                    backend,
                    "--model",
                    "/models/qwen",
                    "--store-dir",
                    "/data/zwt/daser_test/bench",
                ]
            )


def test_run_bench_parser_rejects_invalid_numeric_args() -> None:
    """Invalid benchmark numeric knobs fail before orchestration starts."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--backend",
                "baseline",
                "--model",
                "/models/qwen",
                "--store-dir",
                "/data/zwt/daser_test/bench",
                "--block-size",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--backend",
                "baseline",
                "--load-generator",
                "vllm-bench",
                "--model",
                "/models/qwen",
                "--store-dir",
                "/data/zwt/daser_test/bench",
                "--bench-request-rate",
                "nan",
            ]
        )


def test_run_bench_shell_entrypoint_is_removed() -> None:
    """The benchmark entrypoint should live in Python, not a shell wrapper."""
    assert not (REPO_ROOT / "benchmarks" / "run_bench.sh").exists()


def test_benchmark_docs_only_reference_python_runner() -> None:
    """Benchmark docs should not keep the removed shell entrypoint."""
    docs = "\n".join(
        [
            (REPO_ROOT / "benchmarks" / "README.md").read_text(),
            (REPO_ROOT / "docs" / "development.md").read_text(),
            (
                REPO_ROOT / "docs" / "optimizations" / "5_prefix_output1_ttft.md"
            ).read_text(),
        ]
    )

    assert "run_bench.sh" not in docs
    assert "python benchmarks/run_bench.py" in docs


def test_grafana_prometheus_datasource_uses_one_second_interval() -> None:
    """Grafana should query Prometheus at the configured scrape interval."""
    datasource = (
        REPO_ROOT
        / "deploy"
        / "monitoring"
        / "grafana"
        / "provisioning"
        / "datasources"
        / "prometheus.yml"
    ).read_text()

    assert "timeInterval: 1s" in datasource


def test_vllm_dashboard_does_not_span_idle_gaps() -> None:
    """vLLM latency panels should not keep showing stale benchmark values."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    for title in (
        "TTFT (Time To First Token)",
        "TPOT (Time Per Output Token)",
        "vLLM Request Rate",
    ):
        custom = panels[title]["fieldConfig"]["defaults"]["custom"]
        assert custom["spanNulls"] is False

    latency_panels = (
        panels["TTFT (Time To First Token)"],
        panels["TPOT (Time Per Output Token)"],
    )
    for panel in latency_panels:
        for target in panel["targets"]:
            expr = target["expr"]
            assert "[$__rate_interval]" in expr
            assert "[5m]" not in expr
            assert "and on()" in expr
            assert "sum(increase(vllm:request_success_total" in expr


def test_vllm_request_rate_panel_uses_aggregate_series() -> None:
    """vLLM request rate should render as one aggregate series."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    expr = panels["vLLM Request Rate"]["targets"][0]["expr"]

    assert expr.startswith("max_over_time")
    assert "sum(increase(vllm:request_success_total" in expr
    assert "[1m]" in expr
    assert "/ 60" in expr
    assert "max_over_time" in expr
    assert "[2m:1s]" in expr
    assert "or vector(0)" in expr


def test_dashboard_service_status_panels_use_scrape_health() -> None:
    """Service status panels should show DOWN when scrape targets have no data."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    assert panels["DaseR"]["targets"][0]["expr"] == 'max(up{job="daser"}) or vector(0)'
    assert panels["vLLM"]["targets"][0]["expr"] == 'max(up{job="vllm"}) or vector(0)'

    for title in ("DaseR", "vLLM"):
        mappings = panels[title]["fieldConfig"]["defaults"]["mappings"]
        assert mappings[0]["options"]["0"]["text"] == "DOWN"
        assert mappings[0]["options"]["1"]["text"] == "UP"


def test_daser_dashboard_hit_rate_panels_render_zero_for_missing_hits() -> None:
    """Hit-rate panels should not go empty when only miss counters exist."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    request_hit_rate = panels["Request Hit Rate"]["targets"][0]["expr"]
    l1_hit_rate = panels["L1 Hit Rate"]["targets"][0]["expr"]

    assert (
        request_hit_rate
        == '(sum(rate(daser_cache_lookup_total{job="daser", result="hit"}[5m])) '
        'or vector(0)) / clamp_min(sum(rate(daser_cache_lookup_total{job="daser"}'
        "[5m])), 1)"
    )
    assert "or vector(0)" in l1_hit_rate


def test_daser_dashboard_uses_fixed_daser_job_label() -> None:
    """DaseR panels should not depend on a browser-local job variable."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )

    assert dashboard.get("templating", {}).get("list", []) == []

    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            if "daser_" in expr:
                assert "$job" not in expr
                assert 'job="daser"' in expr


def test_daser_dashboard_transfer_latency_omits_p99() -> None:
    """Transfer latency should show p50 and p95 only."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    legends = {
        target["legendFormat"] for target in panels["Transfer Latency"]["targets"]
    }

    assert legends == {"p50 {{op}}", "p95 {{op}}"}


def test_daser_dashboard_does_not_span_idle_gaps() -> None:
    """DaseR panels should leave benchmark idle gaps disconnected."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    for title in (
        "Cache Lookups",
        "Prefix Reuse Distribution (tokens)",
        "Evictions & Late Commits",
        "Transfer Latency",
        "Throughput GB/s",
        "Chunk Size",
        "L1 Hit Rate",
        "L1 Usage",
    ):
        custom = panels[title]["fieldConfig"]["defaults"]["custom"]
        assert custom["spanNulls"] is False


def test_daser_dashboard_l1_panels_tolerate_missing_l1_misses() -> None:
    """L1 panels should use aggregate expressions that survive missing labels."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    hit_rate = panels["L1 Hit Rate"]["targets"][0]["expr"]
    usage = panels["L1 Usage"]["targets"][0]["expr"]

    assert "sum(increase(daser_l1_hits_total" in hit_rate
    assert "sum(increase(daser_l1_misses_total" in hit_rate
    assert "max_over_time" in hit_rate
    assert "[2m:1s]" in hit_rate
    assert "or vector(0)" in hit_rate
    assert "sum(daser_l1_bytes_used" in usage
    assert "sum(daser_l1_bytes_capacity" in usage
    assert "max_over_time" in usage
    assert "[2m:1s]" in usage
    assert "or vector(0)" in usage


def test_daser_dashboard_sparse_series_panels_render_zero_fallbacks() -> None:
    """Sparse benchmark panels should emit a visible zero series outside runs."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    l1_hit_rate = panels["L1 Hit Rate"]["targets"][0]["expr"]
    l1_usage = panels["L1 Usage"]["targets"][0]["expr"]
    request_rate = panels["vLLM Request Rate"]["targets"][0]["expr"]

    assert "increase(daser_l1_hits_total" in l1_hit_rate
    assert "increase(daser_l1_misses_total" in l1_hit_rate
    assert "max_over_time" in l1_hit_rate
    assert "[2m:1s]" in l1_hit_rate
    assert "or vector(0)" in l1_hit_rate
    assert "max_over_time" in l1_usage
    assert "[2m:1s]" in l1_usage
    assert "or vector(0)" in l1_usage
    assert "increase(vllm:request_success_total" in request_rate
    assert "max_over_time" in request_rate
    assert "[2m:1s]" in request_rate
    assert "/ 60" in request_rate
    assert "or vector(0)" in request_rate


def test_daser_dashboard_sparse_panels_use_visible_series_styling() -> None:
    """Sparse panels should make short benchmark samples visible."""
    dashboard = json.loads(
        (
            REPO_ROOT
            / "deploy"
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "daser-overview.json"
        ).read_text()
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    for title in ("L1 Hit Rate", "L1 Usage", "vLLM Request Rate"):
        panel = panels[title]
        custom = panel["fieldConfig"]["defaults"]["custom"]
        assert panel["options"]["legend"]["displayMode"] == "list"
        assert custom["showPoints"] == "auto"
        assert custom["lineWidth"] >= 2
        assert custom["fillOpacity"] == 0


def test_daser_metrics_probe_reports_prometheus_scrape_state(
    monkeypatch, capsys
) -> None:
    """The DaseR probe should report whether Prometheus scraped the target."""
    calls: list[str] = []

    class _Response:
        def __init__(self, text: str, payload: dict[str, object] | None = None) -> None:
            self.status_code = 200
            self.text = text
            self._payload = payload

        def json(self) -> dict[str, object]:
            assert self._payload is not None
            return self._payload

    def fake_get(url: str, **_kwargs: object) -> _Response:
        calls.append(url)
        if url == "http://127.0.0.1:2026/metrics":
            return _Response("daser_up 1\n")
        if url.endswith("/api/v1/targets"):
            return _Response(
                "",
                {
                    "status": "success",
                    "data": {
                        "activeTargets": [
                            {
                                "labels": {"job": "daser"},
                                "scrapeUrl": "http://host.docker.internal:2026/metrics",
                                "health": "up",
                                "lastError": "",
                            }
                        ]
                    },
                },
            )
        if url.endswith("/api/v1/query"):
            return _Response(
                "",
                {
                    "status": "success",
                    "data": {
                        "result": [
                            {
                                "metric": {"job": "daser"},
                                "value": [123.0, "1"],
                            }
                        ]
                    },
                },
            )
        raise AssertionError(url)

    monkeypatch.setattr("benchmarks.run_bench.httpx.get", fake_get)
    monkeypatch.setattr("benchmarks.run_bench.time.sleep", lambda _seconds: None)

    _probe_daser_metrics(
        BenchmarkManifest(
            run_id="run1",
            backend="daser",
            reuse_mode="prefix",
            model="/models/qwen",
            store_dir="/data/zwt/daser_bench/run1/daser-prefix",
            l1_size_bytes=1024,
            l2_size_bytes=2048,
            skip_l2=True,
            endpoints={
                "daser": ServiceEndpoint("http://127.0.0.1:2026"),
                "vllm": ServiceEndpoint("http://127.0.0.1:8001"),
            },
            log_dir="/data/zwt/daser_bench/run1/daser-prefix/logs",
            pid_file="/data/zwt/daser_bench/run1/daser-prefix/pids.json",
        ),
        phase="startup",
        prometheus_url="http://127.0.0.1:9090",
        settle_seconds=0,
    )

    captured = capsys.readouterr().out
    assert "daser_metrics_startup_status: ready" in captured
    assert (
        "prometheus_daser_target_startup: health=up "
        "scrape_url=http://host.docker.internal:2026/metrics" in captured
    )
    assert "prometheus_daser_up_startup: value=1" in captured
    assert "http://127.0.0.1:9090/api/v1/targets" in calls


def test_run_bench_python_entrypoint_prints_backend_progress(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Python orchestration prints readable stage separators and summaries."""
    run_root = tmp_path / "run_20260102_030405"
    commands: list[list[str]] = []
    metric_probe_labels: list[str] = []

    def fake_run_command(command: list[str]) -> None:
        commands.append(command)
        if "--prepare-only" in command:
            out_path = Path(command[command.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "derived_l1_size_bytes": 1024,
                            "derived_l2_size_bytes": 2048,
                        }
                    }
                )
            )
            return
        if any(item.endswith("bench_start_servers.py") for item in command):
            store_dir = Path(command[command.index("--store-dir") + 1])
            store_dir.mkdir(parents=True, exist_ok=True)
            backend = command[command.index("--backend") + 1]
            label = store_dir.name
            reuse_mode = "none"
            if backend == "daser":
                reuse_mode = command[command.index("--cache-reuse-mode") + 1]
            endpoints = {"vllm": {"url": "http://127.0.0.1:8001"}}
            if backend == "daser":
                endpoints["daser"] = {"url": "http://127.0.0.1:2026"}
            (store_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run1",
                        "backend": backend,
                        "reuse_mode": reuse_mode,
                        "model": "/models/qwen",
                        "store_dir": str(store_dir),
                        "l1_size_bytes": 1024,
                        "l2_size_bytes": 2048,
                        "skip_l2": True,
                        "endpoints": endpoints,
                        "log_dir": str(store_dir / "logs"),
                        "pid_file": str(store_dir / "pids.json"),
                    }
                )
            )
            return
        out_path = Path(command[command.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        label = out_path.parent.name
        phase_result = {
            "result": {
                "warm": {
                    "summary": {
                        "ttft_ms_mean": 12.5,
                        "phase_elapsed_ms": 222.0,
                        "phase_prompt_tok_per_s": 345.6,
                        "backend_server_cache_hit_rate": 0.75,
                        "answer_contains_accuracy": 1.0,
                    }
                }
            }
        }
        if label == "baseline":
            phase_result = {
                "result": {
                    "baseline": {
                        "summary": {
                            "ttft_ms_mean": 30.0,
                            "phase_elapsed_ms": 333.0,
                            "phase_prompt_tok_per_s": 111.0,
                            "answer_contains_accuracy": 0.5,
                        }
                    }
                }
            }
        elif label == "lmcache":
            phase_result = {
                "result": {
                    "cold": {
                        "summary": {
                            "ttft_ms_mean": 40.0,
                            "phase_elapsed_ms": 444.0,
                            "answer_contains_accuracy": 0.25,
                        }
                    },
                    "warm": {
                        "summary": {
                            "ttft_ms_mean": 12.5,
                            "phase_elapsed_ms": 222.0,
                            "phase_prompt_tok_per_s": 345.6,
                            "backend_server_cache_hit_rate": 0.75,
                            "answer_contains_accuracy": 1.0,
                        }
                    },
                },
                "correctness": {
                    "cold_warm_exact_match": {
                        "available": True,
                        "accuracy": 0.9,
                    }
                },
            }
        elif label == "daser-chunk":
            phase_result = {
                "result": {
                    "cold": {
                        "uploaded_documents": 1,
                        "upload_ms": 55.0,
                    },
                    "warm": {
                        "summary": {
                            "ttft_ms_mean": 10.0,
                            "phase_elapsed_ms": 111.0,
                            "phase_prompt_tok_per_s": 456.0,
                            "backend_server_cache_hit_rate": 0.8,
                            "answer_contains_accuracy": 1.0,
                        }
                    },
                }
            }
        elif label == "daser-prefix":
            phase_result = {
                "result": {
                    "cold": {
                        "summary": {
                            "ttft_ms_mean": 35.0,
                            "phase_elapsed_ms": 350.0,
                            "answer_contains_accuracy": 0.75,
                        }
                    },
                    "warm": {
                        "summary": {
                            "ttft_ms_mean": 9.0,
                            "phase_elapsed_ms": 99.0,
                            "phase_prompt_tok_per_s": 567.0,
                            "backend_server_cache_hit_rate": 0.85,
                            "answer_contains_accuracy": 1.0,
                        }
                    },
                },
                "correctness": {
                    "cold_warm_exact_match": {
                        "available": True,
                        "accuracy": 1.0,
                    }
                },
            }
        out_path.write_text(json.dumps(phase_result))

    monkeypatch.setattr("benchmarks.run_bench._run_command", fake_run_command)
    monkeypatch.setattr(
        "benchmarks.run_bench.stop_from_pid_file",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "benchmarks.run_bench.time.strftime",
        lambda _fmt: "20260102_030405",
    )
    monkeypatch.setattr(
        "benchmarks.run_bench._probe_daser_metrics",
        lambda *_args, **kwargs: metric_probe_labels.append(kwargs["phase"]),
    )

    result = run_benchmark(
        RunBenchArgs(
            backend="all",
            model="/models/qwen",
            store_dir=str(tmp_path),
            dataset="longbench",
            longbench_dir="/data/longbench",
            datasets="triviaqa",
            max_samples=1,
            block_size=128,
        )
    )

    captured = capsys.readouterr().out
    assert result == run_root
    assert "== PREPARE ==" in captured
    assert "== BASELINE START ==" in captured
    assert "== DASER-PREFIX START ==" in captured
    assert "== DASER-PREFIX COLD/WARM LOAD ==" in captured
    assert "reuse_mode: none" not in captured
    assert "== COMPARISON SUMMARY ==" in captured
    assert "baseline:" in captured
    assert "baseline_ttft_ms_mean: 30.0" in captured
    assert "lmcache:" in captured
    assert "cold_ttft_ms_mean: 40.0" in captured
    assert "warm_ttft_ms_mean: 12.5" in captured
    assert "cold_warm_exact_match_available: True" in captured
    assert "cold_warm_exact_match_accuracy: 0.9" in captured
    assert "daser-chunk:" in captured
    assert "cold_uploaded_documents: 1" in captured
    assert "cold_upload_ms: 55.0" in captured
    assert "daser-prefix:" in captured
    assert "warm_prompt_tok_per_s: 567.0" in captured
    assert "warm_answer_contains_accuracy: 1.0" in captured
    assert f"run_root: {run_root}" in captured
    assert any(command[1] == "benchmarks/bench_load.py" for command in commands)
    block_size_commands = [
        command
        for command in commands
        if any(
            item.endswith(
                (
                    "bench_load.py",
                    "bench_start_servers.py",
                )
            )
            for item in command
        )
    ]
    assert block_size_commands
    for command in block_size_commands:
        assert command[command.index("--block-size") + 1] == "128"
    start_commands = [
        command
        for command in commands
        if any(item.endswith("bench_start_servers.py") for item in command)
    ]
    for command in start_commands:
        backend = command[command.index("--backend") + 1]
        if backend in ("vllm", "lmcache"):
            assert "--cache-reuse-mode" not in command
        else:
            assert "--cache-reuse-mode" in command
    assert "[bench] start backend" not in captured
    assert metric_probe_labels == [
        "startup",
        "post-load",
        "startup",
        "post-load",
    ]


def test_run_bench_stage_title_formats_backend_names() -> None:
    """Stage titles are compact visual separators."""
    assert _stage_title("daser-prefix", "cold/warm load") == (
        "== DASER-PREFIX COLD/WARM LOAD =="
    )


def test_run_bench_explicit_openai_subset_excludes_daser_chunk() -> None:
    """Comma-separated OpenAI-compatible rows can omit DaseR chunk."""
    runs = _expand_backend_runs("baseline,lmcache,daser-prefix")

    assert runs == [
        BackendRun("baseline", "vllm", "none"),
        BackendRun("lmcache", "lmcache", "none"),
        BackendRun("daser-prefix", "daser", "prefix"),
    ]


def test_vllm_bench_rejects_chunk_backends() -> None:
    """vLLM bench load generation cannot exercise DaseR chunk endpoints."""
    with pytest.raises(ValueError, match="daser-chunk"):
        _validate_backend_runs(
            _expand_backend_runs("all"),
            load_generator="vllm-bench",
        )
    with pytest.raises(ValueError, match="daser-chunk"):
        _validate_backend_runs(
            _expand_backend_runs("daser-chunk"),
            load_generator="vllm-bench",
        )


def test_run_benchmark_validates_direct_args_before_creating_run_dir(
    tmp_path: Path,
) -> None:
    """Direct RunBenchArgs callers get the same preflight validation as CLI."""
    with pytest.raises(ValueError, match="block_size"):
        run_benchmark(
            RunBenchArgs(
                backend="baseline",
                model="/models/qwen",
                store_dir=str(tmp_path),
                block_size=0,
            )
        )

    assert list(tmp_path.iterdir()) == []


def test_vllm_bench_prepare_config_uses_synthetic_lengths(tmp_path: Path) -> None:
    """Synthetic vLLM bench sizing uses configured input length and block size."""
    args = RunBenchArgs(
        backend="baseline,lmcache,daser-prefix",
        model="/models/qwen",
        store_dir=str(tmp_path),
        load_generator="vllm-bench",
        block_size=128,
        bench_num_prompts=10,
        bench_input_len=1024,
        bench_output_len=1,
        bench_random_prefix_len=128,
        bench_random_range_ratio=0.5,
    )

    config = _bench_prepare_config(args, tmp_path)

    assert config["dataset"] == "vllm-bench-random"
    assert config["num_samples"] == 10
    assert config["block_size"] == 128
    assert config["bench_input_len"] == 1024
    assert config["bench_output_len"] == 1
    assert config["max_prompt_tokens"] == 1664
    assert config["max_prompt_blocks"] == 13
    assert config["total_blocks"] == 130


def test_vllm_bench_command_uses_random_dataset(tmp_path: Path) -> None:
    """vLLM bench commands target completions with deterministic random load."""
    args = RunBenchArgs(
        backend="daser-prefix",
        model="/models/qwen",
        store_dir=str(tmp_path),
        load_generator="vllm-bench",
        bench_num_prompts=12,
        bench_input_len=4096,
        bench_output_len=2,
        bench_max_concurrency=4,
        bench_request_rate="inf",
        bench_seed=123,
        bench_burstiness=2.0,
        bench_random_prefix_len=256,
        bench_random_range_ratio=0.25,
    )
    raw_path = tmp_path / "raw.json"

    command = _vllm_bench_command(
        args,
        ServiceEndpoint("http://127.0.0.1:8001"),
        raw_path,
    )

    assert command[:3] == ["vllm", "bench", "serve"]
    assert command[command.index("--backend") + 1] == "openai"
    assert command[command.index("--base-url") + 1] == "http://127.0.0.1:8001"
    assert command[command.index("--endpoint") + 1] == "/v1/completions"
    assert command[command.index("--dataset-name") + 1] == "random"
    assert command[command.index("--num-prompts") + 1] == "12"
    assert command[command.index("--input-len") + 1] == "4096"
    assert command[command.index("--output-len") + 1] == "2"
    assert command[command.index("--max-concurrency") + 1] == "4"
    assert command[command.index("--request-rate") + 1] == "inf"
    assert command[command.index("--seed") + 1] == "123"
    assert command[command.index("--burstiness") + 1] == "2.0"
    assert command[command.index("--temperature") + 1] == "0.0"
    assert command[command.index("--top-p") + 1] == "1.0"
    assert command[command.index("--percentile-metrics") + 1] == "ttft,tpot,itl,e2el"
    assert "--save-detailed" in command
    assert command[command.index("--random-prefix-len") + 1] == "256"
    assert command[command.index("--random-range-ratio") + 1] == "0.25"
    assert command[command.index("--result-filename") + 1] == raw_path.name


def test_vllm_bench_normalises_result_summary(tmp_path: Path) -> None:
    """vLLM bench JSON metrics are converted into benchmark summaries."""
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "completed": 10,
                "total_input_tokens": 10240,
                "total_output_tokens": 10,
                "benchmark_duration": 2.5,
                "mean_ttft_ms": 100.0,
                "mean_tpot_ms": 3.0,
                "mean_e2el_ms": 120.0,
            }
        )
    )

    summary = _normalise_vllm_bench_result(raw)

    assert summary["num_requests"] == 10
    assert summary["ttft_ms_mean"] == 100.0
    assert summary["latency_ms_mean"] == 120.0
    assert summary["phase_elapsed_ms"] == 2500.0
    assert summary["phase_prompt_tok_per_s"] == 4096.0
    assert summary["prompt_tokens_total"] == 10240
    assert summary["completion_tokens_total"] == 10


def test_vllm_bench_compares_detailed_outputs(tmp_path: Path) -> None:
    """Detailed vLLM bench JSON enables cold/warm exact-match correctness."""
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    cold.write_text(
        json.dumps(
            {
                "completed": 3,
                "generated_texts": ["A", "B", "C"],
                "errors": [None, None, None],
            }
        )
    )
    warm.write_text(
        json.dumps(
            {
                "completed": 3,
                "generated_texts": ["A", "X", "C"],
                "errors": [None, None, None],
            }
        )
    )

    correctness = _compare_vllm_bench_outputs(cold, warm)

    assert correctness == {
        "cold_warm_exact_match": {
            "available": True,
            "matches": 2,
            "total": 3,
            "accuracy": 2 / 3,
            "length_mismatch": False,
        }
    }


def test_vllm_bench_compares_outputs_detail_shape(tmp_path: Path) -> None:
    """vLLM bench output-list details also support correctness comparison."""
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    cold.write_text(
        json.dumps({"outputs": [{"generated_text": "A"}, {"generated_text": "B"}]})
    )
    warm.write_text(
        json.dumps({"outputs": [{"generated_text": "A"}, {"generated_text": "X"}]})
    )

    correctness = _compare_vllm_bench_outputs(cold, warm)

    assert correctness["cold_warm_exact_match"]["available"] is True
    assert correctness["cold_warm_exact_match"]["matches"] == 1
    assert correctness["cold_warm_exact_match"]["total"] == 2
    assert correctness["cold_warm_exact_match"]["accuracy"] == 0.5
    assert correctness["cold_warm_exact_match"]["length_mismatch"] is False


def test_vllm_bench_correctness_counts_length_mismatch(tmp_path: Path) -> None:
    """Missing cold/warm outputs should lower exact-match accuracy."""
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    cold.write_text(json.dumps({"generated_texts": ["A", "B", "C"]}))
    warm.write_text(json.dumps({"generated_texts": ["A", "B"]}))

    correctness = _compare_vllm_bench_outputs(cold, warm)

    assert correctness["cold_warm_exact_match"]["available"] is True
    assert correctness["cold_warm_exact_match"]["matches"] == 2
    assert correctness["cold_warm_exact_match"]["total"] == 3
    assert correctness["cold_warm_exact_match"]["accuracy"] == 2 / 3
    assert correctness["cold_warm_exact_match"]["length_mismatch"] is True


def test_vllm_bench_correctness_marks_missing_details_unavailable(
    tmp_path: Path,
) -> None:
    """Missing detailed outputs should not look like a zero-sample pass."""
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    cold.write_text(json.dumps({"completed": 2}))
    warm.write_text(json.dumps({"completed": 2}))

    correctness = _compare_vllm_bench_outputs(cold, warm)

    assert correctness["cold_warm_exact_match"]["available"] is False
    assert correctness["cold_warm_exact_match"]["total"] == 0
    assert correctness["cold_warm_exact_match"]["accuracy"] is None
    assert "reason" in correctness["cold_warm_exact_match"]


def test_vllm_bench_phase_metrics_report_backend_token_hit_rate(monkeypatch) -> None:
    """vLLM bench phases record backend token hit rates from metric deltas."""
    before_metrics = {
        "vllm_prometheus": {},
        "backend_prometheus": {
            "daser_cache_requested_tokens_total": 1000,
            "daser_cache_matched_tokens_total": 100,
        },
        "backend_status": {},
    }
    after_metrics = {
        "vllm_prometheus": {},
        "backend_prometheus": {
            "daser_cache_requested_tokens_total": 3000,
            "daser_cache_matched_tokens_total": 1600,
            "daser_external_prefix_cache_queries_total": 10,
            "daser_external_prefix_cache_hits_total": 10,
            'daser_cache_lookup_total{result="hit"}': 10,
            "daser_cache_lookup_total": 10,
        },
        "backend_status": {},
    }

    async def fake_collect(
        manifest: BenchmarkManifest,
        before: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del manifest
        if before is None:
            return before_metrics
        return {
            "vllm_prometheus": {},
            "backend_prometheus": {
                key: value - before["backend_prometheus"].get(key, 0.0)
                for key, value in after_metrics["backend_prometheus"].items()
            },
            "backend_status": {},
            "hit_ratios": {
                "daser_prometheus_tokens": 0.75,
                "daser_external_prefix_cache_queries_total": 10,
                "daser_external_prefix_cache_hits_total": 10,
                "daser_external_prefix": 1.0,
                "daser_prometheus_requests": 1.0,
            },
        }

    manifest = BenchmarkManifest(
        run_id="run1",
        backend="daser",
        reuse_mode="prefix",
        model="/models/qwen",
        store_dir="/bench",
        l1_size_bytes=1024,
        l2_size_bytes=2048,
        skip_l2=True,
        endpoints={"vllm": ServiceEndpoint("http://127.0.0.1:8001")},
        log_dir="/bench/logs",
        pid_file="/bench/pids.json",
    )
    monkeypatch.setattr(
        "benchmarks.run_bench.collect_phase_metrics",
        fake_collect,
    )

    metrics, hit_rate = _collect_vllm_bench_phase_metrics(manifest, before_metrics)

    assert metrics["backend_prometheus"]["daser_cache_requested_tokens_total"] == 2000
    assert metrics["backend_prometheus"]["daser_cache_matched_tokens_total"] == 1500
    assert metrics["hit_ratios"]["daser_prometheus_tokens"] == 0.75
    assert metrics["hit_ratios"]["daser_external_prefix"] == 1.0
    assert metrics["hit_ratios"]["daser_prometheus_requests"] == 1.0
    assert hit_rate == 0.75


def test_run_bench_vllm_bench_entrypoint_runs_openai_rows(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """vLLM bench mode starts OpenAI-compatible rows and writes a summary."""
    run_root = tmp_path / "run_20260102_030405"
    commands: list[list[str]] = []
    lmcache_waits: list[str] = []

    def fake_run_command(command: list[str]) -> None:
        commands.append(command)
        if any(item.endswith("bench_start_servers.py") for item in command):
            store_dir = Path(command[command.index("--store-dir") + 1])
            store_dir.mkdir(parents=True, exist_ok=True)
            backend = command[command.index("--backend") + 1]
            reuse_mode = "none"
            if backend == "daser":
                reuse_mode = command[command.index("--cache-reuse-mode") + 1]
            endpoints = {"vllm": {"url": "http://127.0.0.1:8001"}}
            if backend == "daser":
                endpoints["daser"] = {"url": "http://127.0.0.1:2026"}
            (store_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run1",
                        "backend": backend,
                        "reuse_mode": reuse_mode,
                        "model": "/models/qwen",
                        "store_dir": str(store_dir),
                        "l1_size_bytes": 1024,
                        "l2_size_bytes": 2048,
                        "skip_l2": True,
                        "endpoints": endpoints,
                        "log_dir": str(store_dir / "logs"),
                        "pid_file": str(store_dir / "pids.json"),
                        "block_size": 128,
                    }
                )
            )
            return
        if command[:3] == ["vllm", "bench", "serve"]:
            result_dir = Path(command[command.index("--result-dir") + 1])
            result_name = command[command.index("--result-filename") + 1]
            backend_label = result_dir.name
            generated_texts = ["A", "B", "C"]
            if (
                backend_label == "daser-prefix"
                and result_name == "vllm_bench_warm.json"
            ):
                generated_texts = ["A", "B", "X"]
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / result_name).write_text(
                json.dumps(
                    {
                        "duration": 2.0,
                        "completed": 3,
                        "failed": 0,
                        "total_input_tokens": 3072,
                        "total_output_tokens": 3,
                        "mean_ttft_ms": 11.0,
                        "mean_e2el_ms": 13.0,
                        "generated_texts": generated_texts,
                        "errors": [None, None, None],
                    }
                )
            )

    monkeypatch.setattr("benchmarks.run_bench._run_command", fake_run_command)
    monkeypatch.setattr("benchmarks.run_bench.stop_from_pid_file", lambda _path: None)
    monkeypatch.setattr(
        "benchmarks.run_bench._wait_with_message",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("vLLM bench LMCache warm wait must not use fixed sleep")
        ),
    )
    monkeypatch.setattr(
        "benchmarks.run_bench.time.strftime",
        lambda _fmt: "20260102_030405",
    )
    monkeypatch.setattr("benchmarks.run_bench._drain_daser", lambda _manifest: None)
    monkeypatch.setattr(
        "benchmarks.run_bench._probe_daser_metrics",
        lambda *_args, **_kwargs: None,
    )

    async def fake_wait_lmcache_quiescent(
        manifest: BenchmarkManifest,
        settle_seconds: float,
    ) -> None:
        lmcache_waits.append(f"{manifest.backend}:{settle_seconds}")

    monkeypatch.setattr(
        "benchmarks.run_bench._wait_lmcache_quiescent",
        fake_wait_lmcache_quiescent,
    )

    async def fake_collect_phase_metrics(
        manifest: BenchmarkManifest,
        before_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del before_metrics
        if manifest.backend == "lmcache":
            return {
                "vllm_prometheus": {},
                "backend_prometheus": {
                    "lmcache_mp_lookup_requested_tokens_total": 3000.0,
                    "lmcache_mp_lookup_hit_tokens_total": 2400.0,
                },
                "backend_status": {},
                "hit_ratios": {"lmcache_prometheus_lookup": 0.8},
            }
        if manifest.backend == "daser":
            return {
                "vllm_prometheus": {},
                "backend_prometheus": {
                    "daser_cache_requested_tokens_total": 3000.0,
                    "daser_cache_matched_tokens_total": 2550.0,
                },
                "backend_status": {},
                "hit_ratios": {"daser_prometheus_tokens": 0.85},
            }
        return {
            "vllm_prometheus": {},
            "backend_prometheus": {},
            "backend_status": {},
            "hit_ratios": {},
        }

    monkeypatch.setattr(
        "benchmarks.run_bench.collect_phase_metrics",
        fake_collect_phase_metrics,
    )

    result = run_benchmark(
        RunBenchArgs(
            backend="baseline,lmcache,daser-prefix",
            load_generator="vllm-bench",
            model="/models/qwen",
            store_dir=str(tmp_path),
            block_size=128,
            bench_num_prompts=3,
            bench_input_len=1024,
            bench_output_len=1,
            bench_max_concurrency=2,
        )
    )

    captured = capsys.readouterr().out
    assert result == run_root
    assert "load_generator: vllm-bench" in captured
    assert "bench_input_len: 1024" in captured
    assert "baseline_ttft_ms_mean: 11.0" in captured
    assert "warm_ttft_ms_mean: 11.0" in captured
    assert "warm_backend_cache_hit_rate: 0.8" in captured
    assert "warm_backend_cache_hit_rate: 0.85" in captured
    assert "lmcache_warm_wait: quiescent" in captured
    assert "cold_warm_exact_match_accuracy: 1.0" in captured
    assert "cold_warm_exact_match_accuracy: 0.6666666666666666" in captured
    assert lmcache_waits == ["lmcache:0.0"]
    assert len([cmd for cmd in commands if cmd[:3] == ["vllm", "bench", "serve"]]) == 5
    for command in commands:
        if command[:3] == ["vllm", "bench", "serve"]:
            assert "--save-detailed" in command
    assert not any(
        "daser-chunk" in str(item) for command in commands for item in command
    )
    lmcache = json.loads((run_root / "lmcache" / "results.json").read_text())
    daser_prefix = json.loads((run_root / "daser-prefix" / "results.json").read_text())
    assert lmcache["correctness"]["cold_warm_exact_match"]["accuracy"] == 1.0
    assert daser_prefix["correctness"]["cold_warm_exact_match"]["accuracy"] == 2 / 3
    assert lmcache["result"]["warm"]["summary"]["backend_server_cache_hit_rate"] == 0.8
    assert (
        daser_prefix["result"]["warm"]["summary"]["backend_server_cache_hit_rate"]
        == 0.85
    )


def test_run_bench_probes_daser_metrics_only_for_daser_backends() -> None:
    """Only DaseR benchmark rows need a DaseR metrics readiness probe."""
    assert _should_probe_daser_metrics(BackendRun("daser", "daser", "chunk"))
    assert _should_probe_daser_metrics(BackendRun("daser-prefix", "daser", "prefix"))
    assert not _should_probe_daser_metrics(BackendRun("baseline", "vllm", "none"))
    assert not _should_probe_daser_metrics(BackendRun("lmcache", "lmcache", "none"))


def test_run_bench_command_prints_before_subprocess(capsys) -> None:
    """Subprocess wrapper prints the exact command before running it."""
    _run_command(["python", "-c", "print('ok')"])

    captured = capsys.readouterr().out
    assert "[bench] run: python -c" in captured


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


def test_server_commands_propagate_custom_block_size(tmp_path: Path) -> None:
    """Custom benchmark block size reaches vLLM, LMCache, and DaseR."""
    manager = ServerManager(
        run_id="run1",
        backend="daser",
        model="/models/qwen",
        store_dir=tmp_path,
        gpu_id="2",
        gpu_util=0.85,
        max_num_seqs=32,
        l1_size_bytes=1024**3,
        l2_size_bytes=2 * 1024**3,
        block_size=128,
    )

    vllm_command = manager.vllm_command(None)
    daser_command = manager._daser_server_command()  # noqa: SLF001
    lmcache_command = manager._lmcache_mp_server_command()  # noqa: SLF001

    assert vllm_command[vllm_command.index("--block-size") + 1] == "128"
    assert daser_command[daser_command.index("--block-tokens") + 1] == "128"
    assert lmcache_command[lmcache_command.index("--chunk-size") + 1] == "128"
    assert manager.manifest().block_size == 128


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


def test_non_daser_manifest_uses_no_reuse_mode(tmp_path: Path) -> None:
    """Non-DaseR backend manifests should not expose DaseR reuse modes."""
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

    assert manager.manifest().reuse_mode == "none"


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
