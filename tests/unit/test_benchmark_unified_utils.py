# SPDX-License-Identifier: Apache-2.0
"""Unit tests for unified benchmark dataset, prompt, and manifest utilities."""

# Standard
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.bench_load import _add_phase_comparison

# First Party
from benchmarks.utils.datasets import ImdbDataset, LongBenchDataset
from benchmarks.utils.loadgen import (
    RequestResult,
    lmcache_metrics_url,
    summarise_results,
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
    build_document_prompt,
    build_full_prompt,
)
from benchmarks.utils.servers import (
    REPO_ROOT,
    BenchmarkManifest,
    ServerManager,
    ServiceEndpoint,
)
from benchmarks.utils.sizing import parse_size_bytes


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
    """DaseR no-evict starts with --skip-l2 while retaining logical slots."""
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
    assert "--l2-size" in cmd
    assert cmd[cmd.index("--l2-size") + 1] == "2048"
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
            )
        ]
    )

    assert summary["num_requests"] == 1
    assert summary["ttft_ms_mean"] == 10.0
    assert summary["cache_hit_rate"] == 0.5


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
