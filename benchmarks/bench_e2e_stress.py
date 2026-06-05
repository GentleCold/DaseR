# SPDX-License-Identifier: Apache-2.0
"""Standalone LongBench benchmark — compare vLLM vs vLLM+LMCache vs vLLM+DaseR.

Starts servers via subprocess, uploads documents / sends completions, and
scores generated answers against ground truth.  Decoupled from the DaseR
codebase so it is not affected by DaseR internal refactors.

Usage::

    python bench_e2e_stress.py --mode vllm --model /path/to/model
    python bench_e2e_stress.py --mode lmcache --model /path/to/model
    python bench_e2e_stress.py --mode daser --model /path/to/model
    python bench_e2e_stress.py --mode all --model /path/to/model
"""

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio
import collections
from dataclasses import dataclass
import functools
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from typing import Any

# Third Party
import httpx

# Force unbuffered output for background runs
print = functools.partial(print, flush=True)  # type: ignore[assignment]

# Force HuggingFace offline mode — the model is always local
os.environ["HF_HUB_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a helpful assistant answering questions using the following documents.\n\n"
)
DOC_SEPARATOR: str = "\n\n---\n\n"
DOCS_MARKER: str = "__DASER_DOCUMENTS__"
BLOCK_TOKENS: int = 16

_DEFAULT_LMCACHE_MP_HOST = "tcp://localhost"
_DEFAULT_LMCACHE_MP_PORT = 5555
_DEFAULT_DATASETS: list[str] = [
    "2wikimqa",  # 多跳QA, p50=25K chars
    "hotpotqa_e",  # 多跳QA, p50=40K chars, 300 samples
    "2wikimqa_e",  # 多跳QA easy版, 300 samples
    "musique",  # 多跳QA, 200 samples
    "triviaqa",  # 百科QA, 200 samples
]
# L1 (pinned CPU) must be ≤ L2 (SSD) per DaseR constraint
_DEFAULT_L1_BYTES = 256 * 1024**3  # 256 GiB pinned / CPU memory
_DEFAULT_L2_BYTES = 300 * 1024**3  # 300 GiB SSD


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One benchmark sample loaded from a JSONL dataset."""

    sample_id: int
    dataset: str
    context: str
    question: str  # the "input" field
    answers: list[str]


@dataclass
class RequestResult:
    """Result of a single inference request."""

    sample_id: int
    dataset: str
    generated_text: str
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None
    # DaseR profiling fields
    server_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_chunks_total: int = 0


@dataclass
class DatasetMetrics:
    """Aggregated metrics for one dataset."""

    dataset: str
    num_samples: int
    num_errors: int
    accuracy_contains: float
    ttft_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p99: float
    latency_ms_mean: float
    prompt_tokens_total: int
    completion_tokens_total: int


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_size_bytes(value: str) -> int:
    """Parse a human-readable byte size (e.g. '10gb', '512mib')."""
    units = {
        "": 1,
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z]*)\s*", value)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")
    number = int(m.group(1))
    unit = m.group(2).lower()
    if unit not in units:
        raise argparse.ArgumentTypeError(f"unsupported size unit: {unit}")
    return number * units[unit]


def _auto_gpu() -> str:
    """Return the GPU index with the most free memory via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "0"
    best = (0, -1)
    for line in out.strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        idx, free_mb = int(parts[0].strip()), int(parts[1].strip())
        if free_mb > best[1]:
            best = (idx, free_mb)
    return str(best[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Standalone LongBench benchmark: vLLM vs LMCache vs DaseR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python bench_e2e_stress.py --mode vllm --model /path/to/model
              python bench_e2e_stress.py --mode daser --model /path/to/model \\
                  --datasets narrativeqa,2wikimqa
              python bench_e2e_stress.py --mode all --model /path/to/model \\
                  --max-inflight 200
        """),
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["vllm", "lmcache", "daser", "all"],
        help="Which comparison to run (or 'all' for sequential runs of all three).",
    )
    p.add_argument("--model", required=True, help="HF model path served by vLLM.")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing LongBench JSONL files.",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names (default: all JSONL files in --data-dir).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max samples per dataset (0 = all).",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=32,
        help="Max concurrent in-flight requests.",
    )
    p.add_argument("--gpu-id", default=None, help="GPU device index (default: auto).")
    p.add_argument(
        "--gpu-util",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization.",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="vLLM max_num_seqs.",
    )
    p.add_argument(
        "--l2-size",
        default=str(_DEFAULT_L2_BYTES),
        type=_parse_size_bytes,
        help="DaseR L2 / LMCache disk capacity (e.g. 90gb, 5gib).",
    )
    p.add_argument(
        "--l1-size",
        default=None,
        type=_parse_size_bytes,
        help="DaseR L1 / LMCache CPU capacity (default: 256gib).",
    )
    p.add_argument(
        "--gpu-monitor-secs",
        type=float,
        default=15.0,
        help="Interval in seconds for GPU utilisation logging (0 = disable).",
    )
    p.add_argument(
        "--store-dir",
        required=True,
        help="Scratch directory for KV store / LMCache disk files.",
    )
    p.add_argument("--vllm-port", type=int, default=8001)
    p.add_argument("--daser-port", type=int, default=2026)
    p.add_argument("--gen-max-tokens", type=int, default=128)
    p.add_argument("--gen-temperature", type=float, default=0.0)
    p.add_argument("--output", default=None, help="JSON results file path.")
    p.add_argument(
        "--no-prefill",
        action="store_true",
        help="Skip DaseR document prefill (docs already cached).",
    )
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for vLLM / DaseR to become healthy.",
    )
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=0,
        help="Filter out samples whose prompt exceeds this many tokens "
        "(0 = no limit). Qwen3-8B has 40960 max; set 40000 to be safe.",
    )
    p.add_argument(
        "--no-dedup-context",
        action="store_true",
        help="Disable context deduplication (every sample runs, "
        "even if multiple samples share the same context).",
    )
    p.add_argument(
        "--cache-reuse-mode",
        default="chunk",
        choices=["chunk", "prefix"],
        help="DaseR connector cache reuse strategy (chunk or prefix).",
    )
    p.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep servers running after benchmark (for debugging).",
    )
    p.add_argument(
        "--socket-path",
        required=True,
        help="Unix domain socket path for DaseR IPC.",
    )
    args = p.parse_args(argv)

    # Resolve defaults that depend on other args
    if args.l1_size is None:
        args.l1_size = _DEFAULT_L1_BYTES
    if args.gpu_id is None:
        args.gpu_id = _auto_gpu()
    if args.datasets is None:
        args.datasets = list(_DEFAULT_DATASETS)
    else:
        args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"longbench_{args.mode}_{ts}.json"
    return args


def _list_datasets(data_dir: str) -> list[str]:
    """Return sorted list of dataset names (stem of each .jsonl file)."""
    p = Path(data_dir)
    if not p.is_dir():
        return []
    names = sorted(f.stem for f in p.glob("*.jsonl"))
    return names


async def _gpu_monitor(gpu_id: str, interval_s: float, label: str) -> None:
    """Periodically log GPU utilisation via nvidia-smi.

    Runs until cancelled.  Logs a one-line summary each *interval_s*
    seconds so the operator can confirm the GPU is saturated.
    """
    gpu_idx = int(gpu_id)
    while True:
        await asyncio.sleep(interval_s)
        try:
            out = await asyncio.to_thread(
                subprocess.check_output,
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
        except Exception:
            continue
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            if int(parts[0]) == gpu_idx:
                print(
                    f"  [gpu:{label}] util={parts[1]}% mem_util={parts[2]}% "
                    f"mem={parts[3]}/{parts[4]} MiB temp={parts[5]}C power={parts[6]}W"
                )
                break


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class ServerManager:
    """Start / stop vLLM and DaseR server processes."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._procs: list[subprocess.Popen[bytes]] = []
        self._log_dir = Path(args.store_dir) / "logs"

    # ---- helpers -----------------------------------------------------------

    @property
    def vllm_url(self) -> str:
        return f"http://127.0.0.1:{self._args.vllm_port}"

    @property
    def daser_url(self) -> str:
        return f"http://127.0.0.1:{self._args.daser_port}"

    def _start(
        self, cmd: list[str], log_name: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.Popen[bytes]:
        """Start a subprocess, tee stdout/stderr to a log file."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / log_name
        fh = log_path.open("wb")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(self._args.gpu_id)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._procs.append(proc)
        print(f"  [started] {' '.join(cmd[:4])}... (log: {log_path})")
        return proc

    async def _wait_healthy(
        self,
        url: str,
        timeout: float = 120.0,
        health_path: str = "/health",
    ) -> None:
        """Poll GET {url}{health_path} until HTTP 200 or timeout."""
        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None
        health_url = f"{url}{health_path}"
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    r = await client.get(health_url, timeout=5.0)
                    if r.status_code == 200:
                        print(f"  [healthy] {url}")
                        return
                except Exception as exc:
                    last_exc = exc
                await asyncio.sleep(2.0)
        detail = f": {last_exc}" if last_exc else ""
        raise RuntimeError(f"{health_url} not healthy after {timeout:.0f}s{detail}")

    # ---- start methods -----------------------------------------------------

    async def start_lmcache_mp_server(self) -> None:
        """Start the LMCache MP (multi-process) cache server."""
        args = self._args
        l1_gb = int(args.l1_size / (1024**3))
        scratch = Path(args.store_dir) / "lmcache_mp_disk"
        scratch.mkdir(parents=True, exist_ok=True)
        cmd = [
            "lmcache",
            "server",
            "--host",
            "localhost",
            "--port",
            str(_DEFAULT_LMCACHE_MP_PORT),
            "--chunk-size",
            str(BLOCK_TOKENS),
            "--max-workers",
            "4",
            "--l1-size-gb",
            str(l1_gb),
            "--eviction-policy",
            "LRU",
            "--l2-adapter",
            json.dumps({"type": "fs", "base_path": str(scratch)}),
            "--http-port",
            "8080",
        ]
        self._start(cmd, "lmcache_mp_server.log")
        # LMCache MP server exposes /healthcheck on its HTTP port
        await self._wait_healthy(
            "http://127.0.0.1:8080",
            timeout=self._args.startup_timeout,
            health_path="/healthcheck",
        )

    async def start_vllm_only(self) -> None:
        """Start vLLM with no KV connector."""
        args = self._args
        cmd = [
            "vllm",
            "serve",
            args.model,
            "--port",
            str(args.vllm_port),
            "--gpu-memory-utilization",
            str(args.gpu_util),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--no-enable-prefix-caching",
        ]
        self._start(cmd, "vllm_vanilla.log")
        await self._wait_healthy(self.vllm_url, timeout=self._args.startup_timeout)

    async def start_vllm_lmcache(self) -> None:
        """Start vLLM with LMCacheMPConnector pointing at the MP server."""
        args = self._args
        kv_config = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": _DEFAULT_LMCACHE_MP_HOST,
                "lmcache.mp.port": _DEFAULT_LMCACHE_MP_PORT,
            },
        }
        env = {"PYTHONHASHSEED": "42"}
        cmd = [
            "vllm",
            "serve",
            args.model,
            "--port",
            str(args.vllm_port),
            "--kv-transfer-config",
            json.dumps(kv_config),
            "--gpu-memory-utilization",
            str(args.gpu_util),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--no-enable-prefix-caching",
        ]
        self._start(cmd, "vllm_lmcache.log", extra_env=env)
        await self._wait_healthy(self.vllm_url, timeout=self._args.startup_timeout)

    async def start_vllm_daser(self) -> None:
        """Start vLLM with DaserConnector."""
        args = self._args
        kv_config = {
            "kv_connector": "DaserConnector",
            "kv_connector_module_path": "daser.connector.daser_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "socket_path": args.socket_path,
                "cache_reuse_mode": args.cache_reuse_mode,
            },
        }
        cmd = [
            "vllm",
            "serve",
            args.model,
            "--port",
            str(args.vllm_port),
            "--kv-transfer-config",
            json.dumps(kv_config),
            "--gpu-memory-utilization",
            str(args.gpu_util),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--no-enable-prefix-caching",
        ]
        self._start(cmd, "vllm_daser.log")
        await self._wait_healthy(self.vllm_url, timeout=self._args.startup_timeout)

    async def start_daser_server(self) -> None:
        """Start the DaseR HTTP + IPC server."""
        args = self._args
        store = Path(args.store_dir) / "daser"
        store.mkdir(parents=True, exist_ok=True)
        # Remove stale store/index files from previous runs (size may differ)
        for stale in ["daser.store", "daser.index", "daser.index.tmp_store"]:
            p = store / stale
            if p.exists():
                p.unlink(missing_ok=True)
        # Remove stale IPC socket
        socket = Path(args.socket_path)
        if socket.exists():
            socket.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "daser.server",
            "--vllm-base-url",
            self.vllm_url,
            "--model-path",
            args.model,
            "--store-dir",
            str(store),
            "--l2-size",
            str(args.l2_size),
            "--l1-size",
            str(args.l1_size),
            "--transfer-mode",
            "iouring",
            "--cache-reuse-mode",
            args.cache_reuse_mode,
            "--host",
            "0.0.0.0",
            "--port",
            str(args.daser_port),
            "--socket-path",
            args.socket_path,
        ]
        self._start(cmd, "daser.log", extra_env={"DASER_LOG_LEVEL": "DEBUG"})
        await self._wait_healthy(self.daser_url, timeout=self._args.startup_timeout)

    # ---- stop ---------------------------------------------------------------

    async def stop_all(self) -> None:
        """SIGTERM all processes, wait, then SIGKILL."""
        for proc in reversed(self._procs):
            if proc.poll() is not None:
                continue
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        # Wait up to 15s for graceful shutdown
        deadline = time.monotonic() + 15.0
        for proc in self._procs:
            timeout = max(0.1, deadline - time.monotonic())
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        # Force-kill stragglers
        for proc in self._procs:
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        self._procs.clear()

    # ---- cleanup ------------------------------------------------------------

    def cleanup_scratch(self) -> None:
        """Remove generated store/index/log files."""
        if self._args.keep_alive:
            return
        for sub in ["daser", "lmcache_disk", "lmcache_mp_disk", "lmcache_cpu"]:
            path = Path(self._args.store_dir) / sub
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        socket = Path(self._args.socket_path)
        if socket.exists():
            socket.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(
    data_dir: str,
    dataset_names: list[str],
    max_samples: int = 0,
    max_context_tokens: int = 0,
    model_path: str | None = None,
) -> dict[str, list[Sample]]:
    """Load JSONL files, return {dataset_name: [Sample, ...]}.

    When *max_context_tokens* > 0 and *model_path* is set, samples whose
    full prompt exceeds the token limit are filtered out at load time.
    """
    tokenizer = None
    if max_context_tokens > 0 and model_path:
        tokenizer = _lazy_tokenizer(model_path)

    samples_by_ds: dict[str, list[Sample]] = {}
    global_id = 0
    total_filtered = 0
    for ds_name in dataset_names:
        path = Path(data_dir) / f"{ds_name}.jsonl"
        if not path.is_file():
            print(f"  [warn] dataset file not found: {path}")
            continue
        ds_samples: list[Sample] = []
        ds_filtered = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Normalise fields
                context = str(obj.get("context", ""))
                question = str(obj.get("input", ""))
                answers = obj.get("answers", [])
                if isinstance(answers, str):
                    answers = [answers]
                answers = [str(a) for a in answers]

                if tokenizer is not None:
                    try:
                        prompt = build_full_prompt(tokenizer, context, question)
                    except Exception:
                        ds_filtered += 1
                        continue
                    n_tokens = len(
                        tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    )
                    if n_tokens > max_context_tokens:
                        ds_filtered += 1
                        continue

                ds_samples.append(
                    Sample(
                        sample_id=global_id,
                        dataset=ds_name,
                        context=context,
                        question=question,
                        answers=answers,
                    )
                )
                global_id += 1
                if max_samples > 0 and len(ds_samples) >= max_samples:
                    break
        if ds_filtered:
            print(
                f"  [filtered] {ds_name}: {ds_filtered} samples "
                f"exceed {max_context_tokens} token limit"
            )
            total_filtered += ds_filtered
        if ds_samples:
            samples_by_ds[ds_name] = ds_samples
            print(f"  [loaded] {ds_name}: {len(ds_samples)} samples")
    if total_filtered:
        print(
            f"  [filtered] total: {total_filtered} samples filtered by context length"
        )
    return samples_by_ds


def _interleave_samples(
    samples_by_ds: dict[str, list[Sample]],
    seed: int = 42,
) -> list[Sample]:
    """Interleave samples from all datasets to eliminate flattening-order bias.

    When datasets are flattened in dict insertion order, samples from later
    datasets always queue behind earlier ones. With a small ``max_num_seqs``
    this creates systemic TTFT bias: later datasets include queue wait time
    from all earlier long-context requests.

    This function shuffles each dataset's samples independently (so relative
    ordering within a dataset is preserved only up to randomisation) then
    round-robins across datasets, yielding a fair interleaving.
    """
    rng = random.Random(seed)
    shuffled = {}
    for ds_name, samples in samples_by_ds.items():
        s = list(samples)
        rng.shuffle(s)
        shuffled[ds_name] = s

    flat: list[Sample] = []
    ds_names = sorted(shuffled.keys())
    indices = {ds: 0 for ds in ds_names}
    while True:
        added = False
        for ds in ds_names:
            if indices[ds] < len(shuffled[ds]):
                flat.append(shuffled[ds][indices[ds]])
                indices[ds] += 1
                added = True
        if not added:
            break
    return flat


def _dedup_by_context(
    samples_by_ds: dict[str, list[Sample]],
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Return a copy with at most one sample per unique context text.

    When multiple samples share the same context (common in RAG benchmarks
    like narrativeqa), DaseR reuses the cached KV while vLLM/LMCache
    re-prefill from scratch.  Deduplicating eliminates this asymmetry so
    all three modes pay the same prefill cost per sample.

    Samples are first shuffled within each dataset then interleaved, so
    the surviving sample for each context is the one that happens to land
    earliest in the interleaved order.
    """
    interleaved = _interleave_samples(samples_by_ds, seed=seed)
    seen: set[str] = set()
    deduped: list[Sample] = []
    skipped = 0
    for s in interleaved:
        if s.context in seen:
            skipped += 1
            continue
        seen.add(s.context)
        deduped.append(s)

    # Rebuild per-dataset dicts
    result: dict[str, list[Sample]] = {}
    for s in deduped:
        result.setdefault(s.dataset, []).append(s)

    if skipped:
        print(
            f"  [dedup] {skipped} samples with duplicate contexts excluded, "
            f"{len(deduped)} unique remaining"
        )
    return result


# ---------------------------------------------------------------------------
# Prompt building (self-contained, mirrors DaseR HTTP layer logic)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _lazy_tokenizer(model_path: str) -> Any:
    """Lazily load a HuggingFace tokenizer (cached)."""
    from transformers import AutoTokenizer  # Third Party

    return AutoTokenizer.from_pretrained(model_path)


def _render_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool
) -> str:
    """Render messages with the tokenizer chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        )
    body = ""
    for msg in messages:
        body += f"{msg['role']}: {msg['content']}\n"
    if add_generation_prompt:
        body += "assistant: "
    return body


def build_full_prompt(
    tokenizer: Any,
    context: str,
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Build a single full prompt by inlining context in the chat template.

    Produces the same prompt that DaseR builds internally, except the
    document text is placed directly rather than via the docs API.
    """
    user_content = f"Documents:\n{DOCS_MARKER}\n\nTask: {question}"
    rendered = _render_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
    )
    if rendered.count(DOCS_MARKER) != 1:
        raise RuntimeError(f"Chat template marker {DOCS_MARKER!r} not unique")
    prefix, suffix = rendered.split(DOCS_MARKER, 1)
    return prefix + context + suffix


# ---------------------------------------------------------------------------
# HTTP clients (async)
# ---------------------------------------------------------------------------


async def _daser_upload_doc(
    client: httpx.AsyncClient,
    daser_url: str,
    title: str,
    text: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Upload one document to DaseR POST /documents, return JSON response."""
    _t0 = time.perf_counter()
    async with sem:
        r = await client.post(
            f"{daser_url}/documents",
            json={"title": title, "text": text},
        )
        r.raise_for_status()
        result = r.json()
    _elapsed_ms = (time.perf_counter() - _t0) * 1000
    _ntokens = len(text) // 4  # rough estimate: ~4 chars per token
    print(f"  [upload] {title[:80]:<80} ntokens~{_ntokens:>6}  {_elapsed_ms:>8.1f}ms")
    return result


async def _daser_infer(
    client: httpx.AsyncClient,
    daser_url: str,
    doc_ids: list[str],
    task: str,
    gen_params: dict[str, Any] | None,
    sem: asyncio.Semaphore,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], float]:
    """POST /infer to DaseR, return (json_body, wall_ms)."""
    body: dict[str, Any] = {
        "doc_ids": doc_ids,
        "task": task,
        "use_kv_cache": True,
        "trace_cache": True,
    }
    if gen_params:
        body["gen_params"] = gen_params
    t0 = time.perf_counter()
    async with sem:
        r = await client.post(
            f"{daser_url}/infer",
            json=body,
            timeout=httpx.Timeout(timeout),
        )
        r.raise_for_status()
    wall_ms = (time.perf_counter() - t0) * 1000
    result = r.json()
    print(
        f"  [infer] {task[:60]:<60} wall={wall_ms:.0f}ms "
        f"ttft={result.get('ttft_ms', 0):.0f}ms"
    )
    return result, wall_ms


async def _vllm_completion_stream(
    client: httpx.AsyncClient,
    vllm_url: str,
    prompt: str,
    gen_params: dict[str, Any] | None,
    sem: asyncio.Semaphore,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], float, float]:
    """Send a streaming completion to vLLM /v1/completions.

    Returns (result_dict, ttft_ms, wall_ms) where result_dict has
    'choices', 'usage' keys mirroring the OpenAI shape.
    """
    payload: dict[str, Any] = {
        "model": "",  # vLLM ignores model when only one is served
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if gen_params:
        payload.update(gen_params)
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    t0 = time.perf_counter()
    first_token_at: float | None = None
    async with sem:
        async with client.stream(
            "POST",
            f"{vllm_url}/v1/completions",
            json=payload,
            timeout=httpx.Timeout(timeout),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                if chunk.get("usage") is not None:
                    usage = dict(chunk["usage"])
                for choice in chunk.get("choices", []):
                    fragment = str(choice.get("text", ""))
                    if not fragment:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(fragment)
    wall_ms = (time.perf_counter() - t0) * 1000
    ttft_ms = ((first_token_at or time.perf_counter()) - t0) * 1000
    return (
        {
            "choices": [{"text": "".join(text_parts)}],
            "usage": usage,
        },
        ttft_ms,
        wall_ms,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and strip text for comparison."""
    return text.strip().lower()


def _check_contains(generated: str, answers: list[str]) -> bool:
    """True if any answer appears as a substring in generated text."""
    g = _normalise(generated)
    return any(_normalise(a) in g for a in answers)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile via linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f]) if c != f else s[f]


def _compute_metrics(
    ds_name: str,
    results: list[RequestResult],
    answers_by_id: dict[int, list[str]],
) -> DatasetMetrics:
    """Aggregate metrics for one dataset."""
    ttfts = [r.ttft_ms for r in results if r.error is None]
    latencies = [r.latency_ms for r in results if r.error is None]
    prompt_total = sum(r.prompt_tokens for r in results)
    completion_total = sum(r.completion_tokens for r in results)
    num_errors = sum(1 for r in results if r.error is not None)

    contains_hits = 0
    for r in results:
        if r.error:
            continue
        answers = answers_by_id.get(r.sample_id, [])
        if _check_contains(r.generated_text, answers):
            contains_hits += 1
    n = max(1, len(results) - num_errors)

    return DatasetMetrics(
        dataset=ds_name,
        num_samples=len(results),
        num_errors=num_errors,
        accuracy_contains=contains_hits / n if n > 0 else 0.0,
        ttft_ms_mean=statistics.mean(ttfts) if ttfts else 0.0,
        ttft_ms_p50=_percentile(ttfts, 50),
        ttft_ms_p99=_percentile(ttfts, 99),
        latency_ms_mean=statistics.mean(latencies) if latencies else 0.0,
        prompt_tokens_total=prompt_total,
        completion_tokens_total=completion_total,
    )


def _print_summary(all_metrics: list[DatasetMetrics], mode: str) -> None:
    """Print a human-readable summary table."""
    print(f"\n{'=' * 80}")
    print(f"  Mode: {mode}")
    print(f"{'=' * 80}")
    header = (
        f"{'Dataset':<24} {'#':>4} {'Err':>4} "
        f"{'Contains':>9} "
        f"{'TTFT_mean':>10} {'TTFT_p50':>9} {'TTFT_p99':>9} "
        f"{'Lat_mean':>10}"
    )
    print(header)
    print("-" * len(header))
    total_contains = 0.0
    total_samples = 0
    for m in all_metrics:
        print(
            f"{m.dataset:<24} {m.num_samples:>4} {m.num_errors:>4} "
            f"{m.accuracy_contains:>8.1%} "
            f"{m.ttft_ms_mean:>9.1f} {m.ttft_ms_p50:>8.1f} {m.ttft_ms_p99:>8.1f} "
            f"{m.latency_ms_mean:>9.1f}"
        )
        total_contains += m.accuracy_contains * m.num_samples
        total_samples += m.num_samples

    if total_samples > 0:
        print("-" * len(header))
        print(
            f"{'AGGREGATE':<24} {total_samples:>4} {'':>4} "
            f"{total_contains / total_samples:>8.1%}"
        )


def _print_error_summary(results: list[RequestResult], label: str) -> None:
    """Print the most common error messages from a run."""
    errors = [r.error for r in results if r.error is not None]
    if not errors:
        return
    counter = collections.Counter(errors)
    print(f"\n  [{label}] error summary ({len(errors)} total):")
    for msg, count in counter.most_common(5):
        # Truncate long messages
        short = str(msg)[:160]
        print(f"    {count:>4}x  {short}")


def _print_daser_profile(results: list[RequestResult]) -> None:
    """Print DaseR-specific profiling breakdown."""
    ok = [r for r in results if r.error is None]
    if not ok or ok[0].server_latency_ms == 0.0:
        return
    n = len(ok)
    srv_lat = [r.server_latency_ms for r in ok]
    srv_mean = statistics.mean(srv_lat)
    ttft = [r.ttft_ms for r in ok]
    ttft_mean = statistics.mean(ttft)
    wall_vs_srv = ttft_mean - srv_mean
    hit_rates = [
        r.cache_hits / max(1, r.cache_chunks_total)
        for r in ok
        if r.cache_chunks_total > 0
    ]
    total_cache_ok = sum(r.cache_hits for r in ok)
    total_cache_chunks = sum(r.cache_chunks_total for r in ok)
    print(f"\n{'=' * 80}")
    print(f"  DaseR profiling (n={n})")
    print(f"{'=' * 80}")
    print(f"  {'TTFT (client)':<30} {ttft_mean:>10.1f} ms")
    print(f"  {'Server latency':<30} {srv_mean:>10.1f} ms")
    print(
        f"  {'TTFT − server latency':<30} {wall_vs_srv:>10.1f} ms  ← HTTP + IPC + queue"
    )
    print(
        f"  {'Cache hit chunks':<30} {total_cache_ok:>10} / {total_cache_chunks} "
        f"({100 * total_cache_ok / max(1, total_cache_chunks):.0f}%)"
    )
    if hit_rates:
        print(
            f"  {'Cache hit rate (mean)':<30} {100 * statistics.mean(hit_rates):>9.1f}%"
        )
    print()


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


async def _wait_disk_quiesce(dir_path: Path, timeout_s: float = 60.0) -> None:
    """Wait until file count in *dir_path* stabilises.

    Polls every 2 seconds; considers the directory stable when the file
    count is unchanged for three consecutive polls, or when *timeout_s*
    elapses (whichever comes first).
    """
    deadline = time.monotonic() + timeout_s
    stable_count = 0
    last_count = -1
    while time.monotonic() < deadline:
        if dir_path.exists():
            count = len(list(dir_path.iterdir()))
        else:
            count = 0
        if count == last_count and count > 0:
            stable_count += 1
            if stable_count >= 3:
                print(f"  [quiesce] stable at {count} files")
                return
        else:
            stable_count = 0
            last_count = count
        await asyncio.sleep(2.0)
    if last_count > 0:
        print(f"  [quiesce] timeout after {timeout_s:.0f}s, last count={last_count}")


async def _run_daser(
    args: argparse.Namespace,
    sm: ServerManager,
    samples_by_ds: dict[str, list[Sample]],
) -> list[DatasetMetrics]:
    """Run DaseR mode: start servers, upload docs, infer, score.

    In chunk mode, documents are uploaded via the DaseR HTTP API and inference
    goes through ``/infer``. In prefix mode, all unique contexts are
    concatenated into a single shared prefix and prompts are sent directly to
    vLLM's ``/v1/completions``, so the DaseR connector caches the shared
    prefix KV on the first request and reuses it on subsequent requests.
    """
    print("\n--- Starting vLLM + DaseR ---")
    await sm.start_vllm_daser()
    await sm.start_daser_server()

    all_samples: list[Sample] = _interleave_samples(samples_by_ds)

    gen_params = {
        "max_tokens": args.gen_max_tokens,
        "temperature": args.gen_temperature,
    }

    if args.cache_reuse_mode == "prefix":
        return await _run_daser_prefix_shared(
            args,
            sm,
            all_samples,
            samples_by_ds,
            gen_params,
        )

    # ---- Chunk mode: upload docs then infer via HTTP ----

    context_to_doc: dict[str, dict[str, Any]] = {}
    for s in all_samples:
        if s.context not in context_to_doc:
            context_to_doc[s.context] = {}

    # ---- Phase 1: Upload documents ----
    if not args.no_prefill:
        print(f"\n--- Uploading {len(context_to_doc)} documents ---")
        sem = asyncio.Semaphore(args.max_inflight)
        upload_timeout = httpx.Timeout(args.timeout)
        async with httpx.AsyncClient(timeout=upload_timeout) as client:
            tasks = []
            for i, (ctx_text, _) in enumerate(context_to_doc.items()):
                title = ctx_text.strip()[:120].replace("\n", " ").strip()
                if not title:
                    title = f"doc_{i}"
                tasks.append(
                    _daser_upload_doc(
                        client,
                        sm.daser_url,
                        title,
                        ctx_text,
                        sem,
                    )
                )
            t0 = time.perf_counter()
            print(
                f"  [upload] starting {len(tasks)} concurrent uploads "
                f"(max_inflight={args.max_inflight})"
            )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            upload_ms = (time.perf_counter() - t0) * 1000
            ok = 0
            for idx, (ctx_text, _) in enumerate(context_to_doc.items()):
                r = results[idx]
                if isinstance(r, Exception):
                    print(f"  [upload error] {type(r).__name__}: {r}")
                    continue
                context_to_doc[ctx_text] = r
                ok += 1
            print(f"  Uploaded {ok}/{len(context_to_doc)} docs in {upload_ms:.0f}ms")
    else:
        print("\n--- Skipping document upload (--no-prefill) ---")

    # ---- Phase 2: Inference (chunk mode) ----
    gen_params = {
        "max_tokens": args.gen_max_tokens,
        "temperature": args.gen_temperature,
    }
    print(
        f"\n--- Running inference ({len(all_samples)} requests, "
        f"max_inflight={args.max_inflight}) ---"
    )
    sem = asyncio.Semaphore(args.max_inflight)
    all_results: list[RequestResult] = []
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        batch_size = min(args.max_inflight * 4, max(100, len(all_samples)))
        for batch_start in range(0, len(all_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(all_samples))
            batch = all_samples[batch_start:batch_end]
            tasks: list[Any] = []
            task_indices: list[int] = []
            for i, s in enumerate(batch):
                doc_info = context_to_doc.get(s.context)
                if not doc_info or "doc_id" not in doc_info:
                    all_results.append(
                        RequestResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            generated_text="",
                            ttft_ms=0,
                            latency_ms=0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            error="document not uploaded",
                        )
                    )
                    continue
                tasks.append(
                    _daser_infer(
                        client,
                        sm.daser_url,
                        [doc_info["doc_id"]],
                        s.question,
                        gen_params,
                        sem,
                        args.timeout,
                    )
                )
                task_indices.append(i)
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for ti, r in zip(task_indices, batch_results, strict=False):
                s = batch[ti]
                if isinstance(r, Exception):
                    all_results.append(
                        RequestResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            generated_text="",
                            ttft_ms=0,
                            latency_ms=0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            error=str(r),
                        )
                    )
                else:
                    body, wall_ms = r
                    text = body.get("text", "")
                    cache_hits = body.get("cache_hits", [])
                    cache_hits_count = sum(1 for c in cache_hits if c.get("chunk_key"))
                    all_results.append(
                        RequestResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            generated_text=text,
                            ttft_ms=body.get("ttft_ms", 0.0),
                            latency_ms=body.get("latency_ms", wall_ms),
                            prompt_tokens=body.get("prompt_tokens", 0),
                            completion_tokens=body.get("completion_tokens", 0),
                            server_latency_ms=body.get("latency_ms", 0.0),
                            cache_hits=cache_hits_count,
                            cache_chunks_total=len(cache_hits),
                        )
                    )
            elapsed = time.perf_counter() - t0
            print(f"  [{batch_end}/{len(all_samples)}] {elapsed:.0f}s")
    _print_error_summary(all_results, "daser")
    _print_daser_profile(all_results)
    return _build_metrics(samples_by_ds, all_results)


async def _run_daser_prefix_shared(
    args: argparse.Namespace,
    sm: ServerManager,
    all_samples: list[Sample],
    samples_by_ds: dict[str, list[Sample]],
    gen_params: dict[str, Any],
) -> list[DatasetMetrics]:
    """Prefix mode: each sample uses its own context, sent directly to vLLM.

    Builds prompts with ``build_full_prompt`` (system + context + question)
    and sends to vLLM ``/v1/completions``. The DaseR connector caches prefix
    KV on the first prefill per unique context; subsequent samples sharing
    the same context hit the prefix cache and skip the shared prefill.

    Context-length filtering is done in ``load_dataset``, so all samples
    here are guaranteed to fit within the model's context window.
    """
    tokenizer = _lazy_tokenizer(args.model)
    prompts: list[str] = []
    for s in all_samples:
        try:
            prompt = build_full_prompt(tokenizer, s.context, s.question)
        except Exception as exc:
            prompts.append(f"Error building prompt: {exc}")
            continue
        prompts.append(prompt)

    # Report context reuse stats
    ctx_counts: dict[str, int] = {}
    for s in all_samples:
        ctx_counts[s.context] = ctx_counts.get(s.context, 0) + 1
    unique_ctx = len(ctx_counts)
    dup_samples = sum(c - 1 for c in ctx_counts.values() if c > 1)
    max_per_ctx = max(ctx_counts.values()) if ctx_counts else 0
    print(
        f"\n  Prefix mode: {unique_ctx} unique contexts, "
        f"{len(all_samples)} samples, "
        f"{dup_samples} prefix-hittable duplicates, "
        f"max {max_per_ctx} questions/context"
    )

    n = len(all_samples)
    print(
        f"\n--- Running prefix inference ({n} requests, "
        f"max_inflight={args.max_inflight}) ---"
    )
    sem = asyncio.Semaphore(args.max_inflight)
    all_results: list[RequestResult] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        t0 = time.perf_counter()
        batch_size = min(args.max_inflight * 4, max(100, n))
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = all_samples[batch_start:batch_end]
            p_batch = prompts[batch_start:batch_end]
            tasks = [
                _vllm_completion_stream(
                    client,
                    sm.vllm_url,
                    prompt,
                    gen_params,
                    sem,
                    args.timeout,
                )
                for prompt in p_batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for s, r in zip(batch, batch_results, strict=False):
                if isinstance(r, Exception):
                    all_results.append(
                        RequestResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            generated_text="",
                            ttft_ms=0,
                            latency_ms=0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            error=str(r),
                        )
                    )
                else:
                    result, ttft_ms, wall_ms = r
                    text = ""
                    if result.get("choices"):
                        text = result["choices"][0].get("text", "")
                    usage = result.get("usage") or {}
                    all_results.append(
                        RequestResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            generated_text=text,
                            ttft_ms=ttft_ms,
                            latency_ms=wall_ms,
                            prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                        )
                    )
            elapsed = time.perf_counter() - t0
            print(f"  [{batch_end}/{n}] {elapsed:.0f}s")

    _print_error_summary(all_results, "daser-prefix")
    return _build_metrics(samples_by_ds, all_results)


async def _run_vllm(
    args: argparse.Namespace,
    sm: ServerManager,
    samples_by_ds: dict[str, list[Sample]],
    label: str,
    skip_sample_ids: set[int] | None = None,
) -> tuple[list[DatasetMetrics], list[RequestResult]]:
    """Run vLLM-only or vLLM+LMCache inference.

    For the LMCache case, the caller should start vLLM with LMCacheConnectorV1
    and run this twice (cold + warm), returning the warm metrics.

    Args:
        skip_sample_ids: if set, skip samples with these IDs (used for LMCache
            warm pass to avoid re-running samples that failed in cold pass).
    """
    gen_params = {
        "max_tokens": args.gen_max_tokens,
        "temperature": args.gen_temperature,
    }

    # Interleave samples to eliminate ordering bias
    all_samples: list[Sample] = _interleave_samples(samples_by_ds)

    # Filter out samples that failed in a prior pass (e.g. LMCache cold)
    if skip_sample_ids:
        all_samples = [s for s in all_samples if s.sample_id not in skip_sample_ids]
        if not all_samples:
            raise RuntimeError("All samples filtered out by skip_sample_ids")

    # Build prompts (context-length filtering is done in load_dataset)
    print(f"  Building prompts for {len(all_samples)} samples...")
    tokenizer = _lazy_tokenizer(args.model)
    prompts: list[str] = []
    for s in all_samples:
        try:
            prompt = build_full_prompt(tokenizer, s.context, s.question)
        except Exception as exc:
            prompts.append(f"Error building prompt: {exc}")
            continue
        prompts.append(prompt)

    print(
        f"\n--- Running {label} inference ({len(all_samples)} requests, "
        f"max_inflight={args.max_inflight}) ---"
    )
    sem = asyncio.Semaphore(args.max_inflight)
    all_results: list[RequestResult] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        t0 = time.perf_counter()
        batch_size = min(args.max_inflight * 4, max(100, len(all_samples)))
        for batch_start in range(0, len(all_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(all_samples))
            batch = all_samples[batch_start:batch_end]
            p_batch = prompts[batch_start:batch_end]
            tasks = [
                _vllm_completion_stream(
                    client,
                    sm.vllm_url,
                    prompt,
                    gen_params,
                    sem,
                    args.timeout,
                )
                for prompt in p_batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for s, r in zip(batch, batch_results, strict=False):
                if isinstance(r, Exception):
                    all_results.append(
                        RequestResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            generated_text="",
                            ttft_ms=0,
                            latency_ms=0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            error=str(r),
                        )
                    )
                else:
                    result, ttft_ms, wall_ms = r
                    text = ""
                    if result.get("choices"):
                        text = result["choices"][0].get("text", "")
                    usage = result.get("usage") or {}
                    all_results.append(
                        RequestResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            generated_text=text,
                            ttft_ms=ttft_ms,
                            latency_ms=wall_ms,
                            prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                        )
                    )
            elapsed = time.perf_counter() - t0
            print(f"  [{batch_end}/{len(all_samples)}] {elapsed:.0f}s")
    _print_error_summary(all_results, label)
    return _build_metrics(samples_by_ds, all_results), all_results


async def _run_lmcache(
    args: argparse.Namespace,
    sm: ServerManager,
    samples_by_ds: dict[str, list[Sample]],
) -> list[DatasetMetrics]:
    """Run LMCache mode (MP): start server, cold pass, then warm pass."""
    # Start LMCache MP server first, then vLLM
    await sm.start_lmcache_mp_server()
    await sm.start_vllm_lmcache()

    # Cold pass
    print("\n--- LMCache cold pass (filling cache) ---")
    cold_metrics, cold_results = await _run_vllm(
        args, sm, samples_by_ds, "lmcache-cold"
    )
    _print_summary(cold_metrics, "lmcache-cold")

    # Collect sample IDs that failed in cold pass — they have no cached KV
    cold_failed_ids: set[int] = {
        r.sample_id for r in cold_results if r.error is not None
    }
    if cold_failed_ids:
        print(
            f"  [lmcache] {len(cold_failed_ids)} cold-pass failures "
            f"will be excluded from warm pass (no cached KV)"
        )

    # Wait for LMCache disk writes to settle (file-count stabilization)
    print("\n  Waiting for LMCache disk writes to settle...")
    await _wait_disk_quiesce(Path(args.store_dir) / "lmcache_mp_disk", timeout_s=120.0)

    # Warm pass — same server session, cache should be populated
    print("\n--- LMCache warm pass ---")
    warm_metrics, _ = await _run_vllm(
        args,
        sm,
        samples_by_ds,
        "lmcache-warm",
        skip_sample_ids=cold_failed_ids,
    )
    return warm_metrics


async def _run_vllm_mode(
    args: argparse.Namespace,
    sm: ServerManager,
    samples_by_ds: dict[str, list[Sample]],
) -> list[DatasetMetrics]:
    """Run vanilla vLLM mode."""
    await sm.start_vllm_only()
    metrics, _ = await _run_vllm(args, sm, samples_by_ds, "vllm")
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_metrics(
    samples_by_ds: dict[str, list[Sample]],
    results: list[RequestResult],
) -> list[DatasetMetrics]:
    """Aggregate per-dataset metrics from raw results."""
    # Build answer lookup
    answers_by_id: dict[int, list[str]] = {}
    for ds_samples in samples_by_ds.values():
        for s in ds_samples:
            answers_by_id[s.sample_id] = s.answers

    # Group results by dataset
    by_ds: dict[str, list[RequestResult]] = {}
    for r in results:
        by_ds.setdefault(r.dataset, []).append(r)

    metrics_list: list[DatasetMetrics] = []
    for ds_name in sorted(by_ds.keys()):
        metrics_list.append(_compute_metrics(ds_name, by_ds[ds_name], answers_by_id))
    return metrics_list


def _save_results(
    args: argparse.Namespace,
    mode: str,
    all_metrics: list[DatasetMetrics],
    wall_s: float,
) -> None:
    """Write JSON results file."""
    output: dict[str, Any] = {
        "config": {
            "mode": mode,
            "model": args.model,
            "l2_size": args.l2_size,
            "l1_size": args.l1_size,
            "gpu_util": args.gpu_util,
            "max_num_seqs": args.max_num_seqs,
            "max_inflight": args.max_inflight,
            "gen_max_tokens": args.gen_max_tokens,
            "gen_temperature": args.gen_temperature,
        },
        "wall_seconds": wall_s,
        "per_dataset": {},
        "aggregate": {},
    }
    total_contains = 0.0
    total_samples = 0
    for m in all_metrics:
        n = m.num_samples - m.num_errors
        output["per_dataset"][m.dataset] = {
            "num_samples": m.num_samples,
            "num_errors": m.num_errors,
            "accuracy_contains": m.accuracy_contains,
            "ttft_ms_mean": m.ttft_ms_mean,
            "ttft_ms_p50": m.ttft_ms_p50,
            "ttft_ms_p99": m.ttft_ms_p99,
            "latency_ms_mean": m.latency_ms_mean,
            "prompt_tokens_total": m.prompt_tokens_total,
            "completion_tokens_total": m.completion_tokens_total,
        }
        total_contains += m.accuracy_contains * n
        total_samples += n

    if total_samples > 0:
        output["aggregate"] = {
            "num_samples": total_samples,
            "accuracy_contains": total_contains / total_samples,
        }
    # Write using a temp file + rename to be atomic (don't lose data on crash)
    out_path = Path(args.output)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    tmp_path.rename(out_path)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main_async(args: argparse.Namespace) -> None:
    """Top-level async entry point."""
    # Load datasets (with optional context-length filtering)
    print(f"\nLoading datasets from {args.data_dir}...")
    samples_by_ds = load_dataset(
        args.data_dir,
        args.datasets,
        args.max_samples,
        max_context_tokens=args.max_context_tokens,
        model_path=args.model if args.max_context_tokens > 0 else None,
    )
    total = sum(len(v) for v in samples_by_ds.values())
    if total == 0:
        print("No samples loaded. Check --data-dir and --datasets.")
        return
    print(f"Total samples: {total} across {len(samples_by_ds)} datasets")

    # Deduplicate by context for fair comparison.
    # Prefix mode skips dedup so duplicate contexts trigger cache hits.
    skip_dedup = args.no_dedup_context or (
        args.mode == "daser" and args.cache_reuse_mode == "prefix"
    )
    if not skip_dedup:
        samples_by_ds = _dedup_by_context(samples_by_ds)
        total = sum(len(v) for v in samples_by_ds.values())
        if total == 0:
            print("No samples remaining after dedup.")
            return
        print(f"After dedup: {total} samples across {len(samples_by_ds)} datasets")
    elif args.mode == "daser" and args.cache_reuse_mode == "prefix":
        print("Prefix mode: dedup disabled to allow prefix cache hits")

    # Warn if max_inflight significantly exceeds max_num_seqs (queue bias)
    if args.max_inflight > args.max_num_seqs * 2:
        print(
            f"  [warn] --max-inflight ({args.max_inflight}) >> "
            f"--max-num-seqs ({args.max_num_seqs}): "
            f"TTFT includes queue wait time, not pure inference time. "
            f"Consider --max-inflight={args.max_num_seqs} for clean TTFT."
        )

    # Determine which modes to run
    modes: list[str]
    if args.mode == "all":
        modes = ["vllm", "lmcache", "daser"]
    else:
        modes = [args.mode]

    all_outputs: dict[str, list[DatasetMetrics]] = {}

    for mode in modes:
        sm = ServerManager(args)
        t0 = time.perf_counter()
        metrics: list[DatasetMetrics] = []
        ok = False

        # Background GPU monitor
        gpu_task: asyncio.Task[None] | None = None
        if args.gpu_monitor_secs > 0:
            gpu_task = asyncio.create_task(
                _gpu_monitor(args.gpu_id, args.gpu_monitor_secs, mode),
                name=f"gpu-monitor-{mode}",
            )

        try:
            if mode == "vllm":
                metrics = await _run_vllm_mode(args, sm, samples_by_ds)
            elif mode == "lmcache":
                metrics = await _run_lmcache(args, sm, samples_by_ds)
            elif mode == "daser":
                metrics = await _run_daser(args, sm, samples_by_ds)
            else:
                print(f"Unknown mode: {mode}")
                continue
            ok = True
            wall_s = time.perf_counter() - t0
            _print_summary(metrics, mode)

            # Save per-mode results
            out_path = Path(args.output)
            stem = out_path.stem
            mode_output = str(out_path.with_stem(f"{stem}_{mode}"))
            args.output = mode_output
            _save_results(args, mode, metrics, wall_s)
            all_outputs[mode] = metrics
        finally:
            if gpu_task is not None:
                gpu_task.cancel()
                try:
                    await gpu_task
                except asyncio.CancelledError:
                    pass
            await sm.stop_all()
            if ok and not args.keep_alive:
                sm.cleanup_scratch()
            elif not ok:
                print(f"\n[{mode}] FAILED — logs preserved at {sm._log_dir}")  # noqa: SLF001

    # If running all modes, also write a combined summary
    if len(modes) > 1:
        combined: dict[str, Any] = {"config": vars(args), "per_mode": {}}
        for mode, metrics in all_outputs.items():
            combined["per_mode"][mode] = {
                ds.dataset: {
                    "accuracy_contains": ds.accuracy_contains,
                    "ttft_ms_mean": ds.ttft_ms_mean,
                }
                for ds in metrics
            }
        # Write combined to same output stem but with _combined suffix
        ts = time.strftime("%Y%m%d_%H%M%S")
        combined_path = Path(f"longbench_all_combined_{ts}.json")
        combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
        print(f"\nCombined comparison saved to {combined_path}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"L2 size: {args.l2_size} bytes  L1 size: {args.l1_size} bytes")
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
