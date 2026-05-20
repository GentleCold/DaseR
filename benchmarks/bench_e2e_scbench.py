# SPDX-License-Identifier: Apache-2.0
"""End-to-end SCBench benchmark: DaseR vs LMCache LocalDiskBackend.

Runs SCBench prompts through vLLM twice per system (cold + warm),
measuring elapsed time and prompt-token throughput. SCBench is a
long-context multi-turn benchmark; each (example, turn) pair becomes
one prompt. Prefix cache is disabled so the NVMe storage tier is the
only source of cross-run speedup.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \\
    python benchmarks/bench_e2e_scbench.py \\
        --num-prompts 200 [--dataset scbench_choice_eng] \\
        [--dataset-dir /data/ld/SCBench] \\
        [--out results.json]
"""

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Deterministic hashing — re-exec with PYTHONHASHSEED set so both LMCache
# scheduler-side token hashing and vLLM's NONE_HASH seed are stable across
# cold/warm LLM rebuilds. Must happen before *any* import that touches
# Python string hashing or vLLM internals.
# ---------------------------------------------------------------------------
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

# Third Party
import torch

# First Party — add project root for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from daser.logging import init_logger
from daser.perf import collect_histograms, load_histograms_json, print_report_from_dict
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

logger = init_logger(__name__)

# Resolve data root from current user.
_USER: str = os.environ.get("USER", "") or os.environ.get("LOGNAME", "")
if not _USER:
    try:
        _USER = os.getlogin()
    except OSError:
        _USER = "tmp"
_DATA_DIR: str = f"/data/{_USER}"

# ---------------------------------------------------------------------------
# Constants — Qwen3-8B KV geometry (matches tests/integration/conftest.py)
# ---------------------------------------------------------------------------
MODEL_PATH_DEFAULT: str = "/data/zwt/model/models/Qwen/Qwen3-8B"
NUM_KV_HEADS: int = 8
HEAD_DIM: int = 128
NUM_LAYERS: int = 36
BLOCK_TOKENS: int = 16
DTYPE_BYTES: int = 2  # bfloat16
SLOT_SIZE: int = NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS * BLOCK_TOKENS * DTYPE_BYTES
# 8 * 128 * 2 * 36 * 16 * 2 = 2,359,296 bytes

MAX_INPUT_TOKENS_DEFAULT: int = 1792
GPU_MEM_UTIL_DEFAULT: float = 0.4
MAX_NUM_SEQS_DEFAULT: int = 64

DATASET_DIR_DEFAULT: str = "/data/ld/SCBench"
DATASET_DEFAULT: str = "scbench_choice_eng"


# ---------------------------------------------------------------------------
# Workload loader — SCBench Parquet / JSONL
# ---------------------------------------------------------------------------


def _try_import_parquet() -> Any:
    """Import pyarrow.parquet, returning None on failure."""
    try:
        import pyarrow.parquet as pq  # noqa: F811

        return pq
    except ImportError:
        return None


def _load_parquet_prompts(path: str, n: int) -> list[str]:
    """Load prompts from a SCBench Parquet file.

    Each row has ``context`` + ``multi_turns`` (list of
    ``{input, answer, options?}`` dicts). Each turn produces one prompt.

    Args:
        path: Path to the ``.parquet`` file.
        n: Max number of prompts to return.

    Returns:
        List of raw prompt strings built from context + turn input.
    """
    pq = _try_import_parquet()
    if pq is None:
        logger.warning("pyarrow not available — falling back to synthetic prompts")
        return _fallback_prompts(n)

    pf = pq.read_table(path)
    out: list[str] = []
    for row_idx in range(pf.num_rows):
        if len(out) >= n:
            break
        context = str(pf.column("context")[row_idx].as_py())
        multi_turns = pf.column("multi_turns")[row_idx].as_py()
        for turn in multi_turns:
            if len(out) >= n:
                break
            question = turn.get("input", "")
            options = turn.get("options")
            prompt = _build_prompt_text(context, question, options)
            if prompt.strip():
                out.append(prompt)

    return out


def _load_jsonl_prompts(path: str, n: int) -> list[str]:
    """Load prompts from a SCBench JSONL file.

    Same structure as Parquet: each line has ``context`` +
    ``multi_turns``.

    Args:
        path: Path to the ``.jsonl`` file.
        n: Max number of prompts to return.

    Returns:
        List of raw prompt strings.
    """
    out: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(out) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            context = rec.get("context", "")
            multi_turns = rec.get("multi_turns", [])
            for turn in multi_turns:
                if len(out) >= n:
                    break
                question = turn.get("input", "")
                options = turn.get("options")
                prompt = _build_prompt_text(context, question, options)
                if prompt.strip():
                    out.append(prompt)
    return out


def _build_prompt_text(context: str, question: str, options: list[str] | None) -> str:
    """Construct a prompt from SCBench context + question + optional choices.

    Args:
        context: Long reference text.
        question: The question for this turn.
        options: Optional list of choice strings (for multiple-choice tasks).

    Returns:
        Formatted prompt string.
    """
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    if options:
        opts = "\n".join(f"  {chr(ord('A') + i)}. {o}" for i, o in enumerate(options))
        prompt += f"\n\nOptions:\n{opts}"
    return prompt


def load_prompts(dataset_dir: str, dataset: str, n: int) -> list[str]:
    """Load N raw prompt strings from a SCBench dataset.

    Tries Parquet first (``<dataset_dir>/<dataset>/test-*.parquet``),
    then JSONL (``<dataset_dir>/data/<dataset>.jsonl``).

    Args:
        dataset_dir: Root SCBench directory.
        dataset: Task name (e.g. ``"scbench_choice_eng"``).
        n: Number of prompts to return.

    Returns:
        List of raw prompt strings. Falls back to synthetic prompts if
        no data file is found.
    """
    # Try Parquet first
    task_dir = os.path.join(dataset_dir, dataset)
    if os.path.isdir(task_dir):
        for fname in sorted(os.listdir(task_dir)):
            if fname.endswith(".parquet"):
                path = os.path.join(task_dir, fname)
                logger.info("loading SCBench prompts from %s", path)
                out = _load_parquet_prompts(path, n)
                logger.info(
                    "loaded %d prompts from %s (requested %d)", len(out), path, n
                )
                return out

    # Try JSONL
    jsonl_path = os.path.join(dataset_dir, "data", f"{dataset}.jsonl")
    if os.path.exists(jsonl_path):
        logger.info("loading SCBench prompts from %s", jsonl_path)
        out = _load_jsonl_prompts(jsonl_path, n)
        logger.info("loaded %d prompts from %s (requested %d)", len(out), jsonl_path, n)
        return out

    logger.warning("SCBench dataset not found: %s/%s", dataset_dir, dataset)
    return _fallback_prompts(n)


def _fallback_prompts(n: int) -> list[str]:
    """Synthetic prompts when SCBench data is missing."""
    logger.warning("SCBench data not found — falling back to synthetic prompts")
    base = [
        (
            "Artificial intelligence is transforming the way we work and live. "
            "From natural language processing to computer vision, machine learning "
            "models are being deployed in healthcare, finance, transportation, and "
            "education. As these systems become more capable, questions about "
            "safety, alignment, and interpretability grow more urgent. Researchers "
            "at universities and companies around the world are working to ensure "
            "that AI systems remain beneficial and controllable as they scale. "
            "Describe the key technical challenges in AI alignment:"
        ),
        (
            "The history of computing spans eight decades, from vacuum tube "
            "machines weighing several tons to pocket-sized devices more powerful "
            "than the supercomputers of the 1990s. The invention of the transistor,"
            " the integrated circuit, and the microprocessor each triggered an "
            "order-of-magnitude leap in capability. Today, GPU clusters connected "
            "by high-bandwidth interconnects power large language models trained on"
            " trillions of tokens. Summarize the most important inflection points "
            "in computer hardware history:"
        ),
    ]
    return [base[i % len(base)] for i in range(n)]


def tokenise_and_truncate(
    prompts: list[str], tokenizer: Any, max_tokens: int, block_tokens: int
) -> list[list[int]]:
    """Tokenise and truncate prompts to max_tokens.

    Args:
        prompts: Raw prompt strings.
        tokenizer: HF tokenizer.
        max_tokens: Per-prompt token ceiling.
        block_tokens: KV block size (tokens per slot).

    Returns:
        List of token-ID lists suitable for vLLM ``TokensPrompt``.
    """
    out: list[list[int]] = []
    for p in prompts:
        encoded = tokenizer(
            p, truncation=True, max_length=max_tokens, add_special_tokens=False
        )
        ids = encoded["input_ids"]
        if len(ids) < block_tokens + 1:
            pad = tokenizer.encode(" ", add_special_tokens=False)
            if pad:
                while len(ids) < block_tokens + 1:
                    ids = ids + pad
                ids = ids[: block_tokens + 1]
        out.append(ids)
    return out


# ---------------------------------------------------------------------------
# LLM build/destroy helpers
# ---------------------------------------------------------------------------


def _destroy_llm(llm: Any) -> None:
    """Shut down a vLLM LLM and free GPU memory."""
    try:
        try:
            llm.llm_engine.engine_core.shutdown(timeout=30.0)
        except TypeError:
            llm.llm_engine.engine_core.shutdown()
    except Exception as exc:
        logger.warning("engine_core.shutdown raised: %s", exc)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# DaseR harness
# ---------------------------------------------------------------------------


class DaserHarness:
    """Owns a DaseR IPCServer + store file for one benchmark run."""

    def __init__(
        self,
        tmpdir: str,
        total_slots: int,
        model_path: str,
        gpu_util: float,
        max_num_seqs: int,
        max_model_len: int,
    ) -> None:
        self.tmpdir = tmpdir
        self.socket_path = os.path.join(tmpdir, "daser.sock")
        self.store_path = os.path.join(tmpdir, "daser.store")
        self.model_path = model_path
        self.total_slots = total_slots
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: IPCServer | None = None

    def start(self) -> None:
        """Pre-allocate store + start IPCServer in a daemon thread."""
        size = self.total_slots * SLOT_SIZE
        with open(self.store_path, "wb") as f:
            f.truncate(size)

        metadata = MetadataStore(total_slots=self.total_slots)
        registry = DocRegistry()
        cm = ChunkManager(
            total_slots=self.total_slots,
            metadata_store=metadata,
            doc_registry=registry,
        )
        core = ServerCore(
            chunk_manager=cm,
            retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
            position_encoder=FixedOffsetEncoder(fixed_offset=0),
            slot_size=SLOT_SIZE,
            block_tokens=BLOCK_TOKENS,
        )
        server = IPCServer(
            socket_path=self.socket_path,
            core=core,
            runtime_config={
                "socket_path": self.socket_path,
                "store_path": self.store_path,
                "slot_size": SLOT_SIZE,
                "block_tokens": BLOCK_TOKENS,
                "model_id": "qwen3-8b",
            },
        )

        loop = asyncio.new_event_loop()
        started = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.start())
            started.set()
            loop.run_forever()

        thread = threading.Thread(target=_run, daemon=True, name="daser-bench-server")
        thread.start()
        assert started.wait(timeout=10.0), "DaseR IPCServer failed to start in 10s"
        self._loop = loop
        self._thread = thread
        self._server = server
        logger.info(
            "[DaseR] server up — store=%s (%.1f GB, %d slots)",
            self.store_path,
            size / 1e9,
            self.total_slots,
        )

    def build_llm(self) -> Any:
        """Construct a vLLM LLM wired to DaserConnector."""
        from vllm import LLM  # Third Party

        kv_transfer_config = {
            "kv_connector": "DaserConnector",
            "kv_connector_module_path": "daser.connector.daser_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "socket_path": self.socket_path,
                "store_path": self.store_path,
                "slot_size": SLOT_SIZE,
                "block_tokens": BLOCK_TOKENS,
                "model_id": "qwen3-8b",
            },
        }
        return LLM(
            model=self.model_path,
            kv_transfer_config=kv_transfer_config,
            gpu_memory_utilization=self.gpu_util,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enable_prefix_caching=False,
            disable_hybrid_kv_cache_manager=True,
        )

    def stop(self) -> None:
        """Stop the IPCServer cleanly."""
        if self._server is not None and self._loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._server.stop(), self._loop)
                fut.result(timeout=10.0)
            except Exception as exc:
                logger.warning("[DaseR] server stop raised: %s", exc)
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=10.0)
        logger.info("[DaseR] server stopped")


# ---------------------------------------------------------------------------
# LMCache harness
# ---------------------------------------------------------------------------


class LMCacheHarness:
    """Configures LMCache via env vars and builds an LMCacheConnectorV1 LLM."""

    def __init__(
        self,
        tmpdir: str,
        total_bytes: int,
        model_path: str,
        gpu_util: float,
        max_num_seqs: int,
        max_model_len: int,
    ) -> None:
        self.tmpdir = tmpdir
        self.model_path = model_path
        self.total_bytes = total_bytes
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self._saved_env: dict[str, str | None] = {}

    def start(self) -> None:
        """Apply LMCache env configuration before LLM init."""
        cpu_gb = max(4.0, self.total_bytes * 0.15 / 1e9)
        env = {
            "LMCACHE_CHUNK_SIZE": str(BLOCK_TOKENS),
            "LMCACHE_LOCAL_CPU": "False",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": f"{cpu_gb:.1f}",
            "LMCACHE_LOCAL_DISK": f"file://{self.tmpdir}/",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": (
                f"{max(5.0, self.total_bytes * 3 / 1e9):.1f}"
            ),
            "LMCACHE_USE_LAYERWISE": "False",
            "LMCACHE_USE_GDS": "False",
            "LMCACHE_LMCACHE_INSTANCE_ID": "daser_vs_lmcache_scbench",
            "PYTHONHASHSEED": "0",
        }
        for k, v in env.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v
        logger.info(
            "[LMCache] env configured — local_disk=%s (%s GB ceiling), cpu=%s GB",
            self.tmpdir,
            env["LMCACHE_MAX_LOCAL_DISK_SIZE"],
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"],
        )

    def build_llm(self) -> Any:
        """Construct a vLLM LLM wired to LMCacheConnectorV1."""
        from vllm import LLM  # Third Party

        kv_transfer_config = {
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_both",
        }
        return LLM(
            model=self.model_path,
            kv_transfer_config=kv_transfer_config,
            gpu_memory_utilization=self.gpu_util,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enable_prefix_caching=False,
        )

    def stop(self) -> None:
        """Restore previous env values."""
        for k, saved in self._saved_env.items():
            if saved is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved
        self._saved_env.clear()
        logger.info("[LMCache] env restored")


# ---------------------------------------------------------------------------
# Timed runner
# ---------------------------------------------------------------------------


def run_system(
    name: str,
    build_llm_fn: Any,
    prompts: list[list[int]],
    warmup_prompt: list[int],
) -> dict[str, Any]:
    """Run cold + warm timed passes for one system.

    We intentionally do NOT rebuild the LLM between cold and warm passes:
    LMCache's LocalDiskBackend keeps its chunk index in-memory and would
    orphan every chunk on rebuild. vLLM's in-GPU KV is recycled between
    generate() calls with enable_prefix_caching=False, so the warm pass
    must fetch from the external storage tier — the signal we measure.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

    params = SamplingParams(temperature=0.0, max_tokens=1)
    warmup_params = SamplingParams(temperature=0.0, max_tokens=1)

    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
    tp_warmup = TokensPrompt(prompt_token_ids=warmup_prompt)

    logger.info("[%s] building LLM", name)
    llm = build_llm_fn()

    logger.info("[%s] cold: warmup", name)
    llm.generate([tp_warmup], warmup_params)

    logger.info("[%s] cold: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    cold_outputs = llm.generate(tp_prompts, params)
    cold_elapsed = time.perf_counter() - t0
    logger.info("[%s] cold elapsed: %.2fs", name, cold_elapsed)

    logger.info("[%s] warm: warmup", name)
    llm.generate([tp_warmup], warmup_params)

    logger.info("[%s] warm: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    warm_outputs = llm.generate(tp_prompts, params)
    warm_elapsed = time.perf_counter() - t0
    logger.info("[%s] warm elapsed: %.2fs", name, warm_elapsed)

    logger.info("[%s] destroying LLM", name)
    _destroy_llm(llm)

    return {
        "cold_elapsed_s": cold_elapsed,
        "warm_elapsed_s": warm_elapsed,
        "cold_outputs": cold_outputs,
        "warm_outputs": warm_outputs,
    }


def correctness_check(
    name: str, cold_outputs: list, warm_outputs: list
) -> dict[str, int]:
    """Compare cold vs warm token IDs per prompt."""
    mismatches = 0
    total = len(cold_outputs)
    for i, (c, w) in enumerate(zip(cold_outputs, warm_outputs, strict=False)):
        if list(c.outputs[0].token_ids) != list(w.outputs[0].token_ids):
            mismatches += 1
            if mismatches <= 3:
                logger.warning(
                    "[%s] prompt %d: cold/warm token_ids differ: %r vs %r",
                    name,
                    i,
                    c.outputs[0].token_ids,
                    w.outputs[0].token_ids,
                )
    if mismatches:
        logger.warning("[%s] %d/%d prompts mismatched", name, mismatches, total)
    else:
        logger.info("[%s] correctness OK (%d/%d match)", name, total, total)
    return {"mismatches": mismatches, "total": total}


# ---------------------------------------------------------------------------
# Profiled DaseR runner (rebuilds LLM between cold/warm)
# ---------------------------------------------------------------------------


def run_daser_phased(
    harness: Any,
    prompts: list[list[int]],
    warmup_prompt: list[int],
    perf_enabled: bool,
    profiler_ctx: Any | None = None,
) -> dict[str, Any]:
    """Run DaseR cold + warm with separate LLM builds for per-phase profiling.

    Rebuilding the LLM between passes triggers ``shutdown()`` on the
    connector, which exports histograms to the path specified by
    ``DASER_PERF_HISTOGRAM_PATH``. The ``IPCServer`` persists
    chunk state across rebuilds, so the warm pass sees all chunks
    written during the cold pass.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

    params = SamplingParams(temperature=0.0, max_tokens=1)
    warmup_params = SamplingParams(temperature=0.0, max_tokens=1)

    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
    tp_warmup = TokensPrompt(prompt_token_ids=warmup_prompt)

    result: dict[str, Any] = {}

    # --- Cold pass ---
    if perf_enabled:
        os.environ["DASER_PERF_HISTOGRAM_PATH"] = os.path.join(
            harness.tmpdir, "histograms_cold.json"
        )
        os.environ["DASER_PERF_CACHE_PATH"] = os.path.join(
            harness.tmpdir, "cache_cold.json"
        )

    logger.info("[DaseR] building LLM for cold pass")
    llm = harness.build_llm()

    logger.info("[DaseR] cold: warmup")
    if profiler_ctx is not None:
        profiler_ctx.measure_generate("cold_warmup", llm, [tp_warmup], warmup_params)
    else:
        llm.generate([tp_warmup], warmup_params)

    logger.info("[DaseR] cold: generate(N=%d)", len(tp_prompts))
    if profiler_ctx is not None:
        cold_outputs = profiler_ctx.measure_generate(
            "cold_generate", llm, tp_prompts, params
        )
        cold_elapsed = profiler_ctx.phases[-1].wall_elapsed_s
    else:
        t0 = time.perf_counter()
        cold_outputs = llm.generate(tp_prompts, params)
        cold_elapsed = time.perf_counter() - t0
    logger.info("[DaseR] cold elapsed: %.2fs", cold_elapsed)

    logger.info("[DaseR] destroying LLM (cold)")
    _destroy_llm(llm)

    if perf_enabled:
        cold_hist = load_histograms_json(
            os.path.join(harness.tmpdir, "histograms_cold.json")
        )
        cold_cache = _load_json_if_exists(
            os.path.join(harness.tmpdir, "cache_cold.json")
        )
        result["cold_histograms"] = cold_hist
        result["cold_cache"] = cold_cache

    # --- Warm pass ---
    if perf_enabled:
        os.environ["DASER_PERF_HISTOGRAM_PATH"] = os.path.join(
            harness.tmpdir, "histograms_warm.json"
        )
        os.environ["DASER_PERF_CACHE_PATH"] = os.path.join(
            harness.tmpdir, "cache_warm.json"
        )

    logger.info("[DaseR] building LLM for warm pass")
    llm = harness.build_llm()

    logger.info("[DaseR] warm: warmup")
    if profiler_ctx is not None:
        profiler_ctx.measure_generate("warm_warmup", llm, [tp_warmup], warmup_params)
    else:
        llm.generate([tp_warmup], warmup_params)

    logger.info("[DaseR] warm: generate(N=%d)", len(tp_prompts))
    if profiler_ctx is not None:
        warm_outputs = profiler_ctx.measure_generate(
            "warm_generate", llm, tp_prompts, params
        )
        warm_elapsed = profiler_ctx.phases[-1].wall_elapsed_s
    else:
        t0 = time.perf_counter()
        warm_outputs = llm.generate(tp_prompts, params)
        warm_elapsed = time.perf_counter() - t0
    logger.info("[DaseR] warm elapsed: %.2fs", warm_elapsed)

    logger.info("[DaseR] destroying LLM (warm)")
    _destroy_llm(llm)

    if perf_enabled:
        warm_hist = load_histograms_json(
            os.path.join(harness.tmpdir, "histograms_warm.json")
        )
        warm_cache = _load_json_if_exists(
            os.path.join(harness.tmpdir, "cache_warm.json")
        )
        result["warm_histograms"] = warm_hist
        result["warm_cache"] = warm_cache

    # Clean up env
    os.environ.pop("DASER_PERF_HISTOGRAM_PATH", None)
    os.environ.pop("DASER_PERF_CACHE_PATH", None)

    result["cold_elapsed_s"] = cold_elapsed
    result["warm_elapsed_s"] = warm_elapsed
    result["cold_outputs"] = cold_outputs
    result["warm_outputs"] = warm_outputs
    return result


def _load_json_if_exists(path: str) -> dict[str, Any] | None:
    """Load a JSON file, returning None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _fmt_elapsed(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f} s"


def _fmt_tps(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{v:,.0f}"


def build_summary(
    daser: dict[str, Any] | None,
    lmcache: dict[str, Any] | None,
    prompt_tokens: int,
) -> dict[str, Any]:
    """Derive tok/s and speedups for the report."""
    summary: dict[str, Any] = {"prompt_tokens_total": prompt_tokens}
    for key, r in (("daser", daser), ("lmcache", lmcache)):
        if r is None or r.get("skipped"):
            summary[key] = {"skipped": True, "reason": (r or {}).get("reason")}
            continue
        cold = r["cold_elapsed_s"]
        warm = r["warm_elapsed_s"]
        summary[key] = {
            "cold_elapsed_s": cold,
            "warm_elapsed_s": warm,
            "cold_tok_per_s": prompt_tokens / cold if cold > 0 else None,
            "warm_tok_per_s": prompt_tokens / warm if warm > 0 else None,
            "warm_cold_speedup": cold / warm if warm > 0 else None,
            "correctness": r.get("correctness"),
        }
    d = summary.get("daser", {})
    lm = summary.get("lmcache", {})
    if not d.get("skipped") and not lm.get("skipped"):
        dw = d.get("warm_tok_per_s") or 0.0
        lw = lm.get("warm_tok_per_s") or 0.0
        summary["warm_tps_ratio_daser_over_lmcache"] = dw / lw if lw > 0 else None
    return summary


def print_report(config: dict[str, Any], summary: dict[str, Any]) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 72)
    print("E2E vLLM Benchmark — DaseR vs LMCache (SCBench)")
    print("=" * 72)
    print(f"Model            : {config['model']}")
    print(f"Dataset          : {config['dataset']}")
    print(f"Prompts          : {config['num_prompts']}")
    print(f"Prompt tokens    : {summary['prompt_tokens_total']:,}")
    print("Sampling         : temperature=0, max_tokens=1")
    print("Prefix cache     : disabled")
    print("-" * 72)
    print(f"{'Metric':<28}{'DaseR':>20}{'LMCache':>20}")
    print("-" * 72)

    d = summary.get("daser", {}) or {}
    lm = summary.get("lmcache", {}) or {}

    def _show(label: str, k: str, fmt: Any) -> None:
        dv = None if d.get("skipped") else d.get(k)
        lv = None if lm.get("skipped") else lm.get(k)
        print(f"{label:<28}{fmt(dv):>20}{fmt(lv):>20}")

    _show("cold elapsed", "cold_elapsed_s", _fmt_elapsed)
    _show("warm elapsed", "warm_elapsed_s", _fmt_elapsed)
    _show("cold tok/s (prompt)", "cold_tok_per_s", _fmt_tps)
    _show("warm tok/s (prompt)", "warm_tok_per_s", _fmt_tps)

    def _speedup(v: Any) -> str:
        return f"{v:.2f}x" if v is not None else "N/A"

    dv_su = None if d.get("skipped") else d.get("warm_cold_speedup")
    lv_su = None if lm.get("skipped") else lm.get("warm_cold_speedup")
    print(f"{'warm/cold speedup':<28}{_speedup(dv_su):>20}{_speedup(lv_su):>20}")

    ratio = summary.get("warm_tps_ratio_daser_over_lmcache")
    print("-" * 72)
    if ratio is not None:
        print(f"DaseR warm tok/s / LMCache warm tok/s = {ratio:.2f}x")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — argparse + orchestration
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--model", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--store-dir", default=f"{_DATA_DIR}/daser_test")
    parser.add_argument(
        "--dataset",
        default=DATASET_DEFAULT,
        help=(
            "SCBench task name (e.g. scbench_choice_eng, scbench_qa_eng, "
            "scbench_repoqa)"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR_DEFAULT,
        help="SCBench root directory",
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=MAX_INPUT_TOKENS_DEFAULT
    )
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=GPU_MEM_UTIL_DEFAULT,
        help="vLLM gpu_memory_utilization (default: 0.4)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=MAX_NUM_SEQS_DEFAULT,
        help="vLLM max_num_seqs (default: 64; lower reduces sampler-warmup memory)",
    )
    parser.add_argument("--skip-daser", action="store_true")
    parser.add_argument("--skip-lmcache", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable GPU-level profiling (torch.cuda.Event + NVTX + histograms).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Export Chrome traces via torch.profiler (implies --profile).",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    profiling = args.profile or args.trace
    trace_dir: str | None = None
    if args.trace:
        trace_dir = os.path.join(args.store_dir, "traces")
    if profiling and os.environ.get("DASER_PERF_LOG", "0") != "1":
        os.environ["DASER_PERF_LOG"] = "1"

    os.makedirs(args.store_dir, exist_ok=True)

    # ---- tokenise prompts ----
    logger.info(
        "loading SCBench prompts from %s (dataset=%s)",
        args.dataset_dir,
        args.dataset,
    )
    raw_prompts = load_prompts(args.dataset_dir, args.dataset, args.num_prompts)
    if len(raw_prompts) < args.num_prompts:
        logger.warning(
            "got %d prompts, requested %d — continuing with what we have",
            len(raw_prompts),
            args.num_prompts,
        )

    from transformers import AutoTokenizer  # Third Party

    logger.info("loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = tokenise_and_truncate(
        raw_prompts, tokenizer, args.max_input_tokens, BLOCK_TOKENS
    )
    token_counts = [len(ids) for ids in prompts]
    prompt_tokens_total = sum(token_counts)
    total_blocks = sum(math.ceil(c / BLOCK_TOKENS) for c in token_counts)
    logger.info(
        "tokenised %d prompts, %d tokens, %d blocks (avg %.1f blocks/prompt)",
        len(prompts),
        prompt_tokens_total,
        total_blocks,
        total_blocks / max(1, len(prompts)),
    )

    # Short warmup prompt — enough tokens for one block + 1 remainder.
    warmup_prompt_ids = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. " * 4,
        add_special_tokens=False,
    )

    # ---- sizes ----
    total_bytes = total_blocks * SLOT_SIZE
    slots_needed = int(math.ceil(1.5 * total_blocks))
    logger.info(
        "store sizing: total_bytes=%.2fGB, slots_needed=%d (1.5x headroom)",
        total_bytes / 1e9,
        slots_needed,
    )

    max_model_len = args.max_input_tokens + 256

    config = {
        "num_prompts": len(prompts),
        "model": args.model,
        "dataset": args.dataset,
        "block_tokens": BLOCK_TOKENS,
        "slot_bytes": SLOT_SIZE,
        "max_input_tokens": args.max_input_tokens,
        "max_model_len": max_model_len,
        "total_blocks": total_blocks,
        "total_bytes": total_bytes,
    }

    # ---- Profiler contexts (created once, used per-system) ----
    daser_ctx: Any = None
    lmcache_ctx: Any = None
    if profiling:
        from benchmarks.profiler import ProfilerContext  # First Party

        daser_ctx = ProfilerContext("DaseR", trace_dir=trace_dir)
        lmcache_ctx = ProfilerContext("LMCache", trace_dir=trace_dir)

    # ---- DaseR run ----
    daser_result: dict[str, Any] | None = None
    if args.skip_daser:
        daser_result = {"skipped": True, "reason": "--skip-daser"}
    else:
        daser_dir = tempfile.mkdtemp(prefix="daser_bench_", dir=args.store_dir)
        h = DaserHarness(
            daser_dir,
            slots_needed,
            args.model,
            args.gpu_util,
            args.max_num_seqs,
            max_model_len,
        )
        try:
            h.start()
            perf_enabled = os.environ.get("DASER_PERF_LOG", "0") == "1"

            if profiling:
                r = run_daser_phased(
                    h, prompts, warmup_prompt_ids, perf_enabled, daser_ctx
                )
            else:
                histogram_path = os.path.join(daser_dir, "histograms.json")
                if perf_enabled:
                    os.environ["DASER_PERF_HISTOGRAM_PATH"] = histogram_path

                r = run_system("DaseR", h.build_llm, prompts, warmup_prompt_ids)

                if os.environ.pop("DASER_PERF_HISTOGRAM_PATH", None):
                    histograms = load_histograms_json(histogram_path)
                    inproc = collect_histograms()
                    for name, h_item in inproc.items():
                        if h_item.count > 0 and name not in histograms:
                            histograms[name] = h_item.to_dict()
                    if histograms:
                        r["histograms"] = histograms
                        print_report_from_dict(histograms)

            r["correctness"] = correctness_check(
                "DaseR", r["cold_outputs"], r["warm_outputs"]
            )
            r.pop("cold_outputs", None)
            r.pop("warm_outputs", None)
            daser_result = r
        finally:
            h.stop()

    # ---- LMCache run ----
    lmcache_result: dict[str, Any] | None = None
    if args.skip_lmcache:
        lmcache_result = {"skipped": True, "reason": "--skip-lmcache"}
    else:
        try:
            import lmcache  # noqa: F401 — import probe
        except ImportError as exc:
            lmcache_result = {"skipped": True, "reason": f"import failed: {exc}"}
        if lmcache_result is None:
            lmcache_dir = tempfile.mkdtemp(prefix="lmcache_bench_", dir=args.store_dir)
            h_lm = LMCacheHarness(
                lmcache_dir,
                total_bytes,
                args.model,
                args.gpu_util,
                args.max_num_seqs,
                max_model_len,
            )
            try:
                h_lm.start()

                if profiling and lmcache_ctx is not None:
                    from vllm import SamplingParams
                    from vllm.inputs import TokensPrompt

                    params = SamplingParams(temperature=0.0, max_tokens=1)
                    warmup_params = SamplingParams(temperature=0.0, max_tokens=1)
                    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
                    tp_warmup = TokensPrompt(prompt_token_ids=warmup_prompt_ids)

                    llm = h_lm.build_llm()

                    logger.info("[LMCache] cold: warmup")
                    lmcache_ctx.measure_generate(
                        "cold_warmup", llm, [tp_warmup], warmup_params
                    )

                    logger.info("[LMCache] cold: generate(N=%d)", len(tp_prompts))
                    cold_outputs = lmcache_ctx.measure_generate(
                        "cold_generate", llm, tp_prompts, params
                    )
                    cold_elapsed = lmcache_ctx.phases[-1].wall_elapsed_s
                    logger.info("[LMCache] cold elapsed: %.2fs", cold_elapsed)

                    logger.info("[LMCache] warm: warmup")
                    lmcache_ctx.measure_generate(
                        "warm_warmup", llm, [tp_warmup], warmup_params
                    )

                    logger.info("[LMCache] warm: generate(N=%d)", len(tp_prompts))
                    warm_outputs = lmcache_ctx.measure_generate(
                        "warm_generate", llm, tp_prompts, params
                    )
                    warm_elapsed = lmcache_ctx.phases[-1].wall_elapsed_s
                    logger.info("[LMCache] warm elapsed: %.2fs", warm_elapsed)

                    logger.info("[LMCache] destroying LLM")
                    _destroy_llm(llm)

                    r = {
                        "cold_elapsed_s": cold_elapsed,
                        "warm_elapsed_s": warm_elapsed,
                        "cold_outputs": cold_outputs,
                        "warm_outputs": warm_outputs,
                    }
                else:
                    r = run_system(
                        "LMCache", h_lm.build_llm, prompts, warmup_prompt_ids
                    )

                r["correctness"] = correctness_check(
                    "LMCache", r["cold_outputs"], r["warm_outputs"]
                )
                r.pop("cold_outputs", None)
                r.pop("warm_outputs", None)
                lmcache_result = r
            finally:
                h_lm.stop()

    # ---- report ----
    summary = build_summary(daser_result, lmcache_result, prompt_tokens_total)
    print_report(config, summary)

    if profiling and daser_ctx is not None and lmcache_ctx is not None:
        from benchmarks.profiler import ProfilerReport  # First Party

        cold_hist = daser_result.get("cold_histograms") if daser_result else None
        warm_hist = daser_result.get("warm_histograms") if daser_result else None
        ProfilerReport(
            daser=daser_ctx,
            lmcache=lmcache_ctx,
            prompt_tokens=prompt_tokens_total,
            cold_histograms=cold_hist,
            warm_histograms=warm_hist,
        ).print()

    if args.out:
        out_obj = {
            "config": config,
            "summary": summary,
            "daser": daser_result,
            "lmcache": lmcache_result,
        }
        Path(args.out).write_text(json.dumps(out_obj, indent=2))
        print(f"\nJSON results written to {args.out}")


if __name__ == "__main__":
    main()
