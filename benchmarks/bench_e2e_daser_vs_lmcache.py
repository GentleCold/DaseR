# SPDX-License-Identifier: Apache-2.0
"""End-to-end inference benchmark: DaseR transfer modes vs LMCache.

Runs the same IMDB-review prompt batch through vLLM twice, once with each
KV connector, measuring cold-pass and warm-pass elapsed time and prompt-token
throughput. Prefix cache is disabled so the NVMe storage tier is the only
source of cross-run speedup.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python benchmarks/bench_e2e_daser_vs_lmcache.py \\
        --model /path/to/model \\
        --store-dir /path/to/benchmark-scratch \\
        --imdb /path/to/imdb.csv \\
        [--num-prompts 200] \\
        [--out results.json]
"""

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio
import csv
from dataclasses import dataclass
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

from daser.connector.ipc_client import IPCClientSync
from daser.logging import init_logger
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — Qwen3-8B KV geometry (matches tests/integration/conftest.py)
# ---------------------------------------------------------------------------
NUM_KV_HEADS: int = 8
HEAD_DIM: int = 128
NUM_LAYERS: int = 36
BLOCK_TOKENS: int = 16
DTYPE_BYTES: int = 2  # bfloat16
SLOT_SIZE: int = NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS * BLOCK_TOKENS * DTYPE_BYTES
# 8 * 128 * 2 * 36 * 16 * 2 = 2,359,296 bytes

BYTES_PER_GIB: int = 1024**3
MAX_MODEL_LEN: int = 2048
MAX_INPUT_TOKENS_DEFAULT: int = 1792
GPU_MEM_UTIL_DEFAULT: float = 0.4
MAX_NUM_SEQS_DEFAULT: int = 64
EVICT_L2_FRACTION: float = 0.95
EVICT_L1_FRACTION: float = 0.9
LMCACHE_LOCAL_SSD_STAGING_GB: float = 0.5
CORRECTNESS_LOGPROBS: int = 0
CORRECTNESS_LOGPROB_TOLERANCE: float = 5e-2

COMPARISON_GDS = "gds-vs-lmcache-local-ssd"
COMPARISON_IOURING_MEM = "iouring-mem-vs-lmcache-local-ssd-mem"


def _bytes_to_lmcache_gb(nbytes: int) -> float:
    """Convert byte capacity to LMCache's GB configuration unit.

    Args:
        nbytes: Capacity in bytes.

    Returns:
        Size value for LMCache GB config knobs. LMCache interprets these
        values with a ``1024**3`` multiplier, so this is a GiB conversion.
    """
    return nbytes / BYTES_PER_GIB


# ---------------------------------------------------------------------------
# Workload loader
# ---------------------------------------------------------------------------


def load_prompts(imdb_path: str, n: int) -> list[str]:
    """Load N IMDB reviews as raw prompt strings.

    Args:
        imdb_path: Path to imdb.csv with a 'review' column.
        n: Number of prompts to return.

    Returns:
        List of raw review strings.
    """
    if not os.path.exists(imdb_path):
        raise FileNotFoundError(f"IMDB CSV not found: {imdb_path}")

    out: list[str] = []
    with open(imdb_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(out) >= n:
                break
            review = row.get("review", "").strip()
            if review:
                out.append(review)
    if len(out) < n:
        logger.warning("IMDB yielded only %d prompts (requested %d)", len(out), n)
    return out


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
        ids = tokenizer.encode(p, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        if len(ids) < block_tokens + 1:
            # Extend trivially short prompts so they cross at least one block
            # boundary with a remainder (ensures non-trivial cache hits).
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
        store_dir: str,
        socket_dir: str,
        total_slots: int,
        model_path: str,
        gpu_util: float,
        max_num_seqs: int,
        transfer_mode: str,
        l1_bytes: int,
    ) -> None:
        """Initialise paths and store file.

        Args:
            store_dir: Directory to hold DaseR store files.
            socket_dir: Short directory to hold the IPC socket.
            total_slots: Pre-allocated slot count for the store.
            model_path: HF model path for vLLM.
            gpu_util: vLLM ``gpu_memory_utilization``.
            max_num_seqs: vLLM ``max_num_seqs``.
            transfer_mode: DaseR transfer backend selected for the run.
            l1_bytes: L1 byte capacity for tiered transfer mode.
        """
        self.store_dir = store_dir
        self.socket_dir = socket_dir
        self.socket_path = os.path.join(socket_dir, "d.sock")
        self.store_path = os.path.join(store_dir, "daser.store")
        self.model_path = model_path
        self.total_slots = total_slots
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.transfer_mode = transfer_mode
        self.l1_bytes = l1_bytes
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: IPCServer | None = None

    def start(self) -> None:
        """Pre-allocate store + start IPCServer in a daemon thread."""
        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.socket_dir, exist_ok=True)
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
                "transfer_mode": self.transfer_mode,
                "l1_size_bytes": self.l1_bytes,
                "l2_size_bytes": size,
                "total_slots": self.total_slots,
                "total_store_bytes": size,
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
            "[DaseR] server up — store=%s (%.1f GiB, %d slots)",
            self.store_path,
            size / BYTES_PER_GIB,
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
            },
        }
        return LLM(
            model=self.model_path,
            kv_transfer_config=kv_transfer_config,
            gpu_memory_utilization=self.gpu_util,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=self.max_num_seqs,
            enable_prefix_caching=False,
            disable_hybrid_kv_cache_manager=True,
        )

    def wait_until_committed(
        self,
        prompts: list[list[int]],
        block_tokens: int,
        require_all_commits: bool,
        require_l2_drain: bool,
        timeout_s: float = 120.0,
    ) -> None:
        """Wait until cold-pass transfer writes are visible in DaseR.

        Args:
            prompts: Tokenized benchmark prompts.
            block_tokens: block size used by the index.
            require_all_commits: when True, require every prompt chunk's store
                commit to complete. Full lookup visibility is measured after
                timing by ``visible_prompt_mask``.
            require_l2_drain: when True, wait for async L2 persistence after
                commit visibility so background cold writes do not interfere
                with warm-load timing.
            timeout_s: Maximum wait time.
        """
        client = IPCClientSync(self.socket_path)
        deadline = time.monotonic() + timeout_s
        expected_commits = sum(
            1 for tokens in prompts if (len(tokens) // block_tokens) * block_tokens > 0
        )
        try:
            while True:
                stats = client.commit_stats()
                committed = int(stats.get("commit_requests", 0))
                if not require_all_commits:
                    if committed == 0:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "timed out waiting for any DaseR commit request"
                            )
                        time.sleep(0.05)
                        continue
                    if require_l2_drain:
                        client.transfer_drain()
                    late = int(stats.get("late_evicted_commits", 0))
                    logger.info(
                        "[DaseR] cold transfer writes drained "
                        "(commits=%d, late_evicted=%d, lookups=%d/%d)",
                        committed,
                        late,
                        int(stats.get("lookup_hits", 0)),
                        int(stats.get("lookup_requests", 0)),
                    )
                    return
                if committed >= expected_commits:
                    if require_l2_drain:
                        client.transfer_drain()
                    logger.info(
                        "[DaseR] all %d cold chunks committed", expected_commits
                    )
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for DaseR commits "
                        f"({committed}/{expected_commits})"
                    )
                time.sleep(0.05)
        finally:
            client.close()

    def visible_prompt_count(
        self,
        prompts: list[list[int]],
        model_id: str,
        block_tokens: int,
    ) -> int:
        """Return how many prompt prefixes are currently visible in DaseR.

        Args:
            prompts: Tokenized benchmark prompts.
            model_id: DaseR model ID used by the harness.
            block_tokens: block size used by the index.

        Returns:
            Number of prompts whose aligned prefix can be looked up.
        """
        return sum(self.visible_prompt_mask(prompts, model_id, block_tokens))

    def visible_prompt_mask(
        self,
        prompts: list[list[int]],
        model_id: str,
        block_tokens: int,
    ) -> list[bool]:
        """Return per-prompt DaseR lookup visibility before a warm pass.

        Args:
            prompts: Tokenized benchmark prompts.
            model_id: DaseR model ID used by the harness.
            block_tokens: block size used by the index.

        Returns:
            Boolean list aligned with ``prompts``.
        """
        client = IPCClientSync(self.socket_path)
        visible: list[bool] = []
        try:
            for tokens in prompts:
                aligned = (len(tokens) // block_tokens) * block_tokens
                if aligned <= 0:
                    visible.append(False)
                    continue
                chunks = client.lookup(tokens[:aligned], model_id)
                visible.append(
                    bool(chunks) and int(chunks[0].get("token_count", 0)) == aligned
                )
        finally:
            client.close()
        return visible

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
        try:
            os.rmdir(self.socket_dir)
        except OSError:
            pass
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
        local_cpu: bool,
        disk_limit_gb: float,
        cpu_limit_gb: float,
    ) -> None:
        """Initialise paths.

        Args:
            tmpdir: Directory used as LMCache's local_disk.
            total_bytes: Expected bytes-on-disk (drives max_local_disk_size).
            model_path: HF model path for vLLM.
            gpu_util: vLLM ``gpu_memory_utilization``.
        """
        self.tmpdir = tmpdir
        self.model_path = model_path
        self.total_bytes = total_bytes
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.local_cpu = local_cpu
        self.disk_limit_gb = disk_limit_gb
        self.cpu_limit_gb = cpu_limit_gb
        self._saved_env: dict[str, str | None] = {}

    def start(self) -> None:
        """Apply LMCache env configuration before LLM init."""
        env = {
            "LMCACHE_CHUNK_SIZE": str(BLOCK_TOKENS),
            "LMCACHE_LOCAL_CPU": "True" if self.local_cpu else "False",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": f"{self.cpu_limit_gb:.6f}",
            "LMCACHE_LOCAL_DISK": f"file://{self.tmpdir}/",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": f"{self.disk_limit_gb:.6f}",
            "LMCACHE_USE_LAYERWISE": "False",
            # Stable instance id + hash seed so cold-pass stores are visible
            # to the warm-pass lookup after the LLM is rebuilt.
            "LMCACHE_LMCACHE_INSTANCE_ID": "daser_vs_lmcache_bench",
            "PYTHONHASHSEED": "0",
        }
        for k, v in env.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v
        logger.info(
            "[LMCache] env configured — local_disk=%s (%s GB-config ceiling)",
            self.tmpdir,
            env["LMCACHE_MAX_LOCAL_DISK_SIZE"],
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
            max_model_len=MAX_MODEL_LEN,
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
    warm_skip_save: bool = False,
    after_cold_fn: Any | None = None,
) -> dict[str, Any]:
    """Run cold + warm timed passes for one system.

    Args:
        name: System label, used only for logging.
        build_llm_fn: Callable returning a fresh LLM instance.
        prompts: Prompt list to pass to generate().
        after_cold_fn: Optional callback run after cold generation and before
            stopping the cold timer. DaseR uses this to include save
            commit/drain cost in cold elapsed time.

    Returns:
        Dict with cold_elapsed_s, warm_elapsed_s, cold_outputs, warm_outputs.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=CORRECTNESS_LOGPROBS,
    )
    warm_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=CORRECTNESS_LOGPROBS,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        if warm_skip_save
        else params
    )

    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]

    # NOTE: we intentionally do NOT destroy and rebuild the LLM between cold
    # and warm passes. LMCache's LocalDiskBackend keeps its chunk index in an
    # in-memory dict and does not scan the directory on startup, so rebuilding
    # the engine would orphan every chunk it just wrote. vLLM's in-GPU KV is
    # recycled between generate() calls with enable_prefix_caching=False, so
    # the warm pass still has to fetch from the external storage tier — which
    # is exactly the signal this benchmark measures.
    logger.info("[%s] building LLM", name)
    llm = build_llm_fn()

    logger.info("[%s] cold: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    cold_outputs = llm.generate(tp_prompts, params)
    if after_cold_fn is not None:
        logger.info("[%s] cold: waiting for save completion", name)
        after_cold_fn()
    cold_elapsed = time.perf_counter() - t0
    logger.info("[%s] cold elapsed: %.2fs", name, cold_elapsed)

    logger.info("[%s] warm: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    warm_outputs = llm.generate(tp_prompts, warm_params)
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
    name: str,
    cold_outputs: list,
    warm_outputs: list,
    prompts: list[list[int]],
    max_num_seqs: int,
) -> dict[str, Any]:
    """Compare cold vs warm outputs with a logprob tolerance.

    Args:
        name: System label used in diagnostics.
        cold_outputs: Outputs from the cold timed pass.
        warm_outputs: Outputs from the warm timed pass.
        prompts: Tokenized benchmark inputs in output order.
        max_num_seqs: vLLM admission limit used for mismatch diagnostics.

    Returns:
        Correctness counters using logprob delta tolerance. Generated token IDs
        are not compared for exact equality.
    """
    mismatches = 0
    mismatch_indices: list[int] = []
    prompt_alignment_mismatches = 0
    max_logprob_delta = 0.0
    total = len(cold_outputs)
    for i, (c, w) in enumerate(zip(cold_outputs, warm_outputs, strict=False)):
        cold_prompt = list(getattr(c, "prompt_token_ids", prompts[i]))
        warm_prompt = list(getattr(w, "prompt_token_ids", prompts[i]))
        if cold_prompt != warm_prompt or cold_prompt != list(prompts[i]):
            prompt_alignment_mismatches += 1
            if prompt_alignment_mismatches <= 3:
                logger.warning(
                    "[%s] prompt %d alignment differs: input=%d cold=%d warm=%d",
                    name,
                    i,
                    len(prompts[i]),
                    len(cold_prompt),
                    len(warm_prompt),
                )
        output_delta = _output_logprob_delta(c, w)
        max_logprob_delta = max(max_logprob_delta, output_delta)
        if output_delta > CORRECTNESS_LOGPROB_TOLERANCE:
            mismatches += 1
            mismatch_indices.append(i)
            if mismatches <= 3:
                logger.warning(
                    "[%s] prompt %d (wave=%d pos=%d len=%d): cold/warm "
                    "logprob delta %.4g exceeds tolerance %.4g",
                    name,
                    i,
                    i // max(1, max_num_seqs),
                    i % max(1, max_num_seqs),
                    len(prompts[i]),
                    output_delta,
                    CORRECTNESS_LOGPROB_TOLERANCE,
                )
    if mismatches:
        logger.warning(
            "[%s] %d/%d prompts mismatched beyond tolerance",
            name,
            mismatches,
            total,
        )
        logger.warning("[%s] mismatch indices: %s", name, mismatch_indices)
    else:
        logger.info(
            "[%s] correctness OK with tolerance %.4g (%d/%d pass)",
            name,
            CORRECTNESS_LOGPROB_TOLERANCE,
            total,
            total,
        )
    if prompt_alignment_mismatches:
        logger.warning(
            "[%s] %d/%d prompt alignments mismatched",
            name,
            prompt_alignment_mismatches,
            total,
        )
    return {
        "mismatches": mismatches,
        "total": total,
        "indices": mismatch_indices,
        "logprob_tolerance": CORRECTNESS_LOGPROB_TOLERANCE,
        "max_logprob_delta": max_logprob_delta,
        "prompt_alignment_mismatches": prompt_alignment_mismatches,
    }


def _output_logprob_delta(cold_output: Any, warm_output: Any) -> float:
    """Return the generated-token logprob delta without token ID equality."""
    cold_completion = cold_output.outputs[0]
    warm_completion = warm_output.outputs[0]
    cold_cumulative = getattr(cold_completion, "cumulative_logprob", None)
    warm_cumulative = getattr(warm_completion, "cumulative_logprob", None)
    if cold_cumulative is not None and warm_cumulative is not None:
        return abs(float(cold_cumulative) - float(warm_cumulative))

    cold_logprobs = _sampled_logprobs(cold_completion)
    warm_logprobs = _sampled_logprobs(warm_completion)
    if len(cold_logprobs) != len(warm_logprobs):
        return math.inf
    if not cold_logprobs and not warm_logprobs:
        return math.inf
    return max(
        abs(cold_value - warm_value)
        for cold_value, warm_value in zip(cold_logprobs, warm_logprobs, strict=False)
    )


def _sampled_logprobs(completion: Any) -> list[float]:
    """Return one sampled-token logprob per generated output position."""
    logprobs = getattr(completion, "logprobs", None)
    token_ids = list(getattr(completion, "token_ids", ()))
    if logprobs is None:
        return []
    sampled: list[float] = []
    for pos, token_id in enumerate(token_ids):
        try:
            position_logprobs = logprobs[pos]
        except (IndexError, KeyError, TypeError):
            return []
        if position_logprobs is None:
            return []
        value = position_logprobs.get(token_id)
        if value is None:
            return []
        sampled.append(float(getattr(value, "logprob", value)))
    return sampled


def correctness_check_with_visibility(
    name: str,
    cold_outputs: list,
    warm_outputs: list,
    prompts: list[list[int]],
    max_num_seqs: int,
    visible_mask: list[bool],
) -> dict[str, Any]:
    """Compare cold/warm outputs and split mismatch counts by visible hits.

    Args:
        name: System label, used only for logging.
        cold_outputs: Outputs from the cold timed pass.
        warm_outputs: Outputs from the warm timed pass.
        prompts: Tokenized benchmark inputs in output order.
        max_num_seqs: vLLM admission limit, used only for diagnostics.
        visible_mask: Per-prompt boolean indicating whether the aligned DaseR
            prefix was visible before the warm pass.

    Returns:
        Dict with total and visible-hit mismatch counters.
    """
    result = correctness_check(name, cold_outputs, warm_outputs, prompts, max_num_seqs)
    visible_total = 0
    visible_mismatches = 0
    visible_max_logprob_delta = 0.0
    for cold, warm, visible in zip(
        cold_outputs, warm_outputs, visible_mask, strict=False
    ):
        if not visible:
            continue
        visible_total += 1
        output_delta = _output_logprob_delta(cold, warm)
        visible_max_logprob_delta = max(visible_max_logprob_delta, output_delta)
        if output_delta > CORRECTNESS_LOGPROB_TOLERANCE:
            visible_mismatches += 1
    result["visible_mismatches"] = visible_mismatches
    result["visible_total"] = visible_total
    result["visible_max_logprob_delta"] = visible_max_logprob_delta
    if visible_total:
        logger.info(
            "[%s] visible-hit correctness: %d/%d mismatched beyond tolerance",
            name,
            visible_mismatches,
            visible_total,
        )
    return result


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
    comparison_mode: str,
) -> dict[str, Any]:
    """Derive tok/s and speedups for the report."""
    summary: dict[str, Any] = {
        "comparison_mode": comparison_mode,
        "prompt_tokens_total": prompt_tokens,
    }
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
            "backend": r.get("backend"),
            "storage_tier": r.get("storage_tier"),
            "warm_skip_save": r.get("warm_skip_save", False),
            "visible_prompt_count": r.get("visible_prompt_count"),
        }
    d = summary.get("daser", {})
    lm = summary.get("lmcache", {})
    if not d.get("skipped") and not lm.get("skipped"):
        dw = d.get("warm_tok_per_s") or 0.0
        lw = lm.get("warm_tok_per_s") or 0.0
        dc = d.get("cold_tok_per_s") or 0.0
        lc = lm.get("cold_tok_per_s") or 0.0
        summary["warm_tps_ratio_daser_over_lmcache"] = dw / lw if lw > 0 else None
        summary["cold_tps_ratio_daser_over_lmcache"] = dc / lc if lc > 0 else None
    return summary


def print_report(config: dict[str, Any], summary: dict[str, Any]) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 72)
    print("E2E vLLM Benchmark — DaseR vs LMCache")
    print("=" * 72)
    print(f"Model            : {config['model']}")
    print(f"Comparison mode  : {config['comparison_mode']}")
    print(f"Prompts          : {config['num_prompts']} (IMDB reviews)")
    print(f"Prompt tokens    : {summary['prompt_tokens_total']:,}")
    print(
        "Sampling         : "
        f"temperature=0, max_tokens=1, logprobs={CORRECTNESS_LOGPROBS}"
    )
    print(f"Correctness      : logprob delta <= {CORRECTNESS_LOGPROB_TOLERANCE:g}")
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
        return f"{v:.2f}×" if v is not None else "N/A"

    dv = None if d.get("skipped") else d.get("warm_cold_speedup")
    lv = None if lm.get("skipped") else lm.get("warm_cold_speedup")
    print(f"{'warm/cold speedup':<28}{_speedup(dv):>20}{_speedup(lv):>20}")

    ratio = summary.get("warm_tps_ratio_daser_over_lmcache")
    print("-" * 72)
    if ratio is not None:
        print(f"DaseR warm tok/s / LMCache warm tok/s = {ratio:.2f}×")
    print("=" * 72)


@dataclass(frozen=True)
class BenchmarkSizing:
    """Derived transfer and cache capacities for one benchmark run.

    Attributes:
        daser_slots: number of DaseR L2 slots.
        daser_store_bytes: DaseR L2 bytes.
        daser_l1_bytes: DaseR L1 bytes.
        lmcache_disk_gb: LMCache local disk limit in its GB config unit.
        lmcache_cpu_gb: LMCache local CPU limit in its GB config unit.
    """

    daser_slots: int
    daser_store_bytes: int
    daser_l1_bytes: int
    lmcache_disk_gb: float
    lmcache_cpu_gb: float


def _derive_sizing(
    total_blocks: int,
    mode: str,
    evict: bool,
) -> BenchmarkSizing:
    """Derive L1/L2 sizes for no-evict and evict benchmark scenarios.

    Args:
        total_blocks: KV blocks in the workload.
        mode: comparison mode.
        evict: when True, choose capacities that force L2 eviction.

    Returns:
        BenchmarkSizing with aligned DaseR and LMCache capacities.
    """
    if evict:
        l2_blocks = max(1, math.floor(total_blocks * EVICT_L2_FRACTION))
        if l2_blocks >= total_blocks:
            l2_blocks = max(1, total_blocks - 1)
        l1_blocks = max(1, math.floor(l2_blocks * EVICT_L1_FRACTION))
    else:
        l2_blocks = max(1, math.ceil(total_blocks * 1.5))
        l1_blocks = max(1, math.ceil(total_blocks * 1.25))

    daser_store_bytes = l2_blocks * SLOT_SIZE
    daser_l1_bytes = l1_blocks * SLOT_SIZE if mode == COMPARISON_IOURING_MEM else 0
    lmcache_disk_gb = _bytes_to_lmcache_gb(daser_store_bytes)
    lmcache_cpu_gb = (
        _bytes_to_lmcache_gb(daser_l1_bytes)
        if mode == COMPARISON_IOURING_MEM
        else LMCACHE_LOCAL_SSD_STAGING_GB
    )
    return BenchmarkSizing(
        daser_slots=l2_blocks,
        daser_store_bytes=daser_store_bytes,
        daser_l1_bytes=daser_l1_bytes,
        lmcache_disk_gb=lmcache_disk_gb,
        lmcache_cpu_gb=lmcache_cpu_gb,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — argparse + orchestration
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--model", required=True)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--imdb", required=True)
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
        help="vLLM max_num_seqs (default: 64).",
    )
    parser.add_argument("--skip-daser", action="store_true")
    parser.add_argument("--skip-lmcache", action="store_true")
    parser.add_argument(
        "--comparison-mode",
        choices=(COMPARISON_GDS, COMPARISON_IOURING_MEM),
        default=COMPARISON_GDS,
    )
    parser.add_argument(
        "--evict",
        action="store_true",
        help="Choose DaseR L2/L1 sizes that force eviction during the workload.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")
    os.makedirs(args.store_dir, exist_ok=True)

    # ---- tokenise prompts ----
    logger.info("loading prompts from %s", args.imdb)
    raw_prompts = load_prompts(args.imdb, args.num_prompts)
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
    max_num_seqs = args.max_num_seqs
    token_counts = [len(ids) for ids in prompts]
    prompt_tokens_total = sum(token_counts)
    total_blocks = sum(c // BLOCK_TOKENS for c in token_counts)
    max_prompt_blocks = max((c // BLOCK_TOKENS for c in token_counts), default=1)
    logger.info(
        "tokenised %d prompts, %d tokens, %d blocks (avg %.1f, max %d blocks/prompt)",
        len(prompts),
        prompt_tokens_total,
        total_blocks,
        total_blocks / max(1, len(prompts)),
        max_prompt_blocks,
    )

    # ---- sizes ----
    total_bytes = total_blocks * SLOT_SIZE
    sizing = _derive_sizing(
        total_blocks=total_blocks,
        mode=args.comparison_mode,
        evict=args.evict,
    )
    transfer_mode = (
        "iouring" if args.comparison_mode == COMPARISON_IOURING_MEM else "gds"
    )
    logger.info(
        "store sizing: total_bytes=%.2fGiB, daser_slots=%d, l1=%.2fGiB, evict=%s",
        total_bytes / BYTES_PER_GIB,
        sizing.daser_slots,
        sizing.daser_l1_bytes / BYTES_PER_GIB,
        args.evict,
    )

    config = {
        "comparison_mode": args.comparison_mode,
        "evict": args.evict,
        "num_prompts": len(prompts),
        "model": args.model,
        "block_tokens": BLOCK_TOKENS,
        "slot_bytes": SLOT_SIZE,
        "max_input_tokens": args.max_input_tokens,
        "max_num_seqs": max_num_seqs,
        "total_blocks": total_blocks,
        "max_prompt_blocks": max_prompt_blocks,
        "total_bytes": total_bytes,
        "daser_transfer_mode": transfer_mode,
        "daser_slots": sizing.daser_slots,
        "daser_store_bytes": sizing.daser_store_bytes,
        "daser_l1_bytes": sizing.daser_l1_bytes,
        "lmcache_disk_gb": sizing.lmcache_disk_gb,
        "lmcache_cpu_gb": sizing.lmcache_cpu_gb,
        "daser_warm_skip_save": True,
    }

    # ---- LMCache run ----
    # Run LMCache before DaseR. The DaseR server opens CUDA IPC buffers in the
    # benchmark parent process, and forking another vLLM EngineCore after that
    # can fail CUDA initialization.
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
                max_num_seqs,
                args.comparison_mode == COMPARISON_IOURING_MEM,
                sizing.lmcache_disk_gb,
                sizing.lmcache_cpu_gb,
            )
            try:
                h_lm.start()
                r = run_system("LMCache", h_lm.build_llm, prompts)
                r["correctness"] = correctness_check(
                    "LMCache",
                    r["cold_outputs"],
                    r["warm_outputs"],
                    prompts,
                    max_num_seqs,
                )
                r.pop("cold_outputs", None)
                r.pop("warm_outputs", None)
                r["backend"] = "lmcache"
                r["storage_tier"] = (
                    "local-ssd-mem"
                    if args.comparison_mode == COMPARISON_IOURING_MEM
                    else "local-ssd"
                )
                r["warm_skip_save"] = False
                r["disk_limit_gb"] = sizing.lmcache_disk_gb
                r["cpu_limit_gb"] = sizing.lmcache_cpu_gb
                lmcache_result = r
            finally:
                h_lm.stop()

    # ---- DaseR run ----
    daser_result: dict[str, Any] | None = None
    if args.skip_daser:
        daser_result = {"skipped": True, "reason": "--skip-daser"}
    else:
        daser_dir = tempfile.mkdtemp(prefix="daser_bench_", dir=args.store_dir)
        socket_dir = tempfile.mkdtemp(prefix="daser_bench_ipc_")
        h = DaserHarness(
            daser_dir,
            socket_dir,
            sizing.daser_slots,
            args.model,
            args.gpu_util,
            max_num_seqs,
            transfer_mode,
            sizing.daser_l1_bytes,
        )
        try:
            h.start()
            visible_mask: list[bool] = []
            r = run_system(
                "DaseR",
                h.build_llm,
                prompts,
                warm_skip_save=True,
                after_cold_fn=lambda: h.wait_until_committed(
                    prompts,
                    BLOCK_TOKENS,
                    require_all_commits=not args.evict,
                    require_l2_drain=(
                        args.evict or args.comparison_mode == COMPARISON_IOURING_MEM
                    ),
                ),
            )
            visible_mask = h.visible_prompt_mask(
                prompts,
                "qwen3-8b",
                BLOCK_TOKENS,
            )
            r["correctness"] = correctness_check_with_visibility(
                "DaseR",
                r["cold_outputs"],
                r["warm_outputs"],
                prompts,
                max_num_seqs,
                visible_mask,
            )
            r.pop("cold_outputs", None)
            r.pop("warm_outputs", None)
            r["backend"] = transfer_mode
            r["storage_tier"] = (
                "local-ssd-mem"
                if args.comparison_mode == COMPARISON_IOURING_MEM
                else "local-ssd"
            )
            r["warm_skip_save"] = True
            r["store_bytes"] = sizing.daser_store_bytes
            r["l1_bytes"] = sizing.daser_l1_bytes
            r["visible_prompt_count"] = sum(visible_mask)
            daser_result = r
        finally:
            h.stop()

    # ---- report ----
    summary = build_summary(
        daser_result,
        lmcache_result,
        prompt_tokens_total,
        args.comparison_mode,
    )
    print_report(config, summary)

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
