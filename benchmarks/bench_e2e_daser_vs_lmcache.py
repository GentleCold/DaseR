# SPDX-License-Identifier: Apache-2.0
"""End-to-end inference benchmark: DaseR vs LMCache.

Runs the same IMDB-review prompt batch through vLLM twice, once with each
KV connector, measuring cold-pass and warm-pass elapsed time and prompt-token
throughput. Prefix cache is disabled so the external KV storage tier is the
only source of cross-run speedup. DaseR can run either `gds` or `iouring-mem`;
LMCache can run LocalDiskBackend or local CPU mode.

Usage:
    source /data/zwt/vllm/bin/activate
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python benchmarks/bench_e2e_daser_vs_lmcache.py \\
        [--num-prompts 200] [--imdb /data/zwt/imdb.csv] \\
        [--transfer-backend iouring-mem] [--evict] \\
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
MODEL_PATH_DEFAULT: str = "/data/zwt/model/models/Qwen/Qwen3-8B"
NUM_KV_HEADS: int = 8
HEAD_DIM: int = 128
NUM_LAYERS: int = 36
BLOCK_TOKENS: int = 16
DTYPE_BYTES: int = 2  # bfloat16
SLOT_SIZE: int = NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS * BLOCK_TOKENS * DTYPE_BYTES
# 8 * 128 * 2 * 36 * 16 * 2 = 2,359,296 bytes

MAX_MODEL_LEN: int = 2048
MAX_INPUT_TOKENS_DEFAULT: int = 1792
GPU_MEM_UTIL_DEFAULT: float = 0.4
MAX_NUM_SEQS_DEFAULT: int = 64
DEFAULT_STORE_HEADROOM: float = 1.5
NON_EVICT_L1_FRACTION: float = 0.5
NON_EVICT_L2_HEADROOM: float = 1.5
EVICT_L1_FRACTION: float = 0.25
EVICT_L2_FRACTION: float = 0.5
LMCACHE_DISK_BUFFER_BYTES: int = 1 * 1024**3


@dataclass(frozen=True)
class CacheSizing:
    """Derived DaseR and LMCache cache capacities for one benchmark run.

    Attributes:
        slots_needed: number of DaseR L2 ring-buffer slots to preallocate.
        l1_bytes: DaseR pinned host L1 bytes.
        l2_bytes: DaseR SSD L2 bytes.
        lmcache_cpu_bytes: LMCache local CPU capacity.
        lmcache_disk_bytes: LMCache local SSD capacity.
        lmcache_mode: LMCache comparison mode label.
        lmcache_local_cpu: whether LMCache local CPU tier is enabled.
        kv_to_l1_ratio: total KV bytes divided by L1 bytes, if any.
        store_to_l1_ratio: L2 bytes divided by L1 bytes, if any.
    """

    slots_needed: int
    l1_bytes: int
    l2_bytes: int
    lmcache_cpu_bytes: int
    lmcache_disk_bytes: int
    lmcache_mode: str
    lmcache_local_cpu: bool
    kv_to_l1_ratio: float | None
    store_to_l1_ratio: float | None


_SIZE_UNITS = {
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

# ---------------------------------------------------------------------------
# Workload loader
# ---------------------------------------------------------------------------


def _parse_bench_size(value: str) -> int:
    """Parse a benchmark byte size with optional suffix.

    Args:
        value: integer byte count or value with kb/mb/gb/kib/mib/gib suffix.

    Returns:
        Size in bytes.
    """
    stripped = value.strip().lower()
    number = ""
    unit = ""
    for char in stripped:
        if char.isdigit():
            if unit:
                raise argparse.ArgumentTypeError(f"invalid size: {value}")
            number += char
        else:
            unit += char
    if not number or unit not in _SIZE_UNITS:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")
    return int(number) * _SIZE_UNITS[unit]


def _slots_for_bytes(size_bytes: int) -> int:
    """Return the number of complete DaseR slots needed for byte capacity."""
    return max(1, math.ceil(size_bytes / SLOT_SIZE))


def _resolve_cache_sizes(
    *,
    total_blocks: int,
    transfer_backend: str,
    evict: bool,
) -> CacheSizing:
    """Resolve aligned DaseR and LMCache cache capacities.

    Args:
        total_blocks: number of DaseR slots required by the prompt batch.
        transfer_backend: DaseR transfer backend under test.
        evict: when True, size iouring-mem as total KV > L2 > L1; otherwise
            size it as L2 > total KV > L1.

    Returns:
        CacheSizing with DaseR and LMCache capacities aligned.

    Raises:
        ValueError: if the backend name is unsupported.
    """
    total_bytes = total_blocks * SLOT_SIZE
    if transfer_backend == "gds":
        slots_needed = math.ceil(DEFAULT_STORE_HEADROOM * total_blocks)
        l2_bytes = slots_needed * SLOT_SIZE
        return CacheSizing(
            slots_needed=slots_needed,
            l1_bytes=0,
            l2_bytes=l2_bytes,
            lmcache_cpu_bytes=0,
            lmcache_disk_bytes=l2_bytes,
            lmcache_mode="local-disk",
            lmcache_local_cpu=False,
            kv_to_l1_ratio=None,
            store_to_l1_ratio=None,
        )

    if transfer_backend != "iouring-mem":
        raise ValueError(f"unsupported transfer backend: {transfer_backend}")

    if evict:
        l1_target = max(SLOT_SIZE, math.floor(total_bytes * EVICT_L1_FRACTION))
        l2_target = max(
            l1_target + SLOT_SIZE,
            math.floor(total_bytes * EVICT_L2_FRACTION),
        )
        if l2_target >= total_bytes:
            l2_target = max(l1_target + SLOT_SIZE, total_bytes - SLOT_SIZE)
        if not total_bytes > l2_target > l1_target > 0:
            raise ValueError(
                "--evict requires enough KV bytes to derive total KV > L2 > L1; "
                "increase --num-prompts or --max-input-tokens"
            )
    else:
        l1_target = max(SLOT_SIZE, math.floor(total_bytes * NON_EVICT_L1_FRACTION))
        l2_target = math.ceil(total_bytes * NON_EVICT_L2_HEADROOM)

    slots_needed = _slots_for_bytes(l2_target)
    l2_bytes = slots_needed * SLOT_SIZE
    if evict and l2_bytes >= total_bytes:
        slots_needed = max(1, total_blocks - 1)
        l2_bytes = slots_needed * SLOT_SIZE
        if l2_bytes <= l1_target:
            l1_target = max(SLOT_SIZE, l2_bytes - SLOT_SIZE)
    l1_bytes = int(l1_target)
    if evict and not total_bytes > l2_bytes > l1_bytes > 0:
        raise ValueError(
            "--evict requires derived total KV > L2 > L1; increase "
            "--num-prompts or --max-input-tokens"
        )
    if not evict and not l2_bytes > total_bytes > l1_bytes > 0:
        raise ValueError(
            "non-evict mode requires derived L2 > total KV > L1; increase "
            "--num-prompts or --max-input-tokens"
        )

    return CacheSizing(
        slots_needed=slots_needed,
        l1_bytes=l1_bytes,
        l2_bytes=l2_bytes,
        lmcache_cpu_bytes=l1_bytes,
        lmcache_disk_bytes=l2_bytes,
        lmcache_mode="local-cpu-disk",
        lmcache_local_cpu=True,
        kv_to_l1_ratio=total_bytes / l1_bytes,
        store_to_l1_ratio=l2_bytes / l1_bytes,
    )


def _fallback_prompts() -> list[str]:
    """Two long prompts from tests/integration/test_vllm_e2e.py (fallback)."""
    return [
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


def load_prompts(imdb_path: str, n: int) -> list[str]:
    """Load N IMDB reviews as raw prompt strings.

    Args:
        imdb_path: Path to imdb.csv with a 'review' column.
        n: Number of prompts to return.

    Returns:
        List of raw review strings. Falls back to synthetic prompts if the
        CSV is missing.
    """
    if not os.path.exists(imdb_path):
        logger.warning(
            "IMDB CSV not found at %s — falling back to synthetic prompts",
            imdb_path,
        )
        base = _fallback_prompts()
        return [base[i % len(base)] for i in range(n)]

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
        tmpdir: str,
        total_slots: int,
        model_path: str,
        gpu_util: float,
        max_num_seqs: int,
        transfer_backend: str = "gds",
        l1_cache_size: int = 0,
    ) -> None:
        """Initialise paths and store file.

        Args:
            tmpdir: Directory to hold socket + store files.
            total_slots: Pre-allocated slot count for the store.
            model_path: HF model path for vLLM.
            gpu_util: vLLM ``gpu_memory_utilization``.
            max_num_seqs: vLLM ``max_num_seqs``.
            transfer_backend: DaseR transfer backend.
            l1_cache_size: pinned host L1 cache bytes.
        """
        self.tmpdir = tmpdir
        self.socket_path = os.path.join(tmpdir, "daser.sock")
        self.store_path = os.path.join(tmpdir, "daser.store")
        self.model_path = model_path
        self.total_slots = total_slots
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.transfer_backend = transfer_backend
        self.l1_cache_size = l1_cache_size
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
                "transfer_backend": self.transfer_backend,
                "l1_cache_size": self.l1_cache_size,
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
            "[DaseR] server up — store=%s (%.1f GB, %d slots) transfer=%s l1=%.1fGB",
            self.store_path,
            size / 1e9,
            self.total_slots,
            self.transfer_backend,
            self.l1_cache_size / 1e9,
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
        max_local_cpu_bytes: int,
        max_local_disk_bytes: int,
        model_path: str,
        gpu_util: float,
        max_num_seqs: int,
        local_cpu: bool = False,
    ) -> None:
        """Initialise paths.

        Args:
            tmpdir: Directory used as LMCache's local_disk.
            total_bytes: Expected bytes-on-disk (drives max_local_disk_size).
            max_local_cpu_bytes: LMCache local CPU capacity in bytes.
            max_local_disk_bytes: LMCache local SSD capacity in bytes.
            model_path: HF model path for vLLM.
            gpu_util: vLLM ``gpu_memory_utilization``.
            max_num_seqs: vLLM ``max_num_seqs``.
            local_cpu: enable LMCache local CPU tier in addition to local disk.
        """
        self.tmpdir = tmpdir
        self.model_path = model_path
        self.total_bytes = total_bytes
        self.max_local_cpu_bytes = max_local_cpu_bytes
        self.max_local_disk_bytes = max_local_disk_bytes
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.local_cpu = local_cpu
        self._saved_env: dict[str, str | None] = {}

    def start(self) -> None:
        """Apply LMCache env configuration before LLM init."""
        local_cpu_bytes = self.max_local_cpu_bytes
        if not self.local_cpu and self.max_local_disk_bytes > 0:
            local_cpu_bytes = max(local_cpu_bytes, LMCACHE_DISK_BUFFER_BYTES)
        env = {
            "LMCACHE_CHUNK_SIZE": str(BLOCK_TOKENS),
            "LMCACHE_LOCAL_CPU": "True" if self.local_cpu else "False",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": f"{local_cpu_bytes / 1e9:.3f}",
            "LMCACHE_LOCAL_DISK": f"file://{self.tmpdir}/",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": f"{self.max_local_disk_bytes / 1e9:.3f}",
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
            "[LMCache] env configured — local_cpu=%s local_disk=%s "
            "(cpu=%sGB disk=%sGB)",
            self.local_cpu,
            self.tmpdir,
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"],
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
    warmup_prompt: list[int],
    drain_prompt: list[int] | None = None,
    skip_warm_save: bool = False,
) -> dict[str, Any]:
    """Run cold + warm timed passes for one system.

    Args:
        name: System label, used only for logging.
        build_llm_fn: Callable returning a fresh LLM instance.
        prompts: Prompt list to pass to generate().
        warmup_prompt: Untimed warmup prompt (short).
        drain_prompt: optional prompt marked with connector params to skip
            saving. Running it between cold and warm forces connector background
            stores from the cold pass to drain before warm measurement.
        skip_warm_save: when True, pass connector params that suppress saving
            during the measured warm read pass.

    Returns:
        Dict with cold_elapsed_s, warm_elapsed_s, cold_outputs, warm_outputs.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

    params = SamplingParams(temperature=0.0, max_tokens=1)
    warm_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        if skip_warm_save
        else params
    )
    warmup_params = SamplingParams(temperature=0.0, max_tokens=1)
    drain_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"kv_transfer_params": {"daser_skip_save": True}},
    )

    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
    tp_warmup = TokensPrompt(prompt_token_ids=warmup_prompt)
    tp_drain = (
        TokensPrompt(prompt_token_ids=drain_prompt)
        if drain_prompt is not None
        else None
    )

    # NOTE: we intentionally do NOT destroy and rebuild the LLM between cold
    # and warm passes. LMCache's LocalDiskBackend keeps its chunk index in an
    # in-memory dict and does not scan the directory on startup, so rebuilding
    # the engine would orphan every chunk it just wrote. vLLM's in-GPU KV is
    # recycled between generate() calls with enable_prefix_caching=False, so
    # the warm pass still has to fetch from the external storage tier — which
    # is exactly the signal this benchmark measures.
    logger.info("[%s] building LLM", name)
    llm = build_llm_fn()

    logger.info("[%s] cold: warmup", name)
    llm.generate([tp_warmup], warmup_params)

    logger.info("[%s] cold: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    cold_outputs = llm.generate(tp_prompts, params)
    cold_elapsed = time.perf_counter() - t0
    logger.info("[%s] cold elapsed: %.2fs", name, cold_elapsed)

    if tp_drain is not None:
        logger.info("[%s] drain: wait for cold stores", name)
        llm.generate([tp_drain], drain_params)

    logger.info("[%s] warm: warmup", name)
    llm.generate([tp_warmup], warmup_params)

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
    daser: dict[str, Any] | None, lmcache: dict[str, Any] | None, prompt_tokens: int
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
    print("E2E vLLM Benchmark — DaseR vs LMCache")
    print("=" * 72)
    print(f"Model            : {config['model']}")
    print(f"Prompts          : {config['num_prompts']} (IMDB reviews)")
    print(f"Prompt tokens    : {summary['prompt_tokens_total']:,}")
    print(f"DaseR transfer   : {config['transfer_backend']}")
    print(f"LMCache mode     : {config['lmcache_mode']}")
    print(f"Eviction pressure: {config['evict']}")
    if config.get("kv_to_l1_ratio") is not None:
        print(f"KV/L1 ratio      : {config['kv_to_l1_ratio']:.2f}x")
        print(f"Store/L1 ratio   : {config['store_to_l1_ratio']:.2f}x")
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
        return f"{v:.2f}×" if v is not None else "N/A"

    dv = None if d.get("skipped") else d.get("warm_cold_speedup")
    lv = None if lm.get("skipped") else lm.get("warm_cold_speedup")
    print(f"{'warm/cold speedup':<28}{_speedup(dv):>20}{_speedup(lv):>20}")

    ratio = summary.get("warm_tps_ratio_daser_over_lmcache")
    print("-" * 72)
    if ratio is not None:
        print(f"DaseR warm tok/s / LMCache warm tok/s = {ratio:.2f}×")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — argparse + orchestration
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--model", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--store-dir", default="/data/zwt/daser_test")
    parser.add_argument("--imdb", default="/data/zwt/imdb.csv")
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
        "--transfer-backend",
        choices=("gds", "iouring-mem"),
        default="gds",
        help="DaseR transfer backend to benchmark.",
    )
    parser.add_argument(
        "--evict",
        action="store_true",
        help=(
            "For iouring-mem, size caches as total KV > L2 > L1. Without this "
            "flag, size as L2 > L1 > total KV."
        ),
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

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
    try:
        sizing = _resolve_cache_sizes(
            total_blocks=total_blocks,
            transfer_backend=args.transfer_backend,
            evict=args.evict,
        )
    except ValueError as exc:
        parser.error(str(exc))
    logger.info(
        "cache sizing: total_bytes=%.2fGB l1=%.2fGB l2=%.2fGB slots=%d evict=%s",
        total_bytes / 1e9,
        sizing.l1_bytes / 1e9,
        sizing.l2_bytes / 1e9,
        sizing.slots_needed,
        args.evict,
    )

    config = {
        "num_prompts": len(prompts),
        "model": args.model,
        "block_tokens": BLOCK_TOKENS,
        "slot_bytes": SLOT_SIZE,
        "max_input_tokens": args.max_input_tokens,
        "total_blocks": total_blocks,
        "total_bytes": total_bytes,
        "l1_bytes": sizing.l1_bytes,
        "l2_bytes": sizing.l2_bytes,
        "store_bytes": sizing.l2_bytes,
        "evict": args.evict,
        "kv_to_l1_ratio": sizing.kv_to_l1_ratio,
        "store_to_l1_ratio": sizing.store_to_l1_ratio,
        "transfer_backend": args.transfer_backend,
        "lmcache_mode": sizing.lmcache_mode,
        "lmcache_local_cpu": sizing.lmcache_local_cpu,
        "lmcache_max_local_cpu_size": sizing.lmcache_cpu_bytes,
        "lmcache_max_local_disk_size": sizing.lmcache_disk_bytes,
    }

    # ---- DaseR run ----
    daser_result: dict[str, Any] | None = None
    if args.skip_daser:
        daser_result = {"skipped": True, "reason": "--skip-daser"}
    else:
        daser_dir = tempfile.mkdtemp(prefix="daser_bench_", dir=args.store_dir)
        h = DaserHarness(
            daser_dir,
            sizing.slots_needed,
            args.model,
            args.gpu_util,
            args.max_num_seqs,
            transfer_backend=args.transfer_backend,
            l1_cache_size=sizing.l1_bytes,
        )
        try:
            h.start()
            r = run_system(
                "DaseR",
                h.build_llm,
                prompts,
                warmup_prompt_ids,
                drain_prompt=warmup_prompt_ids,
                skip_warm_save=True,
            )
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
                sizing.lmcache_cpu_bytes,
                sizing.lmcache_disk_bytes,
                args.model,
                args.gpu_util,
                args.max_num_seqs,
                local_cpu=sizing.lmcache_local_cpu,
            )
            try:
                h_lm.start()
                r = run_system(
                    "LMCache",
                    h_lm.build_llm,
                    prompts,
                    warmup_prompt_ids,
                    drain_prompt=warmup_prompt_ids,
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
