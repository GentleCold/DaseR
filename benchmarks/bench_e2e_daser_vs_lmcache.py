# SPDX-License-Identifier: Apache-2.0
"""End-to-end inference benchmark: DaseR transfer modes vs LMCache.

Runs the same IMDB-review prompt batch through vLLM twice, once with each
KV connector, measuring cold-pass and warm-pass elapsed time and prompt-token
throughput. Prefix cache is disabled so the NVMe storage tier is the only
source of cross-run speedup.

Usage:
    python benchmarks/bench_e2e_daser_vs_lmcache.py \\
        --model /path/to/model \\
        --store-dir /path/to/benchmark-scratch \\
        --imdb /path/to/imdb.csv \\
        [--num-prompts 200] \\
        [--out results.json]
"""

# ruff: noqa: E402

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any
import uuid

# ---------------------------------------------------------------------------
# Deterministic hashing — re-exec with PYTHONHASHSEED set so both LMCache
# scheduler-side token hashing and vLLM's NONE_HASH seed are stable across
# cold/warm LLM rebuilds. Must happen before *any* import that touches
# Python string hashing or vLLM internals.
# ---------------------------------------------------------------------------
BENCHMARK_SEED_ENV = "42"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != BENCHMARK_SEED_ENV:
    os.environ["PYTHONHASHSEED"] = BENCHMARK_SEED_ENV
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

# Select the benchmark GPU before importing torch or vLLM. The regular
# argparse parser is built later; this minimal parser intentionally ignores
# all other options.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu-id", default="auto")
_gpu_args, _ = _gpu_parser.parse_known_args()

# First Party — add project root for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.utils import (
    BYTES_PER_GIB,
    COMPARISON_GDS,
    COMPARISON_IOURING_MEM,
    apply_gpu_selection,
    derive_benchmark_sizing,
    derive_capacity_limits,
    load_prompts,
    set_global_seed,
    tokenise_and_truncate,
)

SELECTED_GPU_ID = (
    apply_gpu_selection(_gpu_args.gpu_id)
    if __name__ == "__main__"
    else os.environ.get("CUDA_VISIBLE_DEVICES")
)

# Third Party
import torch

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

MAX_MODEL_LEN: int = 2048
MAX_INPUT_TOKENS_DEFAULT: int = 1792
GPU_MEM_UTIL_DEFAULT: float = 0.9
MAX_NUM_SEQS_DEFAULT: int = 64
BENCHMARK_SEED: int = 42


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
                "cache_reuse_mode": "prefix",
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
        """Construct a vLLM LLM wired to DaserConnector.

        Returns:
            Configured vLLM ``LLM`` instance.
        """
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
            seed=BENCHMARK_SEED,
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
        digest = hashlib.sha1(tmpdir.encode("utf-8")).hexdigest()[:12]
        self.instance_id = f"daser_vs_lmcache_{digest}"
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
            "LMCACHE_LMCACHE_INSTANCE_ID": self.instance_id,
            "PYTHONHASHSEED": BENCHMARK_SEED_ENV,
        }
        for k, v in env.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v
        logger.info(
            "[LMCache] env configured — local_disk=%s (%s GB-config ceiling)",
            self.tmpdir,
            env["LMCACHE_MAX_LOCAL_DISK_SIZE"],
        )
        self._reset_process_state()

    def build_llm(self) -> Any:
        """Construct a vLLM LLM wired to LMCacheConnectorV1.

        Returns:
            Configured vLLM ``LLM`` instance.
        """
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
            seed=BENCHMARK_SEED,
            enable_prefix_caching=False,
        )

    def wait_for_disk_quiescence(
        self,
        timeout_s: float = 120.0,
        stable_for_s: float = 1.0,
        poll_s: float = 0.1,
    ) -> None:
        """Wait until LMCache local-disk files stop changing.

        LMCache's LocalDiskBackend submits SSD writes to a background worker and
        adds a key to the lookup index only after that key's write completes.
        The benchmark cannot call a public LMCache drain API, so it waits for
        the observable local-disk tier to become quiescent before the warm pass.

        Args:
            timeout_s: Maximum wait time in seconds.
            stable_for_s: Required duration with unchanged file count and bytes.
            poll_s: Poll interval in seconds.

        Raises:
            TimeoutError: If the local-disk snapshot does not become stable.
        """
        root = Path(self.tmpdir)
        deadline = time.monotonic() + timeout_s
        last_snapshot: tuple[int, int] | None = None
        stable_since: float | None = None
        while time.monotonic() < deadline:
            snapshot = self._disk_snapshot(root)
            if snapshot == last_snapshot and snapshot[0] > 0:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= stable_for_s:
                    logger.info(
                        "[LMCache] local disk quiescent: files=%d bytes=%d",
                        snapshot[0],
                        snapshot[1],
                    )
                    return
            else:
                last_snapshot = snapshot
                stable_since = None
            time.sleep(poll_s)
        raise TimeoutError(
            "LMCache local-disk writes did not become quiescent within "
            f"{timeout_s:.1f}s under {root}"
        )

    def stop(self) -> None:
        """Restore previous env values."""
        self._reset_process_state()
        for k, saved in self._saved_env.items():
            if saved is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved
        self._saved_env.clear()
        logger.info("[LMCache] env restored")

    def _reset_process_state(self) -> None:
        """Reset LMCache process-global vLLM integration state.

        LMCache caches its env-derived config and vLLM cache engine in
        process-global singletons. This benchmark creates multiple isolated
        LMCache harnesses in one process, so each harness clears that state
        before reading a new scratch-dir configuration.
        """
        try:
            from lmcache.integration.vllm import utils as vllm_utils  # type: ignore

            vllm_utils._config_instance = None  # noqa: SLF001
        except Exception as exc:
            logger.debug("[LMCache] config singleton reset skipped: %s", exc)
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME  # type: ignore
            from lmcache.v1.cache_engine import LMCacheEngineBuilder  # type: ignore

            LMCacheEngineBuilder.destroy(ENGINE_NAME)
        except Exception as exc:
            logger.debug("[LMCache] engine singleton reset skipped: %s", exc)

    @staticmethod
    def _disk_snapshot(root: Path) -> tuple[int, int]:
        """Return current LMCache local-disk file count and total bytes.

        Args:
            root: LMCache local-disk root.

        Returns:
            Tuple of ``(file_count, total_bytes)`` for stored ``.pt`` files.
        """
        count = 0
        total = 0
        for path in root.rglob("*.pt"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            count += 1
            total += stat.st_size
        return count, total


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
        seed=BENCHMARK_SEED,
    )
    warm_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
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
    llm.generate(tp_prompts, params)
    if after_cold_fn is not None:
        logger.info("[%s] cold: waiting for save completion", name)
        after_cold_fn()
    cold_elapsed = time.perf_counter() - t0
    logger.info("[%s] cold elapsed: %.2fs", name, cold_elapsed)

    logger.info("[%s] warm: generate(N=%d)", name, len(tp_prompts))
    t0 = time.perf_counter()
    llm.generate(tp_prompts, warm_params)
    warm_elapsed = time.perf_counter() - t0
    logger.info("[%s] warm elapsed: %.2fs", name, warm_elapsed)

    logger.info("[%s] destroying LLM", name)
    _destroy_llm(llm)

    return {
        "cold_elapsed_s": cold_elapsed,
        "warm_elapsed_s": warm_elapsed,
    }


def run_correctness_system(
    name: str,
    build_llm_fn: Any,
    prompts: list[list[int]],
    max_num_seqs: int,
    warm_skip_save: bool = False,
    after_cold_fn: Any | None = None,
    visible_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """Run an untimed cold/warm exact correctness pass.

    Args:
        name: System label, used only for logging.
        build_llm_fn: Callable returning an LLM instance.
        prompts: Prompt list to pass to generate().
        max_num_seqs: vLLM admission limit used for diagnostics.
        warm_skip_save: when True, skip DaseR duplicate warm stores.
        after_cold_fn: Optional callback run after cold correctness generation
            and before the warm correctness generation. DaseR uses this to
            make store commits visible before warm loads.
        visible_mask: Optional DaseR visible-hit mask for per-hit diagnostics.

    Returns:
        Correctness dictionary from ``correctness_check``.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=BENCHMARK_SEED,
    )
    warm_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        if warm_skip_save
        else params
    )
    tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]

    logger.info("[%s] correctness: building exact-check LLM", name)
    llm = build_llm_fn()
    try:
        logger.info("[%s] correctness: cold generate(N=%d)", name, len(tp_prompts))
        cold_outputs = llm.generate(tp_prompts, params)
        if after_cold_fn is not None:
            logger.info("[%s] correctness: waiting for save completion", name)
            after_cold_fn()
        logger.info("[%s] correctness: warm generate(N=%d)", name, len(tp_prompts))
        warm_outputs = llm.generate(tp_prompts, warm_params)
        if visible_mask is None:
            return correctness_check(
                name,
                cold_outputs,
                warm_outputs,
                prompts,
                max_num_seqs,
            )
        return correctness_check_with_visibility(
            name,
            cold_outputs,
            warm_outputs,
            prompts,
            max_num_seqs,
            visible_mask,
        )
    finally:
        logger.info("[%s] correctness: destroying LLM", name)
        _destroy_llm(llm)


def run_lmcache_correctness(
    store_dir: str,
    total_bytes: int,
    model_path: str,
    gpu_util: float,
    max_num_seqs: int,
    local_cpu: bool,
    disk_limit_gb: float,
    cpu_limit_gb: float,
    prompts: list[list[int]],
) -> dict[str, Any]:
    """Run LMCache exact correctness in an isolated scratch store.

    Args:
        store_dir: Base directory for LMCache benchmark scratch files.
        total_bytes: Workload byte size used for LMCache sizing.
        model_path: HF model path for vLLM.
        gpu_util: vLLM GPU memory utilization.
        max_num_seqs: vLLM max_num_seqs.
        local_cpu: Whether LMCache L1 CPU tier is enabled.
        disk_limit_gb: LMCache local-disk limit in GiB units.
        cpu_limit_gb: LMCache local-CPU limit in GiB units.
        prompts: Tokenized prompts for correctness.

    Returns:
        Exact correctness result dictionary.
    """
    lmcache_dir = tempfile.mkdtemp(prefix="lmcache_correctness_", dir=store_dir)
    h_lm = LMCacheHarness(
        lmcache_dir,
        total_bytes,
        model_path,
        gpu_util,
        max_num_seqs,
        local_cpu,
        disk_limit_gb,
        cpu_limit_gb,
    )
    try:
        h_lm.start()
        return run_correctness_system(
            name="LMCache",
            build_llm_fn=h_lm.build_llm,
            prompts=prompts,
            max_num_seqs=max_num_seqs,
            after_cold_fn=h_lm.wait_for_disk_quiescence,
        )
    finally:
        h_lm.stop()


def run_daser_correctness(
    store_dir: str,
    model_path: str,
    gpu_util: float,
    max_num_seqs: int,
    transfer_mode: str,
    l1_bytes: int,
    total_slots: int,
    prompts: list[list[int]],
    require_all_commits: bool,
    require_l2_drain: bool,
) -> dict[str, Any]:
    """Run DaseR exact correctness in an isolated server/store.

    Args:
        store_dir: Base directory for DaseR benchmark scratch files.
        model_path: HF model path for vLLM.
        gpu_util: vLLM GPU memory utilization.
        max_num_seqs: vLLM max_num_seqs.
        transfer_mode: DaseR transfer backend.
        l1_bytes: DaseR L1 byte capacity.
        total_slots: DaseR L2 slots.
        prompts: Tokenized prompts for correctness.
        require_all_commits: Whether all chunks must commit before warm.
        require_l2_drain: Whether tiered transfer must drain L2 before warm.

    Returns:
        Exact correctness result dictionary with visible-hit counters.
    """
    daser_dir = tempfile.mkdtemp(prefix="daser_correctness_", dir=store_dir)
    socket_dir = tempfile.mkdtemp(prefix="daser_correctness_ipc_")
    h = DaserHarness(
        daser_dir,
        socket_dir,
        total_slots,
        model_path,
        gpu_util,
        max_num_seqs,
        transfer_mode,
        l1_bytes,
    )
    try:
        h.start()
        from vllm import SamplingParams  # Third Party
        from vllm.inputs import TokensPrompt  # Third Party

        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
        )
        warm_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=BENCHMARK_SEED,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        tp_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]

        logger.info("[DaseR] correctness: building isolated exact-check LLM")
        llm = h.build_llm()
        try:
            logger.info("[DaseR] correctness: cold generate(N=%d)", len(tp_prompts))
            cold_outputs = llm.generate(tp_prompts, params)
            logger.info("[DaseR] correctness: waiting for save completion")
            h.wait_until_committed(
                prompts,
                BLOCK_TOKENS,
                require_all_commits=require_all_commits,
                require_l2_drain=require_l2_drain,
            )
            visible_mask = h.visible_prompt_mask(prompts, "qwen3-8b", BLOCK_TOKENS)
            logger.info("[DaseR] correctness: warm generate(N=%d)", len(tp_prompts))
            warm_outputs = llm.generate(tp_prompts, warm_params)
            return correctness_check_with_visibility(
                "DaseR",
                cold_outputs,
                warm_outputs,
                prompts,
                max_num_seqs,
                visible_mask,
            )
        finally:
            logger.info("[DaseR] correctness: destroying LLM")
            _destroy_llm(llm)
    finally:
        h.stop()


def correctness_check(
    name: str,
    cold_outputs: list,
    warm_outputs: list,
    prompts: list[list[int]],
    max_num_seqs: int,
) -> dict[str, Any]:
    """Compare cold vs warm generated output exactly.

    Args:
        name: System label used in diagnostics.
        cold_outputs: Outputs from the cold timed pass.
        warm_outputs: Outputs from the warm timed pass.
        prompts: Tokenized benchmark inputs in output order.
        max_num_seqs: vLLM admission limit used for mismatch diagnostics.

    Returns:
        Correctness counters. Only exact generated text and token-ID matches
        are accepted.
    """
    mismatches = 0
    mismatch_indices: list[int] = []
    mismatch_details: list[dict[str, Any]] = []
    prompt_alignment_mismatches = 0
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
        if _generated_token_ids(c) == _generated_token_ids(w) and _output_text(
            c
        ) == _output_text(w):
            continue

        mismatches += 1
        mismatch_indices.append(i)
        detail = {
            "index": i,
            "wave": i // max(1, max_num_seqs),
            "position": i % max(1, max_num_seqs),
            "prompt_tokens": len(prompts[i]),
            "cold_token_ids": _generated_token_ids(c),
            "warm_token_ids": _generated_token_ids(w),
            "cold_text": _output_text(c),
            "warm_text": _output_text(w),
        }
        mismatch_details.append(detail)
        if mismatches <= 3:
            logger.warning(
                "[%s] prompt %d (wave=%d pos=%d len=%d): text mismatch cold=%s warm=%s",
                name,
                i,
                detail["wave"],
                detail["position"],
                detail["prompt_tokens"],
                detail["cold_token_ids"],
                detail["warm_token_ids"],
            )
    if mismatches:
        logger.warning(
            "[%s] exact text/token mismatches=%d/%d",
            name,
            mismatches,
            total,
        )
        logger.warning("[%s] mismatch indices: %s", name, mismatch_indices)
    else:
        logger.info(
            "[%s] exact text/token correctness OK (%d requests)",
            name,
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
        "mismatch_details": mismatch_details,
        "prompt_alignment_mismatches": prompt_alignment_mismatches,
    }


def _generated_token_ids(output: Any) -> list[int]:
    """Return generated token IDs from a vLLM RequestOutput.

    Args:
        output: vLLM request output.

    Returns:
        Generated token IDs, or an empty list when unavailable.
    """
    if not getattr(output, "outputs", None):
        return []
    return [int(token_id) for token_id in getattr(output.outputs[0], "token_ids", [])]


def _output_text(output: Any) -> str:
    """Return generated text from a vLLM RequestOutput.

    Args:
        output: vLLM request output.

    Returns:
        Generated text, or an empty string when unavailable.
    """
    if not getattr(output, "outputs", None):
        return ""
    return str(getattr(output.outputs[0], "text", ""))


def correctness_check_with_visibility(
    name: str,
    cold_outputs: list,
    warm_outputs: list,
    prompts: list[list[int]],
    max_num_seqs: int,
    visible_mask: list[bool],
) -> dict[str, Any]:
    """Compare cold/warm outputs and split exact mismatches by visible hits.

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
    for cold, warm, visible in zip(
        cold_outputs, warm_outputs, visible_mask, strict=False
    ):
        if not visible:
            continue
        visible_total += 1
        if _generated_token_ids(cold) == _generated_token_ids(warm) and _output_text(
            cold
        ) == _output_text(warm):
            continue
        visible_mismatches += 1
    result["visible_mismatches"] = visible_mismatches
    result["visible_total"] = visible_total
    if visible_total:
        logger.info(
            "[%s] visible-hit correctness: mismatches=%d (%d requests)",
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


def _fmt_count(v: Any) -> str:
    """Format an integer counter metric.

    Args:
        v: Numeric counter value or None.

    Returns:
        Human-readable counter string.
    """
    if v is None:
        return "N/A"
    return str(v)


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
        daser_correctness = d.get("correctness") or {}
        lmcache_correctness = lm.get("correctness") or {}
        daser_mismatches = daser_correctness.get("mismatches")
        lmcache_mismatches = lmcache_correctness.get("mismatches")
        if daser_mismatches is not None and lmcache_mismatches is not None:
            delta = int(daser_mismatches) - int(lmcache_mismatches)
            summary["correctness_mismatch_delta_daser_minus_lmcache"] = delta
            summary["correctness_parity_ok"] = delta <= 1
    return summary


def print_report(config: dict[str, Any], summary: dict[str, Any]) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 72)
    print("E2E vLLM Benchmark — DaseR vs LMCache")
    print("=" * 72)
    print(f"Model            : {config['model']}")
    print(f"Comparison mode  : {config['comparison_mode']}")
    print(f"Prompts          : {config['num_prompts']} (IMDB reviews)")
    print(f"Seed             : {config['seed']}")
    print(f"Prompt tokens    : {summary['prompt_tokens_total']:,}")
    print("Sampling         : temperature=0, max_tokens=1")
    print("Correctness src  : exact generated token IDs and output text")
    print("Correctness      : cold/warm outputs must match exactly")
    print("Correctness rule : DaseR mismatches <= LMCache mismatches + 1")
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

    def _correctness_value(system: dict[str, Any], key: str) -> Any:
        correctness = system.get("correctness") or {}
        return correctness.get(key)

    print(
        f"{'exact mismatches':<28}"
        f"{_fmt_count(_correctness_value(d, 'mismatches')):>20}"
        f"{_fmt_count(_correctness_value(lm, 'mismatches')):>20}"
    )
    parity = summary.get("correctness_parity_ok")
    if parity is not None:
        delta = summary.get("correctness_mismatch_delta_daser_minus_lmcache")
        print(f"{'mismatch delta':<28}{_fmt_count(delta):>20}{'limit <= 1':>20}")
        print(f"{'correctness parity':<28}{str(bool(parity)):>20}{'':>20}")

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
    set_global_seed(BENCHMARK_SEED)
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
        help="vLLM gpu_memory_utilization (default: 0.9)",
    )
    parser.add_argument(
        "--gpu-id",
        default="auto",
        help=(
            "GPU ID to expose through CUDA_VISIBLE_DEVICES. Use 'auto' to pick "
            "the GPU with most free memory, or 'current' to keep the current env."
        ),
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
    store_root = os.path.join(args.store_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(store_root, exist_ok=False)
    logger.info("benchmark scratch root: %s", store_root)
    logger.info(
        "selected GPU: %s (CUDA_VISIBLE_DEVICES=%s)",
        SELECTED_GPU_ID if SELECTED_GPU_ID is not None else "current",
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

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
    capacity_limits = derive_capacity_limits(store_root, SELECTED_GPU_ID)
    sizing = derive_benchmark_sizing(
        total_blocks=total_blocks,
        max_prompt_blocks=max_prompt_blocks,
        slot_size=SLOT_SIZE,
        mode=args.comparison_mode,
        evict=args.evict,
        capacity_limits=capacity_limits,
    )
    transfer_mode = (
        "iouring" if args.comparison_mode == COMPARISON_IOURING_MEM else "gds"
    )
    logger.info(
        "cache sizing: workload=%.2fGiB, daser_l2_slots=%d, "
        "daser_l1=%.2fGiB, evict=%s, capped=%s, "
        "max_l1=%.2fGiB, max_l2=%.2fGiB",
        total_bytes / BYTES_PER_GIB,
        sizing.daser_slots,
        sizing.daser_l1_bytes / BYTES_PER_GIB,
        args.evict,
        sizing.capacity_capped,
        capacity_limits.max_l1_bytes / BYTES_PER_GIB,
        capacity_limits.max_l2_bytes / BYTES_PER_GIB,
    )

    config = {
        "comparison_mode": args.comparison_mode,
        "evict": args.evict,
        "num_prompts": len(prompts),
        "correctness_num_prompts": len(prompts),
        "seed": BENCHMARK_SEED,
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
        "daser_l2_bytes": sizing.daser_l2_bytes,
        "daser_l1_bytes": sizing.daser_l1_bytes,
        "lmcache_disk_gb": sizing.lmcache_disk_gb,
        "lmcache_cpu_gb": sizing.lmcache_cpu_gb,
        "selected_gpu_id": SELECTED_GPU_ID,
        "gpu_util": args.gpu_util,
        "capacity_limits": {
            "max_l1_bytes": capacity_limits.max_l1_bytes,
            "max_l2_bytes": capacity_limits.max_l2_bytes,
            "memory_available_bytes": capacity_limits.memory_available_bytes,
            "disk_available_bytes": capacity_limits.disk_available_bytes,
            "capacity_capped": sizing.capacity_capped,
        },
        "daser_warm_skip_save": True,
        "correctness_metric": "exact_generated_token_ids_and_text",
    }
    correctness_prompts = prompts

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
            lmcache_dir = tempfile.mkdtemp(prefix="lmcache_bench_", dir=store_root)
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
                r = run_system(
                    "LMCache",
                    h_lm.build_llm,
                    prompts,
                    after_cold_fn=h_lm.wait_for_disk_quiescence,
                )
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
            if lmcache_result is not None:
                lmcache_result["correctness"] = run_lmcache_correctness(
                    store_root,
                    total_bytes,
                    args.model,
                    args.gpu_util,
                    max_num_seqs,
                    args.comparison_mode == COMPARISON_IOURING_MEM,
                    sizing.lmcache_disk_gb,
                    sizing.lmcache_cpu_gb,
                    correctness_prompts,
                )

    # ---- DaseR run ----
    daser_result: dict[str, Any] | None = None
    if args.skip_daser:
        daser_result = {"skipped": True, "reason": "--skip-daser"}
    else:
        daser_dir = tempfile.mkdtemp(prefix="daser_bench_", dir=store_root)
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
            r["backend"] = transfer_mode
            r["storage_tier"] = (
                "local-ssd-mem"
                if args.comparison_mode == COMPARISON_IOURING_MEM
                else "local-ssd"
            )
            r["warm_skip_save"] = True
            r["l2_bytes"] = sizing.daser_l2_bytes
            r["l1_bytes"] = sizing.daser_l1_bytes
            daser_result = r
        finally:
            h.stop()
        if daser_result is not None:
            correctness_require_l2_drain = (
                args.evict or args.comparison_mode == COMPARISON_IOURING_MEM
            )
            daser_result["correctness"] = run_daser_correctness(
                store_root,
                args.model,
                args.gpu_util,
                max_num_seqs,
                transfer_mode,
                sizing.daser_l1_bytes,
                sizing.daser_slots,
                correctness_prompts,
                require_all_commits=not args.evict,
                require_l2_drain=correctness_require_l2_drain,
            )
            daser_result["visible_prompt_count"] = int(
                daser_result["correctness"].get("visible_total", 0)
            )

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
