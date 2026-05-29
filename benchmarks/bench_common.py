# SPDX-License-Identifier: Apache-2.0
"""Shared benchmark harnesses, runners, and reporters for DaseR vs LMCache.

Import this module from individual benchmark scripts (IMDB, LongBench, etc.).
"""

# Future
from __future__ import annotations

# Standard
import asyncio
import gc
import hashlib
import os
import tempfile
import threading
import time
from typing import Any

# Third Party
import torch

from daser.connector.helpers import hash_tokens
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Second pass: CUDA driver may release IPC memory after first
        # synchronize, so collect again for LMCache→DaseR handover.
        gc.collect()
        torch.cuda.empty_cache()


def wait_gpu_memory(
    gpu_util: float,
    timeout_s: float = 60.0,
    poll_s: float = 1.0,
) -> None:
    """Block until the selected GPU has enough free memory for a new LLM.

    vLLM V1 EngineCore subprocesses may hold GPU memory briefly after
    shutdown.  This polls ``torch.cuda.mem_get_info`` until free memory
    reaches ``total * gpu_util``.

    Args:
        gpu_util: vLLM ``gpu_memory_utilization`` for the next LLM.
        timeout_s: Maximum wait time in seconds.
        poll_s: Interval between polls in seconds.

    Raises:
        RuntimeError: If free memory never reaches the required threshold
            within *timeout_s*.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        free, total = torch.cuda.mem_get_info()
        free_gib = free / (1024**3)
        total_gib = total / (1024**3)
        needed = total_gib * gpu_util
        if free_gib >= needed:
            logger.info(
                "[GPU] %.2f GiB free >= %.2f GiB needed — proceeding",
                free_gib,
                needed,
            )
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU memory not freed after {timeout_s:.0f}s: "
                f"{free_gib:.2f}/{total_gib:.2f} GiB free, "
                f"need {needed:.2f} GiB"
            )
        logger.debug(
            "[GPU] waiting for memory: %.2f/%.2f GiB free, need %.2f GiB",
            free_gib,
            total_gib,
            needed,
        )
        time.sleep(poll_s)


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
        max_model_len: int = MAX_MODEL_LEN,
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
            max_model_len: vLLM ``max_model_len`` (default: 2048).
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
        self.max_model_len = max_model_len
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
        from benchmarks.utils import BYTES_PER_GIB

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
            max_model_len=self.max_model_len,
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
        model_id = "qwen3-8b"

        # ---- compute unique chunk keys (diagnostic only) ----
        chunk_keys: set[str] = set()
        for tokens in prompts:
            aligned = (len(tokens) // block_tokens) * block_tokens
            if aligned > 0:
                chunk_keys.add(hash_tokens(tokens[:aligned]))
        if len(chunk_keys) < len(prompts):
            logger.info(
                "[DaseR] unique aligned chunks: %d/%d prompts (%d duplicate prefixes)",
                len(chunk_keys),
                len(prompts),
                len(prompts) - len(chunk_keys),
            )

        # ---- build prompt prefix list for visibility check ----
        prompt_prefixes: list[list[int]] = []
        for tokens in prompts:
            aligned = (len(tokens) // block_tokens) * block_tokens
            prompt_prefixes.append(tokens[:aligned] if aligned > 0 else [])

        try:
            while True:
                if not require_all_commits:
                    stats = client.commit_stats()
                    committed = int(stats.get("commit_requests", 0))
                    if committed > 0:
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
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "timed out waiting for any DaseR commit request"
                        )
                    time.sleep(0.05)
                    continue

                # require_all_commits: check every prompt prefix is visible.
                # We cannot rely on commit_requests count because intra-run
                # cache hits mean some unique chunk keys are never committed
                # (an earlier request with a matching shorter prefix already
                # committed that chunk).
                missing = 0
                for prefix in prompt_prefixes:
                    if not prefix:
                        continue
                    try:
                        chunks = client.lookup(prefix, model_id)
                    except Exception:
                        chunks = []
                    if not chunks:
                        missing += 1
                if missing == 0:
                    if require_l2_drain:
                        client.transfer_drain()
                    stats = client.commit_stats()
                    logger.info(
                        "[DaseR] all %d prompts visible (commits=%d, lookups=%d/%d)",
                        len(prompts),
                        int(stats.get("commit_requests", 0)),
                        int(stats.get("lookup_hits", 0)),
                        int(stats.get("lookup_requests", 0)),
                    )
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for DaseR commits "
                        f"({len(prompts) - missing}/{len(prompts)} prompts visible)"
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
        max_model_len: int = MAX_MODEL_LEN,
    ) -> None:
        """Initialise paths.

        Args:
            tmpdir: Directory used as LMCache's local_disk.
            total_bytes: Expected bytes-on-disk (drives max_local_disk_size).
            model_path: HF model path for vLLM.
            gpu_util: vLLM ``gpu_memory_utilization``.
            max_model_len: vLLM ``max_model_len`` (default: 2048).
        """
        self.tmpdir = tmpdir
        self.model_path = model_path
        self.total_bytes = total_bytes
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.local_cpu = local_cpu
        self.disk_limit_gb = disk_limit_gb
        self.cpu_limit_gb = cpu_limit_gb
        self.max_model_len = max_model_len
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
            "LMCACHE_LMCACHE_INSTANCE_ID": self.instance_id,
            # The BENCHMARK_SEED_ENV value ("42") is imported by callers.
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "42"),
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
            max_model_len=self.max_model_len,
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
        from pathlib import Path

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
        for k, saved in self._saved_env.items():
            if saved is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved
        self._saved_env.clear()
        logger.info("[LMCache] env restored")

    @staticmethod
    def _disk_snapshot(root: Any) -> tuple[int, int]:
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


# ---------------------------------------------------------------------------
# Convenience: harness + correctness in one call
# ---------------------------------------------------------------------------


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
    max_model_len: int = MAX_MODEL_LEN,
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
        max_model_len: vLLM ``max_model_len`` (default: 2048).

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
        max_model_len=max_model_len,
    )
    try:
        h_lm.start()
        wait_gpu_memory(gpu_util)
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
    max_model_len: int = MAX_MODEL_LEN,
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
        max_model_len: vLLM ``max_model_len`` (default: 2048).

    Returns:
        Exact correctness result dictionary with visible-hit counters.
    """
    from vllm import SamplingParams  # Third Party
    from vllm.inputs import TokensPrompt  # Third Party

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
        max_model_len=max_model_len,
    )
    try:
        h.start()
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
        wait_gpu_memory(gpu_util)
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


# ---------------------------------------------------------------------------
# Correctness checkers
# ---------------------------------------------------------------------------


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
    label = config.get("dataset", "IMDB reviews")
    print(f"Prompts          : {config['num_prompts']} ({label})")
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
