# SPDX-License-Identifier: Apache-2.0
"""Service lifecycle and manifest helpers for benchmark runs."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import httpx

from benchmarks.utils.constants import BLOCK_TOKENS
from benchmarks.utils.sizing import (
    LMCACHE_EVICTION_TRIGGER_WATERMARK,
    bytes_to_lmcache_gb,
    bytes_to_lmcache_gb_for_effective_l1,
)

LMCACHE_MP_HOST = "tcp://localhost"
LMCACHE_MP_PORT = 5555
LMCACHE_HTTP_PORT = 8080
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServiceEndpoint:
    """HTTP endpoint for a benchmark service.

    Args:
        url: Base URL.

    Thread-safety:
        Immutable value object.
    """

    url: str


@dataclass(frozen=True)
class BenchmarkManifest:
    """Service run manifest shared between start and load scripts.

    Args:
        run_id: Run identifier.
        backend: Backend name.
        reuse_mode: DaseR reuse mode, or ``none``.
        model: Model path.
        store_dir: Scratch directory.
        l1_size_bytes: L1 size in bytes.
        l2_size_bytes: L2 size in bytes.
        skip_l2: Whether the service should run with no L2 tier.
        endpoints: Named service endpoints.
        log_dir: Log directory.
        pid_file: JSON file containing subprocess PIDs.
        block_size: vLLM KV block size in tokens.

    Thread-safety:
        Immutable value object.
    """

    run_id: str
    backend: str
    reuse_mode: str
    model: str
    store_dir: str
    l1_size_bytes: int
    l2_size_bytes: int
    skip_l2: bool
    endpoints: dict[str, ServiceEndpoint]
    log_dir: str
    pid_file: str
    block_size: int = BLOCK_TOKENS

    def write(self, path: str | Path) -> None:
        """Write manifest JSON atomically enough for local benchmark use."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "BenchmarkManifest":
        """Read a manifest JSON file.

        Args:
            path: Manifest path.

        Returns:
            BenchmarkManifest instance.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        endpoints = {
            name: ServiceEndpoint(**endpoint)
            for name, endpoint in payload["endpoints"].items()
        }
        payload["endpoints"] = endpoints
        payload.setdefault("block_size", BLOCK_TOKENS)
        return cls(**payload)


class ServerManager:
    """Start and stop benchmark service subprocesses."""

    def __init__(
        self,
        run_id: str,
        backend: str,
        model: str,
        store_dir: str | Path,
        gpu_id: str,
        gpu_util: float,
        max_num_seqs: int,
        l1_size_bytes: int,
        l2_size_bytes: int,
        max_num_batched_tokens: int | None = None,
        block_size: int = BLOCK_TOKENS,
        reuse_mode: str = "chunk",
        transfer_mode: str = "iouring",
        vllm_port: int = 8001,
        daser_port: int = 2026,
        startup_timeout: float = 240.0,
        max_model_len: int | None = None,
        skip_l2: bool = False,
    ) -> None:
        """Initialize the service manager.

        Args:
            run_id: Run identifier.
            backend: Backend name.
            model: Model path.
            store_dir: Scratch directory.
            gpu_id: GPU ID exposed through CUDA_VISIBLE_DEVICES.
            gpu_util: vLLM GPU memory utilization.
            max_num_seqs: vLLM max_num_seqs.
            max_num_batched_tokens: Optional vLLM scheduler token budget.
            block_size: vLLM KV block size in tokens.
            l1_size_bytes: L1 size.
            l2_size_bytes: L2 size.
            reuse_mode: DaseR cache reuse mode.
            transfer_mode: DaseR transfer backend.
            vllm_port: vLLM HTTP port.
            daser_port: DaseR HTTP port.
            startup_timeout: Health-check timeout.
            max_model_len: Optional vLLM max model length.
            skip_l2: Disable L2 persistence/adapters for L1-only no-evict runs.
        """
        self.run_id = run_id
        self.backend = backend
        self.model = model
        self.store_dir = Path(store_dir)
        self.gpu_id = gpu_id
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.l1_size_bytes = l1_size_bytes
        self.l2_size_bytes = l2_size_bytes
        self.reuse_mode = reuse_mode
        self.transfer_mode = transfer_mode
        self.vllm_port = vllm_port
        self.daser_port = daser_port
        self.startup_timeout = startup_timeout
        self.max_model_len = max_model_len
        self.skip_l2 = skip_l2
        self.log_dir = self.store_dir / "logs"
        self.pid_file = self.store_dir / "pids.json"
        self.socket_path = self.store_dir / "daser.sock"
        self._procs: list[subprocess.Popen[bytes]] = []

    @property
    def vllm_url(self) -> str:
        """Return the vLLM base URL."""
        return f"http://127.0.0.1:{self.vllm_port}"

    @property
    def daser_url(self) -> str:
        """Return the DaseR base URL."""
        return f"http://127.0.0.1:{self.daser_port}"

    async def start(self) -> BenchmarkManifest:
        """Start services for the configured backend and return a manifest."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_ports_available()
        if self.backend == "vllm":
            await self.start_vllm_only()
        elif self.backend == "lmcache":
            await self.start_lmcache_mp_server()
            await self.start_vllm_lmcache()
        elif self.backend == "daser":
            await self.start_vllm_daser()
            await self.start_daser_server()
        else:
            raise ValueError(f"unsupported backend: {self.backend}")
        self._write_pids()
        manifest = self.manifest()
        manifest.write(self.store_dir / "manifest.json")
        return manifest

    def manifest(self) -> BenchmarkManifest:
        """Return the current service manifest."""
        endpoints = {"vllm": ServiceEndpoint(self.vllm_url)}
        if self.backend == "daser":
            endpoints["daser"] = ServiceEndpoint(self.daser_url)
        return BenchmarkManifest(
            run_id=self.run_id,
            backend=self.backend,
            reuse_mode=self.reuse_mode if self.backend == "daser" else "none",
            model=self.model,
            store_dir=str(self.store_dir),
            l1_size_bytes=self.l1_size_bytes,
            l2_size_bytes=self.l2_size_bytes,
            skip_l2=self.skip_l2,
            endpoints=endpoints,
            log_dir=str(self.log_dir),
            pid_file=str(self.pid_file),
            block_size=self.block_size,
        )

    async def start_lmcache_mp_server(self) -> None:
        """Start the LMCache MP server."""
        cmd = self._lmcache_mp_server_command()
        proc = self._start(cmd, "lmcache_mp_server.log")
        await self._wait_healthy(
            f"http://127.0.0.1:{LMCACHE_HTTP_PORT}",
            "/healthcheck",
            self.startup_timeout,
            proc,
        )

    def _lmcache_mp_server_command(self) -> list[str]:
        """Build the LMCache MP server command.

        Returns:
            Command argv. When ``skip_l2`` is true, no ``--l2-adapter`` is
            emitted, so LMCache keeps data in its L1 manager only.

        Thread-safety:
            Pure helper except for creating the L2 scratch directory when the
            adapter is enabled.
        """
        l1_gb = (
            bytes_to_lmcache_gb(self.l1_size_bytes)
            if self.skip_l2
            else bytes_to_lmcache_gb_for_effective_l1(self.l1_size_bytes)
        )
        cmd = [
            "lmcache",
            "server",
            "--host",
            "localhost",
            "--port",
            str(LMCACHE_MP_PORT),
            "--chunk-size",
            str(self.block_size),
            "--max-workers",
            "4",
            "--l1-size-gb",
            str(l1_gb),
            "--eviction-policy",
            "LRU",
            "--eviction-trigger-watermark",
            str(LMCACHE_EVICTION_TRIGGER_WATERMARK),
            "--http-port",
            str(LMCACHE_HTTP_PORT),
        ]
        if not self.skip_l2:
            scratch = self.store_dir / "lmcache_mp_disk"
            scratch.mkdir(parents=True, exist_ok=True)
            cmd.extend(
                [
                    "--l2-adapter",
                    json.dumps({"type": "fs", "base_path": str(scratch)}),
                ]
            )
        return cmd

    async def start_vllm_only(self) -> None:
        """Start vanilla vLLM."""
        await self._start_vllm("vllm_vanilla.log", None)

    async def start_vllm_lmcache(self) -> None:
        """Start vLLM with LMCache MP connector."""
        kv_config = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": LMCACHE_MP_HOST,
                "lmcache.mp.port": LMCACHE_MP_PORT,
            },
        }
        await self._start_vllm(
            "vllm_lmcache.log",
            kv_config,
            extra_env={"PYTHONHASHSEED": "42"},
        )

    async def start_vllm_daser(self) -> None:
        """Start vLLM with DaseR connector."""
        kv_config = {
            "kv_connector": "DaserConnector",
            "kv_connector_module_path": "daser.connector.daser_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "socket_path": str(self.socket_path),
                "cache_reuse_mode": self.reuse_mode,
            },
        }
        await self._start_vllm("vllm_daser.log", kv_config)

    async def start_daser_server(self) -> None:
        """Start DaseR HTTP + IPC server."""
        cmd = self._daser_server_command()
        proc = self._start(cmd, "daser.log")
        await self._wait_healthy(self.daser_url, "/health", self.startup_timeout, proc)

    def _daser_server_command(self) -> list[str]:
        """Build the DaseR server command.

        Returns:
            Command argv. When ``skip_l2`` is true, no ``--l2-size`` is
            emitted and DaseR derives its L1-only logical slot capacity from
            ``--l1-size``.

        Thread-safety:
            Not thread-safe with respect to concurrent store directory cleanup.
        """
        store = self.store_dir / "daser"
        store.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "daser.server",
            "--vllm-base-url",
            self.vllm_url,
            "--model-path",
            self.model,
            "--store-dir",
            str(store),
            "--l1-size",
            str(self.l1_size_bytes),
            "--transfer-mode",
            self.transfer_mode,
            "--cache-reuse-mode",
            self.reuse_mode,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.daser_port),
            "--socket-path",
            str(self.socket_path),
            "--block-tokens",
            str(self.block_size),
        ]
        if self.skip_l2:
            cmd.append("--skip-l2")
        else:
            cmd.extend(["--l2-size", str(self.l2_size_bytes)])
        return cmd

    async def stop_all(self) -> None:
        """Terminate all child processes."""
        for proc in reversed(self._procs):
            if proc.poll() is not None:
                continue
            proc.terminate()
        deadline = time.monotonic() + 15.0
        for proc in self._procs:
            if proc.poll() is not None:
                continue
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
        self._procs.clear()

    async def _start_vllm(
        self,
        log_name: str,
        kv_transfer_config: dict[str, Any] | None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        proc = self._start(
            self.vllm_command(kv_transfer_config),
            log_name,
            extra_env=extra_env,
        )
        await self._wait_healthy(self.vllm_url, "/health", self.startup_timeout, proc)

    def vllm_command(
        self,
        kv_transfer_config: dict[str, Any] | None,
    ) -> list[str]:
        """Build the vLLM serve command for benchmark services.

        Args:
            kv_transfer_config: Optional KV transfer configuration.

        Returns:
            Command list suitable for subprocess.Popen.

        Thread-safety:
            Pure calculation over immutable instance configuration.
        """
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.vllm_port),
            "--gpu-memory-utilization",
            str(self.gpu_util),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--no-enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--block-size",
            str(self.block_size),
        ]
        if self.max_model_len is not None and self.max_model_len > 0:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens > 0:
            cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if kv_transfer_config is not None:
            cmd.extend(["--kv-transfer-config", json.dumps(kv_transfer_config)])
        return cmd

    def _start(
        self,
        cmd: list[str],
        log_name: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / log_name
        fh = log_path.open("wb")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(REPO_ROOT)
        )
        if extra_env:
            env.update(extra_env)
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._procs.append(proc)
        return proc

    async def _wait_healthy(
        self,
        base_url: str,
        path: str,
        timeout: float,
        proc: subprocess.Popen[bytes],
    ) -> None:
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{base_url}{path} startup process exited with "
                        f"code {proc.returncode}"
                    )
                try:
                    response = await client.get(f"{base_url}{path}", timeout=5.0)
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        raise RuntimeError(f"{base_url}{path} not healthy after {timeout:.0f}s")

    def _write_pids(self) -> None:
        payload = [
            {
                "pid": proc.pid,
                "cmd": proc.args,
                "cuda_visible_devices": str(self.gpu_id),
            }
            for proc in self._procs
        ]
        self.pid_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _ensure_ports_available(self) -> None:
        ports = [self.vllm_port]
        if self.backend == "lmcache":
            ports.extend([LMCACHE_MP_PORT, LMCACHE_HTTP_PORT])
        if self.backend == "daser":
            ports.append(self.daser_port)
        busy = [port for port in ports if _is_port_open("127.0.0.1", port)]
        if busy:
            joined = ", ".join(str(port) for port in busy)
            raise RuntimeError(f"benchmark service port(s) already in use: {joined}")


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def stop_from_pid_file(path: str | Path) -> None:
    """Terminate processes listed in a benchmark pid file.

    Args:
        path: PID JSON path.

    Thread-safety:
        Sends process signals and should be called by one owner.
    """
    pid_path = Path(path)
    if not pid_path.is_file():
        return
    payload = json.loads(pid_path.read_text(encoding="utf-8"))
    for item in reversed(payload):
        pid = int(item["pid"])
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15.0
    for item in reversed(payload):
        pid = int(item["pid"])
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
