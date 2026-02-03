from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class LMCacheGDSConfig:
    """Minimal LMCache config for using GDS backend.

    LMCache v1 adapter in vLLM reads configuration from `LMCACHE_CONFIG_FILE`.
    We generate a small YAML config and point the env var to it.

    Notes:
      - Even if GPUDirect Storage is not available, LMCache may fall back to
        POSIX I/O internally.
      - `gds_path` should be a directory. LMCache will create subdirs/files.
    """

    gds_path: str
    chunk_size: int = 256
    # IMPORTANT: LMCache interprets this value as MiB (and internally multiplies
    # by 1024**2 to convert to bytes). Do NOT pass bytes here.
    cufile_buffer_size: int | None = None

    # NOTE:
    # - `local_cpu` controls whether LMCache keeps a CPU *hot cache* (in-memory).
    # - Even when `local_cpu` is False, LMCache still uses a pinned-CPU staging
    #   buffer (sized by max_local_cpu_size) to move KV off GPU before it can be
    #   written to disk.
    local_cpu: bool = False
    max_local_cpu_size: float = 2.0

    # Extra toggles passed to LMCache as a dict. Keep None by default.
    extra_config: dict | None = None


def write_lmcache_config_file(path: str | Path, cfg: LMCacheGDSConfig) -> Path:
    """Write an LMCache YAML config file and return its absolute path."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # We avoid a YAML dependency by emitting a small YAML subset.
    lines: list[str] = []
    lines.append(f"chunk_size: {int(cfg.chunk_size)}")
    # IMPORTANT: LMCache default hashing is "builtin" (Python hash), which is
    # randomized per-process unless PYTHONHASHSEED is fixed. Using a stable
    # vLLM hash avoids cross-process 0-hit lookups.
    lines.append("pre_caching_hash_algorithm: sha256_cbor")
    # For prefix reuse across different suffixes (warmup: text, inference:
    # text+task), saving partial chunks is usually counterproductive:
    # a partial chunk key depends on the exact end position, so even a small
    # suffix change makes the key different and lookup returns 0.
    # Disabling it makes reuse happen at full chunk boundaries.
    lines.append("save_unfull_chunk: false")
    lines.append(f"local_cpu: {'true' if cfg.local_cpu else 'false'}")
    lines.append(f"max_local_cpu_size: {float(cfg.max_local_cpu_size)}")
    gds_path = str(Path(cfg.gds_path).expanduser().resolve())
    # LMCache parses `file://.../` into a filesystem path.
    lines.append(f"local_disk: file://{gds_path}/")
    lines.append("max_local_disk_size: 100")
    lines.append("remote_url: null")
    lines.append("enable_pd: false")
    lines.append("enable_p2p: false")
    # NOTE: Use either local_disk or gds_path depending on your backend.
    # lines.append(f"gds_path: {cfg.gds_path}")
    if cfg.cufile_buffer_size is not None:
        # LMCache expects MiB here; passing bytes will explode to PB-scale.
        if int(cfg.cufile_buffer_size) > 1_000_000:
            raise ValueError(
                "LMCache 'cufile_buffer_size' must be specified in MiB. "
                "The provided value is suspiciously large; did you pass bytes?"
            )
        lines.append(f"cufile_buffer_size: {int(cfg.cufile_buffer_size)}")

    if cfg.extra_config:
        lines.append("extra_config:")
        for k, v in cfg.extra_config.items():
            # Minimal YAML literal formatting.
            if isinstance(v, bool):
                vv = "true" if v else "false"
            elif v is None:
                vv = "null"
            elif isinstance(v, (int, float)):
                vv = str(v)
            else:
                escaped = str(v).replace('"', '\\"')
                vv = f'"{escaped}"'
            lines.append(f"  {k}: {vv}")
    else:
        lines.append("extra_config: null")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def configure_lmcache_env(config_file: str | Path) -> None:
    """Point vLLM's LMCache adapter to the given config file."""
    os.environ["LMCACHE_CONFIG_FILE"] = str(Path(config_file).expanduser().resolve())


@dataclass(frozen=True)
class WarmupConfig:
    # Generate at least 1 token so vLLM executes a full forward.
    max_tokens: int = 1
    temperature: float = 0.0
    top_p: float = 1.0


class LMCacheGDSKVCacheManager:
    """High-level helper for LMCache(GDS) KV caching with vLLM LMCacheConnectorV1.

    Provides two workflows:
      1) KV generation workflow: prefill `text` prompts to populate LMCache
         and store KV blocks to disk (GDS backend path).
      2) Online inference workflow: run `text + task` prompts; LMCacheConnector
         will prefetch KV to GPU for cached prefixes.
    """

    def __init__(
        self,
        *,
        model: str,
        lmcache_cfg: LMCacheGDSConfig,
        lmcache_cfg_file: str | Path,
        enforce_eager: bool = True,
        use_native_adapter: bool = False,
        kv_role: str = "kv_both",
        kv_connector_extra_config: dict[str, Any] | None = None,
        **llm_kwargs,
    ) -> None:
        # Ensure cache directory exists.
        Path(lmcache_cfg.gds_path).expanduser().resolve().mkdir(parents=True, exist_ok=True)

        cfg_path = write_lmcache_config_file(lmcache_cfg_file, lmcache_cfg)
        configure_lmcache_env(cfg_path)

        kv_transfer_config = {
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": str(kv_role),
            "kv_connector_extra_config": {
                # vLLM's LMCacheConnectorV1 reads this.
                "use_native": bool(use_native_adapter),
            },
        }

        if kv_connector_extra_config:
            kv_transfer_config["kv_connector_extra_config"].update(kv_connector_extra_config)

        from vllm import LLM  # type: ignore

        self._llm = LLM(
            model=model,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            kv_transfer_config=kv_transfer_config,
            **llm_kwargs,
        )

    def _shutdown_llm(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass

    @property
    def llm(self) -> Any:
        return self._llm

    def warmup_text_prefixes(
        self,
        texts: Sequence[str],
        *,
        warmup: WarmupConfig | None = None,
    ) -> None:
        """KV generation workflow: compute + persist KV for text-only prefixes."""
        if not texts:
            return
        warmup = warmup or WarmupConfig()
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            max_tokens=warmup.max_tokens,
            temperature=warmup.temperature,
            top_p=warmup.top_p,
        )

        # IMPORTANT:
        # LMCache's disk persistence path uses a pinned CPU staging buffer.
        # Warming up with a large list of prompts makes vLLM schedule many
        # concurrent stores/puts, which can exhaust the staging pool and cause
        # LMCache to either drop stores or busy-wait with repeated warnings.
        # Running warmup sequentially dramatically reduces peak staging usage
        # and makes disk persistence reliable.
        for t in texts:
            if not t:
                continue
            _ = self._llm.generate([t], params)

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: Any,
    ):
        """Online inference workflow: run batched generation (with KV prefetch)."""
        return self._llm.generate(list(prompts), sampling_params)

    def close(self) -> None:
        self._shutdown_llm()
        return

    @staticmethod
    def build_text_task_prompts(
        texts: Sequence[str],
        task: str,
        *,
        separator: str = "\n\n",
    ) -> list[str]:
        return [f"{t}{separator}{task}" for t in texts]
