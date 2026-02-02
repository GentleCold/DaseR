from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from typing import Any


@dataclass(frozen=True)
class WarmupConfig:
    # Generate at least 1 token so vLLM executes a full forward.
    max_tokens: int = 1
    temperature: float = 0.0
    top_p: float = 1.0


class SSDKVCacheManager:
    """High-level helper for 2-phase prefix KV caching on SSD.

    Intended workflow for CSV-like workloads:
      1) Warmup stage: prompt=text; run a tiny generation to force prefill;
         OffloadingConnector stores full KV blocks for the text prefix to SSD.
      2) Task stage: prompt=text+task; connector loads cached prefix blocks
         and vLLM continues prefill from the first cache-miss token.

    Notes / constraints:
      - Requires vLLM v1 KV connector path and uses OffloadingConnector.
      - Uses daser.kvcache.ssd_offloading_spec.SSDOffloadingSpec via
        kv_transfer_config.kv_connector_extra_config.
      - For cross-process / restart hits, you should fix PYTHONHASHSEED.
    """

    def __init__(
        self,
        *,
        model: str,
        ssd_cache_dir: str | Path,
        max_blocks: int = 100_000,
        enforce_eager: bool = True,
        pythonhashseed: str | None = "0",
        **llm_kwargs,
    ) -> None:
        from vllm import LLM  # type: ignore

        if pythonhashseed is not None and "PYTHONHASHSEED" not in os.environ:
            # Must be set before block hashing is initialized to be fully effective.
            os.environ["PYTHONHASHSEED"] = str(pythonhashseed)

        cache_dir = Path(ssd_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        kv_transfer_config = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "SSDOffloadingSpec",
                "spec_module_path": "daser.kvcache.ssd_offloading_spec",
                "ssd_cache_dir": str(cache_dir),
                "max_blocks": int(max_blocks),
            },
        }

        self._llm = LLM(
            model,
            enforce_eager=enforce_eager,
            kv_transfer_config=kv_transfer_config,
            **llm_kwargs,
        )

    @property
    def llm(self) -> Any:
        return self._llm

    def warmup_text_prefixes(
        self,
        texts: Sequence[str],
        *,
        warmup: WarmupConfig | None = None,
    ) -> None:
        """Compute and persist KV blocks for text-only prefixes."""
        if not texts:
            return
        warmup = warmup or WarmupConfig()
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            max_tokens=warmup.max_tokens,
            temperature=warmup.temperature,
            top_p=warmup.top_p,
        )

        _ = self._llm.generate(list(texts), params)

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: Any,
    ):
        return self._llm.generate(list(prompts), sampling_params)

    def build_text_task_prompts(
        self,
        texts: Sequence[str],
        task: str,
        *,
        separator: str = "\n\n",
    ) -> list[str]:
        return [f"{t}{separator}{task}" for t in texts]

    def close(self) -> None:
        # vLLM cleans up workers on process exit; this is a placeholder for symmetry.
        return
