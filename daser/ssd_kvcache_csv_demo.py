"""Demo: 2-phase SSD KV cache reuse for CSV text+task workloads.

This file intentionally avoids args/env-vars.
Edit the constants below to configure the run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from vllm import SamplingParams

from daser.kvcache import SSDKVCacheManager


# =========================
# Simple config (edit here)
# =========================
DATA_CSV = "/data/zwt/imdb.csv"  # must have a 'review' column
LIMIT = 100
MODEL = "/data/zwt/model/models/Qwen/Qwen3-8B"
CACHE_DIR = f"/data/zwt/vllm_kv_ssd"
MAX_BLOCKS = 200_000
BATCH_SIZE = 16


TASK = (
    "You are a sentiment classifier.\n"
    "Given the above movie review text, output exactly one word: positive or negative."
)


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    if "review" not in df.columns:
        raise ValueError(f"Expected a 'review' column, got {list(df.columns)}")

    texts = df["review"].astype(str).tolist()[:LIMIT]

    mgr = SSDKVCacheManager(
        model=MODEL,
        ssd_cache_dir=Path(CACHE_DIR),
        max_blocks=MAX_BLOCKS,
        enforce_eager=True,
        # You should also export PYTHONHASHSEED in the shell for full safety.
        pythonhashseed="0",
    )

    print(f"Warming up text prefixes: n={len(texts)}")
    mgr.warmup_text_prefixes(texts, batch_size=BATCH_SIZE)

    prompts = mgr.build_text_task_prompts(texts, TASK)
    params = SamplingParams(max_tokens=8, temperature=0.0)

    print("Running text+task generation (should reuse prefix KV from SSD)...")
    outs = mgr.generate(prompts, params, batch_size=BATCH_SIZE)

    for i, out in enumerate(outs[: min(5, len(outs))]):
        text = out.outputs[0].text.strip() if out.outputs else ""
        print(f"[{i}] {text}")


if __name__ == "__main__":
    main()
