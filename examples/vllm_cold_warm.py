# SPDX-License-Identifier: Apache-2.0
"""Minimal cold -> warm demo of DaseR + vLLM.

Run a DaseR server first (see ``run_daser_server.py``), then launch this
script. It performs two phases:

1. Cold: build a fresh ``LLM``; generate on two prompts; DaseR stores the
   resulting KV blocks to NVMe.
2. Warm: tear the engine down, build a second ``LLM`` with empty GPU KV cache;
   DaseR looks up the cached prefixes and streams them back from NVMe.

We report wall-clock generation time for both phases and verify that the
decoded text is identical. This is a runnable counterpart to
``tests/integration/test_vllm_e2e.py``.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python examples/vllm_cold_warm.py \\
        --model /path/to/Qwen3-8B \\
        --store-path /tmp/daser_example/daser.store \\
        --socket-path /tmp/daser_example/daser.sock
"""

# Future
from __future__ import annotations

# Standard
import argparse
import gc
from time import perf_counter

# Third Party
import torch
from vllm import LLM, SamplingParams

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

# Qwen3-8B: 36 layers * 8 KV heads * 128 head_dim * 16 block_tokens * bf16 * 2 (K+V).
DEFAULT_SLOT_SIZE: int = 8 * 128 * 2 * 36 * 16 * 2
DEFAULT_BLOCK_TOKENS: int = 16
DEFAULT_MAX_NEW_TOKENS: int = 64

# Two prompts long enough (>= 64 tokens each) to produce several full KV blocks.
PROMPTS: list[str] = [
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


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for the example client."""
    parser = argparse.ArgumentParser(
        description="Cold -> warm demo of DaseR via DaserConnector + vLLM.",
    )
    parser.add_argument("--model", required=True, help="Path or HF id of the model")
    parser.add_argument("--store-path", required=True, help="DaseR NVMe store path")
    parser.add_argument(
        "--socket-path",
        default="/tmp/daser.sock",
        help="Unix socket of the running DaseR server",
    )
    parser.add_argument(
        "--slot-size",
        type=int,
        default=DEFAULT_SLOT_SIZE,
        help=f"Bytes per KV slot (default Qwen3-8B: {DEFAULT_SLOT_SIZE})",
    )
    parser.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS)
    parser.add_argument("--model-id", default="qwen3-8b")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    return parser.parse_args()


def _make_llm(args: argparse.Namespace) -> LLM:
    """Build a vLLM ``LLM`` wired to the DaseR connector.

    Args:
        args: parsed CLI namespace (must contain model / DaseR connection info).

    Returns:
        A fully initialized ``LLM`` ready for ``generate`` calls.
    """
    kv_transfer_config = {
        "kv_connector": "DaserConnector",
        "kv_connector_module_path": "daser.connector.daser_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "socket_path": args.socket_path,
            "store_path": args.store_path,
            "slot_size": args.slot_size,
            "block_tokens": args.block_tokens,
            "model_id": args.model_id,
        },
    }
    return LLM(
        model=args.model,
        kv_transfer_config=kv_transfer_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_hybrid_kv_cache_manager=True,
    )


def _destroy_llm(llm: LLM) -> None:
    """Tear down a vLLM engine and release GPU memory.

    Mirrors the shutdown sequence used by the E2E integration test so that the
    second ``LLM`` starts with a clean GPU KV cache.
    """
    try:
        llm.llm_engine.engine_core.shutdown(timeout=30.0)
    except TypeError:
        llm.llm_engine.engine_core.shutdown()
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        logger.warning("[EXAMPLE] engine_core.shutdown raised: %s", exc)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _run_phase(
    label: str,
    args: argparse.Namespace,
    params: SamplingParams,
) -> tuple[list[str], float]:
    """Run one generation phase end-to-end.

    Args:
        label: human-readable phase name (``cold`` or ``warm``).
        args: parsed CLI namespace.
        params: ``SamplingParams`` used for both phases to keep outputs comparable.

    Returns:
        Tuple of (decoded texts per prompt, wall-clock generation time in seconds).
    """
    logger.info("[EXAMPLE] %s: constructing LLM", label)
    llm = _make_llm(args)
    logger.info("[EXAMPLE] %s: generating", label)
    t0 = perf_counter()
    outputs = llm.generate(PROMPTS, params)
    elapsed = perf_counter() - t0
    texts = [out.outputs[0].text for out in outputs]
    logger.info("[EXAMPLE] %s: done in %.2fs", label, elapsed)
    _destroy_llm(llm)
    return texts, elapsed


def main() -> None:
    """Run the cold phase, then the warm phase, and log the comparison."""
    args = _parse_args()
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    cold_texts, cold_time = _run_phase("cold", args, params)
    warm_texts, warm_time = _run_phase("warm", args, params)

    all_match = all(c == w for c, w in zip(cold_texts, warm_texts, strict=False))
    speedup = cold_time / warm_time if warm_time > 0 else float("inf")
    logger.info(
        "[EXAMPLE] summary: cold=%.2fs warm=%.2fs speedup=%.2fx match=%s",
        cold_time,
        warm_time,
        speedup,
        all_match,
    )
    for i, (cold, warm) in enumerate(zip(cold_texts, warm_texts, strict=False)):
        logger.info("[EXAMPLE] prompt %d cold: %s", i, cold.replace("\n", " "))
        logger.info("[EXAMPLE] prompt %d warm: %s", i, warm.replace("\n", " "))


if __name__ == "__main__":
    main()
