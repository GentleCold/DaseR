# SPDX-License-Identifier: Apache-2.0

"""Capture complete canonical BF16 slots from local Llama weights, offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed, redistributable input, deliberately unrelated to codebook fitting.
TEXT = """A retrieval system divides documents into passages and keeps an index.
When a reader asks a question, the index finds relevant passages. The language
model attends to those passages and to the question before writing its answer.
Cached keys and values can avoid repeated computation, but moving those values
between a processor and storage also takes time. An experiment should measure
the complete transfer, preserve every bit, and report its assumptions clearly.
The notebook records the weather, a train timetable, a recipe for bread, and a
discussion of how trees grow. Each topic adds context to the next observation.
"""


def main() -> None:
    """Parse CLI arguments and save slots plus provenance; synchronous CLI only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--block-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.slots <= 0 or args.block_tokens <= 0:
        parser.error("slots and block-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer(TEXT, add_special_tokens=False)["input_ids"]
    count = args.slots * args.block_tokens
    ids = ([tokenizer.bos_token_id] + tokens * ((count // len(tokens)) + 1))[:count]
    # Load on host first: no accelerate/device-map dependency is necessary.
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    config = model.config
    if config.model_type != "llama" or config.num_hidden_layers != 32:
        raise ValueError("this capture protocol requires Llama-3.1-8B geometry")
    with torch.inference_mode():
        # The base model avoids an unnecessary vocabulary-sized logits tensor.
        result = model.model(
            input_ids=torch.tensor([ids], device="cuda"),
            use_cache=True,
        )
    shape = (
        args.slots,
        config.num_hidden_layers,
        2,
        args.block_tokens,
        config.num_key_value_heads,
        config.head_dim,
    )
    slots = np.lib.format.open_memmap(
        args.output / "slots.npy", mode="w+", dtype="<u2", shape=shape
    )
    for layer, entry in enumerate(result.past_key_values):
        key, value = entry[:2]
        for kv, tensor in enumerate((key, value)):
            # HF [batch, heads, tokens, dim] -> canonical [slot, token, head, dim].
            bits = tensor[0].transpose(0, 1).contiguous().view(torch.uint16).cpu()
            slots[:, layer, kv] = bits.numpy().reshape(shape[0], *shape[3:])
    slots.flush()
    slot_bytes = int(slots[0].nbytes)
    manifest = {
        "model": args.model.name,
        "config": json.loads(config.to_json_string(use_diff=False)),
        "seed": 20261020,
        "text": TEXT,
        "text_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
        "token_ids": ids,
        "token_ids_sha256": hashlib.sha256(np.array(ids, dtype="<i4")).hexdigest(),
        "source": "transformers LlamaModel, BF16 SDPA, post-RoPE K and V",
        "slots": args.slots,
        "block_tokens": args.block_tokens,
        "shape": shape,
        "slot_bytes": slot_bytes,
        "dtype": "BF16 bits in little-endian uint16",
        "layout": "slot, layer, K/V, token, head, dim",
        "slot_sha256": [hashlib.sha256(slot).hexdigest() for slot in slots],
        "data_sha256": hashlib.sha256(slots).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy")
        },
    }
    # Model configs can embed local resource paths; provenance uses the model
    # basename and a path-free config so checked-in reports remain portable.
    manifest["config"].pop("_name_or_path", None)
    manifest["config"].pop("name_or_path", None)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: manifest[key] for key in ("slots", "slot_bytes", "data_sha256")}
        )
    )


if __name__ == "__main__":
    main()
