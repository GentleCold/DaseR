# SPDX-License-Identifier: Apache-2.0
"""Prompt construction and tokenization helpers for benchmarks."""

from __future__ import annotations

from typing import Any

from benchmarks.utils.constants import (
    BLOCK_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
)
from benchmarks.utils.datasets import BenchmarkSample

DOCS_MARKER: str = "__DASER_DOCUMENTS__"


def render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render chat messages through the tokenizer when available.

    Args:
        tokenizer: Hugging Face tokenizer or compatible object.
        messages: Chat messages.
        add_generation_prompt: Whether to append the assistant generation tag.

    Returns:
        Rendered prompt string.

    Thread-safety:
        Depends on tokenizer implementation; this function keeps no state.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        )
    body = ""
    for msg in messages:
        body += f"{msg['role']}: {msg['content']}\n"
    if add_generation_prompt:
        body += "assistant: "
    return body


def build_full_prompt(
    tokenizer: Any,
    context: str,
    question: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Build a full RAG prompt by inserting context into the chat template.

    Args:
        tokenizer: Hugging Face tokenizer or compatible object.
        context: Document text.
        question: Task/question text.
        system_prompt: System message.

    Returns:
        Prompt string ready for vLLM completions.

    Thread-safety:
        Depends on tokenizer implementation; this function keeps no state.
    """
    user_content = f"Documents:\n{DOCS_MARKER}\n\nTask: {question}"
    rendered = render_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
    )
    if rendered.count(DOCS_MARKER) != 1:
        raise RuntimeError(f"chat template marker {DOCS_MARKER!r} not unique")
    prefix, suffix = rendered.split(DOCS_MARKER, 1)
    return prefix + context + suffix


def build_document_prompt(
    tokenizer: Any,
    documents: list[str],
    question: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    doc_separator: str = "\n\n---\n\n",
) -> str:
    """Build the prompt used by DaseR document inference.

    Args:
        tokenizer: Hugging Face tokenizer or compatible object.
        documents: Document texts in prompt order.
        question: Task/question text.
        system_prompt: System message.
        doc_separator: Separator between multiple documents.

    Returns:
        Prompt string identical to ``build_full_prompt`` for a single document.

    Thread-safety:
        Depends on tokenizer implementation; this function keeps no state.
    """
    return build_full_prompt(
        tokenizer,
        doc_separator.join(documents),
        question,
        system_prompt,
    )


def build_prompts(tokenizer: Any, samples: list[BenchmarkSample]) -> list[str]:
    """Build full prompts for benchmark samples.

    Args:
        tokenizer: Hugging Face tokenizer or compatible object.
        samples: Benchmark samples.

    Returns:
        Prompt strings aligned with samples.

    Thread-safety:
        Depends on tokenizer implementation; this function keeps no state.
    """
    return [
        build_full_prompt(tokenizer, sample.context, sample.question)
        for sample in samples
    ]


def count_prompt_tokens(tokenizer: Any, prompts: list[str]) -> list[int]:
    """Count prompt tokens without adding special tokens.

    Args:
        tokenizer: Hugging Face tokenizer or compatible object.
        prompts: Prompt strings.

    Returns:
        Token counts aligned with prompts.

    Thread-safety:
        Depends on tokenizer implementation; this function keeps no state.
    """
    counts: list[int] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False)
        counts.append(len(encoded["input_ids"]))
    return counts


def workload_blocks(
    token_counts: list[int], block_tokens: int = BLOCK_TOKENS
) -> tuple[int, int]:
    """Return total and max aligned KV block counts for prompts.

    Args:
        token_counts: Prompt token counts.
        block_tokens: KV block size.

    Returns:
        ``(total_blocks, max_prompt_blocks)``.

    Thread-safety:
        Pure function.
    """
    blocks = [max(1, count // block_tokens) for count in token_counts]
    return sum(blocks), max(blocks, default=1)


def filter_by_token_limit(
    samples: list[BenchmarkSample],
    prompts: list[str],
    token_counts: list[int],
    max_context_tokens: int,
) -> tuple[list[BenchmarkSample], list[str], list[int]]:
    """Filter samples whose full prompt exceeds a token limit.

    Args:
        samples: Benchmark samples.
        prompts: Prompt strings aligned with samples.
        token_counts: Token counts aligned with samples.
        max_context_tokens: Maximum allowed prompt tokens; 0 disables filter.

    Returns:
        Filtered ``(samples, prompts, token_counts)``.

    Thread-safety:
        Pure function.
    """
    if max_context_tokens <= 0:
        return samples, prompts, token_counts
    kept_samples: list[BenchmarkSample] = []
    kept_prompts: list[str] = []
    kept_counts: list[int] = []
    for sample, prompt, count in zip(samples, prompts, token_counts, strict=False):
        if count > max_context_tokens:
            continue
        kept_samples.append(sample)
        kept_prompts.append(prompt)
        kept_counts.append(count)
    return kept_samples, kept_prompts, kept_counts


def tokenise_and_truncate(
    prompts: list[str],
    tokenizer: Any,
    max_tokens: int,
    block_tokens: int = BLOCK_TOKENS,
) -> list[list[int]]:
    """Tokenize prompts and avoid exact block-aligned terminal lengths.

    vLLM/DaseR external-prefix accounting expects at least one token beyond a
    full cached block. When a prompt would otherwise end exactly on a block
    boundary, this helper appends one pad-space token when possible.

    Args:
        prompts: Raw prompt strings.
        tokenizer: Hugging Face tokenizer or compatible object.
        max_tokens: Per-prompt token ceiling.
        block_tokens: KV block size.

    Returns:
        Token ID lists.

    Thread-safety:
        Depends on tokenizer implementation; this function keeps no state.
    """
    out: list[list[int]] = []
    for prompt in prompts:
        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        if len(ids) > 0 and len(ids) % block_tokens == 0:
            pad = list(tokenizer.encode(" ", add_special_tokens=False))
            if pad and len(ids) < max_tokens:
                ids = [*ids, pad[0]]
            elif ids:
                ids = ids[:-1]
        while len(ids) < block_tokens + 1:
            pad = list(tokenizer.encode(" ", add_special_tokens=False))
            if not pad:
                break
            ids.append(pad[0])
        out.append(ids)
    return out
