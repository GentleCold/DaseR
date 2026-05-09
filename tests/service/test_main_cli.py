# SPDX-License-Identifier: Apache-2.0

# Standard
import sys

# Third Party
import pytest

# First Party
from daser.service.__main__ import _parse_args


def _run_parse(argv: list[str]):
    saved = sys.argv
    sys.argv = ["daser.service", *argv]
    try:
        return _parse_args()
    finally:
        sys.argv = saved


def test_infer_cache_mode_doc_rope_parses() -> None:
    args = _run_parse(
        [
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--model",
            "m",
            "--tokenizer",
            "tok",
            "--infer-cache-mode",
            "doc-rope",
        ]
    )

    assert args.infer_cache_mode == "doc-rope"


def test_infer_cache_mode_rejects_unknown_value() -> None:
    with pytest.raises(SystemExit):
        _run_parse(
            [
                "--vllm-base-url",
                "http://127.0.0.1:8001",
                "--model",
                "m",
                "--tokenizer",
                "tok",
                "--infer-cache-mode",
                "bad",
            ]
        )
