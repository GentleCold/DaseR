# SPDX-License-Identifier: Apache-2.0

# Standard
import io
import logging

# First Party
from daser.logging import init_logger


def test_init_logger_colors_daser_records() -> None:
    """DaseR log records should be visually distinct from vLLM records."""
    stream = io.StringIO()
    logger = init_logger("tests.daser.color", level="INFO", stream=stream)
    logger.propagate = False

    logger.info("[HTTP] ready")

    output = stream.getvalue()
    assert "\033[38;2;102;178;255mDaseR\033[0m" in output
    assert "[HTTP] ready" in output


def test_init_logger_replaces_existing_daser_handler_stream() -> None:
    """Tests can inject streams without stacking duplicate DaseR handlers."""
    first = io.StringIO()
    second = io.StringIO()
    logger = init_logger("tests.daser.replace", level="INFO", stream=first)
    logger = init_logger("tests.daser.replace", level="INFO", stream=second)
    logger.propagate = False

    logger.info("single")

    assert first.getvalue() == ""
    assert second.getvalue().count("single") == 1
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
