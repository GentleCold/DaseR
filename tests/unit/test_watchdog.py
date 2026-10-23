# SPDX-License-Identifier: Apache-2.0
"""Tests for the agent-trace arm watchdog."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmarks.research.watchdog import parse_args, watch


def test_wrapper_exit_is_not_a_service_exit_while_its_group_runs(
    tmp_path: Path,
) -> None:
    """A service whose wrapper exited but whose server still runs is healthy."""
    out = tmp_path / "art"
    service = tmp_path / "scr"
    out.mkdir()
    (service / "logs").mkdir(parents=True)
    # Stand-in replay client: the watchdog finds it by module name and dir.
    replay = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(2)",
            "benchmarks.mooncake.benchmark",
            str(out.resolve()),
        ]
    )
    # The wrapper (like ``nsys profile --duration``) leaves its server running.
    wrapper = subprocess.Popen(
        ["sh", "-c", "sleep 60 & echo $!"],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert wrapper.stdout is not None
    wrapper.stdout.readline()  # the server has started
    wrapper.wait(timeout=5)
    (service / "pids.json").write_text(json.dumps([{"pid": wrapper.pid}]))
    try:
        ok, message = watch(
            parse_args(
                [
                    "--output-dir",
                    str(out),
                    "--service-dir",
                    str(service),
                    "--vllm-url",
                    "http://127.0.0.1:9",
                    "--interval",
                    "0.4",
                    "--data-path",
                    str(tmp_path),
                    "--min-free-data-gib",
                    "0",
                    "--min-free-root-gib",
                    "0",
                    "--no-stop",
                ]
            )
        )
    finally:
        os.killpg(wrapper.pid, signal.SIGKILL)
        replay.wait(timeout=5)
    assert ok, message
    assert message.startswith("replay exited")


def _startup_args(out: Path, service: Path, tmp_path: Path) -> list[str]:
    # These tests exercise process discovery and log classification. Host disk
    # capacity must not preempt those checks on a busy development machine.
    return [
        "--output-dir",
        str(out),
        "--service-dir",
        str(service),
        "--vllm-url",
        "http://127.0.0.1:9",
        "--interval",
        "0.4",
        "--startup-timeout",
        "30",
        "--data-path",
        str(tmp_path),
        "--min-free-data-gib",
        "0",
        "--min-free-root-gib",
        "0",
        "--no-stop",
    ]


def _runner(out: Path) -> subprocess.Popen:
    """Stand-in arm runner: found by module name and the artifact dir."""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(1)",
            "benchmarks.research.agent_trace_arms",
            str(out.parent.resolve()),
        ]
    )


def test_runner_exit_before_replay_is_an_anomaly(tmp_path: Path) -> None:
    """A runner whose services never came up ends the watch without a replay."""
    out = tmp_path / "art" / "daser-compressed"
    service = tmp_path / "scr"
    out.mkdir(parents=True)
    runner = _runner(out)
    try:
        ok, message = watch(parse_args(_startup_args(out, service, tmp_path)))
    finally:
        runner.wait(timeout=5)
    assert not ok
    assert message == "runner exited before the replay client started"


def test_runner_exit_with_requests_is_a_finished_replay(tmp_path: Path) -> None:
    """A replay that finished between polls is not reported as a startup failure."""
    out = tmp_path / "art" / "daser-compressed"
    service = tmp_path / "scr"
    out.mkdir(parents=True)
    (out / "requests.jsonl").write_text("")
    runner = _runner(out)
    try:
        ok, message = watch(parse_args(_startup_args(out, service, tmp_path)))
    finally:
        runner.wait(timeout=5)
    assert ok, message


def test_agentx_harness_is_watched_as_the_replay(tmp_path: Path) -> None:
    """A replay recorded in ``replay.pid`` is watched whatever its argv."""
    out = tmp_path / "art" / "daser-compressed"
    service = tmp_path / "scr"
    (out / "agentx").mkdir(parents=True)
    # Like the AgentX harness after setproctitle: nothing in argv to match.
    harness = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    (out / "replay.pid").write_text(str(harness.pid))
    try:
        ok, message = watch(parse_args(_startup_args(out, service, tmp_path)))
    finally:
        harness.wait(timeout=5)
    assert ok, message
    assert message.startswith("replay exited")
    assert f"pid={harness.pid}" in message


def _watch_with_service_log(tmp_path: Path, lines: list[str]) -> tuple[bool, str]:
    """Watch a short replay while a service log holds ``lines``."""
    out = tmp_path / "art" / "lmcache"
    service = tmp_path / "scr"
    (out / "agentx").mkdir(parents=True)
    (service / "logs").mkdir(parents=True)
    (service / "logs" / "lmcache_mp_server.log").write_text("\n".join(lines) + "\n")
    harness = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    (out / "replay.pid").write_text(str(harness.pid))
    try:
        return watch(parse_args(_startup_args(out, service, tmp_path)))
    finally:
        harness.terminate()
        harness.wait(timeout=5)


def test_lmcache_lazy_heap_memcheck_is_not_an_anomaly(tmp_path: Path) -> None:
    """LMCache's memcheck race during lazy pinning does not stop the arm."""
    ok, message = _watch_with_service_log(
        tmp_path,
        [
            "LMCache ERROR: Memory allocator size is inconsistent",
            "LMCache ERROR: This implies a bug in the memory allocator",
        ],
    )
    assert ok, message
    assert message.startswith("replay exited")


def test_other_service_error_is_an_anomaly(tmp_path: Path) -> None:
    """Any other ERROR line in a service log stops the arm."""
    ok, message = _watch_with_service_log(
        tmp_path, ["LMCache ERROR: Failed to store chunk"]
    )
    assert not ok
    assert "Failed to store chunk" in message
