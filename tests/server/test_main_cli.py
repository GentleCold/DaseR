# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

# Third Party
import pytest

# First Party
from daser.config import DEFAULT_IOURING_L1_BYTES
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.__main__ import (
    DASER_ASCII_BANNER,
    DASER_BANNER_COLOR,
    DASER_BANNER_RESET,
    VLLMStartupError,
    _build_daser_config,
    _build_http_config,
    _build_index_components,
    _consume_completed_task,
    _ensure_store_file,
    _log_startup_banner,
    _parse_args,
    _parse_size_bytes,
    _read_vllm_model_id,
    _resolve_model_paths,
    _shutdown_server,
    main,
)


def _run_parse(argv: list[str]):
    saved = sys.argv
    sys.argv = ["daser.server", *argv]
    try:
        return _parse_args()
    finally:
        sys.argv = saved


def _write_model_config(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 28,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )


def test_parse_size_bytes_accepts_human_readable_units() -> None:
    assert _parse_size_bytes("10gb") == 10 * 1000**3
    assert _parse_size_bytes("10gib") == 10 * 1024**3
    assert _parse_size_bytes("512mb") == 512 * 1000**2
    assert _parse_size_bytes("512mib") == 512 * 1024**2
    assert _parse_size_bytes("2097152") == 2097152


def test_documented_flags_populate_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
            "--socket-path",
            "/tmp/daser.sock",
            "--transfer-mode",
            "iouring",
            "--l1-size",
            "1gb",
        ]
    )
    cfg = _build_daser_config(args)

    assert args.port == 2026
    assert cfg.model_path == str(model_path)
    assert cfg.vllm_model_id == str(model_path)
    assert cfg.store_path == str(store_dir / "daser.store")
    assert cfg.ipc_socket_path == "/tmp/daser.sock"
    assert cfg.index_path == str(store_dir / "daser.index")
    assert cfg.transfer_mode == "iouring"
    assert cfg.l1_size_bytes == 1000**3
    assert cfg.l2_size_bytes == cfg.aligned_store_bytes
    assert cfg.total_slots > 0
    assert cfg.aligned_store_bytes <= 10 * 1000**3
    assert cfg.aligned_store_bytes == cfg.total_slots * cfg.resolved_slot_size()

    http_cfg = _build_http_config(args)
    assert http_cfg.vllm_base_url == "http://127.0.0.1:8001"
    assert http_cfg.model == str(model_path)
    assert http_cfg.tokenizer == str(model_path)
    assert http_cfg.align_document_chunks is True
    assert args.cache_reuse_mode == "chunk"
    runtime = cfg.runtime_config()
    assert runtime["transfer_mode"] == "iouring"
    assert runtime["l1_size_bytes"] == 1000**3
    assert runtime["l2_size_bytes"] == cfg.aligned_store_bytes


def test_default_transfer_mode_is_iouring(tmp_path: Path) -> None:
    """The server defaults to iouring unless a transfer mode is specified."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
        ]
    )

    cfg = _build_daser_config(args)

    assert args.transfer_mode == "iouring"
    assert cfg.transfer_mode == "iouring"
    assert cfg.l1_size_bytes == min(DEFAULT_IOURING_L1_BYTES, cfg.l2_size_bytes)
    assert cfg.runtime_config()["transfer_mode"] == "iouring"
    assert cfg.runtime_config()["l1_size_bytes"] == cfg.l1_size_bytes


def test_skip_l2_populates_memory_only_runtime_config(tmp_path: Path) -> None:
    """--skip-l2 keeps logical slots but disables store/index persistence."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
            "--skip-l2",
        ]
    )

    cfg = _build_daser_config(args)
    runtime = cfg.runtime_config()

    assert cfg.skip_l2 is True
    assert cfg.transfer_mode == "iouring"
    assert cfg.l1_size_bytes == min(DEFAULT_IOURING_L1_BYTES, cfg.l2_size_bytes)
    assert runtime["skip_l2"] is True
    assert runtime["transfer_mode"] == "iouring"
    assert runtime["store_path"] == ""
    assert runtime["l2_size_bytes"] == cfg.aligned_store_bytes
    assert runtime["total_slots"] == cfg.total_slots


def test_skip_l2_rejects_gds_with_clear_message(tmp_path: Path) -> None:
    """GDS requires an L2 store file, so it cannot run with --skip-l2."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--transfer-mode",
            "gds",
            "--skip-l2",
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        _build_daser_config(args)

    message = str(exc_info.value)
    assert "--skip-l2 is incompatible with --transfer-mode=gds" in message
    assert "GDS requires an L2 store file" in message


def test_cache_reuse_mode_chunk_selects_chunk_components(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--cache-reuse-mode",
            "chunk",
        ]
    )

    retrieval, position = _build_index_components(args.cache_reuse_mode, 16)
    http_cfg = _build_http_config(args)

    assert isinstance(retrieval, ChunkReuseIndex)
    assert isinstance(position, ChunkPositionEncoder)
    assert http_cfg.align_document_chunks is True


def test_cache_reuse_mode_prefix_selects_prefix_components(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--cache-reuse-mode",
            "prefix",
        ]
    )

    retrieval, position = _build_index_components(args.cache_reuse_mode, 16)

    assert isinstance(retrieval, PrefixHashIndex)
    assert isinstance(position, FixedOffsetEncoder)


def test_l2_size_must_fit_at_least_one_slot(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="at least one slot"):
        _build_daser_config(args)


def test_ensure_store_file_truncates_larger_legacy_file(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
        ]
    )
    cfg = _build_daser_config(args)
    store_dir.mkdir()
    with open(cfg.store_path, "wb") as f:
        f.truncate(cfg.total_store_bytes)

    _ensure_store_file(cfg)

    assert Path(cfg.store_path).stat().st_size == cfg.aligned_store_bytes


def test_ensure_store_file_rejects_smaller_existing_file(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
        ]
    )
    cfg = _build_daser_config(args)
    store_dir.mkdir()
    with open(cfg.store_path, "wb") as f:
        f.truncate(cfg.aligned_store_bytes - 1)

    with pytest.raises(ValueError, match="has size"):
        _ensure_store_file(cfg)


def test_skip_l2_does_not_create_store_file(tmp_path: Path) -> None:
    """Memory-only mode must not allocate daser.store at startup."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--skip-l2",
        ]
    )
    cfg = _build_daser_config(args)

    _ensure_store_file(cfg)

    assert store_dir.exists()
    assert not Path(cfg.store_path).exists()


def test_model_path_is_optional_when_vllm_model_is_local_path(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--store-dir",
            str(tmp_path / "store"),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
        ]
    )

    model_id, resolved_model_path = _resolve_model_paths(
        args,
        vllm_model_id=str(model_path),
    )

    assert model_id == str(model_path)
    assert resolved_model_path == str(model_path)


def test_model_path_is_required_when_vllm_model_is_not_local_path(
    tmp_path: Path,
) -> None:
    args = _run_parse(
        [
            "--store-dir",
            str(tmp_path / "store"),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
        ]
    )
    with pytest.raises(ValueError, match="--model-path is required"):
        _resolve_model_paths(args, vllm_model_id="served-alias")


@pytest.mark.asyncio
async def test_read_vllm_model_id_explains_when_vllm_is_unavailable(
    monkeypatch,
) -> None:
    """Startup should tell users to start vLLM before DaseR."""

    class FakeVLLMClient:
        def __init__(self, base_url: str, model: str) -> None:
            self.base_url = base_url
            self.model = model

        async def list_models(self) -> list[str]:
            raise OSError("connection refused")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("daser.server.__main__.VLLMClient", FakeVLLMClient)

    with pytest.raises(RuntimeError) as exc_info:
        await _read_vllm_model_id("http://127.0.0.1:8001")

    message = str(exc_info.value)
    assert "vLLM is not reachable at http://127.0.0.1:8001" in message
    assert "Please start vLLM before starting DaseR" in message
    assert "connection refused" in message


def test_startup_banner_logs_daser_ascii_art(monkeypatch) -> None:
    """DaseR startup should print a recognizable terminal banner."""
    expected_banner = """

████▄   ▄▄▄   ▄▄▄▄ ▄▄▄▄▄ █████▄
██  ██ ██▀██ ███▄▄ ██▄▄  ██▄▄██▄
████▀  ██▀██ ▄▄██▀ ██▄▄▄ ██   ██
"""
    expected_colored = (
        "\n"
        "\n████▄   ▄▄▄   ▄▄▄▄ ▄▄▄▄▄ "
        f"{DASER_BANNER_COLOR}█████▄{DASER_BANNER_RESET}\n"
        "██  ██ ██▀██ ███▄▄ ██▄▄  "
        f"{DASER_BANNER_COLOR}██▄▄██▄{DASER_BANNER_RESET}\n"
        "████▀  ██▀██ ▄▄██▀ ██▄▄▄ "
        f"{DASER_BANNER_COLOR}██   ██{DASER_BANNER_RESET}\n"
    )
    messages: list[str] = []
    monkeypatch.setattr(
        "daser.server.__main__.logger.info",
        lambda message, *args: messages.append(message % args if args else message),
    )

    _log_startup_banner()

    assert DASER_ASCII_BANNER == expected_banner
    assert DASER_BANNER_COLOR == "\033[38;2;102;178;255m"
    assert DASER_BANNER_RESET == "\033[0m"
    assert expected_colored in messages


def test_main_exits_cleanly_when_vllm_is_unavailable(monkeypatch) -> None:
    """The CLI should show the vLLM hint without a Python traceback."""
    messages: list[str] = []

    async def fake_run_server(_args: Any) -> None:
        raise VLLMStartupError("Please start vLLM before starting DaseR")

    monkeypatch.setattr("daser.server.__main__._parse_args", lambda: object())
    monkeypatch.setattr("daser.server.__main__.run_server", fake_run_server)
    monkeypatch.setattr(
        "daser.server.__main__.logger.error",
        lambda message, *args: messages.append(message % args if args else message),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert messages == ["[SERVER] Please start vLLM before starting DaseR"]


def test_main_exits_cleanly_on_invalid_startup_config(monkeypatch) -> None:
    """The CLI should print configuration errors without a Python traceback."""
    messages: list[str] = []

    async def fake_run_server(_args: Any) -> None:
        raise ValueError(
            "--skip-l2 is incompatible with --transfer-mode=gds because "
            "GDS requires an L2 store file"
        )

    monkeypatch.setattr("daser.server.__main__._parse_args", lambda: object())
    monkeypatch.setattr("daser.server.__main__.run_server", fake_run_server)
    monkeypatch.setattr(
        "daser.server.__main__.logger.error",
        lambda message, *args: messages.append(message % args if args else message),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert messages == [
        "[SERVER] --skip-l2 is incompatible with --transfer-mode=gds because "
        "GDS requires an L2 store file"
    ]


def test_store_dir_is_required(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    _write_model_config(model_path)
    with pytest.raises(SystemExit):
        _run_parse(
            [
                "--model-path",
                str(model_path),
                "--vllm-base-url",
                "http://127.0.0.1:8001",
            ]
        )


def test_http_flags_are_required():
    with pytest.raises(SystemExit):
        _run_parse(["--store-dir", "/tmp/store"])


def test_consume_completed_task_ignores_cancelled_http_task() -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(cancelled())
        with pytest.raises(asyncio.CancelledError):
            loop.run_until_complete(task)

        _consume_completed_task(task)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_shutdown_server_stops_acceptance_before_saving(tmp_path: Path) -> None:
    events: list[str] = []
    index_path = str(tmp_path / "daser.index")

    class FakeHTTPTask:
        def done(self) -> bool:
            return False

    class FakeHTTPServer:
        should_exit = False

    class FakeChunkManager:
        def save(self, path: str) -> None:
            events.append(f"save:{path}")

    class FakeCore:
        chunk_manager = FakeChunkManager()

    class FakeIPCServer:
        async def stop_accepting(self) -> None:
            events.append("stop_accepting")

        async def close(self) -> None:
            events.append("close")

    async def fake_wait_for(task: Any, timeout: float) -> None:
        assert isinstance(task, FakeHTTPTask)
        assert timeout == 5
        events.append("http_wait")

    http_server = FakeHTTPServer()
    await _shutdown_server(
        http_server=http_server,
        http_task=FakeHTTPTask(),
        ipc_server=FakeIPCServer(),
        core=FakeCore(),
        index_path=index_path,
        wait_for=fake_wait_for,
    )

    assert http_server.should_exit is True
    assert events == ["http_wait", "stop_accepting", f"save:{index_path}", "close"]


@pytest.mark.asyncio
async def test_shutdown_server_saves_after_cancelled_http_task(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    index_path = str(tmp_path / "daser.index")

    class FakeHTTPTask:
        def done(self) -> bool:
            return False

    class FakeHTTPServer:
        should_exit = False

    class FakeChunkManager:
        def save(self, path: str) -> None:
            events.append(f"save:{path}")

    class FakeCore:
        chunk_manager = FakeChunkManager()

    class FakeIPCServer:
        async def stop_accepting(self) -> None:
            events.append("stop_accepting")

        async def close(self) -> None:
            events.append("close")

    async def cancelled_wait_for(_task: Any, _timeout: float) -> None:
        events.append("http_cancelled")
        raise asyncio.CancelledError

    http_server = FakeHTTPServer()
    await _shutdown_server(
        http_server=http_server,
        http_task=FakeHTTPTask(),
        ipc_server=FakeIPCServer(),
        core=FakeCore(),
        index_path=index_path,
        wait_for=cancelled_wait_for,
    )

    assert http_server.should_exit is True
    assert events == [
        "http_cancelled",
        "stop_accepting",
        f"save:{index_path}",
        "close",
    ]


@pytest.mark.asyncio
async def test_shutdown_server_skips_index_save_when_l2_is_skipped(
    tmp_path: Path,
) -> None:
    """Memory-only mode must not persist daser.index during shutdown."""
    events: list[str] = []
    index_path = str(tmp_path / "daser.index")

    class FakeHTTPTask:
        def done(self) -> bool:
            return True

    class FakeHTTPServer:
        should_exit = False

    class FakeChunkManager:
        def save(self, path: str) -> None:
            events.append(f"save:{path}")

    class FakeCore:
        chunk_manager = FakeChunkManager()

    class FakeIPCServer:
        async def stop_accepting(self) -> None:
            events.append("stop_accepting")

        async def close(self) -> None:
            events.append("close")

    await _shutdown_server(
        http_server=FakeHTTPServer(),
        http_task=FakeHTTPTask(),
        ipc_server=FakeIPCServer(),
        core=FakeCore(),
        index_path=index_path,
        skip_l2=True,
    )

    assert events == ["stop_accepting", "close"]
    assert not Path(index_path).exists()
