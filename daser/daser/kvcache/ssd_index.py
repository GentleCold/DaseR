from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SSDIndexMeta:
    block_bytes: int
    format_version: int = 1


class SSDIndex:
    """A tiny persistent index: block_hash -> slot_id using plain files.

    This intentionally avoids sqlite. It keeps an in-memory dict and persists
    periodic snapshots to disk via atomic file replace.

        Layout (under a directory):
            - meta.json
            - mapping.pkl       (dict[bytes, int])
    """

    def __init__(self, path: str | Path, meta: SSDIndexMeta):
        # Keep constructor signature compatible with the prior sqlite variant.
        # `path` may be either a directory or a file path; we normalize to a directory.
        p = Path(path)
        # If a directory is provided, use it.
        # If a file path is provided, use its parent directory.
        self.dir = p if p.suffix == "" else p.parent
        self.dir.mkdir(parents=True, exist_ok=True)

        self._meta_path = self.dir / "meta.json"
        self._mapping_path = self.dir / "mapping.pkl"

        self._mapping: dict[bytes, int] = {}

        self._load_from_disk()
        self._ensure_meta(meta)

    def close(self) -> None:
        # No persistent connections.
        return

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _load_from_disk(self) -> None:
        if self._mapping_path.exists():
            with open(self._mapping_path, "rb") as f:
                self._mapping = pickle.load(f)

    def _ensure_meta(self, meta: SSDIndexMeta) -> None:
        if not self._meta_path.exists():
            payload = {
                "format_version": int(meta.format_version),
                "block_bytes": int(meta.block_bytes),
            }
            self._atomic_write_bytes(self._meta_path, json.dumps(payload).encode("utf-8"))
            return

        with open(self._meta_path, "rb") as f:
            payload = json.loads(f.read().decode("utf-8"))
        existing = int(payload.get("block_bytes", -1))
        if existing != int(meta.block_bytes):
            raise ValueError(
                f"SSDIndex block_bytes mismatch: existing={existing}, requested={meta.block_bytes}. "
                "Use a separate cache dir per (model, dtype, layout, block_size)."
            )

    def iter_ready(self) -> Iterable[tuple[bytes, int]]:
        for block_hash, slot_id in self._mapping.items():
            yield block_hash, int(slot_id)

    def get(self, block_hash: bytes) -> int | None:
        slot = self._mapping.get(block_hash)
        return None if slot is None else int(slot)

    def set(self, block_hash: bytes, slot_id: int) -> None:
        self._mapping[block_hash] = int(slot_id)

    def delete(self, block_hash: bytes) -> None:
        self._mapping.pop(block_hash, None)

    def commit(self) -> None:
        # Simple, best-effort persistence.
        # We keep atomic file replacement to avoid torn writes.
        self._atomic_write_bytes(
            self._mapping_path, pickle.dumps(self._mapping, protocol=pickle.HIGHEST_PROTOCOL)
        )

    # No rollback/transactions. If a write fails, caller can recreate the cache dir.
