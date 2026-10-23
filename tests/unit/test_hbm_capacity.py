# SPDX-License-Identifier: Apache-2.0
"""CPU allocation checks for the public HBM capacity accounting contract."""

from typing import Any

import pytest

torch = pytest.importorskip("torch", minversion="2.0")

from daser.compression.attn_tables import DeviceTables  # noqa: E402
from daser.ops.hbm_kv import HbmKvStore, page_bytes  # noqa: E402


def test_memory_accounting_uses_actual_stride_and_tensor_element_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accounting must include row padding and metadata without a CUDA read."""
    original_zeros = torch.zeros

    def cpu_zeros(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("pin_memory", None)
        return original_zeros(*args, **kwargs)

    # Only the asynchronous polling destination is pinned in the constructor;
    # these tests never enqueue kernels or use that destination.
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    caches = [torch.zeros((3, page_bytes(1) + 16), dtype=torch.uint8) for _ in range(2)]
    tables = DeviceTables(
        torch.zeros((2, 2, 1, 256), dtype=torch.uint8),
        torch.zeros((2, 2, 1, 2), dtype=torch.int32),
        torch.zeros((2, 2, 1, 16), dtype=torch.uint8),
    )
    store = HbmKvStore(caches, 1, tables, 128, staging_slots=2, overflow_records=3)

    def no_device_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("capacity observation must only read host metadata")

    monkeypatch.setattr(torch.Tensor, "cpu", no_device_read)
    monkeypatch.setattr(torch.Tensor, "item", no_device_read)
    monkeypatch.setattr(torch.cuda, "synchronize", no_device_read)
    usage = store.memory_usage()
    assert usage["arena"] == 2 * 3 * (page_bytes(1) + 16)
    assert usage["staging"] == 2 * 2 * 2 * 32768
    assert usage["scratch"] == 3 * 2 * 32768
    assert usage["overflow"] == 3 * 32768
    assert usage["codebooks"] == 2 * 2 * (256 + 2 * 4 + 16)
    # 16 counters, per-block slot/tags/fill/live metadata, two free slots,
    # one overflow bitmap word, two 128-entry lists, one null BF16 record,
    # 12 int64 addresses + 12 int32 lengths and per-layer base/stride arrays.
    assert usage["metadata"] == (
        16 * 4
        + 3 * 4
        + 4 * 3 * 4
        + 2 * 4
        + 4
        + 2 * 128 * 4
        + 32768
        + 12 * 8
        + 12 * 4
        + 2 * 3 * 4
        + 3 * 4
        + 2 * 2 * 8
    )
