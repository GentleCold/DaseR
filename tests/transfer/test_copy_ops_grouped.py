# SPDX-License-Identifier: Apache-2.0

"""Grouped pinned-to-CUDA copies."""

import pytest

from daser.transfer.iouring import copy_ops
from daser.transfer.iouring.pinned_pool import PinnedMemoryPool
from tests.transfer.cuda import require_cuda


class _StreamDestination:
    """CUDA destination that owns a private copy stream, like staging IPC."""

    def __init__(self, array: object, stream: object) -> None:
        self.array = array
        self.stream = stream

    def __getitem__(self, item: object) -> object:
        """Return a CuPy slice of the destination."""
        return self.array[item]

    @property
    def copy_stream_ptr(self) -> int:
        """Return the non-default stream the copies are queued on."""
        return int(self.stream.ptr)


def _scattered_copy(piece: int) -> tuple[bytes, bytes, int]:
    """Copy eight scattered pinned pieces to permuted GPU offsets.

    Returns the expected bytes, the GPU result and the reported run count.
    """
    cupy = require_cuda()
    count = 8
    stride = 3 * piece + 512
    pool = PinnedMemoryPool(stride * count + piece)
    try:
        source = pool.allocate(stride * count + piece)
        assert source is not None
        view = source.view()
        for i in range(len(view)):
            view[i] = (i * 7 + 3) % 251
        order = [5, 2, 7, 0, 3, 6, 1, 4]
        chunks: list[copy_ops.CopyChunk] = [
            (slot * piece, source, i * stride, piece) for i, slot in enumerate(order)
        ]
        expected = bytearray(count * piece)
        for target, _data, offset, nbytes in chunks:
            expected[target : target + nbytes] = view[offset : offset + nbytes]
        # The scattered copies overwrite every byte. A default-stream memset
        # would race their nonblocking private stream and could zero bytes
        # after the copy even though the final device synchronize succeeds.
        dst = cupy.empty(count * piece, dtype=cupy.uint8)
        target = _StreamDestination(dst, cupy.cuda.Stream(non_blocking=True))
        runs = copy_ops.copy_grouped_to_cuda_dst(target, chunks)
        cupy.cuda.Device().synchronize()
        result = cupy.asnumpy(dst).tobytes()
        source.close()
        return bytes(expected), result, runs
    finally:
        pool.close()


@pytest.mark.parametrize("piece", [4096, 4100])
def test_scattered_copy_matches_source(piece: int) -> None:
    """Permuted pinned pieces, aligned or not, land at their targets."""
    expected, result, runs = _scattered_copy(piece)
    assert runs == 8
    assert result == expected
