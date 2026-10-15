# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC buffers open on the exporter's GPU across device-visibility sets."""

import os
import subprocess
import sys
import textwrap
import uuid

import pytest

cupy = pytest.importorskip("cupy")

from daser.transfer.cuda_ipc import (  # noqa: E402
    cuda_array_pci_bus_id,
    open_cuda_ipc_buffer,
)


def _device_count() -> int:
    try:
        return int(cupy.cuda.runtime.getDeviceCount())
    except cupy.cuda.runtime.CUDARuntimeError:
        return 0


# The exporter sees only the last visible GPU, so its ordinal is 0 while the
# receiver (this process) knows that GPU under a different ordinal.
_EXPORTER = textwrap.dedent(
    """
    import sys
    import cupy
    from daser.transfer.cuda_ipc import cuda_array_pci_bus_id, export_cuda_ipc_handle

    array = cupy.arange(256, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    handle = export_cuda_ipc_handle(array).hex()
    sys.stdout.write(handle + " " + cuda_array_pci_bus_id(array) + "\\n")
    sys.stdout.flush()
    sys.stdin.readline()
    """
)


@pytest.mark.skipif(_device_count() < 2, reason="needs two visible CUDA devices")
def test_ipc_buffer_opens_on_exporter_gpu_despite_different_ordinals() -> None:
    """A handle exported as device 0 elsewhere opens on the same physical GPU."""
    receiver_last = _device_count() - 1
    raw_uuid = cupy.cuda.runtime.getDeviceProperties(receiver_last)["uuid"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = f"GPU-{uuid.UUID(bytes=bytes(raw_uuid[:16]))}"
    exporter = subprocess.Popen(
        [sys.executable, "-c", _EXPORTER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        handle_hex, exported_bus_id = exporter.stdout.readline().split()
        opened = open_cuda_ipc_buffer(
            handle=bytes.fromhex(handle_hex),
            nbytes=256,
            pci_bus_id=exported_bus_id,
        )
        try:
            assert opened.array.device.id == receiver_last
            assert cuda_array_pci_bus_id(opened.array) == exported_bus_id
            assert cupy.asnumpy(opened.array).tolist() == list(range(256))
        finally:
            opened.close()
    finally:
        exporter.communicate("\n", timeout=60)
    assert exporter.returncode == 0
