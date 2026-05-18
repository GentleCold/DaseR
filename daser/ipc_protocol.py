# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import socket
from typing import Any

# Third Party
import msgpack

HEADER_SIZE = 4


def pack_frame(payload: dict[str, Any]) -> bytes:
    """Encode a payload as a length-prefixed msgpack frame.

    Args:
        payload: msgpack-serializable request or response body.

    Returns:
        Four-byte big-endian length header followed by msgpack bytes.
    """
    data = msgpack.packb(payload, use_bin_type=True)
    return len(data).to_bytes(HEADER_SIZE, "big") + data


def unpack_frame(data: bytes) -> dict[str, Any]:
    """Decode msgpack frame payload bytes.

    Args:
        data: msgpack payload bytes without the length header.

    Returns:
        Decoded frame body.
    """
    return msgpack.unpackb(data, raw=False)


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    """Receive exactly ``nbytes`` bytes from a blocking socket.

    Args:
        sock: connected blocking socket.
        nbytes: number of bytes to receive.

    Returns:
        Received bytes.

    Raises:
        ConnectionError: if the peer closes before enough bytes arrive.
    """
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("socket closed before receiving all bytes")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    """Read and decode one frame from a blocking socket.

    Args:
        sock: connected blocking socket.

    Returns:
        Decoded frame body.
    """
    header = recv_exact(sock, HEADER_SIZE)
    length = int.from_bytes(header, "big")
    return unpack_frame(recv_exact(sock, length))


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read and decode one frame from an asyncio stream.

    Args:
        reader: stream reader.

    Returns:
        Decoded frame body.

    Async/thread-safety:
        Performs async socket I/O on the caller's event loop.
    """
    header = await reader.readexactly(HEADER_SIZE)
    length = int.from_bytes(header, "big")
    return unpack_frame(await reader.readexactly(length))


async def write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    """Encode and write one frame to an asyncio stream.

    Args:
        writer: stream writer.
        payload: msgpack-serializable request or response body.

    Async/thread-safety:
        Performs async socket I/O on the caller's event loop.
    """
    writer.write(pack_frame(payload))
    await writer.drain()
