# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.ipc_protocol import pack_frame, unpack_frame


def test_pack_frame_uses_length_prefixed_msgpack() -> None:
    payload = {"op": "lookup", "tokens": [1, 2, 3], "model_id": "m"}

    frame = pack_frame(payload)

    length = int.from_bytes(frame[:4], "big")
    assert length == len(frame) - 4
    assert unpack_frame(frame[4:]) == payload
