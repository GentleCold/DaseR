# SPDX-License-Identifier: Apache-2.0

"""Strict-lossless KV compression format and offline codec APIs."""

from daser.compression.codec import (
    EncodedSlot,
    build_compressed_store,
    calibrate_codebooks,
    decode_slot,
    encode_slot,
)
from daser.compression.format import (
    CompressedSlotRef,
    CompressedStoreGeometry,
    CompressedStoreIndex,
    SlotMode,
)

__all__ = [
    "CompressedSlotRef",
    "CompressedStoreGeometry",
    "CompressedStoreIndex",
    "EncodedSlot",
    "SlotMode",
    "build_compressed_store",
    "calibrate_codebooks",
    "decode_slot",
    "encode_slot",
]
