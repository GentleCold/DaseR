# SPDX-License-Identifier: Apache-2.0

from daser.replacement.base import ReplacementPolicy
from daser.replacement.compression_density import (
    CompressionDensityReplacementPolicy,
)
from daser.replacement.lru import LRUReplacementPolicy

__all__ = [
    "CompressionDensityReplacementPolicy",
    "LRUReplacementPolicy",
    "ReplacementPolicy",
]
