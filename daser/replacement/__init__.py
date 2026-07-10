# SPDX-License-Identifier: Apache-2.0

from daser.replacement.base import ReplacementPolicy
from daser.replacement.lru import LRUReplacementPolicy
from daser.replacement.prefix_aware_lru import PrefixAwareLRUReplacementPolicy

__all__ = [
    "LRUReplacementPolicy",
    "PrefixAwareLRUReplacementPolicy",
    "ReplacementPolicy",
]
