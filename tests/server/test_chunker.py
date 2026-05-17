# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.server.http.chunker import Chunker, hash_tokens


def test_pad_to_chunk_boundary_extends_tail_without_mutating_input() -> None:
    chunker = Chunker(block_tokens=4, chunk_blocks=2)
    tokens = [1, 2, 3, 4, 5]

    padded = chunker.pad_to_chunk_boundary(tokens, pad_token=0)

    assert tokens == [1, 2, 3, 4, 5]
    assert padded == [1, 2, 3, 4, 5, 0, 0, 0]


def test_chunk_pads_tail_when_pad_token_is_provided() -> None:
    chunker = Chunker(block_tokens=4, chunk_blocks=1)

    chunks = chunker.chunk([1, 2, 3, 4, 5], pad_token=0)

    assert [chunk.tokens for chunk in chunks] == [[1, 2, 3, 4], [5, 0, 0, 0]]
    assert [chunk.chunk_key for chunk in chunks] == [
        hash_tokens([1, 2, 3, 4]),
        hash_tokens([5, 0, 0, 0]),
    ]


def test_single_chunk_pads_whole_segment_to_block_boundary() -> None:
    chunker = Chunker(block_tokens=4, chunk_blocks=8)

    chunk = chunker.single_chunk([1, 2, 3, 4, 5], pad_token=0)

    assert chunk.tokens == [1, 2, 3, 4, 5, 0, 0, 0]
    assert chunk.chunk_key == hash_tokens([1, 2, 3, 4, 5, 0, 0, 0])
