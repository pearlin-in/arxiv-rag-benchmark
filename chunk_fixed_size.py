"""
Fixed-size token chunking for RAG pipelines.

Splits raw text into overlapping token-window chunks using tiktoken's
`cl100k_base` encoding (the same encoding family used by GPT-4-class
models; it's a reasonable, widely-available stand-in for "roughly how
many tokens will this cost/represent" even though your actual embedding
model may tokenize slightly differently).

Usage as a library:
    from chunk_fixed_size import fixed_size_chunk
    chunks = fixed_size_chunk(text, paper_id="2305.12345")

Usage as a CLI (for spot-checking one file):
    python chunk_fixed_size.py data/raw_text/2305.12345.txt
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import tiktoken

ENCODING_NAME = "cl100k_base"
_encoding = tiktoken.get_encoding(ENCODING_NAME)


@dataclass
class Chunk:
    """A single fixed-size chunk of a source document."""

    chunk_id: str       # e.g. "2305.12345_fixed_0"
    paper_id: str       # e.g. "2305.12345"
    text: str           # decoded chunk text
    token_count: int    # number of tokens in this chunk
    start_token: int    # inclusive start index into the full token sequence
    end_token: int      # exclusive end index into the full token sequence

    def to_dict(self) -> dict:
        return asdict(self)


def fixed_size_chunk(
    text: str,
    paper_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """Split `text` into overlapping fixed-size token chunks.

    Args:
        text: Raw document text (e.g. extracted paper text).
        paper_id: Identifier of the source document, used to build
            chunk_ids and to tag each chunk back to its source.
        chunk_size: Target number of tokens per chunk.
        chunk_overlap: Number of tokens each chunk overlaps with the
            previous one, so context isn't lost at chunk boundaries.

    Returns:
        A list of Chunk objects covering the full text in order. Returns
        an empty list if `text` is empty/whitespace-only.

    Raises:
        ValueError: if chunk_size <= 0, chunk_overlap < 0, or
            chunk_overlap >= chunk_size (which would either loop forever
            or produce a non-advancing window).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}), or the window never advances"
        )

    if not text or not text.strip():
        return []

    tokens = _encoding.encode(text)
    total_tokens = len(tokens)
    stride = chunk_size - chunk_overlap

    # Text shorter than one chunk: return it as a single chunk.
    if total_tokens <= chunk_size:
        return [
            Chunk(
                chunk_id=f"{paper_id}_fixed_0",
                paper_id=paper_id,
                text=_encoding.decode(tokens),
                token_count=total_tokens,
                start_token=0,
                end_token=total_tokens,
            )
        ]

    chunks: List[Chunk] = []
    start = 0
    index = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        window_tokens = tokens[start:end]

        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}_fixed_{index}",
                paper_id=paper_id,
                text=_encoding.decode(window_tokens),
                token_count=len(window_tokens),
                start_token=start,
                end_token=end,
            )
        )

        # Stop once this chunk reached the end of the text (avoids a
        # tiny trailing chunk that's mostly a re-hash of the overlap).
        if end == total_tokens:
            break

        start += stride
        index += 1

    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chunk_fixed_size.py <path_to_text_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    sample_text = input_path.read_text(encoding="utf-8")
    paper_id = input_path.stem

    result_chunks = fixed_size_chunk(sample_text, paper_id=paper_id)

    print(f"Source file: {input_path}")
    print(f"Paper ID: {paper_id}")
    print(f"Total tokens (encoding={ENCODING_NAME}): "
          f"{len(_encoding.encode(sample_text))}")
    print(f"Number of chunks: {len(result_chunks)}\n")

    for chunk in result_chunks[:5]:
        preview = chunk.text[:150].replace("\n", " ")
        print(f"[{chunk.chunk_id}] tokens {chunk.start_token}-{chunk.end_token} "
              f"({chunk.token_count} tokens)")
        print(f"  preview: {preview}...\n")

    if len(result_chunks) > 5:
        print(f"... and {len(result_chunks) - 5} more chunk(s) not shown.")