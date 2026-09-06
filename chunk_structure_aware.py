"""
Structure-aware chunking for RAG pipelines over academic (arXiv) papers.

Strategy:
    1. Split the document into sections using its section headers
       (Abstract, Introduction, Method, Results, etc.), tolerating common
       numbering styles ("1 Introduction", "2. Methods", "II. Results").
    2. Any section that fits within `chunk_size` tokens becomes a single
       chunk.
    3. Any section longer than `chunk_size` tokens is split along
       paragraph breaks ("\n\n") into ~chunk_size-token chunks with
       chunk_overlap tokens of overlap between consecutive chunks. A
       single paragraph that itself exceeds chunk_size (e.g. a dense,
       unbroken benchmark table) is hard-split on raw token windows as a
       last resort.
    4. If no recognizable headers are found at all, the whole document is
       treated as one section and chunked the same way.

Usage as a library:
    from chunk_structure_aware import structure_aware_chunk
    chunks = structure_aware_chunk(text, paper_id="2305.12345")

Usage as a CLI (for spot-checking one file):
    python chunk_structure_aware.py data/raw_text/2305.12345.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import tiktoken

ENCODING_NAME = "cl100k_base"
_encoding = tiktoken.get_encoding(ENCODING_NAME)

# Section header keywords this parser recognizes. Matching requires the
# *entire* (stripped) line to be one of these, optionally preceded by
# numbering ("1", "1.1", "II.") and followed by an optional colon — this
# keeps it from matching the word "introduction" mid-sentence.
_HEADER_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "model architecture",
    "architecture",
    "experimental setup",
    "experiments",
    "experiment",
    "results",
    "evaluation",
    "analysis",
    "ablation study",
    "ablation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
]

# Sections that are near-never the source of a real answer, and whose
# short, keyword-dense text (titles, author names, repo links, citation
# keywords) tends to score artificially high on cosine similarity relative
# to how informative it actually is. Excluded from indexing by default.
DEFAULT_EXCLUDE_SECTIONS = {"Preamble", "Acknowledgments", "Acknowledgements", "References"}

# Any chunk shorter than this (in tokens) is dropped as a second safety
# net, catching stray degenerate chunks that don't fall under a named
# excluded section but are still too short to contain a real answer.
DEFAULT_MIN_CHUNK_TOKENS = 20

_NUMBERING = r"(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?)\s+"
_HEADER_RE = re.compile(
    rf"^(?:{_NUMBERING})?"
    rf"({'|'.join(re.escape(w) for w in sorted(_HEADER_KEYWORDS, key=len, reverse=True))})"
    rf"\s*:?\s*$",
    re.IGNORECASE,
)


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split raw text into (header_label, section_text) pairs.

    Text before the first recognized header is labeled "Preamble" (title,
    authors, venue info, etc.). Empty sections are dropped.
    """
    lines = text.split("\n")
    raw_sections: List[Tuple[str, List[str]]] = []
    current_header = "Preamble"
    current_lines: List[str] = []

    for line in lines:
        match = _HEADER_RE.match(line.strip())
        if match:
            raw_sections.append((current_header, current_lines))
            current_header = match.group(1).title()
            current_lines = []
        else:
            current_lines.append(line)
    raw_sections.append((current_header, current_lines))

    sections = []
    for header, section_lines in raw_sections:
        section_text = "\n".join(section_lines).strip()
        if section_text:
            sections.append((header, section_text))
    return sections


def _split_long_section(
    section_text: str,
    chunk_size: int,
    chunk_overlap: int,
    section_start_offset: int,
) -> List[Tuple[str, int, int, int]]:
    """Split one over-long section into ~chunk_size-token chunks.

    Packs paragraphs (split on blank lines) greedily until the next
    paragraph would push the running total over `chunk_size`, then
    finalizes that chunk and carries the last `chunk_overlap` tokens
    forward as the start of the next chunk. A paragraph that alone
    exceeds `chunk_size` is hard-split on raw token windows.

    Returns (text, token_count, start_token, end_token) tuples, where the
    token offsets are document-global (section_start_offset + local
    position). Because overlapping windows revisit tokens, these offsets
    track approximate cumulative position rather than exact non-overlapping
    slices — the same convention used by the fixed-size chunker.
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    if not paragraphs:
        paragraphs = [section_text]

    results: List[Tuple[str, int, int, int]] = []
    local_cursor = 0
    accumulated_ids: List[int] = []
    has_new_content = False

    def flush() -> None:
        nonlocal accumulated_ids, local_cursor, has_new_content
        if not accumulated_ids or not has_new_content:
            return
        text = _encoding.decode(accumulated_ids)
        tok_count = len(accumulated_ids)
        start = local_cursor
        end = start + tok_count
        results.append((text, tok_count, start, end))

        overlap_ids = accumulated_ids[-chunk_overlap:] if chunk_overlap > 0 else []
        local_cursor = end - len(overlap_ids)
        accumulated_ids = list(overlap_ids)
        has_new_content = False

    for para in paragraphs:
        para_ids = _encoding.encode(para)

        if len(para_ids) > chunk_size:
            # Flush whatever's pending, then hard-split this oversized
            # paragraph on raw token windows (paragraph breaks don't help
            # here — typically a dense, unbroken table or equation block).
            flush()
            start = 0
            while start < len(para_ids):
                end = min(start + chunk_size, len(para_ids))
                window_ids = para_ids[start:end]
                tok_count = len(window_ids)
                global_start = local_cursor
                results.append(
                    (_encoding.decode(window_ids), tok_count,
                     global_start, global_start + tok_count)
                )
                if end == len(para_ids):
                    local_cursor = global_start + tok_count
                    break
                stride = chunk_size - chunk_overlap
                local_cursor += stride
                start += stride
            accumulated_ids = []
            has_new_content = False
            continue

        if has_new_content and len(accumulated_ids) + len(para_ids) > chunk_size:
            flush()

        accumulated_ids.extend(para_ids)
        has_new_content = True

    flush()

    return [
        (text, tok_count, section_start_offset + start, section_start_offset + end)
        for (text, tok_count, start, end) in results
    ]


def structure_aware_chunk(
    text: str,
    paper_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    exclude_sections: set = DEFAULT_EXCLUDE_SECTIONS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
) -> List[Dict]:
    """Chunk a paper's text by section, falling back to fixed-size
    splitting for any section longer than `chunk_size` tokens.

    Args:
        text: Raw document text (e.g. extracted paper text).
        paper_id: Identifier of the source document, used to build
            chunk_ids and tag each chunk back to its source.
        chunk_size: Target max tokens per chunk.
        chunk_overlap: Token overlap between consecutive chunks produced
            when splitting an over-long section.
        exclude_sections: Section header labels to skip entirely (never
            emitted as chunks). Defaults to boilerplate sections — title/
            author preambles, acknowledgments, references — that are
            rarely the source of a real answer but can score artificially
            high on similarity due to short, keyword-dense text. Pass an
            empty set to disable exclusion.
        min_chunk_tokens: Drop any chunk shorter than this many tokens,
            regardless of section, as a safety net for stray short chunks.
            Pass 0 to disable.

    Returns:
        A list of dicts, each with: chunk_id, paper_id, text, token_count,
        section_header, token_start, token_end. Returns [] for
        empty/whitespace-only text.

    Raises:
        ValueError: if chunk_size <= 0, chunk_overlap < 0, or
            chunk_overlap >= chunk_size.
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

    sections = _split_into_sections(text)
    if len(sections) == 1 and sections[0][0] == "Preamble":
        # No section headers were recognized anywhere in the document —
        # relabel so the output honestly reflects "no structure found"
        # rather than implying everything before an Abstract/Introduction.
        sections = [("Full Document", sections[0][1])]

    chunks: List[Dict] = []
    doc_offset = 0
    idx = 0

    for header, section_text in sections:
        section_ids = _encoding.encode(section_text)
        section_tok_count = len(section_ids)
        if section_tok_count == 0:
            continue

        is_excluded = header in exclude_sections

        if not is_excluded:
            if section_tok_count <= chunk_size:
                if section_tok_count >= min_chunk_tokens:
                    chunks.append(
                        {
                            "chunk_id": f"{paper_id}_structured_{idx}",
                            "paper_id": paper_id,
                            "text": section_text,
                            "token_count": section_tok_count,
                            "section_header": header,
                            "token_start": doc_offset,
                            "token_end": doc_offset + section_tok_count,
                        }
                    )
                    idx += 1
            else:
                sub_chunks = _split_long_section(
                    section_text, chunk_size, chunk_overlap, doc_offset
                )
                for sub_text, tok_count, start, end in sub_chunks:
                    if tok_count < min_chunk_tokens:
                        continue
                    chunks.append(
                        {
                            "chunk_id": f"{paper_id}_structured_{idx}",
                            "paper_id": paper_id,
                            "text": sub_text,
                            "token_count": tok_count,
                            "section_header": header,
                            "token_start": start,
                            "token_end": end,
                        }
                    )
                    idx += 1

        # Always advance doc_offset by the section's true token count,
        # even when excluded, so token positions for later sections stay
        # aligned to the document rather than shifting when noise is dropped.
        doc_offset += section_tok_count

    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chunk_structure_aware.py <path_to_text_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    sample_text = input_path.read_text(encoding="utf-8")
    paper_id = input_path.stem

    result_chunks = structure_aware_chunk(sample_text, paper_id=paper_id)

    print(f"Source file: {input_path}")
    print(f"Paper ID: {paper_id}")
    print(f"Number of chunks: {len(result_chunks)}\n")

    for chunk in result_chunks:
        preview = chunk["text"][:120].replace("\n", " ")
        print(
            f"[{chunk['chunk_id']}] section={chunk['section_header']!r} "
            f"tokens {chunk['token_start']}-{chunk['token_end']} "
            f"({chunk['token_count']} tokens)"
        )
        print(f"  preview: {preview}...\n")