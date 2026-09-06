"""
BM25 keyword retrieval over a chunk JSONL file (chunks_fixed.jsonl or
chunks_structured.jsonl), using rank_bm25. This is the sparse/keyword
counterpart to your dense Chroma retrieval — good at exact terms,
acronyms, and literal phrase matches that embeddings can dilute.

Usage as a library:
    from bm25_retriever import BM25Retriever
    retriever = BM25Retriever("data/chunks_fixed.jsonl")
    results = retriever.search("Low-Rank Adaptation", top_k=5)

Usage as a CLI:
    python bm25_retriever.py "Low-Rank Adaptation" data/chunks_fixed.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization for BM25 indexing/queries.

    Simple on purpose: BM25 doesn't need the tokenizer to match your
    embedding model's tokenizer, just to be consistent between indexing
    and querying.
    """
    return _TOKEN_RE.findall(text.lower())


def load_chunks(path: Path) -> List[Dict]:
    """Load chunks from a JSONL file, one JSON object per line."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[skip] malformed JSON on line {line_num}: {e}")
    return chunks


class BM25Retriever:
    """Wraps a BM25 index built from a chunk JSONL file, keeping the
    original chunk dicts aligned to BM25's internal document order so
    results can be mapped back to full chunk metadata.
    """

    def __init__(self, chunks_path: str | Path):
        self.chunks_path = Path(chunks_path)
        self.chunks: List[Dict] = load_chunks(self.chunks_path)
        if not self.chunks:
            raise ValueError(f"No chunks loaded from {self.chunks_path}")

        tokenized_corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top-k chunks for `query`, each with a `bm25_score`
        and `rank` (1-indexed) added. Chunks with a zero score (no term
        overlap at all) are excluded, since a "top-k" of pure noise isn't
        useful for fusion or display.
        """
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            if scores[idx] <= 0:
                break  # scores are sorted descending; once we hit 0, stop
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            chunk["rank"] = rank
            results.append(chunk)

        return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python bm25_retriever.py "your query" <path_to_chunks.jsonl>')
        sys.exit(1)

    query = sys.argv[1]
    chunks_path = sys.argv[2]

    print(f"Loading and indexing chunks from {chunks_path}...")
    retriever = BM25Retriever(chunks_path)
    print(f"Indexed {len(retriever.chunks)} chunks.\n")

    results = retriever.search(query, top_k=5)

    print(f"Query: {query!r}")
    print(f"Top {len(results)} BM25 results:\n")

    for r in results:
        preview = r["text"][:200].replace("\n", " ")
        print(f"#{r['rank']}  chunk_id={r['chunk_id']}  bm25_score={r['bm25_score']:.4f}")
        print(f"      paper_id={r.get('paper_id')}", end="")
        if "section_header" in r:
            print(f"  section={r['section_header']}", end="")
        print()
        print(f"      text: {preview}...\n")