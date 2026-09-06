"""
Hybrid retrieval: combines dense vector search (Chroma) and BM25 keyword
search using Reciprocal Rank Fusion (RRF), so exact terms/acronyms that
dense search dilutes (see: "what does LoRA stand for?") can still surface
via keyword overlap, while conceptual queries still benefit from dense
search's semantic matching.

RRF, briefly: each retriever produces a ranked list. A chunk's fused score
is the sum, across retrievers that returned it, of 1 / (k + rank), where
rank is that chunk's 1-indexed position in that retriever's list and k is
a constant (60 is the standard default from the original RRF paper) that
flattens the influence of any single retriever's exact rank positions.
Chunks near the top of *either* list score well; chunks near the top of
*both* score best.

Usage as a library:
    from hybrid_retrieval import hybrid_search
    results = hybrid_search(
        "what does LoRA stand for?",
        collection_name="structured_chunks",
        chunks_path="data/chunks_structured.jsonl",
    )

Usage as a CLI:
    python hybrid_retrieval.py "what does LoRA stand for?" structured_chunks data/chunks_structured.jsonl
"""

from __future__ import annotations

import sys
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from bm25_retriever import BM25Retriever

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHROMA_DB_DIR = "data/chroma_db"
RRF_K = 60  # standard constant from the original RRF paper

# Caches so repeated calls in one process don't reload the model / rebuild
# the BM25 index every time. Keyed by chunks_path since a process might
# want to query both the fixed and structured collections in one run.
_embedding_model = None
_chroma_client = None
_bm25_retrievers: Dict[str, BM25Retriever] = {}


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _chroma_client


def _get_bm25_retriever(chunks_path: str) -> BM25Retriever:
    if chunks_path not in _bm25_retrievers:
        _bm25_retrievers[chunks_path] = BM25Retriever(chunks_path)
    return _bm25_retrievers[chunks_path]


def dense_search(query: str, collection_name: str, top_k: int) -> List[Dict]:
    """Dense retrieval via Chroma, returning chunks with `rank` (1-indexed)."""
    model = _get_embedding_model()
    client = _get_chroma_client()
    collection = client.get_collection(name=collection_name)

    query_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        chunk = {
            "chunk_id": chunk_id,
            "text": text,
            "paper_id": metadata.get("paper_id"),
            "similarity": 1 - distance,
            "rank": rank,
        }
        if "section_header" in metadata:
            chunk["section_header"] = metadata["section_header"]
        chunks.append(chunk)

    return chunks


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    rrf_k: int = RRF_K,
) -> List[Dict]:
    """Fuse multiple ranked lists of chunks (each chunk dict must have
    `chunk_id` and `rank`) into one ranked list by RRF score.

    Returns chunks sorted by descending fused score, each annotated with
    `rrf_score` and the per-retriever ranks it appeared at.
    """
    fused: Dict[str, Dict] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        for chunk in ranked_list:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in fused:
                # Keep the original chunk data (text, paper_id, etc.) from
                # whichever retriever we see it in first.
                fused[chunk_id] = {**chunk, "rrf_score": 0.0, "retriever_ranks": {}}

            rrf_contribution = 1.0 / (rrf_k + chunk["rank"])
            fused[chunk_id]["rrf_score"] += rrf_contribution
            fused[chunk_id]["retriever_ranks"][f"retriever_{list_idx}"] = chunk["rank"]

    return sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)


def hybrid_search(
    query: str,
    collection_name: str,
    chunks_path: str,
    top_k: int = 5,
    dense_k: int = 20,
    bm25_k: int = 20,
    rrf_k: int = RRF_K,
) -> List[Dict]:
    """Run dense + BM25 retrieval and fuse them with RRF.

    dense_k / bm25_k control how deep each individual retriever searches
    before fusion (wider than `top_k` so RRF has enough candidates to work
    with); `top_k` controls how many fused results are returned.
    """
    dense_results = dense_search(query, collection_name, top_k=dense_k)

    bm25_retriever = _get_bm25_retriever(chunks_path)
    bm25_results = bm25_retriever.search(query, top_k=bm25_k)

    fused = reciprocal_rank_fusion([dense_results, bm25_results], rrf_k=rrf_k)

    return fused[:top_k]


def hybrid_search_with_rewrite(
    query: str,
    collection_name: str,
    chunks_path: str,
    top_k: int = 5,
    dense_k: int = 20,
    bm25_k: int = 20,
    rrf_k: int = RRF_K,
) -> Dict:
    """Rewrite `query` with an LLM before running hybrid search.

    Targets acronym/definition questions where the natural-language query
    shares no tokens with the answer (e.g. "what does LoRA stand for?" vs
    a chunk containing "Low-Rank Adaptation") — expanding the query first
    gives both BM25 and dense search something concrete to match on.

    Falls back to the original, un-rewritten query if the rewrite call
    fails for any reason (e.g. API/network error), so a rewrite failure
    never breaks retrieval outright — it just degrades to plain hybrid
    search.

    Returns a dict with: original_query, rewritten_query, results.
    """
    from query_rewrite import rewrite_query  # lazy import: only needed here,
    # so plain hybrid_search() doesn't require google-generativeai/dotenv.

    try:
        rewritten_query = rewrite_query(query)
    except Exception as e:
        print(f"[query rewrite failed, falling back to original query] {e}")
        rewritten_query = query

    results = hybrid_search(
        rewritten_query, collection_name, chunks_path,
        top_k=top_k, dense_k=dense_k, bm25_k=bm25_k, rrf_k=rrf_k,
    )

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "results": results,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    use_rewrite = "--rewrite" in args
    if use_rewrite:
        args.remove("--rewrite")

    if len(args) != 3:
        print(
            'Usage: python hybrid_retrieval.py "your query" '
            "<collection_name> <path_to_chunks.jsonl> [--rewrite]"
        )
        sys.exit(1)

    query, collection_name, chunks_path = args

    if use_rewrite:
        rewrite_result = hybrid_search_with_rewrite(query, collection_name, chunks_path, top_k=5)
        print(f"Original query:  {rewrite_result['original_query']!r}")
        print(f"Rewritten query: {rewrite_result['rewritten_query']!r}\n")
        results = rewrite_result["results"]
    else:
        results = hybrid_search(query, collection_name, chunks_path, top_k=5)

    print(f"Collection: {collection_name}  |  BM25 source: {chunks_path}")
    print(f"Top {len(results)} hybrid (RRF-fused) results:\n")

    for i, r in enumerate(results, start=1):
        preview = r["text"][:200].replace("\n", " ")
        print(f"#{i}  chunk_id={r['chunk_id']}  rrf_score={r['rrf_score']:.5f}  "
              f"ranks={r['retriever_ranks']}")
        print(f"      paper_id={r.get('paper_id')}", end="")
        if "section_header" in r:
            print(f"  section={r['section_header']}", end="")
        print()
        print(f"      text: {preview}...\n")