"""
Phase 3 (structure-aware chunks): embeds every chunk in
data/chunks_structured.jsonl using BAAI/bge-base-en-v1.5 and stores the
embeddings in a persistent ChromaDB collection ("structured_chunks") using
cosine similarity.

This mirrors build_fixed_index.py exactly, except it reads from
chunks_structured.jsonl, writes to the "structured_chunks" collection, and
additionally carries the `section_header` field into each chunk's metadata
(since structure-aware chunks have that field and fixed-size ones don't).

Usage:
    pip install sentence-transformers chromadb
    python build_structured_index.py
"""

import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks_structured.jsonl")
CHROMA_DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "structured_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 256


def load_chunks(path: Path) -> list[dict]:
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


def clean_metadata(chunk: dict) -> dict:
    """Build a metadata dict for Chroma with the required fields, ensuring
    no None values sneak in (Chroma metadata must be str/int/float/bool).
    """
    metadata = {
        "paper_id": chunk.get("paper_id"),
        "token_count": chunk.get("token_count"),
        "token_start": chunk.get("token_start"),
        "token_end": chunk.get("token_end"),
        "section_header": chunk.get("section_header"),
    }

    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            # Fall back to a safe default per expected type rather than
            # ever writing None into Chroma metadata.
            cleaned[key] = -1 if key in ("token_count", "token_start", "token_end") else ""
        else:
            cleaned[key] = value
    return cleaned


def main():
    start_time = time.time()

    if not CHUNKS_PATH.exists():
        print(f"Chunks file not found: {CHUNKS_PATH.resolve()}")
        return

    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        print("No chunks to embed. Exiting.")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Initializing persistent ChromaDB at {CHROMA_DB_DIR}...")
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = chunks[batch_start:batch_end]

        ids = [chunk["chunk_id"] for chunk in batch]
        documents = [chunk["text"] for chunk in batch]
        metadatas = [clean_metadata(chunk) for chunk in batch]

        embeddings = model.encode(
            documents,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        elapsed = time.time() - start_time
        print(
            f"Batch {batch_idx + 1}/{num_batches}: embedded and stored "
            f"chunks {batch_start}-{batch_end - 1} of {total} "
            f"(elapsed: {elapsed:.1f}s)"
        )

    total_elapsed = time.time() - start_time
    print(f"\nDone. Stored {total} chunks in collection '{COLLECTION_NAME}'.")
    print(f"ChromaDB persisted at: {CHROMA_DB_DIR.resolve()}")
    print(f"Total elapsed time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()