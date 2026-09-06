"""
Inspect full chunk text from a Chroma collection to debug retrieval contents.
Usage:
    python inspect_chunks.py "What does LoRA stand for?" structured_chunks --top_k 5
"""

import argparse
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = Path("data/chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def inspect_collection_chunks(query_text: str, collection_name: str, top_k: int = 5):
    if not CHROMA_DB_DIR.exists():
        print(f"ChromaDB directory not found at {CHROMA_DB_DIR.resolve()}")
        return

    print(f"\n[Query]: {query_text}")
    print(f"[Collection]: {collection_name}")
    print("=" * 80)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode([query_text], convert_to_numpy=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Collection '{collection_name}' not found: {e}")
        return

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    if not results["ids"] or not results["ids"][0]:
        print("No results returned.")
        return

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance

        print(f"Rank {i + 1} | Similarity: {similarity:.4f} | Chunk ID: {chunk_id}")
        print(f"Metadata: {meta}")
        print("-" * 80)
        print(doc)
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect full retrieved chunk text for debugging.")
    parser.add_argument("query", type=str, help="The search query text.")
    parser.add_argument(
        "collection", 
        type=str, 
        default="structured_chunks", 
        nargs="?",
        choices=["fixed_chunks", "structured_chunks"], 
        help="ChromaDB collection to query."
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")
    
    args = parser.parse_args()
    inspect_collection_chunks(args.query, args.collection, args.top_k)