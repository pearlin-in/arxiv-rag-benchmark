"""
Quick retrieval sanity-check: embeds a text query with the same model used
to build the index, queries a given Chroma collection, and prints the
top-5 most similar chunks with their similarity scores.

Usage:
    python test_retrieval.py "what does LoRA stand for?" fixed_chunks
    python test_retrieval.py "what does LoRA stand for?" structured_chunks
    python test_retrieval.py "what does LoRA stand for?"   # defaults to fixed_chunks
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = "data/chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 5


def main():
    if len(sys.argv) < 2:
        print('Usage: python test_retrieval.py "your query here" [collection_name]')
        sys.exit(1)

    query = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "fixed_chunks"

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Connecting to ChromaDB at {CHROMA_DB_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Could not open collection '{collection_name}': {e}")
        print("Has it been built yet? (build_fixed_index.py / build_structured_index.py)")
        sys.exit(1)

    query_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nQuery: {query!r}")
    print(f"Collection: {collection_name}")
    print(f"Top {len(ids)} results:\n")

    for rank, (chunk_id, doc_text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        # Chroma's cosine space returns a distance (0 = identical); convert
        # to a similarity score (1 = identical, closer to 0 = dissimilar)
        # so it reads intuitively.
        similarity = 1 - distance

        preview = doc_text[:300].replace("\n", " ")
        print(f"#{rank}  chunk_id={chunk_id}  similarity={similarity:.4f}")
        print(f"      paper_id={metadata.get('paper_id')}", end="")
        if "section_header" in metadata:
            print(f"  section={metadata.get('section_header')}", end="")
        print()
        print(f"      text: {preview}...\n")


if __name__ == "__main__":
    main()