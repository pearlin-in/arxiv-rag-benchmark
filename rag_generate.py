"""
Retrieval-augmented generation: retrieves top-k chunks from a Chroma
collection, builds a prompt combining the question and retrieved chunks,
and calls the Gemini API to generate an answer that cites which paper(s)
it drew from.

Usage as a library:
    from rag_generate import generate_answer
    result = generate_answer("What does LoRA stand for?", collection_name="fixed_chunks")
    print(result["answer"])
    print(result["sources"])

Usage as a CLI:
    python rag_generate.py "What does LoRA stand for?" fixed_chunks
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = "data/chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
GEMINI_MODEL_NAME = "gemini-3.6-flash"
TOP_K = 5
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 5  # backs off as base_delay * 2^attempt on rate limits

# Lazily-initialized singletons so repeated calls in one process don't
# reload the embedding model or reconfigure the API every time.
_embedding_model = None
_chroma_client = None
_gemini_configured = False


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


def _ensure_gemini_configured() -> None:
    global _gemini_configured
    if _gemini_configured:
        return
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Make sure a .env file with "
            "GEMINI_API_KEY=your-key-here exists in the working directory."
        )
    genai.configure(api_key=api_key)
    _gemini_configured = True


def retrieve_chunks(
    question: str,
    collection_name: str = "fixed_chunks",
    top_k: int = TOP_K,
) -> List[Dict]:
    """Embed `question` and retrieve the top-k most similar chunks.

    Returns a list of dicts with: chunk_id, text, paper_id, similarity,
    and section_header (only present for structure-aware collections).
    """
    model = _get_embedding_model()
    client = _get_chroma_client()
    collection = client.get_collection(name=collection_name)

    query_embedding = model.encode([question], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunk = {
            "chunk_id": chunk_id,
            "text": text,
            "paper_id": metadata.get("paper_id"),
            "similarity": 1 - distance,
        }
        if "section_header" in metadata:
            chunk["section_header"] = metadata["section_header"]
        chunks.append(chunk)

    return chunks


def build_prompt(question: str, chunks: List[Dict]) -> str:
    """Combine the question and retrieved chunks into a single prompt that
    instructs the model to cite which paper(s) it drew from.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        section_note = f" (section: {chunk['section_header']})" if "section_header" in chunk else ""
        context_blocks.append(
            f"[Source {i} | paper_id: {chunk['paper_id']}{section_note}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant answering questions using ONLY the provided excerpts from academic papers. Do not use outside knowledge.

Instructions:
- Answer the question using only the excerpts below.
- If the excerpts don't contain enough information to answer, say so explicitly.
- Cite the paper_id(s) you used to support your answer, e.g. "(paper_id: 2305.12345)".
- Be concise and precise, especially with numbers, names, and technical terms.

Excerpts:
{context_text}

Question: {question}

Answer:"""
    return prompt


def _call_gemini_with_retry(prompt: str) -> str:
    """Call the Gemini API, retrying with exponential backoff if the
    free-tier rate limit is hit, so a long eval run doesn't just crash.
    """
    _ensure_gemini_configured()
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "quota" in error_str or "rate" in error_str
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                print(f"  Rate limit hit, retrying in {delay}s... "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            raise

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


def generate_answer(
    question: str,
    collection_name: str = "fixed_chunks",
    top_k: int = TOP_K,
) -> Dict:
    """Full RAG step: retrieve top-k chunks, build a citation-instructing
    prompt, and call Gemini to generate an answer.

    Returns a dict with:
        answer: the generated answer text
        sources: list of {paper_id, chunk_id, similarity} for the chunks used
        prompt: the full prompt sent to Gemini (useful for debugging)
    """
    chunks = retrieve_chunks(question, collection_name=collection_name, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant chunks were found in the collection for this question.",
            "sources": [],
            "prompt": None,
        }

    prompt = build_prompt(question, chunks)
    answer_text = _call_gemini_with_retry(prompt)

    sources = [
        {"paper_id": c["paper_id"], "chunk_id": c["chunk_id"], "similarity": c["similarity"]}
        for c in chunks
    ]

    return {"answer": answer_text, "sources": sources, "prompt": prompt}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag_generate.py "your question here" [collection_name]')
        sys.exit(1)

    question = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "fixed_chunks"

    result = generate_answer(question, collection_name=collection_name)

    print(f"Question: {question}")
    print(f"Collection: {collection_name}\n")
    print("Answer:")
    print(result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  paper_id={s['paper_id']}  chunk_id={s['chunk_id']}  "
              f"similarity={s['similarity']:.4f}")