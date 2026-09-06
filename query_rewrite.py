"""
Query rewriting: expands a natural-language question into a keyword-rich
search query before it hits retrieval, using Gemini. This targets the
specific failure mode you found — "what does LoRA stand for?" shares no
tokens with "Low-Rank Adaptation", so neither BM25 nor dense search (nor
their RRF fusion) had any way to reward the chunk that actually defines
it. Rewriting the query to include the expansion terms up front gives
both retrievers something to match on.

Usage as a library:
    from query_rewrite import rewrite_query
    expanded = rewrite_query("what does LoRA stand for?")
    # -> "LoRA Low-Rank Adaptation full name definition parameter-efficient fine-tuning"

Usage as a CLI:
    python query_rewrite.py "what does LoRA stand for?"
"""

from __future__ import annotations

import os
import sys
import time

import google.generativeai as genai
from dotenv import load_dotenv

GEMINI_MODEL_NAME = "gemini-3.6-flash"
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 5

_gemini_configured = False

_REWRITE_PROMPT_TEMPLATE = """You are a search query optimizer for a hybrid keyword + vector search engine over a corpus of academic papers on efficient LLMs (topics: quantization, pruning, distillation, LoRA, KV cache).

Rewrite the question below into an expanded search query optimized for retrieval:
- If the question mentions an acronym or abbreviation, include both the acronym AND its full expanded name/term.
- Add a few closely related technical terms that a relevant passage would likely contain.
- Do NOT change the meaning of the question or invent unrelated content.
- Do NOT answer the question — only rewrite it as a search query.
- Return ONLY the rewritten query on a single line. No quotes, no explanation, no preamble.

Question: {question}

Rewritten query:"""


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


def _call_gemini_with_retry(prompt: str) -> str:
    """Call Gemini, retrying with exponential backoff on rate-limit errors."""
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


def rewrite_query(question: str) -> str:
    """Expand `question` into a keyword-rich search query via Gemini.

    Returns the original question unchanged if Gemini returns an empty
    or whitespace-only response (fails safe rather than sending an empty
    query to retrieval). Raises if the API call itself fails after
    retries — callers should decide whether to catch and fall back to
    the original query (see hybrid_retrieval.hybrid_search_with_rewrite
    for that pattern).
    """
    prompt = _REWRITE_PROMPT_TEMPLATE.format(question=question)
    raw_response = _call_gemini_with_retry(prompt)

    rewritten = raw_response.strip().strip('"').strip("'").strip()
    if not rewritten:
        return question

    return rewritten


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python query_rewrite.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    rewritten = rewrite_query(question)

    print(f"Original:  {question}")
    print(f"Rewritten: {rewritten}")