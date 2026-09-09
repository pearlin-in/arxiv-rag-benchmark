"""
Scans your chunked JSONL files (chunks_fixed.jsonl, chunks_structured.jsonl)
for a keyword or phrase, and prints the matching chunk_id + paper_id + a
snippet of surrounding text, so you can confirm ground truth for your
evaluation questions without manually grepping through files.

USAGE EXAMPLES
---------------
# Search both chunk files for a keyword
python find_chunk_id.py "reciprocal rank fusion"

# Search only the structured chunks
python find_chunk_id.py "lambda" --files data/chunks_structured.jsonl

# Narrow to one paper (e.g. you already know the arxiv id)
python find_chunk_id.py "perplexity" --paper-id 2312.12345

# Show more surrounding context per match
python find_chunk_id.py "4-bit quantization" --context 400

# Only show the first 5 matches (default is 10)
python find_chunk_id.py "KV cache" --limit 5

NOTES ON SCHEMA
----------------
This script tries a few common field name variants automatically:
  - chunk id:  "chunk_id", "id"
  - paper id:  "paper_id", "arxiv_id", "id" (paper-level)
  - text:      "text", "chunk_text", "content"
  - title:     "title", "paper_title", "source_title"

If your JSONL uses different field names, edit the FIELD_CANDIDATES
dict below to match — or just tell Claude your exact schema and ask
for an updated version of this script.
"""

import argparse
import json
import re
import sys
from pathlib import Path

FIELD_CANDIDATES = {
    "chunk_id": ["chunk_id", "id"],
    "paper_id": ["paper_id", "arxiv_id"],
    "text": ["text", "chunk_text", "content"],
    "title": ["title", "paper_title", "source_title"],
}

DEFAULT_FILES = ["data/chunks_fixed.jsonl", "data/chunks_structured.jsonl"]


def get_field(record: dict, field: str):
    for candidate in FIELD_CANDIDATES[field]:
        if candidate in record:
            return record[candidate]
    return None


def make_snippet(text: str, keyword: str, context_chars: int) -> str:
    match = re.search(re.escape(keyword), text, re.IGNORECASE)
    if not match:
        return text[:context_chars] + ("..." if len(text) > context_chars else "")
    start = max(0, match.start() - context_chars // 2)
    end = min(len(text), match.end() + context_chars // 2)
    snippet = text[start:end]
    # Bold-ish highlight using brackets since terminals vary in color support
    snippet = re.sub(
        re.escape(keyword),
        lambda m: f"[[{m.group(0)}]]",
        snippet,
        flags=re.IGNORECASE,
    )
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def search_file(path: Path, keyword: str, paper_id_filter: str, context_chars: int, limit: int):
    if not path.exists():
        print(f"  (skipping — file not found: {path})")
        return 0

    matches_found = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  (warning: could not parse line {line_num} in {path})")
                continue

            text = get_field(record, "text") or ""
            paper_id = get_field(record, "paper_id") or "UNKNOWN_PAPER_ID"
            chunk_id = get_field(record, "chunk_id")
            if chunk_id is None:
                chunk_id = f"line_{line_num}"
            title = get_field(record, "title") or ""

            if paper_id_filter and paper_id_filter.lower() not in str(paper_id).lower():
                continue

            if keyword.lower() in text.lower():
                matches_found += 1
                print(f"\n--- Match {matches_found} ---")
                print(f"  file:      {path}")
                print(f"  chunk_id:  {chunk_id}")
                print(f"  paper_id:  {paper_id}")
                if title:
                    print(f"  title:     {title}")
                print(f"  snippet:   {make_snippet(text, keyword, context_chars)}")

                if matches_found >= limit:
                    print(f"\n  (stopping at --limit {limit} matches for this file; refine your keyword for more precision)")
                    break

    return matches_found


def main():
    parser = argparse.ArgumentParser(description="Find chunk IDs by keyword for eval question ground-truth.")
    parser.add_argument("keyword", help="Keyword or phrase to search for (case-insensitive)")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="JSONL files to search")
    parser.add_argument("--paper-id", default=None, help="Restrict search to chunks from this paper/arxiv id")
    parser.add_argument("--context", type=int, default=250, help="Characters of context to show around the match")
    parser.add_argument("--limit", type=int, default=10, help="Max matches to print per file")
    args = parser.parse_args()

    print(f"Searching for: {args.keyword!r}")
    if args.paper_id:
        print(f"Restricted to paper_id containing: {args.paper_id}")

    total = 0
    for file_str in args.files:
        path = Path(file_str)
        print(f"\n=== {path} ===")
        total += search_file(path, args.keyword, args.paper_id, args.context, args.limit)

    print(f"\nTotal matches: {total}")
    if total == 0:
        print("No matches. Try a shorter/simpler keyword, check spelling, or drop --paper-id.")
    elif total > 5:
        print("Tip: many matches — narrow with --paper-id or a more specific phrase before you pick a ground-truth chunk.")


if __name__ == "__main__":
    sys.exit(main())