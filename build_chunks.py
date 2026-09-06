"""
Runs both chunking strategies (fixed-size and structure-aware) over every
extracted paper in data/raw_text/, and saves each strategy's output to its
own JSONL file so they can be compared later.

Output:
    data/chunks_fixed.jsonl       (one JSON object per line, from fixed_size_chunk)
    data/chunks_structured.jsonl  (one JSON object per line, from structure_aware_chunk)

Every chunk is already tagged with its source paper via the `paper_id`
field (derived from the .txt filename), so both files can be grouped/
joined back to the metadata CSV by paper_id.

Requires chunk_fixed_size.py and chunk_structure_aware.py to be in the
same folder as this script.

Usage:
    python build_chunks.py
"""

import json
from pathlib import Path
from dataclasses import asdict

from chunk_fixed_size import fixed_size_chunk
from chunk_structure_aware import structure_aware_chunk

INPUT_DIR = Path("data/raw_text")
FIXED_OUTPUT = Path("data/chunks_fixed.jsonl")
STRUCTURED_OUTPUT = Path("data/chunks_structured.jsonl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main():
    txt_paths = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_paths:
        print(f"No .txt files found in {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(txt_paths)} extracted text files in {INPUT_DIR}")

    FIXED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fixed_count = 0
    structured_count = 0
    failed_files = []

    with open(FIXED_OUTPUT, "w", encoding="utf-8") as fixed_f, \
         open(STRUCTURED_OUTPUT, "w", encoding="utf-8") as structured_f:

        for txt_path in txt_paths:
            paper_id = txt_path.stem  # filename without .txt, e.g. "2305.12345"

            try:
                text = txt_path.read_text(encoding="utf-8")

                fixed_chunks = fixed_size_chunk(
                    text, paper_id=paper_id,
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                )
                for chunk in fixed_chunks:
                    fixed_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                fixed_count += len(fixed_chunks)

                structured_chunks = structure_aware_chunk(
                    text, paper_id=paper_id,
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                )
                for chunk in structured_chunks:
                    structured_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                structured_count += len(structured_chunks)

                print(f"[ok] {paper_id}: {len(fixed_chunks)} fixed chunks, "
                      f"{len(structured_chunks)} structured chunks")

            except Exception as e:
                print(f"[FAILED] {paper_id}: {e}")
                failed_files.append((paper_id, str(e)))

    print(f"\nDone. {len(txt_paths) - len(failed_files)}/{len(txt_paths)} papers processed.")
    print(f"Fixed-size chunks:      {fixed_count} total -> {FIXED_OUTPUT.resolve()}")
    print(f"Structure-aware chunks: {structured_count} total -> {STRUCTURED_OUTPUT.resolve()}")

    if failed_files:
        print(f"\n{len(failed_files)} file(s) failed:")
        for paper_id, error in failed_files:
            print(f"  {paper_id}: {error}")


if __name__ == "__main__":
    main()