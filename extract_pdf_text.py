"""
Extracts raw text from every PDF in data/raw_pdfs/ and saves it as a .txt
file (same basename) in data/raw_text/. Logs any files that fail to extract.

Uses PyMuPDF (imported as `fitz`) rather than pypdf: for academic PDFs with
multi-column layouts, PyMuPDF is generally more reliable at preserving
reading order and tends to choke less often on malformed PDF structures.

Usage:
    pip install pymupdf
    python extract_pdf_text.py
"""

from pathlib import Path

import fitz  # PyMuPDF

INPUT_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/raw_text")
FAILURE_LOG = Path("data/extraction_failures.log")


def extract_text(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF, joined with double newlines."""
    doc = fitz.open(pdf_path)
    try:
        pages_text = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages_text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(pdf_paths)} PDFs in {INPUT_DIR}")

    succeeded = 0
    failures = []

    for pdf_path in pdf_paths:
        txt_path = OUTPUT_DIR / (pdf_path.stem + ".txt")

        if txt_path.exists():
            print(f"[skip, already extracted] {pdf_path.name}")
            succeeded += 1
            continue

        try:
            text = extract_text(pdf_path)
            if not text.strip():
                raise ValueError("extracted text is empty (possibly a scanned/image-only PDF)")
            txt_path.write_text(text, encoding="utf-8")
            print(f"[ok] {pdf_path.name} -> {txt_path.name} ({len(text)} chars)")
            succeeded += 1
        except Exception as e:
            print(f"[FAILED] {pdf_path.name}: {e}")
            failures.append((pdf_path.name, str(e)))

    # Write a log of failures so you can revisit them later.
    if failures:
        with open(FAILURE_LOG, "w", encoding="utf-8") as f:
            for filename, error in failures:
                f.write(f"{filename}\t{error}\n")
        print(f"\n{len(failures)} file(s) failed. Details logged to {FAILURE_LOG.resolve()}")
    else:
        print("\nNo failures.")

    print(f"Extracted {succeeded}/{len(pdf_paths)} PDFs successfully.")
    print(f"Text files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()