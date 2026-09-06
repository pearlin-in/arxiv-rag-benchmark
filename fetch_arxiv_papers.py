
"""
Pulls papers from arXiv (cs.CL and cs.LG) for a set of efficient-LLM terms,
downloads the PDFs, and saves a combined metadata CSV.
 
Usage:
    python fetch_arxiv_papers.py
 
Requires:
    pip install arxiv pandas
"""
 
import csv
import time
import urllib.request
from pathlib import Path
 
import arxiv
 
# ---- Config -----------------------------------------------------------
 
TERMS = ["quantization", "pruning", "distillation", "LoRA", "KV cache"]
CATEGORIES = ["cs.CL", "cs.LG"]
RESULTS_PER_TERM = 50          # aim for the 40-60 range the roadmap wants
OUTPUT_DIR = Path("data/raw_pdfs")
METADATA_CSV = Path("data/metadata.csv")
REQUEST_DELAY_SECONDS = 3.0    # be polite to the arXiv API
 
 
def build_query(term: str) -> str:
    """Build a query string like: (cat:cs.CL OR cat:cs.LG) AND all:"quantization" """
    cat_clause = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    # Quote multi-word terms (e.g. "KV cache") so they're matched as a phrase.
    term_clause = f'all:"{term}"'
    return f"({cat_clause}) AND {term_clause}"
 
 
def sanitize_filename(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")
 
 
def fetch_for_term(client: arxiv.Client, term: str) -> list[dict]:
    """Search arXiv for one term and return a list of metadata dicts.
 
    Iterates the results generator manually (instead of a plain `for` loop)
    so that if arXiv's API hiccups partway through a term (a known source of
    `UnexpectedEmptyPageError` in the `arxiv` package), we keep whatever
    records we already collected instead of losing the whole term.
    """
    print(f"\nSearching for term: {term!r}")
    search = arxiv.Search(
        query=build_query(term),
        max_results=RESULTS_PER_TERM,
        sort_by=arxiv.SortCriterion.Relevance,
    )
 
    records = []
    results_iter = client.results(search)
    while True:
        try:
            result = next(results_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f"  [stopped early for {term!r} after {len(records)} results: {e}]")
            break
 
        arxiv_id = result.get_short_id()  # e.g. "2305.12345v1"
        pdf_path = OUTPUT_DIR / f"{sanitize_filename(arxiv_id)}.pdf"
 
        if pdf_path.exists():
            print(f"  [skip, already downloaded] {arxiv_id}")
        else:
            try:
                print(f"  Downloading {arxiv_id}: {result.title[:70]}...")
                # Use urllib directly against result.pdf_url instead of relying on
                # library methods that vary between arxiv package versions.
                req = urllib.request.Request(
                    result.pdf_url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req) as response, open(pdf_path, "wb") as out_file:
                    out_file.write(response.read())
                time.sleep(REQUEST_DELAY_SECONDS)
            except Exception as e:
                print(f"  [FAILED to download {arxiv_id}] {e}")
                # Clean up any partially-written file so a retry doesn't think it's done.
                if pdf_path.exists():
                    pdf_path.unlink()
                continue
 
        records.append(
            {
                "arxiv_id": arxiv_id,
                "title": result.title.strip().replace("\n", " "),
                "authors": "; ".join(a.name for a in result.authors),
                "abstract": result.summary.strip().replace("\n", " "),
                "matched_term": term,
                "pdf_filename": pdf_path.name,
                "primary_category": result.primary_category,
                "published": result.published.date().isoformat(),
            }
        )
 
    print(f"  -> {len(records)} results for {term!r}")
    return records
 
 
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
 
    client = arxiv.Client(page_size=RESULTS_PER_TERM, delay_seconds=5.0, num_retries=5)
 
    all_records: list[dict] = []
    for term in TERMS:
        try:
            all_records.extend(fetch_for_term(client, term))
        except Exception as e:
            print(f"[FAILED term search: {term!r}] {e}")
 
    # Dedupe by arxiv_id, keeping the first occurrence but tracking all matched terms.
    deduped: dict[str, dict] = {}
    for rec in all_records:
        aid = rec["arxiv_id"]
        if aid not in deduped:
            deduped[aid] = rec
        else:
            # Merge matched terms if the same paper matched multiple search terms.
            existing_terms = set(deduped[aid]["matched_term"].split(", "))
            existing_terms.add(rec["matched_term"])
            deduped[aid]["matched_term"] = ", ".join(sorted(existing_terms))
 
    final_records = list(deduped.values())
 
    fieldnames = [
        "arxiv_id",
        "title",
        "authors",
        "abstract",
        "matched_term",
        "pdf_filename",
        "primary_category",
        "published",
    ]
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_records)
 
    print(f"\nDone. {len(all_records)} raw results, {len(final_records)} unique papers.")
    print(f"PDFs saved to: {OUTPUT_DIR.resolve()}")
    print(f"Metadata CSV saved to: {METADATA_CSV.resolve()}")
 
 
if __name__ == "__main__":
    main()
 
