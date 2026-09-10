"""
Evaluates a retriever against your hand-labeled eval set, computing
precision@5, recall@5, and MRR, and printing a results table.

Works with any of your three retrievers (dense / BM25 / hybrid) via a
small pluggable interface, so you can run this once per combination
(Week 3's "run the eval on all 4 combinations" step).

Eval set CSV format (one row per question):
    question_id,question,question_type,correct_paper_ids,correct_chunk_ids_fixed,correct_chunk_ids_structured,answer

- correct_chunk_ids_fixed / correct_chunk_ids_structured: semicolon-
  separated chunk_ids you've hand-verified contain the answer, one
  column per chunking strategy (a chunk boundary in the fixed-size
  version isn't the same chunk as in the structured version, so each
  needs its own ground truth). Whichever one matches the chunk set
  you're evaluating against is used for precision/recall/MRR.
- correct_paper_ids, question_type, answer: kept for your own
  reference/output. NOT used in the metric math.
- Rows with no ground truth for the relevant chunk set are skipped
  (with a warning), since precision/recall/MRR are undefined without
  chunk-level ground truth.

Usage:
    python eval_retrieval.py eval_questions.csv dense fixed_chunks
    python eval_retrieval.py eval_questions.csv dense structured_chunks
    python eval_retrieval.py eval_questions.csv bm25 data/chunks_fixed.jsonl
    python eval_retrieval.py eval_questions.csv hybrid structured_chunks data/chunks_structured.jsonl

Optional: add an output path as the last argument to save per-question
results to CSV:
    python eval_retrieval.py eval_questions.csv dense fixed_chunks results_dense_fixed.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Set

from bm25_retriever import BM25Retriever
from hybrid_retrieval import dense_search, hybrid_search

K = 5              # precision@k, recall@k
MRR_DEPTH = 10      # look this deep for MRR's "first relevant hit"


# ---- Eval set loading ---------------------------------------------------

def infer_chunk_set(identifier: str) -> str:
    """Infer whether `identifier` (a collection name or file path) refers
    to the fixed-size or structure-aware chunk set, based on substring
    matching. Raises if it can't tell, rather than silently guessing.
    """
    lowered = identifier.lower()
    is_fixed = "fixed" in lowered
    is_structured = "structured" in lowered

    if is_fixed and not is_structured:
        return "fixed"
    if is_structured and not is_fixed:
        return "structured"

    raise ValueError(
        f"Could not infer chunk set (fixed vs structured) from {identifier!r}. "
        "Expected 'fixed' or 'structured' to appear in the collection name / "
        "file path (e.g. 'fixed_chunks', 'data/chunks_structured.jsonl')."
    )


def load_eval_set(path: Path, chunk_set: str) -> List[Dict]:
    """Load the hand-labeled eval CSV into a list of dicts:
    {question_id, question, question_type, answer,
     correct_chunk_ids (set, from the column matching `chunk_set`),
     correct_paper_ids (set)}.

    `chunk_set` must be "fixed" or "structured" — selects whether
    correct_chunk_ids_fixed or correct_chunk_ids_structured is used as
    ground truth.
    """
    if chunk_set not in ("fixed", "structured"):
        raise ValueError(f"chunk_set must be 'fixed' or 'structured', got {chunk_set!r}")

    ground_truth_column = f"correct_chunk_ids_{chunk_set}"

    eval_rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if ground_truth_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {ground_truth_column!r} not found in {path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):  # header is line 1
            question = row.get("question", "").strip()
            if not question:
                continue

            chunk_ids_raw = row.get(ground_truth_column, "").strip()
            paper_ids_raw = row.get("correct_paper_ids", "").strip()

            correct_chunk_ids: Set[str] = {
                c.strip() for c in chunk_ids_raw.split(";") if c.strip()
            }
            correct_paper_ids: Set[str] = {
                p.strip() for p in paper_ids_raw.split(";") if p.strip()
            }

            if not correct_chunk_ids:
                q_id = row.get("question_id", f"row {row_num}")
                print(f"[warn] {q_id}: no {ground_truth_column}, skipping "
                      f"question: {question!r}")
                continue

            eval_rows.append({
                "question_id": row.get("question_id", "").strip(),
                "question": question,
                "question_type": row.get("question_type", "").strip(),
                "answer": row.get("answer", "").strip(),
                "correct_chunk_ids": correct_chunk_ids,
                "correct_paper_ids": correct_paper_ids,
            })

    return eval_rows


# ---- Retriever adapters --------------------------------------------------
# Each adapter has signature: (question: str, top_n: int) -> List[Dict],
# where each dict has at least a "chunk_id" key, ranked best-first.

def make_dense_retriever(collection_name: str) -> Callable[[str, int], List[Dict]]:
    def retrieve(question: str, top_n: int) -> List[Dict]:
        return dense_search(question, collection_name, top_k=top_n)
    return retrieve


def make_bm25_retriever(chunks_path: str) -> Callable[[str, int], List[Dict]]:
    bm25 = BM25Retriever(chunks_path)

    def retrieve(question: str, top_n: int) -> List[Dict]:
        return bm25.search(question, top_k=top_n)
    return retrieve


def make_hybrid_retriever(
    collection_name: str, chunks_path: str
) -> Callable[[str, int], List[Dict]]:
    def retrieve(question: str, top_n: int) -> List[Dict]:
        return hybrid_search(
            question, collection_name, chunks_path,
            top_k=top_n, dense_k=max(20, top_n), bm25_k=max(20, top_n),
        )
    return retrieve


# ---- Metric computation ---------------------------------------------------

def evaluate_question(
    retrieved_ranked: List[Dict],
    correct_chunk_ids: Set[str],
    k: int = K,
) -> Dict:
    """Compute precision@k, recall@k, and reciprocal rank for one question.

    `retrieved_ranked` should have at least `max(k, MRR_DEPTH)` items
    (best-first) for MRR to have a fair chance of finding a hit.
    """
    top_k = retrieved_ranked[:k]
    relevant_in_top_k = sum(1 for c in top_k if c["chunk_id"] in correct_chunk_ids)

    precision = relevant_in_top_k / k if k > 0 else 0.0
    recall = relevant_in_top_k / len(correct_chunk_ids) if correct_chunk_ids else 0.0

    reciprocal_rank = 0.0
    for rank, chunk in enumerate(retrieved_ranked, start=1):
        if chunk["chunk_id"] in correct_chunk_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "relevant_in_top_k": relevant_in_top_k,
        "total_relevant": len(correct_chunk_ids),
    }


def run_evaluation(
    eval_set: List[Dict],
    retrieve_fn: Callable[[str, int], List[Dict]],
    k: int = K,
    mrr_depth: int = MRR_DEPTH,
) -> List[Dict]:
    """Run every question in eval_set against retrieve_fn, returning a list
    of per-question result dicts (question, precision, recall, reciprocal_rank, ...).
    """
    results = []
    top_n = max(k, mrr_depth)

    for row in eval_set:
        retrieved = retrieve_fn(row["question"], top_n)
        metrics = evaluate_question(retrieved, row["correct_chunk_ids"], k=k)
        results.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "question_type": row["question_type"],
            **metrics,
        })

    return results


# ---- Reporting --------------------------------------------------------

def print_results_table(results: List[Dict], k: int = K) -> None:
    print(f"\n{'Question':<60} {'P@' + str(k):>8} {'R@' + str(k):>8} {'RR':>8}")
    print("-" * 88)
    for r in results:
        q_display = r["question"][:57] + "..." if len(r["question"]) > 60 else r["question"]
        print(f"{q_display:<60} {r['precision']:>8.3f} {r['recall']:>8.3f} "
              f"{r['reciprocal_rank']:>8.3f}")

    n = len(results)
    if n == 0:
        print("\nNo evaluated questions.")
        return

    avg_precision = sum(r["precision"] for r in results) / n
    avg_recall = sum(r["recall"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n

    print("-" * 88)
    print(f"{'AVERAGE (n=' + str(n) + ')':<60} {avg_precision:>8.3f} {avg_recall:>8.3f} "
          f"{mrr:>8.3f}")
    print(f"\nPrecision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}:    {avg_recall:.4f}")
    print(f"MRR:          {mrr:.4f}")


def save_results_csv(results: List[Dict], output_path: Path, k: int = K) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "question", "question_type", "precision",
                        "recall", "reciprocal_rank", "relevant_in_top_k", "total_relevant"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nPer-question results saved to: {output_path.resolve()}")


# ---- CLI --------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print(
            "Usage:\n"
            "  python eval_retrieval.py <eval_set.csv> dense <collection_name> [output.csv]\n"
            "  python eval_retrieval.py <eval_set.csv> bm25 <chunks.jsonl> [output.csv]\n"
            "  python eval_retrieval.py <eval_set.csv> hybrid <collection_name> <chunks.jsonl> [output.csv]"
        )
        sys.exit(1)

    eval_set_path = Path(args[0])
    retriever_type = args[1].lower()

    if retriever_type == "dense":
        collection_name = args[2]
        chunk_set = infer_chunk_set(collection_name)
        retrieve_fn = make_dense_retriever(collection_name)
        output_path = Path(args[3]) if len(args) > 3 else None
        label = f"dense ({collection_name})"

    elif retriever_type == "bm25":
        chunks_path = args[2]
        chunk_set = infer_chunk_set(chunks_path)
        retrieve_fn = make_bm25_retriever(chunks_path)
        output_path = Path(args[3]) if len(args) > 3 else None
        label = f"bm25 ({chunks_path})"

    elif retriever_type == "hybrid":
        if len(args) < 4:
            print("hybrid requires: <collection_name> <chunks.jsonl>")
            sys.exit(1)
        collection_name = args[2]
        chunks_path = args[3]
        chunk_set = infer_chunk_set(chunks_path)
        retrieve_fn = make_hybrid_retriever(collection_name, chunks_path)
        output_path = Path(args[4]) if len(args) > 4 else None
        label = f"hybrid ({collection_name} + {chunks_path})"

    else:
        print(f"Unknown retriever type: {retriever_type!r} (expected dense/bm25/hybrid)")
        sys.exit(1)

    if not eval_set_path.exists():
        print(f"Eval set not found: {eval_set_path.resolve()}")
        sys.exit(1)

    eval_set = load_eval_set(eval_set_path, chunk_set)
    print(f"Loaded {len(eval_set)} labeled question(s) from {eval_set_path} "
          f"(ground truth: correct_chunk_ids_{chunk_set})")
    print(f"Evaluating retriever: {label}\n")

    results = run_evaluation(eval_set, retrieve_fn, k=K, mrr_depth=MRR_DEPTH)
    print_results_table(results, k=K)

    if output_path:
        save_results_csv(results, output_path, k=K)


if __name__ == "__main__":
    main()