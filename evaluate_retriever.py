
import json
import chromadb

def evaluate_retriever(eval_json_path, collection_name, top_k=5, db_path="./chroma_db"):
    # Initialize Chroma client and collection
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        
    precisions = []
    recalls = []
    mrr_scores = []
    
    for item in eval_data:
        query = item['question']
        true_chunk_ids = set(item['correct_chunk_ids'])
        
        # Run retrieval against Chroma
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract retrieved IDs (adjust if you store chunk IDs in metadata)
        retrieved_chunk_ids = results['ids'][0]
        
        # Precision@k: fraction of retrieved chunks that are relevant
        hits = [cid for cid in retrieved_chunk_ids if cid in true_chunk_ids]
        precision = len(hits) / top_k
        precisions.append(precision)
        
        # Recall@k: fraction of all true chunks found in top-k
        if len(true_chunk_ids) > 0:
            recall = len(hits) / len(true_chunk_ids)
        else:
            recall = 0.0
        recalls.append(recall)
        
        # Reciprocal Rank for MRR
        rr = 0.0
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in true_chunk_ids:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)
        
    # Aggregate scores across all questions
    mean_precision = sum(precisions) / len(precisions) if precisions else 0
    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    mean_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    
    # Output results table
    print(f"\nEvaluation Results for Collection: {collection_name}")
    print(f"{'Metric':<20} | {'Score':<10}")
    print("-" * 33)
    print(f"{f'Precision@{top_k}':<20} | {mean_precision:.4f}")
    print(f"{f'Recall@{top_k}':<20} | {mean_recall:.4f}")
    print(f"{'MRR':<20} | {mean_mrr:.4f}\n")

if __name__ == "__main__":
    # Example execution for structured chunks
    evaluate_retriever(
        eval_json_path='data/eval_dataset.json', 
        collection_name='structured_chunks', 
        top_k=5
    )

