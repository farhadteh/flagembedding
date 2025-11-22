"""
Key concepts

- Dual-encoder: separate encoders for query and document that map text to dense embeddings. Retrieval := nearest neighbors in embedding space (dot product or cosine).
- Similarity: recommended is inner product on L2-normalized vectors (equivalent to cosine).
- Indexing: FAISS (IndexFlatIP for small corpora, IVF/HNSW for production). Persist embeddings to disk for reuse.
- Evaluation metrics: Recall@K (primary), MRR@K (secondary).

Recommended workflow

1. Preprocess dataset and split train/val/test (e.g., 80/10/10).
2. Mine hard negatives: use BM25 and/or FAISS (embedding nearest neighbors) to create candidate pools; optionally re-rank with a cross-encoder offline and select hard negatives.
3. Fine-tune dual-encoder with:
   - in-batch negatives (MultipleNegativesRankingLoss) and/or explicit hard negatives (triplet or N-pair style).
4. Encode all docs, build FAISS index, and evaluate retrieval.
5. Deploy: model.encode() for queries, query FAISS index for top-K.
"""
