"""
Goal: produce hard_negatives.jsonl mapping query_id -> list of hard negative doc_ids.

Candidate pool creation (choose one or union)

- BM25: fast and works well for keyword-heavy job texts.
  - Tokenize docs and use rank_bm25 to retrieve top-K.
- FAISS (embedding ANN): encode documents with a base embedder (all-MiniLM-L6-v2 or bge-base) and use FAISS to get nearest neighbors.
- Combined: union(BM25_topK, FAISS_topK) for diversity.

Optional cross-encoder re-ranking (offline)

- Use a CrossEncoder (sentence-transformers or available BGE reranker) to re-score candidate (query, doc_text) pairs to pick the most challenging negatives (highest cross-encoder score but not labeled positive).
- Workflow:
  1. Retrieve candidate pool per query (K=100).
  2. Remove ground-truth positives (labels 1/2).
  3. Re-score remaining candidates with cross-encoder.
  4. Store top-M as hard negatives (M=5).

Output format

- JSONL lines:
  {"query_id":"q1","query":"...","hard_negatives":["d12","d34","d9", ...]}

Performance notes for g5(22GB)
- Cross-encoder re-ranking batch_size 32–64 depending on GPU; enable fp16 if supported.
- Build FAISS embeddings offline and persist if corpus large.
"""
