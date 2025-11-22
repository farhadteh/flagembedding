"""
Overview

Purpose: End-to-end retrieval pipeline for job-search: given a user query (keywords), return top relevant job ads using an embedding-based dual-encoder and FAISS index.

Scope: Text-only (queries and job ad content). Training a dual-encoder (retriever). Optionally use a cross-encoder offline to mine hard negatives; the cross-encoder is not used at query time.

Label handling: label==1 and label==2 are treated as positives; label==0 is negative. This is configurable in the training config.

Goals

- Provide runnable scripts for:
  - mining hard negatives (FAISS/BM25 candidate + optional cross-encoder re-rank)
  - training a dual-encoder using positives + mined hard negatives
  - building and querying a FAISS index
- Provide small sample dataset and default configs tuned for a 22GB GPU (g5 EC2).

Acceptance criteria

- End-to-end notebook/script can run on a single 22GB GPU and produce a trained model and FAISS index.
- Training uses in-batch negatives + explicit hard negatives when available.
- Evaluation includes Recall@K and example queries.
"""
