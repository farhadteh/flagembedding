"""
Building the FAISS index

- Encode all docs with trained document encoder:
  embeddings = model.encode(docs, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
  normalize embeddings with faiss.normalize_L2
- Choose index:
  - small corpus: IndexFlatIP
  - medium/large: IndexIVFFlat / HNSW (configure nlist, probes)
- Persist index with faiss.write_index and save doc_id mapping.

Query-time retrieval

- Encode query: q_emb = model.encode([query], convert_to_numpy=True)
- Normalize and search: D,I = index.search(q_emb, top_k)
- Map I indices back to doc_ids and return scores.

Example snippet:
- (See examples/build_index_and_search.py)

Scaling tips

- Use GPU FAISS when possible for faster indexing/search.
- Pre-compute document embeddings and shard index for very large corpora.
"""
