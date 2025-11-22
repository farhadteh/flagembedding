"""
#!/usr/bin/env python3
"""
Encode corpus with model, build FAISS index, and perform a sample query.
"""
import json, argparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_docs(jsonl_path):
    docs = {}
    with open(jsonl_path) as f:
        for line in f:
            o=json.loads(line); docs[o['doc_id']] = o['doc_text']
    return docs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='output/dual_encoder')
    parser.add_argument('--index_path', default='faiss.index')
    parser.add_argument('--doc_map', default='doc_map.json')
    args = parser.parse_args()

    docs = load_docs(args.data)
    doc_ids = list(docs.keys())
    texts = [docs[d] for d in doc_ids]
    model = SentenceTransformer(args.model)
    emb = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, args.index_path)
    with open(args.doc_map,'w') as fw:
        json.dump(doc_ids, fw)
    print('Saved index and doc_map')

    # sample query
    q = input('Enter a query: ')
    q_emb = model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D,I = index.search(q_emb, 10)
    for score, idx in zip(D[0], I[0]):
        print(score, doc_ids[idx])
"""
