"""
#!/usr/bin/env python3
"""
Mine hard negatives using BM25 and/or FAISS candidate retrieval and an optional cross-encoder for re-ranking.
Outputs hard_negatives.jsonl with lines: {"query_id":"q1","query":"...","hard_negatives":["d12",...]}
"""
import json
import argparse
from collections import defaultdict
import numpy as np
try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
except Exception as e:
    raise ImportError("Install required libs: pip install sentence-transformers faiss-cpu rank_bm25")


def load_dataset(path):
    queries = {}
    docs = {}
    q2labels = defaultdict(list)
    with open(path,'r') as f:
        for line in f:
            obj = json.loads(line)
            qid = obj['query_id']
            queries[qid] = obj['query']
            did = obj['doc_id']
            docs[did] = obj['doc_text']
            q2labels[qid].append({'doc_id':did,'label':obj.get('label',0)})
    return queries, docs, q2labels


def build_bm25(docs):
    corpus = [docs[d].split() for d in docs]
    dids = list(docs.keys())
    bm25 = BM25Okapi(corpus)
    return bm25, dids


def build_faiss(encoder, docs, batch_size=64):
    dids = list(docs.keys())
    texts = [docs[d] for d in dids]
    emb = encoder.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, dids, emb


def retrieve_candidates(queries, docs, bm25=None, bm25_dids=None, faiss_index=None, faiss_dids=None, encoder=None, K=100):
    candidates = {}
    for qid,qtext in queries.items():
        cand = []
        if bm25:
            tokens = qtext.split()
            scores = bm25.get_scores(tokens)
            top = np.argsort(scores)[::-1][:K]
            cand += [bm25_dids[i] for i in top]
        if faiss_index and encoder:
            q_emb = encoder.encode([qtext], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            D,I = faiss_index.search(q_emb, K)
            cand += [faiss_dids[i] for i in I[0]]
        # unique preserve order
        seen=set(); uniq=[]
        for d in cand:
            if d not in seen:
                uniq.append(d); seen.add(d)
        candidates[qid]=uniq[:K]
    return candidates


def rerank_with_crossencoder(queries, docs, candidates, q2labels, reranker_name, M=5, batch_size=32):
    reranker = CrossEncoder(reranker_name, max_length=256)
    hard_neg = {}
    for qid, cand_list in candidates.items():
        pairs = [(queries[qid], docs[did]) for did in cand_list]
        if not pairs:
            hard_neg[qid] = []
            continue
        scores = reranker.predict(pairs, batch_size=batch_size)
        scored = list(zip(cand_list, scores))
        positives = {x['doc_id'] for x in q2labels[qid] if x.get('label',0) in (1,2)}
        filtered = [(d,s) for d,s in scored if d not in positives]
        filtered.sort(key=lambda x: x[1], reverse=True)
        hard_neg[qid] = [d for d,_ in filtered[:M]]
    return hard_neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', choices=('bm25','faiss','combined'), default='combined')
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--reranker', default=None, help='optional CrossEncoder HF name')
    args = parser.parse_args()

    queries, docs, q2labels = load_dataset(args.input)

    bm25=None; bm25_dids=None; faiss_index=None; faiss_dids=None
    base_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    if args.mode in ('bm25','combined'):
        bm25, bm25_dids = build_bm25(docs)
    if args.mode in ('faiss','combined'):
        faiss_index, faiss_dids, _ = build_faiss(base_encoder, docs)

    candidates = retrieve_candidates(queries, docs, bm25=bm25, bm25_dids=bm25_dids,
                                     faiss_index=faiss_index, faiss_dids=faiss_dids, encoder=base_encoder, K=args.K)

    if args.reranker:
        hard_neg = rerank_with_crossencoder(queries, docs, candidates, q2labels, reranker_name=args.reranker, M=args.M)
    else:
        # fallback: take top-M from candidates excluding positives
        hard_neg={}
        for qid, cand in candidates.items():
            positives = {x['doc_id'] for x in q2labels[qid] if x.get('label',0) in (1,2)}
            filtered = [d for d in cand if d not in positives]
            hard_neg[qid] = filtered[:args.M]
    out_path = 'hard_negatives.jsonl'
    with open(out_path,'w') as fw:
        for qid, items in hard_neg.items():
            fw.write(json.dumps({'query_id':qid,'query':queries[qid],'hard_negatives':items}) + '\n')
    print('Wrote', out_path)

if __name__ == '__main__':
    main()
"""
