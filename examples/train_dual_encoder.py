#!/usr/bin/env python3
"""
Train a dual-encoder retriever using sentence-transformers.
- Uses positives (labels 1+2) and explicit hard negatives if provided.
- Default behavior: use MultipleNegativesRankingLoss (in-batch contrastive) when no explicit hard negatives are provided. If hard negatives are provided, use Triplet-style loss by default.
"""
import json, argparse, random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def load_data(jsonl_path):
    queries = {}
    docs = {}
    q2labels = {}
    with open(jsonl_path) as f:
        for line in f:
            o = json.loads(line)
            qid = o['query_id']
            queries.setdefault(qid, o['query'])
            docs[o['doc_id']] = o['doc_text']
            q2labels.setdefault(qid, []).append({'doc_id': o['doc_id'], 'label': o.get('label', 0)})
    return queries, docs, q2labels

def build_train_examples_inbatch(queries, docs, q2labels):
    # For MultipleNegativesRankingLoss, create (query, positive) pairs
    examples = []
    for qid, qtext in queries.items():
        positives = [d['doc_id'] for d in q2labels.get(qid, []) if d.get('label', 0) in (1, 2)]
        if not positives:
            continue
        # create one example per positive
        for pos in positives:
            examples.append(InputExample(texts=[qtext, docs[pos]]))
    return examples

def build_train_examples_triplet(queries, docs, q2labels, hard_neg_path=None):
    hard_neg = {}
    if hard_neg_path:
        with open(hard_neg_path) as f:
            for line in f:
                try:
                    o = json.loads(line)
                    hard_neg[o['query_id']] = o.get('hard_negatives', [])
                except Exception:
                    continue
    examples = []
    for qid, qtext in queries.items():
        positives = [d['doc_id'] for d in q2labels.get(qid, []) if d.get('label', 0) in (1, 2)]
        if not positives:
            continue
        for pos in positives:
            if qid in hard_neg and hard_neg[qid]:
                for neg in hard_neg[qid]:
                    examples.append(InputExample(texts=[qtext, docs[pos], docs[neg]]))
            else:
                # sample random negative
                neg = random.choice(list(docs.keys()))
                if neg == pos:
                    continue
                examples.append(InputExample(texts=[qtext, docs[pos], docs[neg]]))
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='JSONL dataset with query_id, query, doc_id, doc_text, label')
    parser.add_argument('--hard_neg', default=None, help='hard_negatives.jsonl (optional)')
    parser.add_argument('--model_name', default='BAAI/bge-base-en-v1.5', help='HuggingFace model name or local path')
    parser.add_argument('--output', default='output/dual_encoder', help='output model directory')
    parser.add_argument('--batch_size', type=int, default=16, help='per-device batch size')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--grad_accum', type=int, default=2, help='gradient accumulation steps')
    parser.add_argument('--use_triplet_if_hard', action='store_true', help='Force using triplet loss even if hard negatives exist')
    args = parser.parse_args()

    queries, docs, q2labels = load_data(args.data)

    # Decide training mode
    use_triplet = False
    if args.hard_neg:
        use_triplet = True
    if args.use_triplet_if_hard:
        use_triplet = True

    model = SentenceTransformer(args.model_name)

    if use_triplet:
        train_examples = build_train_examples_triplet(queries, docs, q2labels, args.hard_neg)
        if not train_examples:
            raise SystemExit('No triplet training examples created. Check data and hard_neg file.')
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.BatchAllTripletLoss(model=model)
    else:
        train_examples = build_train_examples_inbatch(queries, docs, q2labels)
        if not train_examples:
            raise SystemExit('No in-batch training examples created. Check data labels.')
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              output_path=args.output,
              show_progress_bar=True)