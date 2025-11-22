"""
#!/usr/bin/env python3
"""
Train a dual-encoder retriever using sentence-transformers.
- Uses positives (labels 1+2) and explicit hard negatives if provided.
- Default loss: MultipleNegativesRankingLoss (in-batch); supports adding triplet examples when hard negatives present.
"""
import json, argparse, random
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

def load_data(jsonl_path):
    queries = {}
    docs = {}
    q2labels = {}
    with open(jsonl_path) as f:
        for line in f:
            o=json.loads(line)
            qid=o['query_id']; queries.setdefault(qid, o['query'])
            docs[o['doc_id']] = o['doc_text']
            q2labels.setdefault(qid, []).append({'doc_id':o['doc_id'],'label':o.get('label',0)})
    return queries, docs, q2labels

def build_train_examples(queries, docs, q2labels, hard_neg_path=None):
    hard_neg = {}
    if hard_neg_path:
        with open(hard_neg_path) as f:
            for line in f:
                o=json.loads(line); hard_neg[o['query_id']] = o.get('hard_negatives',[])
    examples = []
    for qid,qtext in queries.items():
        positives = [d['doc_id'] for d in q2labels.get(qid,[]) if d.get('label',0) in (1,2)]
        if not positives: continue
        for pos in positives:
            if qid in hard_neg and hard_neg[qid]:
                for neg in hard_neg[qid]:
                    examples.append(InputExample(texts=[qtext, docs[pos], docs[neg]]))
            else:
                # sample random negative
                neg = random.choice(list(docs.keys()))
                if neg == pos: continue
                examples.append(InputExample(texts=[qtext, docs[pos], docs[neg]]))
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--hard_neg', default=None)
    parser.add_argument('--model_name', default='all-MiniLM-L12-v2')
    parser.add_argument('--output', default='output/dual_encoder')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--grad_accum', type=int, default=2)
    args = parser.parse_args()

    queries, docs, q2labels = load_data(args.data)
    train_examples = build_train_examples(queries, docs, q2labels, args.hard_neg)
    if not train_examples:
        raise SystemExit("No training examples created. Check data labels.")

    model = SentenceTransformer(args.model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    # Use TripletLoss if using InputExample(query, pos, neg)
    train_loss = losses.BatchAllTripletLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              output_path=args.output,
              show_progress_bar=True,
              # you can set optimizer params or use default
              )
"""
