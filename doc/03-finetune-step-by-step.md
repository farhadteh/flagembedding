"""
Data format

- JSONL where each line is:
  {"query_id":"q1","query":"python backend engineer","doc_id":"d1","doc_text":"We are hiring ...","label":2}
- Labels: 2 or 1 → positive; 0 → negative.

Splits

- Split by query_id to avoid leakage (train/val/test on distinct queries). Default 80/10/10.

Preprocessing

- Clean text: strip HTML, trim whitespace; minimal tokenization (model tokenizer will handle token boundaries).
- Deduplicate docs by doc_id.

Training recipe (defaults tuned for 22GB GPU)

- Base encoder: BAAI/bge-base-en-v1.5 (or another HF-compatible sentence-transformer). You can swap the model_name in config.
- Loss: MultipleNegativesRankingLoss (in-batch softmax) or ContrastiveLoss; to add explicit hard negatives use triplet-style InputExample (query, positive, negative).
- Hyperparameters (templates/finetune_config.yaml):
  - model_name: BAAI/bge-base-en-v1.5
  - train_batch_size: 32
  - gradient_accumulation_steps: 2 (effective batch 64)
  - epochs: 3
  - learning_rate: 2e-5
  - fp16: true
  - max_seq_length: 128
  - candidate_pool_k: 100
  - hard_negative_m: 5
- Logging and checkpointing: save best by validation Recall@10.

Training flow (high level)

1. Create training triples: for each query, pair query with positive doc(s) and sample negatives (prefer hard negatives).
2. Use DataLoader with shuffle, and a suitable batch size for your GPU.
3. Train with Mixed Precision (fp16) and gradient accumulation.
4. Evaluate on val set after each epoch; compute Recall@K and MRR.

Command-line example (after pushing scripts):

- Mine hard negatives:
  python examples/mine_hard_negatives.py --input examples/data/sample_job_data.jsonl --mode combined --K 100 --M 5 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
- Train:
  python examples/train_dual_encoder.py --data examples/data/sample_job_data.jsonl --hard_neg hard_negatives.jsonl --config templates/finetune_config.yaml
"""
