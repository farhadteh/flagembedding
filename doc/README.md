Overview

This folder contains documentation and runnable examples focused on text retrieval for a job-search application using FlagEmbedding. The materials are tailored for fine-tuning a dual-encoder retrieval model (embeddings -> FAISS) and optionally using a cross-encoder for offline hard-negative mining.

Table of contents
- 01-introduction.md — Purpose, label conventions, goals
- 02-retrieval-overview.md — embedding / retrieval concepts and metrics
- 03-finetune-step-by-step.md — end-to-end fine-tuning guide for dual-encoder retrieval
- 04-hard_negative_mining.md — candidate retrieval + optional cross-encoder mining
- 06-inference-and-serving.md — index building and query-time retrieval examples
- 07-tasks.md — checklist and next steps

Examples and scripts are under examples/ and templates/ for configs.