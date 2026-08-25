---
name: embedding-model-training
description: World-class guidance for TRAINING text/retrieval embedding and reranker models — bi-encoders
  (dual-encoders), cross-encoder rerankers, and the retrieve-then-rerank pattern. Use when training,
  fine-tuning, distilling, or evaluating an embedding/retriever/reranker (contrastive/InfoNCE loss,
  in-batch negatives, hard-negative mining, false-negative removal, cross-encoder distillation,
  Matryoshka/MRL nested dims, ColBERT multi-vector, instruction-tuned embeddings, multilingual/
  multimodal), or when picking embedding dimension/pooling/normalization and evaluating on MTEB/BEIR
  (nDCG@10, recall). This is how the MODELS are built; for serving embeddings, ANN indexes, and RAG
  retrieval infra see [[rag-vector-databases]].
---

# Embedding Model Training

Apply the judgment of someone who has trained state-of-the-art retrievers and rerankers and shipped
them into production search/RAG systems. The hard part is almost never the loss function — it is the
**data**: which positives, and above all which negatives, you put in front of the model. Get
hard-negative mining and false-negative removal right and a small model beats a big one.

## How to use this skill

1. **Read `embedding-model-training-guide.md`** in this directory — the full reference (architectures,
   contrastive learning, hard-negative mining, training recipes, evaluation, production). Apply it.
2. For concrete artifacts to imitate — an InfoNCE bi-encoder training sketch, a positive-aware
   hard-negative-mining config, and a cross-encoder distillation outline — read **`examples.md`**.
3. Match the surrounding codebase/framework conventions ([[ml-frameworks]], [[training-frameworks]]);
   apply the data-quality and evaluation rules regardless.

## The essentials (full rationale in the guide)

- **Pick the right architecture for the latency budget.** Bi-encoder (encode query and doc separately,
  compare with cosine/dot) is cheap and indexable — use it for first-stage retrieval over millions of
  docs. Cross-encoder (joint query+doc → score) is far more accurate but cannot be pre-indexed — use it
  as a **reranker** over the top-k. The default production shape is **retrieve-then-rerank**
  ([[rag-vector-databases]], [[recsys-ranking]]). If a reranker was the real need, do not train a
  bi-encoder.
- **Contrastive learning is the workhorse.** InfoNCE with **in-batch negatives**: every other doc in
  the batch is a negative for the current query. Large batches help because they give more negatives per
  step (DPR, E5). Tune **temperature** (often ~0.01–0.05 for normalized embeddings). Decide pooling
  (mean vs CLS/last-token) and **L2-normalize** when training with cosine.
- **Hard negatives are the crux.** Random/in-batch negatives are too easy and accuracy plateaus. Mine
  hard negatives with a retriever (top-ranked non-positives) — ANCE-style, refreshed periodically.
- **The false-negative problem dominates web data.** Naive top-k mining mislabels many *true* relevant
  passages as negatives. **Positive-aware / threshold-based mining** (NV-Retriever) anchors on the
  positive's score and drops candidates scoring too close to it → faster training and higher accuracy.
- **Recipe: contrastive pretrain on weakly-supervised pairs → fine-tune on labeled + mined-hard-negative
  data → optionally distill from a cross-encoder teacher** (soft scores / MarginMSE / KL). Instruction-
  prefixed embeddings let one model serve many tasks.
- **Matryoshka (MRL)** trains nested dimensions so you can truncate (e.g. 1024→256) at deploy time for
  cheaper storage/ANN with graceful accuracy loss. **ColBERT** multi-vector trades index size for
  accuracy via late interaction.
- **Evaluate on MTEB (retrieval subset = BEIR)** with **nDCG@10** and recall@k — but watch
  **contamination/overfitting** and treat leaderboard ranks as time-scoped (verify against current docs).
- **Production:** the embedding **dimension and distance metric must match the index** exactly; quantize
  embeddings (int8/binary) to cut cost; and **re-embed the whole corpus when the model version changes**
  — query and corpus embeddings must come from the same model. Version your indexes.

## Related skills
- `[[rag-vector-databases]]` — serving embeddings: ANN indexes (HNSW/IVF), hybrid search, RAG pipelines.
- `[[recsys-ranking]]` — two-tower retrieval and learning-to-rank; the same retrieve-then-rank shape.
- `[[multimodal-ml]]` — CLIP-style image/text and other multimodal embeddings (same contrastive core).
- `[[fine-tuning-peft]]` — LoRA/adapter fine-tuning of LLM-based embedding backbones.
- `[[ml-frameworks]]` / `[[training-frameworks]]` — PyTorch/JAX, FSDP/DeepSpeed, large-batch training.
- `[[ml-evaluation-evals]]` — building trustworthy eval sets and avoiding benchmark contamination.
- `[[aiml-on-kubernetes]]` — running the embedding/mining/eval jobs on GPUs at scale.
