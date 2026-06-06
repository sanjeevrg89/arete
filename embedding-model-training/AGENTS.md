# AGENTS.md — Embedding & Reranker Model Training

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`embedding-model-training-guide.md`** next to this file —
> read it before training, fine-tuning, distilling, or evaluating an embedding/retriever/reranker.
> Concrete artifacts to imitate (InfoNCE bi-encoder sketch, positive-aware mining config, cross-encoder
> distillation outline) are in **`examples.md`**. This file is the always-on summary.
>
> Scope: how the **models** are built. For ANN indexes, hybrid search, and RAG retrieval infra that
> *consume* embeddings, see **[[rag-vector-databases]]**. The field moves fast — treat every benchmark
> number and "SOTA" claim as **time-scoped; verify against current docs/leaderboards.**

## Apply by default when training/evaluating embedding or reranker models

- **Pick architecture for the latency budget.** Bi-encoder (separate query/doc encodes, cosine/dot,
  indexable) = **first-stage retrieval** over millions of docs. Cross-encoder (joint encode, one score,
  not indexable) = **reranker** over top-k. Default production shape is **retrieve-then-rerank**. If
  recall is fine but the top results are wrong, you needed a **reranker**, not a better bi-encoder.
- **Contrastive learning is the core.** InfoNCE with **in-batch negatives**; **large batches help**
  (more negatives/step) via cross-GPU gather / GradCache. Tune **temperature** (~0.01–0.05 for
  normalized embeddings). Fix **pooling** (mean / CLS / last-token) and **L2-normalize** for cosine.
- **Hard negatives are the crux.** Random/in-batch negatives plateau. Mine hard negatives with a strong
  retriever (top-ranked non-positives); refresh ANCE-style as the model improves.
- **Always remove false negatives.** Naive top-k mining mislabels many *true* positives as negatives
  (large fractions on web data) and trains the model to reject correct answers. Use **positive-aware /
  threshold-based mining** (NV-Retriever): anchor on the positive's score, drop near-positive
  candidates. Never feed raw top-k as negatives on sparsely-labeled data.
- **Recipe:** weakly-supervised contrastive **pretrain** → **fine-tune** on labeled + mined-hard-negative
  data → optionally **distill from a cross-encoder** teacher (MarginMSE / KL). Keep **instruction
  prefixes** identical at train and serve.
- **Matryoshka (MRL):** train nested dims so you can **truncate** at deploy time (cheaper storage/ANN,
  graceful accuracy loss). **ColBERT** multi-vector trades a bigger index for higher accuracy.
- **Evaluate on MTEB (Retrieval subset = BEIR).** Use **nDCG@10** for ranking, **recall@k** for the
  first-stage retriever. Watch **contamination/overfitting**; build a **held-out in-domain eval set**
  and treat it as the primary signal. Leaderboard ranks are snapshots — verify current.
- **Match the index exactly:** same **dimension**, same **distance metric**, same **normalization**, same
  instruction prefixes — train and serve. Mismatch = silent ranking collapse.
- **Re-embed the whole corpus on every model change.** Query and corpus vectors must come from the same
  model version; **version your indexes** and blue/green swap. Mixing versions is the most common silent
  regression.
- **Quantize** embeddings (int8 ~4×, binary ~32×) to cut cost; validate the recall hit on your eval set.

## Anti-patterns (reject these)
Random/in-batch negatives only · ignoring false negatives · evaluating on contaminated benchmarks ·
dimension/distance-metric mismatch with the index · not re-embedding after a model change ·
train/serve skew in pooling/normalization/instruction prefixes · training a bi-encoder when a reranker
was the real need · tiny batches · mining negatives once and never refreshing.

## Related skills
`[[rag-vector-databases]]` · `[[recsys-ranking]]` · `[[multimodal-ml]]` · `[[fine-tuning-peft]]` ·
`[[ml-frameworks]]` · `[[training-frameworks]]` · `[[ml-evaluation-evals]]` · `[[aiml-on-kubernetes]]`
