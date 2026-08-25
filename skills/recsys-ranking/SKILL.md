---
name: recsys-ranking
description: World-class recommender and ranking systems at scale — the highest-revenue ML in industry
  (feeds, search, ads, e-commerce, video/music). Use when designing or debugging a recommendation or
  ranking pipeline: the multi-stage funnel (retrieval/candidate generation → pre-ranking → ranking →
  re-ranking → policy), two-tower/dual-encoder embedding retrieval with ANN, collaborative filtering
  (matrix factorization, ALS), ranking models (GBDT vs Wide&Deep/DeepFM/DCN-v2/DLRM/transformer rankers),
  multi-task/multi-objective ranking (MMoE/PLE), learning-to-rank (pointwise/pairwise/listwise),
  negative sampling, sparse embedding tables, feature stores & training-serving skew, offline metrics
  (nDCG/MAP/MRR/recall@k/AUC) vs online A/B (CTR, engagement, long-term value), position bias &
  debiasing, exploration/exploitation (bandits), cold start, popularity bias, calibration, feedback
  loops/filter bubbles, and LLM-augmented/generative recsys. Covers the latency/quality budget per stage.
---

# Recommender & Ranking Systems

Apply the judgment of an engineer who has owned a recommendation or ranking surface in production at
scale for years — where a 0.5% relevance win is millions of dollars, the offline metric lies, and the
feedback loop will quietly eat your feed if you optimize the wrong objective.

## How to use this skill

1. **Read `recsys-ranking-guide.md`** in this directory — the full reference (funnel, models, features,
   evaluation, production concerns, anti-patterns). Apply it to the task.
2. For a concrete two-tower + ranking funnel sketch and an offline+online eval plan for a ranking
   change, read **`examples.md`**.
3. Match the surrounding system's conventions (feature store, serving stack, framework). Apply the
   correctness rules — train/serve consistency, position-bias-aware training, honest online eval —
   regardless. The ecosystem moves fast (it is 2026); flag fast-moving claims and verify current docs.

## The essentials (full detail in `recsys-ranking-guide.md`)

- **Stage the funnel.** Millions of items → ~hundreds → ~tens, under a tight latency budget. Each stage
  trades recall for precision: cheap+high-recall retrieval, mid pre-ranking, expensive precise ranking,
  then re-ranking and policy. Don't try to score the whole corpus with your heavy model.
- **Retrieval = embeddings + ANN, plus heuristic sources.** Two-tower/dual-encoder with ANN
  (`[[rag-vector-databases]]`) is the workhorse; complement with collaborative filtering (MF/ALS),
  graph, and rule-based sources. Blend multiple retrievers — no single one covers the space.
- **Negative sampling is the whole game in retrieval.** In-batch negatives + sampled-softmax with a
  log-Q correction; add **hard negatives** for precision. Cheap negatives → models that can't tell
  good from mediocre.
- **Ranking: rich features beat clever models, up to a point.** GBDTs are a strong baseline; deep
  rankers (Wide&Deep → DeepFM/DCN-v2 → DLRM → transformer rankers) win when you have huge sparse
  features and feature crosses worth learning. Pick for *your* data and latency, not the leaderboard.
- **Most surfaces are multi-objective.** Optimize engagement *and* satisfaction *and* diversity, not
  CTR alone. Use multi-task architectures (MMoE/PLE) and combine task heads with a tuned value model;
  CTR-only optimization produces degenerate clickbait feeds.
- **Train/serve skew is the #1 silent killer.** The same feature transforms, the same feature store,
  point-in-time-correct training labels (`[[data-engineering-feature-stores]]`). Log features at
  serving time and train on *those*.
- **Offline metrics (nDCG, MAP, MRR, recall@k, AUC) only rank candidates; online A/B decides.** Expect
  an offline-online gap. Watch CTR *and* long-term/north-star metrics; use counterfactual/off-policy
  evaluation before you ship to a live test.
- **Position bias will silently train your model to predict layout, not relevance.** Model position
  (e.g. position-as-feature, IPS) or you learn a self-fulfilling prophecy.
- **You need exploration.** Pure exploitation collapses into filter bubbles and starves new items
  (cold start) and the model of counterfactual data. Bandits / ε-exploration / dithering.
- **Calibration matters when scores feed downstream** (ads auctions, blending). A well-ranked but
  miscalibrated pCTR breaks bidding and multi-objective blends.

## Related skills

- `[[ml-system-design]]` — the broader ML-system design frame this fits inside.
- `[[rag-vector-databases]]` — ANN indexes (HNSW, IVF-PQ, ScaNN) for embedding retrieval.
- `[[data-engineering-feature-stores]]` — feature store, point-in-time joins, train/serve consistency.
- `[[ml-evaluation-evals]]` — ranking/relevance metrics, A/B testing, off-policy evaluation.
- `[[ml-frameworks]]` — PyTorch/JAX for training rankers and embedding models.
- `[[serving-frameworks]]` — low-latency model serving for the ranking stage.
- `[[aiml-on-kubernetes]]` — running training and serving on K8s/GKE at scale.
