# AGENTS.md — Recommender & Ranking Systems

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`recsys-ranking-guide.md`** next to this file — read it
> before designing or debugging a recommendation/ranking system, and apply it. A worked two-tower +
> ranking funnel sketch and an offline+online eval plan are in **`examples.md`**. This is the always-on
> summary.
>
> Recsys moves fast (it is 2026). Treat specific model names, generative/LLM-recsys techniques, and any
> numbers as **verify against current docs/papers** — never fabricate benchmarks.

## When working on a recommendation or ranking system, apply these by default:

- **Stage the funnel.** retrieval/candidate-gen (recall, cheap/item, whole corpus) → optional
  pre-ranking → ranking (precise, rich features, heavy) → re-ranking (list-level: diversity, dedup,
  freshness) → policy/business rules. Give each stage an explicit latency budget and a recall target.
  Never score the whole corpus with the heavy ranker.
- **Retrieval = embeddings + ANN + heuristic sources, blended.** Two-tower/dual-encoder (independent
  item tower → precompute embeddings → ANN search, see `[[rag-vector-databases]]`) is the default;
  complement with collaborative filtering (MF/ALS), graph/co-visitation, trending/fresh/lexical sources.
  Tag each candidate with its source. Keep the towers **late-interaction** (dot product only) or you lose
  precomputability.
- **Negative sampling makes or breaks retrieval.** In-batch negatives **with the log-Q /
  sampled-softmax correction** (skip it and you suppress popular items), plus mined **hard negatives**
  for precision. Easy-negatives-only → a retriever that can't tell good from mediocre.
- **Ranking: rich features first.** GBDTs (XGBoost/LightGBM, LambdaMART) are a strong baseline on
  tabular/dense features; go deep (Wide&Deep → DeepFM/DCN-v2 → DLRM → transformer/sequence rankers)
  for massive sparse IDs, learned crosses, and multi-task heads. Choose for *your* data+latency, not the
  leaderboard. Early feature interaction belongs here, not in retrieval.
- **Most surfaces are multi-objective.** Optimize engagement *and* satisfaction *and* diversity, not
  CTR alone. Use MMoE/PLE multi-task heads; blend into a final value with weights tuned **online against
  the north-star**, not by offline loss. CTR-only → degenerate clickbait feeds.
- **Train/serve skew is the #1 silent failure.** Log features at serving time and **train on those**
  (log-and-train); share transform code; do **point-in-time-correct** label/feature joins (no
  time-travel leakage). Use a feature store for consistency (`[[data-engineering-feature-stores]]`).
- **Offline metrics filter; online A/B decides.** Offline: recall@k (retrieval), nDCG/MAP/MRR, AUC/GAUC,
  LogLoss. Online: CTR + satisfaction/dwell + **long-term/north-star (retention, LTV)** with guardrails
  (latency, diversity, fairness, abuse) and long-term holdouts. Expect an offline-online gap.
  See `[[ml-evaluation-evals]]`.
- **Handle position bias** or you train the model to predict layout, not relevance, and lock in a
  feedback loop. Position-as-feature (neutralized at serving) or IPS. Log propensities so
  counterfactual/off-policy eval (IPS / doubly-robust) is possible *before* spending A/B traffic.
- **You need exploration.** ε-greedy/dithering or bandits (Thompson/UCB, contextual) — for cold start,
  to break filter bubbles, and to feed OPE. Budget it explicitly; judge it on long-term payoff.
- **Calibrate** when scores feed auctions/blending/thresholds (pCTR ≈ actual). Monitor reliability/ECE;
  add isotonic/Platt if ranking-only training left it miscalibrated.
- **Mind sparse embedding tables** (the dominant cost): hashing trick, embedding sharding/model-parallel,
  frequency-aware sizing/pruning, quantization. See `[[ml-frameworks]]`.
- **Cold start & popularity bias & feedback loops are first-class.** Content-tower retrieval + fresh
  sources + exploration for new items/users; diversity re-ranking and debiasing for the tail.
- **Generative/LLM-augmented recsys** (LLM encoders, semantic-ID generative retrieval, LLM rerankers) is
  a fast-moving frontier — apply surgically at the small final stages; **verify current** and don't
  assume it replaces the funnel.

## Definition of done for a ranking/recsys change
- Stage latency budgets respected; retrieval stage-recall@k measured and healthy.
- No train/serve skew: serving-logged features, shared transforms, point-in-time-correct labels.
- Position bias handled; propensities logged where exploration/OPE is used.
- Offline metrics used only to gate; an **online A/B plan** with north-star + guardrails + adequate
  power/duration exists before launch; calibration checked if scores feed downstream.
- Objective is the real (multi-objective) value, validated against long-term metrics — not CTR alone.

## Reviewing a recsys design/diff
Walk the funnel stage by stage; check the anti-patterns and the troubleshooting table at the end of
`recsys-ranking-guide.md`.
