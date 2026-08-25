---
name: ml-system-design
description: World-class playbook for designing end-to-end ML systems — the "tie it all together" architect
  skill, covering both real-world production architecture and the ML-system-design interview. Use when asked
  to design an ML/AI system, scope an ML feature, do an ML system design interview, or reason about a
  recommendation/ranking, search/retrieval, feed, ads/CTR, classification, fraud/anomaly, or LLM/RAG/agent
  system end to end. Provides a repeatable framework (clarify problem & business metric → ML objective →
  data/labels & the third pipeline → features → model → training pipeline → offline+online evaluation →
  serving (online/batch/streaming, latency budget) → monitoring & iteration), the canonical problem
  archetypes, the key design axes/tradeoffs (online vs batch, latency vs throughput vs cost, candidate
  generation→ranking→re-ranking funnels, freshness, training-serving skew, cold start, feedback loops,
  scale & failure modes), how to choose and defend business/ML/guardrail metrics, and the anti-patterns.
---

# ML System Design

Apply the judgment of a staff ML systems architect who has shipped and operated large-scale ML and LLM
systems for years — and who has run the design interview from both sides of the table. The job is not to
name a model. It is to translate a fuzzy business need into a system: an **objective**, the **data and
pipelines** that feed it, an **evaluation** that proves it works, a **serving path** that meets a latency
and cost budget, and a **monitoring + iteration** loop that keeps it working as the world drifts.

## How to use this skill

1. **Read `ml-system-design-guide.md`** in this directory — the full framework, the problem archetypes,
   the design axes and tradeoffs, the metric hierarchy, the scale/estimation math, and the anti-patterns.
   Apply it to the design at hand.
2. For a complete worked design walked through the framework end to end (with the tradeoffs called out),
   read **`examples.md`**.
3. Whether the task is a real architecture or an interview: **scope first, never jump to the model.**
   State the business metric and the ML objective before drawing a single box. Match the surrounding
   codebase/platform conventions; apply the framework rigor regardless.

## The essentials (full detail in `ml-system-design-guide.md`)

- **Clarify before you design.** Pin the *business metric* you're moving and any constraints (scale, QPS,
  latency budget, cost, privacy) before proposing anything. A model you can't tie to a moveable KPI is a
  red flag.
- **Translate to a crisp ML problem and ML objective.** Name the task (binary/multiclass classification,
  regression, ranking/learning-to-rank, retrieval, generation) and the exact thing you optimize. The
  *business KPI*, the *ML metric*, and the *proxy label* are three different things — say how they relate.
- **ML systems have a third pipeline beyond train and serve: the data/feature pipeline** that produces
  labels and features for *both*. Most production failures live here (training-serving skew, label leakage,
  stale features). Design it explicitly. See `[[data-engineering-feature-stores]]`.
- **Evaluation is offline + online, and they disagree.** Offline metrics (AUC, NDCG, recall@k, calibration)
  gate launch; online A/B against business + **guardrail** metrics decides it. Expect an offline/online
  gap and plan the experiment. See `[[ml-evaluation-evals]]`.
- **Serving path follows the latency budget.** Online (per-request, ms budget) vs batch (precompute) vs
  streaming (near-real-time). Large/expensive models go behind a **candidate generation → ranking →
  re-ranking funnel**: cheap recall first, expensive precision last. See `[[serving-frameworks]]`.
- **Name the design axes out loud:** online vs batch, latency vs throughput vs cost, model complexity vs
  maintainability, build vs buy vs API, freshness vs staleness. Every choice is a tradeoff — defend it.
- **Plan for cold start, feedback loops, and failure modes.** New users/items, the model shaping its own
  future training data, and what happens when the model server is down (fallback to a heuristic/cached
  ranking — degrade, don't fail).
- **Estimate scale early:** QPS, data volume, feature/embedding size, index size, p99 latency, $/query.
  Back-of-envelope numbers drive every architecture decision (sharding, caching, replication, async).
- **Monitoring and iteration are part of the design, not an afterthought.** Track input drift, prediction
  drift, online metrics, and data quality; have a retraining trigger and a rollback. See `[[mlops-lifecycle]]`
  and `[[ml-observability-monitoring]]`.
- **Modern LLM/RAG/agent systems use the same framework** with new boxes: retrieval + vector index, prompt
  + context assembly, the generation model (API vs self-hosted), evals as the new offline metric, and
  cost/latency dominated by tokens. See `[[rag-vector-databases]]` and `[[llm-app-agent-frameworks]]`.
- **Avoid the anti-patterns:** jumping to a model before the metric, ignoring data/serving, no
  monitoring/iteration story, over-engineering a v1, and optimizing a metric you can't actually move.

## Related skills

- `[[recsys-ranking]]` — the recommendation/ranking archetype in depth (candidate gen, LTR, two-tower).
- `[[mlops-lifecycle]]` — the lifecycle around the system: CI/CD, retraining, registries, rollout.
- `[[ml-evaluation-evals]]` — offline metrics, A/B testing, and LLM evals to gate and decide launches.
- `[[data-engineering-feature-stores]]` — the third pipeline: feature stores, freshness, training-serving skew.
- `[[rag-vector-databases]]` — retrieval, embeddings, and vector indexes for the LLM/RAG archetype.
- `[[llm-app-agent-frameworks]]` — orchestrating LLM apps, tools, and agents.
- `[[serving-frameworks]]` — vLLM/Triton/KServe/Ray Serve for the serving box.
- `[[aiml-on-kubernetes]]` — running training and serving at scale on Kubernetes/GKE.
- `[[ml-observability-monitoring]]` — drift, data quality, and production monitoring.

---

# Reference — ml-system-design

# ML System Design — The Full Reference

This is the architect competency: turning a fuzzy business need into a running ML system, and defending
every decision. It applies equally to a real production design and to the ML-system-design interview — the
interview is a 45-minute compression of the real thing. The discipline is the same: **scope the problem,
pin the metric, design the data and the pipelines, choose a model you can defend, evaluate offline and
online, serve it within a budget, and monitor + iterate.**

> The ecosystem moves fast (it is 2026). Concrete tools, model families, vector-DB features, serving stacks,
> and managed APIs change quarterly — **verify against current docs** before committing to a specific
> version or quoted limit. The *framework* below is stable; the *tool names* are illustrative.

---

## 1. Mental model: an ML system is three pipelines + a loop

A classic software system is request → logic → response. An ML system is different because the "logic" is
*learned from data* and *decays over time*. The single most useful mental model:

> **An ML system is three pipelines, not two.** Beyond **(1) training** and **(2) serving/inference**, there
> is **(3) the data/feature pipeline** that produces labels and features for *both* of the others. Plus a
> **feedback loop**: predictions generate logs that become tomorrow's training data.

```
                 ┌─────────────────── (3) DATA / FEATURE PIPELINE ───────────────────┐
                 │  raw events ─► clean ─► join ─► label ─► features ─► feature store  │
                 └───────────────┬───────────────────────────────┬───────────────────┘
                                 │ (offline, batch)               │ (online, low-latency read)
                                 ▼                                ▼
                        (1) TRAINING PIPELINE              (2) SERVING PIPELINE
                        data ─► train ─► eval ─►            request ─► features ─► model ─► response
                        registry ─► (gate)                  └────────── logs ──────────┐
                                 ▲                                                      │
                                 └──────────────── FEEDBACK LOOP ──────────────────────┘
                                      (logged predictions + outcomes → new labels)
```

Why this matters: **most production ML failures are not model failures — they are pipeline failures.**
Training-serving skew (features computed differently offline vs online), label leakage (a feature that
encodes the answer), stale features, and a broken feedback loop kill more launches than a 1% AUC gap. If
your design has only "train" and "serve" boxes, it is incomplete. (See `[[data-engineering-feature-stores]]`.)

---

## 2. The repeatable design framework

Run every design — interview or real — through these steps **in order**. In an interview, roughly:
problem framing 5–7 min, high-level design 2–3 min, data & features ~10 min, modeling ~10 min, evaluation &
serving ~7 min, then deep dives. In real life the same steps span a design doc and a few review cycles.

### Step 1 — Clarify the problem and the business metric (do not skip)
- **What are we actually building, for whom, and why now?** Get the use case, the user, and the pain.
- **What business metric does success move?** Revenue, engagement (DAU, session length, retention),
  conversion, watch time, fraud-loss-prevented, support-deflection, cost-saved. *Write it down.* Everything
  downstream is justified by its effect on this number.
- **Constraints up front:** expected scale (users, QPS), latency budget (p99 ms), cost ceiling, privacy/PII
  and regulatory constraints (GDPR, fairness), freshness requirements, and what "good enough" looks like.
- Ask the clarifying questions a senior engineer asks: *Is this personalized? Real-time or batch? How many
  items? How fresh must it be? What's the current baseline (a heuristic? nothing?)?*

### Step 2 — Translate to an ML problem and ML objective
- **Frame the ML task precisely:** binary/multiclass classification, regression, ranking / learning-to-rank,
  retrieval/nearest-neighbor, sequence/generation, clustering/anomaly. The framing is a design decision —
  "recommend videos" can be pointwise CTR classification, pairwise ranking, or sequential next-item
  prediction, and each implies a different system.
- **Define the ML objective (the loss you optimize):** e.g. binary cross-entropy on click, a ranking loss,
  weighted multi-task loss over click + watch-time + like. State it explicitly.
- **Distinguish three different things:**
  | Layer | Example (video feed) | Who cares |
  |---|---|---|
  | **Business KPI** | daily watch time, retention | leadership |
  | **ML metric (offline)** | AUC, NDCG@10, recall@100 | you, to gate launch |
  | **Proxy label** | did the user click / watch ≥10s | what you actually train on |
- The art is choosing a **proxy label that correlates with the KPI without a perverse incentive.** Training
  on raw clicks optimizes clickbait; training on "watch ≥ X% of the video" or a satisfaction signal aligns
  better. **Naming this gap is a senior signal.**

### Step 3 — Data: sources, labels, and the third pipeline
- **Sources:** user/item profiles, interaction/event logs, content features, context (time, device, query),
  third-party. Note volume and freshness for each.
- **Labels — the hard part.** Where does ground truth come from?
  - *Implicit feedback* (clicks, watches, purchases) — abundant but biased (position bias, exposure bias,
    presentation bias — you only observe outcomes for items you showed).
  - *Explicit feedback* (ratings, thumbs) — sparse, often skewed.
  - *Human labeling* — costly, slow, for cold-start and eval sets.
  - *Weak/programmatic labels, distant supervision* — scale at the cost of noise.
- **Sampling & class imbalance:** fraud/ads CTR are extreme-imbalance (positives ≪ 1%). Decide negative
  sampling (random vs hard negatives), down/up-sampling, and class weighting — and remember to **recalibrate**
  predicted probabilities afterward.
- **The data pipeline is a first-class component.** Specify: ingestion (batch vs streaming), cleaning, joins,
  point-in-time-correct label generation (no leakage from the future), feature computation, and the **feature
  store** that serves the *same* features to training (offline, historical) and serving (online, fresh). The
  contract that the online and offline feature values match is what prevents training-serving skew.

### Step 4 — Features
- **Feature types:** categorical (one-hot / hashing / **embeddings** for high-cardinality IDs), numeric
  (normalize/standardize, bucketize, log-transform skew), text (tokenize → embeddings), image/audio (pretrained
  encoders), cross/interaction features, and **aggregations** (counts/averages over windows — "items bought
  last 7d").
- **Embeddings** are the workhorse for high-cardinality IDs (users, items, queries) and for retrieval.
- **Feature freshness is a spectrum:** static (profile) → slowly-changing (daily aggregates) → real-time
  (last 5 actions). Real-time features are powerful and the main source of skew — they must be computed the
  same way online and offline. Push that logic into the feature store, not duplicated in two codebases.
- **Avoid leakage:** any feature that wouldn't be available at prediction time, or that encodes the label,
  must go. Point-in-time joins enforce this.

### Step 5 — Model choice
- **Start with a baseline.** A heuristic (most-popular, rules) or a simple model (logistic regression /
  gradient-boosted trees) is the honest comparison point and often ships first. **Always propose a baseline
  before the deep model** — it's a maturity signal and it de-risks the project.
- **Pick the simplest model that hits the bar, then justify complexity:**
  | Family | Good for | Cost |
  |---|---|---|
  | Logistic regression / linear | interpretable baseline, huge sparse features | low |
  | Gradient-boosted trees (XGBoost/LightGBM) | tabular, CTR/fraud, strong default | low–med |
  | Two-tower / embedding retrieval | candidate generation at scale (ANN) | med |
  | Deep nets / wide-and-deep / DLRM | ranking with rich features & interactions | med–high |
  | Sequence/transformer | session/next-item, NLP, time series | high |
  | Pretrained LLM (API or self-hosted) | generation, RAG, agents, zero/few-shot | high (tokens) |
- **Multi-task & multi-objective:** real ranking models predict several heads (click, watch, like, share)
  and combine them with a tunable weighted score — lets product trade objectives without retraining.
- State architecture specifics when relevant: layers, embedding dims, activation, regularization (dropout,
  L2), and how you handle the loss. (See `[[ml-frameworks]]`, `[[recsys-ranking]]`.)

### Step 6 — Training pipeline
- **Components:** data extraction (point-in-time) → feature transform → train → eval/gate → **model registry**
  → (promote). Make it reproducible (versioned data + code + config) and scheduled (retrain cadence driven by
  drift, not vibes).
- **Scale knobs:** distributed data-parallel for big data; the embedding tables (not the dense net) usually
  dominate memory for recsys/ads. Use the right hardware and parallelism. (See `[[training-frameworks]]`.)
- **Retraining strategy:** full retrain vs incremental/online learning vs warm-start. Cadence: ads/feed often
  daily or faster; a stable classifier maybe weekly/monthly. Tie cadence to observed drift.
- **The registry gates promotion** on offline eval thresholds; promotion to serving is a separate, audited
  step. (See `[[mlops-lifecycle]]`.)

### Step 7 — Evaluation: offline + online (they will disagree)
- **Offline (gates launch):** choose metrics that match the task —
  - Classification: AUC-ROC, AUC-PR (use PR for imbalance), precision/recall/F1 at an operating threshold,
    **calibration** (predicted prob ≈ observed rate — critical for ads/bidding/fraud thresholds).
  - Ranking/recsys: **NDCG@k, MAP, MRR, recall@k, hit-rate@k**, diversity/coverage.
  - Regression: RMSE/MAE, MAPE.
  - LLM/generation: task evals, rubric/LLM-as-judge, exact-match/ROUGE where applicable — see
    `[[ml-evaluation-evals]]`.
- **The offline/online gap is real.** Offline metrics are computed on logged data that *the old model
  produced* — selection bias, no feedback effects. A model that wins offline can lose online. Plan for it.
- **Online (decides launch): A/B test** against the business KPI **plus guardrail metrics** (latency, error
  rate, and "do no harm" metrics — e.g. you improved CTR but tanked watch-time or increased blocks). Watch
  novelty effects; run long enough for significance; consider interleaving for ranking (lower variance than
  A/B). Shadow/canary the system before a full A/B. (See `[[ml-evaluation-evals]]`.)

### Step 8 — Serving pipeline (driven by the latency budget)
- **Choose the serving mode from the budget:**
  | Mode | When | Mechanism |
  |---|---|---|
  | **Batch / precompute** | recs that don't need per-request freshness; latency budget is hours | compute offline, store in a KV store, look up at request time |
  | **Online / real-time** | per-request prediction (CTR, fraud, ranking) with ms budget | feature fetch + model inference in the request path |
  | **Streaming / near-real-time** | features/labels from a live event stream (fraud, trending) | stream processor updates features/state continuously |
- **The candidate-generation → ranking → re-ranking funnel** is *the* pattern for large item spaces (recs,
  search, feed, ads):
  - **Candidate generation / retrieval:** millions → hundreds. Cheap, high-recall. ANN over embeddings
    (two-tower), inverted index, co-visitation, multiple sources unioned.
  - **Ranking:** hundreds → tens. Expensive, high-precision. The heavy model with rich features.
  - **Re-ranking:** apply business logic, diversity, dedup, freshness boosts, policy/safety filters.
  - This funnel is how you afford an expensive model: it only ever scores a few hundred items, not millions.
- **Latency tactics:** caching (feature cache, embedding cache, full-response cache for hot queries),
  precompute item embeddings offline, ANN indexes (HNSW/IVF), model optimization (quantization, distillation,
  batching), and async/precompute anything off the critical path. (See `[[serving-frameworks]]`,
  `[[inference-optimization]]`.)
- **Failure modes & graceful degradation:** the model server *will* fail or time out. Design the fallback —
  a cached ranking, a popularity baseline, the previous model — so the product **degrades, not fails.** Set
  per-stage timeouts; the funnel should return *something*.

### Step 9 — Monitoring and iteration (part of the design)
- **Monitor four things:** (1) system health (latency, QPS, error rate, saturation), (2) **input/feature
  drift** (distribution shift vs training), (3) **prediction drift** (output distribution moving), (4)
  **model quality** (online business metric + delayed ground-truth metrics as labels arrive).
- **Data-quality checks** at the pipeline edges catch the silent killers (a feature suddenly all-null after
  an upstream schema change).
- **Closed loop:** drift/decay → alert → retrain trigger → eval gate → canary → A/B → promote → rollback if
  guardrails trip. This loop *is* the system staying alive. (See `[[mlops-lifecycle]]`,
  `[[ml-observability-monitoring]]`.)

---

## 3. The canonical problem archetypes

Recognize which archetype you're in; each has a default architecture you can adapt.

### Recommendation / ranking
Personalized item selection from a large catalog. **Architecture:** candidate generation (two-tower ANN +
co-visitation) → ranking (multi-task deep model on click/watch/like) → re-ranking (diversity, freshness,
policy). Implicit feedback labels; cold start is central. **The reference archetype — go deep in
`[[recsys-ranking]]`.**

### Search / retrieval
Query → relevant results. **Architecture:** query understanding → retrieval (lexical inverted index +
semantic/dense ANN, often hybrid) → learning-to-rank → re-rank. Relevance labels from clicks + human
judgments; metrics NDCG/MRR. Same funnel as recs, driven by a query instead of a user context.

### Feed ranking
Mixed-content timeline. **Architecture:** candidate sources (follow graph, recommended, ads) → multi-task
ranking (engagement + integrity/quality + diversity) → re-rank with business rules and ad insertion.
Heavy on guardrails (you can wreck the ecosystem optimizing one engagement metric). Strong feedback loops.

### Ads / CTR & conversion prediction
Predict P(click) / P(conversion) to drive bidding/auction. **Architecture:** retrieval of eligible ads →
CTR model (wide-and-deep / DLRM, massive sparse features, real-time features) → auction (bid × pCTR) →
budget pacing. **Calibration is non-negotiable** (the probability feeds a money decision). Extreme class
imbalance; negative sampling + recalibration. Latency budget is tight (the auction is in the page load).

### Classification (the general case)
Spam, content moderation, churn, intent, support routing. **Architecture:** features → classifier (GBT or
fine-tuned encoder) → threshold tuned to the precision/recall tradeoff the business wants. Watch threshold
choice and calibration; cost of FP vs FN is a product decision, not a default.

### Fraud / anomaly detection
Rare, adversarial, real-time. **Architecture:** streaming features (velocity/aggregations over windows) →
fast model (GBT + rules) → real-time score → action (block/review/allow). **Streaming pipeline is core.**
Extreme imbalance; adversaries adapt so the model decays fast (retrain often); precision/recall tradeoff is
a cost decision (review capacity); often a rules layer + ML hybrid for explainability and fast response.

### Modern LLM / RAG / agent systems
Generation grounded in your data or acting via tools. **Architecture:**
- **RAG:** ingest → chunk → embed → vector index; at query time: embed query → retrieve top-k (often hybrid
  + rerank) → assemble prompt/context → LLM generate → post-process/cite. The "retrieval" box is recsys with
  a generation head; the "model" is an API or a self-hosted LLM. See `[[rag-vector-databases]]`.
- **Agents:** an LLM orchestration loop with tools, memory, and control flow. See `[[llm-app-agent-frameworks]]`.
- **What's different:** evals replace classic offline metrics (`[[ml-evaluation-evals]]`); **cost and latency
  are dominated by tokens** (context length, model tier, number of calls); **build vs buy vs API** is a
  central axis (managed LLM API vs self-hosted on `[[serving-frameworks]]`); guardrails for hallucination,
  injection, and PII are first-class; caching (prompt/response/semantic cache) is a major lever.
- **What's the same:** clarify the metric, design the data/retrieval pipeline, evaluate, serve within a
  budget, monitor and iterate. The framework holds.

---

## 4. Key design axes and tradeoffs (name these out loud)

Every design is a set of explicit tradeoffs. Stating them — and defending your pick — is the core skill.

- **Online vs batch inference.** Batch (precompute, store in KV) is cheap, simple, and fast to serve but
  stale and can't use request-time context; online is fresh and contextual but costs latency and infra.
  Many systems are hybrid: batch-precompute candidates, online-rank.
- **Latency vs throughput vs cost.** Bigger batches and bigger models raise throughput/quality but hurt
  p99 latency and $/query. The funnel exists to resolve this. Quantization/distillation trade a little
  quality for a lot of latency/cost.
- **Model complexity vs maintainability.** A deep model that's 2% better but needs a team to operate may be
  the wrong call for v1. Earn complexity. Ship the boosted-tree baseline, then iterate.
- **Build vs buy vs API.** Self-host (control, cost-at-scale, latency, privacy) vs managed API (speed to
  ship, no ops, but per-call cost, data egress, vendor lock, rate limits). Acute for LLMs.
- **Freshness vs staleness vs cost.** How stale can features/recommendations be? Real-time costs streaming
  infra and is the main skew risk; daily batch is cheap and robust. Match freshness to the actual need.
- **Feature freshness & training-serving skew.** The deepest recurring trap: features computed one way in
  the training job and another way in the serving path. Single source of truth (feature store), log serving
  features for training, and validate parity. See `[[data-engineering-feature-stores]]`.
- **Candidate gen → ranking → re-rank funnel.** Recall vs precision vs compute, staged. The architectural
  answer to "how do I afford an expensive model over millions of items."
- **Cold start.** New user/item/query with no history. Tactics: content-based features and metadata,
  popularity/demographic priors, exploration (bandits/epsilon-greedy), onboarding signals, and falling back
  to non-personalized until enough signal accrues.
- **Feedback loops.** The model shapes what users see, which shapes the next training set — runaway
  popularity, filter bubbles, degenerate feedback. Mitigate with exploration, position-bias correction
  (inverse-propensity weighting), and logging *what was shown*, not just what was clicked.
- **Scalability & failure modes (distributed-systems fundamentals).** At scale you are doing distributed
  systems: **replication** (availability of model/feature servers), **sharding/partitioning** (embedding
  tables, ANN index, feature store by key), **caching** (features, embeddings, responses — with TTLs and
  invalidation), **queues** (decouple ingestion and async scoring; backpressure), and **consistency**
  (eventual consistency for features is usually fine; be explicit where it isn't). Plan for partial failure
  and graceful degradation. (See `[[aiml-on-kubernetes]]`, `[[serving-frameworks]]`.)

---

## 5. Choosing and defending metrics

- **Three layers, always:** **business KPI** (what the org wants), **ML metric** (offline, gates the model),
  **guardrail metrics** (must-not-regress: latency, errors, integrity/quality, fairness). The skill is
  connecting them: *this offline NDCG gain should produce this watch-time lift, without regressing these
  guardrails.*
- **Pick a metric you can actually move and that resists gaming.** Raw CTR invites clickbait; pick a
  satisfaction-weighted target. A metric you can't influence with the system you're building is a red flag.
- **Operating point ≠ AUC.** AUC is threshold-free; production runs at a *threshold* chosen from the
  precision/recall tradeoff and the business cost of FP vs FN. Choose and defend it.
- **Calibration matters whenever the probability is consumed downstream** (bidding, fraud thresholds,
  expected-value ranking). A discriminative-but-miscalibrated model breaks those.
- **Experimentation:** A/B with proper sizing/power; guardrails as gates; watch novelty/primacy effects;
  consider interleaving for ranking; shadow then canary then ramp. Trust online over offline when they
  conflict. See `[[ml-evaluation-evals]]`.

---

## 6. Scale estimation (do the back-of-envelope)

A senior design states numbers; they drive the architecture. Estimate, don't agonize over precision:

- **QPS:** DAU × actions/user/day ÷ 86,400, then × peak factor (≈2–5×). 100M DAU × 20 req ≈ 2B/day ≈ ~23k
  QPS average, ~70k+ at peak.
- **Latency budget:** split the end-to-end p99 across stages (e.g. 200 ms total → 20 ms retrieval, 50 ms
  feature fetch, 100 ms ranking, 30 ms slack). The split tells you what must be precomputed/cached.
- **Data volume:** events/day × bytes/event → storage and pipeline throughput. Tells you batch vs streaming
  and how much history you can keep.
- **Embeddings / index size:** num_items × dim × 4 bytes. 100M items × 128-dim float32 ≈ 51 GB — does the
  ANN index fit in RAM on one node, or must it shard? This decides the retrieval architecture.
- **Cost (LLM/RAG):** tokens/request × $/token × QPS. Token cost usually dominates LLM serving economics and
  decides model tier, context length, caching, and build-vs-API. **Verify current pricing — it changes
  fast.**

Use these to justify sharding, caching, replication, async, and the choice of serving mode.

---

## 7. How to run the design (interview or real)

1. **Scope.** Ask clarifying questions; state the use case, users, business metric, and constraints. Resist
   designing for 60 seconds.
2. **Estimate scale.** QPS, data size, latency budget, index size, cost. Numbers on the board.
3. **State the ML problem & objective.** Task framing, proxy label, the metric hierarchy.
4. **Draw the boxes (high-level first).** Data pipeline → training → registry → serving funnel → monitoring.
   One clean diagram before any deep dive.
5. **Go deep where it matters.** Pick the highest-leverage components (usually data/labels, the funnel, and
   evaluation) and detail them. Don't deep-dive everything.
6. **Name the tradeoffs at each box** and defend your choice against the alternative.
7. **Address evolution.** Cold start, feedback loops, retraining, drift, scale-up path, and what v2 adds.
   Show you've thought past launch.

**The strongest signal is judgment under tradeoffs, not naming the fanciest model.**

---

## 8. Anti-patterns (the traps that fail designs)

- **Jumping to the model before the problem and metric.** "I'd use a transformer" before stating the
  business metric and ML objective is the #1 failure. Architecture is downstream of the problem.
- **Ignoring the data and serving story.** Hand-waving "we collect data and train" skips the part where 80%
  of the engineering and 90% of the failures live (labels, the third pipeline, training-serving skew, the
  latency budget).
- **No monitoring or iteration story.** A design that ends at "deploy the model" is incomplete — ML systems
  decay. No drift detection, retraining trigger, or rollback = not a real system.
- **Over-engineering v1.** Proposing a 5-stage deep funnel with online learning when a boosted-tree baseline
  on batch features would validate the idea. Match complexity to the stage; earn it.
- **Optimizing a metric you can't move (or that's gameable).** A proxy label disconnected from the KPI, or
  one that invites clickbait/perverse behavior, dooms the system even at high offline scores.
- **Training-serving skew / leakage left unaddressed.** Different feature logic online vs offline, or a
  feature that encodes the future. Silent, lethal, and a senior reviewer will probe for it.
- **No baseline.** Jumping to the complex model with nothing to compare against — you can't claim a win.
- **One global model where context demands segmentation** (or vice-versa: a model per tiny segment that
  can't train). Match model granularity to data volume.
- **Treating LLM/RAG as exempt from the framework.** "Just call the API" with no retrieval design, no evals,
  no cost/latency budget, no guardrails. Same rigor applies.

---

## 9. Canonical references

- **Chip Huyen, *Designing Machine Learning Systems* (O'Reilly)** — the reference book for this skill;
  framing, data, the train/serve/monitor loop, and the failure modes. Also her *AI Engineering* for the
  LLM-era systems.
- **Hello Interview — ML System Design** — structured delivery framework and worked problems:
  https://www.hellointerview.com/learn/ml-system-design
- **Patrick Halina — ML Systems Design Interview Guide:**
  https://patrickhalina.com/posts/ml-systems-design-interview-guide/
- **Google — *Rules of Machine Learning* (Martin Zinkevich):** durable engineering rules
  (baseline first, simple model, metric discipline) — https://developers.google.com/machine-learning/guides/rules-of-ml
- **Eugene Yan — applied ML/recsys system writeups:** https://eugeneyan.com/
- **Cross-skill:** `[[recsys-ranking]]`, `[[mlops-lifecycle]]`, `[[ml-evaluation-evals]]`,
  `[[data-engineering-feature-stores]]`, `[[rag-vector-databases]]`, `[[llm-app-agent-frameworks]]`,
  `[[serving-frameworks]]`, `[[aiml-on-kubernetes]]`, `[[ml-observability-monitoring]]`.

> Treat tool names, model families, vector-DB/serving features, and pricing as **fast-moving — verify
> against current docs.** The framework and the tradeoff reasoning are what to commit to memory.

---

# ML System Design — Worked Example

One design walked end-to-end through the framework in `ml-system-design-guide.md`, with the tradeoffs called
out at each step. The point is not the specific numbers (illustrative — **verify any real figure against
current data**) but the *shape of the reasoning*. A second, compressed RAG example follows to show the same
framework applied to an LLM system.

---

## Example A — "Recommend videos for the home feed"

A short-form-video app wants a personalized home feed. This is the **recommendation/ranking archetype**
(see `[[recsys-ranking]]`). We run it through all nine steps.

### Step 1 — Clarify the problem & business metric
**Questions asked:** Personalized per-user? (yes). Catalog size? (~100M videos, millions new/day). Users?
(~100M DAU). Latency? (feed must load fast — p99 budget ~200 ms for the recommendation call). What's the
current baseline? (chronological + most-popular). What's success?

**Business metric chosen:** **long-term daily watch time / retention.** Not raw clicks — optimizing clicks
would surface clickbait and hurt retention. *Naming this is the senior signal.*

**Constraints:** ~100M DAU, tight latency budget, must respect content-policy/integrity (don't amplify
harmful content), cold start for new users and new videos is core (millions of new videos/day).

### Step 2 — ML problem & objective
- **Task framing:** model it as **ranking** a candidate set of videos by expected satisfaction. Concretely,
  a **multi-task** model predicting several heads — P(click), P(watch ≥ X% / completion), P(like/share),
  P(skip) — combined into a tunable score `w1·pWatch + w2·pEngage − w3·pSkip`. Multi-task lets product trade
  objectives without retraining and aligns better with watch-time than a single click head.
- **Proxy label:** primarily **"meaningful watch"** (watched ≥ a duration/percentage threshold), not raw
  click. This is the explicit business-KPI ↔ proxy-label link.
- **Metric hierarchy:** business KPI = watch time/retention; offline ML metric = **NDCG@k / recall@k +
  calibration** of the watch head; proxy label = meaningful-watch events.

### Step 3 — Data, labels, the third pipeline
- **Sources:** interaction logs (impressions, clicks, watch duration, likes, skips), user profile/history,
  video content features (creator, topic, embeddings of title/thumbnail/audio-visual), context (time, device).
- **Labels — implicit feedback** from watch logs. **Biases to handle:** position bias (top items get more
  watches regardless of quality) and **exposure bias** (we only observe outcomes for videos we *showed* —
  the candidate generator's blind spots never get labels). Mitigations: log *what was shown* (impressions,
  not just engagements), inverse-propensity weighting / position features, and exploration in serving.
- **The data/feature pipeline (third pipeline):** stream watch events → clean/dedupe → **point-in-time
  correct** joins to build labels (no peeking at the future) → compute features → write to a **feature
  store** that serves identical features to training (historical) and serving (online). This contract is
  what prevents **training-serving skew** — the deepest trap here. (`[[data-engineering-feature-stores]]`)

### Step 4 — Features
- **User:** embedding (learned from interaction history), recent-watch sequence (last N video embeddings —
  a *real-time* feature, the main skew risk), demographics, long-term topic affinities.
- **Video:** ID embedding, creator embedding, content embeddings (multimodal), age, aggregate watch-through
  rate, topic.
- **Context:** time-of-day, device, session position.
- **Cross/aggregate:** user-topic affinity, user×creator history, counts over windows.
- **Leakage guard:** the candidate's *future* watch-through can't be a feature; only-available-at-serve-time
  values, enforced by point-in-time joins.

### Step 5 — Model choice
- **Baseline first:** most-popular + a gradient-boosted-tree CTR model on basic features. Honest comparison
  point; likely the first thing in production.
- **Retrieval model:** **two-tower** (user tower, video tower) trained so the dot product approximates
  affinity → enables **ANN retrieval** over 100M video embeddings.
- **Ranking model:** a **deep multi-task network** (shared bottom + per-task heads) over the rich feature
  set, scoring the few hundred candidates. *Tradeoff:* a heavier transformer over the watch sequence is ~x%
  better but costs latency/ops — **earn it in v2**; ship the multi-task DNN first.

### Step 6 — Training pipeline
- Point-in-time feature extraction → transform → distributed train (embedding tables dominate memory, not
  the dense net — shard them) → offline eval gate → **model registry** → promote.
- **Retraining cadence:** frequently (daily or faster) — feeds drift quickly and new content floods in.
  Warm-start from the previous model to converge fast. Cadence is justified by observed drift, not habit.
  (`[[training-frameworks]]`, `[[mlops-lifecycle]]`)

### Step 7 — Evaluation: offline + online
- **Offline (gates launch):** **recall@k** for the retrieval tower (did the right videos make the candidate
  set?), **NDCG@k** for ranking, and **calibration** of the watch head. Computed on logged data — biased by
  what the old model showed, so treat as a gate, not the verdict.
- **Online (decides launch):** **A/B test on watch time/retention** + **guardrails**: p99 latency, error
  rate, **integrity** (no rise in policy-violating watches), and diversity (don't collapse into one topic —
  a feedback-loop risk). Consider **interleaving** for the ranking change (lower variance than A/B). Watch
  the novelty effect; run long enough for retention significance. Shadow → canary → ramp. (`[[ml-evaluation-evals]]`)

### Step 8 — Serving pipeline (latency budget ~200 ms)
The **candidate generation → ranking → re-ranking funnel**:
1. **Candidate generation (100M → ~1000):** union of sources — two-tower **ANN** (HNSW) over precomputed
   video embeddings, co-visitation, follow-graph, trending/fresh, and an **exploration** slice for cold
   videos. Cheap, high-recall, ~20 ms.
2. **Ranking (~1000 → ~50):** the multi-task DNN, scoring only the candidates. Fetch user/video features
   from the **feature store** (cache hot users/videos), ~100 ms. *This is why the funnel exists* — the
   expensive model never touches 100M items.
3. **Re-ranking (~50 → feed):** diversity/dedup, freshness boost, **policy/integrity filters**, ad insertion.

**Serving-mode tradeoffs:** video embeddings are **batch-precomputed** offline (don't change per request);
the user-context score is **online** (needs recent-watch sequence). Hybrid. **Latency tactics:** ANN index,
feature/embedding caches, model batching/quantization, precompute everything off the critical path.
**Graceful degradation:** if the ranker times out, serve the candidate set ordered by a cheap score or a
cached/popularity ranking — **degrade, don't fail.** (`[[serving-frameworks]]`, `[[inference-optimization]]`)

### Step 9 — Monitoring & iteration
- **System:** p99 latency, QPS, error rate, ANN/feature-store saturation.
- **Drift:** feature-distribution shift (new content trends), prediction-distribution shift.
- **Quality:** online watch time/retention + delayed labels as watches resolve; **integrity & diversity
  guardrails.**
- **Data quality:** null-rate/schema checks at pipeline edges (an upstream change nulling a feature is the
  classic silent killer).
- **Loop:** drift/decay → retrain trigger → eval gate → canary → A/B → promote, with rollback if a guardrail
  trips. (`[[mlops-lifecycle]]`, `[[ml-observability-monitoring]]`)

### Scale sanity-check (illustrative)
- **QPS:** 100M DAU × ~20 feed loads/day ≈ 2B/day ≈ ~23k QPS avg, ~70k+ peak → ranking must shard + cache.
- **Index:** 100M videos × 128-dim float32 ≈ ~51 GB → ANN index **must shard** across nodes; can't sit on one.
- **Latency split:** 200 ms budget → ~20 ms retrieval + ~100 ms ranking (incl. feature fetch) + ~30 ms
  re-rank + slack. The split *dictates* what is precomputed vs online.

### Tradeoffs called out (the deliverable)
| Decision | Chose | Over | Why |
|---|---|---|---|
| Proxy label | meaningful-watch | raw click | clicks → clickbait, hurts retention KPI |
| Funnel | retrieve→rank→rerank | rank everything | can't run a deep model over 100M items in 200 ms |
| Video embeddings | batch-precompute | online | don't change per request → cheaper, faster |
| Ranking model v1 | multi-task DNN | sequence transformer | earn complexity later; ship the simpler win |
| Retrain cadence | daily/warm-start | weekly full | fast feed + content drift demand freshness |
| On ranker failure | cached/popularity fallback | error | degrade, don't fail |

---

## Example B — "Company-docs Q&A assistant" (RAG), compressed

Same framework, LLM archetype (`[[rag-vector-databases]]`, `[[llm-app-agent-frameworks]]`).

- **Problem & metric.** Help employees get correct answers from the company knowledge base. **Business metric:**
  support-ticket deflection / time-to-answer. **Guardrails:** no confident-but-wrong answers, no leaking
  restricted docs. Constraint: answer within a few seconds; bounded cost/query.
- **ML problem.** Retrieval-augmented **generation**, not training a model. "Objective" = grounded,
  cited, correct answers. No proxy-label training; success is measured by **evals**.
- **Data pipeline.** Ingest docs → chunk → embed → **vector index**, with per-doc **access-control metadata**
  carried through retrieval (a doc the user can't see must never be retrievable for them). Re-index on doc
  changes — freshness is a real requirement.
- **Retrieval + model.** Query → embed → **hybrid retrieval** (dense + lexical) → **rerank** top-k → assemble
  prompt with retrieved context → **LLM generate** with citations → post-process. *Build vs buy vs API:*
  managed LLM API (ship fast, per-token cost, data-egress/privacy review) vs self-hosted on
  `[[serving-frameworks]]` (control, cost at scale). Decide on cost + privacy.
- **Evaluation.** **Offline evals** replace classic metrics: a labeled eval set scored for
  **retrieval quality** (did we fetch the right chunks?) and **answer quality** (correctness/groundedness,
  rubric or LLM-as-judge), plus hallucination/refusal rate. **Online:** A/B on deflection + thumbs +
  guardrails. (`[[ml-evaluation-evals]]`)
- **Serving & cost.** Latency and **cost are dominated by tokens** (context length × model tier × calls) —
  the funnel here is *retrieve few, rerank, send a tight context*. **Semantic/prompt caching** for repeat
  questions. Guardrails for prompt injection, PII, and access control. Verify model pricing/limits against
  current docs.
- **Monitoring & iteration.** Track retrieval hit-rate, answer ratings, hallucination flags, cost/query,
  latency; grow the eval set from real failures; re-index on doc churn; loop. Same shape as Example A.

**Takeaway:** the archetype and the boxes differ, but the framework — clarify the metric, design the data
pipeline, evaluate, serve within a budget, monitor and iterate — is identical.
