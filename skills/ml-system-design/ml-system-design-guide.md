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
