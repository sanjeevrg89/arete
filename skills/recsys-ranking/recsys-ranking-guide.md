# Recommender & Ranking Systems — Field Guide

Recommendation and ranking is the highest-revenue ML in industry: feeds (social, video, music),
search ranking, ads, and e-commerce recommendations all reduce to *"given a user and context, order
a huge catalog so the best items come first."* The corpus is millions to billions of items, the
budget is tens of milliseconds, the labels are biased by what you previously showed, and the metric
you can measure offline is not the metric the business cares about. This guide is the production
mental model plus the specific traps.

This is a fast-moving field (it is 2026). Architectures and especially the LLM/generative-recsys
frontier change quarterly — treat specific model names and any numbers as **verify against current
docs/papers**, not gospel.

---

## 1. Mental model: the multi-stage funnel

You cannot run a heavyweight model over millions of items per request inside ~tens of ms. So every
serious system is a **funnel** that progressively narrows the candidate set, spending more compute per
item as the set shrinks:

```
corpus (10^6–10^9 items)
   │  retrieval / candidate generation        ~O(10^6 → 10^3)   recall-oriented, very cheap/item
   ▼
candidates (10^3)
   │  pre-ranking (optional lightweight rank)  ~O(10^3 → 10^2)   distilled/cheap ranker
   ▼
shortlist (10^2)
   │  ranking (the expensive, precise model)   ~O(10^2)          rich features, deep model
   ▼
ranked (10^2)
   │  re-ranking (list-level)                  diversity, dedup, freshness, MMR, sequence model
   ▼
   │  policy / business rules                  ads load, compliance, blocklists, pinning, caps
   ▼
final page (10^1)
```

**Why staged:** the latency/quality budget. Retrieval must touch the whole corpus, so it must be
cheap per item (embedding dot product + ANN, inverted-index lookups) and is tuned for **recall** — its
job is "don't lose the good items." Ranking sees only ~hundreds of candidates, so it can afford a
heavy model with hundreds of features and is tuned for **precision** at the top. Re-ranking operates on
the *list as a whole* (diversity, dedup, business mix) because pointwise ranking can't see the page.

**Budget discipline.** Give each stage an explicit p99 latency and a quality target. A rough split for
a feed surface might be: retrieval a few ms, pre-ranking a few ms, ranking 10–30 ms, re-ranking a few
ms — but the exact budget is *yours to measure*. The funnel only works if each stage's recall is high
enough that the next stage isn't starved of good candidates. The metric that ties stages together is
**stage recall**: fraction of the items the *final* model would have wanted that survived each upstream
cut. If retrieval recall@1000 vs the ranker's top-10 is low, no ranking improvement can save you.

---

## 2. Retrieval / candidate generation

Goal: from the full corpus, return ~hundreds–thousands of plausibly-relevant candidates, fast, with
high recall. You almost always **blend several retrieval sources** and dedupe — no single source covers
the space (personalized, fresh, popular, exploratory, follow-graph, query-matched).

### 2.1 Collaborative filtering / matrix factorization

The classic: factor the (sparse) user×item interaction matrix `R ≈ U Vᵀ` into low-rank user and item
embeddings. **ALS (Alternating Least Squares)** is the scalable workhorse for implicit feedback
(clicks/plays, not ratings) — it alternates solving for U with V fixed and vice versa, each a set of
independent least-squares problems, trivially parallelizable. Implicit-feedback ALS weights observed
interactions by confidence and treats unobserved as weak negatives (Hu, Koren & Volinsky 2008).

- Strengths: simple, strong baseline, cheap to serve (precomputed item embeddings + ANN or
  precomputed per-user top-K).
- Weaknesses: **pure CF can't use content/context features** → brutal cold start for new users/items;
  it only knows the interaction graph.

### 2.2 Two-tower / dual-encoder (the modern default)

Two separate networks — a **user/query tower** and an **item/candidate tower** — each producing an
embedding in a shared space; relevance = dot product (or cosine). Train so that a positive (user, item)
pair scores higher than negatives, typically with **sampled softmax**:

```
score(u, i) = <f_user(u_features), f_item(i_features)>
loss        = sampled-softmax / contrastive over (positive item vs sampled negatives)
```

The decisive property: the **item tower is independent of the user**, so you precompute *all* item
embeddings offline, build an **ANN index** over them, and at serving time embed the user once and do an
approximate nearest-neighbor search. That is what makes corpus-scale retrieval in single-digit ms
possible. See `[[rag-vector-databases]]` for ANN indexes (HNSW, IVF-PQ, ScaNN) and their recall/latency
tradeoffs.

Two-tower with in-batch sampled softmax and the log-Q correction is the canonical recipe (Yi et al.,
"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations," RecSys 2019 — verify).

**Negative sampling — the single most important design choice in retrieval:**

- **In-batch negatives:** within a training batch, every other item is a negative for a given user.
  Free and effective, but the negative distribution is biased toward **popular** items (they appear in
  more batches), so you must apply a **log-Q / sampled-softmax correction** subtracting the estimated
  log-probability of sampling each item. Skip the correction and you systematically suppress popular
  items.
- **Hard negatives:** in-batch negatives are mostly *easy* (obviously irrelevant), so the model never
  learns fine distinctions. Mine **hard negatives** — items that are close to relevant but wrong (e.g.
  high-scoring non-clicked, or near-neighbors of the positive). A blend of easy + hard is standard;
  all-hard destabilizes training.
- **Sampled softmax** over a sampled negative set (rather than full softmax over millions of items)
  makes the loss tractable.

A two-tower model that *concatenates* user and item early (so they interact before the dot product) is
NOT a two-tower model for retrieval — you lose precomputability. Early interaction belongs in
**ranking**, not retrieval.

### 2.3 Graph-based and heuristic sources

- **Graph / co-occurrence:** random walks (node2vec-style), "users who interacted with X also
  interacted with Y," session-based co-visitation. Strong for "more like this" and session continuation.
- **Heuristic / rule sources:** trending/popular, freshly published, geo/locale, followed authors,
  recently-viewed-category, query term match (for search: an inverted index / BM25 / lexical retrieval
  is itself a retrieval source). These are cheap, interpretable, and cover the cold-start and freshness
  gaps that learned embeddings miss.

Blend, dedupe, and tag each candidate with its source(s) — source is a useful ranking feature and a
debugging lifeline.

---

## 3. Ranking models

Ranking sees ~hundreds of candidates and orders them precisely using a **rich feature vector** per
(user, item, context). Here, unlike retrieval, **early feature interaction is the point**.

### 3.1 GBDTs vs deep learning

- **Gradient-boosted decision trees** (XGBoost / LightGBM, often with LambdaMART for ranking) are a
  *very* strong baseline on tabular, dense, lower-cardinality features. Fast to train, robust, handle
  monotonic constraints, easy to reason about. Many production rankers are still GBDTs and it's often
  the right call. GBDTs struggle with extremely high-cardinality sparse IDs and with reusing learned
  embeddings.
- **Deep rankers** win when you have **massive sparse categorical features** (millions of item/user/ad
  IDs), want **learned embeddings** and **explicit feature crosses**, or need multi-task/multi-objective
  heads. The cost is training/serving infrastructure and tuning.

Choose for *your* feature profile and latency budget. "Deep because it's modern" is an anti-pattern;
so is "GBDT forever" when you have billions of sparse IDs.

### 3.2 The deep-ranker lineage (know what each adds)

| Model | arXiv (verify) | Key idea / what it adds |
|---|---|---|
| **Wide & Deep** | 1606.07792 | Jointly train a *wide* linear model (memorization of feature crosses) + a *deep* network (generalization). The template for hybrid rankers. |
| **DeepFM** | 1703.04247 | Replaces manual wide crosses with a **factorization machine** for 2nd-order feature interactions, sharing embeddings with the deep part. No hand-crafted crosses. |
| **DCN / DCN-v2** | 1708.05123 / 2008.13535 | **Cross Network** explicitly learns bounded-degree feature crosses; DCN-v2 makes the cross layers low-rank and more expressive for web-scale. |
| **DLRM** | 1906.00091 | Meta's deep learning recommendation model: big embedding tables for categorical features + MLPs + pairwise dot-product interaction. The reference for **sparse-embedding-heavy** click prediction. |
| **Transformer-based rankers** | (e.g. BST, behavior-sequence transformers; verify current) | Self-attention over the user's **behavior sequence** and/or over the candidate set; captures long/ordered user history and list context. The current frontier blends sequence modeling and generative approaches. |

These are reference points, not a tier list. The right model is the simplest one that captures the
feature interactions that matter for your data, within latency.

### 3.3 Multi-task & multi-objective

Real surfaces optimize **several objectives at once** — e.g. click *and* dwell/satisfaction *and*
share *and* "not-a-regret," plus diversity. Two reasons: (a) different labels are noisy/biased in
different ways, (b) the business cares about long-term value, not one click.

- **Hard parameter sharing** (shared bottom, per-task heads) is the simple baseline but suffers when
  tasks conflict (negative transfer).
- **MMoE (Multi-gate Mixture-of-Experts, Ma et al. KDD 2018 — verify):** shared experts with a
  per-task gating network, so tasks can use experts differently. Reduces negative transfer.
- **PLE (Progressive Layered Extraction, Tan et al. RecSys 2020 — verify):** separates **shared** vs
  **task-specific** experts explicitly and stacks extraction layers; generally more robust than MMoE
  when tasks conflict.
- **Combining heads into one score:** the per-task predictions are blended into a final ranking value
  — a weighted/learned combination (often `Σ wₖ · pₖ`, sometimes products for multiplicative effects).
  The weights are a **product/policy decision** and are tuned via online experiments against the
  north-star metric, *not* by offline loss. This value model is where "what do we actually want the
  feed to be" lives.

### 3.4 Learning-to-rank: pointwise / pairwise / listwise

- **Pointwise:** predict an absolute label per item (pCTR, pConversion) and sort by it. Simplest,
  composes naturally with calibration and multi-objective blending. Most production click models are
  pointwise.
- **Pairwise:** learn that item A should rank above item B (RankNet, LambdaRank, LambdaMART). Optimizes
  *relative* order, closer to what ranking actually needs.
- **Listwise:** optimize a list-level metric directly (ListNet, ListMLE, LambdaMART approximating
  nDCG). Best aligned with nDCG but heavier and trickier to train.

Pointwise + good calibration + a value model is the pragmatic default; pairwise/listwise (LambdaMART)
shine when you have explicit graded relevance and care about exact top-K order (classic search ranking).

---

## 4. Features & data

### 4.1 Feature taxonomy

- **User features:** demographics (sparingly/responsibly), long-term preference embeddings, aggregate
  behavior (historical CTR, category affinities), and the **behavior sequence** (last-N interactions)
  — sequence features are among the most powerful.
- **Item features:** content embeddings (text/image/video via `[[multimodal-ml]]`), category, author,
  age/freshness, historical engagement stats, price/inventory (e-commerce).
- **Context features:** time of day, day of week, device, surface/placement, session position, query
  (search), preceding items on the page.
- **Cross features:** user×item affinity, "has this user engaged with this author before," geo match.
  These crosses are exactly what DCN/DeepFM learn automatically.

### 4.2 Real-time vs batch features and the feature store

Features split by freshness need:

- **Batch/offline features:** aggregates computed periodically (7-day item CTR, user category
  affinities). Cheap, but stale.
- **Real-time/streaming features:** counters updated within seconds (last-5-minute CTR, "items the user
  just clicked this session"). Essential for freshness and reactivity but operationally hard.

A **feature store** (`[[data-engineering-feature-stores]]`) serves the *same* feature definitions to
training (offline, point-in-time-correct) and serving (online, low-latency lookup). This is the only
sane way to keep them consistent.

### 4.3 Training-serving skew (read this twice)

**Train/serve skew is the most common silent cause of "great offline, flat online."** It happens when
the features (or labels) a model trains on differ from what it sees at serving:

- Different transform code/library offline vs online → subtly different values.
- **Time-travel leakage:** training on a feature value that wasn't available at the moment of the
  impression (e.g. using an item's *final* engagement count instead of its value at request time).
  Always do **point-in-time-correct** joins.
- Labels attributed at the wrong moment, or features computed after the label event.

**The robust fix:** *log features at serving time* and train on exactly those logged feature vectors
("log-and-train" / feature logging). Then there is no second code path to drift. Where you must
recompute offline, share the *same* transform code between train and serve and add skew-detection
monitoring (`[[ml-observability-monitoring]]`).

### 4.4 The massive sparse-embedding-table problem

The dominant cost in deep recsys is **embedding tables** for high-cardinality IDs (item, user, ad,
cross-feature). Tables can reach hundreds of GB to terabytes — far bigger than the dense network. Tactics:

- **Hashing trick:** hash IDs into a fixed-size table (the "hashing trick"), accepting collisions to
  bound memory. Double-hashing / quotient-remainder reduces collision damage.
- **Embedding sharding / model parallelism:** shard tables across hosts/accelerators
  (parameter-server style, or modern sharded embedding via e.g. TorchRec — verify current). Dense
  compute is data-parallel; sparse tables are model-parallel. See `[[ml-frameworks]]`.
- **Frequency-aware sizing & pruning:** rare IDs get tiny/shared embeddings; prune stale rows. Mixed-
  dimension embeddings allocate capacity to the heads of the long tail.
- **Quantization / compression** of embeddings for serving.

---

## 5. Evaluation

### 5.1 Offline metrics (necessary, never sufficient)

These measure ranking quality on logged data; use them to **filter** candidate models, not to decide
launches.

| Metric | What it measures | Use |
|---|---|---|
| **AUC / LogLoss** | Pointwise classification quality / calibration of pCTR | Click models. **GAUC** (per-user AUC, averaged) is more meaningful than global AUC for ranking, since global AUC rewards separating *users* not *items within a user*. |
| **Recall@k** | Did the relevant items survive into the top-k | **Retrieval** quality (the funnel-recall metric). |
| **Precision@k / MAP** | Relevance density in top-k | Set-relevance. |
| **MRR** | Reciprocal rank of the first relevant item | When the first good hit dominates (search, QA-like). |
| **nDCG@k** | Graded relevance, position-discounted | The gold-standard ranking metric for ordered lists. |

### 5.2 The offline-online gap

Offline metrics are computed on **logged data the old model produced**, so they're biased toward what
was already shown, ignore presentation/feedback effects, and can't measure long-term value. A model can
win offline and lose online (or vice versa). **Online A/B is the source of truth.**

### 5.3 Online evaluation

- **A/B test** with proper randomization (usually by user/cohort, not request, to avoid contamination),
  enough power, and a fixed duration covering weekly cycles. Watch:
  - immediate engagement: **CTR**, conversions, watch time, plays;
  - **satisfaction / quality**: dwell, completion, surveys, "see fewer like this," reported-content
    rate;
  - **long-term / north-star**: retention, DAU/WAU, sessions, lifetime value — the metrics that
    actually pay. Short-term CTR wins that hurt retention are *losses*. Use holdback/long-term
    holdout groups to catch these.
- **Guardrail metrics:** latency, diversity, creator/seller fairness, abuse rate — a launch that wins
  engagement but blows a guardrail does not ship.

### 5.4 Counterfactual / off-policy evaluation (OPE)

Before a live test, estimate how a *new* policy would have performed using logs from the *old* policy.
**Inverse Propensity Scoring (IPS)** reweights logged outcomes by `1/p(shown | old policy)`;
**doubly-robust** estimators combine IPS with a reward model to cut variance. OPE is noisy (high
variance when propensities are small) but lets you prune bad candidates before spending A/B traffic.
This is why **logging propensities / showing some randomization** at serving time is so valuable — it
makes OPE possible. See `[[ml-evaluation-evals]]`.

### 5.5 Position bias and debiasing

Users click higher positions more **regardless of relevance**. If you train pCTR on raw clicks, the
model learns to predict *position/layout*, not relevance — and since it then controls position, you get
a **self-fulfilling feedback loop**. Mitigations:

- **Position as a feature** at training time, set to a neutral/fixed value (e.g. position 1, or a
  learned "examination" constant) at serving — so the model factors out position.
- **IPS for ranking:** weight each click by inverse examination propensity (estimated via result
  randomization, intervention experiments, or an EM-based examination model).
- **Two-tower examination/relevance factorization** (e.g. PAL / position-bias-aware learning — verify).

Train position-bias-blind and your metrics will look great while relevance quietly degrades.

---

## 6. Production concerns

### 6.1 Serving latency & freshness

- Ranking is the latency hot spot: batch candidate scoring, quantize, distill the ranker, cache
  user/item embeddings. Co-locate the ANN index with the retrieval service.
- **Candidate freshness:** new items must enter the retrieval index quickly (near-real-time embedding +
  index upsert) or they're invisible — a freshness pipeline is part of the system, not an afterthought.
  See `[[serving-frameworks]]`, `[[aiml-on-kubernetes]]`.

### 6.2 Feedback loops & filter bubbles

The model trains on data it generated by choosing what to show. Left unchecked this **narrows** the feed
(filter bubble), **amplifies popularity**, and starves the model of counterfactual signal. You must
deliberately inject diversity (re-ranking, MMR, category caps) and exploration.

### 6.3 Exploration / exploitation & bandits

Pure exploitation is a trap: you never learn about items/users you don't already show. Add exploration:

- **ε-greedy / dithering:** show some random/perturbed-rank items.
- **Bandits:** Thompson sampling or UCB over arms (items/creatives), especially for **ads creatives**
  and **cold-start items** where you need to learn reward fast. **Contextual bandits** condition on
  user/context features.
- Exploration is also what feeds OPE and breaks feedback loops. Budget it explicitly (a small % of
  traffic/slots); measure its long-term payoff, not its short-term CTR cost.

### 6.4 Cold start

- **New item:** no interactions → CF/embedding retrieval can't place it. Lean on **content features**
  (content-tower embeddings), heuristic/fresh sources, and exploration to gather signal fast.
- **New user:** no history → popularity/trending + onboarding signals + context; personalize as signal
  accrues. A two-tower user tower built only from content/context (not ID) degrades gracefully.

### 6.5 Popularity bias

Popular items get shown more → get more clicks → look even better to the model → shown even more.
Counter with the log-Q correction in retrieval, popularity-debiasing in ranking (e.g. popularity as a
feature you can neutralize, or sampling corrections), diversity in re-ranking, and exploration for the
tail.

### 6.6 Calibration

When the score feeds a downstream decision — **ad auctions** (bid = pCTR × value), multi-objective
blending, thresholding — it must be **calibrated**: predicted pCTR ≈ actual click rate. Ranking-only
training (pairwise/listwise) doesn't guarantee calibration; add a calibration layer (Platt/isotonic) or
train pointwise with LogLoss and monitor calibration (reliability curves, ECE) in production.

### 6.7 Generative / LLM-augmented recsys (frontier — verify current)

An active, fast-moving direction (2025–2026). Treat all of this as **verify against current research**:

- **LLMs as feature generators / encoders:** rich text/semantic embeddings for items and queries;
  zero-/few-shot cold-start understanding; reasoning over user history.
- **Generative retrieval / semantic IDs:** represent items as token sequences (semantic IDs) and
  *generate* candidate IDs autoregressively instead of ANN lookup (e.g. TIGER-style approaches —
  verify). Promising; not yet a universal replacement for the funnel.
- **LLM-as-ranker / reranker:** prompt or fine-tune an LLM to rerank a shortlist; powerful but
  latency/cost-heavy — usually confined to the final, small re-ranking stage.
- Reality check: classic two-tower + deep ranker funnels still dominate large-scale production on
  latency/cost grounds. LLM components are added surgically where their semantics pay for the cost.

---

## 7. Anti-patterns / gotchas

- **Offline-metric tunnel vision.** Tuning to nDCG/AUC and shipping without an A/B, or trusting offline
  to predict the launch. Offline filters; online decides.
- **Train/serve skew.** Different feature code paths, non-point-in-time labels, recomputed-offline
  features that drift from serving. Log-and-train; share transforms.
- **Position-bias-blind training.** Training pCTR on raw clicks with no position handling → the model
  predicts layout and the feedback loop locks it in.
- **No exploration / pure exploitation.** Filter bubbles, frozen popularity, starved cold-start items,
  and no counterfactual data for OPE.
- **Optimizing CTR only.** Degenerate clickbait/outrage feeds, dwell and retention rot. Optimize a
  multi-objective value, validate on long-term metrics.
- **Ignoring long-term value.** Shipping every short-term-engagement win; use long-term holdouts.
- **Early-interaction "two-tower."** Crossing user and item before the dot product kills
  precomputability and corpus-scale retrieval.
- **Skipping the sampling correction.** In-batch negatives without log-Q correction systematically
  suppress popular items (or, conversely, over-recommend the tail).
- **Easy-negatives-only retrieval.** No hard negatives → a retriever that can't distinguish good from
  mediocre.
- **Uncalibrated scores into an auction/blend.** Good ranking, broken bidding/blending economics.
- **Letting retrieval recall rot.** Spending all effort on the ranker while upstream recall silently
  caps the whole funnel. Monitor stage recall.
- **One mega-model instead of a funnel.** Trying to score the whole corpus with the heavy ranker —
  blows the latency budget.

---

## 8. Troubleshooting (symptom → likely cause → fix)

- **Great offline (nDCG/AUC up), flat or negative online.** → Train/serve skew, position-bias artifact,
  or offline metric not aligned with the north-star. → Audit feature logging & point-in-time joins;
  add position handling; check the value/objective the A/B measures.
- **Ranking improvements plateau.** → Retrieval recall is the bottleneck. → Measure stage recall@k;
  improve/blend retrieval sources before touching the ranker.
- **Feed collapses to popular/clickbait over time.** → Feedback loop + CTR-only objective + no
  exploration. → Add multi-objective value, diversity re-ranking, exploration; check popularity debias.
- **New items/creators get no traffic.** → Cold start + popularity bias + no exploration. → Content-
  based retrieval, fresh source, bandit exploration budget.
- **pCTR looks well-ranked but ad revenue / blend is off.** → Miscalibration. → Reliability curves/ECE;
  add isotonic/Platt calibration; train pointwise LogLoss.
- **Retrieval recall good, ranking weirdly prefers obscure items.** → Missing/incorrect log-Q
  correction or popularity feature leakage. → Re-check sampling correction and feature neutralization.
- **Latency p99 spikes under load.** → ANN index or embedding-table lookups. → Profile per stage; shard
  embeddings, tune ANN params (recall/latency), batch scoring, distill the ranker. See
  `[[serving-frameworks]]`.
- **Training instability in two-tower.** → Too many hard negatives / bad temperature in softmax. →
  Blend easy+hard, tune softmax temperature, verify the log-Q correction.

---

## 9. Version awareness

Recsys moves fast, especially at the model and LLM-augmentation frontier (2026). Treat the following as
**verify against current docs/papers** before relying on them:

- Specific deep-ranker variants and which one is "best" — benchmark on *your* data; published leaderboard
  wins rarely transfer.
- Generative-retrieval / semantic-ID / LLM-ranker techniques — an active research area; APIs and SOTA
  shift quarterly.
- Framework specifics for sparse-embedding sharding (e.g. TorchRec and equivalents) — verify current
  versions and capabilities in `[[ml-frameworks]]`.
- ANN library/index choices and tunables — see `[[rag-vector-databases]]` and verify current.
- **Never trust a benchmark number you didn't reproduce.** No fabricated CTR/latency/recall figures.

---

## 10. Canonical references (real URLs; verify currency)

- Wide & Deep Learning — https://arxiv.org/abs/1606.07792
- DeepFM — https://arxiv.org/abs/1703.04247
- Deep & Cross Network (DCN) — https://arxiv.org/abs/1708.05123
- DCN-v2 — https://arxiv.org/abs/2008.13535
- DLRM (Deep Learning Recommendation Model) — https://arxiv.org/abs/1906.00091
- MMoE (Modeling Task Relationships in Multi-task Learning) — KDD 2018 (Ma et al.) — verify.
- PLE (Progressive Layered Extraction) — RecSys 2020 (Tan et al.) — verify.
- Sampling-bias-corrected two-tower / sampled-softmax retrieval — Yi et al., RecSys 2019 — verify.
- Implicit-feedback matrix factorization (ALS) — Hu, Koren & Volinsky, 2008 — verify.
- Chip Huyen, *Designing Machine Learning Systems* (O'Reilly) — the practitioner reference for the
  system-level concerns (feature stores, train/serve skew, online eval, the funnel framing).
- Cross-links: `[[ml-system-design]]`, `[[ml-evaluation-evals]]`, `[[data-engineering-feature-stores]]`,
  `[[rag-vector-databases]]`, `[[ml-frameworks]]`, `[[serving-frameworks]]`, `[[aiml-on-kubernetes]]`,
  `[[ml-observability-monitoring]]`, `[[multimodal-ml]]`.
