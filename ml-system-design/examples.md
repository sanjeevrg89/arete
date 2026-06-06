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
