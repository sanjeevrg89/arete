# AGENTS.md — ML System Design

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ml-system-design-guide.md`** next to this file — read it
> before designing an ML system or doing an ML-system-design interview, and apply it. A complete worked
> design is in **`examples.md`**. This file is the always-on summary.
>
> This is the "tie it all together" architect skill: turn a fuzzy business need into a running ML system
> (real-world *or* the design interview). The job is never "name a model" — it is objective → data/pipelines
> → evaluation → serving budget → monitoring/iteration. Tool names move fast (it is 2026) — **verify
> against current docs**; never fabricate benchmarks or pricing.

## When designing an ML/AI system or doing an ML-system-design interview, apply these by default:

- **Scope before you design. Never jump to the model.** Pin the **business metric** you're moving and the
  constraints (scale, QPS, latency budget, cost, privacy) first. A model you can't tie to a moveable KPI is
  a red flag.
- **Translate to a precise ML problem + ML objective.** Name the task (classification / regression / ranking
  / retrieval / generation) and the exact loss. Keep three layers distinct: **business KPI**, **offline ML
  metric**, **proxy label** — and say how they relate. Pick a proxy that resists gaming (raw clicks →
  clickbait).
- **Design three pipelines, not two.** Beyond **train** and **serve** there is the **data/feature pipeline**
  feeding both, plus a **feedback loop** (predictions → tomorrow's labels). Most production failures are
  pipeline failures: **training-serving skew, label leakage, stale features.** Make the data pipeline a
  first-class box. (`[[data-engineering-feature-stores]]`)
- **Always propose a baseline** (heuristic or logistic/GBT) before the deep model. Earn complexity; ship the
  simplest thing that hits the bar.
- **Evaluate offline AND online — they disagree.** Offline (AUC/PR, NDCG/recall@k, calibration, RMSE, LLM
  evals) gates launch; online **A/B vs business KPI + guardrail metrics** decides it. Plan for the
  offline/online gap; shadow → canary → A/B. (`[[ml-evaluation-evals]]`)
- **Serving mode follows the latency budget:** batch/precompute vs online/real-time vs streaming. Large item
  spaces use the **candidate generation → ranking → re-ranking funnel** (cheap recall → expensive precision
  → business/diversity/policy). It's how you afford an expensive model. (`[[serving-frameworks]]`)
- **Name the tradeoff axes out loud and defend each pick:** online vs batch, latency vs throughput vs cost,
  complexity vs maintainability, build vs buy vs API, freshness vs staleness.
- **Plan cold start, feedback loops, and failure modes.** New users/items (content features, popularity,
  exploration/bandits); position-bias correction and logging *what was shown*; and **graceful degradation** —
  on model/timeout failure fall back to a cached/popularity ranking, degrade don't fail.
- **Estimate scale early:** QPS, data volume, p99 latency split across stages, embedding/index size, $/query
  (token cost for LLMs). Numbers drive sharding, caching, replication, async, and serving mode.
- **At scale it's distributed systems:** replication, sharding/partitioning, caching (with TTL/invalidation),
  queues/backpressure, consistency (eventual is usually fine — be explicit where it isn't).
- **Monitoring + iteration are part of the design:** system health, input/feature drift, prediction drift,
  model quality; data-quality checks; retraining trigger + eval gate + rollback. (`[[mlops-lifecycle]]`,
  `[[ml-observability-monitoring]]`)
- **LLM/RAG/agent systems use the same framework** with new boxes: retrieval + vector index, prompt/context
  assembly, generation model (API vs self-hosted), **evals as the offline metric**, **cost/latency dominated
  by tokens**, and hallucination/injection/PII guardrails. (`[[rag-vector-databases]]`,
  `[[llm-app-agent-frameworks]]`)

## Anti-patterns to refuse
Jumping to a model before the problem/metric · ignoring the data & serving story · no monitoring/iteration
story · over-engineering v1 · optimizing a metric you can't move or that's gameable · leaving
training-serving skew / leakage unaddressed · no baseline · treating LLM/RAG as exempt from the framework.

## How to run a design (interview or real)
Scope → estimate scale/QPS/data/latency → state ML problem & objective → draw the boxes (one clean diagram)
→ deep-dive the high-leverage components → name tradeoffs at each box → address evolution (cold start,
feedback loops, retraining, scale-up). **The signal is judgment under tradeoffs, not the fanciest model.**

Reference book: Chip Huyen, *Designing Machine Learning Systems*. Full detail and worked example:
`ml-system-design-guide.md` and `examples.md`.
