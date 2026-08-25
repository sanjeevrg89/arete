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
