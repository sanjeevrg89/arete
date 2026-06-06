---
name: ml-observability-monitoring
description: Production ML/LLM observability and monitoring — the discipline of knowing when a deployed model
  is silently degrading, with no errors and no stack trace. Use when designing or debugging model monitoring:
  data/covariate drift, concept/label drift, prediction drift, training-serving skew, feature-attribution
  drift; drift detectors (PSI, KL/JS divergence, KS test, Wasserstein, embedding drift) and their pitfalls;
  data-quality validation (schema/range/null/cardinality/freshness, Great Expectations / TFDV style); model-
  performance monitoring with delayed/absent ground truth, proxy metrics, slice/segment & fairness analysis,
  calibration; LLM observability (OpenTelemetry GenAI tracing, spans for chains/agents/tools, Langfuse/Phoenix/
  LangSmith, token/cost/latency TTFT/ITL, hallucination signals, online LLM-as-judge eval, guardrail hit rates);
  alerting, retraining triggers, incident response for model regressions, and dashboards. Covers Evidently,
  Arize, WhyLabs, Fiddler, Vertex AI Model Monitoring, Prometheus/Grafana. Not CI/CD or feature pipelines.
---

# ML Observability & Monitoring

Apply the judgment of an engineer who owns the pager for production models and has watched accuracy rot for a
week before anyone noticed. **The defining fact of ML monitoring: a model can be 100% healthy by every software
SLO — no errors, p99 latency fine — while its predictions are quietly wrong.** Your job is to catch that
*before* the business does.

## How to use this skill

1. **Read `ml-observability-monitoring-guide.md`** in this directory — the full reference (why ML monitoring
   differs from software monitoring, the four monitoring layers, drift detection math and pitfalls, data-quality
   checks, performance monitoring under label lag, LLM observability, closing the loop, tooling, anti-patterns).
   Apply it to the system at hand.
2. For a concrete Evidently-style drift report config and an OpenTelemetry/Langfuse LLM tracing + online-eval
   instrumentation snippet to imitate, read **`examples.md`**.
3. Match the existing observability stack (Prometheus/Grafana, the chosen ML-monitoring vendor) and conventions;
   apply the correctness rules regardless. Treat managed-product features and SDK surfaces (Vertex AI Model
   Monitoring, Evidently, Langfuse, OTel GenAI semantic conventions) as **verify against current docs** — this
   space moves fast and it is 2026.

## Essentials (full detail in `ml-observability-monitoring-guide.md`)

- **Models fail silently.** There is no exception when a model degrades — the prediction just gets worse. Monitor
  the *statistics of inputs, outputs, and outcomes*, not just the service health. Software SLOs are necessary,
  not sufficient.
- **Monitor four layers, in this order of leading-ness:** (1) data quality (schema/null/range/freshness),
  (2) **training-serving skew** and drift (inputs change), (3) prediction drift (outputs change), (4) actual
  performance vs ground truth (the lagging truth). Layers 1–3 are *leading* signals you have today; layer 4 is
  *lagging* and may arrive days or weeks late — or never.
- **Ground-truth lag is the central problem.** Labels for a fraud/churn/conversion model can take days to months.
  Design for it: proxy metrics, delayed-label backfill/joins, human labeling on a sample, and drift as an
  early-warning stand-in for unmeasurable accuracy.
- **Training-serving skew is the highest-ROI thing to catch** and the most common production failure: the feature
  the model sees at serving time differs from training (different code path, stale feature, unit/encoding
  mismatch, time-travel leakage). Compare the *serving* feature distribution to the *training* distribution, and
  log served feature vectors. See `[[data-engineering-feature-stores]]`.
- **Pick the drift test for the data type and know its failure mode.** PSI/JS-divergence for binned numeric/
  categorical, KS/Wasserstein for continuous, chi-square for categorical, embedding-distance/domain-classifier
  for text/images. **All drift tests are sample-size sensitive** — at high volume everything is "significant";
  threshold on *effect size* (PSI, Wasserstein), not raw p-values, and fix a reference window.
- **Drift ≠ decay.** Input drift is a *hypothesis* that performance may drop, not proof. Concept drift (the
  X→y relationship changes) can crater accuracy with *zero* input drift. Always tie drift alerts back to a
  performance or business metric before you retrain.
- **Slice everything.** Aggregate accuracy hides per-segment collapse. Monitor key segments (geo, device, new vs
  returning, language, customer tier) and protected groups for **fairness**, and track **calibration** (predicted
  probability vs observed rate), not just a point metric.
- **LLM observability is tracing + eval, not a single accuracy number.** Emit OpenTelemetry GenAI spans for each
  chain/agent/tool/retrieval/LLM call (model, prompt, tokens, cost, latency, TTFT/ITL); log prompt+response; run
  **online LLM-as-judge** evals and reference-free quality/groundedness checks on sampled live traffic; track
  guardrail hit rates. See `[[ml-evaluation-evals]]` and `[[ai-security-on-gke]]`.
- **Close the loop or it's just dashboards.** Every monitor needs an owner, a threshold with hysteresis, a
  runbook, and a defined action: alert, auto-rollback, open incident, or trigger retraining (continuous training
  lives in `[[mlops-lifecycle]]`). Tune for signal — alert fatigue kills monitoring programs.
- **Anti-patterns that bite:** accuracy-only (no leading signals), no skew detection, no slicing, no ground-truth
  pipeline at all, drift alerts with no performance tie-in, and per-feature alerts that page on every wiggle.

## Related skills
- `[[mlops-lifecycle]]` — CI/CD, model registry, and continuous training that retraining triggers feed into.
- `[[ml-evaluation-evals]]` — offline/online eval methodology and LLM-as-judge that online evals reuse.
- `[[data-engineering-feature-stores]]` — feature pipelines, online/offline parity, the source of skew.
- `[[ai-security-on-gke]]` — guardrails whose hit rates you monitor as a safety signal.
- `[[aiml-on-kubernetes]]` / `[[autoscaling-kubernetes]]` / `[[gke-master]]` — infra-side metrics (GPU util,
  queue depth, autoscaling) that complement model-quality signals.
