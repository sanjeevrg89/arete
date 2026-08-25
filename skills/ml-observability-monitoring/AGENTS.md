# AGENTS.md — ML / LLM Observability & Monitoring

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ml-observability-monitoring-guide.md`** next to this file —
> read it before designing or debugging model monitoring, and apply it. A concrete Evidently-style drift
> report and an OTel/Langfuse LLM tracing + online-eval snippet are in **`examples.md`**. This is the
> always-on summary.
>
> Scope: knowing when a **deployed** model is silently degrading. Not CI/CD/registry (`mlops-lifecycle`),
> not feature pipelines (`data-engineering-feature-stores`), not infra metrics (`gke-master`,
> `autoscaling-kubernetes`).

## Apply by default when monitoring production ML/LLM systems

- **Models fail silently** — no error when a prediction goes wrong. Software SLOs (uptime, latency, errors)
  are necessary but **not sufficient**. Monitor the *statistics of inputs, outputs, and outcomes*.
- **Monitor four layers, leading → lagging:** (1) data quality, (2) drift + **training-serving skew**,
  (3) prediction drift, (4) performance vs ground truth. Lean on 1–3 (available now); layer 4 is delayed.
- **Ground-truth lag is the core constraint.** Labels arrive days–months late or never. Build a label-join/
  backfill pipeline on day one, use proxy metrics + sampled human labels, and use drift as an early warning.
- **Training-serving skew is the #1 silent failure.** Log the exact feature vector served; compare to the
  training distribution. Fix by computing features once and sharing via a feature store.
- **Pick the drift test for the data type, threshold on EFFECT SIZE not p-values:** PSI/JS for binned,
  KS/Wasserstein for continuous, chi-square for categorical, embedding-distance/domain-classifier for
  text/images. All drift tests are sample-size sensitive — at scale p-values always fire. Fix a reference window.
- **Drift ≠ decay.** Input drift is a hypothesis, not proof. Concept drift cuts accuracy with zero input drift.
  Always confirm a drift alert against a performance/business metric before retraining.
- **Data quality first.** Schema, range, null-rate, cardinality, freshness/staleness, missing-feature policy.
  Most "model broke" pages are data broke. Gate the batch (Great Expectations / TFDV / Pandera), don't just log.
- **Slice everything + calibration.** Aggregate accuracy hides per-segment collapse. Monitor key segments,
  protected groups (fairness), and calibration (predicted prob vs observed rate) — not just one point metric.
- **LLM = tracing + online eval + cost/latency + safety.** Emit OTel GenAI spans per chain/agent/tool/LLM call
  (model, prompt, tokens, cost, latency, TTFT/ITL); log prompt+response; run online LLM-as-judge / groundedness
  on sampled traffic; track guardrail hit rates. Pin & calibrate the judge — it drifts too.
- **Close the loop:** every monitor gets an owner, an effect-size threshold with hysteresis, a runbook, and an
  action (alert / auto-rollback / incident / retrain trigger). Keep last-good model deployable. Tune for signal —
  alert fatigue kills monitoring.
- **Avoid:** accuracy-only, no skew detection, no slicing, no ground-truth pipeline, drift alerts with no
  performance tie-in, per-feature alert spam, p-value thresholds at scale, LLM "vibes" monitoring.

## Verify against current docs (fast-moving)
OTel **GenAI semantic conventions**, **Vertex AI Model Monitoring** modes/limits, and **Evidently / Langfuse /
Phoenix / Arize / WhyLabs / Fiddler / Great Expectations / TFDV** APIs change often. The concepts are durable;
the API surfaces are not — don't hardcode from memory. It is 2026.

## Related
`[[mlops-lifecycle]]` · `[[ml-evaluation-evals]]` · `[[data-engineering-feature-stores]]` ·
`[[ai-security-on-gke]]` · `[[aiml-on-kubernetes]]` · `[[autoscaling-kubernetes]]` · `[[gke-master]]`
