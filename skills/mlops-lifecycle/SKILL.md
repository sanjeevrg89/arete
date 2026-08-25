---
name: mlops-lifecycle
description: Production MLOps lifecycle — the discipline that reliably gets ML models from notebook to
  production and keeps them healthy. Use when designing or reviewing ML delivery: building CI/CD/CT
  pipelines, choosing a pipeline orchestrator (Kubeflow Pipelines, Vertex AI Pipelines, Argo, Airflow,
  Flyte, Metaflow), standing up a model registry, experiment tracking (MLflow, Weights & Biases, Vertex
  Experiments), ML metadata/lineage, data & model validation gates, deployment patterns (shadow, canary,
  blue-green, A/B, champion/challenger), online vs batch vs streaming inference, reproducibility,
  train/serve skew, continuous training triggers, LLMOps deltas (prompt versioning, eval gates, RAG/agent
  deploy), or assessing MLOps maturity. Covers the MLOps maturity model (level 0→1→2), the core technical
  capabilities, governance hooks, and anti-patterns.
---

# MLOps Lifecycle

Apply the judgment of an engineer who has run ML systems in production for years — who has been paged at
3am for a silently-stale model, debugged train/serve skew, and rolled back a bad model behind a canary.
MLOps is DevOps for ML plus the parts software engineering never had to solve: **the model is a function
of data, so the pipeline must be a first-class artifact and retraining (CT) is a first-class operation.**

## How to use this skill

1. **Read `mlops-lifecycle-guide.md`** in this directory — the full reference (maturity model, CI/CD/CT,
   the core capabilities, deployment patterns, reproducibility/skew, LLMOps, anti-patterns, maturity
   checklist). Apply it to the task.
2. For a concrete **Kubeflow/Vertex pipeline sketch** (train→eval→register→deploy with a CT trigger) and a
   **model-registry promotion flow**, read **`examples.md`** and imitate the structure.
3. Match the surrounding stack's conventions (orchestrator, registry, cloud). Apply the
   correctness/governance rules — versioning, validation gates, lineage, rollback — regardless of stack.
4. The ecosystem moves fast (it is 2026). Treat product/API names, SDK signatures, and managed-service
   feature claims as **verify against current docs** before relying on them.

## The essentials (full detail in `mlops-lifecycle-guide.md`)

- **Maturity model.** Level 0 = manual, notebook-to-prod, script-driven. Level 1 = automated **ML
  pipeline** (the pipeline, not a one-off model, is the deliverable; enables continuous training).
  Level 2 = automated **CI/CD** for the pipeline itself. Know which level you're at and the next unlock.
- **CT is the property unique to ML.** Beyond CI (test code) and CD (ship pipelines, not just a service),
  **continuous training** retrains on fresh data via a trigger: schedule, new data, or a drift/decay
  signal from monitoring. No CT ⇒ models rot. Drift detection lives in `[[ml-observability-monitoring]]`.
- **Ship pipelines, not models.** In a mature setup CD delivers a *training pipeline* that produces models;
  the model is an output, reproducible from versioned code + data + config + environment.
- **Validation is a gate, not a notebook cell.** Validate data (schema/distribution/anomalies) *and* the
  model (eval metric ≥ threshold, ≥ current champion, slice/fairness checks) before any promotion.
- **Model registry is the source of truth.** Versioning, staging/promotion, lineage to the producing run +
  data, approval gates, model cards. Nothing reaches prod that isn't a registered, approved version.
- **Experiment tracking ≠ registry.** Tracking (MLflow / W&B / Vertex Experiments) logs params/metrics/
  artifacts of *runs*; the registry governs the *blessed* versions you promote. Wire them together.
- **Deploy progressively, design rollback first.** Shadow → canary → progressive → full, or blue-green for
  instant cutover; champion/challenger and A/B for online comparison. Always keep the previous version
  one command away.
- **Train/serve skew is the classic prod failure.** Same feature transforms in train and serve — share the
  code or use a feature store (`[[data-engineering-feature-stores]]`); validate serving inputs against the
  training schema; watch for the "third pipeline" (data/feature) drifting from the training assumptions.
- **Reproducibility is non-negotiable.** Pin code (git SHA), data (snapshot/version), config, container
  image, and seeds. ML metadata ties every model to exactly what produced it.
- **Governance is wired into the pipeline.** Approval gates, model cards, audit trail, and lineage are
  pipeline steps, not afterthoughts — see `[[responsible-ai-governance]]`.
- **LLMOps deltas:** version prompts/templates and model+params as artifacts; **eval gates in CI**
  (`[[ml-evaluation-evals]]`) replace "retrain"; deploy RAG/agents (index + retriever + prompt + model) as
  a versioned bundle. "CT" often becomes "continuous evaluation + prompt/index refresh."
- **Anti-patterns kill you slowly:** notebook-to-prod by hand, no CT, no registry, no lineage, drift
  unmonitored, train/serve skew, untested pipelines. The maturity checklist at the end of the guide is the
  fast scorecard.

## Related skills

- `[[ml-observability-monitoring]]` — model monitoring, drift/decay detection, the signals that *trigger*
  CT and rollback. Reach for it for the "is the model healthy in prod" half.
- `[[data-engineering-feature-stores]]` — data/feature pipelines, feature stores, the train/serve skew fix
  at the data layer. The "third pipeline."
- `[[ml-evaluation-evals]]` — eval harnesses and eval-gate design, especially for LLMs/agents in CI.
- `[[responsible-ai-governance]]` — model cards, approval/sign-off, audit, fairness; the governance hooks
  MLOps pipelines must enforce.
- `[[aiml-on-kubernetes]]` / `[[gke-master]]` / `[[ray-on-kubernetes]]` — the compute substrate that runs
  the training/serving steps of these pipelines.
