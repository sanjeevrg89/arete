# AGENTS.md — MLOps Lifecycle

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`mlops-lifecycle-guide.md`** next to this file — read it before
> designing or reviewing ML delivery, and apply it. A concrete Kubeflow/Vertex pipeline sketch and a
> model-registry promotion flow are in **`examples.md`**. This file is the always-on summary.
>
> Scope: production MLOps lifecycle (notebook → prod → healthy). Model monitoring/drift is the sibling
> `[[ml-observability-monitoring]]`; data/feature pipelines + feature stores are
> `[[data-engineering-feature-stores]]`. Reference them; don't duplicate.
>
> It is 2026 and the tooling churns. Treat product names, SDK signatures, and managed-service feature
> claims as **verify against current docs**. Concepts are stable; tool surface area is not. Never fabricate
> APIs, fields, flags, or benchmark numbers.

## When designing or reviewing ML delivery, apply these by default:

- **Know the maturity level.** Level 0 = manual/notebook-to-prod (anti-pattern for anything that must stay
  good). Level 1 = automated **ML pipeline** (unlocks continuous training). Level 2 = **CI/CD of the
  pipeline itself**. State the current level and the next unlock.
- **Ship pipelines, not models.** The deliverable is a versioned, tested training *pipeline*; the model is
  its reproducible output, registered and promoted separately.
- **CT is the ML-unique property.** Beyond CI (test code+data+model) and CD (deliver pipelines), add
  **continuous training** on a deliberate trigger (schedule and/or drift/decay signal from
  `[[ml-observability-monitoring]]`). No CT ⇒ silent model decay.
- **Never auto-promote.** Every model (including CT output) clears the same gates before promotion: data
  validation (schema/distribution) + model validation (metric ≥ threshold, ≥ current champion, sliced
  checks) + operational checks (latency/size/load).
- **Model registry is the source of truth.** Immutable versions, stages/promotion, full lineage (run +
  data snapshot + code SHA + container), approval gates, model cards. Nothing reaches prod that isn't a
  registered, approved version. Keep "previous Production" one step from redeploy.
- **Experiment tracking ≠ registry.** Tracking (MLflow / W&B / Vertex Experiments) compares *runs*; the
  registry governs *blessed* versions. Wire run → registered-version lineage.
- **Deploy progressively, design rollback first.** Shadow → canary → progressive, or blue-green for instant
  cutover; champion/challenger + A/B for online comparison. Bind rollout to guardrail metrics that
  auto-abort a regressing canary.
- **Pick serving mode per use case** (online/batch/streaming) and back all modes with the *same* model
  version + feature logic to avoid cross-mode skew.
- **Reproducibility = pin all five:** code SHA, data snapshot/version, config, container image digest (not
  `:latest`), seeds. If the registry entry can't recreate the model, it's not reproducible.
- **Train/serve skew is the classic failure.** Share transform code between train and serve, or use a
  feature store (`[[data-engineering-feature-stores]]`); validate serving inputs vs the training schema.
  Account for the **third pipeline** (data/feature) in lineage and validation.
- **Governance is pipeline steps, not afterthoughts:** approval gate before prod, generated model card,
  immutable audit/lineage, policy checks that can block promotion (`[[responsible-ai-governance]]`).
- **Choose the orchestrator on lineage + reproducibility + authoring model**, not hype: Kubeflow Pipelines,
  Vertex AI Pipelines, Argo, Airflow, Flyte, Metaflow.
- **LLMOps deltas:** the "model" is prompt + base model + params + retriever/tools — version them as a
  bundle. **CT becomes continuous evaluation + prompt/index/model refresh.** **Eval gates in CI**
  (`[[ml-evaluation-evals]]`) gate promotion; re-embed when the embedding model changes; track cost +
  latency as deploy metrics.

## Anti-patterns to flag on sight
Notebook-to-prod by hand · no CT · no model registry · no lineage/metadata · drift unmonitored ·
train/serve skew · untested pipelines · CT that auto-deploys without a gate · manual promotion with no
approval trail.

## Reviewing an ML delivery design/diff
Run the **maturity checklist** at the end of `mlops-lifecycle-guide.md` as the scorecard; the no's are the
backlog.
