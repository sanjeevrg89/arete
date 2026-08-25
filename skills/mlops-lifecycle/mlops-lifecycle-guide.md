# MLOps Lifecycle — Full Reference

The discipline of getting ML models reliably from notebook to production **and keeping them healthy**.
This guide is the operating manual: the maturity model, CI/CD/CT, the core technical capabilities, the
model registry, deployment patterns, reproducibility and train/serve skew, the "third pipeline," LLMOps
deltas, anti-patterns, and a maturity checklist.

Scope boundary: **model monitoring, drift, and decay detection** are the sibling skill
`[[ml-observability-monitoring]]` — this guide references them as CT/rollback *triggers* but does not
re-derive them. **Data/feature pipelines and feature stores** are `[[data-engineering-feature-stores]]`.
Cross-link, don't duplicate.

> Version awareness: it is 2026 and the MLOps tooling market churns constantly. Product names, managed-
> service feature matrices, and SDK signatures below are illustrative of the canonical idiom — **verify
> against current docs** before depending on a specific API field, flag, or feature claim. Concepts
> (maturity levels, CT, registry semantics, deployment patterns) are stable; tool surface area is not.

## Mental model: why ML needs more than DevOps

A traditional service is `code → build → deploy`. An ML system is `code + data + config → train → model →
deploy`, and the model's quality is a function of *data it has never seen at build time*. Three
consequences drive everything in MLOps:

1. **The artifact is derived, not authored.** You don't write the model; a pipeline produces it from data.
   So the *pipeline* must be the versioned, tested, deployable thing — not just the model binary.
2. **The world moves, so the model decays.** Data distributions shift; yesterday's good model is today's
   bad one. You need a *retraining* operation — **Continuous Training (CT)** — that DevOps never had.
3. **Two codebases compute features: training and serving.** If they diverge, you get **train/serve skew**
   — the single most common silent production failure in ML.

MLOps = the practices, automation, and platform capabilities that make 1–3 tractable: reproducible
pipelines, validation gates, a registry as source of truth, progressive deployment with rollback, lineage,
and monitoring-driven retraining. The canonical framing is Google's *Practitioner's Guide to MLOps* and the
GCP architecture doc *MLOps: Continuous delivery and automation pipelines in ML* (links at the end).

## The MLOps maturity model (level 0 → 1 → 2)

This is the most useful single lens. From the GCP architecture doc:

### Level 0 — Manual process
Every step is manual and script/notebook-driven. Data scientist trains in a notebook, hands a model
("over the wall") to engineers who wrap it in a service. Characteristics:
- Manual, script-driven, interactive. No CI/CD for the ML side.
- Infrequent releases (a few times a year). Training and serving code are *disconnected*.
- No active performance monitoring; no automated retraining. **Models silently rot.**

This is fine for a one-off analysis. It is the dominant **anti-pattern** for anything that must stay good.

### Level 1 — ML pipeline automation
The unit of delivery becomes an **automated, orchestrated training pipeline**, not a hand-built model.
This is the level that unlocks **continuous training**. Characteristics:
- The pipeline runs end-to-end (ingest → validate data → transform → train → validate model → register)
  on a trigger (schedule, new data, drift signal) and produces a fresh, registered model automatically.
- **Continuous Training (CT)** in production: the pipeline retrains on new data without a human kicking it.
- **Symmetry of environments:** the same pipeline runs in dev/experiment and in prod (parameterized).
- **Data & model validation are explicit pipeline steps** (gates), not ad-hoc checks.
- Requires: pipeline orchestrator, ML metadata store, feature store (optional but common), model registry.

What level 1 still lacks: the *pipeline source code itself* is deployed manually. New pipeline logic (new
features, new architecture) is a manual rollout.

### Level 2 — CI/CD pipeline automation
You add CI/CD **for the pipeline**. A change to feature engineering, model code, or pipeline topology flows
through automated build/test and is delivered as a new pipeline to staging then prod. Characteristics:
- **CI:** commit triggers build + unit/integration tests of components, pipeline assembly, and packaging.
- **CD:** the built pipeline artifact is automatically delivered to the target environment.
- **CT** runs inside the deployed pipeline on its triggers.
- Source control, automated testing, and a pipeline registry/artifact store are mandatory.

Net: at level 2 you can ship a *new way of producing models* as routinely as a web team ships a feature.
Most serious ML orgs target level 1 broadly and level 2 for high-value, frequently-iterated models.

| Capability                    | Level 0 | Level 1 | Level 2 |
|-------------------------------|:-------:|:-------:|:-------:|
| Automated training pipeline   | no      | yes     | yes     |
| Continuous training (CT)      | no      | yes     | yes     |
| Data & model validation gates | no      | yes     | yes     |
| CI/CD of the pipeline itself  | no      | no      | yes     |
| Deliverable                   | model   | pipeline| pipeline + automated rollout |

## CI / CD / CT

DevOps has CI and CD. ML adds **CT**. All three are distinct properties; don't conflate them.

### Continuous Integration (CI) — test data + code + model
ML CI tests more than code. On every commit:
- **Code:** unit tests for transforms, feature logic, pipeline components.
- **Data:** schema and statistics tests on sample/fixture data; assert the feature engineering produces
  expected shapes/ranges; catch nulls/cardinality blowups.
- **Model:** smoke-train on a small slice and assert it converges; assert the model meets a *minimum*
  quality bar and matches expected input/output signature; test that it loads in the serving runtime.
- Pipeline assembly: the components wire together and the DAG compiles.

### Continuous Delivery (CD) — deliver pipelines, not just a service
In mature ML, CD's deliverable is an **automated training pipeline** plus a **prediction service**. CD must
verify: model compatibility with the target serving infra (hardware, runtime, library versions), prediction
service latency/throughput SLOs under load, and that the pipeline reproduces in the target environment.
"Deploy" usually means deploy *the pipeline*; the model it later produces is registered and deployed
separately (often via CT + a promotion gate).

### Continuous Training (CT) — the property unique to ML
CT is the automated retraining of the model in production. It is what keeps the model from decaying.

**CT triggers** (pick deliberately; usually combine):
- **Scheduled** — retrain nightly/weekly. Simple, predictable; can waste compute or lag fast shifts.
- **New-data** — retrain when N new labeled rows / a new data partition lands.
- **Drift/decay-triggered** — a monitoring signal (input drift, prediction drift, or a drop in a live
  quality metric) fires the pipeline. This closes the loop with `[[ml-observability-monitoring]]`, which
  owns the detection; MLOps owns the *reaction*.
- **On-demand** — manual or event-driven (e.g., a major upstream schema change).

**CT must not auto-promote blindly.** A retrained model goes through the *same* validation gates
(data validation, model evaluation vs threshold and vs current champion, slice checks) before it can be
registered as a promotion candidate. CT that auto-deploys without a quality gate is an outage generator.

### Pipeline orchestration
The orchestrator runs the DAG (containerized steps), passes artifacts/metadata between steps, handles
retries/caching, and exposes the trigger surface. Canonical options (verify current capabilities):

| Orchestrator           | Sweet spot / notes                                                          |
|------------------------|-----------------------------------------------------------------------------|
| **Kubeflow Pipelines** | K8s-native, container-step DAGs, built-in ML metadata + artifact lineage. The open core many platforms build on. |
| **Vertex AI Pipelines**| Managed, runs KFP/TFX pipelines serverlessly; integrates Vertex Model Registry, Experiments, Metadata. |
| **Argo Workflows**     | General K8s workflow engine; KFP runs *on* Argo. Reach for it when you want raw container DAGs. |
| **Apache Airflow**     | Mature, ubiquitous DAG scheduler; great for data orchestration, weaker on ML-native artifact/lineage semantics. |
| **Flyte**              | Strongly-typed, K8s-native, data-aware, reproducibility/caching first-class; good for typed ML pipelines. |
| **Metaflow**           | Python-first DX, easy local→cloud scaling, human-friendly; popular for data-science-led teams. |

Choose on: K8s-nativeness, artifact/metadata lineage support, typing/caching, managed vs self-hosted, and
how your team writes pipelines (YAML DSL vs Python). Don't pick by hype; pick by lineage + reproducibility +
your team's authoring model.

## The core MLOps technical capabilities

A platform (DIY or managed) provides these. Map your stack to this list; gaps are your roadmap.

- **Source/version control** — code *and* config *and* pipeline definitions in git. Data and models are
  versioned too (snapshots/DVC/registry), referenced by version, not copied ad hoc.
- **Test & build (CI)** — automated tests (code/data/model) and reproducible container builds for each step.
- **Deployment services (CD)** — automated, repeatable deployment of pipelines and prediction services to
  online (real-time), batch, and streaming targets.
- **Model registry** — the governed store of model *versions*: staging/promotion, lineage, approval, model
  cards. Source of truth for "what is/should be in prod." (Deep dive below.)
- **Feature store** — consistent feature definitions and values across training and serving; the primary
  train/serve-skew fix at the data layer. Owned by `[[data-engineering-feature-stores]]` — link, don't
  duplicate. Examples: Feast, Vertex Feature Store, Tecton, Databricks Feature Store.
- **ML metadata & artifact tracking** — record every run's inputs, params, artifacts, and *lineage* (which
  data + code + container produced which model). Backbone of reproducibility and audit. (e.g. ML Metadata /
  MLMD, Vertex ML Metadata.)
- **Pipeline orchestration** — see table above.
- **Experiment tracking** — log params, metrics, and artifacts of training *runs* for comparison and
  selection (MLflow Tracking, Weights & Biases, Vertex AI Experiments).
- **Data & model validation** — automated gates (below).

### Experiment tracking vs the model registry (don't confuse them)
- **Experiment tracking** answers *"which run did best, and how?"* — it logs many runs (params, metrics,
  curves, artifacts) so you can compare and reproduce experiments. MLflow Tracking, W&B, Vertex Experiments.
- **Model registry** answers *"which version is blessed for staging/prod, and who approved it?"* — it
  governs the lifecycle of *selected* models.
- Wire them: a tracked run produces a candidate model → you register that model version → it carries a back-
  pointer (lineage) to the run, data snapshot, and code SHA. MLflow ships both Tracking and a Model Registry;
  even so, keep the *conceptual* distinction — tracking is for exploration breadth, registry is for governed
  promotion.

### Data & model validation (the gates)
- **Data validation:** schema conformance (expected columns/types), distribution checks vs a baseline
  (detect skew/anomalies/drift in incoming training data), required-feature presence, label sanity. Tools:
  TensorFlow Data Validation (TFDV), Great Expectations, Pandera, Deequ.
- **Model validation:** absolute quality (metric ≥ threshold), **relative** quality (≥ current champion /
  no regression), **sliced** evaluation (per-segment metrics, not just aggregate — catches fairness/edge
  regressions), and operational checks (inference latency, model size, serving-runtime load test). Only a
  model that clears *all* gates becomes a promotion candidate. Eval design for LLMs/agents lives in
  `[[ml-evaluation-evals]]`.

## Model registry (deep dive)

The registry is the contract between "training produced a model" and "production serves a model." Treat it
as the system of record. Core responsibilities:

- **Versioning** — every model is an immutable version with a unique ID; you never overwrite, you add.
- **Stages / promotion** — versions move through stages (e.g. `None → Staging → Production → Archived`, or
  custom: `dev → shadow → canary → prod`). Promotion is an explicit, auditable transition.
- **Lineage** — each version links to: the producing pipeline run, the data snapshot/version, the code SHA,
  the container image, hyperparameters, and the eval results. From a prod model you can answer "what made
  this?" in one hop. This is the backbone of reproducibility and incident forensics.
- **Approval gates** — promotion to production requires sign-off (automated gate passing + optional human
  approval). Encodes governance/segregation-of-duties. See `[[responsible-ai-governance]]`.
- **Model cards** — structured documentation per version: intended use, training data summary, eval
  metrics (including sliced/fairness), limitations, owner. Generate these in the pipeline and attach to the
  registered version. Governance detail in `[[responsible-ai-governance]]`.
- **Metadata/tags** — framework, metrics, signature (input/output schema), serving requirements.

Examples: MLflow Model Registry, Vertex AI Model Registry, plus registries fronted by artifact platforms
(e.g. JFrog ML / Artifactory-style model registries). The promotion flow — register → validate → stage →
approve → deploy → (optionally) auto-rollback — is shown concretely in `examples.md`.

## Deployment patterns

You are deploying a *new model version* (or a new pipeline). Design for safe rollout and instant rollback.

### Serving modes (pick per use case)
- **Online / real-time** — synchronous low-latency request/response (REST/gRPC). Strict latency SLOs,
  autoscaling, often the hardest to operate. Skew risk is highest here.
- **Batch** — score a large dataset on a schedule (e.g. nightly recommendations table). Throughput over
  latency; easiest to reason about.
- **Streaming** — score events as they flow (Kafka/Pub-Sub/Flink). Continuous, stateful-ish, ordering and
  late-data concerns.

A model often serves both online and batch; ensure the *same* model version and feature logic back both, or
you reintroduce skew across serving modes.

### Rollout strategies
| Pattern               | What it does                                                              | Use when |
|-----------------------|--------------------------------------------------------------------------|----------|
| **Shadow (dark launch)** | New model receives mirrored live traffic; predictions logged, not served. | De-risk before any user impact; validate latency + behavior on real traffic. |
| **Canary**            | Route a small % of live traffic to the new version; watch metrics; ramp.   | Standard progressive rollout. |
| **Blue-green**        | Stand up new version in parallel; switch 100% at once; old stays warm.      | Need instant cutover + instant rollback; can afford double capacity. |
| **A/B test**          | Split traffic across versions to compare a *business/quality* metric statistically. | Deciding *which model is better* for users, not just "is it safe." |
| **Champion/challenger** | Incumbent (champion) serves; one or more challengers run (often in shadow or on a slice); promote a challenger that wins on the metric. | Continuous, ongoing model competition; pairs naturally with CT. |

**Rollback is a first-class design requirement, not a recovery afterthought.** Keep the previous registered
version deployable in one step; bind promotion to automated guardrail metrics so a canary that regresses
auto-aborts. Blue-green and canary give you the fastest rollback; ensure the registry's "previous
Production" pointer makes "redeploy the last good version" trivial.

## Reproducibility, train/serve skew, and the third pipeline

### Reproducibility
A model is reproducible only if you pinned **all five**: code (git SHA), data (immutable snapshot/version),
config/hyperparameters, environment (container image digest, not `:latest`), and randomness (seeds). ML
metadata records the binding so any prod model maps back to exactly what produced it. If you can't recreate
a model from its registry entry, you can't safely debug, audit, or retrain it.

### Train/serve skew — the classic prod failure
Skew = the features (or their distributions) at serving time differ from training time, so a model that
scored great offline performs badly live. Sources:
- **Code skew:** training computes a feature one way (pandas, batch), serving another (a hand-written
  service). Even subtle differences (default fill values, time-zone handling, tokenization) silently break.
- **Data skew:** serving inputs drift from training distribution (this shades into drift — see
  `[[ml-observability-monitoring]]`).
- **Time-travel/leakage skew:** training used information not available at serving time.

Fixes: **share the transform code** between train and serve (one library, or export the preprocessing as
part of the model graph), or use a **feature store** so train and serve read the *same* feature
definitions/values (`[[data-engineering-feature-stores]]`). Validate serving inputs against the *training*
schema at runtime, and monitor serving feature distributions vs the training baseline.

### The "third pipeline"
Beyond the obvious two pipelines (training and serving/inference) there is a **third: the data/feature
pipeline** that produces features for *both*. Most train/serve skew and many silent failures originate
here — a feature definition changes in the data pipeline and the training assumptions go stale. Treat the
feature pipeline as a versioned, tested, monitored artifact in its own right. It is owned by
`[[data-engineering-feature-stores]]`; MLOps must account for it in lineage and validation.

## Governance hooks

Governance is not a separate process bolted on at the end — it is **pipeline steps and registry gates**:
- Approval gate before `Production` promotion (automated checks + human sign-off where required).
- **Model card** generated by the pipeline and attached to the registered version.
- Immutable **audit trail / lineage**: who trained/approved/deployed what, from which data and code.
- Policy checks (license, data-use, fairness thresholds) as gates that can *block* promotion.
Depth, model-card schemas, fairness, and sign-off workflows are in `[[responsible-ai-governance]]`.

## LLMOps deltas

For LLM/RAG/agent systems the lifecycle reshapes — same spine, different artifacts and gates:

- **The "model" is often a prompt + base model + params + tools/retriever**, not a trained weight file.
  Version the **prompt/template** (versioned, reviewed, registered like code), the model id + decoding
  params, the tool/function schemas, and the retrieval index together as a deployable bundle.
- **CT → continuous *evaluation* + refresh.** You rarely retrain a foundation model; instead you
  continuously *evaluate* the system and refresh prompts, few-shot examples, retrieval index, or model
  version. The "is it still good?" loop is eval-driven.
- **Eval gates in CI are the core gate.** A prompt/model/index change must pass an offline eval suite
  (quality, refusal/safety, regression vs current, sliced by use case) before promotion. Use LLM-as-judge,
  rubric scoring, golden datasets — design lives in `[[ml-evaluation-evals]]`. Treat the eval suite like a
  test suite: it gates merges.
- **RAG/agent deployment** = version the whole stack: index/embeddings (and the embedding model — re-embed
  on change), retriever config, prompt, model, and tool definitions. Roll out behind canary/shadow and
  watch online quality + cost + latency, same as any model.
- **Prompt/version management:** prompts live in source control and/or a prompt registry with versions,
  environments, and rollback — never hardcoded-and-edited-in-prod. Track which prompt version served which
  response (lineage) for debugging.
- Cost and latency become first-class deploy metrics (tokens, p95 latency, cache hit rate) alongside
  quality.

## Anti-patterns (and the failure they cause)

- **Notebook-to-prod by hand.** No pipeline, no tests, irreproducible. Works once, rots immediately, can't
  be safely changed. → Automate to at least maturity level 1.
- **No continuous training.** Model trained once, never refreshed; quietly decays as the world shifts. →
  Add a CT trigger (schedule + drift-driven) with validation gates.
- **No model registry.** Models live in buckets with names like `model_final_v3_REALLY.pkl`. No lineage, no
  promotion control, no idea what's in prod. → Adopt a registry as the source of truth.
- **No lineage / metadata.** Can't answer "what data and code produced this prod model?" → ungovernable and
  undebuggable. → Record ML metadata for every run; link registry versions back to runs.
- **Drift unmonitored.** No serving-time monitoring, so decay/skew is invisible until a business metric
  craters. → `[[ml-observability-monitoring]]`; feed its signals into CT and rollback.
- **Train/serve skew.** Separate, divergent feature code in train and serve. → Share transforms / feature
  store; validate serving inputs vs training schema.
- **Untested pipelines.** The pipeline that builds your model has no tests; a refactor silently corrupts
  features. → CI tests for components, data, and model.
- **CT that auto-deploys without a gate.** Retrains and ships straight to prod. → Always gate on
  data + model validation and champion comparison.
- **Manual promotion with no approval trail.** Someone clicks "deploy." → Approval gates + audit in the
  registry.

## Maturity checklist (fast scorecard)

Score yes/no; the no's are your backlog.

- [ ] Training is an **automated pipeline**, not a notebook (≥ level 1).
- [ ] Pipeline + config + prompts are in **source control**; runs are reproducible from a registry entry.
- [ ] **CI** tests code, data, and model on every change.
- [ ] **Continuous training** runs on a deliberate trigger (schedule and/or drift), gated by validation.
- [ ] **Data validation** and **model validation** (absolute + vs-champion + sliced) gate every promotion.
- [ ] A **model registry** is the source of truth: versioning, stages, lineage, approval, model cards.
- [ ] **Experiment tracking** is wired to the registry (run → registered version lineage).
- [ ] **ML metadata/lineage** ties every prod model to its data + code + container.
- [ ] Deployment is **progressive** (shadow/canary/blue-green) with **one-step rollback**.
- [ ] **Train/serve skew** is prevented (shared transforms / feature store) and monitored.
- [ ] **Monitoring** (`[[ml-observability-monitoring]]`) feeds drift/decay signals back into CT/rollback.
- [ ] **Governance hooks** (approval, model card, audit) are pipeline/registry steps, not manual.
- [ ] (LLM) **Eval gates** run in CI; prompts/index/model versioned and deployed as a bundle.
- [ ] (Level 2) **CI/CD of the pipeline itself** — new pipeline logic ships through automated build/test.

## Rationalizations & rebuttals

The excuses for skipping MLOps discipline, each rebutted:

- *"Notebook-to-prod by hand is fine, it's just one model."* — One model is how every unmanaged fleet
  starts. The moment you must retrain, debug, or change it you're irreproducible. Automate to ≥ level 1
  before it ships, not after it breaks.
- *"No registry, just copy the file to the bucket."* — A bucket has no versioning, lineage, promotion
  control, or approval trail. `model_final_v3_REALLY.pkl` is not a source of truth; you won't know what's
  in prod or how to roll back. Register the version.
- *"Skip CT, we'll retrain manually when accuracy drops."* — You only notice the drop after a business
  metric craters, and manual retrains slip. The world shifts continuously; add a deliberate trigger
  (schedule and/or drift) gated by validation.
- *"No lineage needed, the data scientist remembers how it was trained."* — Memory is not an audit trail
  and people leave. Without code SHA + data snapshot + container digest you cannot reproduce, debug, or
  defend the model. Record ML metadata for every run.
- *"Just deploy straight to 100%, the offline eval looked great."* — Offline eval doesn't catch
  train/serve skew, latency regressions, or sliced failures on live traffic. Roll out behind
  shadow/canary with guardrail metrics and one-step rollback.
- *"We share the feature code in spirit; train and serve are basically the same."* — "Basically the same"
  is exactly how skew hides — fill values, time zones, tokenization differ silently. Share the actual
  transform (one library or in-graph) or a feature store, and validate serving inputs vs the training
  schema.
- *"CT can auto-promote; the pipeline produced it, so it's good."* — A retrained model that bypasses the
  gates is an outage generator. Run the *same* data + model validation and champion comparison before any
  promotion candidate is registered.

## Red flags

Stop and reconsider if you see any of these:

- **No model registry** — models live in buckets/named files; nobody can state what's in prod or who
  approved it.
- **Manual deploys** — someone hand-wraps a notebook model or clicks "deploy" with no automated pipeline
  or approval trail.
- **Drift unmonitored** — no serving-time monitoring; decay and skew are invisible until a business
  metric drops (`[[ml-observability-monitoring]]`).
- **Not reproducible** — you cannot recreate a prod model from its registry entry (missing code SHA, data
  snapshot, container digest, or seeds).
- **No rollback path** — the previous good version isn't one-step deployable; a bad rollout means an
  incident, not a revert.
- **Train/serve skew risk** — separate, divergent feature code in training and serving; no input
  validation against the training schema.
- **CT without gates** — retraining auto-deploys with no data/model validation or champion comparison.
- **Untested pipelines** — the pipeline that builds your model has no CI; a refactor can silently corrupt
  features.
- **(LLM) Prompts edited in prod** — prompt/index/model not versioned as a bundle; no eval gate before
  promotion.

## Verification gate (definition of done)

The work is not done until all of these are true:

- [ ] **Pipeline is reproducible** — training runs as an automated pipeline (≥ level 1), and any prod model
  recreates from its registry entry (code SHA + data snapshot + container digest + config + seeds pinned).
- [ ] **Registry + versioning + lineage** — every model is an immutable registered version with stages,
  approval, and one-hop lineage back to its producing run, data, code, and image.
- [ ] **CI/CD/CT gates wired** — CI tests code/data/model; CD verifies serving compatibility + SLOs; CT
  runs on a deliberate trigger. Every promotion passes data validation and a **model eval gate** (absolute
  threshold + vs-champion + sliced; for LLMs, the offline eval suite gates the merge).
- [ ] **Progressive deploy + rollback** — rollout is shadow/canary/blue-green with guardrail metrics, and
  the previous Production version is one-step redeployable.
- [ ] **Monitoring wired** — serving-time monitoring is live and its drift/decay signals feed back into
  CT and rollback (`[[ml-observability-monitoring]]`).
- [ ] **Skew prevented** — train and serve share transform logic (or a feature store); serving inputs are
  validated against the training schema.
- [ ] **Governance hooks in place** — approval gate, model card, and audit/lineage are pipeline/registry
  steps, not manual afterthoughts.

## Canonical references (verify current before relying on specifics)

- Google, *Practitioner's Guide to MLOps* —
  https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf
- GCP, *MLOps: Continuous delivery and automation pipelines in machine learning* —
  https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Vertex AI — *Introduction to MLOps on Vertex AI* —
  https://cloud.google.com/vertex-ai/docs/start/introduction-mlops
- ml-ops.org — *MLOps Principles* — https://ml-ops.org/content/mlops-principles
- JFrog — *What is a Model Registry?* — https://jfrog.com/learn/mlops/model-registry/
- Kubeflow Pipelines docs — https://www.kubeflow.org/docs/components/pipelines/
- MLflow (Tracking + Model Registry) — https://mlflow.org/docs/latest/
- TensorFlow Data Validation (TFDV) — https://www.tensorflow.org/tfx/guide/tfdv
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015) — the foundational
  "ML systems are mostly not ML code" paper; motivates most of this guide.
