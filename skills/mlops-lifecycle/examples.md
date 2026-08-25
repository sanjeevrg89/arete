# MLOps Lifecycle — Worked Examples

Canonical artifacts to imitate. Two examples:

1. A **Kubeflow / Vertex AI pipeline sketch** — `train → eval (gate) → register → deploy`, with a
   **continuous-training (CT) trigger**.
2. A **model-registry promotion flow** — register → validate → stage → approve → deploy → rollback.

> These are illustrative of the canonical idiom. SDK names/signatures (Kubeflow Pipelines DSL `kfp`,
> `google-cloud-aiplatform`) move fast — **verify against current docs** before running. The *shape* (DAG,
> gate, register, deploy, CT trigger) is the durable part. Don't fabricate fields you're unsure of.

---

## 1. Kubeflow Pipelines / Vertex AI pipeline: train → eval → register → deploy

A pipeline is a DAG of containerized components. Each `@component` is a step; artifacts (datasets, models,
metrics) flow between steps and are recorded in ML metadata for lineage. The **eval step gates** the
register step — a model below threshold or below the current champion never gets registered.

```python
# pipeline.py  — Kubeflow Pipelines v2 DSL (kfp). Compile and run on Kubeflow Pipelines
# or Vertex AI Pipelines. Verify component/SDK details against current kfp + Vertex docs.
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn"])
def validate_data(raw: Input[Dataset], valid: Output[Dataset]):
    """Schema + distribution checks (TFDV/Great Expectations/Pandera in real code).
    FAIL THE STEP (raise) on schema drift or anomalies so the pipeline stops here."""
    import pandas as pd
    df = pd.read_parquet(raw.path)
    assert set(["feature_a", "feature_b", "label"]).issubset(df.columns), "schema drift"
    assert df["label"].notna().all(), "null labels"
    df.to_parquet(valid.path)

@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn", "joblib"])
def train(data: Input[Dataset], model: Output[Model], seed: int = 42):
    import pandas as pd, joblib
    from sklearn.ensemble import GradientBoostingClassifier
    df = pd.read_parquet(data.path)
    X, y = df[["feature_a", "feature_b"]], df["label"]
    clf = GradientBoostingClassifier(random_state=seed)  # pin the seed for reproducibility
    clf.fit(X, y)
    model.metadata["framework"] = "sklearn"
    joblib.dump(clf, model.path)

@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn", "joblib"])
def evaluate(model: Input[Model], data: Input[Dataset], metrics: Output[Metrics]) -> float:
    """Compute absolute + sliced metrics. Return the headline metric so the DAG can gate on it."""
    import pandas as pd, joblib
    from sklearn.metrics import roc_auc_score
    clf = joblib.load(model.path)
    df = pd.read_parquet(data.path)
    auc = float(roc_auc_score(df["label"], clf.predict_proba(df[["feature_a", "feature_b"]])[:, 1]))
    metrics.log_metric("auc", auc)
    # In real code: also log per-slice AUC (fairness/edge regressions) and operational metrics.
    return auc

@dsl.component(base_image="python:3.11", packages_to_install=["google-cloud-aiplatform"])
def register_and_deploy(model: Input[Model], champion_auc: float, candidate_auc: float):
    """Register the version + deploy. Only reached when the eval gate passed.
    Compare vs the current champion; register as a candidate, attach lineage + model card,
    and route via canary rather than 100% cutover (see promotion flow below)."""
    from google.cloud import aiplatform
    aiplatform.init(project="PROJECT", location="us-central1")
    registered = aiplatform.Model.upload(  # verify signature against current Vertex SDK
        display_name="churn-model",
        artifact_uri=model.uri,
        serving_container_image_uri="REGIONAL-DOCKER-REPO/serving:latest",
        labels={"candidate_auc": str(round(candidate_auc, 4)),
                "champion_auc": str(round(champion_auc, 4))},
    )
    # Deploy behind a canary split rather than 100%; promote after guardrails hold.
    # endpoint.deploy(model=registered, traffic_split={"0": 90, registered.name: 10}, ...)

@dsl.pipeline(name="churn-training-pipeline")
def churn_pipeline(raw_data_uri: str, eval_threshold: float = 0.80, champion_auc: float = 0.0):
    raw = dsl.importer(artifact_uri=raw_data_uri, artifact_class=Dataset).output
    validated = validate_data(raw=raw)
    trained = train(data=validated.outputs["valid"])
    ev = evaluate(model=trained.outputs["model"], data=validated.outputs["valid"])

    # --- THE GATE: register/deploy only if eval clears threshold AND beats the champion ---
    with dsl.If(ev.output >= eval_threshold, name="meets-threshold"):
        with dsl.If(ev.output > champion_auc, name="beats-champion"):
            register_and_deploy(
                model=trained.outputs["model"],
                champion_auc=champion_auc,
                candidate_auc=ev.output,
            )
    # else: pipeline ends without promoting; the incumbent champion keeps serving.
```

Key properties to copy:

- **Data validation is the first step and can fail the run** — bad data never reaches training.
- **The eval step returns a metric the DAG branches on.** Promotion is conditional on *threshold AND
  champion comparison* — never unconditional.
- **Each artifact (Dataset/Model/Metrics) is tracked in ML metadata**, giving automatic lineage:
  prod model → eval run → training run → validated data.
- **Seeds and pinned images** make the run reproducible.
- Register and deploy are separated from training conceptually; deploy uses a **canary split**, not a
  100% cutover.

### The continuous-training (CT) trigger

The pipeline above is the unit you *trigger*. Wire triggers explicitly (verify scheduler/SDK current):

```python
# Scheduled CT (e.g. nightly) — Vertex AI Pipelines scheduling. Verify against current SDK.
from google.cloud import aiplatform
job = aiplatform.PipelineJob(
    display_name="churn-training-pipeline",
    template_path="churn_pipeline.yaml",      # compiled from pipeline.py
    parameter_values={"raw_data_uri": "gs://bucket/latest/", "eval_threshold": 0.80,
                      "champion_auc": 0.83},  # pass the *current* champion so the gate is relative
)
job.create_schedule(cron="0 2 * * *", display_name="nightly-ct")  # schedule-driven CT
```

```text
# Drift/decay-triggered CT (the loop that closes with [[ml-observability-monitoring]]):
#   monitoring detects input/prediction drift or a live-metric drop
#     -> emits an event (Pub/Sub / webhook / alert)
#       -> event triggers this same pipeline (e.g. Cloud Function / Eventarc -> PipelineJob.run)
#         -> pipeline retrains, re-validates, and only promotes if it beats the champion gate.
# Detection is owned by [[ml-observability-monitoring]]; the *reaction* (retrain + gate) is MLOps.
```

CT trigger choices, restated: **scheduled** (simple, may lag/waste), **new-data** (retrain on fresh
partition), **drift-triggered** (monitoring fires it — closes the loop), **on-demand**. Combine
schedule + drift in practice. Whatever the trigger, **the validation gates above always run** — CT never
auto-promotes a model that hasn't beaten threshold and champion.

---

## 2. Model-registry promotion flow

The registry is the source of truth. A version moves through stages by **explicit, auditable transitions**,
each gated. Example with MLflow Model Registry (the same lifecycle applies to Vertex Model Registry /
JFrog-style registries — verify the specific API):

```python
# Register a candidate produced by a tracked run, then promote through stages with gates.
import mlflow
from mlflow import MlflowClient
client = MlflowClient()

# 1. REGISTER — a tracked run produced a model; register it as a new immutable version.
#    Lineage: the version back-points to the run (params/metrics/artifacts), and the run
#    records data snapshot + git SHA + container digest. Verify run/data are pinned.
run_id = "abc123"                                  # from experiment tracking
result = mlflow.register_model(f"runs:/{run_id}/model", name="churn-model")
version = result.version

# 2. VALIDATE — attach eval results + a model card; assert gates passed.
client.set_model_version_tag("churn-model", version, "auc", "0.86")
client.set_model_version_tag("churn-model", version, "min_slice_auc", "0.79")   # sliced check
client.set_model_version_tag("churn-model", version, "model_card_uri", "gs://.../card.md")

# 3. STAGE — move to Staging only when data+model validation passed. (Newer MLflow uses
#    alias-based promotion instead of stages; verify which your version supports.)
client.transition_model_version_stage("churn-model", version, stage="Staging")

# 4. APPROVE — automated guardrails + (where required) human sign-off recorded as an audit event.
#    Encodes governance / segregation of duties — see [[responsible-ai-governance]].
client.set_model_version_tag("churn-model", version, "approved_by", "ml-platform-oncall")

# 5. PROMOTE TO PRODUCTION — but capture the outgoing champion first so rollback is one step.
prod = client.get_latest_versions("churn-model", stages=["Production"])
previous = prod[0].version if prod else None
client.transition_model_version_stage(
    "churn-model", version, stage="Production",
    archive_existing_versions=True,    # incumbent -> Archived, recoverable
)

# 6. ROLLBACK (first-class, not an afterthought) — if canary guardrails regress, re-promote
#    the previous version. Because it's still in the registry, this is a single transition.
if previous:
    client.transition_model_version_stage("churn-model", previous, stage="Production")
```

Stage flow and the gate at each hop:

```text
   (training run)            data + model validation passed?         approval passed?
        |                              |                                   |
        v                              v                                   v
   [ Registered ] --register--> [ Staging ] --canary + sign-off--> [ Production ]
        ^                                                                  |
        |                          guardrail metric regresses             v
        +------------------------------ rollback -------------------- [ Archived ]
                          (re-promote previous Production version)
```

Properties to copy:

- **Immutable versions** — never overwrite; every promotion is a new version + an auditable transition.
- **Gate at every hop** — validation before Staging; approval before Production. CT output enters at
  step 1 and passes the *same* gates.
- **Lineage on the version** — run, data snapshot, code SHA, container, eval results, model card. From a
  prod model you reach "what produced this?" in one hop.
- **Rollback is trivial because the previous Production version is archived, not deleted** — re-promote it.
- **Pair with progressive deployment** (canary/blue-green from the guide): the registry says *what* is
  blessed; the deployment strategy controls *how fast* traffic shifts and how fast you can revert.
