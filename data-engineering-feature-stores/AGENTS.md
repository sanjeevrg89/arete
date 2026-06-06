# AGENTS.md — Data Engineering & Feature Stores for ML

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`data-engineering-feature-stores-guide.md`** next to this
> file — read it before building or debugging ML data pipelines / feature stores, and apply it. Concrete
> artifacts to imitate (Feast feature definition, point-in-time training-set retrieval, TFDV / Great
> Expectations validation) are in **`examples.md`**. This file is the always-on summary.
>
> **The law: garbage in, garbage out.** Data quality, not model choice, decides production outcomes. The
> three failure classes to prevent are training-serving skew, temporal/label leakage, and silent
> data-quality regressions.

## When working on ML data pipelines, feature stores, or data validation, apply by default:

- **One feature definition, served to both training (offline) and serving (online).** Never reimplement a
  feature in the serving layer — duplicated/divergent feature code is the root cause of training-serving
  skew. Use a feature store (Feast/Tecton/Vertex AI Feature Store/Featureform) or a shared, tested
  transform library.
- **Point-in-time-correct (as-of) joins, always.** For each label/decision event, fetch each feature's
  value *as of that event's timestamp* (most recent row with `feature_ts <= label_ts`, honoring TTL).
  Never join the "latest" value into training — that leaks the future and inflates offline metrics. Be
  suspicious of any feature that looks implausibly predictive.
- **Validate every dataset against an explicit schema/contract before it trains or serves.** Schema
  (types, nullability, domains) + distribution checks (drift/skew/null spikes) + volume/freshness. Use
  Great Expectations or TFDV. **Fail the pipeline / quarantine the partition** — don't warn into a log.
- **Offline store ≠ online store.** Offline (warehouse/lakehouse: BigQuery/Snowflake/Iceberg/Delta/Parquet)
  for high-throughput training reads; online (Redis/DynamoDB/Bigtable) for millisecond point lookups.
  **Materialization** syncs offline → online and sets **feature freshness** — give it an SLO and alert on lag.
- **Registry is the source of truth** for feature definitions, owners, and lineage. No undocumented
  features. A feature without an owner/description/definition is a future incident.
- **Default to batch; reach for streaming only when staleness measurably moves the metric.** Avoid two
  drifting implementations (batch + streaming) of the same feature — share the logic (Beam/Flink) or let
  the feature store target both.
- **Version data and features, not just code.** Pin every training run to a concrete data version —
  lakehouse snapshot (Delta/Iceberg time travel), DVC rev, or LakeFS commit — and record it with the model.
  "Which data trained this model?" must have a precise answer. Pipelines must be idempotent and backfillable.
- **Embeddings are features:** embed with the *same model + preprocessing* in train and serve; pin the
  embedder version (re-embedding is a migration + backfill, not a config tweak).
- **Label quality caps accuracy.** Measure inter-annotator agreement; use weak supervision / active
  learning to scale; audit noisy labels. Fixing labels often beats any model change.
- **LLM corpora: dedup (exact + near-dup) and decontaminate against eval sets** before training — train/eval
  contamination silently inflates benchmarks. Document the dataset (data card: source, biases, PII, license).
- **Govern the data:** end-to-end lineage; classify/mask/tokenize PII, enforce access + retention + deletion;
  data contracts between producers and consumers.

## Definition of done for ML data work — prove it, don't claim it
- Same transformation path feeds offline (training) and online (serving); skew checked.
- Training sets built with point-in-time joins; no future leakage.
- Data validated against a schema/contract; the pipeline fails on violation. Report results.
- Training run pinned to a concrete, reproducible data version.
- Feature freshness within SLO; features documented with owners in the registry.

## Version awareness
This ecosystem moves fast (2026). **Verify against current docs** for feature-store APIs (esp. Vertex AI
Feature Store — architecture changed across generations — and Feast), lakehouse format support per engine,
orchestrator lineage features, and GE/TFDV API surfaces. Don't fabricate APIs, version numbers, or benchmarks.
