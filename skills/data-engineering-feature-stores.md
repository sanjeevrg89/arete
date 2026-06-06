---
name: data-engineering-feature-stores
description: Expert data engineering for ML — the pipelines, feature stores, and data-quality discipline
  that decide whether models work in production ("garbage in, garbage out"). Use when building or debugging
  ML data pipelines (ingestion, validation, transformation, batch vs streaming), orchestration (Airflow,
  Dagster, Flyte, Spark, Beam), the lakehouse (Delta Lake, Apache Iceberg, Hudi, Parquet), or data
  versioning (DVC, LakeFS, lakeFS). Use for feature stores (Feast, Tecton, Vertex AI Feature Store,
  Featureform) — offline vs online store, the registry, materialization, feature freshness,
  point-in-time-correct joins, and eliminating training-serving skew. Use for data quality & validation
  (Great Expectations, TFDV, schema/distribution/anomaly checks, data contracts), labeling & dataset
  curation (weak supervision, active learning, dedup/decontamination for LLM corpora), embedding/feature
  pipelines, and governance (lineage, PII). Triggers on symptoms like train/serve skew, label leakage from
  non-point-in-time joins, stale features, missing data validation, undocumented features, or duplicated
  one-off feature code across training and serving.
---

# Data Engineering & Feature Stores for ML

Apply the judgment of a data/ML engineer who has run feature platforms in production for years: most
"model" failures are data failures. The data pipeline — not the model architecture — usually sets the
ceiling on production performance. Make the same feature logic serve training and serving, validate
every dataset against an explicit contract, and never join a label to a feature computed after the label
existed.

## How to use this skill

1. **Read `data-engineering-feature-stores-guide.md`** in this directory — the full reference (pipeline
   architecture, lakehouse, the feature-store mental model, point-in-time joins, data validation,
   labeling/curation, governance, anti-patterns, troubleshooting). Apply it to the task at hand. For
   concrete artifacts to imitate (a Feast feature definition, point-in-time training-set retrieval, a
   TFDV/Great Expectations validation snippet), read **`examples.md`**.
2. Match the surrounding stack's conventions (existing orchestrator, warehouse/lakehouse, feature store,
   validation tool). Apply the correctness rules — one feature definition for train and serve,
   point-in-time joins, an explicit data contract — regardless of stack.
3. Before declaring data work "done," prove it: the pipeline emits validated data against a schema, the
   same transformation path feeds offline and online, and you can answer "what was this feature's value
   at prediction time?" Report skew/freshness/validation results, not impressions.

## The essentials (full rationale in `data-engineering-feature-stores-guide.md`)

- **Garbage in, garbage out is the law.** Data quality, not model choice, decides production outcomes.
  Spend the effort on pipelines, validation, and labels accordingly.
- **A feature store exists to kill training-serving skew.** Define a feature once; serve the *identical*
  computation to offline (training) and online (low-latency inference). Skew = the model sees different
  feature distributions in prod than in training, and silently degrades.
- **Point-in-time correctness is non-negotiable.** Build training sets with as-of joins: for each label
  event, fetch the feature value *as it was at that timestamp*. A naive join to "latest" feature value
  leaks the future and inflates offline metrics that collapse in production.
- **Offline store ≠ online store.** Offline (warehouse/lakehouse: BigQuery, Snowflake, Parquet/Iceberg)
  serves high-throughput historical reads for training; online (Redis, DynamoDB, Bigtable) serves
  millisecond single-entity reads for inference. Materialization syncs offline → online.
- **Validate every dataset against an explicit schema/contract** before it trains or serves a model.
  Schema (types, required, domains) + distribution checks (drift, skew, missing-value spikes) catch the
  silent failures. Great Expectations or TFDV; fail the pipeline, don't warn into a log nobody reads.
- **Batch vs streaming is a freshness/cost tradeoff,** not a religion. Most features are batch; reach for
  streaming only when staleness actually moves the metric. Keep one transformation definition across both.
- **Lakehouse over raw files for ML data at scale:** Delta/Iceberg/Hudi give ACID, schema evolution, and
  time travel on Parquet — reproducible training snapshots without copying data.
- **Version data and features, not just code.** DVC/LakeFS for datasets; a feature registry as the source
  of truth for definitions, owners, and lineage. An undocumented feature is a future incident.
- **Label quality dominates.** Bad labels cap accuracy no matter the model. Measure inter-annotator
  agreement; use weak supervision / active learning to scale; audit and reconcile noisy labels.
- **For LLM corpora, curation is the work:** dedup (exact + near-dup/MinHash), quality filtering, and
  **decontamination** against eval sets — train/test leakage silently inflates benchmark scores.
- **Govern the data:** lineage end-to-end, PII handling (minimize, mask, tokenize, access-control), and
  data contracts between producers and consumers. See `[[responsible-ai-governance]]`.

## Related skills

- `[[mlops-lifecycle]]` — where feature/data pipelines sit in the end-to-end ML lifecycle and CI/CD/CT.
- `[[ml-observability-monitoring]]` — detecting feature drift / skew in production; closing the loop.
- `[[rag-vector-databases]]` — embedding/ingestion pipelines for RAG share this data-engineering spine.
- `[[recsys-ranking]]` — the canonical heavy consumer of feature stores and point-in-time joins.
- `[[responsible-ai-governance]]` — lineage, PII, data contracts, and dataset documentation.
- `[[aiml-on-kubernetes]]` — running Spark/orchestrators/feature-store infra on K8s/GKE.

---

# Reference — data-engineering-feature-stores

# Data Engineering & Feature Stores for ML — Full Reference

The single source of truth for this skill. Data engineering for ML is the discipline of getting correct,
fresh, well-documented data into models — for both training and serving — and proving it stays that way.
"Garbage in, garbage out" is not a slogan; it is the dominant failure mode of production ML. This guide
covers ML data pipelines, the lakehouse, feature stores, data quality/validation, labeling and dataset
curation, embedding pipelines, governance, and the anti-patterns that bite.

The ecosystem moves fast (it is 2026). Where a specific API surface, store type, or limit matters,
**verify against current docs** — flagged inline. Don't quote version numbers you can't confirm.

---

## 1. Mental model: data is the model's contract with reality

A model is a function from features to predictions. Its quality is bounded by:

1. **The data it learned from** (labels + features at training time), and
2. **The data it sees in production** (features at inference time).

Three failure classes dominate, and all are data-engineering problems, not modeling problems:

- **Training-serving skew** — the feature pipeline at serving time computes something subtly different
  from the training pipeline (different code, different defaults, different aggregation window, stale
  data). The model's inputs no longer match what it learned. This is the single most common reason a
  model that "worked offline" underperforms in production.
- **Label leakage / temporal leakage** — a feature encodes information that was not available at the
  decision time the label represents (e.g. joining "current account balance" to a 6-month-old fraud
  label). Offline metrics look great; production collapses because that information isn't there yet.
- **Silent data-quality regressions** — an upstream schema change, a units change, a null spike, a new
  enum value, or a distribution shift flows into the model untyped and unchecked. No error is thrown;
  the model just gets worse.

The architecture in this guide exists to prevent these: **one feature definition** for train and serve
(kills skew), **point-in-time joins** (kill leakage), and **validation against an explicit contract**
(kills silent regressions).

Reference architectures to ground in (verify current): Google Cloud's *MLOps: Continuous delivery and
automation pipelines in machine learning*
(`cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning`),
`ml-ops.org`, the Feast docs (`docs.feast.dev`), and Vertex AI Feature Store docs.

---

## 2. ML data pipelines: ingestion → validation → transformation

The canonical ML data pipeline, regardless of tooling:

1. **Ingestion** — land raw data from sources (operational DBs via CDC, event streams, files, APIs,
   third-party feeds) into a raw/bronze layer. Capture *event time* and *ingestion time* separately;
   you need event time for correct point-in-time joins.
2. **Validation** — check the raw/landed data against a schema and distribution expectations *before*
   it propagates. Fail fast on contract violations. (Section 6.)
3. **Transformation / feature engineering** — clean, join, aggregate, encode into features. This is the
   logic that **must be shared** between training and serving. ELT (transform in the warehouse/lakehouse
   with SQL/dbt) is the default for batch ML data; reserve heavy out-of-warehouse compute (Spark) for
   scale or non-SQL transforms.
4. **Materialization / publishing** — write features to the offline store (for training) and sync to the
   online store (for serving). (Section 5.)
5. **Consumption** — training jobs read point-in-time-correct datasets; serving reads online features by
   entity key.

### Batch vs streaming — a freshness/cost tradeoff, not a religion

| Dimension | Batch | Streaming |
|---|---|---|
| Latency / freshness | minutes → hours → daily | seconds → sub-second |
| Complexity & cost | low | high (windowing, watermarks, late data, exactly-once) |
| Typical features | aggregates over days, profile attributes, embeddings | "events in last 5 min", live counters |
| Tooling | Spark, dbt, warehouse SQL, Airflow/Dagster | Flink, Spark Structured Streaming, Beam, Kafka |

Default to batch. Reach for streaming **only when staleness measurably moves the model's metric** (fraud,
real-time recsys, dynamic pricing). The trap is maintaining *two* implementations of the same feature
(batch backfill + streaming online) that drift apart — the classic skew bug. Prefer tools/feature stores
that let one definition target both, or use a stream-batch unified engine (Beam/Flink) with shared logic.

### Orchestration

The orchestrator runs the DAG, handles retries/backfills, scheduling, and lineage. Pick by model:

- **Airflow** — the incumbent; task-centric DAGs, huge operator ecosystem, time-based scheduling and
  backfills. Great for "run these steps on a schedule." Weaker on data-asset/lineage semantics
  (improving in newer versions — verify current).
- **Dagster** — asset-centric ("software-defined assets"): you declare the *data assets* and their
  dependencies; strong typing, data-aware scheduling, built-in lineage. Good fit for ML data platforms.
- **Flyte** — strongly-typed, containerized tasks with data lineage and caching; Kubernetes-native;
  popular for ML/data-science workflows and reproducibility.
- **Spark** — the distributed compute engine for large transforms/feature backfills (not an orchestrator
  per se; orchestrated by the above). PySpark for feature engineering at scale.
- **Apache Beam** — a unified batch+streaming *programming model* (runs on Dataflow/Flink/Spark runners);
  valuable when you want one pipeline definition for both batch and streaming.

Whatever you pick: make pipelines **idempotent** (re-running a partition yields the same result),
**partitioned by event date**, and **backfillable**. Parameterize by date; never write "today" logic that
can't be replayed for a historical window.

---

## 3. The lakehouse and storage formats

- **Parquet** — columnar, compressed, the de-facto file format for analytical/ML data. Column pruning +
  predicate pushdown make feature reads cheap. Almost everything below sits on Parquet.
- **Lakehouse table formats — Delta Lake, Apache Iceberg, Apache Hudi** — add a transaction log/metadata
  layer over Parquet files to provide:
  - **ACID** transactions (no half-written partitions corrupting a training read),
  - **schema evolution** (add/rename columns safely),
  - **time travel** (read the table *as of* a snapshot/version — reproducible training data without
    copying), and
  - efficient upserts/merges and partition evolution.
  Choose based on your engine/catalog ecosystem (Spark, Trino, Flink, BigQuery/BigLake, Snowflake support
  varies — **verify current** as support changes frequently). Iceberg has become a broadly adopted open
  standard; Delta is tightly integrated with Spark; Hudi targets streaming upserts/incremental.
- **Why this matters for ML:** time travel + immutable snapshots give you **reproducible training
  datasets**. Pin training to a table version/snapshot ID and you can regenerate the exact dataset later —
  essential for debugging and for `[[responsible-ai-governance]]` audits.

### Data & feature versioning

- **DVC** — Git-style versioning for datasets and model artifacts; stores large files in remote object
  storage and tracks pointers in Git. Good for reproducible experiments and small/medium dataset lineage.
- **LakeFS** — Git-like branching/commits/merges over an object store at scale; lets you create an
  isolated branch of your data lake for an experiment, validate, then merge. Pairs with lakehouse formats.
- Pin every training run to a concrete data version (DVC rev, LakeFS commit, or lakehouse snapshot ID) and
  record it with the model. "Which data trained this model?" must have a precise answer.

---

## 4. Feature stores: the mental model

A **feature store** is a centralized system that **standardizes feature definitions and serves the same
feature logic to training (offline) and serving (online)** — its core purpose is to **eliminate
training-serving skew**. It is the contract layer between data engineering and ML.

It solves four concrete problems:

1. **Skew** — one definition, served to both training and inference (the headline benefit).
2. **Reuse / discovery** — features defined once are discoverable and reusable across teams/models via a
   **registry**, instead of every model reimplementing "7-day purchase count."
3. **Point-in-time correctness** — the store builds training sets with correct as-of joins (Section 4.3).
4. **Low-latency serving** — an online store answers "give me these features for entity X" in
   milliseconds.

### 4.1 Core components

- **Registry** — the source of truth for feature definitions, entities, data sources, and metadata
  (owner, description, types). This is what makes features discoverable and governable.
- **Offline store** — historical feature values for training and batch scoring. High-throughput,
  high-latency reads over large ranges. Backed by a warehouse/lakehouse: BigQuery, Snowflake, Redshift,
  or Parquet/Iceberg/Delta files.
- **Online store** — latest feature values for online inference. Low-latency point lookups by entity key.
  Backed by a KV store: Redis, DynamoDB, Bigtable, Cassandra, or similar.
- **Transformation / compute** — where feature logic runs (batch backfill, streaming, on-demand at
  request time). Some stores compute transforms; others (e.g. Feast in its classic form) expect features
  precomputed upstream and focus on *registry + serving*. **Verify the compute model of your chosen tool.**
- **Materialization** — the process that loads feature values from the offline store into the online store
  so serving sees fresh data. Materialization cadence sets **feature freshness**.

### 4.2 Key objects (Feast-style vocabulary; concepts generalize)

- **Entity** — the thing a feature describes and the join key (e.g. `user_id`, `merchant_id`).
- **Data source** — where raw feature values live (a warehouse table, Parquet path, stream), with the
  **event-timestamp column** the store uses for point-in-time joins.
- **Feature view** — a named group of features tied to an entity and a source, with a TTL/freshness. The
  reusable unit registered in the store.
- **Feature service** — a curated bundle of features a specific model consumes (its input contract).
- **On-demand / request-time feature** — computed at request time from request inputs (and optionally
  other features), e.g. a ratio of two looked-up values. Same definition used offline and online — this is
  how you keep request-time transforms skew-free.

### 4.3 Point-in-time-correct joins (the heart of it)

To build a training set you have **label/decision events**, each with an entity and a timestamp. For each
event you must fetch the feature values **as they were at that event's timestamp** — never the latest.

A correct point-in-time (as-of) join, per event row:

- find feature rows for the same entity with `feature.event_timestamp <= label.event_timestamp`
- pick the **most recent** such row
- (optionally) reject it if it's older than the feature's TTL (too stale → null, mirroring serving)

This avoids **label leakage**: a feature computed *after* the decision time must never be visible to that
training row. A naive `JOIN ... ON entity_id` to the current value silently leaks the future and produces
offline metrics that evaporate in production.

Feature stores implement this for you (Feast: `get_historical_features` with an entity dataframe that
carries the timestamps). If you hand-roll it in SQL, use a windowed/as-of join keyed on event time — see
`examples.md`. **Test it:** a feature that "knows the future" usually shows up as an implausibly strong
offline signal. Be suspicious of features that are too good.

### 4.4 Freshness and materialization

- **Freshness** = how stale online feature values can be. Set by materialization cadence (batch sync every
  N minutes/hours) or streaming ingestion. Define an SLO per feature view; alert when materialization lags.
- **Backfill** historical feature values into the offline store so training has full history; **materialize**
  recent values into the online store for serving. These are different operations with different cadences.
- **Online/offline consistency:** the value served online must equal what an as-of join would have produced
  offline for the same timestamp. Monitor this; divergence is skew.

### 4.5 Tooling landscape (verify current capabilities — this space moves fast)

| Tool | Shape | Notes |
|---|---|---|
| **Feast** | Open-source, lightweight | Registry + offline/online serving + point-in-time joins; bring-your-own warehouse/KV; classic Feast expects features precomputed upstream. Good default OSS. |
| **Tecton** | Managed, full platform | Defines + computes batch/streaming/on-demand features; managed materialization, monitoring, SLAs. |
| **Vertex AI Feature Store** | Managed (Google Cloud) | Registry + online serving; newer generation serves features directly from BigQuery as the offline source with an online serving layer. **Verify the current architecture/API — it has changed across generations.** |
| **Featureform** | Open-source, "virtual" store | A declarative abstraction/registry layer over your existing infra (warehouse + KV) rather than a new datastore. |

Don't over-index on the tool. The discipline — one definition, point-in-time joins, a registry, freshness
SLOs — matters more than the brand. A feature store is justified when you have **multiple models/teams
sharing features and online serving needs**; a single batch model may not need one.

---

## 5. Embedding & feature pipelines for modern ML

- **Embeddings are features.** Text/image/user embeddings are produced by an upstream model and then
  stored and served like any other feature. The skew rule still applies: embed with the **same model and
  preprocessing** at training and serving time, and pin the embedding model version (a silent embedder
  upgrade re-embeds your space and breaks everything downstream).
- **Versioning:** treat the embedding model version as part of the feature's identity. Re-embedding is a
  migration, not a config tweak — plan a backfill and a cutover.
- **Link to RAG:** RAG ingestion (chunk → embed → index) is the same data-engineering spine — ingestion,
  transformation, validation, versioning — feeding a vector index instead of an online KV store. See
  `[[rag-vector-databases]]` for the retrieval side; the *pipeline* discipline here applies directly.
- **Recsys** is the canonical heavy feature-store consumer: user/item/context features, point-in-time
  joins on impression logs, real-time freshness. See `[[recsys-ranking]]`.

---

## 6. Data quality & validation

Validation is the immune system of the pipeline. Without it, bad data is indistinguishable from good data
until the model degrades in production.

### What to check

- **Schema** — column presence, types, nullability, allowed domains/enums, value ranges. Catches upstream
  breaking changes and units mistakes.
- **Distribution / statistics** — per-feature stats (mean, std, missing %, cardinality, quantiles) vs a
  reference (a trusted previous dataset or the training distribution). Catches drift and skew.
- **Volume / freshness** — row counts in expected band, partitions present, max event-timestamp recent
  enough. Catches silent upstream stoppages.
- **Cross-field / referential** — relationships that must hold (e.g. `end_date >= start_date`, FK exists).
- **Train/serve skew check** — compare serving feature distributions to training; a divergence is the
  warning that skew has set in. (Feeds `[[ml-observability-monitoring]]`.)

### Tools

- **Great Expectations** — declarative "expectations" (assertions) about data; produces validation results
  and Data Docs (human-readable data documentation). Wire into the pipeline to **fail the run** on
  violation. Good general-purpose, source-agnostic choice.
- **TensorFlow Data Validation (TFDV)** — infer a `schema` from a baseline, compute statistics, and detect
  anomalies / training-serving skew / drift against that schema. Strong in TFX pipelines; the
  *skew-and-drift* comparisons are its signature strength. Works standalone too.
- Lightweight column tests in **dbt** (e.g. `not_null`, `unique`, `accepted_values`, relationship tests)
  cover warehouse-side ELT validation cheaply — use alongside the above.

**Fail closed.** A validation failure should stop the pipeline / quarantine the partition and page an
owner — not write a warning into a log nobody reads. The whole point is to stop bad data *before* it
reaches a model.

### Data contracts

A **data contract** is an explicit, versioned agreement between a data *producer* and its *consumers*:
schema, semantics, freshness, quality guarantees, and ownership. It moves validation upstream (the
producer can't silently break the schema) and makes breakage a contract violation with a clear owner,
not a mystery model regression. Implement as schema + expectations in the producer's CI, enforced before
publish. See `[[responsible-ai-governance]]`.

---

## 7. Labeling, datasets, and corpus curation

Model accuracy is capped by label quality; this is often the highest-leverage place to spend effort.

- **Label quality** — measure **inter-annotator agreement** (e.g. Cohen's/Fleiss' kappa); low agreement
  means an ambiguous task or guidelines, not just "bad annotators." Audit a sample; reconcile conflicts;
  track label provenance. Noisy labels put a hard ceiling on achievable accuracy — fixing labels often
  beats any model change.
- **Weak supervision** — instead of hand-labeling everything, write **labeling functions** (heuristics,
  rules, distant supervision, other models) and combine their noisy votes into probabilistic labels (the
  Snorkel paradigm). Scales labeling to large unlabeled corpora; the label model learns each source's
  accuracy/correlations. Validate against a small gold set.
- **Active learning** — label the most *informative* examples first (uncertainty / margin / disagreement
  sampling, or diversity-based), iteratively retraining. Cuts labeling cost for a target accuracy; watch
  for sampling bias in the resulting set.
- **Dataset curation & dedup** — deduplicate (exact hashing + near-duplicate detection via MinHash/LSH or
  embedding similarity); duplicates waste compute, bias the model toward repeated content, and inflate
  eval if they straddle splits.
- **Decontamination (critical for LLM corpora)** — remove training documents that overlap your evaluation
  / benchmark sets (n-gram overlap or hashing against eval data). Train/eval contamination silently
  inflates benchmark scores and makes a model look better than it is. Do this *before* training and
  document the procedure. See `[[ml-evaluation-evals]]` for the eval side.
- **Documentation** — every dataset needs a datasheet/data card: source, collection method, intended use,
  known biases/limitations, license, PII status. Undocumented datasets are governance and reproducibility
  debt.

---

## 8. Governance: lineage, PII, contracts

- **Lineage** — end-to-end traceability from raw source → transform → feature → model → prediction. Needed
  for debugging ("which upstream change moved this feature?"), impact analysis, audits, and incident
  response. Asset-aware orchestrators (Dagster), lakehouse catalogs, and feature registries each capture a
  slice; stitch them.
- **PII handling** — minimize collection; classify and tag PII columns; mask/tokenize/hash where the raw
  value isn't needed; encrypt at rest/in transit; enforce access control and retention; honor
  deletion/right-to-be-forgotten down through derived features and training snapshots (immutable snapshots
  make deletion genuinely hard — design for it). Never put raw PII into feature names, logs, or embeddings
  carelessly.
- **Data contracts** (Section 6) are the governance mechanism between teams.
- Full treatment in `[[responsible-ai-governance]]`.

---

## 9. Anti-patterns (these are the ones that bite)

- **Training-serving feature skew** — separate code paths for training features (offline batch) and serving
  features (online). They drift; the model silently degrades. *Fix:* one definition served to both (feature
  store / shared transform library).
- **Leakage via non-point-in-time joins** — joining the *current* (or any future-relative) feature value to
  a historical label. Inflated offline metrics, production collapse. *Fix:* as-of joins on event time
  with TTLs; be suspicious of features that are "too predictive."
- **No data validation** — trusting upstream data implicitly. A units change, schema change, or null spike
  flows straight into the model. *Fix:* schema + distribution checks that fail the pipeline.
- **Undocumented features** — a feature with no owner, description, or definition. Nobody knows what it
  means, whether it's safe, or what breaks if the upstream changes. *Fix:* registry with required metadata.
- **One-off feature code duplicated across train & serve** — copy-pasted (and subtly divergent) feature
  logic in the training notebook and the serving service. The root cause of most skew. *Fix:* shared,
  tested feature definitions; never reimplement a feature in the serving layer.
- **"Latest" used as a feature value in training** — same as leakage; the latest value didn't exist at the
  historical decision time.
- **Streaming and batch implementations of the same feature that drift** — two sources of truth. *Fix:*
  unified definition (Beam/Flink shared logic, or a feature store that targets both).
- **Mutating data in place** — non-reproducible training. *Fix:* immutable, versioned snapshots
  (lakehouse time travel / LakeFS / DVC).
- **Validating only at train time, not at serve time** — skew goes undetected in production. *Fix:* monitor
  serving feature distributions vs training (`[[ml-observability-monitoring]]`).

---

## 10. Performance & scale

- **Push transforms to the data** (ELT in the warehouse / Spark over the lakehouse) rather than pulling raw
  data into Python. Columnar Parquet + predicate/column pushdown make feature reads cheap; select only the
  columns/partitions you need.
- **Partition by event date** and prune; avoid full-table scans for incremental feature builds. Bucket/sort
  by entity key where it helps joins.
- **Online store sizing:** size for QPS and p99 latency of point lookups; keep only the features actually
  served online hot (don't materialize the entire offline catalog). Watch hot keys and TTL/eviction.
- **Materialization cost vs freshness:** more frequent materialization = fresher features = more compute.
  Set per-feature freshness SLOs and pay only where freshness moves the metric.
- **Backfills are expensive** — make them incremental and idempotent; checkpoint progress; run on spot/
  preemptible where the orchestrator can retry.
- **Skew in Spark joins** (hot keys) and small-files problems on the lakehouse are the usual scale
  pain points — salt hot keys, compact small files, tune partition counts.

---

## 11. Troubleshooting (symptom → likely cause → fix)

- **Offline metrics excellent, production poor** → training-serving skew **or** leakage. Diff the
  serving feature values against an as-of offline computation for the same entities/timestamps; audit the
  most-important features for future information.
- **A feature looks implausibly predictive** → temporal leakage. Check the feature's event timestamp vs the
  label timestamp; ensure the as-of join and TTL are correct.
- **Model degraded gradually with no deploy** → upstream data drift or a silent schema/units change. Run
  distribution checks (TFDV/GE) against the training reference; inspect recent partitions.
- **Online predictions use stale features** → materialization lag. Check materialization job freshness vs
  SLO; check online-store TTL/eviction; alert on max-event-timestamp age.
- **Nulls/defaults in serving that weren't in training** → entity missing in online store, or different
  null handling between paths. Align default/imputation logic across train and serve (put it in the shared
  definition).
- **Can't reproduce a past training run** → unversioned data. Pin to a lakehouse snapshot / DVC rev /
  LakeFS commit and record it with the model going forward.
- **Pipeline "succeeded" but downstream is wrong** → validation gap. Add schema + volume + distribution
  checks that fail the run; the success was vacuous.

---

## 12. Version awareness

This ecosystem changes fast (2026). Specifically **verify against current docs** before relying on:

- Feature-store APIs and architectures — especially **Vertex AI Feature Store** (its design changed across
  generations: an older entity-type/featurestore model vs a newer BigQuery-backed online-serving model) and
  Feast's compute/registry capabilities.
- Lakehouse format support in a given engine/catalog (Iceberg/Delta/Hudi support in BigQuery/BigLake,
  Snowflake, Trino, Spark, Flink shifts frequently).
- Orchestrator data-asset/lineage features (Airflow, Dagster, Flyte) and streaming-engine semantics.
- Great Expectations / TFDV API surfaces (both have had notable API changes across major versions).

Don't quote version numbers or limits you can't confirm; prefer describing the capability and pointing to
the current docs.

---

## 13. Canonical references (real URLs)

- Google Cloud — *MLOps: Continuous delivery and automation pipelines in machine learning*:
  `https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning`
- Vertex AI Feature Store: `https://cloud.google.com/vertex-ai/docs/featurestore`
- Feast (open-source feature store): `https://docs.feast.dev/`
- MLOps principles: `https://ml-ops.org/`
- Great Expectations: `https://docs.greatexpectations.io/`
- TensorFlow Data Validation (TFDV): `https://www.tensorflow.org/tfx/guide/tfdv`
- Apache Iceberg: `https://iceberg.apache.org/` · Delta Lake: `https://delta.io/` ·
  Apache Hudi: `https://hudi.apache.org/`
- Apache Beam: `https://beam.apache.org/` · Apache Spark: `https://spark.apache.org/`
- Dagster: `https://docs.dagster.io/` · Apache Airflow: `https://airflow.apache.org/` ·
  Flyte: `https://docs.flyte.org/`
- DVC: `https://dvc.org/doc` · LakeFS: `https://docs.lakefs.io/`
- Tecton: `https://docs.tecton.ai/` · Featureform: `https://docs.featureform.com/`
- Snorkel / weak supervision (paper): `https://arxiv.org/abs/1711.10160`
- Deduplicating training data (LLM corpora): `https://arxiv.org/abs/2107.06499`

---

# Examples — Data Engineering & Feature Stores for ML

Canonical, imitable artifacts for the highest-leverage patterns in
`data-engineering-feature-stores-guide.md`: a feature definition, **point-in-time** training-set
retrieval (Feast-style and equivalent SQL), and data validation (TFDV + Great Expectations).

These are runnable-in-spirit. Feature-store and validation APIs **move fast (2026) — verify field names
and signatures against current docs** before depending on them. Imports trimmed for brevity.

---

## 1. Feature definition (Feast-style)

Define each feature **once** and register it; the same definition drives both the offline training join
and the online serving lookup. This is what eliminates training-serving skew.

```python
# feature_repo/features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float32, Int64

# The entity = the join key for these features.
user = Entity(name="user", join_keys=["user_id"])

# Where the precomputed feature values live. event_timestamp_field is what makes
# point-in-time joins correct — it records WHEN each feature value became valid.
user_stats_source = FileSource(
    path="s3://lake/features/user_stats/",       # Parquet / Delta / Iceberg path or a warehouse table
    timestamp_field="event_timestamp",
    created_timestamp_column="created",          # optional: tiebreaker for same event_timestamp
)

# A FeatureView: a reusable, registered group of features for an entity, with a freshness TTL.
# TTL = how stale a feature value may be before it is treated as missing (mirrors serving behavior).
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=3),
    schema=[
        Field(name="purchases_7d", dtype=Int64),
        Field(name="avg_order_value_30d", dtype=Float32),
        Field(name="account_age_days", dtype=Int64),
    ],
    online=True,                                  # also materialize to the online store for serving
    source=user_stats_source,
)

# A FeatureService = the exact feature contract a specific model consumes.
fraud_model_v1 = FeatureService(name="fraud_model_v1", features=[user_stats_fv])
```

```bash
feast apply          # register entities/views/services in the registry (source of truth)
# Materialize recent values offline -> online so serving sees fresh data (sets freshness):
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

### On-demand (request-time) feature — same definition offline and online

For transforms that need request inputs, define them once so train and serve compute identically:

```python
from feast import RequestSource, on_demand_feature_view
from feast.types import Float64

txn_request = RequestSource(name="txn", schema=[Field(name="amount", dtype=Float64)])

@on_demand_feature_view(
    sources=[user_stats_fv, txn_request],
    schema=[Field(name="amount_vs_avg", dtype=Float64)],
)
def amount_vs_avg(inputs):
    import pandas as pd
    out = pd.DataFrame()
    # safe ratio; identical math offline (in the training join) and online (at request time)
    out["amount_vs_avg"] = inputs["amount"] / (inputs["avg_order_value_30d"] + 1.0)
    return out
```

---

## 2. Point-in-time-correct training-set retrieval

The entity dataframe carries **one timestamp per label/decision event**. The store fetches each feature
**as of that timestamp** — never the latest — so no future information leaks into the training row.

```python
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Label events: each row is "for THIS user, AT THIS time, the outcome was THIS".
entity_df = pd.DataFrame({
    "user_id":        [1001, 1002, 1001],
    "event_timestamp": pd.to_datetime([                 # the decision time — drives the as-of join
        "2026-05-01 10:00:00", "2026-05-02 12:30:00", "2026-05-20 09:15:00",
    ]),
    "is_fraud":       [0, 1, 0],                          # the label, carried through untouched
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("fraud_model_v1"),
).to_df()
# For each row, purchases_7d/avg_order_value_30d/account_age_days are the values that were valid
# at event_timestamp (most recent feature row with feature_ts <= event_timestamp, within TTL).
```

At **serving** time the *same* features are read by entity key from the online store (latest values):

```python
features = store.get_online_features(
    features=store.get_feature_service("fraud_model_v1"),
    entity_rows=[{"user_id": 1001}],
).to_dict()   # identical feature logic as training -> no skew
```

### The equivalent as-of join in plain SQL (when you hand-roll it)

If you build training sets without a feature store, the join MUST be point-in-time. The bug to avoid is
a plain `JOIN ON user_id` to a "current" value — that leaks the future.

```sql
-- ✅ CORRECT: for each label event, take the most recent feature row at-or-before the event time,
--    and only if it is within the freshness window (TTL).
SELECT
  l.user_id,
  l.event_timestamp,
  l.is_fraud,
  f.purchases_7d,
  f.avg_order_value_30d
FROM labels AS l
LEFT JOIN LATERAL (
  SELECT *
  FROM   user_stats AS f
  WHERE  f.user_id = l.user_id
    AND  f.event_timestamp <= l.event_timestamp                       -- no future leakage
    AND  f.event_timestamp >  l.event_timestamp - INTERVAL '3 days'   -- TTL: too-stale -> NULL
  ORDER BY f.event_timestamp DESC
  LIMIT 1
) AS f ON TRUE;

-- ❌ WRONG: leaks the future and inflates offline metrics.
-- SELECT l.*, f.purchases_7d
-- FROM labels l JOIN user_stats_latest f ON f.user_id = l.user_id;   -- "latest" != "as of decision time"
```

---

## 3. Data validation — TFDV (schema + skew/drift)

TFDV's signature strength: compute stats, infer/curate a schema, then detect anomalies and
**training-serving skew / drift** against it. Wire the anomaly check to fail the pipeline.

```python
import tensorflow_data_validation as tfdv

# 1) Baseline: stats + a curated schema from a trusted training dataset.
train_stats = tfdv.generate_statistics_from_csv("train.csv")
schema = tfdv.infer_schema(train_stats)   # then review & tighten by hand (domains, required, ranges)

# 2) Validate a new/serving batch against that schema.
serving_stats = tfdv.generate_statistics_from_csv("serving.csv")
anomalies = tfdv.validate_statistics(serving_stats, schema)

# 3) Explicit skew/drift comparators (the reason to use TFDV):
#    skew_comparator -> training vs serving; drift_comparator -> over time.
purchases = tfdv.get_feature(schema, "purchases_7d")
purchases.skew_comparator.infinity_norm.threshold = 0.01   # tune per feature

skew_anomalies = tfdv.validate_statistics(
    statistics=serving_stats, schema=schema, serving_statistics=serving_stats)

if anomalies.anomaly_info or skew_anomalies.anomaly_info:
    # FAIL CLOSED: stop the pipeline / quarantine the batch; page the owner. Do not just log.
    raise ValueError(f"data validation failed: {anomalies.anomaly_info}")
```

---

## 4. Data validation — Great Expectations (schema + contract gates)

Declarative expectations that double as human-readable **data documentation**. Run in the pipeline and
fail on violation.

```python
import great_expectations as gx

context = gx.get_context()
batch = context.data_sources.add_pandas("ml").add_dataframe_asset("features") \
    .add_batch_definition_whole_dataframe("today").get_batch(batch_parameters={"dataframe": df})

suite = context.suites.add(gx.ExpectationSuite("feature_contract"))
# Schema / contract:
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="purchases_7d", type_="int64"))
# Domain / range (catches units & corruption bugs):
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="account_age_days", min_value=0, max_value=20000))
# Distribution / freshness sanity (catches silent upstream regressions):
suite.add_expectation(gx.expectations.ExpectColumnMeanToBeBetween(
    column="avg_order_value_30d", min_value=1.0, max_value=10000.0))

result = batch.validate(suite)
if not result.success:
    raise SystemExit("data contract violated — blocking publish")  # fail closed, then build Data Docs
```

> Both GE and TFDV have changed their APIs across major versions. **Treat the exact method/argument names
> above as illustrative and verify against the version you have installed.** The discipline — schema +
> distribution + freshness checks that *fail the pipeline* — is the part that doesn't change.

---

## 5. Idempotent, point-in-time-aware pipeline task (orchestrator-agnostic)

The shape every feature-build task should have: parameterized by date, idempotent (re-runnable for any
partition), event-time partitioned, and validated before publish.

```python
def build_user_stats(run_date: str):           # e.g. "2026-05-20"; backfillable for any past date
    raw = read_source(partition=run_date)       # read ONE event-date partition (prune; no full scan)
    validate_or_fail(raw)                        # schema/volume checks BEFORE transform (fail closed)
    feats = transform(raw)                       # the ONE shared definition used by train & serve
    feats["event_timestamp"] = to_event_time(raw)
    write_offline(feats, partition=run_date, mode="overwrite")  # idempotent: replace, don't append
    # online materialization (freshness) happens separately, on its own cadence
```

Pin the training run to a concrete data version (lakehouse snapshot id / DVC rev / LakeFS commit) and
record it with the model so the dataset is exactly reproducible later.
