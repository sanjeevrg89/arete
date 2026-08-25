---
name: data-engineering-feature-stores
description: Expert data engineering for ML — the pipelines, feature stores, and data-quality discipline
  that decide whether models work in production ("garbage in, garbage out"). Use when building or debugging
  ML data pipelines (ingestion, validation, transformation, batch vs streaming), orchestration (Airflow,
  Dagster, Flyte, Spark, Beam), the lakehouse (Delta Lake, Apache Iceberg, Hudi, Parquet), or data
  versioning (DVC, LakeFS, lakeFS). Use for streaming & real-time features (Kafka/Pulsar, Flink, Spark
  Structured Streaming, Beam, CDC/Debezium, windowed aggregations, watermarks, late/out-of-order data,
  exactly-once, online/offline consistency) and real-time inference (fraud, recsys). Use for the analytics
  /query side — data warehouses (BigQuery, Snowflake, Redshift) and OLAP/lakehouse query engines (Spark
  SQL, Trino/Presto, DuckDB), columnar/partitioning/clustering and query cost, SQL for ML (window
  functions, point-in-time/as-of joins, cohorting), and dbt-style transformation/label/feature derivation.
  Use for feature stores (Feast, Tecton, Vertex AI Feature Store,
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
- **Streaming features (Kafka/Pulsar log → Flink/Spark Structured Streaming/Beam → online store; CDC via
  Debezium):** compute on **event time** with windows, watermarks, and an explicit late-data policy; aim for
  exactly-once (or at-least-once + idempotent sink). The hard part is **online/offline consistency** — the
  online value must equal the offline as-of value; use one definition for both and diff them. Streaming
  features earn their cost only for fraud / session recsys / dynamic pricing. See `[[distributed-systems-fundamentals]]`.
- **The analytics/query side is where most labels and batch features are derived:** warehouses
  (BigQuery/Snowflake/Redshift) and OLAP query engines (Spark SQL, Trino, DuckDB) over columnar data.
  Partition by event date and prune; avoid `SELECT *`/full scans (bytes scanned = cost). Master SQL window
  functions and **point-in-time/as-of joins** (the #1 leakage bug), and use **dbt** for versioned,
  tested transformation/label derivation. dbt/SQL *derives*; the feature store *serves* — keep the boundary.
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
- `[[distributed-systems-fundamentals]]` — partitioning, ordering, delivery semantics, and exactly-once
  underpinning streaming feature pipelines (Kafka/Flink) and online stores.
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
   it propagates. Fail fast on contract violations. (Section 8.)
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

## 6. Streaming & real-time data systems for ML

Section 2 framed batch vs streaming as a freshness/cost tradeoff. This section is the depth: how the
streaming side actually works, and the one hard problem it forces on you — **online/offline consistency**.
Reach for any of this only when staleness measurably moves the model's metric (fraud, real-time recsys,
dynamic pricing, abuse/anomaly detection). When it does, getting it wrong is the dominant source of
training-serving skew.

### 6.1 The architecture: log + compute + sink

Real-time feature systems are layered. Keep the layers distinct in your head:

- **The log (durable, replayable event bus)** — **Apache Kafka** (the de-facto standard) or **Apache
  Pulsar**. This is the source of truth for events: an append-only, partitioned, ordered-within-partition,
  retained log you can **replay**. Replayability is what makes streaming features reproducible — you can
  reprocess history through the same logic. **CDC (Change Data Capture)** with **Debezium** turns an
  operational database's write-ahead log into a Kafka topic, so feature freshness tracks the source DB
  without dual-writes or polling. CDC is the most common way to keep online features fresh from an OLTP
  system.
- **The compute (stream processor)** — consumes the log, computes stateful aggregations, writes results:
  - **Apache Flink** — the reference stateful stream processor: true event-time processing, rich windowing,
    large managed keyed state with checkpointing, exactly-once sinks. The strongest choice for heavy
    stateful streaming features.
  - **Spark Structured Streaming** — micro-batch (and a lower-latency continuous mode); attractive when you
    already run Spark and want the *same* DataFrame/SQL code for batch and stream (helps consistency).
  - **Apache Beam** — a unified batch+streaming *programming model* (one pipeline, multiple runners:
    Dataflow/Flink/Spark). Its event-time/windowing model is the cleanest way to express "one definition,
    both modes."
  - **Kafka Streams / ksqlDB** — lighter-weight, library/SQL-based stream processing co-located with Kafka;
    fine for simpler per-key aggregations without standing up Flink.
- **The sink** — the online store (Redis/DynamoDB/Bigtable/Cassandra) for low-latency serving, and/or the
  offline store (lakehouse/warehouse) for training history. Streaming features typically write **both**.

The streaming engine is a distributed stateful system; its correctness rests on the foundations in
`[[distributed-systems-fundamentals]]` (partitioning, ordering, consensus/checkpointing, delivery
semantics). The concepts below are those foundations applied to feature computation.

### 6.2 Event time, windows, watermarks, late data

The four concepts that make or break streaming feature correctness:

- **Event time vs processing time.** Compute features on **event time** (when the event happened), never
  processing time (when your job saw it) — otherwise a consumer lag spike silently changes feature values.
  This is the streaming analogue of point-in-time correctness.
- **Windows** — bound an aggregation over time: **tumbling** (fixed, non-overlapping: "count per 1-min
  bucket"), **sliding/hopping** (overlapping: "count over the last 5 min, updated every 30 s"), **session**
  (gap-delimited activity bursts). "Transactions in the last 5 minutes" is a sliding event-time window.
- **Watermarks** — the engine's estimate of "event time has progressed to T; assume no events older than T
  will still arrive." A watermark lets a window *close* and emit. Set it from the **allowed lateness** you
  can tolerate: too tight drops legitimately late events (undercount); too loose holds state and delays
  emission (latency + memory). This is a direct freshness-vs-correctness knob.
- **Late / out-of-order data** — events arrive after their window closed (mobile clients, network delays,
  CDC backlog). Decide explicitly: drop, route to a side output, or allow-late-and-update (recompute and
  re-emit). The choice changes the feature value; pick it deliberately and make batch match it.
- **Exactly-once** — duplicate or lost events corrupt counters/aggregates. Achieved end-to-end via
  checkpointed operator state plus transactional/idempotent sinks (Flink checkpoints + transactional Kafka;
  idempotent writes to the online store). "At-least-once + idempotent sink" is the common pragmatic target.
  Understand the guarantee your pipeline actually provides — see `[[distributed-systems-fundamentals]]`.

### 6.3 Online feature stores & the online/offline consistency problem

A streaming feature must be computed in the stream **and** be reconstructable as point-in-time-correct
history for training. The hazard: you write a streaming job for the online value and a *separate* batch job
for the training history, and they diverge — the headline training-serving skew bug, now with two codebases
instead of one.

The **online/offline consistency** requirement: the value served online at time `t` must equal what an
as-of join would produce offline for that same entity and `t`. Concretely, "5-minute purchase count"
must use the same window definition, the same late-data policy, and the same null/default handling in both
paths. How platforms address it (**verify current capabilities — this space moves fast**):

- **Single definition, dual execution.** A feature platform (Tecton-style; the broad pattern popularized by
  Uber's published *Michelangelo* platform) takes **one feature definition** and runs it as a streaming job
  for online freshness *and* as a backfill over historical logs for the offline training set — same logic,
  two runtimes. This is the structural fix for streaming skew.
- **Push computed features into the store's online layer.** **Feast** supports a stream/push path
  (`push` / a stream source) so a Flink/Spark job writes freshly computed features into the online store
  while the offline store retains history for point-in-time joins. Feast itself does not heavily *compute*
  streaming aggregations — you compute upstream and push. **Verify the current Feast stream API.**
- **Unify the engine.** Express the feature once in Beam or in Spark/Flink SQL and run the *same* code as a
  bounded (batch backfill) and unbounded (streaming) job. Fewer moving parts, less drift.
- **Continuously test consistency.** Periodically diff online-served values against an offline as-of
  recomputation for the same entities/timestamps; alert on divergence. This is a feature-level skew monitor
  and belongs in `[[ml-observability-monitoring]]`.

**Freshness vs cost.** Streaming is materially more expensive and operationally heavier than batch
(standing stateful jobs, state backends, checkpoint storage, on-call). Set a **freshness SLO per feature**
and pay for streaming only where the model's metric is sensitive to staleness. Many "real-time" features
are fine at minute-level micro-batch; reserve sub-second streaming for the few features that need it.

### 6.4 Real-time inference pipelines — when streaming features matter

The end-to-end real-time path: event → stream processor updates a feature in the online store → at request
time the model reads online features by entity key (plus on-demand/request-time features computed from the
request itself) → prediction. Streaming features earn their cost when **the signal is in very recent
behavior**:

- **Fraud / abuse / anomaly detection** — "velocity" features (count/amount in the last N seconds/minutes,
  count of distinct devices/IPs) are the canonical case; a 5-minute-stale counter misses the attack.
- **Real-time recsys / ranking** — session and within-session signals (what the user clicked *this
  session*), trending-now item counts. Often **batch features for stable profile/embedding signals +
  streaming features for session freshness**, combined at request time. See `[[recsys-ranking]]`.
- **Dynamic pricing, real-time bidding, live ops** — decisions on the current state of a fast-moving system.

For everything else, batch features served from the online store are simpler, cheaper, and correct. The
question is never "batch or streaming?" but "which *features* need streaming freshness?" — usually a small
subset, mixed with a batch majority in the same model.

---

## 7. Analytical data, SQL & OLAP — the warehouse/query side

Feature stores are the *serving* side. This section is the *analytics & derivation* side: the warehouse and
query engines where you explore data, derive labels, build cohorts, and compute batch features in SQL before
they are published to the offline store and materialized online. Most batch features and nearly all labels
originate here. (How this feeds the feature pipeline: derive in SQL/dbt → land in the offline store →
point-in-time-join for training and materialize for serving, per Sections 2 and 4.)

### 7.1 Warehouses and lakehouse query engines

- **Cloud data warehouses** — **BigQuery**, **Snowflake**, **Amazon Redshift**: managed, columnar,
  massively-parallel (MPP) SQL engines that separate (or elastically scale) storage and compute. The default
  home for analytical SQL and ELT-based feature/label derivation.
- **Lakehouse query engines** — read SQL directly over open table formats (Iceberg/Delta/Hudi on Parquet,
  Section 3) without loading into a warehouse:
  - **Spark SQL** — SQL over the lakehouse on the Spark engine; same engine you use for heavy transforms.
  - **Trino / Presto** — distributed MPP SQL query engine; federates across the lakehouse, warehouses, and
    operational stores; strong for interactive analytics over large data without ingestion.
  - **DuckDB** — in-process (single-node) columnar OLAP engine; superb for local/medium-scale analytics,
    notebook feature exploration, and reading Parquet/Iceberg directly. Often the fastest path for "analyze
    this dataset" without standing up a cluster.

### 7.2 Columnar / OLAP fundamentals (why these are fast — and how to keep them cheap)

- **Columnar storage** — values of one column are stored contiguously, so a query reads only the columns it
  references (**column pruning**) and compresses well (run-length/dictionary encoding on like values). This
  is why analytical reads of a few columns over billions of rows are cheap; row stores (OLTP) are the
  opposite tradeoff. Parquet is the on-disk embodiment.
- **OLAP vs OLTP** — OLAP (these engines) is scan-and-aggregate over many rows, few columns; OLTP is
  point read/write of whole rows. Don't run analytics on the OLTP DB (and don't serve point lookups from
  the warehouse — that's the online store's job).
- **Partitioning** — physically split a table by a column (usually **event date**) so the engine scans only
  relevant partitions (**partition pruning**). The single biggest lever on query cost. Always partition
  large feature/event tables by event date and filter on it.
- **Clustering / sorting / bucketing** — order data within partitions by frequently-filtered columns
  (BigQuery clustering, Snowflake clustering keys, Spark/Iceberg sort/bucket) so the engine skips blocks
  (min/max **data skipping**) and joins/aggregations are cheaper.
- **Query cost** — on consumption-priced warehouses (e.g. BigQuery bytes-scanned, Snowflake credits) cost is
  driven by **bytes scanned**: prune columns (never `SELECT *` on wide tables), prune partitions (filter on
  the partition column), and pre-aggregate. A careless full-table scan over a fact table is a real bill.
  Materialize expensive repeated aggregations rather than recomputing them per query.

### 7.3 SQL proficiency for ML

The SQL that matters for ML data work goes well beyond `GROUP BY`:

- **Window functions** — `OVER (PARTITION BY entity ORDER BY event_time ...)` with
  `ROW_NUMBER`/`RANK`/`LAG`/`LEAD` and framed aggregates (`SUM(...) OVER (... ROWS/RANGE BETWEEN ...)`).
  These compute per-entity running aggregates and "last value before time T" **without leaking the future**
  — the SQL backbone of feature derivation and of hand-rolled point-in-time joins.
- **Point-in-time / as-of joins** — the most important and most error-prone pattern: for each label event,
  pick the most recent feature row with `feature_ts <= label_ts` (honoring a TTL). Expressed via a lateral
  join or a windowed `ROW_NUMBER() ... QUALIFY row_num = 1` over the union of label and feature timestamps.
  A plain `JOIN ON entity_id` to a "current" value leaks the future (Section 4.3). See `examples.md` for the
  full pattern; it is the #1 leakage bug.
- **Cohorting & funnels** — grouping entities by an acquisition/first-event period and tracking behavior over
  subsequent periods (retention/cohort tables, funnel step conversion). The basis of many labels (churn,
  conversion, LTV) and of segment features.
- **Sessionization** — grouping events into sessions by inactivity gap (window functions over event-time
  diffs) — the batch counterpart of streaming session windows (Section 6.2); keep the two definitions aligned.

### 7.4 dbt-style transformation & modeling

**dbt** (and equivalents) brings software engineering to warehouse SQL: transformations as version-controlled
`SELECT` models with a dependency DAG, environments, and tests. For ML it is the standard way to derive
labels and batch features in the warehouse/lakehouse before they reach the feature store:

- **Models & DAG** — each transform is a `ref()`-linked model; dbt builds the dependency graph, handles
  incremental materializations (process only new partitions), and gives you lineage.
- **Tests as a data contract** — built-in `not_null` / `unique` / `accepted_values` / `relationships` tests
  plus custom tests run in CI and **fail the build** on violation — cheap warehouse-side validation that
  complements Great Expectations/TFDV (Section 8). This is where many data contracts (Section 8) live.
- **Docs & lineage** — generated model docs and a lineage graph make derived features discoverable and
  auditable, feeding governance (Section 10).
- **Where it sits** — dbt/SQL is **analytics and feature/label *derivation***; the feature store is
  *registration, point-in-time training retrieval, and low-latency serving*. dbt produces the offline-store
  tables a feature view points at; it does **not** replace point-in-time joins or the online store. Keep the
  boundary clear: derive in SQL, serve through the store.

**Verify against current docs:** warehouse-specific syntax differs (BigQuery `QUALIFY`, Snowflake
`QUALIFY`, `ASOF JOIN` support varies by engine — DuckDB and some engines have a native `ASOF JOIN`; others
require the lateral/window pattern). dbt's materialization and testing APIs also evolve across versions.

---

## 8. Data quality & validation

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

## 9. Labeling, datasets, and corpus curation

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

## 10. Governance: lineage, PII, contracts

- **Lineage** — end-to-end traceability from raw source → transform → feature → model → prediction. Needed
  for debugging ("which upstream change moved this feature?"), impact analysis, audits, and incident
  response. Asset-aware orchestrators (Dagster), lakehouse catalogs, and feature registries each capture a
  slice; stitch them.
- **PII handling** — minimize collection; classify and tag PII columns; mask/tokenize/hash where the raw
  value isn't needed; encrypt at rest/in transit; enforce access control and retention; honor
  deletion/right-to-be-forgotten down through derived features and training snapshots (immutable snapshots
  make deletion genuinely hard — design for it). Never put raw PII into feature names, logs, or embeddings
  carelessly.
- **Data contracts** (Section 8) are the governance mechanism between teams.
- Full treatment in `[[responsible-ai-governance]]`.

---

## 11. Anti-patterns (these are the ones that bite)

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
  unified definition (Beam/Flink shared logic, or a feature store that targets both); diff online vs an
  offline as-of recomputation and alert on divergence (Section 6.3).
- **Aggregating streaming features on processing time** — a consumer-lag spike silently shifts feature
  values, and late events undercount. *Fix:* event-time windows with explicit watermarks and a deliberate
  late-data policy (Section 6.2); make the batch backfill use the *same* window/late-data semantics.
- **Full-table scans / `SELECT *` on fact tables** — slow queries and (on consumption-priced warehouses) a
  real bill, from scanning columns/partitions you don't need. *Fix:* prune columns, filter the partition
  column, pre-aggregate/materialize repeated heavy aggregations (Section 7.2).
- **Mutating data in place** — non-reproducible training. *Fix:* immutable, versioned snapshots
  (lakehouse time travel / LakeFS / DVC).
- **Validating only at train time, not at serve time** — skew goes undetected in production. *Fix:* monitor
  serving feature distributions vs training (`[[ml-observability-monitoring]]`).

---

## 12. Performance & scale

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

## 13. Troubleshooting (symptom → likely cause → fix)

- **Offline metrics excellent, production poor** → training-serving skew **or** leakage. Diff the
  serving feature values against an as-of offline computation for the same entities/timestamps; audit the
  most-important features for future information.
- **A feature looks implausibly predictive** → temporal leakage. Check the feature's event timestamp vs the
  label timestamp; ensure the as-of join and TTL are correct.
- **Model degraded gradually with no deploy** → upstream data drift or a silent schema/units change. Run
  distribution checks (TFDV/GE) against the training reference; inspect recent partitions.
- **Online predictions use stale features** → materialization lag (batch) or stream-processor consumer lag
  /backpressure (streaming). Check materialization job freshness vs SLO; check online-store TTL/eviction;
  check the stream processor's consumer lag and checkpoint health; alert on max-event-timestamp age.
- **Streaming feature value disagrees with the offline as-of value** → online/offline inconsistency: the
  stream and the batch backfill use different window/late-data/null semantics, or processing vs event time.
  Align both to one definition; diff them continuously (Section 6.3).
- **Warehouse query slow or expensive** → no partition pruning / `SELECT *` / unclustered scan. Filter on the
  partition (event-date) column, select only needed columns, add clustering/sort keys, pre-aggregate
  (Section 7.2).
- **Nulls/defaults in serving that weren't in training** → entity missing in online store, or different
  null handling between paths. Align default/imputation logic across train and serve (put it in the shared
  definition).
- **Can't reproduce a past training run** → unversioned data. Pin to a lakehouse snapshot / DVC rev /
  LakeFS commit and record it with the model going forward.
- **Pipeline "succeeded" but downstream is wrong** → validation gap. Add schema + volume + distribution
  checks that fail the run; the success was vacuous.

---

## 14. Version awareness

This ecosystem changes fast (2026). Specifically **verify against current docs** before relying on:

- Feature-store APIs and architectures — especially **Vertex AI Feature Store** (its design changed across
  generations: an older entity-type/featurestore model vs a newer BigQuery-backed online-serving model) and
  Feast's compute/registry capabilities.
- Lakehouse format support in a given engine/catalog (Iceberg/Delta/Hudi support in BigQuery/BigLake,
  Snowflake, Trino, Spark, Flink shifts frequently).
- Orchestrator data-asset/lineage features (Airflow, Dagster, Flyte) and streaming-engine semantics
  (Flink/Spark Structured Streaming windowing, watermark, and exactly-once details; Feast's stream/push API).
- Great Expectations / TFDV API surfaces (both have had notable API changes across major versions).
- Warehouse/engine SQL surface — `QUALIFY`, native `ASOF JOIN` support, and clustering/partitioning syntax
  differ across BigQuery, Snowflake, Trino, Spark SQL, and DuckDB; dbt materialization/test APIs evolve too.

Don't quote version numbers or limits you can't confirm; prefer describing the capability and pointing to
the current docs.

---

## Rationalizations & rebuttals

The excuses that precede a skew or leakage incident, each with the one-line answer:

- *"It's faster to just rewrite the feature in the serving service than wire up the store."* → That copy
  is the #1 cause of training-serving skew (Section 11). Two implementations drift the day after you
  ship. One tested definition served to both, or it isn't done.
- *"Random k-fold CV is fine for this."* → Random splits leak the future across folds for any
  time-ordered or entity-grouped data. Use time-based / out-of-time (and entity-grouped) splits, or your
  offline score is fiction (Sections 4.3, 7.3).
- *"Point-in-time joins are overkill — I'll just join the current feature value."* → A plain
  `JOIN ON entity_id` to the latest value leaks information that didn't exist at decision time; metrics
  inflate offline and collapse in production. As-of join on event time with a TTL is non-negotiable
  (Section 4.3).
- *"The upstream data is clean, we don't need validation."* → Clean today, a silent units/schema/null
  change tomorrow — no error thrown, model just degrades (Section 8). Schema + distribution + freshness
  checks that fail closed cost a day and save a quarter.
- *"Undocumented features are fine, the author knows what they mean."* → The author leaves, the upstream
  changes, and nobody knows if the feature is safe or what breaks. Registry entry with owner +
  description + types, or it's not a feature, it's debt (Sections 4.1, 11).
- *"We'll validate at training time; serving is the same data."* → It demonstrably isn't — that
  assumption *is* skew. Monitor serving feature distributions against the training reference
  (Sections 8, 11).
- *"Processing-time windows are close enough for the streaming feature."* → A consumer-lag spike silently
  shifts every value and late events undercount. Event-time windows with explicit watermarks and a
  deliberate late-data policy, matched in the batch backfill (Section 6.2).
- *"It's a one-off backfill, I'll mutate the table in place."* → Now the training run is unreproducible
  forever. Immutable, versioned snapshots (lakehouse time travel / LakeFS / DVC) (Sections 3, 11).

---

## Red flags (stop and reconsider)

- **Feature logic exists in more than one place** — the training notebook/job and the serving service
  each compute "7-day count." Skew is already shipping (Sections 4, 11).
- **A join to a feature with no event-time / TTL condition** in training-set construction — non-point-in-time
  join, i.e. leakage by construction (Section 4.3).
- **A feature that is "too predictive"** / an implausibly strong offline signal — almost always temporal
  leakage; audit feature timestamp vs label timestamp (Sections 4.3, 13).
- **No validation step between ingestion and the model** — or validation that logs a warning instead of
  failing the run (does not "fail closed") (Section 8).
- **Stale online features** — materialization/consumer lag with no freshness SLO or alert on
  max-event-timestamp age; serving silently reads old values (Sections 4.4, 6.3).
- **Schema/units drift entering untyped** — a new enum value, a renamed column, or a millis-vs-seconds
  change flowing through with no contract to catch it (Sections 8, 11).
- **Streaming and batch implementations of the same feature** with no continuous online-vs-offline
  consistency diff — two sources of truth that will diverge (Sections 6.3, 11).
- **Validating only at train time, never at serve time** — skew goes undetected in production
  (Sections 8, 11).
- **A training run with no pinned data version** (lakehouse snapshot / DVC rev / LakeFS commit) — "which
  data trained this?" has no precise answer (Sections 3, 13).
- **Raw PII in feature names, logs, or embeddings**, or no deletion path through derived features and
  snapshots (Section 10).

---

## Verification gate (definition of done)

The work is not done until every item is true:

- [ ] **Single feature definition** for train and serve — no feature logic reimplemented in the serving
  layer; the shared definition (feature store / shared transform lib) is what both paths call.
- [ ] **Point-in-time-correct joins** for every training set — as-of join on event time honoring TTL; no
  `JOIN`-to-current-value anywhere in training-set construction.
- [ ] **Time-based (and entity-grouped where relevant) splits** — no random CV on time-ordered data.
- [ ] **Data validation gates the pipeline and fails closed** — schema + distribution + volume/freshness +
  cross-field checks; a violation stops the run / quarantines the partition and pages an owner.
- [ ] **Serve-time validation / skew monitor in place** — serving feature distributions compared to the
  training reference; alert on divergence.
- [ ] **Streaming features prove online/offline consistency** (if any) — online-served value diffed
  against an offline as-of recomputation; event-time windows, watermarks, and late-data policy matched in
  batch.
- [ ] **Freshness SLO defined per online feature** and alerted on (materialization/consumer lag,
  max-event-timestamp age).
- [ ] **Every served feature is registered** with owner, description, and types; every dataset has a
  datasheet/data card.
- [ ] **Lineage captured** raw source → transform → feature → model, and the training run is **pinned to a
  concrete data version** (snapshot ID / DVC rev / LakeFS commit) recorded with the model.
- [ ] **PII classified, tagged, and handled** — masked/tokenized where raw isn't needed; access control,
  retention, and a deletion path through derived features and snapshots.

> Note on fast-moving tooling: feature-store APIs (Vertex AI Feature Store, Feast stream/push),
> lakehouse-format support per engine, Great Expectations/TFDV surfaces, and streaming-engine semantics
> change frequently (2026) — verify the exact API/capability against current docs before relying on it
> (Section 14).

---

## 15. Canonical references (real URLs)

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
- Apache Kafka: `https://kafka.apache.org/documentation/` · Apache Pulsar: `https://pulsar.apache.org/docs/` ·
  Debezium (CDC): `https://debezium.io/documentation/`
- Apache Flink: `https://nightlies.apache.org/flink/flink-docs-stable/` ·
  Spark Structured Streaming: `https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html`
- Uber *Michelangelo* ML platform (published overview): `https://www.uber.com/blog/michelangelo-machine-learning-platform/`
- dbt: `https://docs.getdbt.com/` · Trino: `https://trino.io/docs/current/` ·
  DuckDB: `https://duckdb.org/docs/` · BigQuery: `https://cloud.google.com/bigquery/docs` ·
  Snowflake: `https://docs.snowflake.com/`
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
retrieval (Feast-style and equivalent SQL), data validation (TFDV + Great Expectations), a **streaming
windowed feature with the online/offline-consistency contract**, and **point-in-time feature derivation in
warehouse SQL** with window functions.

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

---

## 6. Streaming windowed feature + the online/offline-consistency contract

A real-time "transactions in the last 5 minutes per user" velocity feature for fraud. The point of this
example is **the consistency contract**: the streaming job (online freshness) and the batch backfill
(offline training history) must compute the *identical* value for the same entity and timestamp, or you
have reintroduced training-serving skew with two codebases. Compute on **event time**, with a watermark and
an explicit late-data policy — and make the batch side use the same window/late-data semantics.

### Online: event-time sliding window in Flink SQL (writes to the online store)

```sql
-- Source: a Kafka topic of transactions; event_time is the EVENT timestamp (not arrival time),
-- with a watermark allowing 2 minutes of out-of-order/late events before a window is finalized.
CREATE TABLE txns (
  user_id     BIGINT,
  amount      DOUBLE,
  event_time  TIMESTAMP(3),
  WATERMARK FOR event_time AS event_time - INTERVAL '2' MINUTE   -- allowed lateness = freshness/correctness knob
) WITH ('connector' = 'kafka', 'topic' = 'txns', /* ...format/bootstrap... */);

-- 5-minute window, updated every 1 minute (sliding/hop) on EVENT time. The result is upserted into the
-- online store keyed by user_id, so serving reads the freshest value by entity key.
INSERT INTO online_user_txn_features              -- sink: Redis/Bigtable/DynamoDB via an upsert connector
SELECT
  user_id,
  window_end                         AS event_timestamp,
  COUNT(*)                           AS txn_count_5m,
  COALESCE(SUM(amount), 0.0)         AS txn_amount_5m
FROM TABLE(
  HOP(TABLE txns, DESCRIPTOR(event_time), INTERVAL '1' MINUTE, INTERVAL '5' MINUTE)
)
GROUP BY user_id, window_start, window_end;
```

### Offline: the SAME window logic as a backfill over the event log (for training history)

```sql
-- Replay the historical event log (Kafka tier / lakehouse table) through the IDENTICAL window definition
-- so the training value for (user, t) equals what the streaming job served online at t. Same 5-minute
-- event-time window, same null/zero handling. This row is then point-in-time-joined to labels (Example 2).
SELECT
  user_id,
  window_end                         AS event_timestamp,     -- the feature's valid-from time
  COUNT(*)                           AS txn_count_5m,
  COALESCE(SUM(amount), 0.0)         AS txn_amount_5m
FROM TABLE(
  HOP(TABLE txns_history, DESCRIPTOR(event_time), INTERVAL '1' MINUTE, INTERVAL '5' MINUTE)
)
GROUP BY user_id, window_start, window_end;
-- Publish to the offline store; a FeatureView points here. get_historical_features then does the as-of join.
```

> **Consistency check (run it continuously):** for a sample of entities/timestamps, diff the value the
> online store served against this offline recomputation. Any divergence = skew — usually a mismatch in
> window size, watermark/late-data policy, or event-vs-processing time. This is a feature-level skew
> monitor; wire it to `[[ml-observability-monitoring]]`. Flink SQL windowing/connector syntax moves fast —
> **verify against the version you run.**

---

## 7. Point-in-time (as-of) feature derivation in warehouse SQL with window functions

When you derive features and the as-of join in the warehouse (no feature store), window functions are the
tool. Compute a per-event "value as of this event's time" without leaking the future, in one pass.

```sql
-- Goal: for each label event, attach the most-recent prior feature snapshot (as-of join), and also a
-- running per-user aggregate computed only from PAST rows. Both must exclude the future.
WITH events AS (
  -- Union labels and feature snapshots on a common (user_id, ts) timeline; tag the source.
  SELECT user_id, event_timestamp AS ts, 'label'   AS src, is_fraud, CAST(NULL AS BIGINT) AS purchases_7d FROM labels
  UNION ALL
  SELECT user_id, event_timestamp AS ts, 'feature' AS src, NULL,     purchases_7d                         FROM user_stats
),
carried AS (
  SELECT
    *,
    -- as-of: the most recent feature value at-or-before this row's ts (NULL until the first feature row).
    LAST_VALUE(purchases_7d IGNORE NULLS) OVER (
      PARTITION BY user_id ORDER BY ts
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS purchases_7d_asof
  FROM events
)
SELECT user_id, ts AS event_timestamp, is_fraud, purchases_7d_asof
FROM carried
WHERE src = 'label';            -- keep only label rows; each now carries its point-in-time feature value
```

> The `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` frame is what guarantees no future leakage — only
> rows at or before the current event contribute. Some engines (DuckDB, others) offer a native `ASOF JOIN`
> that expresses this directly; **verify `ASOF JOIN`/`QUALIFY` support and window-frame semantics for your
> warehouse.** A plain `JOIN ON user_id` to a current value is the leakage bug this pattern exists to avoid.
