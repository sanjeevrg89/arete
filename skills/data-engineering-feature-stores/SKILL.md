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
