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
