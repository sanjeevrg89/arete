# Recsys/Ranking — Worked Examples

Two canonical artifacts to imitate:
1. A **two-tower retrieval + ranking funnel** (architecture + pseudo-config).
2. An **offline + online evaluation plan** for a ranking change.

Pseudo-config is illustrative — adapt field names to your framework/feature store. It is not a literal
schema. Numbers (dims, k, budgets) are *examples* to be re-derived for your system; do not treat them
as benchmarks.

---

## 1. Two-tower retrieval + ranking funnel

### 1.1 Architecture

```
                         ┌─────────────────────── OFFLINE / NEAR-REAL-TIME ──────────────────────┐
                         │  item features ──► item tower f_item ──► item embeddings ──► ANN index │
                         │  (content, category, author, stats)        (built nightly + NRT upsert)│
                         └───────────────────────────────────────────────────────────────────────┘
 request (user, context)
        │
        ▼
  ┌─────────────┐   user tower f_user(user+context) ──► user emb ─┐
  │  RETRIEVAL  │                                                  ├─► ANN top-K  ──┐
  │             │   + heuristic sources (trending, fresh, follows,─┘                │
  │             │     recently-viewed, lexical/BM25 for search)                     │
  └─────────────┘                                                                   ▼
                                                          blend + dedupe + tag source  (~1–2k candidates)
        │
        ▼
  ┌─────────────┐   lightweight distilled ranker (subset of features)
  │ PRE-RANKING │   score → keep top-N                                              (~200 candidates)
  └─────────────┘
        │
        ▼
  ┌─────────────┐   heavy multi-task ranker (DLRM/DCN-v2/sequence transformer)
  │   RANKING   │   heads: p_click, p_dwell, p_share, p_neg ; value = Σ wₖ·pₖ        (~200 scored)
  └─────────────┘
        │
        ▼
  ┌─────────────┐   diversity (MMR/category caps), dedupe, freshness boost,
  │ RE-RANKING  │   sequence-aware list model, exploration injection                (ordered list)
  └─────────────┘
        │
        ▼
  ┌─────────────┐   ads load / sponsored mix, compliance & safety blocklists,
  │   POLICY    │   pinning, frequency caps, per-creator fairness caps              (final page ~10–20)
  └─────────────┘
```

Latency split (illustrative — **measure and own yours**): retrieval ~3–6 ms, pre-ranking ~3–5 ms,
ranking ~10–30 ms, re-ranking + policy ~3–5 ms, within a single-digit-to-~50 ms p99 surface budget.

### 1.2 Pseudo-config — two-tower retrieval

```yaml
two_tower_retrieval:
  user_tower:
    inputs: [user_id_emb, user_history_seq, geo, device, time_of_day, context_surface]
    sequence_encoder: { type: transformer, layers: 2, pooling: attention }  # last-N behavior
    mlp: [512, 256, 128]
    output_dim: 128
    l2_normalize: true                # cosine-style; keep dot-product retrieval
  item_tower:
    inputs: [item_id_emb, content_emb, category, author_id_emb, age_bucket, pop_stats]
    mlp: [512, 256, 128]
    output_dim: 128                   # MUST match user_tower output_dim
    l2_normalize: true
    # item tower is user-independent  -> precompute ALL item embeddings offline

  loss:
    type: sampled_softmax
    in_batch_negatives: true
    logQ_correction: true             # subtract est. log P(sample item); else popular items suppressed
    hard_negatives:
      source: ann_near_positives + high_score_non_click
      ratio: 0.3                      # blend easy(in-batch) + hard; all-hard destabilizes
    temperature: 0.05                 # tune; affects training stability

  serving:
    index: { engine: see_rag-vector-databases, type: HNSW_or_IVFPQ, top_k: 1000 }
    embed_user: online (1 forward pass)
    embed_item: offline batch + near-real-time upsert for fresh items   # freshness pipeline
  monitoring:
    - stage_recall@1000_vs_ranker_top10   # is retrieval starving the ranker?
    - fresh_item_index_lag
    - train_serve_feature_skew
```

### 1.3 Pseudo-config — multi-task ranker

```yaml
ranker:
  arch: multi_task            # MMoE or PLE shared/task-specific experts; or DCN-v2 / DLRM bottom
  shared_bottom:
    embedding_tables:         # the dominant memory cost -> shard / hash
      item_id:   { dim: 64, hashing: true,  table_shards: 8 }
      user_id:   { dim: 64, hashing: true,  table_shards: 8 }
      author_id: { dim: 32 }
    dense_features: [item_ctr_7d, item_ctr_5m, user_cat_affinity, price, freshness, source_tags]
    cross_network: dcn_v2     # explicit bounded-degree feature crosses
    position_feature:         # debias: feed true position at train, neutral value at serve
      train: actual_slot
      serve: 1                # neutralize layout signal
  experts: { count: 6, type: mlp }            # MMoE/PLE gating per task
  task_heads:
    p_click:   { loss: logloss, calibrated: true }
    p_dwell:   { loss: logloss }
    p_share:   { loss: logloss }
    p_negative:{ loss: logloss }              # "see fewer like this" / report
  value_model:                                 # final ranking score
    formula: w_click*p_click + w_dwell*p_dwell + w_share*p_share - w_neg*p_negative
    weights_tuned_by: online_ab_vs_north_star  # NOT offline loss; this is a product decision
  serving:
    feature_source: feature_store (same defs as training)  # see data-engineering-feature-stores
    log_features_at_serving: true              # log-and-train -> no skew
```

Notes: the **value-model weights are tuned online**, against retention/north-star, not by offline loss.
`p_click` is **calibrated** (it may feed blending/ads). `position_feature` neutralization is what keeps
the model predicting relevance, not layout.

---

## 2. Offline + online evaluation plan for a ranking change

Scenario: replace the current ranker with a new multi-task DCN-v2 ranker. Plan to ship responsibly.

### Phase 0 — guard against skew before measuring anything
- Confirm the new model consumes **serving-logged features** (or shares exact transform code).
- Verify **point-in-time-correct** labels: features as-of impression time, labels attributed in the
  correct window; no time-travel leakage.
- Run a **skew check**: replay a day of serving logs, diff offline-computed vs serving-logged feature
  vectors; fail the launch if non-trivial drift. See `[[ml-observability-monitoring]]`.

### Phase 1 — offline (gate, do not decide)
- **Retrieval unchanged?** If yes, hold the candidate set fixed and evaluate ranking on the *same*
  candidates so you isolate the ranker. If retrieval also changed, measure **stage recall@k** too.
- Metrics on a held-out, time-split set (no random split — split by time to avoid leakage):
  - **GAUC** (per-user AUC averaged) and LogLoss for the click head — GAUC reflects within-user ordering,
    which is what ranking actually does.
  - **nDCG@k / MAP / MRR** against graded relevance (engagement-weighted).
  - **Calibration:** reliability curve + ECE for `p_click` (it feeds blending).
  - Per-segment slices: new vs returning users, head vs tail items, cold-start items, key locales —
    catch regressions hidden by the average.
- **Gate:** the new model must not regress key offline metrics or any segment badly. Passing offline is
  *necessary, not sufficient* — it only earns a ticket to the A/B.

### Phase 2 — counterfactual / off-policy (cheap pre-screen)
- Using logs **with logged propensities / exploration**, run **IPS** (and a **doubly-robust** estimator
  to cut variance) to estimate the new policy's reward vs the current one.
- High-variance result (small propensities) → treat as weak signal; still useful to kill obviously-bad
  candidates before spending live traffic. See `[[ml-evaluation-evals]]`.

### Phase 3 — online A/B (source of truth)
- **Randomization:** by user/cohort (not per-request) to avoid cross-arm contamination; sticky
  assignment. Compute required sample size for the target effect; run ≥1–2 full weekly cycles to cover
  day-of-week seasonality.
- **Primary / north-star:** retention or sessions or LTV (the metric that pays) — not raw CTR.
- **Engagement (secondary):** CTR, conversions/watch-time, dwell, completion.
- **Quality/satisfaction:** "see fewer like this," reported-content rate, survey scores — a CTR win that
  raises these is suspect.
- **Guardrails (a breach blocks launch regardless of engagement):** p99 serving latency, list
  **diversity**/category concentration, per-creator/seller **fairness**, abuse/policy-violation rate,
  error/timeout rate.
- **Long-term holdout:** keep a small holdback that *never* gets the new model for weeks, to detect
  long-term-value erosion that a 2-week test misses (the classic short-term-CTR-up / retention-down
  trap).

### Phase 4 — decision & ramp
- Ship only if: primary metric up (or flat with a clear secondary win) **and** no guardrail breach
  **and** offline-online directions are consistent (a sign-flip means a skew/position-bias bug — go
  investigate, don't ship). Then ramp 1% → 5% → 25% → 100% watching guardrails at each step; keep the
  long-term holdout running.
- Post-launch: monitor calibration drift, feature skew, and feedback-loop effects (popularity
  concentration, diversity) continuously, not just at launch.

### Common failure this plan catches
- **Great offline, flat online** → almost always train/serve skew or a position-bias artifact (Phase 0
  / `position_feature`), or the A/B measuring the wrong objective (north-star vs CTR).
- **CTR up, retention down** → caught by the long-term holdout and satisfaction guardrails, not by the
  primary engagement metric.
