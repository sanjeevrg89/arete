# ML / LLM Observability & Monitoring — Field Guide

The authoritative reference for this skill. Read it fully, then apply it. Scope: knowing when a **deployed**
model is degrading. Build/train/registry/CI live in `[[mlops-lifecycle]]`; feature pipelines in
`[[data-engineering-feature-stores]]`; eval methodology in `[[ml-evaluation-evals]]`; infra metrics (GPU, queue
depth, autoscaling) in `[[gke-master]]` / `[[autoscaling-kubernetes]]` / `[[aiml-on-kubernetes]]`.

---

## 1. Mental model — why ML monitoring is not software monitoring

A web service fails *loudly*: it throws, returns 5xx, latency spikes, a probe goes red. You alert on the
symptom. A model fails *silently*: it keeps returning a well-formed `0.87` for every request, on time, with no
error — the number is just **wrong**, and it gets wronger as the world drifts away from the data it was trained
on. There is no exception to catch.

So ML observability monitors a different object. Not "is the service up?" but **"are the predictions still
trustworthy?"** That requires watching the *statistics* of three things over time:

- **Inputs** — the feature distributions the model receives.
- **Outputs** — the prediction distribution the model emits.
- **Outcomes** — what actually happened (ground truth), and the performance metric computed against it.

Software SLOs (availability, latency, error rate) are **necessary but not sufficient**. You still need them
(see infra skills) — a model that's down is also broken — but a green dashboard tells you nothing about whether
the model is right.

### The ground-truth lag / feedback-loop problem

The metric you actually care about — accuracy, AUC, RMSE, conversion lift — requires labels, and **labels are
delayed, partial, or absent**:

- A fraud model: the chargeback that confirms "fraud" lands 30–90 days later.
- A churn model: you learn the true label at the end of the subscription period.
- A loan-default model: ground truth arrives over the *loan's lifetime*.
- A recommendation: you get a weak, biased signal (clicks) immediately, the real one (retention) much later — and
  it's confounded by the recommendation itself (**feedback loop**: the model shapes the data it's later judged on).

This lag is the organizing constraint of the whole discipline. You cannot wait weeks to discover a regression. So
you build a **ladder of leading-to-lagging signals** and use the leading ones as early warnings for the metric you
can't yet measure.

---

## 2. The four monitoring layers (leading → lagging)

Design every monitoring system as these four layers. Earlier layers are *leading* (available now, cheap) and
*proxies*; the last is *lagging* (the truth, but late).

| Layer | Question | Signal availability | Examples |
|---|---|---|---|
| 1. Data quality | Is the input even valid? | Immediate | schema, nulls, ranges, cardinality, freshness |
| 2. Drift / **skew** | Has the input world changed vs training? | Immediate | PSI, KS, embedding drift, train-vs-serve skew |
| 3. Prediction drift | Has the output distribution shifted? | Immediate | mean score shift, class-balance shift |
| 4. Performance | Is the model actually right? | **Delayed / partial** | accuracy, AUC, RMSE, calibration, business KPI |

The trap is monitoring only layer 4 (because it's the metric leadership asks for) and discovering a problem two
weeks after it started. The leverage is in layers 1–3, which you have *today*. Layer 4, when labels arrive,
*validates and recalibrates* your leading thresholds.

---

## 3. Drift & decay

### 3.1 Vocabulary — be precise, these are not synonyms

- **Data drift / covariate shift:** P(X) changes; P(y|X) unchanged. The input distribution moves (new user
  cohort, new device, seasonality). The model may still be valid — but it's now extrapolating.
- **Concept drift:** P(y|X) changes — the *relationship* the model learned is now wrong. This is the dangerous
  one: accuracy can collapse with **zero** input drift (e.g. spammers change tactics, prices in a recession).
  - *Sudden* (regime change), *gradual* (slow erosion), *incremental*, and *recurring/seasonal* concept drift
    each need different windows.
- **Label drift / prior probability shift:** P(y) changes — the class balance shifts (fraud rate triples).
- **Prediction drift:** the model's *output* distribution shifts. A leading proxy you can always compute — if the
  output moved and inputs didn't, suspect a pipeline/feature bug; if inputs moved too, suspect real drift.
- **Training-serving skew:** training-time features ≠ serving-time features for the *same* entity. Not drift over
  time — a *static mismatch* between two code paths. The #1 silent production failure (see §3.5).
- **Feature-attribution drift:** the *importance ranking* of features (e.g. mean |SHAP|) shifts, even if marginal
  distributions look stable. Catches subtler concept drift and broken features that a marginal-distribution test
  misses. Vertex AI Model Monitoring offers this as feature-attribution monitoring (verify current docs).

**Drift is a hypothesis, not a verdict.** Drift means "the world looks different; performance *may* have dropped."
It is not proof of decay. Always confirm against a performance or business metric before acting — alerting on
drift alone produces noise and erodes trust.

### 3.2 Detection methods — pick by data type, threshold on effect size

| Method | Use for | Notes / pitfalls |
|---|---|---|
| **PSI** (Population Stability Index) | binned numeric / categorical | Industry default. Rules of thumb: <0.1 stable, 0.1–0.25 moderate, >0.25 significant. Sensitive to binning; undefined on empty bins (add ε). |
| **KL divergence** | distribution shift | Asymmetric, unbounded, blows up when ref has 0 where current has mass. Rarely thresholded directly. |
| **JS divergence** | distribution shift | Symmetric, bounded [0,1] (log2), well-behaved version of KL. Good general categorical/binned choice. |
| **KS test** (Kolmogorov–Smirnov) | continuous, univariate | Compares CDFs. **p-value is sample-size sensitive** — at high N everything is "significant." Use the KS *statistic* (effect size), not p. |
| **Wasserstein / Earth Mover's** | continuous | Measures how *far* mass moved, not just "different." More robust and interpretable than KS for magnitude. |
| **Chi-square** | categorical | Same N-sensitivity caveat as KS. |
| **Embedding drift** | text / images / high-dim | Embed inputs, compare distributions: centroid/MMD distance, or a **domain classifier** (train a model to tell reference vs current — high AUC ⇒ drift). The practical way to drift-monitor unstructured data and LLM inputs/outputs. |

### 3.3 Pitfalls that make drift monitoring lie to you

- **Sample-size sensitivity / p-value abuse.** Statistical-significance tests detect *any* difference at scale.
  At millions of requests, KS/chi-square always reject. **Threshold on effect-size metrics** (PSI, JS, Wasserstein,
  classifier AUC), and pick the threshold from a *backtest* on a stable period, not a textbook constant.
- **Multiple comparisons.** 500 features × a daily test = a flood of false alarms. Apply correction (Bonferroni/BH),
  rank by importance, or monitor a single multivariate signal — don't page on every feature.
- **Reference-window choice.** A drifting reference (rolling 7d) hides slow drift; a fixed reference (training set)
  catches it but flags benign seasonality. Often you want *both*: vs-training (decay risk) and vs-recent (anomaly).
- **Seasonality.** Weekday/weekend, holidays, campaigns. Compare like-to-like windows or deseasonalize first.
- **Binning artifacts.** PSI depends entirely on bin edges. Fix bins from the reference and reuse them; handle new
  categories and empty bins explicitly.
- **Mixing missingness into the distribution.** A spike in nulls can masquerade as drift (or hide it). Monitor
  missingness as its own signal (§4).
- **Aggregate-only drift.** Global stability can hide a drifting *segment*. Slice (§5.3).

### 3.4 Drift on the prediction side

Always monitor **prediction drift** — it needs no labels and is your earliest output-side signal. For
classifiers: mean predicted probability, class-balance, and score-histogram drift. For regressors: output
distribution shift. A jump in prediction drift with stable inputs almost always means a **broken feature or
pipeline**, not the world changing — check skew and data quality first.

### 3.5 Training-serving skew — the highest-ROI catch

Skew is when the model sees a different feature value at serving time than it would have at training time, for the
same input. Causes:

- Different code computing the feature in the training pipeline vs the serving path (the classic).
- A **stale or missing feature** at serving (online store didn't have it; defaulted to 0/NaN/imputed mean).
- Unit / encoding / scaling mismatch (cents vs dollars, different category map, different tokenizer).
- **Time-travel / leakage:** training used a value that isn't actually available at prediction time.

Detection: log the **exact feature vector served to the model** and compare its distribution to the training
distribution (TFDV's `validate_statistics` / skew comparators, or Vertex AI Model Monitoring's training-serving
skew mode — verify current docs). The durable fix is to **compute features once and share** training and serving
through a feature store with point-in-time-correct reads — see `[[data-engineering-feature-stores]]`.

---

## 4. Data-quality monitoring (layer 1 — do this first)

Most "model is broken" pages are actually **data** broken. Cheapest, most actionable layer. Validate every batch
or a sample of every request:

- **Schema:** expected columns present, correct types, no surprise columns. Pin a schema and diff against it.
- **Range / domain:** numeric within `[min,max]`; categoricals in the known set (flag **new categories**).
- **Null / missing rate:** per-feature null fraction vs reference; alert on spikes. Decide *and monitor* the
  missing-feature policy (impute, default, reject) — a feature defaulting silently to 0 is a stealth outage.
- **Cardinality:** unique-count drift catches IDs leaking in, hashing changes, exploding categoricals.
- **Freshness / staleness:** timestamp of the feature vs now; an upstream pipeline that stopped at 02:00 serves
  yesterday's features all day with no error. Monitor data age and row counts/volume.
- **Uniqueness / duplicates / referential integrity** where relevant.

**Pipeline validation tooling:** **Great Expectations** (expectation suites, checkpoints, data docs) for tabular
batch validation; **TensorFlow Data Validation (TFDV)** for schema inference, statistics, and skew/drift
comparators in TFX-style pipelines; **Evidently** for combined data-quality + drift reports/test-suites; **Pandera**
for dataframe schema contracts; **Deequ/PyDeequ** for Spark-scale checks. Wire validation as a **gate**: fail the
batch / hold the prediction / page, don't just log.

---

## 5. Model-performance monitoring (layer 4 — the lagging truth)

### 5.1 Living with delayed and partial labels

- **Delayed-label join/backfill:** persist every prediction with a key and timestamp; when ground truth arrives
  (chargeback, label event, manual review), join it back and compute metrics on the now-complete window. Your
  performance dashboard is therefore always "as of labels available," lagging real time.
- **Proxy metrics:** when true labels lag, monitor correlated signals you *do* get fast — clicks/dwell for recsys,
  user edits/thumbs for assistants, complaint/refund rate, downstream override rate. Validate the proxy correlates
  with the real metric before trusting it; proxies are biased (feedback loops) — treat as directional.
- **Sampled human labeling:** label a random (and a *targeted*, e.g. low-confidence) sample continuously to get an
  unbiased-ish accuracy estimate without waiting for organic labels.
- **Confidence / uncertainty:** drops in mean confidence or rises in near-decision-boundary rate are leading
  proxies for trouble.

### 5.2 Calibration

Track **calibration**, not just a point metric: when the model says 0.8, does the event happen ~80% of the time?
Use reliability diagrams and ECE (Expected Calibration Error). Models drift *out of calibration* before — and more
detectably than — they lose ranking power (AUC). Critical anywhere the probability is used directly (pricing, bidding,
risk thresholds).

### 5.3 Slice / segment analysis and fairness

**Aggregate metrics lie.** A model can hold 92% overall while collapsing on a 5% segment that just doubled in
volume. Always monitor performance **by slice**: geography, device, language, new vs returning, customer tier,
product category, time-of-day. Auto-surface the **worst-performing slices** rather than eyeballing a global number.

**Fairness monitoring** is slicing on protected/sensitive groups: track performance and error-rate parity
(e.g. FPR/FNR gaps, demographic parity, equalized-odds-style gaps) across groups over time — fairness regresses
with drift even when overall accuracy holds. See `[[responsible-ai-governance]]` for the governance framing.

---

## 6. LLM observability

LLMs break the classic playbook: outputs are free text, there's rarely a label, "accuracy" isn't one number,
apps are multi-step (RAG/agents/tools), and cost+latency are first-class. LLM observability = **tracing +
online evaluation + cost/latency + safety signals**.

### 6.1 Tracing (the backbone)

Instrument with **distributed tracing** where each LLM/retrieval/tool/agent step is a **span** in a trace. Use the
**OpenTelemetry GenAI semantic conventions** (`gen_ai.*` attributes — system, request/response model, input/output
token counts, etc.; conventions are evolving — **verify current names against the OTel spec**). Capture per span:

- model + parameters (temperature, max tokens), the **prompt/input and response/output**,
- **token counts** (prompt/completion) and **cost**,
- **latency**, split into **TTFT** (time-to-first-token) and **ITL/TPOT** (inter-token latency) for streaming,
- tool name + args + result for tool calls; retrieved chunks + scores for retrieval steps,
- status/error, and a session/user/trace id to follow a whole conversation.

Tooling: **Langfuse** (open-source, self-hostable), **Arize Phoenix** (OSS, OTel-native), **LangSmith** (LangChain),
plus Arize/WhyLabs/Fiddler on the platform side. Prefer OTel-native instrumentation so you're not locked in. Trace
**chains and agents** end-to-end — for agents the value is seeing the *whole decision path* (which tool, why a loop,
where latency/cost went). App/agent framework details are in `[[llm-app-agent-frameworks]]`.

### 6.2 Online evaluation in production

Offline evals gate releases (`[[ml-evaluation-evals]]`); **online evals** watch live traffic. On a sample of real
requests, run:

- **LLM-as-judge** scoring on dimensions you care about (helpfulness, correctness, tone, instruction-following).
- **Reference-free quality checks:** groundedness/faithfulness (is the answer supported by the retrieved context?
  — the core RAG hallucination signal, see `[[rag-vector-databases]]`), relevance, answer completeness,
  toxicity/PII.
- **Heuristics:** regex/format/schema-valid (did it return valid JSON?), refusal rate, response length, language.

Caveats: LLM-as-judge is **itself a model that drifts and is biased** (position, verbosity, self-preference) — pin
the judge model/version, calibrate it against human labels periodically, and treat its scores as a *monitored
metric*, not ground truth. Sample (don't judge 100% — cost) but **stratify** so rare/risky cases are represented.

### 6.3 Cost, latency, throughput, and safety signals

- **Cost/token:** tokens and \$ per request, per route, per user/tenant, per model — the fastest-moving and
  easiest-to-blow-up dimension. Alert on cost-per-request and total spend; watch for prompt bloat and runaway agent
  loops.
- **Latency:** TTFT and end-to-end p50/p95/p99; for streaming, ITL. Tie to serving infra (`[[serving-frameworks]]`,
  `[[gke-inference-gateway]]`, `[[inference-optimization]]`).
- **Quality/hallucination signals:** groundedness scores, contradiction/uncertainty cues, citation-coverage, user
  thumbs/edits/regenerations as implicit feedback.
- **Guardrail hit rates:** rate of prompt-injection/jailbreak/PII/toxicity guardrail triggers (in and out). A *change*
  in hit rate is signal — both a spike (attack, drift) and a drop to zero (guardrail silently broke). See
  `[[ai-security-on-gke]]`.
- **Prompt/response logging:** log inputs and outputs (with PII handling) — you cannot debug, eval, or build a
  regression set without them. Mind privacy/retention.

---

## 7. Closing the loop — alerting, triggers, incident response

Dashboards nobody acts on are theater. Every monitor must have an **owner, a threshold, a runbook, and an action.**

- **Thresholds with hysteresis.** Backtest thresholds on historical stable + incident periods. Require persistence
  (e.g. N consecutive windows or a sustained breach) to fire — single-window blips are noise. Use effect-size
  metrics, not raw p-values.
- **Retraining triggers.** Define what fires a retrain: sustained performance drop, drift past threshold *confirmed*
  by a performance/proxy regression, or scheduled cadence. The retrain/continuous-training pipeline itself lives in
  `[[mlops-lifecycle]]` — monitoring's job is to *trigger and validate*, not to run training.
- **Incident response for model regressions.** Treat a model regression like an outage: detect → triage (data?
  skew? concept drift? bad deploy?) → **mitigate fast** (roll back to the last-good model/prompt, shadow/canary the
  fix) → root-cause → backfill labels to confirm. Keep the last-known-good model deployable for instant rollback.
- **Dashboards** layered to the four layers (§2): a top-line health view (perf + business KPI + cost), drilldowns to
  drift/skew/data-quality, and per-slice/per-segment views. The on-call should answer "is the model OK, and if not
  which layer broke?" in under a minute.
- **Alert fatigue is the failure mode.** Too many alerts ⇒ all ignored. Consolidate per-feature noise into
  importance-weighted or multivariate signals; route by severity; review and prune alerts regularly.

---

## 8. Tooling landscape (verify current capabilities against vendor docs)

| Tool | Niche |
|---|---|
| **Evidently** (OSS + cloud) | Tabular + text/LLM: drift, data-quality, performance reports & test-suites; good default for self-hosted tabular monitoring. |
| **Arize** / **Phoenix** (OSS) | ML + LLM observability; Phoenix is OTel-native tracing/eval, OSS and self-hostable. |
| **WhyLabs** / **whylogs** | Lightweight data-logging *profiles* (privacy-preserving aggregates) for drift/data-quality at scale. |
| **Fiddler** | Model monitoring + explainability + LLM observability. |
| **Langfuse** / **LangSmith** | LLM tracing, prompt mgmt, online/offline eval (Langfuse OSS; LangSmith LangChain-centric). |
| **Vertex AI Model Monitoring** | Managed skew/drift (and feature-attribution) monitoring for models on Vertex; integrates with feature store/registry. Verify current modes/limits. |
| **Great Expectations / TFDV / Pandera / Deequ** | Data-quality validation (§4). |
| **Prometheus / Grafana / OpenTelemetry** | Infra + service metrics (QPS, latency, GPU util, errors) and the substrate for custom ML metrics and GenAI traces. Pair an ML-monitoring tool *with* these — they answer different questions. |

Heuristics: don't buy a platform before you have layers 1–3 instrumented; **emit OTel-native** signals to avoid
lock-in; keep raw prediction+feature logs (you'll need them for backfill, eval sets, and root-cause) regardless of
which dashboard you use.

---

## 9. Anti-patterns (the ways monitoring programs fail)

- **Accuracy-only monitoring.** Waiting on the one lagging metric ⇒ you learn about regressions weeks late and have
  no leading signal. Build layers 1–3.
- **No training-serving skew detection.** The single most common silent production failure goes uncaught.
- **No slicing.** A green aggregate hiding a dead segment / fairness regression.
- **No ground-truth pipeline.** Predictions logged with no path to ever attach the label ⇒ you can *never* compute
  real performance or build a training set. Design the label join on day one.
- **Drift alerts with no performance tie-in.** Paging on input drift that didn't actually hurt the model ⇒ noise ⇒
  ignored alerts.
- **Per-feature alert spam.** 500 features × a test = alert fatigue. Importance-weight / consolidate / multivariate.
- **p-value thresholding at scale.** Everything is "significant"; you drown. Threshold effect size.
- **Static reference forever (or a drifting one only).** Use the right reference(s) for the question (§3.3).
- **LLM "vibes" monitoring.** No tracing, no online eval, no cost tracking — just spot-checks. You'll miss quality
  regressions, hallucination spikes, and cost blowups until a user or a finance report tells you.
- **Monitoring you never act on.** No owner, no runbook, no trigger ⇒ it's decoration.

---

## Rationalizations & rebuttals

The excuses for shipping a model with no real monitoring — and why each is wrong.

- *"Accuracy looks fine on the eval set, so we're good."* — The eval set is a frozen snapshot of the past;
  production is drifting away from it right now. Offline accuracy says nothing about live performance under
  drift, skew, or a broken serving feature. Monitor layers 1–3 on *live* traffic (§2).
- *"No errors, latency's green, dashboards are healthy."* — Software SLOs are necessary but not sufficient (§1).
  A model fails silently: it returns a well-formed, on-time, *wrong* number with no exception to catch. Green
  infra tells you nothing about whether predictions are right.
- *"We'll skip drift detection — drift doesn't mean the model is wrong."* — Correct that drift ≠ decay (§3.1),
  but drift is your only *immediate* leading signal while real performance is weeks late. Don't skip it; tie it
  to a performance/proxy metric so you alert on *confirmed* drift, not raw input shift (§3.4, §7).
- *"Ground truth is delayed weeks, so performance monitoring is pointless."* — The label lag is the reason to
  build the leading-to-lagging ladder, not skip it. Persist predictions with keys, backfill-join labels when
  they land, and use proxies / sampled human labels / confidence as early warnings (§5.1). Designing the label
  join on day one is what lets you *ever* compute real performance.
- *"One global accuracy number is enough — no slicing needed."* — Aggregates lie. A model can hold 92% overall
  while collapsing on a 5% segment that just doubled in volume, and fairness regresses with drift even when
  overall accuracy holds (§5.3). Auto-surface worst slices instead of eyeballing one number.
- *"It's an LLM, you can't really measure it — we'll spot-check."* — Spot-checks miss quality regressions,
  hallucination spikes, and cost blowups until a user or a finance report finds them. Tracing + online eval +
  cost/latency + safety signals are all measurable and required (§6).
- *"Train and serve share the same features, so skew can't happen."* — Skew is a *static mismatch between two
  code paths* and the #1 silent production failure (§3.5): different feature code, a stale/missing online value
  defaulting to 0, unit/encoding/tokenizer mismatch. Log the exact served vector and compare to training.

## Red flags

Stop and reconsider if any of these are true of your monitoring:

- **Accuracy/AUC is the only thing monitored** — no leading layers, so you learn of regressions weeks late (§2).
- **No training-serving skew detection** — the most common silent failure is uncaught; nobody compares the
  exact served feature vector to the training distribution (§3.5).
- **No drift/skew monitoring on inputs or predictions** — no immediate signal exists between deploy and the
  next label arriving (§3).
- **No ground-truth pipeline** — predictions logged with no key/timestamp or no path to attach labels later,
  so real performance can *never* be computed and no fresh training set can be built (§5.1).
- **Drift alerts with no performance tie-in** — paging on input drift that didn't hurt the model; or
  per-feature alert spam (500 features × a daily test) ⇒ alert fatigue ⇒ everything ignored (§3.3, §7).
- **p-values thresholded at scale** — KS/chi-square always "significant" at millions of rows; you're drowning
  in false alarms instead of thresholding effect size (§3.3).
- **No slicing / no fairness tracking** — only a global metric, so a dead segment or a fairness regression
  hides behind a green aggregate (§5.3).
- **LLM with no tracing or online eval** — free-text outputs, multi-step agents, and cost are unobserved;
  "vibes" monitoring only (§6).
- **Monitors with no owner, threshold, runbook, or trigger** — dashboards nobody acts on are decoration (§7).

## Verification gate (definition of done)

Monitoring for a deployed model is "done" only when all of the following are wired and firing:

- [ ] **Data quality (layer 1)** monitored on every batch/sampled request: schema, ranges/new categories, null
      rate, cardinality, freshness/volume — wired as a *gate* that fails/holds, not just logs (§4).
- [ ] **Drift detected with effect-size metrics** (PSI/JS/Wasserstein/classifier-AUC, *not* raw p-values) on
      inputs and on **prediction output**, with thresholds backtested on a stable period and multiple-comparison
      noise consolidated (§3.2, §3.3).
- [ ] **Training-serving skew** detected: the exact served feature vector is logged and compared to the training
      distribution; missing-feature policy is monitored (§3.5).
- [ ] **Performance under label lag** tracked: predictions persisted with key+timestamp, label backfill-join
      computes metrics "as of labels available," plus proxies / sampled human labels / confidence and
      **calibration (ECE)** as leading signals (§5.1–5.2).
- [ ] **Slice & fairness monitoring** live: per-segment performance with worst-slice auto-surfacing, and
      error-rate parity tracked across protected groups over time (§5.3).
- [ ] **LLM apps:** OTel-native **tracing** (spans for LLM/retrieval/tool steps with model, tokens, cost,
      TTFT/ITL, I/O) + **online eval** on a stratified sample (LLM-as-judge with a pinned judge, groundedness,
      heuristics) + cost/latency/guardrail-hit-rate monitoring (§6).
- [ ] **Alerting closed loop:** every monitor has an owner, a thresholded alert with hysteresis/persistence, a
      runbook, and **retrain/rollback triggers** wired (sustained drop or confirmed drift ⇒ retrain;
      last-known-good model kept deployable for instant rollback) (§7).

## 10. Version awareness

This ecosystem moves fast (it is 2026). **Verify against current docs** before relying on specifics: the
OpenTelemetry **GenAI semantic conventions** (attribute names and stability are still evolving); **Vertex AI Model
Monitoring** modes, supported model types, and limits; **Evidently**, **Langfuse**, **Phoenix**, **Arize**,
**WhyLabs**, **Fiddler** APIs and feature sets; and **Great Expectations** / **TFDV** APIs (GX's API changed
substantially across major versions). The *concepts* here (the four layers, drift vocabulary, effect-size
thresholding, ground-truth lag, tracing+online-eval for LLMs) are durable; the **API surfaces are not** — don't
hardcode from memory.

---

## 11. Canonical references (real URLs; verify currency)

- ML-Ops principles & monitoring: https://ml-ops.org/
- Evidently — drift detection & metrics: https://docs.evidentlyai.com/
- Evidently — LLM evaluation/observability: https://www.evidentlyai.com/llm-guide
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Vertex AI Model Monitoring: https://cloud.google.com/vertex-ai/docs/model-monitoring/overview
- TensorFlow Data Validation (TFDV): https://www.tensorflow.org/tfx/guide/tfdv
- Great Expectations: https://docs.greatexpectations.io/
- Langfuse (LLM observability): https://langfuse.com/docs
- Arize Phoenix (OSS tracing/eval): https://docs.arize.com/phoenix
- whylogs / WhyLabs: https://docs.whylabs.ai/
- Google "Rules of ML" (best-practice background): https://developers.google.com/machine-learning/guides/rules-of-ml
