# Examples — drift detection & LLM tracing/online-eval

Two canonical, imitate-this artifacts: (1) a tabular drift + data-quality report config (Evidently-style), and
(2) an OpenTelemetry-GenAI + Langfuse LLM tracing + online-eval instrumentation snippet. Both are
runnable-in-spirit; **verify the exact SDK surface against current docs** — these libraries change across
versions (it is 2026). Imports are real but feature names may have moved.

---

## 1. Tabular drift + data-quality report (Evidently-style)

The shape that matters: a **fixed reference** (training or a stable production window), the **current** window,
a **column mapping** (so prediction/target/categoricals are treated correctly), and **effect-size** drift tests
with explicit thresholds — not raw p-values. Run on a schedule; fail/alert on the test suite, not the eyeball.

```python
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestShareOfMissingValues,
    TestNumberOfColumnsWithMissingValues,
    TestColumnDrift,
    TestShareOfDriftedColumns,
)

# Reference = training data (catches decay vs the world the model learned).
# Current   = the last window of LIVE features actually SERVED to the model
#             (logging the served vector is what catches training-serving skew).
reference: pd.DataFrame = load_training_features()
current:   pd.DataFrame = load_served_features(window="last_24h")

column_mapping = ColumnMapping(
    target="label",                 # may be absent/delayed in production
    prediction="pred_score",        # always present -> prediction-drift monitoring
    numerical_features=["amount", "account_age_days", "txn_velocity_1h"],
    categorical_features=["country", "device_type", "merchant_category"],
)

# --- Diagnostic report: drift + data quality + target/prediction drift -------------
report = Report(metrics=[
    DataDriftPreset(
        # Threshold on EFFECT SIZE; pick the test per column type.
        num_stattest="wasserstein",  cat_stattest="jensenshannon",
        num_stattest_threshold=0.1,  cat_stattest_threshold=0.1,
        # alternative for binned/categorical: stattest="psi", threshold=0.2
    ),
    DataQualityPreset(),             # nulls, ranges, cardinality, new categories
    TargetDriftPreset(),             # prediction (and target, when labels exist) drift
])
report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
report.save_html("drift_report.html")   # human drilldown / data-docs

# --- Gate: a TestSuite that PASSES/FAILS, wired to alerting/CI ----------------------
suite = TestSuite(tests=[
    # Data quality (layer 1) — usually the real cause of "model broke".
    TestNumberOfColumnsWithMissingValues(eq=0),
    TestShareOfMissingValues(column_name="account_age_days", lte=0.01),
    # Drift (layer 2) — alert only when a MEANINGFUL share of features moved.
    TestShareOfDriftedColumns(lt=0.30),
    # Watch the high-importance features individually (not all 500).
    TestColumnDrift(column_name="txn_velocity_1h", stattest="wasserstein", lt=0.1),
    TestColumnDrift(column_name="merchant_category", stattest="psi", lt=0.2),
])
suite.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

result = suite.as_dict()
if result["summary"]["failed_tests"] > 0:
    # DON'T auto-retrain on drift alone. Drift is a hypothesis:
    # open an incident / page on-call, then CONFIRM against a performance or
    # proxy metric (backfilled labels, override rate) before triggering a retrain.
    raise_alert(result)
```

**Why it's built this way**
- **Reference = training, current = *served* features** — comparing the served vector to training is what
  surfaces **training-serving skew**, the #1 silent failure. Log the exact vector the model scored.
- **`prediction` mapped** ⇒ you get prediction drift with zero labels — your earliest output-side signal.
- **Effect-size tests with thresholds** (Wasserstein/JS/PSI), not p-values — at high volume a KS/chi-square
  p-value is always "significant."
- **`TestShareOfDriftedColumns` + a few per-feature tests on important features** instead of one alert per
  column — avoids the 500-features alert-fatigue trap.
- The diagnostic **Report** is for humans; the **TestSuite** is the machine gate that pages.

---

## 2. LLM tracing + online eval (OpenTelemetry GenAI + Langfuse)

Each LLM/retrieval/tool step is a **span** in a trace, annotated with OTel `gen_ai.*` attributes (model, token
counts, cost) plus latency split into **TTFT/ITL**. A **sampled** subset of live traffic gets an **online
LLM-as-judge / groundedness** eval whose score is attached to the trace as a monitored metric.

> Verify attribute names against the current OTel GenAI semantic-conventions spec — they are still evolving.

```python
import time, random
from opentelemetry import trace
from langfuse import Langfuse           # OSS LLM observability; Phoenix/LangSmith are alternatives

tracer = trace.get_tracer("rag-service")
lf = Langfuse()

def answer(question: str, user_id: str) -> str:
    # One trace per user request; carries session/user id to follow a conversation.
    with tracer.start_as_current_span("rag.request") as root:
        root.set_attribute("gen_ai.system", "openai")
        root.set_attribute("user.id", user_id)

        # --- retrieval span: log chunks + scores (groundedness needs the context) ---
        with tracer.start_as_current_span("retrieve") as rspan:
            chunks = vector_search(question, k=5)        # see [[rag-vector-databases]]
            rspan.set_attribute("retrieval.k", len(chunks))
            rspan.set_attribute("retrieval.top_score", chunks[0].score)

        # --- LLM span: model, tokens, COST, latency split TTFT / ITL ----------------
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
            t0 = time.perf_counter(); first_token_t = None; out = []
            for tok in stream_completion(question, chunks):
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                out.append(tok)
            t_end = time.perf_counter()
            answer_text = "".join(out)

            usage = last_usage()  # provider-reported token usage
            span.set_attribute("gen_ai.usage.input_tokens",  usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            span.set_attribute("gen_ai.cost.usd",            estimate_cost(usage))
            span.set_attribute("gen_ai.latency.ttft_ms", (first_token_t - t0) * 1000)  # streaming UX
            span.set_attribute("gen_ai.latency.e2e_ms",  (t_end - t0) * 1000)
            if usage.completion_tokens > 1:                                            # ITL / TPOT
                span.set_attribute("gen_ai.latency.itl_ms",
                                   (t_end - first_token_t) * 1000 / (usage.completion_tokens - 1))

        # --- guardrail signal: monitor the HIT RATE, in and out (see [[ai-security-on-gke]]) ---
        flagged = output_guardrail(answer_text)
        root.set_attribute("guardrail.flagged", flagged)

        # --- prompt/response logging: required for debug, eval sets, regression tests ---
        gen = lf.trace(name="rag.request", user_id=user_id, input=question,
                       output=answer_text)

        # --- ONLINE eval on a SAMPLE (cost) — stratify so risky cases are represented ---
        if flagged or random.random() < 0.05:
            # Groundedness = is the answer supported by retrieved context? The core RAG
            # hallucination signal. LLM-as-judge methodology lives in [[ml-evaluation-evals]].
            scores = run_online_eval(question, answer_text, context=chunks,
                                     judge_model="gpt-4o", judge_version="2026-xx")  # PIN the judge
            gen.score(name="groundedness", value=scores["groundedness"])
            gen.score(name="answer_relevance", value=scores["relevance"])
            gen.score(name="toxicity", value=scores["toxicity"])

        return answer_text
```

**Why it's built this way**
- **Tracing is the backbone** — for RAG/agents the value is the *whole decision path* (which chunks, which tool,
  where cost/latency went), not a single number.
- **Tokens, cost, TTFT/ITL on the span** — cost-per-request and TTFT are the fastest-moving, easiest-to-blow-up
  dimensions; watch for prompt bloat and runaway agent loops.
- **Online eval is sampled and stratified** — 100% judging is too expensive, but always eval the **flagged/risky**
  slice plus a random sample so rare failures show up.
- **Pin the judge model+version** — LLM-as-judge is itself a drifting, biased model; treat its score as a
  *monitored metric*, recalibrate it against human labels periodically.
- **Guardrail hit rate is a monitored signal** both ways: a spike (attack/drift) and a drop to zero (guardrail
  silently broke) are both alerts.
- **Log prompt+response** — you can't debug, build eval sets, or create regression tests without them (mind
  PII/retention).

> Pair these model-quality signals with infra metrics (GPU util, queue depth, autoscaling) from
> `[[serving-frameworks]]` / `[[gke-inference-gateway]]` / `[[autoscaling-kubernetes]]` — they answer different
> questions, and you need both to root-cause an LLM latency or quality regression.
