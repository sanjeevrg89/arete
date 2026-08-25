---
name: ml-evaluation-evals
description: World-class evaluation of ML & LLM systems — the discipline that separates a demo from production ("evals are the moat"). Use whenever you need to measure model/system quality: choosing or computing metrics, building an eval set, or grading outputs. Covers classical ML metrics (precision/recall/F1, ROC-AUC vs PR-AUC, calibration/ECE/Brier, regression, ranking nDCG/MRR/MAP, slice & fairness), LLM/generative evaluation (benchmarks like MMLU/GPQA/HumanEval/MT-Bench and their contamination limits, BLEU/ROUGE/BERTScore weakness), LLM-as-a-judge (pairwise vs pointwise, position/verbosity/self-preference bias mitigation, human calibration), RAG eval (faithfulness/groundedness, context precision/recall, RAGAS), agent eval (task success, trajectory/tool-use correctness), online A/B testing & interleaving & guardrails, and eval-ops (lm-eval-harness, OpenAI Evals, Inspect, Promptfoo, DeepEval, Langfuse, evals in CI, golden sets, contamination/leakage).
---

# ML & LLM Evaluation (Evals)

Apply the judgment of an engineer who has shipped ML and LLM systems to production for years and knows
that **the model is a commodity — the eval set and the harness around it are the moat.** You cannot
improve, ship safely, or even know if a change helped without a trustworthy evaluation. Replace
vibes-only judgment with measurable, representative, bias-controlled, regression-gated evals.

## How to use this skill

1. **Read `ml-evaluation-evals-guide.md`** in this directory — the full reference (eval mindset,
   classical metrics, LLM/generative eval, LLM-as-a-judge, RAG & agent eval, online experimentation,
   eval-ops). Apply it to the task at hand.
2. For concrete artifacts to imitate — an LLM-as-a-judge rubric+prompt with bias mitigations, a
   RAGAS-style metric config, and an eval-in-CI gate — read **`examples.md`**.
3. Match the surrounding stack's conventions and tooling; apply the correctness rules (representative
   set, no leakage/contamination, judge bias controls, human calibration) regardless.
4. The ecosystem moves fast (2026): **never quote a benchmark score from memory**, and verify harness
   APIs, judge models, and leaderboards against current docs before relying on them.

## The essentials (full rationale in `ml-evaluation-evals-guide.md`)

- **You can't improve what you can't measure.** Build the eval *before* you optimize; treat the eval
  set, scoring functions, and judge prompts as versioned, reviewed code. Eval-driven development:
  every change is a hypothesis tested against the eval, not vibes.
- **The eval set is the asset.** Make it representative of production (right distribution), sliced by
  the segments you care about, frozen as a golden set, and seeded with your hardest/failed cases.
- **Avoid contamination & leakage.** Group/time-split (not row-split) to stop train/eval leakage;
  assume any public benchmark older than the model's cutoff is partially contaminated — use private,
  held-out eval sets.
- **Classical metrics, chosen deliberately.** Accuracy lies under imbalance; use **PR-AUC** for rare
  positives (not ROC-AUC); report the confusion matrix at the deployed threshold; check **calibration**
  (ECE/Brier) when probabilities drive decisions; use **nDCG/MRR/MAP** for ranking ([[recsys-ranking]]).
- **Always slice.** An aggregate metric hides per-segment regressions; disaggregate by segment, and do
  fairness as slice-based evaluation across groups ([[responsible-ai-governance]]).
- **Reference metrics are weak for open generation.** BLEU/ROUGE/BERTScore miss factuality and valid
  paraphrase; prefer deterministic checks (schema, code execution, regex) wherever the property is
  checkable, and LLM-as-a-judge for the genuinely subjective parts.
- **LLM-as-a-judge: prefer pairwise over fine-grained absolute scores; control biases** (position →
  swap order; verbosity → ignore length; self-preference → different judge family) and **calibrate
  against human labels** vs the human–human agreement ceiling before trusting it at scale.
- **Evaluate RAG's two surfaces separately:** retrieval (context precision/recall) vs generation
  (faithfulness/groundedness, answer relevance) — RAGAS vocabulary ([[rag-vector-databases]]).
- **Agents are trajectories:** measure task success *and* trajectory/tool-use correctness; per-step
  errors compound, so run each case N times and report success rate + variance ([[llm-app-agent-frameworks]]).
- **Online proves it.** Offline gates what's allowed to ship; A/B tests, interleaving, guardrail
  metrics, and feedback signals decide what ships. Mind the offline-online gap ([[ml-observability-monitoring]]).
- **Wire evals into CI.** Golden/regression suite on every PR; gate on relative (pairwise) comparison;
  handle non-determinism with N runs + tolerance; keep a human-in-the-loop audit ([[mlops-lifecycle]]).

## Related skills

- `[[ml-system-design]]` — where evaluation fits in the end-to-end design of an ML system.
- `[[mlops-lifecycle]]` — eval gates in CI/CD, regression suites, release management.
- `[[ml-observability-monitoring]]` — online metrics, telemetry, feedback signals, drift; the online half.
- `[[rag-vector-databases]]` — the RAG pipeline whose retrieval+generation you evaluate (RAGAS).
- `[[llm-app-agent-frameworks]]` — the agents/LLM apps whose trajectories and tool use you evaluate.
- `[[recsys-ranking]]` — ranking/recsys metrics (nDCG/MRR/MAP) and online interleaving in depth.
- `[[fine-tuning-peft]]` — evals to select checkpoints and detect regressions from fine-tuning.
- `[[responsible-ai-governance]]` — fairness criteria, disaggregated evaluation, safety gates.

---

# Reference — ml-evaluation-evals

# ML & LLM Evaluation — The Full Reference

Evaluation is the discipline that separates a demo from a production system. The model is a commodity;
the **eval set and the harness around it are the moat**. If you cannot measure quality, you cannot
improve it, you cannot ship safely, and you cannot tell whether your last change helped or hurt. This
guide covers classical ML metrics, LLM/generative evaluation, LLM-as-a-judge, RAG and agent evaluation,
online experimentation, and eval-ops — written for someone who has to make these calls in production.

The ecosystem moves fast (it is 2026): benchmark leaderboards, judge models, and harness APIs change
monthly. **Never quote a benchmark score from memory — verify against current docs/leaderboards.** This
guide gives you the durable mental models and the traps; treat all specific tool flags and numbers as
"verify against current docs."

---

## 1. The eval mindset

**You cannot improve what you cannot measure.** The first deliverable of any serious ML/LLM project is
not the model — it is a trustworthy evaluation you believe in. Build the eval before you optimize the
system, and treat the eval itself as a versioned, reviewed artifact.

**Eval-driven development.** Mirror test-driven development: collect failing/representative cases first,
encode them as an eval set with a scoring function, then iterate on prompts/models/retrieval/data until
the score moves. Every prompt tweak, model swap, or retrieval change is a hypothesis you confirm or
reject against the eval — not vibes.

**Offline vs online.**
- *Offline* — run a fixed dataset through the system in a harness; deterministic-ish, cheap, fast,
  runs in CI. Measures capability on a frozen distribution. Good for regression gating and model
  selection. Limitation: it is a *proxy* for real usage and drifts from production.
- *Online* — measure real users/traffic via A/B tests, guardrail metrics, feedback signals. The ground
  truth for business impact, but slow, noisy, expensive, and risky. See [[ml-observability-monitoring]].
- You need **both**. Offline catches regressions before they ship; online validates that offline gains
  translate to user value. The gap between them (the *offline-online gap*) is itself a signal.

**The proxy-metric problem.** Every metric is a proxy for what you actually care about. Optimizing the
proxy hard enough breaks the correlation with the true objective (Goodhart's law: "when a measure
becomes a target, it ceases to be a good measure"). ROUGE is a proxy for summary quality; click-through
is a proxy for satisfaction; a judge score is a proxy for human preference. Keep a *basket* of metrics,
watch guardrails, and periodically re-validate the proxy against ground truth (human labels).

**A representative eval set.** The single highest-leverage artifact. Properties of a good one:
- **Matches the production distribution** — sampled from real (or realistic) inputs, not just easy or
  textbook cases. If prod is 60% short queries and 5% adversarial, the eval set should reflect that.
- **Sliced** — stratified by the segments you care about (language, user tier, query type, difficulty,
  demographic groups) so an aggregate win can't hide a per-slice regression.
- **Stable + versioned** — a frozen "golden set" you can compare across runs; expand it deliberately,
  version it, review changes like code.
- **Includes hard/failure cases** — bugs you've fixed become permanent regression tests.
- **Right size** — large enough for statistical power on the slices you split by; for expensive
  judge/human grading, a few hundred well-chosen cases beat thousands of redundant easy ones.

**Contamination and leakage.** The two ways an eval lies to you:
- *Train/test leakage* — eval examples (or near-duplicates) appear in training data, or a feature
  encodes the label. The classic ML version: deduplicate across splits, split by *entity/time* not by
  row when records correlate, never let future information leak into past predictions (temporal split
  for time series; group split when one user has many rows).
- *Benchmark contamination* — the eval (e.g., MMLU, HumanEval) leaked into a model's pretraining
  corpus, so the score measures memorization, not capability. Endemic for public LLM benchmarks. Defend
  with private/held-out eval sets, freshly collected data, canary strings, and contamination probes.
  **Assume any public benchmark older than your model's training cutoff is partially contaminated.**

---

## 2. Classical ML metrics

### Classification

Start from the **confusion matrix** (TP/FP/FN/TN) and derive everything; report metrics *at a chosen
operating threshold*, not just threshold-free aggregates.

- **Precision** = TP/(TP+FP) — of what you flagged, how much was right (cost of false positives).
- **Recall / TPR / sensitivity** = TP/(TP+FN) — of what's actually positive, how much you caught (cost
  of false negatives).
- **F1** = harmonic mean of precision & recall; **Fβ** weights recall (β>1) or precision (β<1). Use
  when you need a single threshold-dependent number and care about both.
- **Specificity / TNR** = TN/(TN+FP); **FPR** = 1 − specificity.
- **Accuracy** is misleading under class imbalance (99% negatives → 99% accuracy by predicting all
  negative). Almost never the headline metric for imbalanced problems.

**Threshold-free, ranking-quality metrics:**
- **ROC-AUC** — area under TPR vs FPR; probability a random positive ranks above a random negative.
  Insensitive to threshold and to class balance — which is also its weakness: on heavily imbalanced
  data it looks optimistic because FPR has a huge negative denominator.
- **PR-AUC (average precision)** — area under precision–recall; **preferred for rare-positive
  problems** (fraud, retrieval, anomaly) because it focuses on the positive class. A random baseline's
  PR-AUC equals the positive prevalence, so always report that baseline.

**Threshold selection** is a product decision, not an ML one: pick it to satisfy a precision or recall
target, or to minimize expected cost given the FP/FN cost ratio. Report the confusion matrix *at the
deployed threshold*, and show how metrics move across thresholds.

**Calibration** — do predicted probabilities mean what they say? A model is calibrated if among samples
scored 0.7, ~70% are positive. Critical when the probability feeds a downstream decision (expected-value
ranking, thresholding, abstention). Measure with a **reliability diagram** and **Expected Calibration
Error (ECE)**; report **Brier score** (mean squared error of probabilities) as a proper scoring rule.
Fix with Platt scaling (logistic) or isotonic regression on a held-out calibration set. Modern deep
nets and especially temperature-unscaled LLM token probabilities are often badly miscalibrated.

**Multiclass:** macro-average (mean over classes — treats classes equally, surfaces rare-class failure)
vs micro-average (aggregate over instances — dominated by frequent classes) vs weighted. Choose
deliberately; report the per-class breakdown, not just the average.

### Regression

- **MAE** — robust, same units, interpretable; treats all errors linearly.
- **RMSE** — penalizes large errors more (squared); use when big misses are disproportionately costly.
- **MAPE / sMAPE** — scale-free percentage error; explodes near zero, biased toward under-prediction.
- **R²** — fraction of variance explained; can be negative for a model worse than the mean.
- **Quantile loss / pinball loss** for quantile regression and prediction intervals.
- Always plot **residuals vs prediction** and **predicted vs actual** — heteroscedasticity and bias
  hide in the aggregate. Report error by segment.

### Ranking & recommendation

When output is an ordered list, position matters. (Deep dive in [[recsys-ranking]].)

- **Precision@k / Recall@k** — relevant items in the top-k / fraction of all relevant items caught in
  top-k. Simple, threshold-by-position.
- **MAP** (Mean Average Precision) — mean over queries of average precision; rewards ranking relevant
  items high; binary relevance.
- **MRR** (Mean Reciprocal Rank) — 1/rank of the first relevant result, averaged. For "find one good
  answer fast" tasks (QA, known-item search).
- **nDCG@k** — Discounted Cumulative Gain with a log position discount, normalized by the ideal
  ordering. The default for **graded** relevance; handles multi-level labels and position. Define gain
  and discount explicitly; nDCG numbers are only comparable under the same formulation.
- **Hit rate / coverage / catalog coverage / diversity / novelty** for recsys health beyond accuracy —
  an accurate recommender that shows everyone the same 10 items is a bad product.

### Segment / slice-based evaluation & fairness

**An aggregate metric is the average of a distribution you should be looking at directly.** A model can
gain 1% overall while regressing 15% on a critical slice. Always cut metrics by meaningful segments
(geography, language, device, user tenure, input length, difficulty, time) — this is how you catch the
failures that aggregate numbers hide.

**Fairness** is slice-based evaluation across protected/sensitive groups. Common (mutually
incompatible) criteria:
- *Demographic parity* — equal positive rate across groups.
- *Equalized odds* — equal TPR and FPR across groups.
- *Equal opportunity* — equal TPR across groups.
- *Calibration within groups* — scores mean the same thing per group.

These cannot generally all hold at once; choosing among them is a values/policy decision, not a
statistical one — coordinate with [[responsible-ai-governance]]. Disaggregated evaluation (report every
metric per group) is the non-negotiable baseline regardless of which criterion you adopt.

---

## 3. LLM / generative evaluation

Open-ended generation is the hard part: there is rarely one correct output, outputs are long and
free-form, and "quality" is multi-dimensional (correctness, relevance, faithfulness, style, safety,
format). The classical-ML toolkit doesn't directly apply.

### Benchmarks (and their limits)

Public benchmarks are useful for *coarse capability triage* and comparing base models, **not** for
evaluating your application. Know the major ones and their failure modes (verify current versions and
scores against leaderboards/papers — do not trust remembered numbers):

| Benchmark | Measures | Known limits |
|-----------|----------|--------------|
| **MMLU / MMLU-Pro** | Broad multiple-choice knowledge | Heavily contaminated; saturating; MC ≠ generation |
| **GPQA (Diamond)** | Graduate-level science, "Google-proof" | Small; still leaks over time |
| **HumanEval / MBPP / LiveCodeBench** | Code generation (pass@k) | HumanEval saturated & contaminated; LiveCodeBench uses fresh problems to fight this |
| **GSM8K / MATH / AIME** | Math reasoning | GSM8K contaminated/saturated; harder sets rotate |
| **MT-Bench / Arena-Hard / Chatbot Arena** | Multi-turn chat, human/judge preference | Judge & population biases; Arena is online human pairwise |
| **SWE-bench (Verified)** | Real GitHub issue resolution | Hard, realistic; harness/setup-sensitive |
| **HELM** | Multi-metric holistic suite | Broad but coarse for any one app |

**The core caveat: benchmark performance is a weak predictor of your task performance.** A model that
tops MMLU may be worse at *your* customer-support summarization. Benchmarks are also a contamination
magnet (§1). Use them to shortlist; never to ship.

**Build task-specific evals.** The thing that matters is a dataset of *your* inputs with a scoring
function for *your* definition of good. This is where to spend effort.

### Reference-based metrics (and why they're weak for gen)

When you have reference outputs, n-gram/embedding overlap metrics are cheap but blunt:
- **BLEU** (MT) — n-gram precision with brevity penalty. Surface-form; misses paraphrase; corpus-level.
- **ROUGE** (summarization) — n-gram/longest-subsequence recall vs reference(s). Rewards lexical
  overlap, blind to factuality and to valid rewordings.
- **METEOR** — adds stemming/synonym matching; better correlation than BLEU but still surface-ish.
- **BERTScore / BLEURT / COMET** — embedding-based semantic similarity; correlate better with humans
  than n-gram metrics, but still reference-bound and weak on factuality/reasoning.
- **Exact match / F1 (token)** — fine for short extractive QA; useless for open generation.

**Why they're weak for open generation:** a correct answer can share almost no n-grams with the
reference; a fluent wrong answer can share many. They cannot judge factuality, instruction-following,
or reasoning. Use them only where references are tight and outputs are short (translation, extractive
QA, constrained summarization), and even then alongside a judge or human check.

### Programmatic / assertion-based checks

The most reliable and cheapest evals are deterministic where the task allows it: schema validation
(does the JSON parse and match the schema?), regex/keyword presence or absence, code execution and unit
tests (for code gen — run it), numeric tolerance, string containment, length/format constraints,
toxicity/PII classifiers. **Prefer a deterministic check over a judge whenever the property is
checkable.** Reserve LLM-as-a-judge for the genuinely subjective dimensions.

---

## 4. LLM-as-a-judge

Use a strong LLM to grade outputs against a rubric. It scales human-like judgment to thousands of
open-ended cases cheaply — and it is biased and noisy, so it must be calibrated and bias-controlled.
(See evidentlyai.com/llm-guide/llm-as-a-judge and langfuse.com's LLM-evaluation guide.)

### Modes

- **Pointwise (direct scoring)** — judge scores one output on a rubric (e.g., 1–5, or pass/fail per
  criterion). Easiest to run in CI; absolute scores are noisy and drift between judge versions. **Bias
  toward binary/low-cardinality rubrics** (pass/fail, or a 1–3 scale with explicit anchors) — fine-
  grained 1–10 scores are mostly noise and poorly reproducible.
- **Pairwise (A vs B)** — judge picks the better of two outputs. More reliable than absolute scoring
  (humans and models compare better than they rate); ideal for regression testing (new vs old) and
  model selection. Cost is O(n) per comparison; for many candidates use a tournament/Elo, not all
  pairs.
- **Reference-guided** — give the judge a gold answer or rubric/reference to ground its decision; raises
  agreement substantially when a reference exists.

### Biases (and mitigations)

LLM judges have systematic, measurable biases. Controlling them is the whole game:

- **Position bias** — favors the first (or a fixed) position in pairwise. *Mitigate:* evaluate both
  orders (A,B) and (B,A) and only count a win if consistent; randomize order; report the flip rate.
- **Verbosity / length bias** — prefers longer, more elaborate answers regardless of quality.
  *Mitigate:* rubric instruction to ignore length; control for length; penalize unnecessary verbosity
  explicitly.
- **Self-preference / self-enhancement bias** — a judge favors text from its own model family.
  *Mitigate:* use a different model family as judge than the one generating; or an ensemble of judges.
- **Sycophancy / authority / formatting bias** — swayed by confident tone, citations (even fake),
  markdown, or assertions of correctness. *Mitigate:* explicit rubric, require evidence, ask for
  reasoning before the verdict.
- **Leniency / score clustering** — pointwise judges bunch scores high. *Mitigate:* anchored rubric
  with concrete examples per level; prefer pairwise.

### Designing a good judge

1. **Decompose** the quality into specific, independently-judged criteria (factuality, relevance,
   completeness, safety, format) rather than one fuzzy "is this good?".
2. **Anchored rubric** — define each score level with a concrete description and ideally an example.
3. **Chain-of-thought then verdict** — ask the judge to reason, *then* emit a structured verdict
   (JSON), so the score is grounded and parseable. Constrain the output format.
4. **Few-shot the judge** with human-labeled exemplars near the decision boundary.
5. **Use a strong judge model**, ideally a different family than the system under test.

### Calibration: agreement with humans

**A judge you haven't validated against humans is just a vibe with extra steps.** Before trusting a
judge at scale, measure its **agreement with human labels** on a sample: accuracy/F1 vs human verdicts
for binary, Cohen's/Fleiss' κ or Spearman/Kendall correlation for graded, and compare to *human–human*
agreement (the realistic ceiling — humans disagree too). If judge–human agreement is below human–human
agreement, fix the rubric before scaling. Re-validate whenever you change the judge model, prompt, or
rubric — judge prompts are versioned artifacts.

### When to trust it

Trust LLM-as-a-judge for *relative* comparisons (A vs B regressions), coarse quality screening at
scale, and dimensions humans agree on. Distrust it for high-stakes absolute scores, anything requiring
domain expertise it lacks, factuality it can't verify without a reference, and as the *sole* gate for a
launch. Keep a human-in-the-loop sample to continuously audit the judge.

---

## 5. RAG evaluation

A RAG system has two failure surfaces — **retrieval** and **generation** — and you must evaluate them
separately, otherwise you can't tell whether a bad answer is a retrieval miss or a generation
hallucination. (Pipeline details in [[rag-vector-databases]].) The RAGAS-style metric set is the
de-facto vocabulary; verify the current RAGAS API against its docs:

**Retrieval quality (does the right context get fetched?):**
- **Context Precision** — of the retrieved chunks, what fraction is relevant (signal vs noise; ranks
  relevant context high). Low → retriever pulls junk that distracts the generator.
- **Context Recall** — of the information needed to answer, how much is present in the retrieved
  context. Low → the answer *can't* be grounded; fix chunking/retrieval/index before touching prompts.

**Generation quality (given the context, is the answer good?):**
- **Faithfulness / Groundedness** — are the answer's claims entailed by the retrieved context, with no
  hallucinated facts? The most important RAG safety metric. Typically scored by decomposing the answer
  into claims and checking each against context (often via a judge or NLI model).
- **Answer Relevance** — does the answer actually address the question (not off-topic, not padded)?
- **Answer Correctness** — answer vs a ground-truth reference (when you have one), combining semantic
  similarity and factual overlap.

**Diagnostic logic:** low context recall → retrieval problem (chunking, embedding, top-k, reranking);
high context recall but low faithfulness → generation/prompt problem (model ignores or contradicts
context); high faithfulness but low answer relevance → the model is grounded but not answering the
question. Evaluate both surfaces on every change. Build a RAG eval set with question / ideal-answer /
ground-truth-context triples; synthesize an initial set (generate Q&A from your corpus) then human-
review it.

---

## 6. Agent evaluation

Agents (multi-step, tool-using, stateful) are the hardest to evaluate: the output is a *trajectory*,
success is often binary-but-rare, and failures compound across steps. (Frameworks in
[[llm-app-agent-frameworks]].)

- **Task success / goal completion** — did the agent achieve the end state? The headline metric. Needs
  a programmatic success check (final DB state, file produced, test passes) wherever possible, or a
  judge against a goal rubric. This is *end-to-end* quality.
- **Trajectory / process evaluation** — was the *path* correct? Right tools chosen, right arguments,
  right order, no unnecessary or destructive steps. An agent can reach the goal by luck through a bad
  path (brittle) or fail late after a good start. Evaluate against a reference trajectory or with a
  judge over the step log.
- **Tool-use correctness** — for each call: was the right tool selected, were arguments well-formed and
  correct, was the result interpreted correctly? Often the dominant failure mode; check tool-call
  accuracy and argument validity directly.
- **Multi-step reliability** — per-step success compounds: 95% per step over 10 steps ≈ 60% end-to-end.
  Measure step count, recovery-from-error rate, and reliability across *repeated* runs (agents are
  stochastic — run each case N times and report success *rate* and variance, not a single pass).
- **Efficiency / cost** — steps, tokens, latency, $ per completed task; a "successful" agent that takes
  40 tool calls is often a failure in production.

Use both outcome (did it work) and process (did it work *for the right reasons*) signals; outcome-only
hides brittleness, process-only misses real success.

---

## 7. Online evaluation & experimentation

Offline says "this should be better." Online proves it on real users. (Monitoring/telemetry in
[[ml-observability-monitoring]].)

- **A/B testing** — randomize users/sessions between control and treatment; compare a primary metric
  with proper statistics (sufficient power, fixed-horizon or sequential test, correction for multiple
  comparisons). The gold standard for causal impact. Watch novelty effects, network interference, and
  sample-ratio mismatch (an SRM check is mandatory — if assignment is skewed, the experiment is
  invalid).
- **Interleaving** — for ranking/search, blend results from two systems in one list and attribute
  clicks; far more sensitive than A/B (each user sees both), so it needs much less traffic. Great for
  retrieval/recsys/RAG-retriever comparisons.
- **Guardrail metrics** — metrics that must *not* regress even if the primary metric wins: latency, cost
  per request, error/refusal rate, safety violations, retention. A win on the primary metric with a
  guardrail breach does not ship.
- **Online LLM-as-judge & feedback signals** — run a judge (often a cheaper model) on a sample of live
  traffic for continuous quality monitoring; collect explicit feedback (thumbs up/down, ratings) and
  implicit signals (edits, copy, retry, abandonment, conversation length, escalation to human). These
  are noisy and biased (responders ≠ population) — triangulate, don't trust any single signal.
- **The offline-online gap.** Offline gains routinely shrink or vanish online (distribution shift,
  proxy-metric divergence, feedback loops, your eval set not matching production). A persistent gap
  means your offline eval is unrepresentative — fix the eval set, don't ignore the gap. Treat
  offline as *necessary but not sufficient*: it gates what's *allowed* to be tested online, online
  decides what *ships*.

---

## 8. Eval ops — harnesses, tooling, CI

Evaluation is infrastructure, not a notebook you run once. Treat eval datasets, scoring functions, and
judge prompts as **versioned, reviewed, tested code**.

**Harnesses & tools** (pick by layer; verify current capabilities/APIs against each project's docs):

| Tool | Layer | Use for |
|------|-------|---------|
| **EleutherAI lm-evaluation-harness** | Academic/benchmark | Standardized base-model benchmarks (MMLU, GSM8K, etc.); reproducible leaderboard-style runs |
| **OpenAI Evals** | Framework | Custom + registry evals, model-graded templates |
| **UK AISI Inspect** | Framework | Rigorous, composable agent/safety evals; solvers, scorers, datasets |
| **Promptfoo** | Dev/CI | Side-by-side prompt/model comparison, assertions, red-teaming; CLI-first, easy in CI |
| **DeepEval** | Dev/CI | Pytest-style LLM unit tests; RAG/agent metrics, judge metrics |
| **RAGAS** | RAG | Faithfulness / answer relevance / context precision & recall |
| **Langfuse** | Platform | Tracing + datasets + online & offline scores + human annotation; production observability + eval |
| **Braintrust / Arize Phoenix / LangSmith** | Platform | Dataset mgmt, experiment tracking, judges, traces (verify current feature sets) |

Choose the smallest tool that fits: programmatic assertions for checkable properties, a dev/CI harness
(Promptfoo/DeepEval) for prompt iteration and gates, a platform (Langfuse/etc.) when you need
production traces + datasets + human annotation in one place. Don't adopt a heavy platform before you
have a representative eval set — the dataset is the asset, the tool is replaceable.

**Eval in CI / regression gates.** Wire offline evals into CI so prompt/model/retrieval changes are
gated on quality, not just unit tests (ties into [[mlops-lifecycle]]):
- **Golden/regression set** runs on every PR; fail the build if the score drops below a threshold or
  any known-fixed case regresses. Each production bug becomes a permanent golden case.
- **Gate on relative comparison** (pairwise new-vs-old) where absolute judge scores are too noisy.
- Account for **non-determinism**: pin temperature/seed where possible; for stochastic systems run N
  times and gate on the mean with a tolerance band, not a single run. Don't make a flaky judge a
  hard blocker — set thresholds with margin and alert on trends.
- Keep the gating suite fast/cheap (a curated subset); run the full expensive suite nightly.
- **Human-in-the-loop** stays in the loop: sample outputs for human review, feed disagreements back
  into the judge calibration and the golden set. The flywheel is human labels → judge calibration →
  scaled judging → sampled human audit → repeat.

---

## 9. Anti-patterns (the traps that bite in production)

- **Vibes-only evaluation** — "looks good to me" on a handful of cherry-picked prompts. No dataset, no
  metric, no regression protection. The default failure mode; the entire point of this skill is to
  replace it.
- **Trusting a contaminated/public benchmark** as your quality bar, or as evidence the model is good at
  *your* task. Public scores measure shortlist-worthiness, not fitness for purpose; assume contamination.
- **A single aggregate metric.** Hides per-slice regressions, masks the proxy-metric problem, and gets
  Goodharted. Always keep a basket + slices + guardrails.
- **LLM-as-a-judge with no bias controls and no human calibration** — position/verbosity/self-preference
  bias make the scores meaningless; an uncalibrated judge is a confident liar.
- **Fine-grained absolute scores** (1–10) treated as precise — they're mostly noise; prefer pairwise or
  low-cardinality anchored rubrics.
- **No online validation** — shipping on offline wins alone; the offline-online gap silently eats your
  gains (or worse, ships a regression).
- **An eval set that doesn't match production** — too easy, stale, wrong distribution, missing the hard
  cases. Then offline numbers go up while users suffer.
- **Train/eval leakage** — duplicates across splits, row-split when you should group/time-split,
  features that encode the label. Inflated offline numbers that collapse in production.
- **Optimizing the judge/metric instead of the product** — tuning prompts to please a biased judge, or
  the metric, rather than users. Re-anchor to human labels and online outcomes periodically.
- **Eval as a one-off, not infrastructure** — a notebook run once at launch, never in CI, never
  versioned. Quality silently rots.

---

## Rationalizations & rebuttals

The excuses for skipping real evaluation, each rebutted:

- **"The outputs look good to me — vibes are enough."** Vibes are a handful of cherry-picked prompts
  with no dataset, no metric, and no regression protection. You can't tell if your next change helped or
  hurt, and you can't catch the regression a user will. Build the representative eval set first.
- **"The public benchmark says this model is SOTA, so it's good for us."** Benchmark rank is a weak
  predictor of *your* task performance — a model that tops MMLU can be worse at your support
  summarization. Public benchmarks are also a contamination magnet; assume any one older than the
  model's training cutoff is partially memorized. Use them to shortlist, never to ship.
- **"One aggregate number went up, so we're better."** An aggregate is the average of a distribution
  you should look at directly. A model can gain 1% overall while regressing 15% on a critical slice, and
  a single target gets Goodharted. Keep a basket of metrics + slices + guardrails.
- **"The LLM judge scored it 8.7/10."** A judge with no bias controls and no human calibration is a
  confident liar: position, verbosity, and self-preference bias make the score meaningless, and
  fine-grained 1–10 scores are mostly noise. Validate judge–human agreement against the human–human
  ceiling, control biases (dual-order pairwise, length controls, cross-family judge), and prefer
  low-cardinality anchored rubrics or pairwise.
- **"Offline gains are clear — skip the online test."** Offline gains routinely shrink or vanish online
  (distribution shift, proxy divergence, feedback loops). Offline is necessary but not sufficient: it
  gates what's *allowed* to be tested; online A/B (with an SRM check and guardrails) decides what ships.
- **"We'll add evals later — ship now."** Eval-as-a-one-off notebook that never runs in CI means quality
  silently rots; every prompt/model/retrieval change becomes an unguarded hypothesis. Wire the golden
  set into CI on day one — it's infrastructure, not a launch chore.
- **"References overlap is high, so the generation is correct."** BLEU/ROUGE/BERTScore are surface or
  semantic overlap, blind to factuality and instruction-following — a fluent wrong answer can share many
  n-grams. Use a faithfulness check (claims entailed by context) or a calibrated judge for correctness.

## Red flags

Stop and reconsider if any of these are true:

- **Your quality bar is a public benchmark** (MMLU/HumanEval/GSM8K) with no contamination defense — the
  score likely measures memorization, not capability, and not fitness for *your* task.
- **A single aggregate metric** drives decisions, with no slices/segments and no guardrails — per-slice
  regressions and proxy-metric divergence are invisible.
- **An LLM judge runs without bias mitigation** — no dual-order pairwise, no length control, same model
  family as the system under test, no anchored rubric.
- **The judge has never been calibrated against human labels** (no agreement measured vs the human–human
  ceiling), or the rubric/judge changed without re-validation.
- **Fine-grained absolute scores (1–10)** are treated as precise signal rather than noise.
- **No online validation** — shipping on offline wins alone, with no A/B/interleaving and no SRM check.
- **The eval set doesn't look like production** — too easy, stale, wrong distribution, or missing the
  hard/failure cases; offline numbers climb while users suffer.
- **Train/eval leakage suspected** — duplicates across splits, row-split where a group/time split is
  required, or a feature that encodes the label; for RAG, retrieval and generation aren't scored
  separately.

## Verification gate (definition of done)

The work is not done until all of these hold:

- [ ] **Representative, uncontaminated eval set** exists: sampled from the production distribution,
  versioned/frozen as a golden set, includes hard/failure cases, deduplicated with entity/time/group
  splits, and (for LLM tasks) defended against benchmark contamination (private/held-out data).
- [ ] **Task-appropriate metrics with slices**: a basket (not one number) chosen for the task —
  e.g. PR-AUC for rare positives, nDCG for graded ranking, faithfulness + answer relevance for RAG,
  task-success + trajectory for agents — reported per meaningful segment with guardrails defined.
- [ ] **Judge is bias-controlled and human-calibrated** (where a judge is used): decomposed anchored
  rubric, chain-of-thought-then-verdict, cross-family judge, dual-order pairwise / length controls, and
  measured judge–human agreement at/above the human–human ceiling, re-validated after any judge/prompt
  change.
- [ ] **Online / A-B validation done**: real-traffic A/B or interleaving with proper power, an SRM
  check, guardrail metrics confirmed not regressed, and the offline-online gap inspected — online (not
  offline alone) decided the ship.
- [ ] **Eval wired into CI**: golden/regression set runs on every PR and fails the build on a score drop
  or any known-fixed case regressing; non-determinism handled (pinned seed/temp or N-run mean with a
  tolerance band); full expensive suite runs nightly; human-in-the-loop sampling feeds disagreements
  back into judge calibration and the golden set.

## Canonical references (verify against current versions)

- HelloInterview — ML system design, evaluation:
  https://www.hellointerview.com/learn/ml-system-design/core-concepts/evaluation
- Evidently AI — LLM-as-a-judge guide: https://www.evidentlyai.com/llm-guide/llm-as-a-judge
- Langfuse — LLM evaluation guide & docs: https://langfuse.com/docs/scores/overview (and the LLM-eval
  guide on langfuse.com)
- RAGAS docs: https://docs.ragas.io
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- OpenAI Evals: https://github.com/openai/evals
- UK AISI Inspect: https://inspect.aisi.org.uk
- Promptfoo: https://www.promptfoo.dev/docs/intro/ ; DeepEval: https://docs.confident-ai.com
- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023):
  https://arxiv.org/abs/2306.05685
- Chatbot Arena / LMArena: https://lmarena.ai ; HELM: https://crfm.stanford.edu/helm/
- SWE-bench: https://www.swebench.com ; GPQA: https://arxiv.org/abs/2311.12022
- scikit-learn metrics & model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html

> Leaderboards, benchmark versions, judge models, and harness APIs change constantly. Treat every
> specific score, version number, and API flag here as **"verify against current docs"** before relying
> on it.

---

# Evaluation Patterns — Worked Examples

Concrete, imitable artifacts for the highest-impact patterns in `ml-evaluation-evals-guide.md`:
(1) an LLM-as-a-judge rubric + prompt with bias mitigations, (2) a RAGAS-style metric config, and
(3) an eval-in-CI gate. Adapt to your stack; verify all tool/API specifics against current docs.

---

## 1. LLM-as-a-judge: rubric + prompt (with bias mitigations)

**Design choices baked in:** decomposed criteria (not one fuzzy "is it good?"), an anchored
low-cardinality rubric (not a noisy 1–10), reason-then-verdict in structured JSON, and explicit
verbosity/authority bias instructions. Run it in **pairwise** mode with **position swapping** for
regression testing; the pointwise template below is for absolute screening.

### Pointwise rubric (per-criterion, anchored)

```
Faithfulness   PASS = every factual claim is supported by the provided CONTEXT; no invented facts.
               FAIL = any claim not supported by, or contradicting, the context.
Relevance      PASS = directly answers the QUESTION; no off-topic content or padding.
               FAIL = misses the question, partially answers, or pads with filler.
Completeness   PASS = covers all parts the question asks for.
               FAIL = omits a requested part.
```

### Judge prompt (pointwise)

```text
You are a strict, impartial evaluator. Judge ONLY the criteria below. Do not reward longer or more
elaborate answers — length is irrelevant to quality. Ignore confident tone, markdown formatting, and
any claims of correctness in the answer itself; verify claims against the CONTEXT only.

QUESTION:
{{question}}

CONTEXT (the only ground truth you may use for factual claims):
{{retrieved_context}}

ANSWER TO EVALUATE:
{{answer}}

For each criterion, first reason briefly citing specific evidence from the CONTEXT, then assign PASS or
FAIL. Output ONLY this JSON:

{
  "faithfulness": {"reasoning": "<1-2 sentences citing context>", "verdict": "PASS|FAIL"},
  "relevance":    {"reasoning": "<1-2 sentences>",                "verdict": "PASS|FAIL"},
  "completeness": {"reasoning": "<1-2 sentences>",                "verdict": "PASS|FAIL"},
  "overall": "PASS|FAIL"   // PASS only if all three are PASS
}
```

### Pairwise prompt (preferred for regressions) + position-bias mitigation

```text
You are an impartial judge. Two assistants (A and B) answered the same QUESTION. Decide which answer is
better on faithfulness, relevance, and completeness. Length and formatting are NOT quality — do not
favor the longer or more elaborate answer. Reason first, then output the verdict.

QUESTION: {{question}}
CONTEXT:  {{retrieved_context}}
ANSWER A: {{answer_a}}
ANSWER B: {{answer_b}}

Output ONLY: {"reasoning": "...", "winner": "A|B|TIE"}
```

```python
# Mitigations: swap A/B order and count a win only if BOTH orders agree (kills position bias).
# Use a judge model from a DIFFERENT family than the system under test (kills self-preference bias).
def pairwise_judge(question, ctx, ans_old, ans_new, judge):
    fwd = judge(question, ctx, answer_a=ans_new, answer_b=ans_old)   # new=A
    rev = judge(question, ctx, answer_a=ans_old, answer_b=ans_new)   # new=B
    new_wins_fwd = fwd["winner"] == "A"
    new_wins_rev = rev["winner"] == "B"
    if new_wins_fwd and new_wins_rev:
        return "new"          # consistent across both orders
    if (not new_wins_fwd) and (not new_wins_rev) and "TIE" not in (fwd["winner"], rev["winner"]):
        return "old"
    return "tie"             # inconsistent => position-sensitive => treat as tie, log the flip
```

**Before trusting this judge at scale — calibrate against humans:**

```python
# Agreement with human labels vs the human-human ceiling. If judge<->human < human<->human, the rubric
# is the problem, not the system. Re-run whenever judge model / prompt / rubric changes.
from sklearn.metrics import cohen_kappa_score
judge_vs_human = cohen_kappa_score(human_labels, judge_labels)   # graded -> spearman/kendall instead
human_vs_human = cohen_kappa_score(annotator_a, annotator_b)     # realistic upper bound
assert judge_vs_human >= 0.8 * human_vs_human, "judge under-agrees with humans; fix rubric before scaling"
```

---

## 2. RAGAS-style metric config

Evaluate retrieval and generation **separately**. Dataset rows carry question / answer / retrieved
contexts / ground-truth, so each metric isolates one failure surface. (Verify the current RAGAS API and
metric names against https://docs.ragas.io — they evolve.)

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,   # retrieval: are retrieved chunks relevant & ranked well? (signal vs noise)
    context_recall,      # retrieval: is all needed info present in retrieved context?
    faithfulness,        # generation: are answer claims entailed by the context? (anti-hallucination)
    answer_relevancy,    # generation: does the answer address the question (no padding/off-topic)?
    answer_correctness,  # answer vs ground-truth reference (semantic + factual overlap)
)

eval_data = Dataset.from_dict({
    "question":     [r["q"]        for r in golden],
    "answer":       [r["model_a"]  for r in golden],   # the system's generated answer
    "contexts":     [r["ctx"]      for r in golden],   # list[str] retrieved chunks, in rank order
    "ground_truth": [r["ideal"]    for r in golden],   # human-reviewed reference answer
})

report = evaluate(
    eval_data,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness],
    # llm=<judge model — different family than the generator>, embeddings=<your embedding model>
)
print(report)  # {'context_precision': .., 'context_recall': .., 'faithfulness': .., ...}
```

**Diagnostic decision table** (read the metrics together, not in isolation):

| context_recall | faithfulness | answer_relevancy | Diagnosis → fix |
|----------------|--------------|------------------|-----------------|
| low | — | — | Retrieval miss → chunking, embedding model, top-k, reranker |
| high | low | — | Generation ignores/contradicts context → prompt, stronger model |
| high | high | low | Grounded but not answering → query understanding, prompt instructions |
| high | high | high | Healthy; iterate on edge slices |

> Build the golden set by synthesizing Q/A pairs from your corpus, then **human-review** them — never
> ship a fully auto-generated eval set as ground truth.

---

## 3. Eval-in-CI gate

Run a fast curated subset on every PR; gate on **relative** (pairwise new-vs-old) quality so noisy
absolute scores don't flake the build; run the full expensive suite nightly. Ties into
`[[mlops-lifecycle]]`.

```yaml
# .github/workflows/eval-gate.yml  (illustrative; pin tool versions, verify CLI flags against docs)
name: eval-gate
on: [pull_request]
jobs:
  offline-evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r eval/requirements.txt
      # Deterministic assertion suite first — cheap, catches format/schema/code-exec failures fast.
      - name: programmatic checks (schema, regex, code execution)
        run: python -m eval.assertions --dataset eval/golden_subset.jsonl
      # Relative quality gate: pairwise judge, current PR vs main baseline, position-swapped.
      - name: pairwise quality gate
        run: |
          python -m eval.pairwise \
            --candidate "$PR_MODEL"  --baseline "$MAIN_MODEL" \
            --dataset eval/golden_subset.jsonl \
            --runs 3 \                       # N runs: handle non-determinism, gate on the mean
            --fail-if-winrate-below 0.45     # tolerance band; new must not be clearly worse than old
```

```python
# eval/pairwise.py (core gate logic)
def run_gate(candidate, baseline, dataset, runs, fail_if_winrate_below):
    wins = ties = losses = 0
    for case in dataset:
        verdicts = [pairwise_judge(case.q, case.ctx,                       # swaps A/B order itself
                                   baseline.answer(case), candidate.answer(case), judge)
                    for _ in range(runs)]
        v = majority(verdicts)                  # reduce stochasticity across N runs
        wins   += v == "new"
        losses += v == "old"
        ties   += v == "tie"
    win_rate = (wins + 0.5 * ties) / len(dataset)
    # Hard fail any KNOWN-FIXED regression case (each prod bug is a permanent golden case).
    assert not any_known_fixed_case_regressed(dataset), "a previously-fixed bug regressed"
    assert win_rate >= fail_if_winrate_below, f"quality gate failed: win_rate={win_rate:.2f}"
    print(f"PASS win_rate={win_rate:.2f}  W{wins}/L{losses}/T{ties}")
```

**Why this shape:** deterministic checks run first (cheapest, no judge noise); the judge gate is
**pairwise** (relative comparisons are far more reliable than absolute scores) with **N runs + a
tolerance band** instead of a brittle single-run threshold; previously-fixed bugs are hard-fail
regression cases. Offline passing only earns the right to test online — A/B test before you ship
(see `[[ml-observability-monitoring]]`).
