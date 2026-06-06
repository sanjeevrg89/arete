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
