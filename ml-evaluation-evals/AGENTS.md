# AGENTS.md — ML & LLM Evaluation Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ml-evaluation-evals-guide.md`** next to this file —
> read it before designing or implementing any evaluation, and apply it. Concrete artifacts to imitate
> (judge rubric+prompt, RAGAS-style config, eval-in-CI gate) are in **`examples.md`**. This is the
> always-on summary.
>
> **The model is a commodity; the eval set and harness are the moat.** You cannot improve, ship safely,
> or know if a change helped without a trustworthy, representative, bias-controlled evaluation. The
> ecosystem moves fast (2026) — **never quote a benchmark score from memory; verify against current docs.**

## When evaluating ML/LLM systems, apply these by default:

- **Eval before you optimize.** Build the eval first; treat eval datasets, scoring functions, and judge
  prompts as versioned, reviewed code. Every prompt/model/retrieval change is a hypothesis tested
  against the eval — not vibes.
- **The eval set is the asset:** representative of the production distribution, sliced by the segments
  you care about, frozen as a golden set, seeded with the hardest/previously-failed cases. Each fixed
  bug becomes a permanent regression case.
- **No leakage / no contamination.** Group- or time-split, never naive row-split, when records
  correlate; deduplicate across splits. Assume any public benchmark older than the model's cutoff is
  partially contaminated — use private held-out sets for shipping decisions.
- **Classical metrics, chosen deliberately:** accuracy lies under class imbalance; use **PR-AUC** for
  rare positives (not ROC-AUC, which flatters imbalanced data); report the confusion matrix at the
  *deployed* threshold; check **calibration** (reliability diagram, ECE, Brier) when probabilities feed
  decisions; **nDCG/MRR/MAP/recall@k** for ranking. Plot residuals for regression.
- **Always slice / disaggregate.** An aggregate metric is the mean of a distribution you should look at
  directly — it hides per-segment regressions. Fairness = slice-based evaluation across protected
  groups (parity / equalized odds / equal opportunity / per-group calibration are mutually
  incompatible; choosing is a policy call).
- **Reference metrics are weak for open generation.** BLEU/ROUGE/BERTScore miss factuality and valid
  paraphrase — use only for tight short outputs (MT, extractive QA). Prefer **deterministic checks**
  (schema/JSON validation, code execution + unit tests, regex, numeric tolerance, classifiers) wherever
  the property is checkable; reserve judges for genuinely subjective dimensions.
- **LLM-as-a-judge — prefer pairwise over fine-grained absolute scores** (1–10 scores are mostly noise;
  use pass/fail or anchored low-cardinality rubrics). **Control biases:** position (swap A/B order, count
  a win only if consistent), verbosity/length (instruct to ignore), self-preference (judge with a
  different model family). Decompose into criteria; reason-then-verdict in structured JSON. **Calibrate
  against human labels** vs the human–human agreement ceiling before trusting at scale; re-validate on
  any judge/prompt/rubric change. A judge is great for relative regression checks, untrustworthy as a
  sole launch gate.
- **RAG: evaluate retrieval and generation separately** — context precision/recall (retrieval) vs
  faithfulness/groundedness + answer relevance (generation), RAGAS vocabulary. Low context recall →
  retrieval bug; high recall + low faithfulness → generation bug.
- **Agents are trajectories:** measure task success (programmatic success check where possible) *and*
  trajectory/tool-use correctness; per-step success compounds, so run each case N times and report
  success *rate* + variance, plus cost/steps/latency.
- **Online proves it.** Offline gates what may ship; **A/B tests** (check sample-ratio mismatch),
  **interleaving** for ranking, **guardrail metrics** (latency/cost/safety must not regress), and
  feedback signals decide what ships. Watch the offline-online gap — a persistent gap means your eval
  set is unrepresentative; fix the eval, not the gap.
- **Evals in CI:** golden/regression suite on every PR, gated on relative (pairwise) comparison or a
  threshold with margin; handle non-determinism (pin seed/temp; or N runs + tolerance band, don't make
  a flaky judge a hard blocker); fast subset on PRs, full suite nightly; human-in-the-loop audit feeds
  judge calibration and the golden set.

## Anti-patterns to reject
Vibes-only evaluation · trusting a contaminated public benchmark as your quality bar · a single
aggregate metric (hides slices, gets Goodharted) · LLM-as-a-judge with no bias controls or human
calibration · fine-grained absolute judge scores treated as precise · shipping on offline wins with no
online validation · an eval set that doesn't match production · train/eval leakage · optimizing the
judge/metric instead of the product · eval as a one-off notebook instead of versioned CI infrastructure.

## Definition of done for an evaluation
A representative, versioned eval set (no leakage/contamination) · the right metrics chosen and reported
*with slices* and guardrails, not a single aggregate · any LLM-judge bias-controlled and human-
calibrated · RAG retrieval+generation evaluated separately / agents evaluated on outcome+trajectory ·
offline gates wired into CI · online validation planned for anything that ships. Full detail and
references in `ml-evaluation-evals-guide.md`.
