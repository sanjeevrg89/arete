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
