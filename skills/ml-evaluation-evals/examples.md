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
