# AI Research Science — Worked Examples

Three artifacts to imitate: a rigorous **ablation-study design template**, an **RLHF/post-training
pipeline in prose** (with knobs and failure modes), and a **read-and-reproduce-a-paper checklist**. These
are research-grade scaffolds, not runnable code — for code/frameworks see the engineering siblings.

---

## 1. Ablation-study design template

Use this before running anything. If you cannot fill in every field, you are not ready to spend compute.

**Title / one-line claim.** e.g. "Adding QK-norm reduces loss-spike frequency at ≥1B params without
hurting final loss."

**Hypothesis (mechanism, not just effect).** *What* and *why*. "QK-norm caps attention-logit magnitude,
preventing the softmax-saturation that precedes the spikes we observe."

**Predictions — confirming and disconfirming.**
- If true: spike rate ↓ across seeds; final eval loss within noise of baseline; attention-logit max stays
  bounded.
- If false: no change in spike rate, OR final loss *worse* (the norm removed useful signal).
- A pre-registered disconfirming prediction is what separates science from post-hoc storytelling.

**Variable under test.** Exactly one. (QK-norm on/off.)

**Controls held fixed.** Tokens seen, data + data order, model shape, optimizer + LR schedule, batch
size, precision, init seed set, eval suite + decoding params. List them explicitly — the ones you forget
are the confounds.

**Baseline.** The strongest honest baseline at *matched* compute/tokens. Reproduce its known number first;
if you can't, stop and debug — your delta is meaningless against a broken baseline.

**Scale ladder.** Run at ≥2–3 sizes (e.g. 150M / 400M / 1B) so you can see whether the effect is
constant, grows, or vanishes with scale (the bitter-lesson check). Cheapest scale where the effect is
visible, then confirm the trend.

**Seeds & variance.** ≥3 seeds per cell; report mean ± std (or min–max). Define the **noise band**; a
delta inside it is null.

**Metrics.** Primary (the claim) + guardrails (don't win on X by silently regressing Y). Prefer a
**continuous** metric over exact-match to avoid manufacturing discontinuities. Plot learning curves
(metric vs tokens/FLOPs), not just endpoints.

**Compute budget & stopping rule.** Total FLOPs, and a pre-committed stop (fixed tokens) so you don't
peek-and-stop on a favorable seed.

**Confounds & threats to validity.** Contamination/leakage in eval ([[ml-evaluation-evals]]), prompt-format
sensitivity, answer-extraction bugs, optimizer-state interactions, "more steps for free" asymmetries.

**Decision rule.** State *before* running what result ships the change, kills it, or triggers a
follow-up. Pre-committing the decision prevents motivated reasoning.

---

## 2. RLHF / post-training pipeline (diagram-in-prose)

The standard alignment-from-feedback pipeline and its three modern variants. Engineering (frameworks,
memory, rollout/learner split) → [[rl-rlhf-frameworks]].

```
Pretrained base
      │  (capable, unsteerable)
      ▼
[1] SFT  ── demonstrations / instruction data ──►  policy π_sft  (also the REFERENCE π_ref)
      │
      ├──────────────► [2] REWARD MODEL  (only for PPO / classic RLHF)
      │                    preference pairs (A≻B) → Bradley-Terry head → r(x,y)
      │
      ▼
[3] OPTIMIZE THE POLICY — choose one path:

  (a) PPO          maximize  E[ r(x,y) − β·KL(π_θ ‖ π_ref) ]   on the policy's OWN samples (on-policy)
                   needs: policy + reference + reward + critic (4 copies), GAE, clipped surrogate

  (b) DPO / kin    classification loss on offline preference pairs; reward reparameterized as
                   r = β·log[π_θ/π_ref];  NO reward model, NO sampling (off-policy)

  (c) GRPO / RLVR  sample a GROUP of G responses per prompt; advantage = group-normalized reward;
                   NO critic;  reward = verifier/rule (math ==, unit tests, format)  → reasoning
      ▼
Aligned / reasoning policy  →  evaluate (held-out, multi-judge, contamination-controlled)
```

### Key knobs
- **KL coefficient β** (PPO/online): the leash. Too low → reward hacking + mode collapse; too high → no
  movement off the SFT policy. Tune against measured KL, not blindly.
- **PPO**: clip ratio, GAE λ, value-loss coefficient, reward/advantage normalization, KL-estimator choice,
  rollout batch / mini-epochs. The critic init is a common failure.
- **DPO family**: β (implicit reward temperature), reference model choice; IPO swaps the loss to curb
  overfit; KTO needs only binary labels; ORPO drops the reference (single-stage with SFT); SimPO is
  reference-free + length-normalized.
- **GRPO/RLVR**: group size G, reward shaping (correctness + format), KL term on/off, verifier strictness;
  PRM (per-step) vs ORM (final-answer) reward.
- **Data**: preference quality/diversity, annotator agreement, on-policy freshness (iterative DPO
  regenerates pairs from the current policy to fight distribution shift).

### Failure modes to watch (map to §6 of the guide)
- **Reward hacking / overoptimization** — proxy reward keeps rising while true quality falls past some KL;
  mitigate with KL leash, RM ensembles, early stop on a held-out judge, verifiable rewards.
- **Length / format bias** — RMs and judges prefer longer, prettier answers; verbosity inflates. Use
  length-debiased or length-normalized rewards; audit response-length drift.
- **Sycophancy** — model agrees with the user because agreement was rewarded; measure explicitly.
- **Mode collapse / diversity & calibration loss** — aligned model is sharper, less diverse, worse
  calibrated than base. Track entropy/diversity.
- **Distribution shift** (offline) — DPO plateaus as the policy moves off the preference data; go iterative.
- **Verifier gaming** (RLVR) — degenerate outputs that pass the rule but aren't real solutions; tighten the
  verifier and reward format too.

---

## 3. "How to read & reproduce a paper" checklist

### Read (in this order — 20 minutes to a verdict)
1. **Abstract + figures + tables first.** What is the claimed effect and how big? Figures usually tell the
   real story faster than the prose.
2. **What's the actual novelty?** Read related-work to separate the new idea from the framing. Most papers
   change one thing — find it.
3. **Is the baseline honest and matched?** Same compute/data/tokens/model size? A win against an
   undertrained or under-tuned baseline is the #1 way results fail to replicate.
4. **Ablations.** Does each component's contribution get isolated? If the headline gain isn't ablated, be
   skeptical it comes from where they say.
5. **Eval integrity.** Contamination/decontamination addressed? Continuous vs exact-match metrics?
   Seeds/variance reported, or a single lucky run? Test-set tuning?
6. **Scale.** Does the effect hold across sizes, or only at one (possibly cherry-picked) scale? Expect
   small-scale tricks to wash out.
7. **Claims vs evidence.** Underline every claim the experiments don't actually support. Form your own
   verdict: *replicable*, *plausible-but-unproven*, or *likely artifact*.

### Reproduce (in this order — cheapest discriminating test first)
1. **Reproduce the BASELINE first** and confirm you can match its reported number. If you can't, fix that
   before touching the proposed method — otherwise your delta is noise.
2. **Smallest scale that should show the effect.** Don't start at the paper's largest run.
3. **Match the controls** from §1: tokens, data order, optimizer/schedule, batch, precision, decoding.
4. **Run multiple seeds**; establish your noise band before judging the delta.
5. **Reproduce the headline ablation**, not just the final number — that's where claims most often break.
6. **Then test the trend with scale** (one or two larger points) before believing it generalizes.
7. **Report honestly**: replicated / partially (effect smaller) / failed-to-replicate, with the conditions.
   A clean negative reproduction is a real research contribution.

### Red flags
No variance/seeds · single scale · unmatched/undertrained baseline · exact-match metric on a task with
near-ties · no decontamination · "we found that" with no ablation · improvements suspiciously near the
noise floor · hyperparameters tuned per-method only for the proposed method.
