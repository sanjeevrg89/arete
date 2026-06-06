---
name: adversarial-ml-robustness
description: Model-level adversarial machine learning and robustness — at the bar of a researcher who breaks
  and defends models for a living. Use when threat-modeling an ML/LLM model, evaluating or claiming
  robustness, or defending against evasion / adversarial examples (FGSM, PGD, C&W, transferable & patch/
  physical attacks), data poisoning, backdoors/trojans (clean-label, trigger-based), model extraction/
  stealing, model inversion, membership inference, or LLM jailbreaks / training-data extraction. Covers the
  NIST Adversarial ML taxonomy (AI 100-2e2025) and MITRE ATLAS; threat models (white/black/gray-box, Lp
  budgets); defenses and their limits (adversarial training, certified/randomized smoothing, why defensive
  distillation & gradient masking are false security); and the core skill — honest robustness evaluation
  with adaptive attacks, AutoAttack, and RobustBench. Defensive, not an attack playbook.
---

# Adversarial ML & Robustness

Apply the judgment of a researcher who breaks defenses for a living and therefore knows how to build and —
above all — **honestly evaluate** robust ones. This is a **defensive** skill: you understand attacks in
order to defend models and to refuse inflated robustness claims. The hardest and most valuable part is
**evaluation** — most published "robust" defenses were broken because they were tested against weak,
non-adaptive attacks. Robustness is **always relative to a stated threat model**; a claim without one is
meaningless.

Boundary: this is **model-level adversarial ML** (perturbations, poisoning, extraction/inversion, membership
inference, jailbreaks-as-evasion, robustness evaluation). Cloud/runtime/infra security, guardrails,
sandboxing, and supply-chain *enforcement* live in `[[ai-security-on-gke]]`. They compose: this skill says
*what* to defend against; that one says *where in the platform* to enforce it.

## How to use this skill

1. **Read `adversarial-ml-robustness-guide.md`** in this directory — the full reference (taxonomy/frameworks,
   attack categories, defenses and their limits, evaluation methodology, production, anti-patterns). Apply it
   to the task.
2. For an **adaptive robustness-evaluation checklist (with AutoAttack)**, a **backdoor/poisoning detection
   note**, and a **threat-model template**, read **`examples.md`**.
3. Match the surrounding codebase/eval-harness conventions; apply the correctness and evaluation-integrity
   rules regardless. The field moves fast (2026): treat attack/defense names, library APIs, leaderboard
   numbers, and arXiv IDs as **verify against current docs**.

## Essentials (full detail in `adversarial-ml-robustness-guide.md`)

- **No threat model, no robustness claim.** Pin down attacker knowledge (white/gray/black-box), goal
  (targeted/untargeted), capability/budget (`Lp` norm + `ε`, or semantic/physical), and access (query rate,
  pipeline/weight access) *before* evaluating. Evaluate the white-box worst case.
- **Anchor on the standard taxonomies.** Use **NIST AI 100-2e2025** vocabulary and map adversary activity to
  **MITRE ATLAS** tactics/techniques so reports are unambiguous and threat-modeling is systematic.
- **Know the attack families to defend them.** Evasion/adversarial examples (FGSM → PGD → C&W,
  transferability, patch/physical); poisoning & **backdoors/trojans** (clean-label, trigger-based);
  **model extraction**, **model inversion**, **membership inference** (`[[privacy-preserving-ml]]`); LLM
  **jailbreaks-as-evasion**, prompt injection (`[[ai-security-on-gke]]`, `[[llm-app-agent-frameworks]]`),
  training-data extraction/memorization (`[[pretraining-data-tokenizers]]`).
- **The only durable empirical defense is adversarial training** (Madry/PGD) — and it costs 3–30× training
  time and clean accuracy, and is budget-specific. State the budget.
- **Certified robustness gives a guarantee — but only inside its ball.** Randomized smoothing (`L2`
  probabilistic certificate, scales) and IBP/verifiers (deterministic, conservative). Report certified
  radius/accuracy; outside the ball you have *no* guarantee.
- **Gradient masking is false security.** Defensive distillation and most input-transform/detection defenses
  lower the *measured* attack, not the real one, and fall to adaptive attacks (BPDA/EOT/transfer). Run the
  gradient-masking sanity checks; treat transforms/detectors as defense-in-depth only.
- **Adaptive attacks are non-negotiable.** The attacker knows your defense. The minimum bar for any empirical
  robustness claim is **AutoAttack** (Croce & Hein, parameter-free ensemble) **plus a defense-aware adaptive
  attack** — AutoAttack alone can still over-estimate a quirky defense.
- **Use RobustBench** for a strong pretrained baseline and to sanity-check your number against SOTA for the
  *same* threat model. Beating SOTA by a wide margin = suspect your evaluation first.
- **Report clean AND robust accuracy** with the exact threat model; surface the **robustness–accuracy
  tradeoff** and the operating point. Distinguish *empirical* ("couldn't break it") from *certified*
  ("provably unbreakable in this ball").
- **Defend the whole pipeline, not just the model.** Poisoning/backdoors enter via **third-party datasets and
  pretrained weights** — verify provenance, dedup, scan, prefer `safetensors`, and trigger-probe before
  trusting a checkpoint (`[[pretraining-data-tokenizers]]`, enforcement in `[[ai-security-on-gke]]`).
- **Monitor and plan IR.** Watch for query/input/output attack signatures (`[[ml-observability-monitoring]]`);
  rate-limit and coarsen outputs to raise extraction/inversion cost; keep model/dataset lineage so you can
  scope and roll back a poisoning incident.
- **Red-team, but don't mistake it for proof.** Structured adversarial probing feeds defense; "no findings"
  is not robustness. Coordinate with `[[ml-evaluation-evals]]` and `[[responsible-ai-governance]]`.
- **Anti-patterns:** no threat model; weak/non-adaptive eval; claiming robustness without AutoAttack +
  adaptive attack; gradient-masking defenses; ignoring poisoning in third-party data/weights; `Lp`-only
  thinking for patch/physical/text threats; reporting only one of clean/robust accuracy.

## Related skills

- `[[ai-security-on-gke]]` — the platform/runtime side: prompt-injection & jailbreak filtering, sandboxing,
  guardrails, supply-chain *enforcement* (signing, `safetensors`). This skill = the model-level adversary.
- `[[privacy-preserving-ml]]` — DP-SGD and the principled defense against membership inference / model
  inversion / training-data extraction.
- `[[ml-evaluation-evals]]` — eval harness and methodology that robustness/red-team evaluation plugs into.
- `[[responsible-ai-governance]]` — policy, model cards/risk docs, and sign-off for red-teaming results.
- `[[llm-app-agent-frameworks]]` — building the agentic apps where jailbreaks/prompt injection land.
- `[[pretraining-data-tokenizers]]` — data dedup/curation; the poisoning/memorization surface of web corpora.
- `[[ml-observability-monitoring]]` — wiring up attack-signature and distribution-shift monitoring.
</content>
