# AGENTS.md — Adversarial ML & Robustness

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`adversarial-ml-robustness-guide.md`** next to this file — read it
> before evaluating, claiming, or defending model robustness, and apply it. Concrete artifacts to imitate
> (adaptive eval checklist, backdoor-detection note, threat-model template) are in **`examples.md`**.
> This file is the always-on summary.
>
> **This is a DEFENSIVE skill**: understand attacks to defend models and to refuse inflated robustness
> claims — not to build attack tooling. **Model-level** adversarial ML; platform/runtime enforcement and
> guardrails live in `[[ai-security-on-gke]]`. The field moves fast (2026) — verify attack/defense names,
> library APIs, leaderboard numbers, and arXiv IDs against current upstream docs.

## When the task touches model robustness / adversarial ML, apply by default:

- **No threat model → no robustness claim.** State attacker **knowledge** (white/gray/black-box), **goal**
  (targeted/untargeted), **capability/budget** (`Lp` norm + `ε`, or semantic/physical), **access** (query
  rate, pipeline/weight access). Evaluate the **white-box worst case**. Use **NIST AI 100-2e2025** vocabulary
  and map activity to **MITRE ATLAS** techniques.
- **Adaptive attacks are mandatory.** The attacker knows the defense. Minimum bar for an empirical claim:
  **AutoAttack** (Croce & Hein, parameter-free ensemble) **+ a defense-aware adaptive attack**. AutoAttack
  alone can over-estimate a quirky defense. Use BPDA for non-differentiable steps, EOT for randomized
  defenses, plus transfer/query-based attacks.
- **Detect gradient masking — it is the #1 cause of fake robustness.** Red flags: black-box beats white-box;
  unbounded attack doesn't reach ~100% success; bigger `ε` doesn't raise attack success; random sampling
  finds adversarials PGD missed; more PGD steps don't help. If any hold, the number is fake. **Defensive
  distillation and most input-transform/detector defenses are gradient masking** — defense-in-depth only.
- **Report clean AND robust accuracy** with the **exact** threat model (norm, `ε`, dataset, attack + steps +
  restarts). Surface the **robustness–accuracy tradeoff** and operating point. Distinguish **empirical**
  ("couldn't break it") from **certified** ("provably safe in this ball, e.g. randomized smoothing").
- **Real defenses:** **adversarial training** (Madry/PGD) is the durable empirical one — costs 3–30× compute
  + clean accuracy, and is budget-specific. **Certified**: randomized smoothing (`L2`, scales) / IBP
  (conservative) — valid **only inside the certified ball**. Sanity-check against **RobustBench** for the
  same threat model; a huge SOTA beat means suspect your eval.
- **Defend the pipeline, not just the model.** **Poisoning & backdoors/trojans** (clean-label, trigger-based)
  enter via third-party **datasets and pretrained weights** — verify provenance, dedup, scan, trigger-probe,
  prefer `safetensors` over pickle (`[[pretraining-data-tokenizers]]`; enforcement in `[[ai-security-on-gke]]`).
- **Privacy attacks** (membership inference, model inversion, training-data extraction/memorization): audit
  with strong attacks (LiRA-style, TPR@low-FPR); principled defense is **DP-SGD** (`[[privacy-preserving-ml]]`).
- **Extraction/inversion at inference:** rate-limit, coarsen outputs (top-1, no logits), monitor query
  distributions; these raise attacker cost, not impossibility.
- **LLMs:** jailbreaks = evasion (and transfer across models); prompt injection is an app/architecture problem
  (`[[ai-security-on-gke]]`, `[[llm-app-agent-frameworks]]`); training-data extraction is memorization-driven.
- **Monitor + IR:** watch query/input/output attack signatures (`[[ml-observability-monitoring]]`); keep
  model/dataset lineage to scope and roll back poisoning incidents; document for `[[responsible-ai-governance]]`.
- **Red-teaming feeds defense but is not proof** — "no findings" ≠ robust. Coordinate with `[[ml-evaluation-evals]]`.

## Anti-patterns (reject these)
No threat model · weak/non-adaptive eval (FGSM-only, single low-step PGD) · robustness claim without
AutoAttack + adaptive attack · gradient-masking defenses (defensive distillation, obfuscated gradients) ·
input-transform/detector as the whole defense · ignoring poisoning in third-party data/weights / loading
pickle weights · reporting only clean or only robust accuracy · conflating empirical with certified ·
`Lp`-only thinking against patch/physical/text attacks · treating red-team "no findings" as a proof.
</content>
