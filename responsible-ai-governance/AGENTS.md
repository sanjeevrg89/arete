# AGENTS.md — Responsible AI Governance, Safety & Compliance

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`responsible-ai-governance-guide.md`** next to this file —
> read it before doing governance, fairness, safety, privacy, or compliance work, and apply it. Artifacts
> to imitate (model card, fairness slice report, NIST-mapped risk register) are in **`examples.md`**.
> This is the always-on summary. This is the governance/ethics layer; adversarial/runtime security is the
> sibling **`[[ai-security-on-gke]]`**.
>
> **Regulation and standards move fast (2026).** Never quote an EU AI Act tier, date, threshold, or
> penalty — or any regulatory number — from memory. Mark it "verify current" and check the primary
> source. Do not fabricate regulation details. Engineers are not lawyers; defer to counsel.

## Apply by default on any responsible-AI / governance / safety / compliance task:

- **Govern / Map / Measure / Manage** (NIST AI RMF) is the spine. Treat governance as a continuous
  control loop embedded in the ML lifecycle, **not** a launch-day checkbox.
- **Stand up the basics:** an **AI inventory**, a **risk register** (map risks to NIST functions), a
  **proportional review board**, and a **named accountable owner per system** (RACI). You cannot govern
  what you cannot enumerate.
- **Frameworks:** NIST AI RMF (voluntary, free, the common language) · **EU AI Act** (law; risk tiers
  unacceptable/high/limited/minimal + GPAI duties — *verify current*) · **ISO/IEC 42001** (certifiable
  AI management system).
- **Documentation = auditability.** Produce a **model card** (intended use, **out-of-scope/misuse**,
  metrics **disaggregated by slice**, caveats — arXiv:1810.03993) and a **datasheet for datasets**
  (arXiv:1803.09010). Version them, regenerate on retrain, link to lineage/provenance.
- **No single definition of "fair."** Name the bias source (historical / sampling / **label** /
  aggregation / feedback-loop). Demographic parity, equalized odds, and calibration are **mathematically
  incompatible** when base rates differ (Kleinberg 2016; Chouldechova 2017). Choose the criterion
  deliberately from the harm; document the trade-off in the model card.
- **Fairness is sliced and continuous.** Disaggregated metrics across protected **and intersectional**
  groups with confidence intervals — never aggregate-only, never measured once. Mitigate
  pre/in/post-processing; monitor for **fairness drift** in prod (`[[ml-observability-monitoring]]`,
  `[[ml-evaluation-evals]]`).
- **LLM safety is a first-class, measurable property.** Build evals against an explicit **harms
  taxonomy**; **red-team every generative system** (manual + automated, logged as evidence); layer
  **guardrails** (probabilistic — complement, never replace, alignment + oversight); treat
  **hallucination/groundedness** as a safety harm; balance **refusal vs over-refusal**. Alignment
  (RLHF / Constitutional AI — `[[rl-rlhf-frameworks]]`) is one layer, not a cure. Distrust public
  benchmarks (leak/saturate) — keep private eval sets.
- **Privacy & data governance:** minimize PII and **scrub LLM logs**; document consent/lawful basis
  (*verify current*); differential privacy (ε budget) and federated learning at concept level;
  **right-to-be-forgotten / machine unlearning is unsolved** — design for it via lineage + retention,
  don't assume you can "untrain"; training-data provenance & **copyright are unsettled law — verify
  current**.
- **Accountability & ops:** **meaningful human oversight** (guard against automation bias); an **AI
  incident plan** with kill switch / rollback / notification **before** an incident; **audit logging**
  that doesn't create new PII liability; **explainability** (SHAP / LIME / integrated gradients) with
  honest limits — they are approximations, weak/unfaithful for LLMs, and ≠ a causal account.
- **Never** treat ethics as a final checkbox, ship without a model card, measure fairness once, report
  aggregate-only metrics, ship a generative system without a red-team, operate without an incident plan,
  ignore evolving regulation, or assume safety is someone else's job.

## Definition of done for a governed AI system
A model card + datasheet exist and are current · risk tier and risk-register entry assigned with an owner
· fairness evaluated sliced (protected + intersectional) and scheduled for re-measurement · for
generative systems, a red-team was run and findings tracked, with guardrails + safety evals in place ·
human oversight and an AI incident/rollback plan are defined · audit logging is on (PII-safe) ·
production monitoring covers fairness/safety drift · any regulatory claim is dated and marked "verify
current."

## Sibling boundary
Fairness/safety/ethics/compliance = here. Prompt injection, jailbreak-as-attack, model/data
exfiltration, and infra hardening = **`[[ai-security-on-gke]]`**. Both are required; share red-team
findings across them.
