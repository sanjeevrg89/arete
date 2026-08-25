---
name: responsible-ai-governance
description: Responsible AI governance, safety, fairness, and compliance — the discipline of building AI that is fair, safe, accountable, transparent, and auditable. Use when setting up AI governance (NIST AI RMF Govern/Map/Measure/Manage, EU AI Act risk tiers, ISO/IEC 42001, AI inventory, risk register, review board); writing transparency artifacts (model cards, datasheets for datasets, system cards, data statements, lineage); doing fairness/bias work (data/label/feedback bias, demographic parity vs equalized odds vs calibration and their impossibility, slice-based eval, pre/in/post-processing mitigation); LLM safety (harms taxonomy, red-teaming, jailbreak/misuse resistance, guardrails, hallucination/groundedness, safety evals, refusal/over-refusal, RLHF/Constitutional AI); privacy & data governance (PII, consent, data minimization, differential privacy, federated learning, machine unlearning / right-to-be-forgotten, training-data provenance & copyright); or accountability/ops (human oversight, AI incident response, audit logging, SHAP/LIME explainability and its limits, fairness-drift monitoring). The governance/ethics layer; adversarial/runtime security is the sibling [[ai-security-on-gke]].
---

# Responsible AI Governance

Apply the judgment of a head of responsible AI / ML risk who has shipped governed systems, sat on a
review board, run red-team exercises, and answered to auditors and regulators. Governance is a
**continuous control loop**, not a launch-day checkbox. This is the fairness/safety/accountability/
compliance layer; adversarial and runtime security is the sibling concern [[ai-security-on-gke]].

> **The law and the standards move fast (it is 2026).** Never quote an EU AI Act date, threshold, tier,
> or penalty — or any other regulatory number — from memory. Flag it "verify current" and check the
> primary source. Engineers are not lawyers; loop in counsel for anything with regulatory teeth.

## How to use this skill

1. **Read `responsible-ai-governance-guide.md`** in this directory — the full reference (frameworks,
   documentation, fairness, LLM safety, privacy, accountability, anti-patterns). Apply it to the task.
2. For artifacts to imitate, read **`examples.md`**: a filled-in model-card skeleton, a fairness slice
   report, and an AI risk register mapped to NIST AI RMF functions.
3. Match the org's existing framework and conventions; apply the correctness/safety/fairness rules and
   the "verify current" discipline regardless.

## Essentials (full detail in `responsible-ai-governance-guide.md`)

- **Govern / Map / Measure / Manage.** Use NIST AI RMF as the organizing spine. Govern = policy, roles,
  accountability; Map = context & risk; Measure = quantify (fairness, safety, privacy); Manage =
  mitigate, decide, monitor, respond. Embed it in the ML lifecycle ([[mlops-lifecycle]]), not beside it.
- **You can't govern what you can't enumerate.** Stand up an **AI inventory** + **risk register** + a
  proportional **review board** with a named accountable owner per system. RACI, not "everyone."
- **Frameworks:** NIST AI RMF (voluntary, the lingua franca); **EU AI Act** (law, risk-tiered:
  unacceptable/high/limited/minimal + GPAI duties — *verify current*); **ISO/IEC 42001** (certifiable AI
  management system).
- **Documentation is auditability.** Ship a **model card** (intended use, out-of-scope/misuse, metrics
  **disaggregated by slice**, caveats — arXiv:1810.03993) and a **datasheet** for each dataset
  (arXiv:1803.09010). Version them; regenerate on retrain; tie to lineage/provenance.
- **There is no single "fair."** Name the bias source (historical/sampling/label/aggregation/
  feedback-loop). Demographic parity, equalized odds, and calibration are **mathematically incompatible**
  when base rates differ (Kleinberg 2016; Chouldechova 2017) — choose the criterion deliberately from the
  harm and document the trade-off.
- **Measure fairness sliced, not aggregate, and never once.** Disaggregated metrics across protected and
  **intersectional** groups with CIs; mitigate pre/in/post-processing; monitor for fairness drift
  ([[ml-observability-monitoring]], [[ml-evaluation-evals]]).
- **LLM safety is measurable.** Build evals against an explicit **harms taxonomy**; **red-team** every
  generative system (manual + automated, logged as evidence); layer **guardrails** (probabilistic — never
  the whole strategy); treat **hallucination/groundedness** as a safety harm; balance **refusal vs
  over-refusal**. Alignment (RLHF / Constitutional AI — [[rl-rlhf-frameworks]]) is one layer, not a cure.
- **Privacy & data governance:** PII minimization (scrub LLM logs); differential privacy (ε budget) and
  federated learning at concept level; **right-to-be-forgotten / unlearning is an unsolved problem** —
  design for it via lineage and retention; training-data provenance & copyright are **unsettled law —
  verify current**.
- **Accountability:** meaningful **human oversight** (beware automation bias); an **AI incident plan**
  with kill switch/rollback/notification *before* you need it; **audit logging** without creating new PII
  liability; **explainability** (SHAP/LIME/IG) with honest limits — approximations, weak for LLMs, ≠ truth.
- **Avoid the anti-patterns:** ethics-as-a-final-checkbox, no model card, fairness measured once,
  aggregate-only metrics, no red-team, no incident plan, ignoring evolving regulation, and "safety is
  someone else's job."

## Related skills

- `[[ai-security-on-gke]]` — sibling: adversarial/runtime security (prompt injection, jailbreak-as-attack,
  model theft, infra hardening). Governance and security overlap and should share red-team findings.
- `[[ml-evaluation-evals]]` — slice-based evaluation, safety/groundedness evals, benchmark contamination.
- `[[ml-observability-monitoring]]` — production monitoring for fairness/safety drift and data drift.
- `[[mlops-lifecycle]]` — where governance, lineage, and gates live in the model lifecycle.
- `[[rl-rlhf-frameworks]]` — RLHF / Constitutional AI / DPO alignment mechanics.
- `[[llm-app-agent-frameworks]]` — where guardrails, system prompts, and tool-use constraints are wired.
- `[[data-engineering-feature-stores]]` — data provenance/lineage feeding governance artifacts.
- `[[rag-vector-databases]]` — retrieval grounding as a hallucination/safety mitigation.
