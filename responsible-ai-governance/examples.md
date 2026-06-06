# Responsible AI Governance — Worked Examples

Three artifacts to imitate: (1) a filled-in **model card** skeleton, (2) a **fairness slice
evaluation** report sketch, and (3) an **AI risk register** entry mapped to the NIST AI RMF functions.
Numbers below are **illustrative placeholders** — replace with your real, dated measurements. Any
regulatory reference is marked *verify current*.

---

## 1. Model card (filled-in skeleton)

Structure follows *Model Cards for Model Reporting* (Mitchell et al., arXiv:1810.03993). Keep it as
versioned markdown/YAML next to the model; regenerate on retrain.

```markdown
# Model Card: loan-default-risk-v3.2

## Model details
- Owner / contact:        Credit Risk ML team — risk-ml@example.com
- Version / date:         3.2.0 — 2026-05-20  (supersedes 3.1.0)
- Type / architecture:    Gradient-boosted trees (binary classifier), 142 features
- Training framework:     scikit-learn / XGBoost; pipeline run abc123 (see lineage)
- License / restrictions: Consumer credit-decisioning only; not for marketing or pricing
- Citation / lineage:     git@.../models/loan-default-risk @ a1b2c3d; dataset DS-credit-2026Q1

## Intended use
- Primary use:    Assist (not replace) human adjudication of consumer loan applications.
- Primary users:  Credit adjudication officers, with human-in-the-loop sign-off on declines.
- Out-of-scope / misuse:  NOT for fully-automated decisions without human review; NOT for
                  employment, insurance, or any non-credit decision; NOT valid outside the
                  US consumer-loan population it was trained on.

## Factors
- Relevant groups:   Evaluated across (and at intersections of) protected attributes per
                     applicable fair-lending law (verify current). Proxies handled with care.
- Instrumentation / environment:  Online scoring at decision time; tabular features from the
                     feature store (see [[data-engineering-feature-stores]]).

## Metrics (DISAGGREGATED — single headline numbers are insufficient)
- Decision threshold: 0.62 (chosen for target precision; see fairness report)
- Overall:   AUC 0.88 | precision 0.71 | recall 0.64 | approval rate 0.58
- By slice:  see companion fairness slice report (§2). Equal-opportunity (TPR) gap 0.012;
             selected fairness criterion = equal opportunity (rationale below).
- All metrics reported with 95% CIs; small slices flagged as low-confidence.

## Evaluation & training data
- Training: DS-credit-2026Q1 (datasheet: datasheets/ds-credit-2026Q1.md), N=1.2M, 2019–2025.
- Eval:     held-out temporal split (2025-Q4..2026-Q1) to test for drift.
- Known gaps: thin-file applicants underrepresented; label = "90+ days delinquent" (a proxy —
             see measurement-bias note in datasheet).

## Ethical considerations
- Fairness criterion chosen: EQUAL OPPORTUNITY (equal TPR across groups) — deliberate choice,
  since the costed harm is a creditworthy applicant wrongly declined. Calibration and equalized
  odds cannot be jointly satisfied here (differing base rates; Kleinberg 2016 / Chouldechova
  2017); trade-off documented and reviewed by the AI review board on 2026-05-15.
- Feedback-loop risk: declines remove future repayment signal — monitored explicitly.

## Caveats & recommendations
- Re-measure fairness quarterly and on every retrain; alert on TPR-gap > 0.03.
- Human override required for all declines; log rationale (PII-safe).
- Re-validate before any expansion to a new population or product.
```

---

## 2. Fairness slice evaluation report (sketch)

Disaggregated, with confidence intervals, across protected **and intersectional** groups. Pair with
[[ml-evaluation-evals]] for methodology and [[ml-observability-monitoring]] for the production version.

```
Model:    loan-default-risk-v3.2     Threshold: 0.62     Eval set: 2025Q4–2026Q1 (N=84,210)
Fairness criterion (selected):  Equal opportunity (equal TPR).   Alert if any gap > 0.03.
Note: group labels used only for fairness measurement under documented governance; proxies avoided.

Slice (illustrative)        N        Approval   TPR     FPR     Precision   AUC    Flag
--------------------------  -------  ---------  ------  ------  ---------   -----  ----
Overall                     84,210   0.58       0.64    0.18    0.71        0.88
Group A                     61,400   0.60       0.65    0.17    0.72        0.89
Group B                     17,900   0.51       0.63    0.21    0.67        0.85
Group C                      4,910   0.49       0.58    0.24    0.63        0.81   low-N (wide CI)
Group B ∩ thin-file          2,030   0.41       0.55    0.27    0.59        0.78   ⚠ TPR gap 0.10

Reference (max−min) TPR gap across reported groups: 0.10  →  EXCEEDS 0.03 threshold  ⚠

Confidence: 95% CIs reported per cell (omitted here for brevity); slices with N<5k flagged as
low-confidence — do not over-interpret point estimates.

Findings:
- Headline TPR gap across top-level groups is 0.02 (within tolerance) BUT the intersection
  (Group B ∩ thin-file) shows a 0.10 TPR gap — aggregate and marginal views hid it. This is the
  whole point of intersectional slicing.
- Root-cause hypothesis: thin-file underrepresentation in training (label/representation bias,
  per datasheet), compounded for Group B.

Mitigations considered (pre / in / post-processing):
- Pre:  targeted data collection / reweighting for thin-file ∩ Group B.
- In:   fairness-constrained training (e.g., Fairlearn exponentiated-gradient on equal opportunity).
- Post: per-group threshold adjustment — REJECTED pending legal review (possible disparate
        treatment; verify current fair-lending law).
Decision: block release for this segment; collect data + retrain; re-run this report. Logged to
the risk register (R-014) and the AI review board.
```

---

## 3. AI risk register entry mapped to NIST AI RMF

One row per risk; map each to the NIST AI RMF function/category it sits under so coverage is auditable.
Maintain alongside the [[mlops-lifecycle]] and review at a defined cadence.

```
RISK ID:        R-014
System:         loan-default-risk-v3.2   (AI inventory: INV-0042)   Risk tier: HIGH
Owner:          Credit Risk ML lead (accountable) / Fairness reviewer (consulted)  [RACI]
Date opened:    2026-05-20     Review cadence: quarterly + on retrain     Status: OPEN-mitigating

NIST AI RMF mapping
  GOVERN:   GOVERN 1 (policies/accountability) — risk owner named; review-board gate at release.
            GOVERN 4 (culture/risk-aware) — fairness sign-off required pre-launch.
  MAP:      MAP 1 (context) — high-stakes consumer credit; affects applicants' access to credit.
            MAP 5 (impacts) — harm = creditworthy applicant wrongly declined (fairness + legal).
  MEASURE:  MEASURE 2.11 (fairness/bias) — equal-opportunity TPR gap; intersectional slices.
            MEASURE 4 (monitoring) — quarterly + drift-triggered re-measurement.
  MANAGE:   MANAGE 1 (prioritize/respond) — block affected segment; collect data + retrain.
            MANAGE 4 (monitor + incident) — prod alert on TPR-gap>0.03; rollback path defined.

Risk description: Intersectional fairness gap (Group B ∩ thin-file, TPR gap 0.10) — disparate
                  performance for an underrepresented segment; fair-lending exposure (verify current).
Likelihood: Medium    Impact: High    Inherent risk: HIGH
Mitigations:  (1) targeted data collection + retrain  (2) fairness-constrained training
              (3) human override mandatory on declines  (4) prod fairness-drift monitoring
Residual risk (post-mitigation, target): Medium — re-assess after retrain + re-run §2 report.
Regulatory note: Fair-lending / EU AI Act high-risk obligations may apply — VERIFY CURRENT; counsel
                 engaged 2026-05-18. Logging + human oversight maintained per high-risk requirements.
Links: model card (§1) · fairness report (§2) · datasheet DS-credit-2026Q1 · incident plan IR-AI-3
```

> The mapping keeps governance honest: every open risk traces to a Govern/Map/Measure/Manage control,
> an owner, a mitigation, and a residual-risk decision — exactly what an auditor or regulator will ask
> for. Don't let the register go stale; an unreviewed register is the same anti-pattern as measuring
> fairness once.
