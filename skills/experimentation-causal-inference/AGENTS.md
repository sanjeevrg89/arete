# AGENTS.md — Experimentation & Causal Inference

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`experimentation-causal-inference-guide.md`** next to this
> file — read it before designing, analyzing, or reviewing an experiment or a causal claim, and apply
> it. Concrete artifacts to imitate (a full design doc, an SRM + CUPED check, a diff-in-differences
> sketch) are in **`examples.md`**. This file is the always-on summary.
>
> **Core stance:** correlation does not establish causation. A trustworthy randomized experiment is the
> gold standard; when you cannot randomize, you must *explicitly argue identification*, not point at a
> chart. A wrongly-run experiment is worse than none. Trust before cleverness. This is a fast-moving
> applied field — **verify tooling, defaults, and library APIs against current sources**; never fabricate
> win-rate or variance-reduction numbers.

## When establishing whether a change CAUSED an outcome, apply by default:

- **Pre-register before looking at data:** hypothesis (direction + mechanism), the single **OEC**,
  **guardrail** metrics, segments, and the decision rule. Anything decided after seeing results is
  **HARKing** — disallowed.
- **Choose the OEC well:** sensitive, hard to game, a leading indicator of long-term value. Reject
  gameable OECs (raw clicks, revenue-per-session). Guardrails (latency, crashes, revenue, opt-outs)
  must not regress, even if the OEC wins.
- **Power the test first.** Set MDE (practical significance), α (≈0.05), power (≈0.80); derive sample
  size (`n ≈ 16σ²/Δ²` rule of thumb) and runtime in whole weeks. Underpowered ⇒ noise + inflated/
  wrong-sign effects (Type M/S).
- **Randomization unit = the coarsest unit that kills interference/carryover and matches the experience
  boundary — usually the user**, sticky via hashed assignment. Analysis unit must be nested in it; use
  delta method / bootstrap for ratio metrics.
- **SRM is the #1 trust check — run it FIRST.** Chi-square on arm counts; p < ~0.001 ⇒ broken
  experiment ⇒ debug, do NOT read metrics. Check on key subpopulations too.
- **Statistics:** a p-value is not P(null) and a 95% CI is not "95% chance the value is inside." Report
  effect + CI. Control multiple comparisons (Benjamini-Hochberg FDR); separate the one decision metric
  from the exploratory scorecard.
- **No peeking** on fixed-horizon tests. To stop early, use **sequential / always-valid inference
  (mSPRT, confidence sequences, group-sequential alpha-spending)**. State which method was used.
- **Reduce variance with CUPED** (pre-period covariate, unbiased, var × (1−ρ²)). Analyze the
  **triggered** population to avoid dilution. Covariates must be pre-treatment.
- **Twyman's law:** surprising/too-good figures are usually instrumentation or SRM bugs — investigate,
  don't celebrate. Watch Simpson's paradox during ramp-up; report percentiles (not means) for latency.
- **Interference (SUTVA violation)** in marketplaces/social graphs/shared budgets breaks naive A/B → use
  **cluster / switchback / budget-split** designs; state the estimand. Account for **novelty/primacy &
  carryover** (run long enough; re-randomize between experiments).
- **At scale:** layered/overlapping experiments (orthogonal hashes; interacting changes share a layer),
  staged **ramp-up** (1%→5%→50%) gated on SRM+guardrails, continuous **A/A** validation, near-real-time
  guardrail monitoring with **auto-shutoff**.
- **When you can't randomize, name the identification strategy + assumptions:** DiD (parallel trends),
  RD (no manipulation at cutoff), IV (relevance + exclusion), PSM/IPW (observed confounders only — no
  protection against unobserved), synthetic control (one treated unit, long pre-period). State the
  threats each design does NOT address.
- **Targeting:** estimate **CATE/uplift** (meta-learners S/T/X/R, causal forests, double ML); evaluate
  with **Qini/uplift curves**, not accuracy. Cleanest from randomized data.
- **ML/recsys:** offline AUC/nDCG only proposes a launch; the **A/B decides** (offline-online gap is
  real). Interleaving is a sensitive pre-filter, not the decision.

## Hard "do not" list
Do NOT: peek/stop fixed-horizon tests early; HARK; read metrics under SRM; ship on a gameable OEC;
report means for latency; treat naive A/B as valid under interference; or claim causality from
observational data without an identification argument. PSM ≠ randomization.

## Reviewing an experiment / causal claim
Check, in order: pre-registration & OEC quality → power/MDE → randomization unit & assignment → **SRM**
→ correct stats (CIs, multiple-comparisons, fixed vs sequential) → CUPED/triggering → interference &
novelty → segment/Simpson sanity. For observational claims: is there an identification strategy, and are
its assumptions stated and checked? Full checklist depth in the guide.
