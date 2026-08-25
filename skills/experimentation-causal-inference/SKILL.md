---
name: experimentation-causal-inference
description: World-class online controlled experiments (A/B testing) and causal inference — the
  gold-standard methodology for data-driven product and ML decisions, run at 20,000+ experiments/year by
  the largest tech companies. Use whenever you need to establish that a change CAUSED an outcome:
  designing an A/B test (OEC/overall evaluation criterion, guardrail metrics, hypothesis, MDE & power
  analysis, sample size, randomization unit — user vs session vs request), doing the statistics right
  (p-values & confidence intervals and their misinterpretation, multiple comparisons/FDR, sequential
  testing / always-valid inference / mSPRT, peeking, variance reduction with CUPED), avoiding the
  pitfalls (Sample Ratio Mismatch/SRM as the #1 trust check, Simpson's paradox, network/interference
  effects with cluster/switchback/budget-split designs, primacy/novelty & carryover, dilution,
  segment heterogeneity, Twyman's law), running experimentation at scale (platforms, overlapping/layered
  experiments, ramp-up, A/A tests, near-real-time monitoring & auto-shutoff), and causal inference when
  you can't randomize (diff-in-differences, regression discontinuity, instrumental variables,
  propensity-score matching, synthetic control, uplift/CATE modeling with causal forests & meta-learners).
  Distinct from model evaluation — see [[ml-evaluation-evals]].
---

# Experimentation & Causal Inference

Apply the judgment of someone who has run a large-scale experimentation platform and an applied causal
inference practice for years. The core belief: **correlation is cheap and usually wrong about
direction; the only reliable way to know whether a change helped is a trustworthy randomized
experiment — and when you cannot randomize, you must explicitly argue identification, not wave at a
chart.** An experiment that is run wrong is worse than no experiment, because it launders a guess into
a "result." Trust comes first; cleverness second.

## How to use this skill

1. **Read `experimentation-causal-inference-guide.md`** in this directory — the full reference
   (potential outcomes & ATE, designing a trustworthy experiment, statistics done right, the pitfalls
   and Twyman's law, experimentation at scale, quasi-experiments & causal inference, uplift/CATE,
   application to ML/recsys). Apply it to the task at hand.
2. For concrete artifacts to imitate — a full experiment design doc (OEC + guardrails + power +
   randomization unit), an SRM check with a CUPED note, and a difference-in-differences sketch for a
   non-randomizable rollout — read **`examples.md`**.
3. Match the surrounding org's metric definitions and platform conventions; apply the trust rules
   (SRM check, pre-registration, correct randomization unit, no peeking without sequential methods)
   regardless of local habit.

## The essentials (full rationale in the guide)

- **Define the OEC before you look at data.** One Overall Evaluation Criterion that is sensitive,
  hard to game, and a leading indicator of long-term value (not raw revenue, not a single-day click).
  Pre-register hypothesis, OEC, guardrails, segments, and decision rule. Anything decided after seeing
  results is HARKing.
- **Guardrail metrics protect the business** while you optimize the OEC: latency, crash rate, revenue,
  unsubscribes. A move that wins on OEC but breaks a guardrail does not ship.
- **Pick the randomization unit deliberately** — almost always the *user* (stable, captures the full
  experience, avoids carryover). Session/request units leak treatment across the same user and
  understate variance. The analysis unit must match (or be nested in) the randomization unit.
- **Power the test first.** Compute the Minimum Detectable Effect, choose α and power (commonly 0.05 /
  0.80), and derive sample size and runtime *before* launch. Underpowered tests produce noise and
  inflated "winners" (Type M/S errors).
- **SRM is the #1 trust check.** If the observed treatment/control split deviates from the designed
  ratio (chi-square p < ~0.001), the experiment is broken — stop and debug; do not read the metrics.
- **A p-value is not P(no effect) and a 95% CI is not "95% probability the true value is inside."**
  Report effect sizes with CIs; correct for multiple comparisons (FDR/Benjamini-Hochberg).
- **Do not peek at fixed-horizon tests.** Continuous monitoring with fixed-α inflates false positives
  badly. If you need to stop early, use **sequential / always-valid inference (mSPRT, group-sequential,
  confidence sequences)**.
- **Reduce variance with CUPED** (use pre-period covariates) to cut required sample size — often a large
  reduction — without bias. Trigger-based analysis (dilution) also sharpens sensitivity.
- **Watch for interference.** In marketplaces, social graphs, and shared-resource systems, SUTVA is
  violated; use **cluster / switchback / budget-split** designs and interpret naive A/B with suspicion.
- **Run A/A tests** to validate the platform (false-positive rate ~ α, no SRM, correct CIs) before
  trusting A/B results. Ramp up exposure (1% → 5% → 50%) to limit blast radius.
- **Twyman's law:** any figure that looks too good or too surprising is probably wrong — investigate
  instrumentation, logging, and SRM before celebrating.
- **When you cannot randomize, name the identification strategy** (DiD, RD, IV, PSM, synthetic control)
  and its assumptions explicitly. Observational "lift" without an identification argument is not causal.

## Related skills

- `[[ml-evaluation-evals]]` — offline/online quality measurement of models; this skill is the
  decision/causality layer above it (does the better-on-eval model actually move the business metric?).
- `[[recsys-ranking]]` — ranking/model changes are validated by online experiments; the offline-online
  gap and interleaving live here.
- `[[ml-observability-monitoring]]` — near-real-time experiment monitoring, metric pipelines, and
  auto-shutoff hook into the same telemetry.
- `[[ml-system-design]]` — experimentation platform as a system; metric stores, assignment service.
- `[[data-engineering-feature-stores]]` — pre-period covariates for CUPED and metric computation come
  from the same feature/metric data plane.
- `[[responsible-ai-governance]]` — guardrails, fairness slices, and decision accountability for
  launches.
