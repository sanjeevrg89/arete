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

---

# Reference — experimentation-causal-inference

# Experimentation & Causal Inference — Full Reference

The discipline of establishing **causality** for product and ML decisions: online controlled
experiments (A/B tests) as the gold standard, the statistics to read them correctly, the pitfalls that
silently invalidate them, how to run thousands of them on a platform, and the quasi-experimental
methods to fall back on when randomization is impossible. This is a fast-moving applied field; the
*methods* below are stable, but tooling, defaults, and library APIs change — **verify specifics against
current sources** (see Canonical references).

---

## 1. Why randomized experiments are the gold standard

### The fundamental problem of causal inference
For a unit (a user) we can observe the outcome under treatment **or** under control, never both. The
*causal effect* for that unit is the difference between two **potential outcomes** Y(1) − Y(0), but one
of them is always counterfactual (Rubin's potential-outcomes / Neyman-Rubin causal model). We cannot
estimate an individual effect; we estimate a *population average*.

The **Average Treatment Effect** is `ATE = E[Y(1) − Y(0)]`. The naive observational comparison
`E[Y | treated] − E[Y | untreated]` equals the ATE *plus selection bias* (treated and untreated units
differ for reasons correlated with the outcome). That bias is the entire reason "users who used feature
X retain better" tells you almost nothing about whether X causes retention.

### What randomization buys you
Random assignment makes treatment **independent of all confounders, observed and unobserved**. In
expectation the two arms are identical on everything except the treatment, so the difference in means is
an unbiased estimate of the ATE. This is why a clean A/B test beats any observational analysis no matter
how sophisticated: it removes confounding by design rather than by assumption. Randomization is also
*humbling* — published industry experience repeatedly shows the majority of well-reasoned feature ideas
fail to move the OEC or move it negatively (the exact rate varies by domain and maturity; treat any
specific percentage as something to **verify against current sources**, e.g. Kohavi/Tang/Xu).

### Key assumption: SUTVA
The Stable Unit Treatment Value Assumption — (a) no interference: one unit's treatment does not affect
another unit's outcome; (b) one version of treatment. SUTVA is the assumption most often violated at
scale (marketplaces, social graphs, shared budgets/caches). See §6.

---

## 2. Designing a trustworthy experiment

Trust is the product. Design the experiment so that, whatever the result, you believe it.

### The hypothesis
A falsifiable statement with a direction and a mechanism: "Showing estimated delivery date on the
product page will *increase* completed checkouts because it reduces purchase uncertainty." Vague
hypotheses ("improve the experience") cannot be tested and invite HARKing.

### The OEC (Overall Evaluation Criterion)
The single metric (or a small weighted combination) you will optimize and use to make the ship/no-ship
decision. Choosing it well is the hardest, highest-leverage design decision.

A good OEC is:
- **Sensitive** — moves measurably within the experiment's runtime at achievable sample sizes.
- **Hard to game** — does not reward degenerate behavior. "Clicks" rewards clickbait; pair with
  downstream success (e.g. dwell, task completion, sessions/user) to penalize regret clicks.
- **A leading indicator of long-term value.** Long-term value (LTV, retention, lifetime revenue) is
  what you care about but is too slow and noisy to measure per-experiment. Use a *surrogate* validated
  against long-term outcomes (e.g. sessions-per-user, normalized engagement) — and revalidate the
  surrogate periodically.
- **Directionally unambiguous** — you know which way is good before you look.

Classic trap: optimizing **revenue-per-session** can be maximized by showing more ads / raising prices,
degrading the experience and long-term revenue. Prefer a structural OEC (e.g. active days, successful
tasks) with revenue and engagement as guardrails.

### Guardrail metrics
Metrics you are *not* trying to improve but refuse to harm: page-load latency, crash/error rate, total
revenue, opt-outs/unsubscribes, support contacts, key-page CTR. A treatment that wins on OEC but
regresses a guardrail beyond a threshold does not ship. Guardrails also include **trustworthiness**
guardrails (SRM, cache-hit parity) that signal a *broken* experiment rather than a bad feature.

### MDE, power, and sample size
- **Minimum Detectable Effect (MDE):** the smallest true effect worth detecting — set by *practical
  significance*, not statistics. A 0.01% lift may be worth shipping at scale, or not worth the
  complexity; that's a business call made up front.
- **Power (1 − β):** probability of detecting an effect of size MDE if it is real. Convention 0.80.
- **α:** false-positive rate, convention 0.05 (two-sided).
- **Sample size** per arm for a difference in means scales roughly as
  `n ≈ (z_{1−α/2} + z_{1−β})² · 2σ² / Δ²` (≈ `16 σ² / Δ²` for α=0.05, power=0.80 as a back-of-envelope).
  Halving the MDE quadruples the sample. Compute runtime = required n / daily eligible traffic, then
  **round up to whole weeks** to cover weekday/weekend seasonality.
- Underpowered experiments are doubly dangerous: low chance of detecting a real effect, and any
  "significant" result that does appear is inflated in magnitude and may have the wrong sign (Type M
  "magnitude" and Type S "sign" errors — Gelman).

### Randomization unit (and its consequences)
| Unit | When | Consequence |
|---|---|---|
| **User** (or device/cookie) | default | Stable experience, no within-user carryover, captures full journey. Most metrics' analysis unit ≠ randomization unit → need the **delta method / bootstrap** for variance (ratio metrics like CTR per user). |
| **Session** | rarely; stateless surface | Same user can land in both arms across sessions → dilutes effect, leaks treatment, underestimates variance. |
| **Request/page-view** | infra/latency-only changes invisible to users | Massive sample but tiny independent unit; never use for experience changes (user sees inconsistent UI). |
| **Cluster** (geo, time-slice, social community) | interference present | See §6. Far fewer effective units → much lower power. |

**Rule:** randomize on the *coarsest* unit that (a) eliminates interference/carryover and (b) matches
the experience boundary. The randomization unit must contain the analysis unit. Persist assignment
(hash userID + experiment salt) so a user's arm is sticky across sessions and deploys.

---

## 3. Statistics done right

### Significance testing, p-values, confidence intervals
- A **p-value** is `P(observe data this extreme or more | null hypothesis true)`. It is **not** the
  probability the null is true, **not** the probability your result is a fluke, and **not** the effect
  size. p = 0.049 and p = 0.051 are essentially the same evidence.
- A **95% confidence interval** is a procedure that covers the true value 95% of the time over repeated
  experiments. For *this* interval the true value is either in it or not; "95% probability it's inside"
  is the common (Bayesian-flavored) misinterpretation. Report the CI on the effect (and relative effect)
  always — it conveys precision and practical significance that a bare p-value hides.
- "Not significant" ≠ "no effect." It means you couldn't rule out zero at this sample size. Report the
  CI so readers see whether you ruled out a *meaningful* effect or just lacked power.

### Multiple comparisons & false discovery rate
Testing many metrics × many segments × many variants inflates false positives: at α=0.05, ~1 in 20
*null* tests "wins" by chance. Control it:
- **Bonferroni** (`α/m`) — simple, conservative; fine for a handful of pre-declared guardrails.
- **Benjamini-Hochberg FDR** — controls the expected *fraction* of false discoveries among rejections;
  the right default for scanning many metrics/segments. Decide the family of tests *before* looking.
- Separate the **decision metric (OEC)** — one pre-registered test, full α — from the **exploratory
  scorecard** — many metrics under FDR control, treated as hypothesis-generating, not decisions.

### Sequential testing / always-valid inference (the peeking problem)
Fixed-horizon tests are valid **only if you decide once, at the pre-set sample size.** Repeatedly
checking and stopping when p < 0.05 ("peeking") can push the real false-positive rate far above α
(continuous peeking drives it toward 1 in the limit). Two correct ways to look early:
- **Group-sequential** (O'Brien-Fleming / Pocock alpha-spending): pre-plan K interim looks, spend α
  across them.
- **Always-valid inference** — the **mixture Sequential Probability Ratio Test (mSPRT)** and
  **confidence sequences** give p-values/CIs valid at *every* sample size, so you can monitor
  continuously and stop whenever, at the cost of some efficiency vs. a correctly-sized fixed test. This
  is the modern default for "anytime" dashboards.
- **CUPED + sequential** compose. Always report whether a result used fixed-horizon or sequential
  methods — mixing the two (sequential monitoring, fixed-horizon p-value) is a silent error.

### Variance reduction: CUPED
**CUPED** (Controlled-experiment Using Pre-Experiment Data, Deng et al.) regresses out a pre-period
covariate `X` (most powerfully the *same metric* measured before the experiment) that is unaffected by
treatment:

```
Y_cuped = Y − θ (X − E[X]),   θ = Cov(Y, X) / Var(X)
```

`Y_cuped` has the same expectation (unbiased) but variance reduced by `1 − ρ²`, where ρ is the
pre/post correlation. High-correlation metrics can see large variance reductions → proportionally
smaller required sample / shorter runtime. Generalizations: regression adjustment / CUPAC with ML-
predicted covariates, stratification, post-stratification. Covariates must be **pre-treatment** —
adjusting on anything affected by treatment reintroduces bias.

### Triggering and dilution
Only a fraction of assigned users actually *encounter* the change (e.g. only users who hit the error
path see the new error page). Including untriggered users **dilutes** the effect toward zero and wastes
power. Analyze the **triggered population** (counterfactual triggering: log who *would* have triggered
in both arms to keep the comparison unbiased), then translate to the launch (all-up) impact for the
business decision.

### Non-normal & ratio metrics
Means of bounded/heavy-tailed metrics (revenue, latency) are skewed. Use the CLT with large n, but for
percentiles (p95 latency) the mean is the wrong summary — bootstrap or quantile-specific methods.
**Report percentiles, not means, for latency-type metrics.** For ratio metrics where analysis unit ≠
randomization unit, use the **delta method** or **bootstrap** for correct variance.

---

## 4. The pitfalls (Twyman's law)

**Twyman's law: any figure that looks interesting or surprising is usually wrong.** Big, exciting
results trigger a *more* skeptical investigation, not a celebration. Most "amazing" lifts are
instrumentation bugs, logging changes, or SRM.

### Sample Ratio Mismatch (SRM) — the #1 trust check
If you designed a 50/50 split but observe, say, 50.2% / 49.8% on millions of users, run a chi-square
goodness-of-fit test on the *counts*. **p < ~0.001 ⇒ SRM ⇒ the experiment is broken — do not interpret
any metric.** Common causes: assignment/bucketing bug, redirect or latency that drops one arm, bot
filtering applied asymmetrically, telemetry loss correlated with treatment, a join that silently drops
rows, ramp/restart timing. SRM is checked *first*, before any metric is read, and on every key
subpopulation.

### Simpson's paradox
A trend that holds in every subgroup can reverse in the aggregate when group sizes/mix differ between
arms (e.g. during a **ramp-up** where the treatment's traffic mix changes day to day). Don't pool across
periods with different allocation; weight segments consistently; be suspicious when aggregate and
per-segment directions disagree.

### Network / interference effects (SUTVA violation)
When one user's treatment affects another's outcome, naive A/B is biased — sometimes severely:
- **Marketplaces / shared budgets:** treating riders/sellers/bidders changes the pool for control too
  (cannibalization); the control is contaminated, biasing the estimate. → **budget-split** (separate
  budget/inventory per arm), **cluster** randomization by market/region, or **switchback** (alternate
  the whole system between treatment and control over time slices) experiments.
- **Social graphs:** a feature that increases sharing spills over to control friends. → **cluster
  randomization** on graph communities, **ego-cluster** designs, or **two-sided / graph-cluster**
  randomization. Pay the power cost (effective n ≈ number of clusters, not users).
- Detect interference by varying treatment *density* (e.g. randomize the fraction treated within
  clusters) and checking whether effect size depends on density.

### Primacy, novelty, carryover
- **Novelty:** users click a new thing because it's new; effect decays. **Primacy:** users are
  initially confused by a change; effect grows as they learn. Both mean the early effect ≠ the
  steady-state effect. → run long enough to see the curve flatten; segment by new vs. tenured users;
  inspect the daily effect trend, not just the pooled number.
- **Carryover:** a previous experiment's after-effects contaminate a unit's behavior in the next.
  Re-randomize users between experiments; use cool-down periods; A/A test the carryover.

### Segment heterogeneity
The ATE can hide large, opposite effects across segments (positive for mobile, negative for desktop;
positive for new users, negative for power users). Always cut the OEC by key pre-registered segments
(platform, country, tenure, traffic source) — but under multiple-testing discipline (§3), treating
post-hoc segment "wins" as exploratory, not decisions. This motivates CATE/uplift modeling (§7).

### Other recurring traps
- Reporting **means where percentiles matter** (latency).
- **Ratio-metric variance** computed as if the analysis unit were the randomization unit.
- **Outliers / bots** dominating revenue metrics → cap/winsorize (pre-declared) and filter bots
  symmetrically.
- **Missing-data / logging asymmetry** between arms (a frequent SRM cause).

---

## 5. Experimentation at scale

### Platform
A mature experimentation platform provides: an **assignment service** (deterministic hash of unit +
experiment salt → arm, sticky and consistent across services), a **metric/metrics framework** (curated,
versioned, owned metric definitions with agreed variance computation), a **scorecard** engine (effect +
CI + p, SRM check, segments, guardrails, CUPED applied automatically), and **diagnostics** (SRM, A/A,
data-quality alarms). Centralizing metric definitions is what makes thousands of experiments comparable
and trustworthy.

### Overlapping / layered experiments
You cannot give each experiment exclusive traffic at scale. Use **layers (a.k.a. domains/universes):**
orthogonal hash spaces so a user is independently randomized in each layer, letting many experiments run
concurrently on the same users without confounding (Tang et al., "overlapping experiment
infrastructure"). Experiments that *interact* (touch the same surface) go in the **same** layer (mutually
exclusive); independent ones go in **different** layers. Track interactions; flag and re-test when two
shipped changes are suspected to interact.

### Ramp-up
Increase exposure in stages — e.g. 1% → 5% → 20% → 50% — gated by guardrails and SRM at each step. Early
stages catch catastrophic bugs with small blast radius; 50/50 maximizes power for the final read. Don't
compare metrics *across* ramp steps (Simpson's paradox); read the effect within a stable-allocation
window. Maximum-power allocation for a two-arm test is 50/50.

### A/A tests
Two identical arms. The platform's continuous validation: the false-positive rate should be ≈ α, there
should be **no SRM**, p-values should be ~uniform, and CIs should have nominal coverage. A/A failures
reveal variance miscalculation, broken randomization, or bad metric pipelines — fix these before
trusting any A/B. Run A/A continuously as a canary.

### Near-real-time monitoring & auto-shutoff
Stream guardrails (crash rate, latency, revenue, error rate) in near-real-time so a clearly harmful
treatment is **auto-ramped-down** within minutes, not days. Distinguish *safety* monitoring (fast,
guardrail-triggered shutoff — false alarms acceptable) from the *decision* read (sequential or
fixed-horizon, careful). Hook this into the same telemetry as [[ml-observability-monitoring]].

---

## 6. (Interference design details — see §4 "Network effects")

Interference is important enough to restate the design menu compactly:

| Design | Idea | Use when | Cost |
|---|---|---|---|
| **User (Bernoulli)** | randomize users | no interference | baseline |
| **Cluster** | randomize groups (geo, market, community) | local interference within groups | power ∝ #clusters; needs many clusters |
| **Switchback** | toggle whole system T/C over time slices | strong marketplace/temporal interference, few clusters | temporal autocorrelation; needs careful slicing & analysis |
| **Budget-split** | separate inventory/budget per arm | shared-budget cannibalization (ads, supply) | half the budget per arm; can change dynamics |
| **Two-sided / graph-cluster** | randomize on both market sides or graph partitions | two-sided marketplaces, social spillover | complex; partial interference remains |

Always state the **estimand** explicitly: total treatment effect (everyone treated vs. no one) differs
from the direct effect, and naive A/B estimates neither when interference is present.

---

## 7. Causal inference when you can't randomize

Sometimes you cannot randomize: ethics, a one-time launch, a pre-existing rollout, a policy/pricing
change, or a competitor action. Then you must *argue identification* — make explicit the assumptions
under which a causal effect is recoverable from observational data (Pearl's do-calculus / back-door
criterion; Imbens & Rubin's potential-outcomes treatment). **Stating the assumptions and their threats
is the deliverable**, not the point estimate.

### Difference-in-Differences (DiD)
Compare the *change* over time in a treated group to the change in a comparable control group; the
double difference cancels fixed group differences and common time trends. **Key assumption: parallel
trends** — absent treatment, the two groups would have moved in parallel. Validate with pre-period
trends (event-study plot); beware staggered adoption (use modern estimators — Callaway-Sant'Anna,
Sun-Abraham — over naive two-way fixed effects, which can be biased with heterogeneous timing).
Good for geo rollouts and policy changes.

### Regression Discontinuity (RD)
When treatment is assigned by a threshold on a running variable (score ≥ cutoff → treated), units just
above and below the cutoff are comparable. Estimate the jump in outcome at the cutoff. **Assumption:**
no precise manipulation of the running variable around the cutoff (test for density discontinuity /
bunching); effect is *local* to the cutoff (LATE at the boundary). Strong design when a threshold exists.

### Instrumental Variables (IV)
Find an instrument Z that affects treatment but influences the outcome *only through* treatment
(exclusion restriction) and is as-good-as-random. Two-stage least squares recovers the **LATE** for
"compliers." **Assumptions:** relevance (strong first stage — weak instruments bias badly), exclusion,
independence/monotonicity. Encouragement designs (randomly nudge some users to adopt) are a clean source
of instruments and a bridge back to randomization.

### Propensity-Score Matching / Weighting (PSM, IPW)
Model `P(treated | observed covariates)` and match/weight to balance the arms on *observed*
confounders. **Critical limitation:** only controls for **measured** confounders — no protection against
unobserved confounding (unlike randomization). Always check covariate balance after matching; report
sensitivity to hidden bias. Treat PSM results as suggestive, not definitive.

### Synthetic Control
Construct a weighted combination of untreated units that reproduces the treated unit's *pre-treatment*
trajectory, then use it as the counterfactual after treatment. Designed for **one (or few) treated
units** with a long pre-period (e.g. one country/market gets a launch). Modern variants: augmented
synthetic control, synthetic DiD. Validate with placebo tests (apply to untreated units; the treated
effect should stand out).

### Threats to validity (apply to all of the above)
Unobserved confounding, selection into treatment, reverse causality, time-varying confounders,
spillover/SUTVA violations, and overfitting the model used for adjustment. The honest report lists which
threat each design does and does *not* address.

### Uplift / Heterogeneous Treatment Effects (CATE)
The **Conditional Average Treatment Effect** `τ(x) = E[Y(1) − Y(0) | X = x]` — the effect *for users
like x* — used for **targeting** (treat only users with positive uplift: the "persuadables," not the
"sure things" or "lost causes"). Estimated from experiment (or quasi-experiment) data with:
- **Meta-learners:** **S-learner** (one model with treatment as a feature), **T-learner** (separate
  models per arm), **X-learner** (better with imbalanced arms), **R-learner** (Robinson
  residualization). 
- **Causal forests / generalized random forests** (Wager & Athey) — honest splitting to estimate τ(x)
  with valid CIs.
- **Double/Debiased ML** (Chernozhukov et al.) — orthogonalize out nuisance models for valid inference.

Evaluate uplift with **Qini / uplift curves** and policy value, *not* classification accuracy — you
never observe the individual ground-truth uplift. CATE estimates from observational data inherit all the
identification assumptions above; the cleanest CATE comes from randomized-experiment data.

---

## 8. Application to ML / recsys

Online experiments are how ML/ranking/recsys changes are actually validated; offline metrics only
*propose* a launch. See [[recsys-ranking]] and [[ml-evaluation-evals]].

- **Offline-online gap:** a model that wins on offline AUC/nDCG frequently does *not* win the online
  OEC — because of feedback loops, position/selection bias in logged data, distribution shift,
  presentation effects, and objective mismatch. Treat offline metrics as a *filter*, the A/B as the
  *decision*. Track the offline→online correlation over time to know how much to trust offline.
- **Interleaving** for ranker comparisons: blend two rankers' results in one list per query and observe
  which side's items get clicked. Far more sensitive than A/B (each user is their own control) — great
  as a *cheap pre-filter* to pick candidates for a full A/B, but it optimizes a click-style signal, not
  the OEC, so confirm with an A/B.
- **Counterfactual / off-policy evaluation** (IPS, doubly-robust estimators) reuses logged data to
  estimate a new policy's value before launch — relies on logging propensities and overlap; validate
  against the eventual A/B.
- **Long-term effects:** ranking changes can shift the content ecosystem (creators, supply) — use
  longer holdbacks / **holdout populations** (a small set never exposed to recent launches) to measure
  cumulative long-term impact.
- ML changes are subject to the *same* SRM, novelty, interference (recommendation feedback loops are
  interference), and CUPED discipline as any other experiment.

---

## 9. Anti-patterns (call these out)

- **Peeking** at a fixed-horizon test and stopping at first significance — inflates false positives.
  Use sequential methods or wait for the planned n.
- **HARKing** — Hypothesizing After Results are Known; trawling segments/metrics post-hoc and
  presenting a chance finding as the hypothesis. Pre-register.
- **Ignoring SRM** — reading metrics from an experiment with a broken split. SRM first, always.
- **Means instead of percentiles** for latency/skewed metrics; ratio-metric variance computed wrong.
- **A gameable OEC** (raw clicks, revenue-per-session) that rewards degrading the product.
- **Ignoring interference** in marketplaces/social graphs and reporting naive A/B as if SUTVA held.
- **Claiming causality from observational data** ("users of X retain better") with no identification
  strategy — pure selection bias.
- **p-hacking / metric-shopping** — running until something is significant, picking the metric that
  won. **Underpowered tests** declared "no effect." **PSM presented as randomization-equivalent**
  (it does not handle unobserved confounders).
- **Not validating the platform** (no A/A) and trusting CIs/variance that may be miscalculated.

---

## 10. Version awareness

This field moves: experimentation platforms, sequential-inference libraries, modern DiD estimators
(Callaway-Sant'Anna, Sun-Abraham, synthetic DiD), and causal-ML packages (causal forests, DoubleML,
EconML, CausalML) evolve their APIs and defaults frequently. The **principles** (randomization, OEC,
SRM, power, identification assumptions) are durable; **library APIs, default α-spending schedules, and
specific tool features are not** — verify against current documentation and the latest editions of the
references below before relying on a specific API or default. Do not cite specific win-rate or
variance-reduction percentages without checking the current source.

---

## 11. Canonical references

- **Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing***
  (Cambridge University Press, 2020). The practitioner bible. Companion site: https://experimentguide.com
- **The American Statistician (2024)** review of online controlled experiments —
  doi:10.1080/00031305.2023.2257237.
- **Pearl, *Causality*** (2nd ed.) and Pearl & Mackenzie, *The Book of Why* — do-calculus, back-door
  criterion, causal graphs.
- **Imbens & Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*** — potential
  outcomes, design-based inference.
- **Deng, Xu, Kohavi, Walker (WSDM 2013)** — CUPED variance reduction.
- **Johari, Pekelis, Walsh, et al.** — always-valid inference / mSPRT and confidence sequences for
  continuous monitoring.
- **Tang, Agarwal, O'Brien, Meyer (KDD 2010)** — overlapping experiment infrastructure (layers).
- **Wager & Athey (JASA 2018)** — causal forests / heterogeneous treatment effects. **Künzel et al.
  (PNAS 2019)** — meta-learners (S/T/X). **Chernozhukov et al.** — double/debiased ML.
- **Callaway & Sant'Anna (2021)** and **Sun & Abraham (2021)** — modern difference-in-differences with
  staggered adoption.
- **Gelman & Carlin** — Type S / Type M errors and design analysis.

Verify editions, links, and library APIs against current sources — this is a fast-moving field.

---

# Worked Examples — Experimentation & Causal Inference

Canonical artifacts to imitate. Numbers below are illustrative placeholders for the *shape* of the
calculation — recompute with real traffic and variance, and never copy a figure as if it were measured.

---

## 1. Experiment design doc (pre-registration)

Write and freeze this **before** launch. If a question is decided after seeing data, it is HARKing.

```
# Experiment: Estimated delivery date on product page (PDP)

## Hypothesis
Showing an estimated delivery date on the PDP will INCREASE completed checkouts, because reducing
delivery-time uncertainty lowers purchase hesitation. Directional, mechanism stated.

## Owner / dates / surface
Owner: <team>. Surface: web + mobile-web PDP. Eligible: logged-in + guest users who view ≥1 PDP.

## OEC (decision metric — single, pre-registered, full alpha)
  net_completed_orders_per_visitor   (purchases net of returns/cancellations within 14d)
Rationale: sensitive at our traffic; hard to game (returns netted out so it can't be inflated by
pushing low-quality conversions); validated as a leading indicator of 90-day revenue per the surrogate
analysis dated <link>. NOT revenue-per-session (gameable by pricing/ads) and NOT raw add-to-cart
(does not capture regret).

## Guardrail metrics (must not regress beyond threshold; FDR-controlled scorecard otherwise)
  - PDP p95 latency           : no regression > 20 ms
  - JS error / crash rate      : no increase
  - return/cancellation rate   : no increase > 0.3 pp
  - customer-support contacts  : no increase
  - SRM (trust guardrail)      : chi-square p must be > 0.001

## Randomization
  unit            : user (hashed userID + experiment salt), sticky across sessions/devices where known
  allocation      : 50 / 50 after ramp (max power)
  ramp plan       : 1% (1d, safety) -> 5% (1d) -> 50% (full run); re-check SRM + guardrails at each step
  analysis unit   : visitor; ratio metrics use delta-method variance (analysis unit nested in rand unit)

## Power analysis
  baseline OEC mean   mu0 = 0.083 orders/visitor
  baseline std (per visitor, from 4-wk pre-period) sigma = 0.42
  MDE (practical)     relative +1.0%  ->  absolute Delta = 0.00083
  alpha = 0.05 (two-sided), power = 0.80
  n_per_arm ~= 16 * sigma^2 / Delta^2
            = 16 * 0.42^2 / 0.00083^2  ~= 4.1e6 visitors/arm
  eligible traffic ~= 1.2e6 visitors/day  =>  ~7 days at 50/50 -> round to 1 full week (>=2 wks if
  weekly seasonality is strong). With CUPED (rho ~= 0.45 on pre-period orders) variance falls by
  ~ (1 - rho^2) ~= 0.80, so required n drops materially -> shorter run. (Confirm rho from real data.)

## Pre-registered segments (exploratory, FDR-controlled — NOT decisions)
  platform (web/mobile-web), country, new-vs-returning, traffic source.

## Decision rule
  Ship iff: OEC effect > 0 and CI excludes 0 at the planned n (fixed-horizon) AND no guardrail breach.
  Inconclusive (CI includes 0) => do not ship; report the CI to show whether a meaningful effect was
  ruled out or we simply lacked power. Early stop only via sequential method (below), not by peeking.

## Inference method
  Fixed-horizon read at planned n for the decision. Near-real-time guardrail monitoring with
  auto-rampdown is SAFETY only (false alarms acceptable), separate from the decision read.
```

---

## 2. SRM check (run FIRST) + CUPED note

### 2a. Sample Ratio Mismatch — before reading any metric

Designed 50/50. Observed: control = 2,498,113, treatment = 2,512,902 (total N = 5,011,015).

```python
from scipy.stats import chisquare

control, treatment = 2_498_113, 2_512_902
n = control + treatment
expected = [n * 0.5, n * 0.5]          # designed allocation
stat, p = chisquare([control, treatment], f_exp=expected)
print(f"chi2={stat:.2f}  p={p:.2e}")
# chi2=44.0  p=3.3e-11   ->  p < 0.001  ->  SRM. STOP. Do NOT read OEC/guardrails.
```

**SRM triggered → the experiment is broken.** Do not interpret metrics. Debug checklist (most common
first): bucketing/assignment bug; redirect or latency dropping one arm; bot filtering applied to one
arm only; telemetry loss correlated with treatment; a join in the metric pipeline silently dropping
rows; ramp/restart timing mixing allocations. Also run SRM per key segment (a global pass can hide a
mobile-only SRM). Only once SRM passes (p > ~0.001) do you read the scorecard.

### 2b. CUPED variance reduction (apply once SRM passes)

Use the *same metric measured in the pre-period* as the covariate `X` (must be pre-treatment).

```python
import numpy as np

# y: in-experiment OEC per user; x: same metric in 4-wk pre-period (pre-treatment, per user)
def cuped_adjust(y, x):
    x = x - x.mean()                       # center
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * x                  # unbiased: E[y_adj] = E[y]
    return y_adj, theta

y_adj_c, theta = cuped_adjust(y_ctrl,  x_ctrl)
y_adj_t, _     = cuped_adjust(y_treat, x_treat)   # reuse theta estimated pooled in practice

rho = np.corrcoef(np.r_[y_ctrl, y_treat], np.r_[x_ctrl, x_treat])[0, 1]
print(f"rho={rho:.2f}  variance reduction ~= {1 - rho**2:.0%}")
# Same point estimate of the effect, smaller CI -> more power for the same N (or less N for same power).
```

Notes: CUPED is unbiased (expectation unchanged), it only shrinks variance. The covariate **must** be
pre-treatment — adjusting on anything affected by treatment reintroduces bias. Pair with **triggered
analysis** (only users who could encounter the change) to remove dilution. CUPED composes with
sequential inference if you are monitoring continuously.

---

## 3. Difference-in-Differences sketch (a rollout you cannot randomize)

Scenario: a feature is launched to **all** users in country A on a fixed date; you cannot A/B it
(business mandate). Use country B (and others) as a comparison group. Estimand: the effect of the
launch on the OEC in A.

```
Identification: parallel trends.
  Assume: absent the launch, A and B's OEC would have moved in parallel.
  Validate BEFORE trusting the estimate -> event-study / pre-trend plot (next block).

Design (panel by country x week):
  treat_i  = 1 if country == A else 0
  post_t   = 1 if week >= launch_week else 0
  y_{it}   = OEC for country i in week t

Two-way fixed-effects DiD (single treated unit, single timing):
  y_{it} = alpha_i + gamma_t + beta * (treat_i * post_t) + eps_{it}
  beta_hat = DiD estimate of the launch effect.
```

```python
import statsmodels.formula.api as smf

# df columns: oec, country, week, treat (0/1), post (0/1)
m = smf.ols("oec ~ C(country) + C(week) + treat:post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"]})   # cluster SE by unit
print(m.params["treat:post"], m.conf_int().loc["treat:post"])
```

### Validate parallel trends with an event study (lead/lag coefficients)
```python
# Interact treated x each week relative to launch (omit the week before launch = -1 as baseline).
# PRE-launch coefficients (leads) should be ~0 with CIs around 0  -> parallel pre-trends hold.
# A non-zero pre-trend invalidates the design; do NOT report the DiD as causal.
es = smf.ols("oec ~ C(country) + C(week) + C(rel_week, Treatment(reference=-1)):treat",
             data=df).fit(cov_type="cluster", cov_kwds={"groups": df["country"]})
```

### Threats to validity to disclose with the result
- **Parallel-trends violation** (different pre-trends) — checked above; the load-bearing assumption.
- **Concurrent shocks** in A only (a marketing push, a holiday, a competitor move) at launch time.
- **Spillover / SUTVA** — does the launch in A affect B (shared marketplace/users)? If so, B is
  contaminated and beta is biased.
- **Composition change** — if A's user mix shifts at launch, weight/segment consistently (Simpson's).
- **Staggered adoption** — if multiple countries launch at different times, do NOT use naive two-way
  FE; use a modern estimator (Callaway-Sant'Anna / Sun-Abraham) which is unbiased under heterogeneous
  timing.

Robustness: placebo "launch" dates in the pre-period (effect should be ~0); alternative control groups;
**synthetic control** (weighted blend of untreated countries matching A's pre-trajectory) as a
cross-check when a single comparison country is weak. Report all of these, plus the assumptions each
design does and does not address — the assumptions are the deliverable, not just the point estimate.
