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

## Rationalizations & rebuttals

The excuses that quietly destroy experiment trust, each with its rebuttal:

- *"It's already significant, let's just stop and ship."* — Peeking at a fixed-horizon test inflates
  the real false-positive rate (toward 1 under continuous peeking). If you want to stop early, use
  group-sequential or always-valid inference (mSPRT / confidence sequences); otherwise wait for the
  planned n. (§3)
- *"The SRM is close — 50.2/49.8 — ignore it."* — On millions of users that imbalance is enormous; run
  the chi-square on counts. p < ~0.001 means the experiment is *broken* (bucketing bug, asymmetric
  telemetry loss, a dropped join), not slightly off — every metric is untrustworthy until it's fixed. (§4)
- *"The mean is fine, skip percentiles."* — For latency and other skewed/heavy-tailed metrics the mean
  is the wrong summary; a regression hides in the tail. Report p95/p99 with bootstrap or quantile
  methods. (§3)
- *"We don't need guardrails, the OEC went up."* — An OEC win that craters latency, crash rate,
  revenue, or opt-outs is not a win. Pre-register guardrails (including trustworthiness guardrails like
  SRM) so a feature that wins on OEC but harms a guardrail does not ship. (§2)
- *"Users of X retain better, so X causes retention."* — That's selection bias: `E[Y|treated] −
  E[Y|untreated] = ATE + bias`. Observational comparison proves correlation, not causality. Randomize,
  or state an identification strategy (DiD/RD/IV/synthetic control) and its threats. (§1, §7)
- *"Run it a couple of days, that's enough."* — Underpowered tests miss real effects *and* inflate the
  magnitude (and can flip the sign) of any "significant" result (Type M/S errors). Compute MDE → power →
  sample size, then round runtime up to whole weeks for seasonality. (§2)
- *"PSM balances the groups, it's basically a randomized test."* — Propensity matching only balances
  *observed* confounders; it gives zero protection against unobserved confounding. Treat PSM as
  suggestive, report sensitivity to hidden bias, and never present it as randomization-equivalent. (§7)
- *"It's a marketplace but A/B is fine."* — When one unit's treatment changes the pool for others
  (cannibalization, spillover), naive A/B is biased and the control is contaminated. Use cluster /
  switchback / budget-split designs and state the estimand. (§4, §6)

## Red flags

Stop and reconsider if you see any of these:

- **Peeking / optional stopping** on a fixed-horizon test, or a result whose p-value method (fixed vs.
  sequential) doesn't match how the test was monitored.
- **HARKing** — the "hypothesis" was discovered by trawling segments/metrics after seeing the results;
  no pre-registered OEC or directional prediction.
- **SRM present but ignored** (or never checked) — metrics being read from an experiment with a broken
  split, or SRM checked only in aggregate and not on key subpopulations.
- **A gameable OEC** — raw clicks, revenue-per-session, or any metric maximizable by degrading the
  product; no downstream-success or structural component, no guardrails.
- **Interference unaddressed** — naive Bernoulli A/B in a marketplace, shared-budget, or social-graph
  setting reported as if SUTVA held; no cluster/switchback/budget-split design and no stated estimand.
- **Causality claimed without identification** — an observational comparison ("users of X…") presented
  as a causal effect with no design, no assumptions stated, no threats listed.
- **Means reported where percentiles matter** (latency/skewed metrics), or ratio-metric variance
  computed as if the analysis unit were the randomization unit (no delta method / bootstrap).
- **No A/A / platform validation** — trusting CIs and variance that have never been checked for nominal
  coverage and uniform p-values; or comparing metrics across ramp steps (Simpson's paradox risk).
- **Aggregate and per-segment directions disagree** and the result is pooled anyway across periods with
  different allocation.

## Verification gate (definition of done)

An experiment (or quasi-experiment) is not done until:

- [ ] **OEC and guardrails pre-registered** — single decision metric (sensitive, hard to game, leading
  indicator) plus guardrails (latency, crash/error, revenue, opt-outs) and trustworthiness guardrails,
  all declared *before* looking.
- [ ] **Power / MDE / sample size computed up front** — practical MDE chosen as a business call, runtime
  derived from required n and daily eligible traffic, rounded up to whole weeks.
- [ ] **SRM checked first** — chi-square on counts overall and on key subpopulations; no metric
  interpreted while SRM is present.
- [ ] **Randomization unit correct** — coarsest unit that removes interference/carryover and matches the
  experience boundary; assignment is sticky (hashed unit + salt); analysis unit contained by it (delta
  method / bootstrap for ratio metrics).
- [ ] **Sequential method or multiple-comparison correction applied to match how it was read** —
  group-sequential / always-valid for early looks; Bonferroni or BH-FDR for the exploratory scorecard;
  full α reserved for the one pre-registered OEC test.
- [ ] **Triggering / dilution handled** — analyzed on the triggered population (counterfactual
  triggering), then translated to all-up launch impact.
- [ ] **Interference considered** — SUTVA assessed; for marketplace/social settings the appropriate
  design used and the estimand stated; novelty/primacy/carryover checked via the daily effect trend.
- [ ] **Results reported as effect + CI** (absolute and relative), with percentiles for latency-type
  metrics and pre-declared outlier/bot handling.
- [ ] **For quasi-experiments: identification stated** — the assumptions (parallel trends / no
  manipulation at cutoff / exclusion + relevance / pre-period fit) and the threats each design does and
  does *not* address are written down, with validation (event-study, density test, placebo, balance).

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
