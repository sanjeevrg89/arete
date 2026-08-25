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
