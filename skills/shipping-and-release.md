---
name: shipping-and-release
description: The shipping/release discipline — getting changes to production safely and reversibly, for
  AI infra and ML systems. Use when planning or reviewing a rollout: deploying a service, config, infra,
  or (especially) a model to production; choosing and sequencing progressive delivery (canary, blue-green,
  shadow/dark launch, feature flags); rolling out an ML model (canary by traffic %, champion/challenger,
  shadow inference, staged ramp); building a pre-ship readiness gate (tested rollback plan, monitoring/
  alerts/SLOs live before ramp, capacity/quota checked, runbook, on-call); running the deploy → watch →
  proceed-or-roll-back loop with the right signals (errors, latency/TTFT, eval/quality, cost); handling a
  bad deploy (roll back first, debug later, blameless review); GitOps and change management; risky cluster
  changes in maintenance windows. The Ship stage of the engineering lifecycle — deployment-pattern mechanics
  live in [[mlops-lifecycle]]; this is the shipping process and go/no-go discipline.
---

# Shipping & Release

Apply the judgment of an engineer who has shipped to production for years and been paged for the
fallout: who has watched a "tiny" config change take down inference for a region, rolled back a model
that passed offline eval but tanked the online quality metric, and learned the hard way that an
untested rollback is not a rollback. The whole discipline reduces to one rule: **ship small, ship
reversibly, watch it.** Never big-bang an irreversible change into production.

## How to use this skill

1. **Read `shipping-and-release-guide.md`** in this directory — the full process and reference. Apply
   it to the rollout at hand. For a concrete model-rollout plan, a pre-ship readiness checklist, and a
   rollback-runbook template to imitate, read **`examples.md`**.
2. This is the **Ship** stage. Code is already written, reviewed ([[code-review-discipline]]) and
   verified ([[verification-and-debugging]]); deployment-pattern *mechanics* and ML delivery pipelines
   are in [[mlops-lifecycle]]. This skill is the *discipline* of how to release without breaking prod.
3. Match the team's existing release tooling and conventions; apply the safety gates regardless. If the
   environment can't support a gate (no rollback, no monitoring), that is a finding — surface it, don't
   skip it silently.

## Essentials (full detail in `shipping-and-release-guide.md`)

- **Ship small, ship reversibly, watch it.** Three properties of every safe release: the change is
  small (bounded blast radius), it can be undone fast, and you are watching the right signals while it
  ramps. Missing any one → you are gambling, not shipping.
- **Progressive delivery, always.** Roll out to a slice before everyone: canary by % of traffic,
  blue-green for instant switch-back, shadow/dark launch for zero-user-impact validation, feature flags
  to decouple deploy from release. Default to canary for anything user-facing.
- **Model rollouts are riskier than code rollouts** — offline eval ≠ online behavior. Always
  **shadow → small canary % → staged ramp → 100%**, gated on online quality/latency/cost, with
  **champion/challenger** comparison. Never swap a model 0→100%. (Mechanics: [[mlops-lifecycle]],
  [[gke-inference-gateway]].)
- **The pre-ship readiness gate is a hard gate, not a checklist you wave through.** Before ramping:
  rollback plan **tested** (executed in staging, not just written), monitoring/alerts/SLOs **live
  before** the first user sees the change ([[ml-observability-monitoring]]), capacity/quota confirmed
  ([[autoscaling-kubernetes]], [[gke-master]]), runbook written, on-call aware, blast radius bounded,
  data migrations reversible. No-go on any → don't ramp.
- **A rollback that has never been executed is a hope, not a plan.** Test it before you need it.
- **Monitoring goes live before the ramp, not after.** "We'll add monitoring later" means you ramp
  blind and find out from users.
- **The ramp loop:** deploy to a slice → watch errors, latency/TTFT, eval/quality, cost against
  explicit abort criteria → proceed to the next step or **roll back**. Bake at each step long enough to
  see real traffic. Don't skip steps because the last one looked fine.
- **When it goes wrong: roll back first, debug later.** Mitigation precedes diagnosis. Restore the
  good state, *then* investigate the bad one ([[verification-and-debugging]]).
- **Reversible migrations.** Expand/contract (additive first, destructive later, after the new code is
  proven). Never deploy code and an irreversible schema change in the same irreversible step.
- **Change management for infra:** GitOps (Git as source of truth, declarative, auditable, revert =
  rollback). Do risky cluster changes in a maintenance window, off-hours, with on-call online.
- **Blameless incident review** after anything that went sideways. Fix the system and the process, not
  the person. Feed the learning back into the readiness gate.
- **Don't ship into a hole.** No Friday-evening ship of a risky change with no on-call. No ship while
  you're the only one who understands it and you're about to be unreachable.

## Related skills

- `[[engineering-lifecycle]]` — the umbrella; Ship is one stage. Start there for the whole arc.
- `[[code-review-discipline]]` — the gate *before* ship; nothing ramps that wasn't reviewed.
- `[[verification-and-debugging]]` — verify before ship; debug *after* you've rolled back.
- `[[mlops-lifecycle]]` — deployment-pattern mechanics and ML delivery/CD pipelines (the *how*).
- `[[ml-observability-monitoring]]` — the signals you watch during the ramp; must be live first.
- `[[gke-inference-gateway]]` — traffic splitting / model routing for canary and shadow on GKE.
- `[[autoscaling-kubernetes]]` — capacity/quota headroom so the ramp doesn't starve.
- `[[gke-master]]` — cluster-level capacity, node pools, and safe maintenance windows.

---

# Reference — shipping-and-release

# Shipping & Release — the discipline of getting changes to production safely

The Ship stage of the engineering lifecycle. Your code is written, reviewed, and verified; now it has
to reach production without breaking it. Shipping is not "press deploy." It is a *risk-management
discipline* whose entire job is to make change **safe and reversible** in a system that real users and
real money depend on. This guide is the process and the judgment behind it, tailored to AI infra and ML
systems where the most dangerous change is often a model, not code.

> Scope note: the *mechanics* of deployment patterns (how canary/blue-green/shadow are wired, ML
> delivery/CD pipelines, registries) live in [[mlops-lifecycle]]. This guide is the *discipline* — when
> to use which, in what order, behind which gate, watching what, and how to back out. Where this and
> mlops overlap, this is the "how to work," that is the "how it's built."

## Overview — the one principle

**Ship small, ship reversibly, watch it.** Every safe release has three properties:

1. **Small** — bounded blast radius. A slice of traffic, a subset of hosts, one region, one tenant,
   behind a flag. If the change is bad, a few percent feel it, not everyone.
2. **Reversible** — you can get back to the last-known-good state *fast* and you have *proven* you can.
   The faster and more certain the rollback, the more aggressively you can ship.
3. **Watched** — you are looking at the right signals while it ramps, with explicit thresholds that
   say "proceed" or "roll back."

Remove any one and you are gambling. A small change watched but irreversible can still wedge you. A
reversible change nobody is watching ramps to 100% before anyone notices. The discipline is holding all
three at once, every time.

The opposite — the thing this skill exists to prevent — is the **big-bang irreversible deploy**: swap
everything at once, no slice, no watch, no way back. It works until the one time it doesn't, and that
time is an outage.

## When to use this skill

- You are about to deploy *anything* to production: a service, a config change, an infra change, a
  schema/data migration, or (highest risk) a model.
- You are designing or reviewing a rollout plan and need the go/no-go gate.
- A deploy is going wrong and you need the "roll back first, debug later" reflex.
- You are setting up change management / GitOps / release process for a team.

The riskier and less reversible the change, the more of this discipline applies. A doc typo behind no
gate is fine. A new model serving 100% of inference is the full ceremony.

## Core concepts — progressive delivery

Progressive delivery = exposing a change to an increasing slice while you watch, instead of all at once.
The patterns (mechanics in [[mlops-lifecycle]]; for ML traffic on GKE see [[gke-inference-gateway]]):

| Pattern | What it does | Reversible by | Best for |
|---|---|---|---|
| **Canary** | Route a small % of live traffic to the new version | Routing % → 0 | Default for user-facing code & models |
| **Blue-green** | Stand up new (green) beside old (blue), switch the router | Switch router back | Fast, atomic cutover with instant rollback |
| **Shadow / dark launch** | Send a *copy* of live traffic to the new version; discard its responses | Stop mirroring (zero user impact) | Validating a model/service under real load with no risk |
| **Feature flag** | Ship the code dark; turn behavior on for a cohort at runtime | Flip the flag off | Decoupling *deploy* from *release*; instant kill switch |
| **Rolling update** | Replace instances batch by batch | Roll back the spec | Stateless code where canary isn't wired |

Key distinctions that matter:

- **Deploy ≠ release.** Deploying ships the bits; releasing exposes the behavior. Feature flags let you
  deploy code to 100% of hosts while the behavior is off, then release it to 1% → 100% independently —
  and kill it in milliseconds without a redeploy. This is the single biggest leverage for reversibility.
- **Canary vs shadow.** Canary serves real users (real risk, real signal). Shadow serves *nobody*
  (zero risk) but lets you compare the new version's behavior/latency/cost on real traffic before any
  user sees it. Shadow first, then canary.
- **Blue-green's gift is the instant switch-back**; its cost is double capacity during the cutover —
  make sure quota covers it ([[autoscaling-kubernetes]]).

### ML model rollouts specifically

Models are the dangerous case because **offline eval does not predict online behavior.** A model can
win on the eval set and lose on live traffic (distribution shift, prompt/feature skew, latency
regressions, a quality metric the eval didn't capture). So model rollouts get extra discipline:

- **Shadow inference first.** Mirror live requests to the challenger, log its outputs, compare against
  the champion offline. No user impact. Catches latency/cost regressions and gross quality failures
  before any traffic.
- **Canary by traffic %.** Route a small slice (start ~1–5%) to the challenger; watch online quality,
  latency/TTFT, and cost on *served* traffic.
- **Champion/challenger.** The current production model is the champion; the new one is the challenger.
  Compare them on the *same* live traffic (or matched slices) on the *same* metrics. The challenger
  must *beat or match* the champion on the agreed metrics to be promoted — "it deployed cleanly" is not
  promotion criteria.
- **Staged ramp.** 1% → 5% → 25% → 50% → 100%, baking at each step long enough to gather a
  statistically meaningful read on the quality metric (often longer than for code, because quality
  signals can lag and be noisy). Ground truth may be delayed — lean on proxy/online metrics and
  [[ml-observability-monitoring]] for drift and quality.
- **Keep the champion warm** for the whole ramp so rollback is "route back to champion," not "redeploy
  the old model."

The signals for an ML ramp are *not* just errors and latency — they include the **eval/quality metric**
(online win rate, task success, click/accept rate, LLM-as-judge score, calibration) and **cost per
request** (tokens, GPU-seconds). A model that is faster and cheaper but quietly worse is still a failed
rollout. See [[ml-observability-monitoring]] for what to instrument.

## The process

A release is a sequence with one hard gate in the middle. Numbered, with checkpoints.

### 1. Plan the rollout (before touching prod)

- State the **change** and its **blast radius**: who/what is affected if it's wrong, and how bad.
- Choose the **delivery pattern** and the **ramp schedule** (the % steps and bake time at each).
- Define **acceptance criteria** and **abort criteria** *up front*, as numbers: which signals, what
  thresholds, over what window trigger "proceed" vs "roll back." Decide them now, while you're calm —
  not at 2am while the graph is red.
- Write the **rollback procedure**: the exact steps/commands to restore last-known-good, and the target
  RTO ("good state within N minutes").
- For data: design the migration **reversible** (expand/contract — see below).

### 2. Pass the pre-ship readiness gate (HARD GATE — go/no-go)

Do **not** ramp until every item is true. This is the heart of the discipline.

- [ ] **Rollback plan tested** — actually executed in staging/pre-prod (or a prior canary), not just
      written. You have *seen* it restore the good state and you know the RTO.
- [ ] **Monitoring, alerts, and SLOs are LIVE before the ramp** — the dashboards and alerts that will
      tell you the change is bad exist and fire *now*, before the first user is exposed
      ([[ml-observability-monitoring]]). Adding them "after" means ramping blind.
- [ ] **Capacity & quota confirmed** — there is headroom for the new version (and for double capacity
      if blue-green), including GPU/TPU/accelerator quota and autoscaling limits
      ([[autoscaling-kubernetes]], [[gke-master]]). The ramp must not starve the cluster.
- [ ] **Runbook written** — what this change is, how to watch it, how to roll it back, who to call.
- [ ] **On-call aware** — whoever owns the pager knows this is shipping and what it does.
- [ ] **Blast radius bounded** — the first step exposes a small, recoverable slice.
- [ ] **Data migrations reversible** — no irreversible/destructive step coupled to this release.
- [ ] **Abort criteria are explicit numbers**, agreed before ramp.

**Any unchecked box → no-go.** Surface the gap; do not quietly ship around it.

### 3. Deploy to the first slice

- Push to the smallest meaningful slice: shadow, then 1% canary (or one host/region/tenant), or behind
  a flag enabled for an early-access/dogfood cohort. For infra, deploy via GitOps so the change is a reviewed,
  revertable commit (see Change management below).
- Confirm the new version is actually serving the slice (don't assume — verify the routing).

### 4. The ramp & watch loop

For each ramp step:

1. **Increase exposure** by one step (e.g., 1% → 5%).
2. **Bake** — let it run long enough to gather real signal under representative traffic. Don't ramp
   faster than your signal arrives. For models, this is often longer (quality is noisy/lagging).
3. **Watch the right signals** against the abort criteria:
   - **Errors** — error rate, exception/crash rate, 5xx, saturation/OOM.
   - **Latency** — p50/p95/p99; for LLMs **TTFT** (time-to-first-token) and **ITL** (inter-token
     latency)/throughput.
   - **Eval / quality** — the online quality metric (win rate, task success, accept rate, judge score,
     calibration), drift ([[ml-observability-monitoring]]).
   - **Cost** — per-request cost, tokens, GPU/TPU-seconds, instance count.
4. **Decide: proceed or roll back.** If signals are healthy and beating/matching the baseline →
   proceed to the next step. If *any* abort criterion trips → roll back (step 6). When in doubt, roll
   back; you can always re-ramp.

Repeat until 100%. **Do not skip steps because the last one looked fine** — a regression can be
sublinear in traffic %, and quality issues surface only at scale.

### 5. Post-deploy verification

At 100% (and at meaningful steps), verify the release **met its acceptance criteria** — the thing you
shipped actually does what it was supposed to, in production, by the numbers you defined in step 1. A
green deploy that didn't achieve its goal is not done. Then clean up: retire the old version once the
new one has baked, remove stale flags, close the change ticket. (Verifying behavior in the running
system: [[verification-and-debugging]].)

### 6. When it goes wrong — roll back first, debug later

- **Mitigate before you diagnose.** The instant an abort criterion trips, execute the rollback /
  flip the kill switch / route back to champion. Restore the good state for users *first*. The urge to
  "just figure out what's wrong real quick" while users suffer is the wrong instinct — every minute of
  diagnosis is a minute of impact you could have stopped.
- Roll back is fast and certain *because you tested it in step 2*.
- Once stable, **then** investigate the bad version offline ([[verification-and-debugging]]): reproduce,
  find root cause, fix, and re-enter the process from step 1.
- After any incident, run a **blameless incident review**: timeline, what failed, what the monitoring
  missed, what slowed the rollback, what to change in the *system and the process* (not who to blame).
  Feed the gaps back into the readiness gate so the same class of failure can't recur silently.

## Change management & GitOps for infra

- **Git is the source of truth.** Declarative desired state in a repo; a controller (Argo CD, Flux,
  etc.) reconciles the cluster to it. Benefits that matter for shipping: every change is **reviewed**
  (PR), **auditable** (history), and **revertable** (revert the commit = roll back the cluster). GitOps
  makes "roll back" a `git revert`, which is the most reversible deploy there is.
- **Sync waves / progressive sync** for ordering and canarying infra changes; don't reconcile a risky
  change cluster-wide in one shot.
- **Risky cluster changes go in a maintenance window**: node-pool upgrades, control-plane/version
  upgrades, CNI/CSI changes, anything that can disrupt running workloads. Pick off-hours/low-traffic,
  announce it, have on-call online, and have the rollback ready ([[gke-master]] for upgrade safety and
  surge settings; [[autoscaling-kubernetes]] for keeping capacity during the churn).
- **One change at a time.** Don't bundle a model swap, a config change, and an infra upgrade into one
  release — if it breaks you won't know which, and rollback gets ambiguous.

## Reversible data migrations (expand/contract)

The trap that turns a reversible deploy into an irreversible one is the schema/data change. Use
**expand → migrate → contract**:

1. **Expand** — add the new column/field/table *additively*; old code ignores it. Reversible (drop it).
2. **Deploy** code that writes both old and new (dual-write) and can read either.
3. **Backfill / migrate** data; verify.
4. **Switch** reads to the new path (behind a flag — reversible).
5. **Contract** — only after the new path is proven and baked, remove the old column/field. This is the
   destructive, irreversible step; it happens *last*, in a *separate* release, long after the code that
   needed the change is stable.

Never couple "deploy new code" with "drop the old column" in one step — you've made rollback impossible.

## Rationalizations & rebuttals

| Rationalization | Rebuttal |
|---|---|
| "Just push to prod, it's fine." | "Fine" is a prediction, not a fact. Canary it; if you're right it costs minutes, if you're wrong it saves an outage. |
| "We'll add monitoring later." | Then you ramp blind and learn it's broken from users, not graphs. Monitoring is a *precondition* of the ramp, not a follow-up. |
| "Rollback is easy, no need to test it." | An untested rollback is a hope. The first time you run it shouldn't be during an incident. Execute it once in staging. |
| "It's a small change." | Blast radius isn't measured in lines. A one-line config or a model swap can take down a region. Small change → still canary, just ramp faster. |
| "Offline eval passed, ship the model to 100%." | Offline ≠ online. Shadow + canary catch the regressions eval can't. Champion/challenger or it doesn't ship. |
| "Let's get it out before the weekend." | Friday-evening risky ships with no on-call are how weekends get ruined. Ship Tuesday morning, or staff the pager. |
| "We'll just fix forward, no need to roll back." | Fix-forward under incident pressure ships *another* untested change. Roll back to known-good first, fix calmly after. |
| "Let me find the bug first, then I'll roll back." | Backwards. Roll back first (stop the bleeding), debug after. Mitigation precedes diagnosis. |

## Red flags — stop and reconsider

- **No rollback** — or a rollback nobody has ever executed.
- **No monitoring/alerts live before the ramp.** You're about to ramp blind.
- **Big-bang deploy** — 0 → 100% in one step, no slice, no canary.
- **Irreversible migration** coupled to the release (destructive schema change in the same step).
- **No canary for a model change** — promoting a model on offline eval alone.
- **Shipping Friday evening** (or before you're unreachable) with no on-call coverage.
- **Bundling** a model + config + infra change in one release.
- **Skipping ramp steps** because the previous step "looked fine."
- **No abort criteria** defined before the ramp — deciding "is this bad enough to roll back?" live.
- **Quota/capacity unchecked** — the ramp (or blue-green's double footprint) will starve the cluster.
- **"Fix forward" as the default** under pressure instead of rolling back.

## Verification gate (definition of done)

A release is done only when all of these are true — report honestly if any aren't:

- [ ] **Rollback was tested** (executed, not just written) and its RTO is known.
- [ ] **Monitoring & alerts were live before the ramp** and stayed green through it.
- [ ] **Progressive delivery was used** — the change ramped through slices, not big-bang.
- [ ] For a model: **shadow + canary** ran and the **challenger beat/matched the champion** on the
      agreed quality/latency/cost metrics before promotion.
- [ ] **Capacity/quota was confirmed** and held through the ramp.
- [ ] **Runbook exists** and on-call was aware.
- [ ] **Post-deploy verification** confirms the release met its acceptance criteria in production.
- [ ] **Data migrations are reversible** (expand/contract; no irreversible step coupled to this ship).
- [ ] If anything went wrong: rolled back first, then a **blameless review** captured the fix to the
      system *and* the process.

## Canonical references

Verify against current docs — the tooling and APIs in this space move fast (it is 2026).

- Google SRE Book & SRE Workbook — release engineering, canarying, error budgets, incident response:
  https://sre.google/books/
- Argo Rollouts (canary/blue-green for Kubernetes): https://argo-rollouts.readthedocs.io/
- Argo CD (GitOps): https://argo-cd.readthedocs.io/  ·  Flux: https://fluxcd.io/
- Progressive delivery overview: https://www.cncf.io/blog/2021/02/05/progressive-delivery-what-it-is-and-why-you-should-care/
- Martin Fowler — Feature Toggles / Flags: https://martinfowler.com/articles/feature-toggles.html
- Martin Fowler — BlueGreenDeployment & CanaryRelease: https://martinfowler.com/bliki/BlueGreenDeployment.html
- Expand/contract (parallel change) migration pattern: https://martinfowler.com/bliki/ParallelChange.html
- OpenFeature (vendor-neutral feature flagging): https://openfeature.dev/

---

# Shipping & Release — worked examples

Concrete artifacts to imitate: a progressive model-rollout plan, a pre-ship readiness checklist, and a
rollback-runbook template. Adapt the names, numbers, and thresholds to your system — the *structure* and
the *gates* are the reusable part.

---

## 1. Progressive model-rollout plan (shadow → canary → ramp → 100%)

Scenario: promoting a new LLM serving the production `assistant` endpoint. The current production model
is the **champion**; the new one is the **challenger**. Traffic split via the inference gateway
([[gke-inference-gateway]]); signals via [[ml-observability-monitoring]].

### Baseline & criteria (agreed BEFORE any traffic)

| Metric | Source | Champion baseline | Challenger must... |
|---|---|---|---|
| Error / 5xx rate | gateway + serving | 0.2% | ≤ 0.3% |
| TTFT p95 | serving | 850 ms | ≤ 900 ms (no >10% regression) |
| Inter-token latency p95 | serving | 35 ms/token | ≤ 40 ms/token |
| Online quality (LLM-as-judge win rate) | eval pipeline | n/a (50/50 ref) | ≥ 50% vs champion |
| Task success / accept rate | product telemetry | 71% | ≥ 71% (no regression) |
| Cost / 1k requests | billing/token meter | $X | ≤ 1.15 × $X |

**Global abort criteria (roll back immediately at any step):** error rate > 1% for 5 min · TTFT p95 >
1.5 × baseline for 10 min · quality win rate < 45% over the bake window · any pod crashloop/OOM · cost
> 1.5 × baseline.

### Stages

| Stage | Traffic to challenger | Bake | Watch | Gate to proceed |
|---|---|---|---|---|
| **0. Shadow** | 0% served; 100% *mirrored* (responses discarded) | 24 h | TTFT/ITL, errors, cost, offline quality on logged outputs | No latency/cost regression; no gross quality failures; no crashes |
| **1. Canary** | 1% | 2–4 h | all metrics on *served* traffic + global abort criteria | All within thresholds; quality ≥ 50% |
| **2. Ramp** | 5% | 4 h | same | within thresholds |
| **3. Ramp** | 25% | 8 h (overnight ok with on-call) | same + slice/segment quality | within thresholds; no bad segment |
| **4. Ramp** | 50% | 8–24 h | same | within thresholds |
| **5. Full** | 100% | — | continue monitoring 48 h | post-deploy verification passes |

### Rules

- **Champion stays warm** through stage 5; rollback = route 100% back to champion (seconds), not a
  redeploy.
- **Bake long enough for the quality signal**, which is noisier and slower than latency. Don't ramp
  faster than your quality read arrives.
- **Don't skip stages** — a regression can be sublinear in traffic; segment-level quality issues only
  surface with volume (stage 3+).
- **Decommission the champion only after 100% has baked** (e.g., 48 h clean), and remove the split
  config last. Keep the rollback path until then.

---

## 2. Pre-ship readiness checklist (the go/no-go gate)

Fill this in before ramping. **Any unchecked box = NO-GO.** Don't ship around a gap — surface it.

```
RELEASE: <what is shipping, version/commit/model-id>
OWNER:   <name>            DATE/WINDOW: <when>            ON-CALL: <name, acknowledged? Y/N>

BLAST RADIUS
[ ] Who/what breaks if this is wrong, and how bad:  ____________________________
[ ] First step exposes a small, recoverable slice:  ____________________________

DELIVERY PLAN
[ ] Pattern chosen (canary / blue-green / shadow / flag): __________
[ ] Ramp schedule (% steps + bake time):  ____________________________
[ ] Acceptance criteria (numbers, what "success in prod" means): ______________
[ ] Abort criteria (numbers, what triggers rollback):  ________________________

REVERSIBILITY  (the heart of the gate)
[ ] Rollback procedure written (exact steps/commands):  see runbook
[ ] Rollback TESTED — executed in staging/pre-prod, not just written.  RTO observed: ____ min
[ ] Kill switch / flag-off path exists and was exercised
[ ] Data migration is reversible (expand/contract; no destructive step in this release)

OBSERVABILITY  (must be LIVE before the ramp)
[ ] Dashboards for errors, latency/TTFT, eval/quality, cost — live now
[ ] Alerts wired to the abort criteria — firing now (tested)
[ ] SLOs / error budget known                          (see [[ml-observability-monitoring]])

CAPACITY
[ ] Headroom for the new version confirmed (incl. double footprint if blue-green)
[ ] GPU/TPU/accelerator quota confirmed                (see [[autoscaling-kubernetes]], [[gke-master]])
[ ] Autoscaling limits won't cap the ramp

PROCESS
[ ] Runbook written and linked
[ ] On-call aware and acknowledged
[ ] Not a Friday-evening / unattended risky ship (or on-call is staffed for it)
[ ] One change only — not bundled with unrelated config/infra changes

GO / NO-GO:  ______   (all boxes checked → GO)
```

---

## 3. Rollback runbook template

One page per release. Written in step 1, *tested* in step 2, used in step 6.

```
# Runbook: <release name / model-id / version>

## What this change is
<1–3 sentences: what shipped, why, what behavior changed.>

## How to watch it (during ramp)
- Dashboard:   <link>
- Key signals & thresholds:
    errors    > 1% / 5min           -> ABORT
    TTFT p95  > 1.5x baseline /10min -> ABORT
    quality   win-rate < 45%         -> ABORT
    cost      > 1.5x baseline        -> ABORT
- Alerts: <links to the alerts wired to the above>

## HOW TO ROLL BACK  (do this FIRST, debug later)
Target RTO: <N> minutes.  (Verified on: <date> in <env>.)

Option A — flag/route (fastest, preferred):
  1. <command to flip flag off / route traffic 100% back to champion-or-blue>
     e.g.  kubectl ... set the traffic split to champion=100, challenger=0
  2. Confirm: <command/dashboard showing traffic on the good version>

Option B — redeploy last-known-good (if A unavailable):
  1. <git revert <sha> && sync   |   deploy <previous-version>>
  2. Confirm healthy: <command/check>

After rollback:
  [ ] Confirm error/latency/quality back to baseline on the dashboard
  [ ] Post in <incident channel>; page <owner> if not already
  [ ] DO NOT re-ramp until root cause is found and fixed

## Data / migration notes
- Reversible? <yes/no + how>.  If a backfill ran: <how to reconcile on rollback>.

## Escalation
- Primary on-call: <name/rotation>   Secondary: <name>
- Owning team / channel: <link>

## After the incident
- [ ] Blameless review scheduled: timeline, what monitoring missed, what slowed rollback
- [ ] Action items filed; readiness-gate updated so this class of failure can't recur silently
```

---

### Anti-pattern (do NOT imitate)

```
# "Ship it" — the big-bang
git push prod main        # 0 -> 100% of traffic, all at once
# no canary, no shadow, no flag
# monitoring: "we'll add it next sprint"
# rollback: "just git revert if something breaks" (never tested)
# migration: ALTER TABLE ... DROP COLUMN  (in the same deploy as the new code)
# 5:40pm Friday, on-call not told
```

Every line above is a red flag: big-bang, no progressive delivery, no monitoring before ramp, untested
rollback, irreversible migration coupled to the deploy, and an unattended Friday ship. This is how a
"small change" becomes a weekend outage.
