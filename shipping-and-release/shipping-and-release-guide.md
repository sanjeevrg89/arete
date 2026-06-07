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
