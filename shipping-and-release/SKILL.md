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
