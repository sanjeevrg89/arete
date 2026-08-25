# AGENTS.md — Shipping & Release Discipline

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative process lives in **`shipping-and-release-guide.md`** next to this file — read
> it before planning or reviewing a rollout, and apply it. A concrete model-rollout plan, a pre-ship
> readiness checklist, and a rollback-runbook template to imitate are in **`examples.md`**. This file is
> the always-on summary.
>
> **This is the Ship stage.** Deployment-pattern mechanics and ML delivery pipelines live in
> [[mlops-lifecycle]]; this is the *discipline* of releasing safely. The one rule: **ship small, ship
> reversibly, watch it.** Never big-bang an irreversible change into production.

## When shipping anything to production, apply these by default:

- **Ship small, ship reversibly, watch it.** Every release must be (1) small — bounded blast radius,
  (2) reversible — fast, *tested* rollback, (3) watched — right signals with explicit abort thresholds.
  Missing any one is gambling, not shipping.
- **Progressive delivery, always.** Canary by % for user-facing changes; blue-green for instant
  switch-back; shadow/dark launch for zero-impact validation; feature flags to decouple deploy from
  release and as a kill switch. **Deploy ≠ release.** Never 0 → 100% in one step.
- **Model rollouts get extra care** — offline eval ≠ online behavior. Always **shadow → small canary %
  → staged ramp → 100%**, with **champion/challenger** on the same live traffic. The challenger must
  *beat or match* the champion on agreed quality/latency/cost before promotion. Keep the champion warm
  for instant rollback. (Mechanics: [[mlops-lifecycle]], [[gke-inference-gateway]].)
- **Pre-ship readiness gate is a HARD go/no-go** — don't ramp until all are true:
  rollback plan **tested** (executed, not just written) · monitoring/alerts/SLOs **live before** the
  ramp ([[ml-observability-monitoring]]) · capacity/quota confirmed ([[autoscaling-kubernetes]],
  [[gke-master]]) · runbook written · on-call aware · blast radius bounded · data migrations
  reversible · abort criteria defined as explicit numbers. Any miss → no-go; surface it, don't skip it.
- **A rollback never executed is not a rollback.** Test it in staging before you need it.
- **Monitoring is a precondition of the ramp, not a follow-up.** "Add it later" = ramp blind.
- **The ramp loop:** deploy to a slice → bake → watch errors, latency/**TTFT**, eval/quality, cost
  against abort criteria → proceed or roll back. Don't skip steps because the last one looked fine.
- **When it goes wrong: roll back first, debug later.** Mitigation precedes diagnosis — restore the good
  state for users, *then* investigate offline ([[verification-and-debugging]]). Don't "fix forward"
  under incident pressure.
- **Reversible data migrations** via expand/contract: additive first, destructive (irreversible) step
  last, in a separate release after the new path is proven. Never couple new code with a destructive
  schema change.
- **Infra = GitOps** (Git as source of truth; revert = rollback). Risky cluster changes go in an
  off-hours maintenance window with on-call online. One change at a time — don't bundle model + config
  + infra.
- **Blameless incident review** after anything that went sideways; fix the system and the process, feed
  it back into the readiness gate.
- **Don't ship into a hole** — no Friday-evening risky ship with no on-call.

## Red flags (stop)
No rollback (or untested rollback) · no monitoring before ramp · big-bang 0→100% · irreversible
migration coupled to the release · no canary for a model change · Friday-evening ship with no on-call ·
bundled changes · skipped ramp steps · no abort criteria · unchecked capacity/quota · "fix forward" by
default.

## Definition of done for a release
Rollback tested · monitoring/alerts live before ramp · ramped via progressive delivery (not big-bang) ·
for models, shadow+canary ran and challenger beat/matched champion · capacity confirmed · runbook
exists & on-call aware · post-deploy verification met acceptance criteria in prod · migrations
reversible · if it broke, rolled back first then blameless review. Report honestly if any fail.
