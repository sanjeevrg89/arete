# AGENTS.md — Engineering Lifecycle (AI Infra / ML Platform)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative process lives in **`engineering-lifecycle-guide.md`** next to this file —
> read it before starting any non-trivial AI-infra/ML change, and apply it. A complete worked example
> and the right-sizing table are in **`examples.md`**. This file is the always-on summary.
>
> This is the **orchestrator**: it right-sizes the process and routes each stage to its owning skill.
> It does not replace those skills — it sequences them and enforces the gate between them.

## At the start of any non-trivial change, do this:

1. **Right-size first.** Ask: *if this is wrong, how many workloads/tenants break, how much compute is
   wasted, is it reversible?* That answer sets how heavy Define/Plan/Review should be. Small/medium/large
   table is in the guide. **Never** drop Verify safety (smoke + eval gate) or Ship safety
   (progressive + rollback), however small it feels.
2. **Run the six stages in order; pass each gate before advancing.** Cycle back when a later stage
   invalidates an earlier assumption — the loop is iterative, not a waterfall.

## The six stages, their gate, and the skill each routes to

- **Define** → `[[spec-driven-development]]`. Problem, requirements/SLOs, **eval/acceptance criteria,
  cost & capacity estimate, failure modes/blast radius**. *Gate:* a written spec a second party agrees
  to. No spec, no plan.
- **Plan** → `[[task-planning-decomposition]]`. Small, sequenced, independently verifiable steps;
  **risky/expensive work first** (smoke run before full run, canary before fleet, dry-run before apply).
  *Gate:* ordered step list, risky-first, each with a rollback/abort answer.
- **Build** → `[[test-driven-development]]`. Implement **test-first**, small commits, tree green.
  *Gate:* slice does what the step intended, tests pass locally, change is reviewable.
- **Verify** → `[[verification-and-debugging]]`. Integration/e2e + **eval gate** (vs criterion &
  baseline) + **reproducibility** (pinned data/versions/seed); debug to **root cause**. *Gate:*
  evidence, not vibes — unit-green is NOT enough.
- **Review** → `[[code-review-discipline]]`. Correctness, security/multi-tenancy, simplicity,
  **blast radius**. *Gate:* independent approval (not self-review) with blocking comments resolved.
- **Ship** → `[[shipping-and-release]]`. **Progressive + reversible** rollout, monitoring incl.
  quality/eval signals, **tested rollback**. *Gate:* canary healthy on infra **and** eval signals,
  rollback proven, alerts live.

## Why this is stricter for AI infra (keep these in mind)

- Cluster changes can **evict live workloads** — shared infra has a workload-sized blast radius.
- Training runs burn **thousands of accelerator-hours** before failures show — fail cheap and early.
- Model rollouts **degrade quality silently** — green latency/error rates do NOT detect a regression;
  the **eval gate** does.
- **Reproducibility is mandatory** — an unreproducible result is unverified.

## Hard rules (do not violate regardless of right-sizing)

- No Build without a Define spec **and an eval/acceptance criterion**.
- No "verified" without an **eval gate** (for any model/serving change) and a **reproducibility recipe**.
- No ship to **100% at once** on shared infra; no ship with **no rollback or an untested rollback**.
- **Irreversible operations** (destructive migration, data deletion, quota change) are NOT routine
  deploys — flag in Define, add backups/sign-off.
- Never loosen an eval threshold or retry-until-green to make Verify "pass." Fix the root cause.

## Definition of done
Every stage's gate satisfied with evidence shown (spec agreed · risky-first plan · test-first build ·
e2e + eval-gate + repro · independent review · proven progressive rollout w/ rollback & monitoring).
If any gate is unmet, the work is not done — report which gate you are at.
