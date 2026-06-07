---
name: engineering-lifecycle
description: The end-to-end engineering lifecycle orchestrator for AI infrastructure and ML platform work — how to take ANY task from idea to production safely through six stages: Define → Plan → Build → Verify → Review → Ship. Use at the START of any non-trivial change (a cluster config edit, a new training pipeline, a model rollout, an autoscaler tweak, a serving-stack upgrade) to decide what process the work needs, run each stage's gate, and route to the right stage skill. Covers why disciplined lifecycle matters MORE for AI infra (GPU-hours are expensive, cluster changes can take down workloads, bad model rollouts degrade production silently), right-sizing process to task size, AI/ML-specific gates (eval criteria, cost estimates, reproducibility, canary rollback), the iterative loop, rationalizations, red flags, and the verification gate per stage. The meta-skill that delegates to spec-driven-development, task-planning-decomposition, test-driven-development, verification-and-debugging, code-review-discipline, and shipping-and-release.
---

# Engineering Lifecycle (AI Infra / ML Platform)

Apply the judgment of a staff engineer who has shipped platform changes that touch thousands of GPUs
and serve live traffic — where a skipped stage means a multi-day outage or tens of thousands of wasted
GPU-hours. This is the orchestrator: it decides **how much process** a task needs and **routes each
stage to the skill that owns it.** It does not replace those skills; it sequences them and enforces the
gates between them.

## How to use this skill

1. Read `engineering-lifecycle-guide.md` in this directory — the full process, the per-stage gates, the
   right-sizing heuristic, and the AI/ML-specific gates. Apply it to the task at hand.
2. For a complete worked example (a TPU training pipeline taken through all six stages with each gate
   shown) and the right-sizing table, read `examples.md`.
3. At the start of any non-trivial AI-infra/ML change: **right-size first** (small / medium / large),
   then run the stages in order, **passing each gate before advancing.** Cycle back when a later stage
   invalidates an earlier assumption — the loop is iterative, not a waterfall.

## The essentials (full detail in `engineering-lifecycle-guide.md`)

- **Six stages, each with a gate that must pass before advancing**, each delegating to one skill:
  - **Define** — problem, requirements, SLOs, eval/acceptance criteria, cost & capacity, failure
    modes. Gate: a written spec a reviewer agrees to. → `[[spec-driven-development]]`
  - **Plan** — decompose into small, sequenced, independently verifiable steps; **order the riskiest
    and most expensive work first.** Gate: an ordered step list with risks called out. →
    `[[task-planning-decomposition]]`
  - **Build** — implement test-first, in small commits. Gate: the slice works and tests pass locally. →
    `[[test-driven-development]]`
  - **Verify** — prove it actually works: integration/e2e, **eval gate, reproducibility**; debug
    failures to root cause. Gate: evidence, not vibes. → `[[verification-and-debugging]]`
  - **Review** — correctness, security, simplicity, **blast radius** before merge. Gate: an approving
    review. → `[[code-review-discipline]]`
  - **Ship** — progressive, **reversible** release with monitoring and a tested rollback. Gate: canary
    healthy, rollback proven. → `[[shipping-and-release]]`
- **Disciplined lifecycle matters MORE for AI infra**: operations are expensive and often irreversible.
  A cluster change can evict running jobs; a training run burns thousands of GPU-hours before you learn
  it diverged; a bad model rollout degrades quality **silently** (latency looks fine, answers get
  worse). Skipping stages is the direct cause of outages and wasted compute.
- **Right-size the process** (heuristic in the guide): a one-line config change is light; a new training
  pipeline or platform component is heavy. Scale Define/Plan/Review to the blast radius. **Never skip
  Verify or Ship safety**, no matter how small it "feels."
- **AI/ML-specific, non-negotiable gates:** Define carries an **eval/acceptance criterion and a cost
  estimate**; Verify carries an **eval gate plus a reproducibility check**; Ship carries a **canary /
  progressive model rollout with a tested rollback**. No eval gate = you cannot tell if you regressed.
- **The loop is iterative.** Verify failing sends you back to Build or Define; a Review finding sends you
  back to Build; a bad canary sends you back through Verify. Cycling back is the system working, not a
  failure.

## Related skills

- `[[spec-driven-development]]` · `[[task-planning-decomposition]]` · `[[test-driven-development]]` ·
  `[[verification-and-debugging]]` · `[[code-review-discipline]]` · `[[shipping-and-release]]` — the six
  stage skills this orchestrator routes to.
- `[[mlops-lifecycle]]` — the ML *system* lifecycle (data → train → deploy → monitor → retrain); this
  skill is the *engineering* lifecycle for a single change within it.
- `[[ml-system-design]]` — designing the system whose changes you take through this lifecycle.
- `[[staff-plus-engineering]]` — scoping, risk, and influence at the level where you own these gates.
- `[[ml-evaluation-evals]]` — how to build the eval that the Define and Verify gates depend on.
