---
name: spec-driven-development
description: The Define stage of the engineering lifecycle — clarify and specify before building. Use
  whenever you receive a vague or underspecified ask (a feature, a training run, a serving deployment,
  a cluster change, a migration) and are tempted to start coding or kick off a job. Covers the
  requirements interview (what/why/for-whom/constraints/success metric), writing a lightweight AI-infra
  /ML spec or design doc (problem & goal, eval/acceptance criteria as testable conditions, scope &
  non-goals, SLOs/SLAs, capacity & cost estimate in GPU/TPU-hours and $, data & dependencies, failure
  modes & blast radius, security/quota/multi-tenancy, rollback/exit plan, open questions), the review
  checkpoint before Plan/Build, and the verification gate. Spec-first beats code-first because building
  the wrong thing is the most expensive failure — doubly so when "building" is a multi-thousand-GPU-hour
  run or a production cluster change.
---

# Spec-Driven Development (the Define stage)

Apply the discipline of an engineer who has watched a quarter of compute burn on the wrong objective and
refuses to repeat it. **The cheapest place to fix a mistake is in the spec; the most expensive is in a
running job or a live cluster.** A spec is text you can edit in minutes. Code, a 4,000-GPU-hour run, or a
node-pool change is not. Write the spec first, get it reviewed, *then* plan and build.

This skill owns one stage: **Define**. It does not plan the work ([[task-planning-decomposition]]), design
the system ([[ml-system-design]]), teach doc-writing craft and stakeholder alignment
([[staff-plus-engineering]]), or design the metrics ([[ml-evaluation-evals]]) — it sequences and gates
those. See [[engineering-lifecycle]] for where Define sits.

## How to use this skill

1. **Read `spec-driven-development-guide.md`** in this directory — the full process with checkpoints,
   the spec contents for AI infra / ML, rationalizations, red flags, and the verification gate.
2. For a fill-in **spec template** and a worked example (serve a model at a p95 / QPS / $-budget target),
   read **`examples.md`** and imitate it.
3. Right-size the spec to the blast radius. A one-line config tweak does not need an RFC; a new serving
   stack or a large training run does. The gate scales with cost and risk — never skip it for anything
   expensive or irreversible.

## Essentials (full detail in `spec-driven-development-guide.md`)

- **Interview before you spec.** A vague ask is not a requirement. Extract *what* (the change), *why*
  (the underlying goal), *for whom* (the consumer), *constraints* (budget, deadline, quota, latency),
  and the *success metric* — before writing a line of spec or code.
- **The metric is the spec.** State the ML/business metric and turn it into **acceptance criteria as
  testable conditions** ("p95 < 200 ms at 500 QPS on the eval set; quality ≥ baseline − 0.5%"). If you
  can't measure done, you're not ready to start.
- **Cost is a first-class section.** Estimate GPU/TPU-hours and dollars *before* the run. An expensive
  job with no cost estimate is a red flag, not a plan.
- **Write scope AND non-goals.** Non-goals prevent scope creep and tell reviewers what you deliberately
  excluded.
- **SLOs/SLAs explicitly:** latency (p50/p95/p99), throughput (QPS/tokens-per-sec), availability. Vague
  "fast" and "reliable" are not commitments.
- **Failure modes & blast radius:** what breaks, who it takes down, the rollback/exit plan. Multi-tenant
  clusters and shared quota mean your failure is someone else's outage.
- **Data & dependencies, security & quota:** where the data comes from and its licensing/PII status;
  what services/teams you depend on; quota and multi-tenancy isolation.
- **Open questions are part of the spec.** Listing unknowns is honest and routes the right reviewers;
  hiding them defers the cost to build time at 100x the price.
- **Checkpoint: the spec is reviewed and agreed before Plan/Build.** A spec written and never read by a
  stakeholder bought you nothing.

## Related skills

- `[[engineering-lifecycle]]` — the umbrella; Define → Plan → Build → Verify → Ship. Start here.
- `[[task-planning-decomposition]]` — the next stage: turn the agreed spec into an ordered plan.
- `[[ml-system-design]]` — designing the architecture the spec commits to (serving stack, training topo).
- `[[ml-evaluation-evals]]` — designing the metric and eval harness your acceptance criteria reference.
- `[[staff-plus-engineering]]` — RFC/design-doc writing craft, driving alignment, and review at scale.
