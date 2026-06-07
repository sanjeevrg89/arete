# Spec-Driven Development — The Define Stage

## Overview

Spec-driven development is the discipline of turning a vague ask into a **reviewed specification with
testable acceptance criteria** *before* any code is written or any job is launched. It is the **Define**
stage of the [[engineering-lifecycle]]: the first gate, before Plan, Build, Verify, and Ship.

The core claim is economic. The cost of fixing a mistake rises by orders of magnitude as it moves
downstream: a wrong assumption caught in a spec costs minutes to fix; the same assumption caught after a
multi-day training run, a serving rollout, or a cluster reconfiguration costs days, dollars, and trust.
**Most wasted engineering effort is not slow code or bad code — it is correct, well-tested code that
solved the wrong problem.** Spec-first is the cheapest insurance against that failure.

This is doubly true in AI infrastructure and ML, where "building" is rarely a few hours of typing:

- A pretraining or large fine-tune run can be **thousands of GPU/TPU-hours** — real money, days of
  wall-clock, and scarce accelerator quota you cannot get back.
- A serving deployment touches **shared, multi-tenant clusters**; a bad rollout is someone else's outage.
- A "quick" data or pipeline change can silently corrupt a dataset that downstream teams depend on.

You cannot cheaply iterate your way to the right answer when each iteration is a 4,000-GPU-hour run. The
spec *is* the cheap iteration loop. Edit the doc, not the cluster.

## When to use this skill

Use it whenever the work is non-trivial and you feel the pull to start building:

- A feature, service, or API whose requirements are fuzzy ("make inference faster", "add a reranker").
- **Any expensive or irreversible action**: a training/fine-tuning run, a large eval sweep, a serving
  stack rollout, a cluster/node-pool change, a data migration, a quota request.
- A request you received second-hand or in one sentence ("can you serve model X for the new feature?").
- Anything where multiple stakeholders, teams, or shared resources are involved.

**When to keep it lightweight (but not skip it):** a one-line config change, a typo fix, a well-trodden
path you have run ten times. Right-size the spec to the blast radius — a sentence of intent and a
success check may suffice. The *gate* (can I state what done looks like, and is the cost acceptable?)
never goes away; the *artifact* shrinks.

**When to skip entirely:** genuinely throwaway exploration where you will discard everything you learn —
a true spike. Even then, write down the question the spike is meant to answer.

## The process

A numbered process with one hard checkpoint. Do not cross the checkpoint without a reviewed spec.

### 1. Interview: extract the real requirement

A vague ask is a symptom, not a specification. Before you write anything, interview the requester (a
human, a ticket, or — if you are an agent — the prompt and its gaps). Ask, and write down the answers:

- **What** is actually being requested? Restate it in your own words and get it confirmed. The literal
  ask ("serve model X") and the real need ("the new feature needs sub-200ms suggestions") often differ.
- **Why** — the underlying goal. The why is what you optimize for; the what is one possible solution.
  Five-whys until you hit a goal, not a mechanism. ("Why serve model X?" → "because the feature needs
  suggestions" → "why a model, not a heuristic?" → ...). The cheapest spec change is discovering you
  don't need the expensive thing at all.
- **For whom** — the consumer. A latency-sensitive interactive feature, a batch pipeline, a back-office
  tool, and an external API have wildly different specs.
- **Constraints** — budget ($/quota), deadline, accelerator availability, latency ceiling, data
  residency, existing systems you must fit into.
- **Success metric** — how will we know it worked? Push until this is a number or a testable condition,
  not an adjective. "Better", "fast", "reliable" are not success metrics.

If the requester cannot answer these, that is the finding: you have surfaced the ambiguity *before*
spending compute on it. Record the gaps as open questions.

> **Agent note:** when the prompt is underspecified, do not silently pick defaults for load-bearing
> decisions (budget, latency target, quality bar). Ask, or state your assumptions explicitly in the
> spec's Open Questions so a reviewer can correct them cheaply.

### 2. Draft the spec

Write a **lightweight RFC / design doc** — prose, not ceremony. The goal is shared understanding and a
testable definition of done, not a 30-page artifact. (For doc-writing craft and how to drive alignment
on a contentious one, see [[staff-plus-engineering]]; for the system architecture the spec commits to,
see [[ml-system-design]].) An AI-infra / ML spec contains:

1. **Problem & goal.** One paragraph: the problem in user/business terms and the goal (the *why* from
   the interview). What does success enable?
2. **The metric & acceptance criteria.** The ML/business metric (accuracy, recall@k, win-rate,
   conversion, cost-per-request) **and** the acceptance criteria expressed as **testable conditions** —
   each one a thing a script or eval can pass/fail. This is the heart of the spec. See "Acceptance
   criteria" below. Defer the design of the metric/eval harness itself to [[ml-evaluation-evals]].
3. **Scope & non-goals.** What is in. What is explicitly **out** — the non-goals are as important as the
   goals; they stop scope creep and tell reviewers what you chose not to do.
4. **SLOs / SLAs.** Latency (p50 / p95 / p99), throughput (QPS, tokens/sec, samples/sec), availability /
   error budget. State them as numbers with conditions ("p95 < 200 ms at 500 QPS, single region").
5. **Capacity & cost estimate.** Accelerator type and count, **GPU/TPU-hours**, wall-clock, and a
   **dollar estimate**. For training: tokens × params → FLOPs → device-hours. For serving: QPS × cost
   per request, or replicas × instance $/hr. A rough number with stated assumptions beats no number.
6. **Data & dependencies.** Datasets (source, size, freshness, **licensing / PII / governance**),
   upstream services and teams you depend on, models/checkpoints, infra (cluster, quota pool, storage).
7. **Failure modes & blast radius.** What can go wrong, how it manifests, and **who it affects**. On
   shared multi-tenant clusters, name the blast radius explicitly: does a bad run starve other tenants'
   quota, saturate the network fabric, or take down a shared endpoint?
8. **Security, quota & multi-tenancy.** Data access and secrets, tenant isolation, quota you will
   consume and from whose pool, rate limits, and authn/authz for any new endpoint.
9. **Rollback / exit plan.** How you undo it (revert, drain, scale to zero, restore checkpoint) and the
   **kill criteria** — the conditions under which you stop early (loss diverges, cost overruns X, eval
   regresses). An expensive run with no exit plan is a runaway waiting to happen.
10. **Open questions.** The honest list of unknowns and decisions deferred. Listing them routes the
    right reviewers and converts a build-time surprise into a review-time comment.

Keep it to the sections that carry weight for *this* change. A serving rollout leans on SLOs and blast
radius; a training run leans on cost, kill criteria, and acceptance criteria.

### 3. Make acceptance criteria testable

This is where most specs fail. "Improve quality" is not testable. Convert every success notion into a
condition with a metric, a threshold, a dataset, and an operating point:

- ❌ "The model should be accurate and fast."
- ✅ "On `eval-v3` (10k held-out): exact-match ≥ 0.82 **and** ≥ baseline − 0.5%; p95 latency < 200 ms at
  500 QPS measured over a 10-minute load test; cost ≤ \$0.30 per 1k requests."

Each criterion must be something you can later run and get a yes/no from. If a criterion can't be
measured, it isn't a criterion — it's a hope, and hopes don't gate a launch.

### 4. ✅ CHECKPOINT — spec reviewed and agreed before Plan/Build

**Do not proceed to [[task-planning-decomposition]] or any building until the spec has been reviewed by
at least one stakeholder and the acceptance criteria + cost/risk are agreed.** A spec you wrote and
nobody read bought you almost nothing — half its value is the review. The review catches the wrong
problem, the missing constraint, the under-estimated cost, the blast radius you didn't see.

For an expensive run, the reviewer who owns the quota/budget must sign off on the cost estimate. For a
serving change, whoever owns the affected cluster/endpoint must sign off on the blast radius.

Output of this stage: an agreed spec. Now hand off to Plan.

### 5. Treat the spec as living

The spec is the contract, not a monument. When reality contradicts it (the cost was 3x, the metric was
wrong, a dependency slipped), **update the spec and re-confirm** — don't silently drift. A spec that no
longer matches reality is worse than none, because people trust it.

## Rationalizations & rebuttals

| Rationalization | Rebuttal |
| --- | --- |
| "I'll figure out the requirements as I code." | You'll figure out *a* set of requirements — the ones that make your current code look done. Discovering at build time costs ~100x what it costs in a spec. |
| "Writing a spec is slow; I could have shipped by now." | A spec is an hour; the wrong 4,000-GPU-hour run is days and real dollars. The spec is the *fast* path to the right thing. |
| "It's obvious what they want." | If it's obvious, the interview takes five minutes and confirms it. If it isn't, you just avoided building the wrong thing. "Obvious" is where most rework is born. |
| "We'll just iterate." | Iteration is cheap on text, ruinous on compute and live clusters. You cannot A/B your way to the right objective at 4k GPU-hours per round. |
| "The acceptance criteria will emerge once I see the results." | Then you'll move the goalposts to wherever you landed. Define done *before* you can see where you landed, or you can't tell success from rationalization. |
| "I don't need a cost estimate, it's probably fine." | "Probably fine" is how a debugging loop quietly spends a quarter of someone's quota. Estimate first; the number is often a surprise. |
| "Non-goals are obvious, I'll skip them." | Non-goals are where scope creep enters. Writing them is the cheapest scope control you have. |
| "Open questions make me look unsure." | Hidden unknowns make you look unsure *at build time, expensively*. A clear open-questions list is what a senior engineer does. |

## Red flags — stop and reconsider

- **No acceptance criteria**, or criteria that are adjectives ("fast", "good", "robust") rather than
  testable conditions with a metric, threshold, dataset, and operating point.
- **Success can't be measured** — there is no eval, no metric, no way to get a yes/no after building.
- **No cost estimate for an expensive run** (any multi-hundred-GPU/TPU-hour job, large sweep, or new
  always-on serving footprint). You are about to spend money you didn't budget.
- **No SLOs** on a serving change — "make it fast" with no number.
- **No blast radius / rollback** on a shared-cluster or production change.
- **No non-goals** — a sign scope hasn't been bounded.
- **You're already writing code / launching jobs** and the spec doesn't exist yet, or exists but nobody
  reviewed it.
- **The "why" is a mechanism, not a goal** ("because we need model X") — you haven't found the real
  requirement and may be building something unnecessary.
- **The requester can't state the success metric** and you proceeded anyway.

## Verification gate (definition of done for the Define stage)

The Define stage is complete only when **all** of these are true:

- [ ] A **written spec exists** (right-sized to the blast radius) covering problem & goal, metric &
      acceptance criteria, scope & non-goals, SLOs, cost, data/dependencies, failure modes/blast radius,
      security/quota, rollback/exit, and open questions.
- [ ] **Acceptance criteria are testable** — each is a condition a script or eval can pass/fail, with a
      metric, threshold, dataset, and operating point.
- [ ] **Cost and risk are assessed** — GPU/TPU-hours + dollar estimate for expensive work; blast radius
      and rollback for cluster/production changes; kill criteria for long runs.
- [ ] **Reviewed and agreed by at least one stakeholder**, including the owner of the budget/quota (for
      expensive runs) and the owner of the affected cluster/endpoint (for serving/infra changes).
- [ ] **Open questions are listed** (not hidden), and any that block the build are resolved or owned.

Only then proceed to [[task-planning-decomposition]] to plan the work, and [[ml-system-design]] to design
the system the spec commits to.

## Canonical references

- Fred Brooks, *The Mythical Man-Month* / *No Silver Bullet* — the cost-of-change and "building the
  wrong thing" arguments that underpin spec-first.
- Barry Boehm, "Software Engineering Economics" — the cost-to-fix-by-phase escalation curve.
- Google, *Software Engineering at Google* (Winters, Manshreck, Wright) — design docs and review culture.
- Joel Spolsky, "Painless Functional Specifications" (parts 1–4) — why lightweight specs pay for
  themselves.
- The IETF RFC tradition — the spirit of a "request for comments": write it down, circulate, get review.
