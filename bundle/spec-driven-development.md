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

---

# Reference — spec-driven-development

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

---

# Spec-Driven Development — Template & Worked Example

A fill-in **AI-infra / ML spec (lightweight RFC / design doc)** template, then one fully worked example.
Imitate the structure. Right-size it: drop sections that carry no weight for your change, but never drop
acceptance criteria, cost (for expensive work), or blast radius/rollback (for cluster/prod changes).

Keep it prose and tables, not ceremony. Defer doc-writing craft and alignment to
[[staff-plus-engineering]], the system architecture to [[ml-system-design]], and the metric/eval-harness
design to [[ml-evaluation-evals]].

---

## Template — copy and fill in

```markdown
# Spec: <short title>

- Author: <name>   Date: <date>   Status: Draft | In review | Agreed
- Reviewers: <budget/quota owner for expensive work; cluster/endpoint owner for serving/infra>

## 1. Problem & goal
<One paragraph in user/business terms. What problem, for whom, and what does success enable?
 The "why" from the interview — a goal, not a mechanism.>

## 2. Metric & acceptance criteria  (the heart of the spec)
Primary metric: <ML/business metric — e.g. recall@k, win-rate, conversion, cost/request>
Acceptance criteria (each a pass/fail a script or eval can produce):
- [ ] <metric ≥/≤ threshold> on <dataset> at <operating point>
- [ ] <latency p95 < N ms at Q QPS over a T-minute load test>
- [ ] <cost ≤ $X per 1k requests / per run>

## 3. Scope & non-goals
In scope: <...>
Non-goals (deliberately excluded): <...>

## 4. SLOs / SLAs
- Latency: p50 <...> / p95 <...> / p99 <...>
- Throughput: <QPS / tokens-per-sec / samples-per-sec>
- Availability / error budget: <...>

## 5. Capacity & cost estimate
- Accelerators: <type × count>, <GPU/TPU-hours>, <wall-clock>
- Cost: <$ estimate with stated assumptions and formula>

## 6. Data & dependencies
- Data: <source, size, freshness, licensing / PII / governance>
- Depends on: <services, teams, models/checkpoints, infra, quota pool>

## 7. Failure modes & blast radius
- <failure> → <how it manifests> → <who it affects> (name shared-cluster impact)

## 8. Security, quota & multi-tenancy
- <data access/secrets, tenant isolation, quota consumed & from whose pool, authn/authz, rate limits>

## 9. Rollback / exit plan & kill criteria
- Rollback: <revert / drain / scale-to-zero / restore checkpoint>
- Kill criteria: <stop early if loss diverges / cost > X / eval regresses>

## 10. Open questions
- <unknowns and deferred decisions; route reviewers; mark which block the build>
```

---

## Worked example — serve model X under a latency / QPS / budget target

> **The vague ask that started it:** *"Can you serve model X for the new suggestions feature?"*
>
> **Interview (step 1) surfaced:** *what* = a low-latency suggestion endpoint, not just "run model X";
> *why* = the new compose-box feature needs inline suggestions that feel instant; *for whom* = an
> interactive, latency-sensitive product surface; *constraints* = launch in 3 weeks, must fit existing
> GPU quota, unit economics must work; *success metric* = p95 latency and cost per 1k requests, with
> quality no worse than the current heuristic. Those answers became the spec below.

```markdown
# Spec: Serve model X for compose-box suggestions

- Author: A. Engineer   Date: 2026-06-06   Status: In review
- Reviewers: quota owner (accelerator pool), endpoint owner (suggestions cluster), product lead

## 1. Problem & goal
The new compose-box feature needs inline text suggestions that feel instant. Today a heuristic returns
weak suggestions; model X is materially better in offline tests. Goal: serve model X behind a
low-latency endpoint so the feature can ship, without blowing the unit economics or destabilizing the
shared suggestions cluster.

## 2. Metric & acceptance criteria
Primary metric: suggestion acceptance rate (online); offline proxy = win-rate vs. heuristic on `sugg-eval-v2`.
Acceptance criteria:
- [ ] Quality: win-rate vs. heuristic ≥ 55% on `sugg-eval-v2` (10k prompts); never below parity.
- [ ] Latency: p95 < 200 ms and p99 < 350 ms at 500 QPS over a 10-minute load test, single region.
- [ ] Throughput: sustains 500 QPS steady-state with < 1% timeouts.
- [ ] Cost: ≤ $0.30 per 1k requests at the target QPS (see §5).
- [ ] Availability: meets the endpoint SLO in §4 in a 24-hour soak.

## 3. Scope & non-goals
In scope: a single-region inference endpoint for model X behind the existing suggestions gateway;
autoscaling; load + soak test; cost validation.
Non-goals: multi-region/HA failover (follow-up); streaming token output; fine-tuning model X;
serving any model other than X; changing the gateway API contract.

## 4. SLOs / SLAs
- Latency: p50 < 90 ms / p95 < 200 ms / p99 < 350 ms at 500 QPS.
- Throughput: 500 QPS steady-state, headroom to 650 QPS peak.
- Availability: 99.9% monthly; error budget shared with the suggestions gateway.

## 5. Capacity & cost estimate
- Model X ≈ 8B params; one accelerator per replica at the target batch/latency.
- Measured ~0.18 s compute per request batched; ~120 req/s per replica at p95 budget →
  500 QPS needs ~5 replicas + 1 for headroom/rollout = 6.
- 6 replicas × $2.50/accelerator-hr (assumption — confirm with quota owner) ≈ $360/day ≈ $11k/month.
- At 500 QPS sustained that is ~1.3B requests/month → ≈ $0.0085 per 1k requests at full load; the
  $0.30/1k criterion holds with large margin even at low utilization. Quota: 6 accelerators from the
  shared inference pool (confirm availability — see Open Questions).

## 6. Data & dependencies
- No new training data; model X checkpoint `model-x@2026-05` from the model registry (license cleared
  for this use; no PII in prompts per product review).
- Depends on: suggestions gateway (routing/authn), the shared inference accelerator pool, the serving
  stack (continuous batching), and metrics/observability ([[ml-observability-monitoring]]).

## 7. Failure modes & blast radius
- Replica OOM / crash-loop on long prompts → elevated p99 and timeouts on the compose feature only,
  but consumes pool quota others share → mitigate with input length cap + per-replica concurrency limit.
- Autoscaler over-provisions under a traffic spike → starves co-tenant workloads in the shared pool →
  cap max replicas at 8; alert the quota owner.
- Bad checkpoint promoted → quality drops below parity → caught by the §2 gate in canary before full rollout.
Blast radius: bounded to the suggestions cluster and the shared accelerator pool's headroom.

## 8. Security, quota & multi-tenancy
- Endpoint behind the gateway's existing authn; per-tenant rate limits inherited from the gateway.
- Quota: 6 accelerators (cap 8) from the shared pool; isolation via the pool's existing scheduling/quota
  mechanism. No new secrets; checkpoint pulled with the existing service identity.

## 9. Rollback / exit plan & kill criteria
- Rollback: route 100% back to the heuristic via the gateway flag (instant); scale model X to zero.
- Canary: 5% traffic for 24h before full rollout.
- Kill criteria: roll back if p95 > 250 ms sustained 10 min, error rate > 2%, win-rate < parity, or
  the pool can't supply the replicas without starving co-tenants.

## 10. Open questions
- Is 6 accelerators available in the shared pool for the launch window? (Blocks build — owner: quota owner.)
- Quality bar 55% vs. 60%? Product to confirm before canary. (Does not block initial build.)
- Confirm the $2.50/accelerator-hr rate used in §5.
```

**Why this gates well:** every line of §2 is something a load test or eval prints a yes/no for; §5 turns
"serve model X" into a defended dollar number *before* the deployment; §7–§9 name who an incident hurts
on the shared cluster and exactly how to back out; §10 puts the one build-blocking unknown (quota) in
front of the reviewer who can resolve it — at review-time cost, not build-time cost. Compare to the
original one-line ask: there was no metric, no cost, no blast radius, no exit. That is the difference
between a spec and a guess.
