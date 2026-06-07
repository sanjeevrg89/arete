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

---

# Reference — engineering-lifecycle

# Engineering Lifecycle — Full Reference (AI Infra / ML Platform)

## Overview

This is the meta-process for taking **any** AI-infrastructure or ML-platform change from idea to
production safely. It defines six stages — **Define → Plan → Build → Verify → Review → Ship** — each
with a single purpose, an explicit **gate that must pass before you advance**, and a **stage skill it
delegates to**. The orchestrator's job is not to do the work of each stage; it is to (1) right-size the
process to the task, (2) run the stages in order, (3) enforce the gate between them, and (4) cycle back
when a later stage invalidates an earlier assumption.

It is deliberately stage-skill-agnostic about *how* each stage is done — that lives in
`[[spec-driven-development]]`, `[[task-planning-decomposition]]`, `[[test-driven-development]]`,
`[[verification-and-debugging]]`, `[[code-review-discipline]]`, and `[[shipping-and-release]]`. This
guide owns the **sequencing, the gates, and the AI/ML-specific discipline** that wraps them.

### Why this matters MORE for AI infrastructure than for ordinary software

In a stateless web service, a mistake is usually cheap and reversible: roll back the binary, the error
budget absorbs a few bad minutes. AI infrastructure breaks both assumptions — **operations are expensive
and often irreversible.**

- **Cluster operations evict live work.** Draining a node pool, changing a taint, rolling a CNI or
  device-plugin DaemonSet, or resizing an accelerator pool can preempt running training jobs and
  long-lived inference replicas. A "quick" change to shared infra has a blast radius measured in
  workloads, not requests.
- **Compute is expensive and the feedback loop is long.** A large training or fine-tuning run burns
  thousands of accelerator-hours and can run for hours-to-weeks before you learn it diverged, OOM'd at
  step 40k, or trained on the wrong data. There is no cheap "just rerun it." A bad batch of jobs can
  vanish a quota allocation that took weeks to obtain.
- **Model rollouts degrade silently.** A new model or serving version that passes every infra health
  check — latency normal, no 5xx, GPUs warm — can still make answers measurably worse. Without an **eval
  gate** and quality monitoring you ship a regression that no dashboard catches until users complain.
- **Reproducibility is a first-class requirement.** "It worked on my pod" is worthless when the run
  depends on data snapshots, library/CUDA versions, RNG seeds, sharding, and accelerator topology. If
  you cannot reproduce a result you cannot verify it, and you cannot roll forward from it.

Skipping a stage doesn't save time on AI infra; it relocates the cost to an outage post-mortem or a
wasted compute bill. The lifecycle is the cheapest insurance you can buy.

## When to use this skill

Use it at the **start of any non-trivial change** to AI infra or an ML platform, including:

- Cluster / platform changes: node-pool or accelerator-pool changes, autoscaler tuning, scheduler/queue
  (e.g. quota or gang-scheduling) config, CNI/CSI/device-plugin upgrades, RBAC/network-policy edits.
- ML pipeline work: a new training or fine-tuning pipeline, a data-prep / tokenization stage, a
  checkpointing or eval harness change.
- Serving changes: a new model version, a serving-framework or runtime upgrade, batching/quantization
  changes, an autoscaling policy on inference.
- Platform components: a new controller/operator, a CRD, an admission webhook, a shared library.

Skip the heavyweight ceremony — but **not the stages** — for genuinely trivial, low-blast-radius work
(see Right-sizing). When in doubt about blast radius, treat it as bigger than it looks: shared AI infra
usually is.

## The process

Run the stages in order. **Do not advance past a gate that has not passed.** Each stage names the skill
that owns its execution.

### Stage 1 — Define  → `[[spec-driven-development]]`

**Purpose:** turn a vague ask into a written, agreed problem statement before any code or `kubectl`.

Produce, sized to the task:
- **Problem & scope** — what is broken / needed, and explicitly what is *out* of scope.
- **Requirements & SLOs** — correctness requirements; for serving, latency/throughput/availability
  targets; for training, throughput (e.g. tokens/sec or step time), convergence target, max wall-clock.
- **Eval / acceptance criteria** — the *measurable* definition of "better/correct." For ML this is an
  eval (quality metric on a held-out or canary set), not just "tests pass." Define it now; you will
  reuse it verbatim at the Verify gate.
- **Cost & capacity estimate** — accelerator-hours, peak accelerators, quota/region availability,
  storage and egress. A rough order-of-magnitude is fine; "unknown" is not.
- **Failure modes** — how this can fail and what the blast radius is (which workloads, which tenants).

**GATE (Define → Plan):** there is a written spec — problem, requirements/SLOs, **eval/acceptance
criteria, a cost estimate, and named failure modes/blast radius** — that a second person (or the
requesting stakeholder) agrees describes the right thing. No spec, no plan.

### Stage 2 — Plan  → `[[task-planning-decomposition]]`

**Purpose:** decompose the spec into small, sequenced, independently verifiable steps.

- Break the work into steps each small enough to build and verify on its own (ideally a single reviewable
  commit / a single deployable slice).
- **Order risky and expensive work first.** Put the experiment that could invalidate the whole approach,
  or the most expensive compute, **early and cheap** — a tiny smoke run before the full training run, a
  single-replica canary before a fleet rollout, a dry-run/diff before applying cluster changes. You want
  to fail (or de-risk) before you spend.
- Identify dependencies, the rollback story per step, and what evidence each step will produce.

**GATE (Plan → Build):** an ordered list of small verifiable steps exists, **risky/expensive steps are
sequenced first**, each step has a rollback or "abort cheaply" answer, and the plan is small enough that
no single step is a leap of faith.

### Stage 3 — Build  → `[[test-driven-development]]`

**Purpose:** implement the planned steps, **tests first**, in small commits.

- Write the test / check before (or alongside) the code: unit tests for logic; for infra, a way to
  assert the desired state (e.g. a dry-run/diff that shows exactly the intended change, a smoke job that
  exercises the new path).
- Keep commits small and self-describing; each should leave the tree green.
- Build the cheap, risky slice from the plan first; don't gold-plate ahead of the spec.

**GATE (Build → Verify):** the implemented slice does what the step intended, its tests pass locally,
and the change is small/reviewable. (This is *local* confidence; proving it in a realistic environment
is the next stage.)

### Stage 4 — Verify  → `[[verification-and-debugging]]`

**Purpose:** prove it actually works — beyond unit tests, in conditions close to production — and debug
any failure to its **root cause** (never paper over a symptom).

- **Integration / e2e:** run the real path. For pipelines, run end-to-end on a small but representative
  input. For cluster changes, apply in a non-prod/staging cluster (or a canary node pool) and observe
  real behavior, not just that `apply` succeeded.
- **Eval gate (AI/ML-specific, mandatory):** run the eval defined in Define against the acceptance
  criterion. A model/serving change that does not pass its eval has *failed Verify*, regardless of green
  infra metrics. Compare against a baseline; watch for silent quality regressions.
- **Reproducibility check (AI/ML-specific, mandatory):** confirm the result is reproducible — pinned
  data snapshot, pinned library/runtime/accelerator-stack versions, recorded seed, recorded config. A
  result you cannot reproduce is not verified.
- **Debug to root cause:** when something fails, find *why* — don't retry-until-green or loosen the
  threshold to make it pass.

**GATE (Verify → Review):** there is **evidence, not vibes** — integration/e2e passed, the **eval gate
passed against the stated criterion**, the result is **reproducible**, and any failure encountered was
fixed at its root cause. Attach the evidence (logs, eval numbers vs baseline, the repro recipe).

### Stage 5 — Review  → `[[code-review-discipline]]`

**Purpose:** an independent check for correctness, security, simplicity, and **blast radius** before
merge.

- **Correctness:** does it do what the spec says; are edge cases and failure modes handled.
- **Security / multi-tenancy:** least-privilege RBAC, no secrets in code/logs, tenant isolation
  preserved, no new public surface by accident.
- **Simplicity:** the simplest design that meets the spec — no speculative abstraction; reviewer can
  understand it.
- **Blast radius:** what does this touch that is *shared*? A change to a shared scheduler config,
  device plugin, or base image needs scrutiny proportional to the number of workloads it can affect.

**GATE (Review → Ship):** an approving review from someone who is not the author, with
correctness/security/simplicity/blast-radius considered. Unresolved blocking comments mean you go back
to Build.

### Stage 6 — Ship  → `[[shipping-and-release]]`

**Purpose:** release **progressively and reversibly**, with monitoring and a **tested** rollback.

- **Progressive rollout:** never flip 100% at once on shared infra. Canary first (one replica / one node
  pool / a small traffic %), bake, then ramp. For models, a canary/shadow or small-percentage rollout
  with the **eval/quality metric monitored live**, not just latency/error rate.
- **Monitoring:** the metrics that would reveal *this* change going wrong are wired up and watched —
  including quality/eval signals for model changes, not only infra health.
- **Tested rollback:** you have a rollback path **and you have confirmed it works** (e.g. the previous
  model version / config / image is retained and you've verified you can revert to it quickly). For
  irreversible operations (a destructive migration, deleting data, a quota change), Define should have
  flagged it and you take extra precautions — backups, a reversible staging, explicit sign-off.

**GATE (Ship → Done):** canary is healthy on both infra **and quality/eval** signals, rollback has been
proven (not assumed), monitoring/alerts are live, and the rollout completed (or is safely ramping). Then
update docs/runbooks and close the loop with the Define stakeholder.

### The loop is iterative

This is **not** a one-pass waterfall. Expect to cycle:
- Verify fails the eval → back to **Build** (fix) or **Define** (the criterion or approach was wrong).
- Review finds a correctness/blast-radius issue → back to **Build**, re-Verify.
- Canary degrades quality → roll back, back to **Verify/Build**, sometimes re-**Define**.
- A late discovery changes scope → revisit **Define** and re-Plan.

Cycling back is the lifecycle doing its job — catching the problem *before* the expensive, irreversible
step. Each loop should make the spec/eval sharper.

## Right-sizing the lifecycle

Scale the *depth* of Define/Plan/Review to the blast radius and cost. **Never** drop Verify safety
(at minimum a smoke test + the eval gate where a model is involved) or Ship safety (progressive +
rollback). Heuristic:

**Ask: if this is wrong, how many workloads/tenants break, how much compute is wasted, and is it
reversible?** The answer sets the depth.

| Task size | Examples | Define | Plan | Build | Verify | Review | Ship |
|-----------|----------|--------|------|-------|--------|--------|------|
| **Small** — low blast radius, reversible, cheap | one-line config/flag change, a log line, bump a non-breaking dep | one-line "why + risk" in the PR | trivial / in your head | TDD on the unit | smoke test; eval gate **if** it can affect model behavior | one reviewer | canary if it touches running workloads; otherwise standard deploy + rollback ready |
| **Medium** — affects one workload type or one tenant, moderate cost | new pipeline stage, autoscaler/quota tuning, serving-framework minor upgrade, a new controller flag | short written spec w/ eval criteria + cost estimate | ordered step list, risky-first | TDD, small commits | integration/e2e + **eval gate** + repro | correctness/security/simplicity + blast radius | progressive rollout, monitored, tested rollback |
| **Large** — shared infra, new component, or expensive/irreversible | new training pipeline, new platform component/operator/CRD, major serving migration, model family rollout, cluster-wide change | full spec: SLOs, evals, cost & capacity, failure modes, sign-off | full decomposition, de-risking spikes first, rollback per step | TDD, incremental, behind flags | full e2e + eval gate + repro + load/scale test + game-day for irreversible ops | multi-reviewer incl. domain + security; design review | staged rollout w/ bake times, live quality monitoring, proven rollback, runbook |

The cost of process should always be **smaller** than the cost of the failure it prevents. For a one-line
change that can't take anything down, a paragraph and a smoke test is the right amount. For a new
training pipeline, the full ceremony is cheap relative to the GPU-hours and time at stake.

## Rationalizations & rebuttals

- *"It's just a quick change, skip the process."* → Right-size it, don't skip it. A "quick" change to
  shared AI infra is exactly how clusters go down. Quick changes still get a smoke test and a rollback.
- *"I'll write the spec after I know it works."* → Then you have no acceptance criterion and no way to
  know it works. Define the eval/acceptance criteria *first*; that's what "works" means.
- *"Tests slow me down; I'll add them later."* → On a multi-hour training run, "later" means you discover
  the bug after burning the compute. Test-first is faster here, not slower.
- *"The unit tests pass, so it's verified."* → Unit-green is the Build gate, not the Verify gate. A model
  change with passing unit tests and a failing eval is a regression. Run the eval and the e2e path.
- *"Infra metrics are green, ship it."* → Green latency/error rates do not detect a quality regression.
  The eval/quality signal is part of the Ship gate for any model change.
- *"We can fix forward if it breaks."* → Not when the operation is irreversible or the run is expensive.
  Default to reversible + canary; earn fix-forward only where rollback is genuinely cheaper and safe.
- *"I can't reproduce it but the numbers look good."* → An unreproducible result is unverified. Pin the
  data/versions/seed and reproduce before you trust it.
- *"Rollback is obvious, I don't need to test it."* → Untested rollback is a hope, not a plan. Confirm
  the old version/config is retained and that you can actually revert.

## Red flags — stop and reconsider

- Jumping straight to **Build** (writing code or running `kubectl`) with **no written spec** and no eval
  criterion.
- The plan **front-loads the easy/cheap work** and defers the risky/expensive experiment to the end.
- "Verified" means **only unit tests** — no integration/e2e, no eval gate, no reproducibility recipe.
- A **model or serving change with no eval gate** — you have no way to detect a quality regression.
- A result you **cannot reproduce** being treated as done.
- **Shipping to 100% at once** on shared infra, or shipping with **no rollback / an untested rollback**.
- An **irreversible operation** (destructive migration, data deletion, quota change) treated like a
  routine deploy.
- Loosening an eval threshold or retrying-until-green to make Verify "pass."
- Review skipped, or self-reviewed, on a change with shared blast radius.
- The process feels heavier than the failure it prevents (you forgot to right-size **down**), *or*
  lighter than the blast radius warrants (you forgot to right-size **up**).

## Verification gate (definition of done for the whole lifecycle)

The change is done only when **every stage's gate is satisfied** and you can show the evidence:

- **Define:** written spec with requirements/SLOs, **eval/acceptance criteria, cost estimate, named
  failure modes & blast radius**, agreed by a second party.
- **Plan:** ordered small verifiable steps, **risky/expensive first**, each with a rollback/abort answer.
- **Build:** implemented test-first, small commits, tree green locally.
- **Verify:** integration/e2e passed; **eval gate passed vs the stated criterion and a baseline**; result
  **reproducible** (pinned data/versions/seed recorded); failures fixed at root cause. Evidence attached.
- **Review:** independent approval covering correctness, security/multi-tenancy, simplicity, blast radius.
- **Ship:** progressive rollout completed/ramping; **canary healthy on infra *and* quality/eval
  signals**; **rollback proven**; monitoring/alerts live; runbook/docs updated; loop closed with the
  Define stakeholder.

If any gate is unmet, the work is not done — you are mid-loop. Report honestly which gate you are at.

## Canonical references

- Google SRE Book & SRE Workbook — error budgets, progressive rollouts, canarying, rollback discipline:
  https://sre.google/books/
- Google "Testing on the Toilet" / Software Engineering at Google (Ch. on testing, code review, large-
  scale changes): https://abseil.io/resources/swe-book
- "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., NeurIPS 2015) — why ML systems
  rot silently: https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- "The ML Test Score" (Breck et al., 2017) — a rubric for ML production-readiness:
  https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
- "Reliable ML through monitoring" / continuous delivery for ML (CD4ML), Martin Fowler:
  https://martinfowler.com/articles/cd4ml.html
- Note (2026): the AI-infra ecosystem moves fast — verify current docs for whatever scheduler, serving
  stack, and eval harness you depend on; the lifecycle is stable, the tools under it are not.

---

# Engineering Lifecycle — Worked Examples

## End-to-end: adding a new TPU training pipeline for a fine-tuned model

A platform team needs a new pipeline that fine-tunes a base model on a fresh dataset, on a TPU pod
slice, and produces a checkpoint that serving can pick up. This is a **large** task (new pipeline,
expensive compute, shared accelerators), so the full lifecycle applies. Below is each stage, what it
produced, and the gate that let it advance.

### Define → `[[spec-driven-development]]`

What was written down:
- **Problem & scope:** add a reproducible fine-tuning pipeline for model X on dataset Y; out of scope:
  changing the base model, changing the serving stack.
- **Requirements/SLOs:** training throughput ≥ a target step time on the chosen TPU slice; full run
  wall-clock under a stated budget; checkpoints written every N steps to durable storage.
- **Eval / acceptance criteria:** the fine-tuned checkpoint must beat the current production model by
  ≥ X on the agreed held-out eval set, with **no regression** on a guardrail/safety eval. *(This exact
  criterion is reused at the Verify gate — see `[[ml-evaluation-evals]]`.)*
- **Cost & capacity estimate:** ~Z accelerator-hours per full run × expected iterations; peak = one pod
  slice of size S; confirmed quota and region availability; storage and egress for checkpoints + data.
- **Failure modes / blast radius:** divergence late in training (wasted compute), preempting other jobs
  by over-requesting quota, a bad checkpoint silently picked up by serving.

> **GATE passed:** a spec with requirements/SLOs, a measurable eval criterion, a cost estimate, and named
> failure modes — reviewed and agreed by the requesting stakeholder. *Without this, there is no
> definition of "the run worked."*

### Plan → `[[task-planning-decomposition]]`

Decomposed into small, sequenced steps, **risky/expensive first**:
1. **De-risk cheap:** a single-step smoke run on a *minimal* TPU slice that exercises data loading,
   sharding, the checkpoint write, and one eval call — to catch config/topology bugs for cents, not
   thousands of accelerator-hours.
2. A short partial run (a few hundred steps) to confirm loss decreases and throughput hits target.
3. The pipeline scaffolding: data snapshot pinning, deterministic seeding, checkpoint cadence.
4. The full run, behind a flag, writing to a staging checkpoint location.
5. Wiring the eval gate to run automatically on each produced checkpoint.

Each step has an abort-cheaply answer; the most expensive step (the full run) is last and gated by all
the cheap de-risking steps.

> **GATE passed:** ordered step list with the riskiest/most expensive work sequenced first and a cheap
> way to fail early.

### Build → `[[test-driven-development]]`

- Wrote unit tests first for the data-prep transform and the config/sharding logic, then implemented.
- Added an assertion-style smoke check for the checkpoint round-trip (write → read → shape/values).
- Small commits, tree green locally; the smoke run from step 1 was run before anything bigger.

> **GATE passed:** the smoke and partial-run slices do what the plan intended; unit tests pass locally;
> commits are small and reviewable. *(Local confidence only — proving the full run is the next stage.)*

### Verify → `[[verification-and-debugging]]`

- **Integration/e2e:** ran the full pipeline end-to-end on the real TPU slice to a real checkpoint.
- **Eval gate (mandatory):** ran the Define eval — the checkpoint beat production by ≥ X **and** held the
  guardrail eval. (First attempt regressed the guardrail; debugged to a data-filtering bug — **fixed at
  root cause**, not by loosening the threshold — then re-ran.)
- **Reproducibility (mandatory):** pinned the data snapshot, recorded library/runtime/XLA and accelerator
  versions, recorded the seed and full config; reran and got matching metrics within tolerance.

> **GATE passed:** e2e green, **eval gate passed vs the stated criterion and the production baseline**,
> result **reproducible** with a recorded recipe, the one failure fixed at root cause. Evidence (eval
> numbers vs baseline, the repro recipe, logs) attached. *Unit-green alone would NOT have passed this.*

### Review → `[[code-review-discipline]]`

- Independent reviewer checked correctness (sharding, checkpoint cadence, eval wiring), **security/
  multi-tenancy** (least-privilege access to data and storage, no secrets in config/logs), **simplicity**
  (no speculative abstraction), and **blast radius** (quota request sized so it cannot starve other
  tenants; the new checkpoint path is staging, not the live serving path yet).

> **GATE passed:** approving review from a non-author covering correctness, security, simplicity, and
> blast radius; blocking comments resolved.

### Ship → `[[shipping-and-release]]`

- **Progressive & reversible:** the new checkpoint is promoted to serving via a **canary** — small
  traffic percentage / shadow first — with the **quality/eval metric monitored live**, not just latency
  and error rate. Previous model version retained.
- **Tested rollback:** confirmed the prior checkpoint is retained and that reverting serving to it is
  fast; rehearsed the revert before ramping.
- Ramped only after the canary held both infra **and** quality signals; updated the runbook and closed
  the loop with the Define stakeholder.

> **GATE passed:** canary healthy on infra **and** eval/quality signals, rollback proven (not assumed),
> monitoring/alerts live, rollout ramped, runbook updated.

### The loop in action

The guardrail-eval failure in Verify sent the work back to **Build** (fix the data filter), then
re-Verify — *before* the full expensive run was trusted or shipped. That cycle is the lifecycle paying
for itself: a silent quality regression caught at the eval gate instead of in production.

---

## Right-sizing the same discipline to other tasks

The depth scales with blast radius and cost; the **Verify eval gate and Ship rollback never drop**.

| Task | Size | What the lifecycle looks like |
|------|------|-------------------------------|
| Bump a non-breaking library version in a serving image | **Small** | One-line "why + risk" in the PR; smoke test; **eval gate because it can change model behavior**; one reviewer; canary the image, rollback ready. |
| Tune an autoscaling/quota policy on an inference deployment | **Small–Medium** | Short spec w/ the SLO it targets + the risk; smoke in staging; verify it doesn't starve other workloads; one reviewer focused on blast radius; progressive rollout, watch saturation, tested rollback. |
| Add a new data-prep / tokenization stage to an existing pipeline | **Medium** | Written spec w/ eval criteria + cost; risky-first plan (tiny sample first); TDD on the transform; **e2e + eval gate + repro** (does downstream quality hold?); correctness/security review; progressive rollout behind a flag, rollback ready. |
| Roll out a new model-serving version (new weights) | **Medium–Large** | Full spec w/ eval criteria, latency/throughput SLOs, cost; plan w/ shadow eval first; build/verify the serving path; **eval gate vs baseline + repro**; review incl. blast radius; **canary/shadow rollout with live quality monitoring + proven rollback** — the silent-regression case, so the eval gate is the whole point. |
| Upgrade a shared device-plugin / CNI DaemonSet across the cluster | **Large** | Full spec w/ failure modes & blast radius; plan that rolls one node pool first; dry-run/diff in Build; verify on a canary pool incl. a real workload; multi-reviewer incl. security; **node-pool-by-node-pool progressive rollout, bake times, proven rollback, game-day for the eviction risk.** |
| New platform component (controller/operator + CRD) | **Large** | Full ceremony: design review in Define, decomposition w/ de-risking spikes, TDD, e2e + scale test + (eval gate if it touches model behavior) + repro, multi-reviewer + security review, staged rollout w/ runbook and proven rollback. |
| One-line, low-blast-radius config (e.g. a log level, a non-behavioral comment) | **Small** | Paragraph of "why + risk"; smoke test; standard deploy with rollback ready. No eval gate needed if it genuinely cannot affect model behavior — but if in doubt, run it. |

Rule of thumb: the cost of the process should be **smaller than the cost of the failure it prevents** —
right-size **down** for trivial reversible changes and **up** for anything that touches shared infra,
burns serious compute, or can degrade quality silently.
