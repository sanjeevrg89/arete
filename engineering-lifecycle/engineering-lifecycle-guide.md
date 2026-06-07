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
