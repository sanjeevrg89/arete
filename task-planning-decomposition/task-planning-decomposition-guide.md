# Task Planning & Decomposition — Guide

The **Plan** stage of `[[engineering-lifecycle]]`. You have a reviewed spec (the *what* and *why*, from
`[[spec-driven-development]]`). Planning produces the *how*: an ordered list of small, independently
verifiable steps, with the riskiest and most expensive work pulled to the front, written down and
reviewed **before** you start Building.

The bar: a senior engineer would not spin up a 64-GPU job, or write a thousand-line pipeline, off a plan
that lives in their head. Planning is where you convert uncertainty into a sequence you can execute and
checkpoint — and where a reviewer can catch a doomed approach for the price of reading a page.

---

## Overview — why plan-then-execute

Diving straight into code feels fast and often is, for a tiny, fully-understood change. For anything
non-trivial — and almost everything in AI infra / ML is non-trivial because it touches distributed
systems, accelerators, and long-running jobs — planning first wins for three concrete reasons:

1. **It surfaces unknowns early, while they're cheap.** The plan forces you to name every step. The act
   of naming exposes the steps you don't actually know how to do ("step 4: get NCCL to use the RDMA
   NICs"). Finding that gap on paper costs an hour. Finding it at hour 30 of a build, after you've laid
   foundations that assume it works, costs the foundations too.

2. **It lets reviewers critique the approach, which is cheap, before the work, which is expensive.**
   Reviewing a plan is reading a page and asking "why this order? what if the collective doesn't hit
   bandwidth? where's the checkpoint?" That is minutes of a colleague's time. Reviewing the *outcome* of
   a wrong approach means re-doing days of work and re-spending the compute. The asymmetry is the whole
   argument: **the approach is reviewable for ~1000× less than the work.**

3. **It creates checkpoints.** A plan of discrete steps is a series of natural stopping points. You can
   stop after step 3, hand off, get preempted, or re-evaluate — and resume with full context. A
   monolithic "build the thing" has no safe stopping point until it's done or abandoned.

The cost of planning is real but small and front-loaded; the cost of *not* planning is large, variable,
and back-loaded (it lands as rework, wasted compute, and integration surprises). On expensive work, the
expected value of planning is strongly positive.

### When to use this skill

- Any task estimated at more than a few hours, or that will consume meaningful compute.
- Anything multi-node, multi-component, or with a long-running step (training, large eval, data jobs).
- Anything with a genuine unknown ("will this fit in memory", "will this saturate the fabric").
- Anything irreversible or expensive to redo (a migration, a big experiment sweep).

When to skip the formality: a one-line bug fix, a config tweak, a change you've made ten times. Even
then, a 30-second mental sequence-and-done-signal is the same discipline, just compressed.

---

## Decomposition — breaking the work into verifiable steps

A good step has three properties: it is **small**, it has a **clear done-signal**, and it is a
**vertical slice** where possible.

### Small enough to verify independently

The unit of a plan is a step you can *finish and check* on its own. The test for "small enough": can you
state, in one sentence, the observation that tells you the step is done and correct? If you can't, the
step is too big or too fuzzy — split it until each piece has an answer.

Rough sizing heuristics (calibrate to your context):

- A step should be reviewable/verifiable in well under a day of work.
- If a step contains the word "and" joining two unrelated outcomes, it's probably two steps.
- If a step's done-signal is "it works", it's not decomposed — *how would you observe* that it works?

### Each step has a clear "done" signal

The done-signal is what makes a step verifiable and what makes the plan honest. It must be **observable
and specific** — a command and its expected output, a metric crossing a threshold, an artifact existing,
a test passing. Examples of good vs weak signals:

| Step | Weak signal | Good done-signal |
|------|-------------|------------------|
| Stand up single-GPU training | "training runs" | `python train.py --steps 50` exits 0; loss decreases over the 50 steps; one checkpoint written to the bucket |
| Verify the collective | "NCCL works" | `all_reduce` of a 1 GB tensor across 2 nodes completes; measured busbw ≥ 80% of the link's line rate |
| Add data loader | "data loads" | one batch has the expected shape/dtype; throughput ≥ N samples/s sustained; no host-OOM over 200 batches |

Where the done-signal is "a test passes", that test is written and run under `[[test-driven-development]]`
— Plan decides *that* a step exists and how it's verified; TDD executes the step.

### Vertical slices over big-bang

Two ways to decompose a system:

- **Horizontal (component-first):** build the data loader fully, then the model fully, then the training
  loop fully, then the checkpointer — and integrate at the end. *Avoid this.* All the risk (integration,
  shapes, device placement, distributed semantics) is concentrated in the last, most expensive phase.
- **Vertical (slice-first):** get the thinnest possible end-to-end path working first — a tiny model, a
  handful of synthetic samples, one GPU, 50 steps, one checkpoint — then thicken each layer. *Prefer
  this.* The first slice proves the path exists and de-risks integration before you scale.

> Integration risk is *the* risk in ML infra. A vertical slice attacks it first; a horizontal
> decomposition defers it to the worst possible moment.

The first vertical slice often doubles as your smoke test for the rest of the project: every later step
re-runs it (or a scaled version) as part of its done-signal.

---

## Sequencing for AI infra

Decomposition gives you a *set* of steps. Sequencing gives you the *order*. For AI infra the ordering
rule is not "easiest first" or "in the order I'll write the files" — it is **de-risk before you commit
the expensive resource.**

### Riskiest / most uncertain / most expensive first

Order so that the step most likely to **invalidate the plan** comes as early as its dependencies allow.
Three overlapping notions of "do this first":

- **Most uncertain** — the step you're least sure will work (a new kernel, an untested topology, a
  library version you haven't run). If it fails, you want to know before building on it.
- **Riskiest** — the step whose failure would force the biggest re-plan or has no fallback.
- **Most expensive** — the step that, once committed, burns the most GPU-hours / dollars / wall-clock.
  You must *not* discover a blocker after spending the budget.

These usually point the same direction: **prove the scary, expensive thing works at the smallest scale
first.** Concretely, before you launch a multi-day 64-GPU run, you should already have proven — cheaply —
that the collectives saturate the fabric, the model+optimizer state fits in memory at your sharding, the
checkpoint writes and *restores*, and one step's math is correct. Each of those is a small, early step.

**Anti-pattern:** leaving "does multi-node networking even work" as a late step. Network/fabric/driver
issues are the single most common reason multi-node jobs die, and they invalidate everything built on
top. That verification belongs near step 1.

### Spikes and prototypes for the unknowns

When a step depends on something you genuinely don't know, don't guess and build — **spike it.** A spike
is a small, timeboxed, throwaway experiment whose only deliverable is *information*: does the kernel fit,
does the API behave as documented, does the collective hit target bandwidth, does the data source have
the schema you assumed.

- **Timebox it** (e.g. half a day). The output is an answer, not production code.
- **Keep it separate** from the build — it's allowed to be ugly; it will be thrown away.
- **The result feeds the plan:** the spike either confirms an assumption (proceed) or refutes it
  (replan, now, while it's cheap).

Decide *what to prototype vs build* by uncertainty × cost: high uncertainty and high downstream cost →
spike first. Low uncertainty (you've done it before) → just build it.

### Dependency ordering

Steps form a DAG: some can't start until others finish (you can't verify checkpoint *restore* before
checkpoint *write* exists). Produce a topological order — but treat it as a *constraint*, not the plan.
Among all steps currently unblocked, pull the **riskiest** one forward. Topological order tells you
what's *legal*; risk tells you what's *next*.

When risk and dependency conflict — the scary step is buried behind prerequisites — see if you can pull
the risk forward another way: a spike that exercises the scary part in isolation, on stubs, without the
full prerequisite chain.

### Plan for partial failure

Long, expensive, or distributed steps *will* be interrupted — preemption, OOM, a node falling out, a
transient fabric error. A plan that assumes every step runs to completion uninterrupted is fragile.
Build resilience into the steps themselves:

- **Checkpoint state** so an interrupted step resumes near where it stopped rather than from zero. For
  training/long ML jobs this means real, frequent, *restorable* checkpoints — see
  `[[ml-checkpointing-orbax]]`. "We checkpoint" is not enough; **restore must be a verified step.**
- **Make steps idempotent** so re-running after a partial failure is safe and converges to the same
  state — no double-writes, no corrupt partial outputs, no "it half-ran and now the dataset is in a
  weird state". Write outputs to a temp path and atomically rename; key outputs by content/run-id;
  design so a retry overwrites rather than appends. These are core distributed-systems properties —
  see `[[distributed-systems-fundamentals]]` for idempotency, retries, and at-least-once reasoning.
- **Define the resume point** for any multi-hour step in the plan: if it dies at 60%, what do you run to
  continue, and how do you know it's safe?

A useful test: for each expensive step, ask "if this is killed halfway, what happens when I re-run it?"
If the answer is "I don't know" or "bad things", that step isn't ready to be in the plan yet.

---

## The plan as a living checklist

The plan is an artifact, not a vibe. It has a lifecycle of its own.

1. **Write it down** where reviewers and your future self will see it — the design doc, the tracking
   issue, the project board. The format is simple: an ordered list of steps, each with its done-signal,
   its dependencies, and a rough effort/cost/risk note.
2. **Get the approach reviewed** before Building (see the gate). The reviewer is checking the *order* and
   the *unknowns*, not the code: "why is the expensive run before the bandwidth check?", "what's the
   resume story for step 5?", "is step 3 really one step?".
3. **Track progress against it.** Check steps off as their done-signals are met. The checklist is your
   single source of truth for "where are we".
4. **Update it as you learn.** Reality will contradict the plan — a spike refutes an assumption, a step
   splits in two, a dependency you missed appears. Edit the plan in the open; a stale plan is worse than
   no plan because people trust it.
5. **Know when to replan.** A done-signal you can't meet, or a result that contradicts an assumption the
   plan rested on, is a *replan trigger* — not a reason to push harder on the existing plan. Stop, revise
   the sequence, re-review if the change is material.

### Estimating effort, cost, and risk per step

Annotate each step with three quick estimates. They don't need to be precise; they need to be *present*,
because they drive sequencing and the spike-vs-build call.

- **Effort** — rough time (hours/days). Surfaces steps that are secretly huge (and should be split).
- **Cost** — for ML, estimate compute: `GPU-hours × $/GPU-hr`, plus data egress / storage where it
  matters. A step that costs $5 and one that costs $5,000 deserve very different scrutiny and ordering.
- **Risk** — confidence the step will work as planned (high/med/low). Low-confidence + high-cost is the
  combination that *must* be de-risked with a spike before you commit.

The estimates are also how you catch a plan that's upside-down: if your highest-cost step is also
low-confidence and scheduled last, the sequence is wrong.

---

## The process (numbered)

1. **Start from the reviewed spec.** Confirm the *what/why* is settled (`[[spec-driven-development]]`).
   Planning a moving spec wastes the plan. If the spec is unclear, go back — don't paper over it.
2. **Enumerate the steps.** Brain-dump every step needed to satisfy the spec. Don't order yet; just get
   them all on paper. Naming them is what exposes the unknowns.
3. **Decompose until each step is verifiable.** Split anything you can't attach a concrete done-signal
   to. Reshape toward **vertical slices** — make sure a thin end-to-end path is one of the early steps.
4. **Attach a done-signal to every step.** A command + expected output, a metric + threshold, an
   artifact. If you can't write one, the step isn't ready.
5. **Flag the unknowns and add spikes.** For each genuinely uncertain step, decide prototype-vs-build;
   add a timeboxed spike where uncertainty × cost is high.
6. **Estimate effort / cost / risk per step.** Especially compute cost. Mark the low-confidence,
   high-cost steps.
7. **Sequence: dependencies, then risk.** Topologically order, then within what's unblocked pull the
   riskiest / most uncertain / most expensive forward. De-risk before committing GPU-hours.
8. **Make expensive/long steps resilient.** Add checkpointing and idempotency; define the resume point
   for every multi-hour step (`[[ml-checkpointing-orbax]]`, `[[distributed-systems-fundamentals]]`).
9. **Write the plan down** as an ordered checklist with done-signals, dependencies, and estimates, in a
   place reviewers will see.

   --- **CHECKPOINT — approach review gate** ---

10. **Get the approach reviewed before Build.** A second person sanity-checks the *sequence, the
    unknowns, and the resilience* — not the code. Resolve their concerns or record why not.
11. **Execute against the checklist**, checking off done-signals, updating the plan as you learn, and
    **replanning** the moment a result contradicts an assumption the plan rested on.

The checkpoint at step 9→10 is the load-bearing gate: **no Building until a written, reviewed plan of
verifiable steps exists with the risky/expensive work first.**

---

## Rationalizations & rebuttals

- *"The task is obvious, I'll just start."* If it's truly obvious, the plan takes five minutes and costs
  nothing. If it isn't, "just start" is how you discover the hard part at hour 30. The feeling of
  obviousness is exactly when unexamined assumptions hide.
- *"Planning is overhead / slows me down."* It front-loads a small cost to avoid a large, variable one
  (rework + wasted compute). On expensive work the math is not close — reviewing the approach is ~1000×
  cheaper than re-doing the work.
- *"I'll keep the plan in my head."* Then it can't be reviewed, can't be checked off, can't survive a
  preemption or a handoff, and drifts without you noticing. A plan in your head is not a plan; it's an
  intention.
- *"I'll figure out the order as I go."* Order is the whole value of planning — doing the risky step
  last is the default failure mode you're trying to prevent. Decide the order while it's still free to
  change.
- *"Let's just launch the big run and see."* That's spending the budget to acquire information you could
  have bought with a $5 spike. Prove it at the smallest scale first.
- *"We'll add checkpointing later."* Later is after the first multi-hour run dies at 90%. Resilience is a
  property of the step, designed in, not bolted on.
- *"The spec might change, so why plan."* A reviewed spec is stable enough to plan against; if it's that
  unstable, the problem is the spec, not the plan — fix it upstream.

---

## Red flags — stop and reconsider

- A step is too big to attach a single concrete done-signal to ("build the training pipeline").
- The most expensive or most uncertain step is scheduled in the back half of the plan.
- There is no spike for a step you've never done before, on a path you've never run.
- The plan exists only in your head / in a chat thread, never written as a checklist.
- An unknown is unflagged — the plan reads as if everything is certain.
- A multi-hour or multi-node step has no checkpoint and no answer to "what if it dies at 60%?".
- Steps aren't idempotent — re-running after a partial failure would corrupt or duplicate state.
- The approach was never reviewed by anyone but the author before Building started.
- Decomposition is horizontal (every component in isolation, integrate last).
- A result already contradicts an assumption the plan rests on, and you're pushing on anyway.

---

## Verification gate (definition of done for the Plan stage)

The Plan stage is complete only when **all** of these are true:

- [ ] A **written plan** exists as an ordered checklist, in a place reviewers can see it.
- [ ] Every step is **small enough to verify independently** and has a **concrete done-signal**
      (command + output, metric + threshold, or artifact).
- [ ] Decomposition favors **vertical slices**; a thin end-to-end path is an early step.
- [ ] The **riskiest / most uncertain / most expensive** steps are sequenced **first** (within
      dependency constraints); de-risking precedes committing GPU-hours.
- [ ] Genuine **unknowns are flagged** and addressed with timeboxed **spikes** where uncertainty × cost
      is high.
- [ ] Each step has an **effort / cost / risk** estimate; low-confidence + high-cost steps are
      explicitly de-risked first.
- [ ] Long/expensive/distributed steps are **resilient to partial failure** — checkpointed, idempotent,
      with a defined resume point.
- [ ] The **approach has been reviewed** by someone other than the author, and concerns are resolved or
      recorded.

Only then proceed to Build.

---

## Canonical references

Real, authoritative sources on planning, decomposition, and de-risking. The ML ecosystem moves fast
(it is 2026) — verify tool-specific details against current docs.

- Fred Brooks, *The Mythical Man-Month* — plans, estimation, and the cost of late integration.
- Tom Gilb, *Principles of Software Engineering Management* — evolutionary delivery / vertical slices.
- Martin Fowler, "SacrificialArchitecture" and "Spike" notes — https://martinfowler.com/bliki/
- Google SRE Book, "Postmortem Culture" & "Managing Critical State" — partial-failure and idempotency
  reasoning — https://sre.google/books/
- Orbax checkpointing docs (for resumable ML steps) — https://orbax.readthedocs.io/
- See also `[[engineering-lifecycle]]`, `[[spec-driven-development]]`, `[[test-driven-development]]`,
  `[[ml-checkpointing-orbax]]`, `[[distributed-systems-fundamentals]]`.
