---
name: task-planning-decomposition
description: The Plan stage of the engineering lifecycle — turn a reviewed spec into a sequenced set of
  small, independently verifiable steps before writing code or burning compute. Use when starting any
  non-trivial AI-infra / ML task (stand up multi-node training, build a data/eval pipeline, migrate a
  serving stack, run an expensive experiment) and you are tempted to "just start coding". Covers
  decomposition into vertical slices with per-step done-signals, riskiest/most-expensive-first
  sequencing to de-risk before committing GPU-hours, spikes/prototypes for unknowns, planning for
  partial failure (checkpointing, idempotent steps), the plan as a living reviewed checklist, and the
  approach-review gate before Build. Triggers: "make a plan", "how should I break this down", "where do
  I start", task estimation, sequencing dependencies, scoping a spike.
---

# Task Planning & Decomposition

Apply the judgment of an engineer who has shipped large ML-infrastructure projects: someone who has
watched a "should be quick" multi-node training job burn a week of cluster time because the plan lived
in someone's head and the riskiest piece was tackled last. **Plan the approach (cheap) before doing the
work (expensive). De-risk before you commit GPU-hours.**

This is the **Plan** stage of `[[engineering-lifecycle]]`: it sits after the spec is reviewed
(`[[spec-driven-development]]`) and before you Build. The output is a written, reviewed plan — an
ordered list of small steps, each with a clear "done" signal — not code.

## How to use this skill

1. **Read `task-planning-decomposition-guide.md`** in this directory — the full process and the
   decomposition/sequencing rules. Apply it to the task at hand.
2. For a fully worked decomposition (a multi-node training task broken into riskiest-first verifiable
   steps) and a reusable planning-checklist template, read **`examples.md`**.
3. Match the team's existing planning artifacts (design doc, tracking issue, project board) — write the
   plan where reviewers will actually see it. Apply the de-risking and written-plan rules regardless.

## Essentials (full detail in `task-planning-decomposition-guide.md`)

- **Plan-then-execute beats diving in.** Planning surfaces unknowns early, lets reviewers critique the
  *approach* (minutes) before the *work* (days + dollars), and creates checkpoints you can stop at.
- **Decompose into steps small enough to verify independently.** Every step has a concrete done-signal
  (a command output, a metric, an artifact). If you can't state how you'd verify a step, it's too big
  or too vague — split it.
- **Vertical slices over big-bang.** Prefer a thin end-to-end path that runs first (1 step on 1 GPU)
  over building every component in isolation and integrating last. Integration risk is the risk.
- **Sequence riskiest / most uncertain / most expensive first.** Do the step most likely to invalidate
  the plan before you build on top of it. Never let "will RDMA/NCCL even work across these nodes" be
  step 9.
- **Spike the unknowns.** For anything you genuinely don't know (does this kernel fit in memory, does
  the collective hit target bandwidth), write a small throwaway prototype to buy information — timeboxed,
  separate from the build.
- **Order by dependency, then by risk within what's unblocked.** A topological order is necessary, not
  sufficient — among ready steps, pull the scary one forward.
- **Plan for partial failure.** Long/expensive steps must be resumable: checkpoint state
  (`[[ml-checkpointing-orbax]]`) and make steps idempotent so a re-run is safe
  (`[[distributed-systems-fundamentals]]`). Assume preemption, OOM, and node failure.
- **The plan is a living checklist.** Write it down, get the approach reviewed, check off steps, and
  update it as reality teaches you. A plan in your head is not a plan.
- **Estimate effort / cost / risk per step** — especially compute cost (GPU-hours × rate). The estimate
  is the input to sequencing and to the prototype-vs-build decision.
- **Know when to replan.** When a step's outcome contradicts an assumption the plan rested on, stop and
  revise the plan — don't grind forward on a plan you know is wrong.

## The gate (before Build)

A written plan of discrete, independently verifiable steps exists; the riskiest/most-expensive work is
sequenced first; unknowns are flagged with spikes; and **the approach has been reviewed** by someone
other than the author. Only then start Building.

## Related skills

- `[[engineering-lifecycle]]` — the surrounding stages; this is the Plan stage.
- `[[spec-driven-development]]` — produces the reviewed spec this stage consumes.
- `[[test-driven-development]]` — within a step, the done-signal is often a test; Plan decides the steps,
  TDD executes one.
- `[[ml-checkpointing-orbax]]` — checkpointing that makes expensive steps resumable.
- `[[distributed-systems-fundamentals]]` — idempotency, retries, and partial-failure reasoning.

---

# Reference — task-planning-decomposition

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

---

# Examples — Task Planning & Decomposition

Worked, imitatable artifacts for the **Plan** stage. The first is a full decomposition of a real
AI-infra task into riskiest-first, verifiable steps. The second is a reusable planning-checklist
template you can copy into a design doc or tracking issue.

---

## Example 1 — Decomposing "stand up multi-node training"

**Reviewed spec (input):** Train model *M* (≈7B params) to convergence on dataset *D* across **4 nodes ×
8 GPUs (32 GPUs)** with FSDP-style sharding, restartable across preemption, checkpoints to object
storage, target ≥ X tokens/s/GPU. (The *what/why* came from `[[spec-driven-development]]`; here we plan
the *how*.)

### Anti-pattern: the order most people reach for (horizontal, scary-thing-last)

> 1. Write the full data pipeline.  2. Write the full model + sharding.  3. Write the training loop.
> 4. Add checkpointing.  5. **Launch the 32-GPU run.**  6. (Only now) discover NCCL won't use the RDMA
> NICs / the run OOMs / checkpoints don't restore — after burning hours of 32-GPU time.

Everything risky (multi-node collectives, memory fit at scale, checkpoint *restore*, throughput) is
concentrated in the final, most expensive step. A blocker there invalidates all the work beneath it.

### The plan: vertical slice first, then riskiest/most-expensive next

Each step lists **deps**, a **done-signal** (observable + specific), and **effort / cost / risk**. Note
the order: a thin end-to-end slice proves the path exists, then we de-risk the things that would kill a
32-GPU run *at small scale* before committing the budget.

| # | Step | Deps | Done-signal (how you know it's done & correct) | Eff / Cost / Risk |
|---|------|------|-----------------------------------------------|-------------------|
| 0 | **Spike: cluster + fabric sanity** | — | `nccl-tests` `all_reduce` of 1 GB across 2 nodes completes; measured busbw ≥ 80% of link line rate. Throwaway. | 0.5d / ~2 GPU-hr / **LOW conf — do first** |
| 1 | **Vertical slice: 1-GPU train, 50 steps** | — | `train.py --steps 50` on a tiny config exits 0; loss strictly decreases over 50 steps; 1 checkpoint written to the bucket. | 1d / ~1 GPU-hr / med |
| 2 | **Spike: memory fit at target sharding** | 1 | Single-node 8-GPU FSDP run holds model+optimizer+activations within HBM at the planned shard/precision; no OOM over 100 steps; record peak mem. | 0.5d / ~8 GPU-hr / **LOW conf** |
| 3 | **Multi-node 2-node (16-GPU) run** | 0,1,2 | 16-GPU run completes 200 steps; loss curve matches single-node within tolerance; tokens/s/GPU measured (early read on the throughput target). | 1d / ~30 GPU-hr / **HIGH risk** (real multi-node) |
| 4 | **Checkpoint write + RESTORE verified** | 1,3 | Kill the run mid-step; resume from last checkpoint; loss/step continues from the saved point (not from zero); weights bit-for-bit (or within tolerance) match. Uses `[[ml-checkpointing-orbax]]`. | 1d / ~10 GPU-hr / **HIGH risk** |
| 5 | **Idempotent / preemption-safe launch** | 4 | Job re-submitted after a simulated preemption converges to same state; outputs written to temp path + atomic rename; no duplicate/corrupt checkpoints. See `[[distributed-systems-fundamentals]]`. | 0.5d / ~5 GPU-hr / med |
| 6 | **Scale to 4 nodes (32 GPUs), short run** | 3,4,5 | 32-GPU run holds 500 steps; tokens/s/GPU ≥ X (the spec target); scaling efficiency vs 16-GPU within tolerance; one preempt/resume cycle survives. | 1d / ~60 GPU-hr / med |
| 7 | **Full convergence run** | 6 | Train to target metric on *D*; periodic checkpoints; survives ≥1 real preemption; final eval meets spec. **Most expensive — last on purpose, because everything that could kill it is already proven.** | days / **largest cost** / low (de-risked) |

### Why this order

- **Spikes (0, 2) come first** because they're the lowest-confidence, and their failure would replan the
  whole approach. They're cheap (a few GPU-hours) and throwaway — they buy *information* before the
  budget is committed.
- **The vertical slice (1)** proves an end-to-end path before any component is "finished" — integration
  risk attacked immediately.
- **Multi-node (3) and checkpoint-restore (4) are pulled forward** because they are the classic
  multi-node-job killers. We prove them at 16 GPUs, not by discovering them at 32.
- **The big expensive run (7) is last** not because it's "the end" but because by then every failure mode
  that could waste its budget is already verified at small scale. That's the entire point of
  riskiest/most-expensive-first: **the cheap steps de-risk the expensive one.**

### Partial-failure design baked in

Steps 4 and 5 exist *specifically* so that the multi-day step 7 can survive preemption. "Restore" is its
own verified step (4) — not an assumption — and idempotency (5) means a re-submit after a node failure is
safe. Before step 7 launches, the answer to "what if it dies at 60%?" is written down and tested.

### Replan triggers (decided up front)

- Spike 0 shows busbw far below line rate → fabric/driver problem; replan networking before anything else.
- Spike 2 OOMs at target sharding → revisit sharding/precision/offload in the spec before continuing.
- Step 3 throughput far below X → renegotiate the target or the parallelism strategy *now*, not after the
  full run.

---

## Example 2 — Planning-checklist template

Copy this into the design doc / tracking issue. Fill it in, get the **approach** reviewed, then Build.

```markdown
# Plan: <task name>

Spec: <link to reviewed spec — the what/why>            Owner: <name>    Reviewer: <name>
Status: DRAFT | IN REVIEW | APPROVED | IN PROGRESS | DONE

## Unknowns (flag everything you're not sure of)
- [ ] <unknown 1> → spike? Y/N  (timebox: __)
- [ ] <unknown 2> → spike? Y/N  (timebox: __)

## Steps  (ordered: dependencies, then riskiest/most-expensive first)
| # | Step (vertical slice where possible) | Deps | Done-signal (command+output / metric+threshold / artifact) | Effort | Cost (GPU-hr × $) | Risk (H/M/L) |
|---|--------------------------------------|------|------------------------------------------------------------|--------|-------------------|--------------|
| 0 | spike: <riskiest unknown>            | —    | <what answer ends the spike>                               |        |                   | L conf       |
| 1 | thin end-to-end slice                | —    |                                                            |        |                   |              |
| 2 | de-risk: <expensive thing, small scale> |   |                                                            |        |                   |              |
| … |                                      |      |                                                            |        |                   |              |
| N | full / expensive run (last)          |      |                                                            |        | largest           | L (de-risked)|

## Partial-failure design
- Long/expensive steps: ___ are checkpointed (how/where: ___) and idempotent (how: ___).
- Resume point for the longest step: "if it dies, run ___; re-running is safe because ___."

## Replan triggers (results that would force a revision)
- If <step> shows <result>, then <replan how>.

## Review gate — ALL must be checked before Build
- [ ] Plan is written here (reviewers can see it).
- [ ] Every step is independently verifiable and has a concrete done-signal.
- [ ] Decomposition is vertical; a thin end-to-end path is an early step.
- [ ] Riskiest / most uncertain / most expensive steps are sequenced first.
- [ ] Unknowns flagged; spikes added where uncertainty × cost is high.
- [ ] Effort / cost / risk estimated per step; low-confidence + high-cost de-risked first.
- [ ] Long/expensive steps are checkpointed + idempotent with a defined resume point.
- [ ] Approach reviewed by someone other than the author; concerns resolved or recorded.
```

---

## Smaller example — when NOT to over-plan

**Task:** bump a serving image's base CUDA version and confirm latency unchanged.

This is small and low-uncertainty, so the "plan" is a 30-second mental sequence with done-signals — the
same discipline, compressed:

1. Build image on new base → `docker build` succeeds; image runs `nvidia-smi` showing the new version.
2. Smoke test → existing serving integration test passes against the new image.
3. Latency check → p50/p99 on the canary within ±5% of baseline over a fixed request set.

No spike (you've done CUDA bumps before), no checkpointing (no long-running state), but each step still
has a concrete done-signal and the riskiest check (latency regression) is explicit. Decompose to the
*level the risk warrants* — not more, not less.
