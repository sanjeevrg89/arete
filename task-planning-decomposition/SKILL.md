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
