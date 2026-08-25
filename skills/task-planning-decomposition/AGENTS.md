# AGENTS.md — Task Planning & Decomposition (the Plan stage)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative process lives in **`task-planning-decomposition-guide.md`** next to this
> file — read it before planning any non-trivial task, and apply it. A fully worked decomposition and a
> reusable checklist template are in **`examples.md`**. This file is the always-on summary.
>
> **This is the Plan stage of `[[engineering-lifecycle]]`:** after the spec is reviewed
> (`[[spec-driven-development]]`), before you Build. The deliverable is a *written, reviewed plan* — an
> ordered list of small, verifiable steps — **not code**.

## When starting a non-trivial AI-infra / ML task, before writing code or launching compute:

- **Plan the approach (cheap) before doing the work (expensive).** Reviewing a plan costs minutes;
  re-doing a wrong approach costs days of work plus the wasted GPU-hours. De-risk before you commit
  compute.
- **Decompose into steps small enough to verify independently.** Every step gets a **concrete
  done-signal**: a command + expected output, a metric crossing a threshold, or an artifact. If you
  can't state how you'd verify a step, it's too big or too vague — split it.
- **Vertical slices over big-bang.** Get a thin end-to-end path running first (tiny model, few samples,
  1 GPU, 50 steps, 1 checkpoint), then thicken. Don't build every component in isolation and integrate
  last — integration risk is *the* risk.
- **Sequence riskiest / most uncertain / most expensive FIRST.** Order by dependency, then within what's
  unblocked pull the scary step forward. Never leave "does multi-node networking even work" as a late
  step. Prove the expensive thing at the smallest scale before launching the big run.
- **Spike the unknowns.** For anything you've never done, write a small, timeboxed, throwaway prototype
  whose only output is information (does it fit, does it hit bandwidth, does the API behave). Decide
  prototype-vs-build by uncertainty × downstream cost.
- **Plan for partial failure.** Long/expensive/distributed steps must be **checkpointed** and
  **idempotent**, with a defined resume point ("if it dies at 60%, what do I run, and is re-running
  safe?"). See `[[ml-checkpointing-orbax]]`, `[[distributed-systems-fundamentals]]`. Restore must be a
  *verified* step, not an assumption.
- **Estimate effort / cost / risk per step** — especially compute cost (`GPU-hours × $/GPU-hr`). Mark
  low-confidence + high-cost steps; those get de-risked first.
- **The plan is a living checklist.** Write it down where reviewers see it; check off done-signals;
  update it as you learn. A plan in your head can't be reviewed, checked off, or survive a preemption.
- **Replan when a result contradicts an assumption the plan rested on** — don't grind forward on a plan
  you know is wrong.
- Within a step, the done-signal is often a test — that's `[[test-driven-development]]` territory. Plan
  decides the steps; TDD executes one.

## The gate (before Build)
Do **not** start Building until **all** hold: a written plan of discrete, independently verifiable steps
exists; the riskiest/most-expensive work is sequenced first; unknowns are flagged with spikes;
long/expensive steps are checkpointed + idempotent with a resume point; and **the approach has been
reviewed by someone other than the author.**

## Common rationalizations to reject
"The task is obvious, just start" · "planning is overhead" · "I'll keep the plan in my head" · "I'll
figure out the order as I go" · "just launch the big run and see" · "we'll add checkpointing later."
Rebuttals in the guide.
