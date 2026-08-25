---
name: skill-self-improvement
description: How to build a self-improvement loop that makes an agent's own Skills (or any file-based
  capability — prompts, runbooks, rubrics) get better over time from real-world feedback, instead of
  doing the same thing on run #50 as on run #1. Use when you want an inner loop (apply a Skill to real
  work and record every run) plus an outer loop (a scheduled/cloud agent that reads the run feedback and
  opens a PR diffing the Skill file), with a verify gate so garbage never gets saved. Covers the
  doer→signal→reviser shape, capturing the feedback signal (human edits, accepted/rejected, graders,
  failing tests, GitHub issue relabels), the scheduled reviser that edits the Skill as a diff, the
  adversarial verify gate before merge (never auto-merge a self-edit), distilling lessons into
  anti-patterns/constraints so they can't regress, and running the loop durably (cron + checkpointing +
  idempotency) on GitHub Actions, Warp/Oz, Inngest/Temporal, or Claude Code scheduled agents. The loop
  layer above the static skill library; for the agent design patterns underneath it see
  llm-app-agent-frameworks.
---

# Skill Self-Improvement Loops

Apply the judgment of an engineer who runs self-improving agents in production — where the failure mode
isn't a crash, it's a loop that quietly optimizes toward the wrong thing because the verifier was weak.
A Skill is a file. A self-improvement loop is the machinery that edits that file from real feedback,
behind a gate, on a schedule — so your judgment is encoded once and compounds while you sleep.

## How to use this skill

1. Read `skill-self-improvement-guide.md` in this directory — the full pattern: the three-ingredient
   shape, the two loops, capturing the signal, the reviser, the verify gate, distillation, and how to
   run it durably. Apply it to the task at hand.
2. For copy-able artifacts — a feedback record, the reviser procedure, a `feedback/`-driven loop wired
   to this repo's CI, a GitHub Actions cron, and a work loop (PR-review) — read `examples.md`.
3. Before building a loop, confirm you have a **real external signal**. No signal → it's not a loop,
   it's a model rewriting the same file forever. Stop and find the signal first.

## The essentials (full detail in `skill-self-improvement-guide.md`)

- **Every self-improvement loop is the same shape: a doer + a signal + a reviser — made durable, behind
  a verify gate.** Everything else is plumbing.
- **Inner loop (the doer):** apply the Skill to real work and **record every run** (input, output, and
  what happened next). A run you didn't record can't teach you anything.
- **The signal is the whole point.** It must be *external truth the doer can't fake*: a human editing
  the result, a PR comment resolved-vs-dismissed, a grader/eval score, a failing test, a relabeled
  issue. A loop with no external signal is theater.
- **Outer loop (the reviser):** a **scheduled** agent reads the accumulated signal and **edits the Skill
  file as a diff**, opened as a **PR**. It does not do the work; it improves the doer that does the work.
- **Never auto-merge a self-edit.** A human (or a strong, independent gate) approves. Self-modifying
  knowledge that merges itself is how a library silently rots.
- **The verify gate decides what the loop becomes.** A loop improves toward whatever the verifier
  rewards — usually verbosity, not correctness. Keep an **adversarial, independent** check (a second
  model told to *refute*, plus tests) → `[[verification-and-debugging]]`, `[[ml-evaluation-evals]]`.
- **Distill, don't just patch.** Fold each lesson into the Skill's *anti-patterns / constraints* and add
  a **regression check**, so the same mistake can't come back. The patch fixes one output; the distilled
  rule fixes the class.
- **Run it durably.** A `while True` in a terminal is not a loop — a deploy/OOM/restart loses it. Run on
  a cron with **checkpointed steps** and **idempotent** side effects (don't re-open the same PR) →
  durable execution in `[[llm-app-agent-frameworks]]` §7.
- **Bound cost and scope.** Only revise skills that have **negative signal** this cycle; cap runs and
  token spend. A daily agent re-reading all history burns money for nothing.
- **Observability is the trust layer.** When an agent wrote the diff, you must be able to see which run,
  which signal, and which gate produced it → `[[ml-observability-monitoring]]`.

## Related skills

- `[[llm-app-agent-frameworks]]` — the agent loop patterns and **durable/long-running orchestration**
  (§7) this loop runs on; the doer is usually an agent built with these.
- `[[verification-and-debugging]]` · `[[ml-evaluation-evals]]` — the verify gate: proving the new Skill
  is actually better (adversarial checks, graders, eval-in-CI), not just different.
- `[[engineering-lifecycle]]` — this loop is a *Verify→Review→Ship* cycle applied to a Skill file; the
  reviser's PR goes through the same gates.
- `[[code-review-discipline]]` — the reviser opens a PR; review it like any change (blast radius, is the
  edit actually correct).
- `[[spec-driven-development]]` — a sharp Skill spec is the seed the loop refines; vague specs can't be
  improved against.
- `[[shipping-and-release]]` · `[[ml-observability-monitoring]]` — ship the revised Skill safely and
  watch whether the change actually helped.
