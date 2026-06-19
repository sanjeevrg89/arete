---
name: staff-operating-modes
description: How to FRAME and DRIVE a non-trivial task to a capable AI coding agent (Claude Code, Codex,
  Gemini CLI, and any agentic IDE or CLI) so it works at a STAFF / DISTINGUISHED engineer's bar — not the
  "senior" default of "it compiles and a happy-path test passes," and not the junior anti-pattern of
  feeding it imperatives ("do this", "write code", "fix this bug"). Use at the START of a task to choose
  the operating mode and set the completion bar: the run-to-done goal wrapper (keep going until the
  architecture and result meet the bar, not until it runs; real end-to-end validation after each step;
  review; commit; track progress), the parallel end-to-end goal for large jobs (decompose into
  independent pieces, dispatch concurrent sub-agents each with its own goal/deliverable/verification),
  production-grade build (requirements -> edge cases -> architecture -> minimal-but-scalable MVP),
  inherit-an-unfamiliar-repo + refactor, root-cause debugging, and performance optimization. The
  distinguished bar adds what "senior" skips: blast radius, simplicity, leverage, second-order effects,
  reversibility, observability, and whether the thing should be built at all. Covers defining "done" so
  the agent can self-check it, validating the real end-to-end path (CLI/browser/clicks/keystrokes),
  pairing every mode with a verification + review gate, and when to keep going vs stop. The "how to
  instruct" layer above engineering-lifecycle; vendor-neutral, works in any agent.
---

# Staff Operating Modes (drive the agent at a distinguished bar)

A capable coding agent is wasted two ways. Drive it like a **junior** — a stream of imperatives — and you
are the bottleneck. Drive it like a **senior** — "build the feature, make the test pass" — and you get
something that works and is quietly wrong at the level that matters: blast radius, simplicity, leverage,
what should have been built instead. Drive it at the **staff / distinguished** bar: give it a goal, the
*standard a principal engineer would hold in review*, and a way to verify — then let it run to that bar
and report. This is the *how-to-instruct* layer; it routes the work into `[[engineering-lifecycle]]` and
the stage skills, which own the gates, and applies the judgment in `[[staff-plus-engineering]]`.

## How to use this skill

1. Read `staff-operating-modes-guide.md` in this directory — the modes, the distinguished standard-of-
   done, the parallel/e2e pattern, and how each mode pairs with a verification + review gate. Apply it.
2. For copy-paste templates (the goal wrapper, the parallel dispatch, the six modes, and a senior-vs-
   distinguished before/after), read `examples.md`. Adapt the bracketed parts; don't paste blindly.
3. **Pick the mode and write the goal + the distinguished bar first.** Then let the agent run, validating
   the real end-to-end path after each meaningful step. Match the team's conventions; hold the bar.

## The essentials (full detail in `staff-operating-modes-guide.md`)

- **Three altitudes, not two.** Junior needs the steps. Senior does the task and stops at "it works."
  **Staff/distinguished** owns architecture, **blast radius**, simplicity, leverage, second-order
  effects, and *whether the thing should exist*. Instruct the agent at the third altitude.
- **Set goals, not imperatives.** "Do X" makes you the planner. "Achieve G to the bar below, verify it,
  keep going until it meets the bar" lets a capable model plan, build, and self-correct.
- **Define "done" at the distinguished bar, so the agent can self-check it.** Not "it compiles / a test
  passed." Done = correct, **simple**, minimal blast radius, observable, **reversible**, and the *right*
  thing to build — what a principal would approve in review. State it in the goal.
- **The run-to-done wrapper:** goal + "keep going until the architecture and result meet the bar, not
  just until it runs" + **real end-to-end validation after each meaningful step** + review + commit +
  **write progress somewhere sensible**. Don't hardcode the progress path; let the agent choose it.
- **Validate the REAL thing.** Exercise the full path — CLI, server, browser, clicks, keystrokes — not
  just unit tests → `[[verification-and-debugging]]`.
- **For large jobs, go parallel + e2e:** decompose into **independent** pieces; spawn concurrent sub-
  agents each with **its own goal, deliverable, verification, and bar**; synthesize and resolve conflicts
  → `[[task-planning-decomposition]]`, multi-agent patterns in `[[llm-app-agent-frameworks]]`.
- **Every mode pairs with a gate.** The mode is the *framing*; a verification pass + an independent,
  staff-level review (`[[code-review-discipline]]`) are what make running-to-done safe, not reckless.
- **Don't stop at partial progress** — unless blocked by **missing credentials/access, destructive
  ambiguity, or conflicting requirements.** Those three are the legitimate stops; "it compiled" is not.
- **The modes** (each a *standard*, not a magic word): run-to-done wrapper · parallel e2e · production-
  grade build (reqs→edge cases→architecture→MVP) · inherit + refactor · root-cause debugging · perf.
- **Composes with the lifecycle, doesn't replace it.** The mode sets the framing and bar;
  `[[engineering-lifecycle]]` runs Define→Plan→Build→Verify→Review→Ship with the gates between.

## Related skills

- `[[staff-plus-engineering]]` — the staff/principal/distinguished *judgment* (blast radius, leverage,
  simplicity, second-order effects) this skill instructs the agent to apply.
- `[[engineering-lifecycle]]` — the stages and gates a run-to-done goal actually moves through; this
  skill picks the framing, that one enforces the process.
- `[[spec-driven-development]]` · `[[task-planning-decomposition]]` — turn a goal into a spec with a
  distinguished standard-of-done, then into independent, parallelizable pieces.
- `[[verification-and-debugging]]` · `[[test-driven-development]]` — the real-end-to-end validation and
  test-first build the run-to-done loop depends on.
- `[[code-review-discipline]]` · `[[shipping-and-release]]` — the staff-level review and the safe
  commit/ship every mode must end on.
- `[[gpu-performance-engineering]]` · `[[inference-optimization]]` — where the performance mode routes
  for AI-infra work.
