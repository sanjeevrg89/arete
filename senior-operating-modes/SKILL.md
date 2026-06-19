---
name: senior-operating-modes
description: How to FRAME and DRIVE a non-trivial task to a capable AI coding agent (Claude Code, Gemini
  CLI, Codex, and any agentic IDE) — set a goal, a standard-of-done, and a verification plan and let it
  run to completion, instead of issuing junior-intern imperatives ("do this", "write code", "fix this
  bug"). Use at the START of a task to pick the operating mode and the completion bar: the run-to-done
  goal wrapper (keep going until it meets the bar, not until it compiles; real-time end-to-end test after
  each step; review; commit; track progress), the parallel end-to-end goal for big jobs (decompose into
  independent pieces, dispatch concurrent sub-agents each with its own goal/deliverable/verification,
  synthesize, resolve conflicts), production-grade build (requirements -> edge cases -> architecture ->
  minimal-but-scalable MVP), inherit-an-unfamiliar-repo + refactor, senior debugging (root cause -> fix
  plan -> production code), and performance optimization. Covers defining "done" so the agent can
  self-check it, validating the real end-to-end path (CLI/browser/clicks/keystrokes), when to keep going
  vs stop, and pairing every mode with a verification + review gate. The "how to instruct" layer above
  engineering-lifecycle (which owns the stages/gates) — vendor-neutral, works in any agent.
---

# Senior Operating Modes (drive the agent like a senior, not a junior)

A capable coding agent is wasted when you use it like a junior intern — "do this", "write that function",
"fix this bug". You become the bottleneck, feeding it steps. Drive it like a **senior**: give it a
**goal**, a **standard-of-done**, and a **way to verify**, then let it run to completion and report. This
skill is the *how-to-instruct* layer; it routes the actual work into `[[engineering-lifecycle]]` and the
stage skills, which own the gates.

## How to use this skill

1. Read `senior-operating-modes-guide.md` in this directory — the modes, the run-to-done standard, the
   parallel/e2e pattern, and how each mode pairs with a verification + review gate. Apply it.
2. For copy-paste templates (the goal wrapper, the parallel dispatch, and the six modes), read
   `examples.md`. Adapt the bracketed parts; don't paste blindly.
3. **Pick the mode and write the goal + standard first.** Then let the agent run, validating the real
   end-to-end path after each meaningful step. Match the team's conventions; hold the bar regardless.

## The essentials (full detail in `senior-operating-modes-guide.md`)

- **Stop issuing imperatives; set goals.** "Do X" makes you the planner. "Achieve G to standard S,
  verify it, and keep going until it meets the bar" lets a capable model plan, build, and self-correct.
- **Define "done" so the agent can self-check it.** Done = every dimension production-grade and *a real
  user can walk in and use it* — not "it compiles" / "tests pass once". State the bar in the goal.
- **The run-to-done wrapper:** goal + "keep going until the architecture and result meet the bar, not
  just until it runs" + **real-time end-to-end test after each meaningful step** + auto-review + commit +
  **write progress somewhere sensible** in the project. Don't hardcode the progress path; let it choose.
- **Validate the REAL thing, not a proxy.** Exercise the full path — CLI, server, browser, clicks,
  keystrokes, whatever it needs — not just unit tests → `[[verification-and-debugging]]`.
- **For big jobs, go parallel + e2e:** decompose into **independent** pieces, spawn concurrent sub-agents
  each with **its own goal, deliverable, verification, and completion bar**, then synthesize and resolve
  conflicts → `[[task-planning-decomposition]]`, multi-agent patterns in `[[llm-app-agent-frameworks]]`.
- **Every mode pairs with a gate.** The mode is the *framing*; a verification pass + an independent
  review (`[[code-review-discipline]]`) are what make running-to-done safe rather than reckless.
- **Don't stop at partial progress** — unless blocked by **missing credentials, destructive ambiguity,
  or conflicting requirements**. Those three are the legitimate stop conditions; "it compiled" is not.
- **The modes:** run-to-done wrapper · parallel e2e · production-grade build (reqs→edge cases→arch→MVP) ·
  inherit + refactor · senior debugging (cause→fix plan→prod code) · performance. Each is a *standard*,
  not a magic word.
- **This composes with the lifecycle, it doesn't replace it.** The mode sets the framing and bar;
  `[[engineering-lifecycle]]` runs Define→Plan→Build→Verify→Review→Ship with the gates between.

## Related skills

- `[[engineering-lifecycle]]` — the stages and gates a run-to-done goal actually moves through; this
  skill picks the framing, that one enforces the process.
- `[[spec-driven-development]]` · `[[task-planning-decomposition]]` — turn a goal into a spec with a
  standard-of-done, then into independent, parallelizable pieces.
- `[[verification-and-debugging]]` · `[[test-driven-development]]` — the real-end-to-end validation and
  the test-first build the run-to-done loop depends on.
- `[[code-review-discipline]]` · `[[shipping-and-release]]` — the review pass and the safe commit/ship
  every mode must end on.
- `[[gpu-performance-engineering]]` · `[[inference-optimization]]` — where the performance mode routes
  for AI-infra work.
- `[[staff-plus-engineering]]` — scoping ambiguous, multi-team work into a goal worth running to done.
