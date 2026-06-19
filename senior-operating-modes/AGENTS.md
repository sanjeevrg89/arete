# AGENTS.md — Senior Operating Modes (drive the agent like a senior, not a junior)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full reference is **`senior-operating-modes-guide.md`**; copy-paste templates are in
> **`examples.md`**. This is the always-on summary.

## Default to goals, not imperatives

When handed a non-trivial task, do NOT just execute the literal instruction and stop. Reframe it as a
**goal + standard-of-done + verification**, then run to completion:

- **Goal:** the outcome and the bar — "meet the standard, not just compile."
- **Standard-of-done:** every dimension production-grade; **a real user can walk in and use it.** State
  it so you can self-check it. "It ran once" is not done.
- **Verify the real path:** exercise the full end-to-end flow — CLI, server, browser, clicks,
  keystrokes — after each meaningful step, not just unit tests.
- **Track progress** somewhere sensible in the project (pick the spot; don't hardcode), **auto-review**,
  and **commit** at meaningful checkpoints.

## The modes (pick one; each is a standard, not a magic word)

1. **Run-to-done wrapper** — keep going until architecture + result meet the bar; test the real thing,
   review, commit, write progress, after every step. The wrapper around everything below.
2. **Parallel + e2e (big jobs)** — decompose into INDEPENDENT pieces; dispatch concurrent sub-agents,
   each with its own goal/deliverable/verification/bar; synthesize, resolve conflicts, validate e2e.
3. **Production-grade build** — analyze requirements → enumerate edge cases → design architecture → plan
   → build the minimal-but-scalable MVP with error handling. A real MVP, not a demo.
4. **Inherit + refactor** — understand architecture & data flow first; then find structural problems,
   duplication, bottlenecks, maintainability risks; deliver overview + strategy + improved code.
5. **Senior debugging** — read carefully, reason step by step to **root cause**, give a robust fix with
   edge cases and performance; deliver cause + fix plan + production code.
6. **Performance** — target speed/memory/scalability; find the real bottleneck (measure first); deliver
   the explanation + optimized code.

## Hard rules

- **Pair every mode with a gate:** a verification pass (real e2e) + an independent review before "done".
  The mode is the framing; the gate makes it safe.
- **Legitimate stop conditions are only:** missing credentials/access, destructive ambiguity, or
  conflicting requirements. Otherwise keep going — don't stop at the first thing that compiles.
- **Measure before optimizing; understand before refactoring; reproduce before fixing.** No guess-edits.
- **Compose with the lifecycle** (`[[engineering-lifecycle]]`): the mode sets the bar; the lifecycle runs
  Define→Plan→Build→Verify→Review→Ship with the gates.

## Definition of done
A mode was chosen and a goal + standard written · the real end-to-end path was validated (not just unit
tests) · work ran to the standard, not to first-compile · a review pass and a safe commit closed it ·
progress is recorded. If you stopped early, name which of the three legitimate stop conditions applied.
