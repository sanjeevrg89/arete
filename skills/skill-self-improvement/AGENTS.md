# AGENTS.md — Skill Self-Improvement Loops

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative pattern lives in **`skill-self-improvement-guide.md`** next to this file —
> read it before building or running a self-improvement loop. Copy-able artifacts (feedback record,
> reviser procedure, CI-wired loop, cron, a work loop) are in **`examples.md`**. This is the always-on
> summary.

## The shape (memorize this)

**A self-improvement loop = a doer + a signal + a reviser — made durable, behind a verify gate.**
If any of the five is missing, you don't have a loop:

1. **Doer (inner loop).** Apply the Skill to real work; **record every run** (input → output → outcome).
2. **Signal.** External truth the doer can't fake — a human edit, a resolved/dismissed review comment, a
   grader score, a failing test, a relabeled issue. **No signal = theater. Find it before you build.**
3. **Reviser (outer loop).** A **scheduled** agent reads the signal and **edits the Skill file as a
   diff**, opened as a **PR**. It improves the doer; it does not do the doer's work.
4. **Verify gate.** Validate + functional/eval checks + an **adversarial, independent** review (a second
   model told to *refute*). The loop improves toward whatever this rewards — keep it strong.
5. **Durable.** Cron + checkpointed steps + idempotent side effects (don't re-open the same PR). A
   terminal `while True` is not durable.

## Hard rules (do not violate)

- **Never auto-merge a self-edit.** A human or a strong independent gate approves the reviser's PR.
- **No loop without a real external signal.** Stop and find it; a model rewriting its own file with no
  outside truth drifts, it does not improve.
- **The reviser edits the Skill, never the run record.** It must not rewrite the feedback/history it
  learns from.
- **Distill, don't just patch.** Each accepted lesson becomes a line in the Skill's *anti-patterns /
  constraints* **plus a regression check**, so the class of mistake can't recur — not a one-off edit.
- **Bound scope and cost.** Only revise skills with negative signal this cycle; cap runs/tokens.
- **Make side effects idempotent.** Derive a stable key (skill + cycle) so a retry/replay doesn't
  double-PR, double-comment, or double-label.

## Definition of done

A loop is complete when: the doer records every run · a real signal is captured in a durable place · a
scheduled reviser turns negative signal into a Skill **diff PR** · that PR must pass the verify gate
(checks + adversarial review) and is **human-merged** · accepted lessons are distilled into the Skill +
a regression check · the runner is durable (survives restart) and idempotent. If any is missing, say
which — the loop is not done.
