# Senior Operating Modes — Full Reference

The most common way to waste a capable coding agent is to drive it like a junior intern: a stream of
imperatives — "do this", "write that function", "now fix this bug". Every step routes back through you.
You are the planner, the integrator, and the bottleneck, and the agent never operates above the altitude
of your last instruction.

The shift is simple and it changes everything: **give the agent a goal, a standard-of-done, and a way to
verify — then let it run to completion and report.** A senior engineer isn't told which lines to write;
they're given an outcome and a bar and trusted to plan, build, test, and self-correct. Instruct the agent
the same way.

This skill is the **how-to-instruct layer**. It does not own the engineering process — `[[engineering-
lifecycle]]` does (Define→Plan→Build→Verify→Review→Ship and the gate between each). This skill decides the
**framing and the bar**; the lifecycle and stage skills do the work. The two compose.

---

## When to use this skill

- At the **start of any non-trivial task** you'd otherwise hand over as a one-line imperative.
- When you catch yourself **feeding the agent steps** instead of an outcome.
- When a job is **big enough to parallelize** and you need to decompose and dispatch.
- When the agent keeps **stopping at "it compiles"** and you want it to run to a real standard.

Not every task needs this — a true one-liner ("rename this var") is fine as an imperative. The modes earn
their overhead on anything with architecture, edge cases, or a quality bar.

---

## The core idea: define "done" so the agent can self-check it

The single highest-leverage move is to **state the standard-of-done in the goal**, in terms the agent can
verify without you:

- **Weak:** "build a feature that does X." → The agent stops at the first version that compiles.
- **Strong:** "build X to production standard: handles the edge cases below, has tests that exercise the
  real end-to-end path, passes review, and a real user can walk in and use it. Keep going until it meets
  that bar, not until it runs."

"Done" is **every dimension at the bar and a real user can use it** — not "it ran once." If the agent
can't tell whether it's done, neither can you; make the bar checkable (tests pass e2e, review clean,
the real flow works).

---

## Mode 1 — The run-to-done wrapper (the meta-mode)

This wraps every other mode. The template (full version in `examples.md`):

> **Goal:** {the task / spec}. Keep going until the architecture and result meet the bar, not just until
> it runs. After every meaningful step: test the real thing end-to-end (CLI, browser, keystrokes —
> whatever it needs), self-review, then commit, and write progress somewhere sensible in the project.
> Finish with one dedicated review pass over everything. Done = every dimension production-grade, a real
> user can use it.

What each clause buys you:

- **"meet the bar, not just run"** — defeats the stop-at-first-compile failure mode.
- **"test the real thing end-to-end after every step"** — catches integration breakage early, when it's
  cheap, instead of at the end. Real path, not a proxy → `[[verification-and-debugging]]`.
- **"self-review then commit"** — small, reviewed, reversible checkpoints instead of one giant diff →
  `[[code-review-discipline]]`, `[[shipping-and-release]]`.
- **"write progress somewhere sensible (don't hardcode the path)"** — the agent records state where the
  project expects it (a task file, a PR description, a scratch doc), so a restart or a handoff resumes
  cleanly. Letting it choose the spot is deliberate: it finds the right place per project.

---

## Mode 2 — Parallel + end-to-end (for big jobs)

When the work is large and decomposable, don't run it serially. The template:

> For this task, write yourself an end-to-end goal: complete the whole plan, not just the next step,
> until architecture, implementation, tests, review, and final result meet the standard. Split it into
> **independent** pieces; spawn as many parallel agents as needed; give each its **own goal that includes
> its expected deliverable, its verification, and its completion bar.** Dispatch concurrently, track
> progress in the right place, synthesize results as they return, resolve conflicts, validate the real
> end-to-end path, and finish with review + commit + a summary.

Discipline that makes it work (not just faster, but better):

- **Parallelize only independent pieces.** Dependent work in parallel produces merge conflicts and
  rework. Decompose so pieces don't share mutable state → `[[task-planning-decomposition]]`. Isolate
  agents that edit files concurrently (separate worktrees/branches).
- **Each sub-agent gets a *complete* goal** — deliverable, verification, and standard — not a vague
  fragment. A sub-agent with no completion bar returns half-work.
- **Synthesize and resolve conflicts as results return**; don't just concatenate. The integration step is
  where quality is won or lost. Multi-agent orchestration patterns → `[[llm-app-agent-frameworks]]` §2.
- **Don't reach for parallel until a single agent demonstrably can't keep up** — it costs more tokens and
  more coordination. Big, genuinely-separable jobs only.

---

## Modes 3–6 — Task-shaped framings

Each is a **standard**, not a magic word. State the deliverables in the goal so the agent self-checks.

### Mode 3 — Production-grade build
> Act as a senior engineer shipping something production-ready. Don't jump to code. First analyze
> requirements, enumerate edge cases, design the architecture, lay out a plan. Build the
> minimal-but-scalable version. Deliver: architecture overview, folder structure, data flow, schema/API
> design where relevant, full implementation, edge-case + error handling, performance notes. A real
> startup MVP, not a demo.

Routes through `[[spec-driven-development]]` (requirements + acceptance criteria) → `[[engineering-
lifecycle]]`. The "don't jump to code" clause is load-bearing: architecture before implementation.

### Mode 4 — Inherit an unfamiliar repo + refactor
> Act as a senior engineer who inherited a large, unfamiliar codebase. First understand the architecture
> and data flow. Then find structural problems, duplication, performance bottlenecks, maintainability
> risks. Deliver: architecture overview, problem areas, refactor strategy, improved architecture + code.

**Understand before you change.** A refactor that starts editing before mapping data flow breaks
invariants it never saw. Pairs with `[[verification-and-debugging]]` (build the mental model) and
`[[code-review-discipline]]` (name the real risks).

### Mode 5 — Senior debugging
> Act as a senior engineer chasing a production bug. Read the code carefully, reason step by step, find
> the **root cause** (not the symptom), give a robust fix accounting for edge cases and performance.
> Deliver three things: the cause, the fix plan, and production-ready code.

**Reproduce before you fix.** A fix for a bug you can't reproduce is a guess → `[[verification-and-
debugging]]` owns the root-cause discipline. Don't patch the symptom and call it done.

### Mode 6 — Performance optimization
> Optimize this as a performance engineer. Targets: speed, memory, scalability. Find the real
> bottlenecks and inefficiencies. Return the explanation + the optimized code.

**Measure first.** Optimizing without a profile optimizes the wrong thing. For AI-infra work this routes
to `[[gpu-performance-engineering]]` (roofline, Nsight, straggler analysis) and `[[inference-
optimization]]` (quantization, batching, decode). Confirm the win with a before/after measurement.

---

## Composing with the lifecycle (which mode routes where)

| Mode | Primary routing |
|---|---|
| Run-to-done wrapper | `[[engineering-lifecycle]]` (runs all six stages + gates) |
| Parallel + e2e | `[[task-planning-decomposition]]` → `[[llm-app-agent-frameworks]]` (multi-agent) |
| Production-grade build | `[[spec-driven-development]]` → `[[test-driven-development]]` |
| Inherit + refactor | `[[verification-and-debugging]]` → `[[code-review-discipline]]` |
| Senior debugging | `[[verification-and-debugging]]` |
| Performance | `[[gpu-performance-engineering]]` / `[[inference-optimization]]` |

The pattern: **this skill frames the task and sets the bar; the lifecycle + stage skills enforce the
gates.** Don't let a mode skip a gate ("it's production-grade because I said so" is not a verification).

---

## Rationalizations & rebuttals

- *"It's faster if I just tell it each step."* → Faster to start, slower to finish — you become the
  bottleneck and the agent never operates above your last instruction. Set a goal and a bar.
- *"It compiled / the test passed, so it's done."* → Done is the *standard*, not the first green. A
  feature that compiles but mishandles the edge cases, or that no real user can actually use, is not done.
- *"Parallel agents will obviously be faster."* → Only for independent work. Parallelizing dependent
  pieces buys you merge conflicts and rework. Decompose first; parallelize the truly separable.
- *"A 'senior engineer' persona is just prompt theater."* → The persona alone is theater; the persona
  **plus a checkable standard-of-done plus a verification gate** is what changes the output. Keep the bar.
- *"Skip the architecture step, go straight to code — it's faster."* → For anything non-trivial, skipping
  design is how you build the wrong thing well. Analyze → edge cases → architecture → then build.
- *"Keep going until 100% means never stop."* → No — stop on the three legitimate conditions (missing
  creds/access, destructive ambiguity, conflicting requirements). Otherwise, don't stop at partial.

---

## Red flags — stop and reconsider

- You're typing a sequence of imperatives instead of one goal with a bar.
- The goal has **no standard-of-done** the agent can self-check.
- "Verified" means a unit test passed, not that the **real end-to-end path** works.
- A mode reached "done" with **no independent review pass**.
- Parallel sub-agents were given **fragments without their own completion bar**, or were set on
  **dependent** work and are now conflicting.
- A refactor started editing **before** the architecture/data flow was understood; a fix landed **before**
  the bug was reproduced; an optimization shipped **before** a measurement.
- The agent stopped at the first compile and called it done (and wasn't actually blocked).

---

## Verification gate (definition of done)

- [ ] A **mode was chosen** and a **goal + standard-of-done** written before work started.
- [ ] The standard is **checkable** (tests pass e2e, review clean, a real user can use it) — not "it
      compiles."
- [ ] The **real end-to-end path** was validated (CLI/browser/clicks/keystrokes as needed), not just unit
      tests.
- [ ] Work ran **to the standard**, with progress tracked and meaningful checkpoints committed.
- [ ] An **independent review pass** closed it; the relevant lifecycle gates were honored, not asserted.
- [ ] If it stopped early, it named which of the **three legitimate stop conditions** applied.

If any box is unchecked, the work isn't done to a senior bar — report which.

---

## Version awareness & references

The *modes* are durable; the **mechanisms** are tool- and version-specific — how you spawn parallel
sub-agents, run background tasks, drive a browser, or persist progress differs across Claude Code, Gemini
CLI, Codex, and agentic IDEs, and changes fast. Verify the current capability of your tool before relying
on a specific mechanism; keep the goal/standard/verification framing constant across all of them.

- `[[engineering-lifecycle]]` — the six-stage process and gates these modes run through.
- `[[spec-driven-development]]` · `[[task-planning-decomposition]]` · `[[verification-and-debugging]]` ·
  `[[code-review-discipline]]` · `[[shipping-and-release]]` — the stage skills each mode routes into.
- `[[staff-plus-engineering]]` — framing ambiguous, multi-team work into a goal worth running to done.
- `[[skill-self-improvement]]` — when a mode is one you repeat, turn it into a loop that improves itself.
