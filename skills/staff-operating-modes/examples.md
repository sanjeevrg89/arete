# Staff Operating Modes — copy-paste templates

Adapt the bracketed parts. The point isn't the wording — it's that each carries a **goal + a checkable
standard-of-done + a verification step**. Works in any agent (Claude Code, Gemini CLI, Codex, IDEs).

---

## 0. The run-to-done wrapper (wrap any task in this)

```
Goal: {your task / the full spec you already agreed on}.

Keep going until the architecture and result meet the bar, not just until it runs.
After every meaningful step: test the real thing end-to-end (full path — CLI, server, browser,
clicks, keystrokes, whatever it needs), self-review, then commit, and write progress somewhere
sensible in the project (you pick the spot — don't expect me to hardcode it).
Finish with one dedicated review pass over everything.

Done = production-grade at a staff bar: correct, simple, minimal blast radius, observable, reversible,
and the right thing to build (not just "it runs" or "a real user can use it").
Only stop early if blocked by missing credentials/access, destructive ambiguity, or conflicting
requirements — otherwise keep going.
```

## 1. Parallel + end-to-end (big jobs)

```
For this task, write yourself a new end-to-end goal: complete the whole plan, not just the next step,
until architecture, implementation, tests, review, and final result meet the standard.

Split that goal into INDEPENDENT pieces. Spawn as many parallel agents as needed to do it better and
faster, and give each agent its own dedicated goal that includes its expected deliverable, its
verification, and its completion standard.

Dispatch them concurrently. Keep tracking progress in the right place, synthesize results as they
return, resolve conflicts, continue implementation, and run real-time validation after important steps
(including browser/computer use, clicks, keyboard actions as needed). Finish with review, commit, and a
final summary. Do not stop after partial progress unless blocked by missing credentials, destructive
ambiguity, or conflicting requirements.
```

## 2. Production-grade build

```
Act as a staff engineer shipping something production-ready, whether it's one feature or a full app.
Don't jump to code. First analyze the requirements, list the edge cases, design the architecture, lay
out a plan. Build the minimal version that's still scalable and maintainable.

Then deliver: architecture overview, folder structure, data flow, database schema and API design where
relevant, full implementation, edge-case and error handling, performance notes. Design it like a real
startup MVP, not a demo. Verify the real end-to-end path before calling it done.
```

## 3. Inherit an unfamiliar repo + refactor

```
Act as a staff engineer who just inherited a large, unfamiliar codebase. First understand the
architecture and data flow — map it before you change anything.

Then find: structural problems, duplicated code, performance bottlenecks, maintainability risks.

Deliver: architecture overview, problem areas, refactor strategy, improved architecture and code. Keep
behavior intact (characterize it with tests first); verify end-to-end after each refactor step.
```

## 4. Root-cause debugging

```
Act as a staff engineer chasing a bug in production. Reproduce it first. Read the code carefully,
reason step by step, find the ROOT CAUSE (not the symptom), and give a robust fix that accounts for
edge cases and performance.

Cover three things: the cause, the fix plan, and production-ready code. Add a regression test that
fails on the old behavior and passes on the fix.
```

## 5. Performance optimization

```
Optimize this code as a performance engineer. Targets: speed, memory, scalability.
Measure first — profile and name the actual bottleneck before changing anything.
Find: bottlenecks, inefficient logic, unnecessary work/re-renders.
Return the explanation, the optimized code, and a before/after measurement that proves the win.
```

---

## Before / after — the whole point in one diff

**Junior (imperative, you're the bottleneck):**
```
Add caching to the user service.
```
→ You now own every follow-up: which cache, invalidation, what to measure, did it help.

**Staff / distinguished (goal + bar + verification):**
```
Goal: cut p99 latency of GET /users to under 100ms under current load.
First profile where the time actually goes (don't guess), and say whether caching is even the right fix
or we're masking a deeper problem. Propose the smallest change, with its invalidation, failure modes, and
blast radius on other services. Implement it behind a flag (reversible), add a load test that proves the
p99, and verify invalidation end-to-end. Keep going until the p99 holds and invalidation is correct;
commit at checkpoints; flag if it needs infra I don't have.
```
→ The agent profiles, questions the premise, picks the simplest reversible change, proves the number, and
self-corrects to the bar — not just "added a cache, looks faster."

---

## Note on portability

These are vendor-neutral. The *mechanisms* differ per tool — how you spawn parallel sub-agents, run
background work, drive a browser, or persist progress — so verify your agent's current capabilities and
keep the goal / standard / verification framing constant. The framing is the part that ports; the buttons
are the part that changes.
