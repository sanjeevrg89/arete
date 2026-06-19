# Using this library to get 10–100x (operating manual)

Installing the skills is 10% of the value. The leverage comes from *how* you operate once they're
installed — and it's the same playbook in **Claude Code, Gemini CLI, Codex,** and any agentic IDE.

> **The idea in one line:** encoded staff/distinguished judgment (the skills) that **ports across every agent** and
> **gets sharper from your usage**. That combination — not any single model — is the multiplier.

This is the honest version. There is no magic prompt. There is a system: install it everywhere, drive
every agent at a staff/distinguished bar, and run a loop that improves the skills from real feedback. Do those three and
the leverage compounds.

---

## 1. Install it in every agent you use

The library ships from **one source of truth per skill** into the format each agent reads, so the *same*
judgment is available whichever tool you're in.

| Agent | One-time install | How it loads |
|-------|------------------|--------------|
| **Claude Code** | `./install.sh claude` | symlinks every skill into `~/.claude/skills`; loads on demand by `description`. `/skills` lists them. |
| **Gemini CLI** | `./install.sh flat <gemini-skills-path>` — or copy a skill dir so its `GEMINI.md` `@./`-imports the guide | flat bundle `skills/*.md` is the universal fallback for any markdown-context loader |
| **Codex / Cursor / IDEs** | drop a skill's `AGENTS.md` + guide into the repo root, or reference it from your existing `AGENTS.md` | always-on project rules; or point at the flat `skills/<name>.md` |
| **Anything else** | point a markdown loader at the `skills/` directory | one self-contained file per skill |
| **Keep current** | `./update.sh` (+ a path for flat copies) | re-link (Claude) / re-copy (flat) |

Do this once for **each** agent you actually use. The payoff of a vendor-neutral library is that you
stop re-learning per tool — switch models or tools, keep the expertise.

---

## 2. Operate at a staff/distinguished bar in all of them (the 3 habits)

### Habit 1 — Frame tasks as goals, not imperatives → [`staff-operating-modes`](staff-operating-modes/)
Stop typing "do this / fix that." Open every non-trivial task with a **goal + standard-of-done +
verification**, and let the agent run to completion. This is the biggest per-task multiplier and it's
identical in every agent. Use the run-to-done wrapper:

> Goal: {task}. Keep going until the architecture and result meet the bar, not just until it runs. After
> every meaningful step, test the real thing end-to-end, self-review, then commit, and write progress
> somewhere sensible. Done = production-grade, a real user can use it.

### Habit 2 — Use the lifecycle on anything with blast radius → [`engineering-lifecycle`](engineering-lifecycle/)
For a real change (a cluster edit, a rollout, a pipeline), run Define→Plan→Build→Verify→Review→Ship with
the gate between each. "Production-grade" becomes *enforced*, not asserted.

### Habit 3 — Make it compound → [`skill-self-improvement`](skill-self-improvement/)
When a skill underperforms, don't just fix the output — append a line to
[`feedback/log.jsonl`](feedback/README.md) (what was wrong, the correct answer). The reviser loop turns
accumulated feedback into PRs that improve the skill, gated by CI + review. Your judgment, encoded once,
gets better while you sleep — across every agent at once.

---

## 3. Your first week

1. **Install into all three agents** (Claude, Gemini CLI, Codex) — prove "works anywhere" once.
2. **Take one real task** and drive it with the run-to-done wrapper instead of imperatives. Feel the
   difference between an intern and a staff engineer.
3. **The first mediocre answer** a skill gives you → add one `feedback/log.jsonl` line. That's the loop
   starting. Run `python scripts/skill_feedback.py` to see your improvement candidates.
4. **Pick your highest-frequency task** (PR review, triage, manifest scaffolding) and make it a habit
   with the right skill. Frequency × leverage = where the hours come back.

---

## What "100x" actually means (read this part)

It's not the model. It's three things multiplying:

- **Encoded judgment** — the skills are a staff/distinguished engineer's playbook, applied consistently instead of
  from memory.
- **Portability** — the same judgment in Claude, Gemini CLI, and Codex; you're never re-learning per
  tool. (This is "token capital" you own, independent of any one model.)
- **A self-improving loop** — it compounds from your real usage rather than staying static.

And the honest caveats, because a tool oversold is a tool distrusted:

- The leverage is real **only if you keep the verify gates strong** and actually run the loop. A library
  you install but operate like a junior is just files.
- These guides are dense starting points authored from public sources — **verify load-bearing claims
  against current docs** (every guide says so). The ecosystem moves fast.
- "Run to done" has three legitimate stop conditions — missing credentials/access, destructive
  ambiguity, conflicting requirements. It is not license to never stop.

---

## Where to look next

- [`REGISTRY.md`](REGISTRY.md) — every skill + an **"agent patterns & operating modes → which skill"**
  table (run-to-done, swarms, durable orchestration, STORM research all map here).
- [`staff-operating-modes`](staff-operating-modes/) · [`research-methods`](research-methods/) ·
  [`skill-self-improvement`](skill-self-improvement/) — the meta-skills that make the rest of the library
  multiply.
- [`tests/VALIDATION.md`](tests/VALIDATION.md) — how a skill earns trust (5 layers); green CI ≠ validated.
