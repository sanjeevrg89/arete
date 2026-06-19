# Skill Self-Improvement Loops — Full Reference

A **self-improvement loop** makes a file-based capability — an agent **Skill**, a prompt, a runbook, a
review rubric — get better over time from real-world feedback, *without a human hand-editing it every
time*. The model's weights don't change; the **system around the model** gets smarter because the files
it reads keep improving. This is the loop layer that sits **above** a static skill library: the library
is the asset, the loop is what compounds it.

This guide synthesizes the same pattern as it appears across the ecosystem so you can build it on
whatever substrate you have:

- **Warp / Oz (`oz-for-oss`)** — an *inner loop* (a Skill applied on every new GitHub issue) and an
  *outer loop* (a daily cloud agent that reads how humans corrected the triage and opens a PR diffing
  the Skill file). The cleanest minimal statement of the pattern.
- **Inngest "agent loop architecture"** — the *durability* layer: the loop is a **cron + a decision**,
  each step is **checkpointed**, and recovery means resuming from the last successful step (not
  re-running side effects). A loop that can't survive a restart isn't a loop.
- **Kimi swarm + Opus verify gate** — the *quality* layer: **verify before you save**. A second,
  more-trustworthy model gates the output, the workflow is **distilled into a reusable Skill**, and
  recurring lessons are baked into a **constraints file** the loop reads every run.

They are one pattern. This guide is how to build it correctly.

---

## When to use this skill

- You have a Skill/prompt/rubric you run **repeatedly** (issue triage, code review, bug-fixing, incident
  response, manifest scaffolding, doc generation) and it makes the **same mistakes** run after run.
- You keep **hand-editing the same file** after each run to fix the same class of error — that manual
  edit *is* the outer loop; automate it.
- You want a capability that **compounds**: sharper on run #50 than run #1, with the improvement
  surviving process restarts and model swaps.
- You maintain a **library of Skills** (like this repo) and want it to improve from usage, not just from
  authoring sessions.

**Do not** build a loop when: the task runs once; you have **no external signal** (see Step 0); or a
single prompt fix ends it. A loop is infrastructure — it has a cost. Earn it.

---

## The shape: doer + signal + reviser, made durable, behind a verify gate

Every self-improvement loop reduces to five parts. Name them before you build; if one is missing you
don't have a loop.

| Part | What it is | Warp analog | This-repo analog |
|---|---|---|---|
| **Doer** (inner loop) | the Skill applied to real work, **with every run recorded** | triage agent on new issue | a skill used in a real session / `functional_test.py` run |
| **Signal** | external truth the doer can't fake | human relabels the issue + comments why | a `feedback/` record, a failing functional check |
| **Reviser** (outer loop) | a **scheduled** agent that reads the signal and **edits the Skill file as a diff** | daily agent → PR to `SKILL.md` | the `skill-self-improvement` reviser → PR to `*-guide.md` |
| **Verify gate** | proves the new Skill is *better*, not just different | (add this — see Step 4) | `validate.py` + functional checks + adversarial review |
| **Durable runner** | cron + checkpointed steps + idempotency | Oz cloud agent on a schedule | GitHub Actions cron / a scheduled agent |

The two loops run at different clocks: the **inner loop** runs per task (fast, often); the **outer
loop** runs on a **schedule** (slow — daily/weekly), observing many inner-loop runs at once. The outer
loop never does the inner loop's work; it improves the *doer that does the work*.

---

## The process

### Step 0 — Confirm a real signal exists (the gate that comes first)

**This is the make-or-break step.** A loop with no external signal is a model rewriting its own file
forever — it drifts toward fluency, not correctness. Before anything else, answer: *what outside-the-
agent truth tells me a run was good or bad?* Valid signals:

- **Human correction** — someone edits the output, relabels, rejects/accepts, or comments why (Warp's
  signal).
- **Automated grader / eval** — a scorer or LLM-judge with a rubric → `[[ml-evaluation-evals]]`.
- **Test / check outcome** — the generated code fails CI; the manifest fails `--dry-run`; a `must_*`
  functional check fails.
- **Downstream outcome** — the triaged issue got reopened; the shipped fix got reverted; the estimate
  missed the real number.

**Gate:** you can name the signal, where it's recorded, and how the reviser will read it. If you can't,
**stop** — build the signal first. No signal, no loop.

### Step 1 — Build the inner loop (the doer) and record every run

Apply the Skill to real work and **persist each run**: the input, the output, and a handle to wherever
the outcome will show up. A run you didn't record teaches nothing.

- **Where runs come from:** a real integration (a GitHub Action that runs the Skill on every new issue;
  a pre-commit/CI step; a scheduled job) or interactive use you log.
- **Where runs are recorded:** an agent trace, a file, a Slack thread, or the task tracker itself.
  Prefer a place the **signal naturally lands in the same record** — e.g. a GitHub issue, where the
  agent's label and the human's correction live on one object.
- Keep records **append-only**. The reviser reads them; it must never rewrite the history it learns
  from.

### Step 2 — Capture the signal

Wire the external truth from Step 0 onto the run record so the reviser can find it later.

- **GitHub-native (Warp-style):** the inner loop labels/comments on an issue or PR; the human's relabel
  + comment is the signal, already attached. The reviser queries closed/triaged items since last run.
- **File-based (this repo):** append a structured record to `feedback/log.jsonl` — `{skill, verdict,
  task, what_was_wrong, correct_answer, source, date}`. Lintable and diff-friendly; see
  `examples.md` and `feedback/README.md`.
- **Test-based:** the failing functional check *is* the signal — its `skill`, prompt, and which
  `must_contain` it missed tell the reviser exactly what to fix.

Capture the **reason**, not just the verdict. "Wrong: should have been *needs-info* because the feature
might need a setting" is a teachable signal; a bare 👎 is not.

### Step 3 — Build the outer loop (the reviser): signal → diff → PR

A **scheduled** agent (cron) that, each run:

1. **Pulls the signal** since its last run (relabels/comments, new `feedback/` records, failing checks).
2. **Groups by Skill** and ranks by how much negative signal each accumulated — only touch skills that
   actually underperformed this cycle.
3. For each, **reads the current Skill file** and the signal, and **produces a diff**: tighten the
   `SKILL.md` router `description` (routing miss), fix/extend the guide (wrong or shallow content), or
   add a rule to the anti-patterns (recurring mistake).
4. **Opens a PR** with the diff, citing the runs that motivated it. Because Skills are just files, "edit
   the Skill" = "make a commit." One PR per Skill per cycle (idempotency — Step 6).

The reviser is itself driven by a Skill (this one) run through a coding agent. It **does not** re-triage
issues or re-do the work — it improves the doer.

### Step 4 — The verify gate (never auto-merge a self-edit)

The reviser's PR must **prove the Skill got better**, not just different. This is where a weak loop goes
wrong: a loop improves toward **whatever the verifier rewards**. A lazy verifier rewards longer, more
confident prose → the Skill bloats and grows false certainty. Make the gate adversarial:

- **Mechanical checks** (cheap, in CI): `validate.py` (structure/links), functional `must_*` checks,
  any eval suite. A regression check added for the fixed bug **must** now pass.
- **Adversarial review** (the Opus-verify-gate idea): a **second, independent** model whose only job is
  to **refute** — "does this diff actually fix the cited failure without breaking other guidance? Is the
  new claim correct? Default to *reject* if unsure." Diversity of model/lens matters more than count.
- **Human merge.** A person (or, for low-stakes loops, the passing adversarial gate) approves. **Self-
  modifying knowledge that merges itself rots silently** — keep a gate that can say no.

**Gate:** checks green **and** the adversarial reviewer affirms the diff is *correct and net-better*.
Otherwise the PR stays open for a human, or the reviser retries with the refutation as input.

### Step 5 — Distill: fold the lesson in so it can't regress

A merged patch fixes **one** output. Distillation fixes the **class**:

- Add the lesson to the Skill's **Anti-patterns / Red flags / constraints** section in plain, general
  terms ("a feature request that may need a config setting is *needs-info*, not *ready*").
- Add a **regression check** (a `must_*` functional check, an eval case) so the same mistake fails CI
  forever after. This is the ratchet — without it, the next model revision can undo the fix.
- This is the Kimi **`constraints.md`** idea: a file the loop reads at the **start** of every run, so
  last cycle's failure is this cycle's hard rule. In this repo, each guide's anti-patterns section *is*
  that constraints file; the reviser appends to it.

### Step 6 — Run it durably (or it isn't a loop)

A `while True` in a terminal — or a long-running process on a VM — loses everything on a deploy, OOM, or
restart: it re-reads data, re-calls the model for decisions it already made, and **re-fires side
effects** (a duplicate PR, a duplicate Slack ping). Make the runner durable:

- **Cron, not a daemon.** The outer loop is *a schedule + a decision* — GitHub Actions `schedule:`, a
  Claude Code scheduled agent, an Inngest cron function, a Temporal schedule. The cron is the heartbeat;
  the agent is the decision in the middle.
- **Checkpoint each step.** Pull-signal, propose-diff, open-PR are distinct steps; on restart, resume
  from the last completed one. This also **saves tokens** — a crash mid-run shouldn't re-call the model
  for steps it already finished. (Durable-execution engines do this for you →
  `[[llm-app-agent-frameworks]]` §7.)
- **Make side effects idempotent.** Derive a **stable key** from `(skill, cycle)` — *not* a fresh UUID —
  so a retry/replay updates the existing PR instead of opening a second one. At-least-once delivery +
  non-idempotent side effect = duplicates → `[[distributed-systems-fundamentals]]`.
- **Handle terminal failure.** If the run exhausts retries (API key expired, provider down), post to an
  ops channel and let the **next scheduled run** pick up — nothing is lost because the signal is durably
  recorded.

### Step 7 — Observe and bound cost

- **The run history is the trace.** When an agent wrote the diff, you must be able to answer: which runs
  triggered it, which signal, which gate passed it? Export per-step history → `[[ml-observability-monitoring]]`.
- **Bound scope and spend.** Only revise skills with negative signal this cycle. Cap runs, token budget,
  and PRs-per-cycle. A daily agent re-reading all history and touching everything burns money for noise.
- **Watch second-order effects.** Track whether merged improvements actually moved the inner loop's
  success rate. If the Skill keeps growing but quality doesn't, your gate or signal is weak — fix that,
  not the Skill.

---

## The loop wired into *this* repo

This library already ships the assets of the loop — it just wasn't automated. The wiring:

```
inner loop:   a skill is applied (a real session, or `functional_test.py` against checks.json)
                   │  records outcome
signal:       feedback/log.jsonl   ←  a 👎 record with the reason   (+ failing functional checks)
                   │  read by
reviser:      skill-self-improvement  →  proposes a diff to <skill>/<skill>-guide.md
                   │                       and a new must_* check in tests/functional/checks.json
verify gate:  validate.py + functional_test.py --lint + adversarial review  →  PR
                   │  human merges
distill:      lesson lands in the guide's Anti-patterns + a regression check; bundle rebuilt
```

- **Signal in:** `feedback/log.jsonl` (format in `feedback/README.md`); `scripts/skill_feedback.py
  --lint` validates records in CI and the default mode ranks **improvement candidates** by negative
  signal.
- **Runner:** `.github/workflows/skill-self-improvement.yml` runs the dry-run (lint + candidate report)
  on a cron with no secrets; the real reviser step (invoke a coding agent → open a PR) is documented
  there, gated, and **never auto-merges**.
- **Gate:** the existing `ci.yml` (`validate.py`, `functional_test.py --lint`) plus the
  `tests/VALIDATION.md` 5-layer procedure — Layer 5 step 4 ("feed failures back") is exactly this loop,
  now automatable.

---

## Running self-improvement loops at work (where the leverage is)

Pick work where you already have judgment worth encoding, then wrap it in inner+outer loops:

| Loop | Doer (inner) | Signal | Reviser updates |
|---|---|---|---|
| **PR review** | a review Skill on every PR → `[[code-review-discipline]]` | which comments got resolved vs. dismissed | the review checklist/rubric |
| **Issue / PR triage** | a triage Skill on new issues (Warp's example) | humans relabeling + reason | the triage rubric |
| **Incident runbooks** | an incident Skill on alert | the postmortem's corrections | the runbook's steps + red flags |
| **Manifest scaffolding** | a scaffolder doer | what reviewers changed in the generated YAML | the scaffolder's defaults |
| **Estimates** | a sizing doer (e.g. memory) | real measured number vs. estimate | the formula / fudge factors |

The move that compounds: **stop re-fixing the same nit by hand. Fix the Skill once, add a regression
check, and let a durable scheduled agent catch the rest.** Three or four of these running quietly *is*
the leverage — your encoded judgment working while you don't.

---

## Anti-patterns / gotchas

- **No external signal ("self-rewarding without ground truth").** The loop optimizes toward the
  verifier's taste, not reality — usually longer, more confident, wronger. The single most common failed
  loop.
- **Auto-merging self-edits.** No human/gate that can say no → the Skill drifts and you find out in
  production. Always a PR, always a gate.
- **Patching the output, not the Skill.** Fixing each run by hand forever; the file never improves. The
  manual fix *is* the signal — move it into the file.
- **No regression ratchet.** A fix with no added check gets silently undone by the next revision.
- **Reviser rewrites its own history.** If the agent can edit the feedback/run log it learns from, the
  signal is corrupted. Records are append-only; the reviser edits the Skill, not the evidence.
- **In-memory / `while True` runner.** Dies on restart, loses progress, re-fires side effects
  (duplicate PRs/pings). Use a cron + checkpoints + idempotency.
- **Non-idempotent side effects behind retries.** Fresh UUID per attempt → a second PR/label/comment on
  every replay. Key side effects by `(skill, cycle)`.
- **Unbounded scope/cost.** Re-reading all history and touching every Skill each run; no cap on PRs or
  tokens. Revise only what has negative signal.
- **Loop with no observability.** An agent wrote the change and you can't reconstruct why. Untrustable —
  export the step history.

---

## Rationalizations & rebuttals

- *"It's self-improving, so I don't need a verifier."* → Backwards. **Self-improvement without a strong
  verifier is self-degradation** — it improves toward whatever you measure. The verifier is the loop.
- *"Let it auto-merge, the agent wrote good code."* → The agent also wrote the code that needs fixing.
  Knowledge that edits and ships itself with no gate rots silently. Keep a PR and a reviewer.
- *"A 👍/👎 is enough signal."* → A verdict without a **reason** can't produce a correct diff. Capture
  *why* it was wrong and what the right answer was.
- *"I'll just run it in a `while` loop on a box."* → That's uptime, not durability. A restart re-runs
  steps and re-fires side effects. Cron + checkpoint + idempotency.
- *"More loops / more agents = more improvement."* → More loops with weak signals = more drift, more
  cost, more PRs to review. One loop with a real signal beats five without.
- *"The Skill keeps getting longer, so it's getting better."* → Length is not quality; it's often the
  verifier rewarding verbosity. Measure the inner loop's success rate, not the file size.

---

## Red flags — stop and reconsider

- You cannot name the **external signal**, or it lives only in someone's head.
- The reviser can **merge its own edits** with no human or adversarial gate.
- There is **no regression check** added when a lesson is "learned."
- The runner is a **long-lived process**, not a cron; a restart would lose or double-fire work.
- Side effects use a **fresh idempotency key** per attempt (or none).
- The Skill file **grows every cycle** but the inner loop's success rate doesn't move.
- The reviser can **edit the feedback/run history** it learns from.
- The loop touches **all** skills every run regardless of signal (cost with no return).

---

## Verification gate (definition of done)

A self-improvement loop is done when:

- [ ] **Signal named & recorded.** A real external signal (human edit / grader / test / outcome) lands
      in a durable, append-only place the reviser reads.
- [ ] **Inner loop records every run** (input → output → outcome handle).
- [ ] **Reviser is scheduled** and turns negative signal into a **Skill diff PR**, scoped to skills with
      signal this cycle, citing the motivating runs.
- [ ] **Verify gate enforced:** mechanical checks green **and** an independent/adversarial review
      affirms the diff is correct and net-better; **a human merges** (no auto-merge).
- [ ] **Lesson distilled:** a general rule added to the Skill's anti-patterns/constraints **plus** a
      regression check that now passes and guards it.
- [ ] **Runner is durable & idempotent:** survives restart (cron + checkpointed steps), and a stable
      `(skill, cycle)` key prevents duplicate PRs/side effects.
- [ ] **Observable & bounded:** the step history reconstructs why each change was made; runs/tokens/PRs
      are capped.

If any box is unchecked, the loop is not done — report which.

---

## Version awareness

It is 2026 and this tooling moves fast. The *pattern* (doer/signal/reviser, durable, gated) is stable;
the **substrates are not**. Verify current APIs before relying on them: Warp/Oz cloud-agent and Skill
formats; Claude Code scheduled/cloud agents and the `~/.claude/skills` layout; Inngest
(`step.run`/`step.invoke`, cron triggers) and Temporal SDKs; GitHub Actions `schedule:`/`workflow_run`
triggers and `GITHUB_TOKEN` permissions for agent-opened PRs. Treat every code/CLI snippet here as
correct-in-shape but **check the exact flags against current docs**.

---

## Canonical references (verify currency)

- Warp — *How to build a self-improvement loop for your Skills* and the `warpdotdev/oz-for-oss` sample
  repo (inner/outer loop with GitHub Issues).
- Inngest — durable functions, cron triggers, `step.run`/`step.invoke`, flow control/concurrency:
  inngest.com/docs.
- Temporal — durable execution, workflow-vs-activity, schedules, idempotency: docs.temporal.io.
- Anthropic — Claude Code Skills, scheduled/cloud agents, and Agent SDK docs.
- This library — `tests/VALIDATION.md` (the 5-layer gate, esp. Layer 5 step 4 "feed failures back"),
  `scripts/validate.py`, `scripts/functional_test.py`, `scripts/skill_feedback.py`, `feedback/README.md`.
- Related: `[[llm-app-agent-frameworks]]` §7 (durable orchestration), `[[ml-evaluation-evals]]`
  (graders/judges as the signal & gate), `[[engineering-lifecycle]]` (the Verify→Review→Ship gates the
  reviser's PR passes through).
