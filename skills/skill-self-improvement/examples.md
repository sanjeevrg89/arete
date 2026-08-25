# Skill Self-Improvement — worked examples

Copy-able artifacts for the doer→signal→reviser→gate→distill loop. See
`skill-self-improvement-guide.md` for the reasoning.

---

## 1. A feedback record (the signal)

`feedback/log.jsonl` — one JSON object per line. Capture the **reason** and the **correct answer**, not
just a verdict:

```json
{"skill":"serving-frameworks","verdict":"bad","task":"pick an engine for a 70B at 200 QPS","what_was_wrong":"named vLLM without asking the latency SLO or context length","correct_answer":"gate the choice on TTFT/ITL + max context first","source":"human-review","date":"2026-06-19"}
{"skill":"k8s-manifest-scaffolder","verdict":"good","task":"manifests for a stateless API","source":"failing-check","date":"2026-06-19"}
```

A bare `{"verdict":"bad"}` is useless — it can't produce a correct diff. The `what_was_wrong` +
`correct_answer` are what the reviser turns into a guide edit and a regression check.

---

## 2. The reviser procedure (the outer loop, one cycle)

What the scheduled agent does — driven by the `skill-self-improvement` skill through a coding agent:

```
1. RANK    python scripts/skill_feedback.py        # skills with the most negative signal this cycle
2. SCOPE   take the top candidate only (bound cost; one PR per skill per cycle)
3. READ    open <skill>/<skill>-guide.md + every 'bad' record for that skill
4. DIFF    edit the guide to fix the cited failures:
             - routing miss   -> tighten SKILL.md `description`
             - wrong claim    -> correct the guide, cite the source
             - recurring miss -> add a line to Anti-patterns / Red flags  (the distilled rule)
5. RATCHET add a must_* check to tests/functional/checks.json that fails on the old behavior
6. BUNDLE  python scripts/build_bundle.py           # regenerate skills/<skill>.md
7. PR      open a branch `skill-improve/<skill>-<cycle>`, cite the motivating records, DO NOT merge
```

Idempotency: the branch name is keyed by `(skill, cycle)` — a retry updates the same PR instead of
opening a second one.

A ready-to-use reviser prompt:

```
Apply the skill-self-improvement skill. For skill "<name>": read feedback/log.jsonl (its 'bad' records)
and its <name>-guide.md. Produce the smallest diff that fixes the cited failures, add the distilled rule
to the guide's Anti-patterns, and add a regression check to tests/functional/checks.json that would have
caught the failure. Regenerate the bundle. Open a PR explaining each change and citing the records — do
not merge. If a fix is uncertain or unsupported by a source, leave it out and say so.
```

---

## 3. The loop wired to this repo (end to end)

```
                INNER LOOP (per task)                          OUTER LOOP (weekly cron)
   ┌─────────────────────────────────────┐        ┌──────────────────────────────────────────┐
   │ a skill is applied to real work      │        │ skill-self-improvement reviser            │
   │  (a session, or functional_test.py)  │        │  reads the signal, diffs the skill        │
   └───────────────┬──────────────────────┘        └───────────────┬──────────────────────────┘
                   │ outcome                                        │ proposes
                   ▼                                                ▼
        feedback/log.jsonl  ──────────  signal  ──────────▶  PR: diff <skill>-guide.md
        (+ failing checks)                                   + new must_* in checks.json
                                                                    │
                                              VERIFY GATE ──────────┤  validate.py + functional_test --lint
                                              (never auto-merge)     │  + adversarial 2nd-model review
                                                                    ▼
                                                            human merges  ──▶  DISTILL: rule lands in
                                                                                Anti-patterns; bundle rebuilt
```

The dry-run runner is `.github/workflows/skill-self-improvement.yml` (cron, no secrets, writes nothing);
it lints the signal and prints the candidates. The real reviser step is documented there, gated, and
disabled by default.

---

## 4. A work loop: self-improving PR review

The same shape applied to a day-job task, using `[[code-review-discipline]]` as the doer:

| Part | Concretely |
|------|-----------|
| **Doer** | a GitHub Action runs the review skill on every PR and posts comments |
| **Signal** | which review comments get **resolved** (useful) vs **dismissed/ignored** (noise) |
| **Reviser** | a weekly agent reads resolved-vs-dismissed rates and diffs the review rubric: drop the categories that are always dismissed, sharpen the ones that catch real bugs |
| **Verify gate** | the rubric diff is a PR; a second model checks it doesn't drop a high-value check; a human merges |
| **Distill** | "stop flagging X (always dismissed)" becomes a rule in the rubric + a check |

After a few cycles the reviewer stops making the nits humans always dismiss and concentrates on what
they act on — your review judgment, encoded once, compounding.

---

## 5. Durable + idempotent runner (why not a `while` loop)

Bad — loses everything on restart, re-fires side effects:

```python
while True:                      # dies on deploy/OOM; on restart re-reads, re-decides, re-opens PRs
    sig = read_feedback()
    pr = open_pr(propose_diff(sig))
    sleep(WEEK)
```

Good — a cron + checkpointed steps + a stable idempotency key (shape shown with Inngest-style steps;
Temporal/GitHub-Actions-cron are equivalent — verify current APIs):

```ts
// cron: "0 10 * * 5"  — the heartbeat; the agent is the decision in the middle
const candidates = await step.run("rank-signal", () => rankFeedback());        // checkpointed
for (const skill of candidates.slice(0, 1)) {                                  // bounded scope
  const diff = await step.run(`propose-${skill}`, () => proposeDiff(skill));   // not re-run on replay
  await step.run(`open-pr-${skill}`, () =>
    openOrUpdatePR(`skill-improve/${skill}-${cycleId}`, diff));                 // idempotent key
}
```

On crash after `propose-${skill}`, replay returns the persisted diff and skips straight to the PR step —
no duplicate model call, no duplicate PR. That is the difference between uptime and durability.
