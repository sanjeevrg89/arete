# Skill feedback — the signal for the self-improvement loop

This directory holds the **signal** that drives the skill self-improvement loop (see the
[`skill-self-improvement`](../skill-self-improvement/) skill). Each record says *a skill was applied to a
real task, and here's what was right or wrong about it.* The outer-loop reviser reads these records,
groups them by skill, and opens a PR diffing the skills that accumulated negative signal.

**A loop is only as good as its signal.** Capture the *reason* a run was wrong and the *correct* answer
— not just a thumbs-down. A bare verdict can't produce a correct diff.

## Format — `log.jsonl`

One JSON object per line (JSON Lines). Append-only — the reviser **reads** this; it must never rewrite
the history it learns from. Fields:

| Field | Required | Meaning |
|-------|----------|---------|
| `skill` | ✅ | the skill's directory name (must exist, e.g. `serving-frameworks`) |
| `verdict` | ✅ | `good` or `bad` |
| `task` | ✅ | one line: what the skill was used for |
| `what_was_wrong` | for `bad` | the specific defect — routing miss / wrong claim / too shallow / missing case |
| `correct_answer` | encouraged | what it *should* have said/done (the teachable part) |
| `source` | optional | where the signal came from (`human-review`, `failing-check`, `prod-outcome`, `example`) |
| `date` | optional | `YYYY-MM-DD` (the loop converts to absolute when distilling) |

Example record:

```json
{"skill":"serving-frameworks","verdict":"bad","task":"pick an engine for a 70B model at 200 QPS","what_was_wrong":"recommended an engine but never asked about the latency SLO or context length","correct_answer":"should gate the recommendation on TTFT/ITL targets and max context before naming an engine","source":"human-review","date":"2026-06-19"}
```

## How records get here

- **By hand** during real use: when a skill underperforms, append a record with the reason.
- **From failing checks:** a failing `tests/functional/checks.json` assertion is itself a signal — the
  reviser can read the check output directly, or you log it here.
- **From an integration (Warp-style):** a GitHub Action runs a skill, a human corrects the result, and a
  job appends the correction here (or the reviser reads the GitHub issue/PR directly).

## Validate & rank

```bash
python scripts/skill_feedback.py --lint   # CI-safe: validates every record (structure + skill exists)
python scripts/skill_feedback.py          # ranks skills by negative signal — the improvement candidates
```

The seed rows in `log.jsonl` have `"source":"example"` — **delete them** and add your own real feedback.
They exist only to show the format and exercise the linter.
