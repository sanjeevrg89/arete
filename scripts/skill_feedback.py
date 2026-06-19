#!/usr/bin/env python3
"""Skill feedback: the *signal* for the self-improvement loop (see the skill-self-improvement skill).

Reads feedback/log.jsonl — one JSON object per line, each recording that a skill was applied to a real
task and whether it was good or bad. Two modes, mirroring functional_test.py:

  python scripts/skill_feedback.py --lint   Validate every record (structure + skill exists). CI-safe.
  python scripts/skill_feedback.py          Aggregate: rank skills by negative signal — the list the
                                            outer-loop reviser should consider improving this cycle.

A missing or empty log is fine (exit 0): no signal yet, nothing to do. Stdlib only.

Record fields: skill (req, must exist), verdict (req: good|bad), task (req), what_was_wrong (req for
bad), correct_answer/source/date (optional). See feedback/README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "feedback" / "log.jsonl"
VALID_VERDICTS = {"good", "bad"}


def load_records() -> list[tuple[int, dict]]:
    """Return (line_number, record) for each non-blank line; exit on malformed JSON."""
    if not LOG.exists():
        print(f"no feedback log at {LOG.relative_to(ROOT)} — nothing to do.")
        return []
    out: list[tuple[int, dict]] = []
    for n, raw in enumerate(LOG.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ERROR line {n}: invalid JSON: {e}")
            sys.exit(1)
        if not isinstance(obj, dict):
            print(f"  ERROR line {n}: each line must be a JSON object")
            sys.exit(1)
        out.append((n, obj))
    return out


def lint(records: list[tuple[int, dict]]) -> int:
    errors: list[str] = []
    for n, c in records:
        for key in ("skill", "verdict", "task"):
            if key not in c or c.get(key) in (None, ""):
                errors.append(f"line {n}: missing '{key}'")
        skill = c.get("skill", "")
        if skill and not (ROOT / skill / "SKILL.md").exists():
            errors.append(f"line {n}: skill '{skill}' has no SKILL.md")
        verdict = c.get("verdict")
        if verdict is not None and verdict not in VALID_VERDICTS:
            errors.append(f"line {n}: verdict '{verdict}' not in {sorted(VALID_VERDICTS)}")
        if verdict == "bad" and not c.get("what_was_wrong"):
            errors.append(f"line {n}: a 'bad' verdict needs 'what_was_wrong' (the teachable reason)")
    print(f"Linted {len(records)} feedback records.")
    for e in errors:
        print(f"  ERROR {e}")
    if errors:
        print("FAILED")
        return 1
    print("OK")
    return 0


def report(records: list[tuple[int, dict]]) -> int:
    if not records:
        print("No feedback recorded yet — the reviser has nothing to improve this cycle.")
        return 0
    bad: dict[str, int] = {}
    good: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    for _, c in records:
        skill = c.get("skill", "?")
        if c.get("verdict") == "bad":
            bad[skill] = bad.get(skill, 0) + 1
            if c.get("what_was_wrong"):
                reasons.setdefault(skill, []).append(c["what_was_wrong"])
        elif c.get("verdict") == "good":
            good[skill] = good.get(skill, 0) + 1

    ranked = sorted(bad.items(), key=lambda kv: kv[1], reverse=True)
    print(f"Feedback: {len(records)} records across "
          f"{len(set(list(bad) + list(good)))} skills.\n")
    if not ranked:
        print("No negative signal — nothing to improve. (Loops need a real signal to act on.)")
        return 0
    print("Improvement candidates (ranked by negative signal):")
    for skill, n in ranked:
        g = good.get(skill, 0)
        print(f"  {n:>3} bad / {g} good  {skill}")
        for r in reasons.get(skill, [])[:3]:
            print(f"        - {r}")
    print("\nThe reviser should open one PR per candidate, diffing its guide + adding a regression "
          "check. It must pass the verify gate and be human-merged (see skill-self-improvement).")
    return 0


def main() -> int:
    records = load_records()
    if "--lint" in sys.argv:
        return lint(records)
    return report(records)


if __name__ == "__main__":
    sys.exit(main())
