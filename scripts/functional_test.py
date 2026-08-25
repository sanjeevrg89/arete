#!/usr/bin/env python3
"""Functional checks for skills: "given this prompt, the output must satisfy X".

Unlike validate.py (which checks metadata), this exercises a skill's *behavior*. Because that needs an
agent in the loop, it runs in two modes:

  python scripts/functional_test.py --lint     Validate the check specs (structure + skill exists).
                                                CI-safe; no agent needed.
  AGENT_CMD='claude -p' python scripts/functional_test.py
                                                Run each prompt through the agent named by $AGENT_CMD
                                                (prompt piped on stdin) and assert must_contain /
                                                must_not_contain regexes against its stdout.

With no AGENT_CMD and no --lint, it prints the available checks and exits 0 (manual mode).

Check specs live in tests/functional/checks.json: a list of objects with
  { "skill", "prompt", "must_contain": [regex...], "must_not_contain": [regex...] (optional), "note" }
Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKS = ROOT / "tests" / "functional" / "checks.json"


def load_checks() -> list[dict]:
    if not CHECKS.exists():
        print(f"no checks file at {CHECKS.relative_to(ROOT)}")
        sys.exit(1)
    try:
        data = json.loads(CHECKS.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"checks.json is not valid JSON: {e}")
        sys.exit(1)
    if not isinstance(data, list):
        print("checks.json must be a JSON array")
        sys.exit(1)
    return data


def find_skill(name: str) -> Path | None:
    """Resolve a skill name to its directory: skills/<name>/ or skills/vendored/<upstream>/<name>/."""
    direct = ROOT / "skills" / name
    if (direct / "SKILL.md").exists():
        return direct
    vendored = ROOT / "skills" / "vendored"
    if vendored.is_dir():
        for upstream in sorted(vendored.iterdir()):
            if (upstream / name / "SKILL.md").exists():
                return upstream / name
    return None


def lint(checks: list[dict]) -> int:
    errors = []
    for i, c in enumerate(checks):
        where = f"check[{i}]"
        for key in ("skill", "prompt", "must_contain"):
            if key not in c:
                errors.append(f"{where}: missing '{key}'")
        skill = c.get("skill", "")
        if skill and find_skill(skill) is None:
            errors.append(f"{where}: skill '{skill}' has no SKILL.md")
        if "must_contain" in c and not isinstance(c["must_contain"], list):
            errors.append(f"{where}: must_contain must be a list")
        for pat in c.get("must_contain", []) + c.get("must_not_contain", []):
            try:
                re.compile(pat)
            except re.error as e:
                errors.append(f"{where}: bad regex {pat!r}: {e}")
    print(f"Linted {len(checks)} functional checks.")
    for e in errors:
        print(f"  ERROR {e}")
    if errors:
        print("FAILED")
        return 1
    print("OK")
    return 0


def run(checks: list[dict], agent_cmd: str) -> int:
    failed = 0
    for c in checks:
        try:
            out = subprocess.run(
                agent_cmd, shell=True, input=c["prompt"], capture_output=True,
                text=True, timeout=600,
            ).stdout
        except subprocess.TimeoutExpired:
            print(f"  FAIL {c['skill']}: agent timed out"); failed += 1; continue
        misses = [p for p in c.get("must_contain", []) if not re.search(p, out, re.I)]
        bad = [p for p in c.get("must_not_contain", []) if re.search(p, out, re.I)]
        if misses or bad:
            failed += 1
            print(f"  FAIL {c['skill']}: missing {misses} forbidden-present {bad}")
        else:
            print(f"  PASS {c['skill']}")
    print(f"\n{len(checks)-failed}/{len(checks)} passed")
    return 1 if failed else 0


def main() -> int:
    checks = load_checks()
    if "--lint" in sys.argv:
        return lint(checks)
    agent_cmd = os.environ.get("AGENT_CMD")
    if not agent_cmd:
        print(f"{len(checks)} functional checks available. Set AGENT_CMD to run them "
              f"(e.g. AGENT_CMD='claude -p'), or use --lint to validate specs.")
        for c in checks:
            print(f"  - {c['skill']}: {c['prompt'][:70]}…")
        return 0
    return run(checks, agent_cmd)


if __name__ == "__main__":
    sys.exit(main())
