#!/usr/bin/env python3
"""Routing eval: does an agent pick the right skill from its descriptions alone?

The `description` in each SKILL.md is the router — it is all an agent sees before deciding what to
load. This script replays one discriminating prompt per skill (tests/skill-routing-checklist.md)
against the catalog an agent would see (`- name: description`, sorted, model-invocable skills only)
and scores whether the agent names the expected skill.

  python scripts/routing_eval.py --lint     Check the cases: every expected/neighbor skill exists and
                                            every first-party skill has a case. CI-safe; no agent.
  AGENT_CMD='claude -p --model haiku --disable-slash-commands --tools "" --no-session-persistence' \\
    python scripts/routing_eval.py --jobs 8
                                            Pipe each routing prompt to $AGENT_CMD on stdin, parse the
                                            skill it names, report top-1 accuracy (strict = expected
                                            skill; lenient = expected or a listed neighbor).
  ... --only SKILL[,SKILL]                  Re-run just those skills' cases after editing a description.
  ... --listing FILE                        Route against a pre-rendered listing instead of the repo's
                                            SKILL.md files (e.g. a listing captured from a real session,
                                            where the agent's budget dropped some descriptions).

Stdlib only (PyYAML is used for frontmatter when installed).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
CHECKLIST = ROOT / "tests" / "skill-routing-checklist.md"

CASE = re.compile(r'^- \[[ x]\] \*\*`([a-z0-9-]+)`\*\* — "(.+)"(?:\s+_\((.+)\)_)?\s*$')

ROUTER_PROMPT = """You are the skill router of a coding agent. Below is the catalog of installed skills \
(name: description). Pick the ONE skill whose instructions you would load first for the task. Reply \
with only the skill name exactly as written in the catalog, or NONE if no skill applies.

Catalog:
{catalog}

Task:
{task}
"""


def frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\r?\n(.*?)\r?\n---", text, re.S)
    if not m:
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(m.group(1))
        return data if isinstance(data, dict) else {}
    except ImportError:
        sys.path.insert(0, str(ROOT / "scripts"))
        import validate  # stdlib fallback parser

        return validate.parse_frontmatter(text) or {}
    except Exception:
        return {}  # invalid YAML: agents see no description (Claude Code) or skip it entirely


def repo_catalog() -> dict[str, str]:
    """name -> description for every model-invocable skill (first-party + vendored)."""
    out: dict[str, str] = {}
    for sm in sorted(SKILLS.rglob("SKILL.md")):
        fm = frontmatter(sm)
        if fm.get("disable-model-invocation") in (True, "true"):
            continue
        out[sm.parent.name] = str(fm.get("description") or "").strip()
    return out


def listing_catalog(path: Path) -> dict[str, str]:
    """Parse a rendered listing: `- name: description` lines (name-only lines have no description)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- "):
            continue
        name, _, desc = line[2:].partition(": ")
        out[name.strip()] = desc.strip()
    return out


def render(catalog: dict[str, str]) -> str:
    return "\n".join(f"- {n}: {d}" if d else f"- {n}" for n, d in sorted(catalog.items()))


def load_cases() -> list[dict]:
    cases = []
    for line in CHECKLIST.read_text(encoding="utf-8").splitlines():
        m = CASE.match(line.strip())
        if m:
            accept = [s.strip() for s in (m.group(3) or "").split(",")]
            cases.append({"expect": m.group(1), "prompt": m.group(2), "accept": accept})
    return cases


def lint(cases: list[dict]) -> int:
    known = {p.parent.name for p in SKILLS.rglob("SKILL.md")}
    first_party = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    errors = []
    for c in cases:
        if c["expect"] not in known:
            errors.append(f"case expects unknown skill '{c['expect']}'")
        for a in c["accept"]:
            if re.fullmatch(r"[a-z0-9-]+", a) and a not in known:
                errors.append(f"case for '{c['expect']}' lists unknown neighbor '{a}'")
    covered = {c["expect"] for c in cases}
    for s in sorted(first_party - covered):
        errors.append(f"skill '{s}' has no routing case in {CHECKLIST.relative_to(ROOT)}")
    for e in errors:
        print(f"  ERROR {e}")
    print(f"{len(cases)} routing cases, {len(errors)} error(s)")
    return 1 if errors else 0


def parse_answer(out: str, names: list[str]) -> str:
    """First catalog name mentioned in the reply (longest match wins at a position)."""
    text = out.strip().strip("`*'\". ")
    if text in names:
        return text
    best = (len(out) + 1, "")
    for n in names:
        m = re.search(rf"(?<![a-z0-9-]){re.escape(n)}(?![a-z0-9-])", out)
        if m and (m.start(), -len(n)) < (best[0], -len(best[1])):
            best = (m.start(), n)
    return best[1] or ("NONE" if "NONE" in out.upper() else "?")


def route(cmd: list[str], prompt: str, names: list[str], timeout: int) -> str:
    for _ in range(2):  # one retry on a failed or empty call
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        if p.returncode == 0 and p.stdout.strip():
            return parse_answer(p.stdout, names)
    return "ERROR"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lint", action="store_true", help="validate the cases only (no agent)")
    ap.add_argument("--listing", type=Path, help="route against a pre-rendered listing file")
    ap.add_argument("--jobs", type=int, default=4, help="parallel agent calls (default 4)")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per agent call")
    ap.add_argument("--json", type=Path, help="write per-case results to this file")
    ap.add_argument("--only", help="comma-separated skills: run only their cases (after editing one)")
    args = ap.parse_args()

    cases = load_cases()
    if args.lint:
        return lint(cases)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        cases = [c for c in cases if c["expect"] in wanted]

    agent = os.environ.get("AGENT_CMD", "").strip()
    if not agent:
        print("Set AGENT_CMD to run the eval, e.g.")
        print("  AGENT_CMD='claude -p --model haiku --disable-slash-commands --tools \"\" "
              "--no-session-persistence' python scripts/routing_eval.py --jobs 8")
        print(f"{len(cases)} cases available; run with --lint to check them.")
        return 0

    catalog = listing_catalog(args.listing) if args.listing else repo_catalog()
    names = sorted(catalog)
    listing = render(catalog)
    cmd = shlex.split(agent)
    with_desc = sum(1 for d in catalog.values() if d)
    print(f"catalog: {len(catalog)} skills ({with_desc} with a description), {len(listing)} chars")

    def run(c: dict) -> dict:
        got = route(cmd, ROUTER_PROMPT.format(catalog=listing, task=c["prompt"]), names, args.timeout)
        return {**c, "got": got}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(run, cases))

    strict = sum(r["got"] == r["expect"] for r in results)
    lenient = sum(r["got"] == r["expect"] or r["got"] in r["accept"] for r in results)
    for r in results:
        if r["got"] != r["expect"]:
            tag = "ok~" if r["got"] in r["accept"] else "MISS"
            print(f"  {tag:4} expected {r['expect']:<34} got {r['got']}")
    n = len(results)
    print(f"strict top-1: {strict}/{n} ({strict / n:.0%})   lenient: {lenient}/{n} ({lenient / n:.0%})")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
