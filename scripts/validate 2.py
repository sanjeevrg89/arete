#!/usr/bin/env python3
"""Validate the skill library: SKILL.md frontmatter, unique names, required files, cross-links.

Stdlib only (no pyyaml) so it runs anywhere. Exits non-zero if any ERROR is found; WARNINGs do not
fail the build. Run from anywhere: `python scripts/validate.py`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that are not skills.
NON_SKILL_DIRS = {"scripts", ".git", ".github"}

# Files every skill directory must contain (a guide is checked separately).
REQUIRED_FILES = ["SKILL.md", "AGENTS.md", "GEMINI.md"]

# Framing we never want published (open, vendor-neutral content).
FORBIDDEN_PHRASES = [
    r"used at google",
    r"google internal",
    r"used internally",
    r"internal(?:ly)? at\b",
]
CO_AUTHOR = re.compile(r"co-authored-by", re.IGNORECASE)
BARE_INTERNALLY = re.compile(r"\binternally\b", re.IGNORECASE)
CROSSLINK = re.compile(r"\[\[([a-z0-9][a-z0-9-]*)\]\]")

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def parse_frontmatter(text: str) -> dict[str, str] | None:
    """Parse a leading `---` YAML-ish frontmatter block. Handles multi-line values."""
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None
    data: dict[str, str] = {}
    key = None
    for line in lines[1:end]:
        m = re.match(r"^([A-Za-z][\w-]*):\s?(.*)$", line)
        if m and not line.startswith((" ", "\t")):
            key = m.group(1)
            data[key] = m.group(2).strip()
        elif key is not None:
            data[key] = (data[key] + " " + line.strip()).strip()
    return data


def skill_dirs() -> list[Path]:
    out = []
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir() or p.name in NON_SKILL_DIRS or p.name.startswith("."):
            continue
        if (p / "SKILL.md").exists():
            out.append(p)
    return out


def main() -> int:
    dirs = skill_dirs()
    if not dirs:
        err("no skill directories found")
        return finish()

    names: dict[str, Path] = {}
    all_slugs: set[str] = {d.name for d in dirs}

    for d in dirs:
        name = d.name
        sm = d / "SKILL.md"
        fm = parse_frontmatter(sm.read_text(encoding="utf-8"))

        # Frontmatter + name/description.
        if fm is None:
            err(f"{name}: SKILL.md has no valid `---` frontmatter block")
        else:
            if "name" not in fm or not fm["name"]:
                err(f"{name}: SKILL.md frontmatter missing `name`")
            elif fm["name"] != name:
                err(f"{name}: frontmatter name '{fm['name']}' != directory name '{name}'")
            else:
                if fm["name"] in names:
                    err(f"{name}: duplicate skill name '{fm['name']}' (also {names[fm['name']].name})")
                names[fm["name"]] = d
            desc = fm.get("description", "")
            if not desc:
                err(f"{name}: SKILL.md frontmatter missing `description` (the router)")
            elif len(desc) < 40:
                warn(f"{name}: description is short ({len(desc)} chars) — make it a stronger router")

        # Required files.
        for f in REQUIRED_FILES:
            if not (d / f).exists():
                err(f"{name}: missing required file {f}")
        if not list(d.glob("*-guide.md")) and not (d / "go-guidelines.md").exists():
            err(f"{name}: missing a deep guide (`*-guide.md` or `go-guidelines.md`)")

        # GEMINI.md should import a guide.
        gem = d / "GEMINI.md"
        if gem.exists() and "@./" not in gem.read_text(encoding="utf-8"):
            warn(f"{name}: GEMINI.md does not `@./`-import its guide")

        # Per-file content checks.
        for md in d.glob("*.md"):
            text = md.read_text(encoding="utf-8")
            low = text.lower()
            for pat in FORBIDDEN_PHRASES:
                if re.search(pat, low):
                    err(f"{name}/{md.name}: forbidden framing matches /{pat}/")
            if CO_AUTHOR.search(text):
                err(f"{name}/{md.name}: contains a Co-Authored-By trailer")
            if BARE_INTERNALLY.search(text):
                warn(f"{name}/{md.name}: uses the word 'internally' — reword if it implies private use")
            for slug in CROSSLINK.findall(text):
                if slug not in all_slugs:
                    warn(f"{name}/{md.name}: cross-link [[{slug}]] has no matching skill directory")

    print(f"Validated {len(dirs)} skills.")
    return finish()


def finish() -> int:
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    print()
    print(f"{len(warnings)} warning(s), {len(errors)} error(s)")
    if errors:
        print("FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
