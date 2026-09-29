#!/usr/bin/env python3
"""Validate the skill library: SKILL.md frontmatter, unique names, required files, cross-links.

Frontmatter is parsed with a strict YAML parser when PyYAML is installed (CI installs it): agents
parse strictly too, and a SKILL.md they can't parse loses its description (Claude Code) or is skipped
outright (`npx skills`). Without PyYAML a heuristic check catches the common failure (an unquoted
`: ` in a plain scalar). Exits non-zero if any ERROR is found; WARNINGs do not fail the build. Run
from anywhere: `python scripts/validate.py`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # stdlib-only fallback: heuristic YAML check
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
VENDORED = SKILLS / "vendored"

# Vendored third-party skills (skills/vendored/<upstream>/<name>/) follow their upstream's layout;
# they are validated loosely (valid frontmatter, spec-legal name + description), not against arete's
# house spec.

# Files every skill directory must contain (a guide is checked separately).
REQUIRED_FILES = ["SKILL.md", "AGENTS.md", "GEMINI.md"]

# The description is the router, and every installed skill's description shares one listing budget
# (Claude Code truncates the skill listing at a fixed character budget; Codex caps its list at 2% of
# context). Keep first-party routers short so the whole library stays visible next to other skills.
DESC_MAX = 250  # house limit, first-party
SPEC_DESC_MAX = 1024  # Agent Skills spec limit, applies to vendored skills too
DESC_BUDGET = 15000  # warn when all first-party routers together exceed this
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")  # spec: lowercase, digits, single hyphens
NAME_MAX = 64
# Finder/iCloud conflict copies ("SKILL 2.md", "validate 2.py") must never be committed.
DUPLICATE_RE = re.compile(r" \d+(\.[^.]+)?$")

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


def frontmatter_block(text: str) -> str | None:
    m = re.match(r"^---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.S)
    return m.group(1) if m else None


def check_yaml(label: str, text: str) -> dict | None:
    """Parse frontmatter the way agents do. Returns the strict mapping (None without PyYAML or on
    error); every problem is reported as an ERROR."""
    block = frontmatter_block(text)
    if block is None:
        return None  # reported by the caller
    why = "agents drop its description or skip the skill; quote the value or reword the ': '"
    if yaml is None:
        # Heuristic: an unquoted plain scalar may not contain ": " or " #" (YAML reads those as a
        # nested mapping / a comment). Continuation lines are folded into their key's value first.
        values: dict[str, str] = {}
        key = None
        for line in block.splitlines():
            m = re.match(r"^([A-Za-z][\w-]*):(?:\s+(.*))?$", line)
            if m:
                key = m.group(1)
                values[key] = (m.group(2) or "").strip()
            elif key is not None and line.strip():
                values[key] += " " + line.strip()
        for k, v in values.items():
            if v[:1] in ("'", '"', "|", ">", "[", "{") or not v:
                continue
            if ": " in v or " #" in v:
                err(f"{label}: frontmatter `{k}` is an unquoted value containing ': ' or ' #' — {why}")
        return None
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as e:
        problem = getattr(e, "problem", None) or str(e).splitlines()[0]
        err(f"{label}: frontmatter is not valid YAML ({problem}) — {why}")
        return None
    if not isinstance(data, dict):
        err(f"{label}: frontmatter is not a YAML mapping")
        return None
    return data


def check_name(label: str, name: str) -> None:
    if len(name) > NAME_MAX or not NAME_RE.match(name):
        err(f"{label}: name '{name}' must be lowercase letters/digits/single hyphens, <= {NAME_MAX} chars")


def duplicate_files() -> list[Path]:
    """Finder/iCloud conflict copies anywhere in the repo ('SKILL 2.md', 'validate 2.py')."""
    out = []
    for p in ROOT.rglob("*"):
        parts = p.relative_to(ROOT).parts
        if parts[0] in (".git", "node_modules") or not DUPLICATE_RE.search(p.name):
            continue
        out.append(p)
    return out


def skill_dirs() -> list[Path]:
    """First-party skills: directories directly under skills/ containing a SKILL.md."""
    out = []
    if not SKILLS.is_dir():
        return out
    for p in sorted(SKILLS.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "SKILL.md").exists():
            out.append(p)
    return out


def vendored_dirs() -> list[Path]:
    """Vendored third-party skills: skills/vendored/<upstream>/<name>/ with a SKILL.md."""
    out = []
    if not VENDORED.is_dir():
        return out
    for upstream in sorted(VENDORED.iterdir()):
        if not upstream.is_dir() or upstream.name.startswith("."):
            continue
        for p in sorted(upstream.iterdir()):
            if p.is_dir() and (p / "SKILL.md").exists():
                out.append(p)
    return out


def main() -> int:
    dirs = skill_dirs()
    vdirs = vendored_dirs()
    if not dirs and not vdirs:
        err("no skill directories found under skills/")
        return finish()

    names: dict[str, Path] = {}
    all_slugs: set[str] = {d.name for d in dirs} | {d.name for d in vdirs}
    budget = 0

    for p in duplicate_files():
        err(f"{p.relative_to(ROOT)}: Finder/iCloud duplicate copy — delete it")

    for d in dirs:
        name = d.name
        sm = d / "SKILL.md"
        text = sm.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        strict = check_yaml(name, text)

        # Frontmatter + name/description.
        if fm is None:
            err(f"{name}: SKILL.md has no valid `---` frontmatter block")
        else:
            if "name" not in fm or not fm["name"]:
                err(f"{name}: SKILL.md frontmatter missing `name`")
            elif fm["name"] != name:
                err(f"{name}: frontmatter name '{fm['name']}' != directory name '{name}'")
            else:
                check_name(name, fm["name"])
                if fm["name"] in names:
                    err(f"{name}: duplicate skill name '{fm['name']}' (also {names[fm['name']].name})")
                names[fm["name"]] = d
            desc = str(strict.get("description") or "") if strict else fm.get("description", "")
            budget += len(desc)
            if not desc:
                err(f"{name}: SKILL.md frontmatter missing `description` (the router)")
            elif len(desc) < 40:
                warn(f"{name}: description is short ({len(desc)} chars) — make it a stronger router")
            elif len(desc) > DESC_MAX:
                err(f"{name}: description is {len(desc)} chars (max {DESC_MAX}) — lead with the "
                    "trigger terms; move detail to the `## Scope and triggers` section")
        if "\n## Scope and triggers\n" not in text:
            warn(f"{name}: SKILL.md has no `## Scope and triggers` section")

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

    # Vendored skills: loose checks only (valid frontmatter, name == dir, unique, description).
    for d in vdirs:
        rel = f"vendored/{d.parent.name}/{d.name}"
        text = (d / "SKILL.md").read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        strict = check_yaml(rel, text)
        if fm is None:
            err(f"{rel}: SKILL.md has no valid `---` frontmatter block")
            continue
        if "name" not in fm or not fm["name"]:
            err(f"{rel}: SKILL.md frontmatter missing `name`")
        elif fm["name"] != d.name:
            err(f"{rel}: frontmatter name '{fm['name']}' != directory name '{d.name}'")
        elif fm["name"] in names:
            err(f"{rel}: duplicate skill name '{fm['name']}' (also {names[fm['name']].name})")
        else:
            check_name(rel, fm["name"])
            names[fm["name"]] = d
        desc = str(strict.get("description") or "") if strict else (fm.get("description") or "")
        if not desc.strip():
            err(f"{rel}: SKILL.md frontmatter missing `description` (the router)")
        elif len(desc) > SPEC_DESC_MAX:
            err(f"{rel}: description is {len(desc)} chars (Agent Skills spec max {SPEC_DESC_MAX})")

    print(f"Validated {len(dirs)} first-party + {len(vdirs)} vendored skills.")
    parser = "strict YAML (PyYAML)" if yaml else "heuristic YAML check (pip install pyyaml for strict)"
    print(f"Frontmatter: {parser}. Router budget: {budget} chars across {len(dirs)} first-party "
          f"descriptions (warn above {DESC_BUDGET}).")
    if budget > DESC_BUDGET:
        warn(f"first-party descriptions total {budget} chars (> {DESC_BUDGET}); every installed "
             "skill shares one listing budget — tighten the longest routers")
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
