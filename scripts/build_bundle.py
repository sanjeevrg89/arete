#!/usr/bin/env python3
"""Build a flat, self-contained markdown bundle from the directory-per-skill source.

Each first-party skill (`skills/<name>/SKILL.md` + guide + `examples.md`) is flattened into a single
file `bundle/<name>.md` (frontmatter preserved, full guide + examples inlined). This is the
loader-agnostic form for tools that load skills as flat markdown files. Vendored third-party skills
(`skills/vendored/<upstream>/<name>/`) are NOT flattened — their extra reference files wouldn't
inline correctly; install them directory-style instead. Adapter files (AGENTS.md / GEMINI.md) are
intentionally excluded — they're redundant once inlined.

Run: `python scripts/build_bundle.py`  (regenerate after adding/editing skills).
Stdlib only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "skills"
OUT = ROOT / "bundle"
SEP = "\n\n---\n\n"


def find_guide(d: Path) -> Path | None:
    g = list(d.glob("*-guide.md"))
    if g:
        return g[0]
    if (d / "go-guidelines.md").exists():
        return d / "go-guidelines.md"
    return None


def main() -> int:
    OUT.mkdir(exist_ok=True)
    built = 0
    skipped = []
    for d in sorted(SRC.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        skill = d / "SKILL.md"
        if not skill.exists():
            continue
        guide = find_guide(d)
        if guide is None:
            skipped.append(f"{d.name} (no guide)")
            continue
        parts = [skill.read_text(encoding="utf-8").rstrip()]
        parts.append(f"# Reference — {d.name}\n\n" + guide.read_text(encoding="utf-8").strip())
        ex = d / "examples.md"
        if ex.exists():
            parts.append(ex.read_text(encoding="utf-8").strip())
        (OUT / f"{d.name}.md").write_text(SEP.join(parts) + "\n", encoding="utf-8")
        built += 1
    print(f"Built {built} flat skill files into {OUT.relative_to(ROOT)}/")
    for s in skipped:
        print(f"  skipped {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
