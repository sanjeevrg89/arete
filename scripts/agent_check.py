#!/usr/bin/env python3
"""Agent check: which agents on this machine can actually see this library's skills?

Installing is not the same as being seen — each agent looks in different directories and parses
frontmatter its own way. For every agent CLI found on PATH this asks the agent itself where it can
(`codex debug prompt-input`, `gemini skills list`, `opencode debug skill`), and otherwise checks the
directory that agent reads. No model calls, no network.

  python scripts/agent_check.py         Table of every agent found; exits 1 if an installed agent
                                        is missing skills (the last column says how to fix it).
  python scripts/agent_check.py --lint  Only scan the library (no agent is run). CI-safe.

Stdlib only.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
HOME = Path.home()
CWD = tempfile.mkdtemp(prefix="agent-check-")  # neutral directory: no project-level skills leak in
atexit.register(shutil.rmtree, CWD, ignore_errors=True)


def run(cmd: list[str], timeout: int = 90) -> tuple[int, str, str]:
    """(exit code, stdout, stderr) — never raises.

    Output goes to files, not pipes: some agent CLIs exit before a pipe drains and lose everything
    past the first 64 KB (a full skill listing is far larger).
    """
    out_path, err_path = Path(CWD) / "stdout.txt", Path(CWD) / "stderr.txt"
    try:
        with open(out_path, "w") as out, open(err_path, "w") as err:
            rc = subprocess.run(cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL, timeout=timeout, cwd=CWD).returncode
        return rc, out_path.read_text(errors="replace"), err_path.read_text(errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def version(binary: str) -> str:
    rc, out, err = run([binary, "--version"], timeout=30)
    m = re.search(r"\d[\w.\-]*", out or err) if rc == 0 else None
    return m.group(0) if m else "?"


def library() -> tuple[set[str], set[str]]:
    """(every skill name, names an agent may load on its own).

    User-invoked skills opt out of implicit loading via `disable-model-invocation: true` (Claude Code)
    and `allow_implicit_invocation: false` in agents/openai.yaml (Codex); agents that honor those
    flags leave them out of the model-visible list.
    """
    names, auto = set(), set()
    for sm in SKILLS.rglob("SKILL.md"):
        name = sm.parent.name
        names.add(name)
        front = sm.read_text(encoding="utf-8").split("---", 2)[1]
        oa = sm.parent / "agents" / "openai.yaml"
        opted_out = re.search(r"^disable-model-invocation:\s*true", front, re.M) or (
            oa.exists() and re.search(r"allow_implicit_invocation:\s*false", oa.read_text(encoding="utf-8"))
        )
        if not opted_out:
            auto.add(name)
    return names, auto


def in_dir(d: Path, names: set[str]) -> set[str]:
    return {n for n in names if (d / n / "SKILL.md").is_file()}


def tilde(p: Path) -> str:
    s = str(p)
    return "~" + s[len(str(HOME)):] if s.startswith(str(HOME)) else s


def check_claude(names: set[str], auto: set[str]) -> tuple[set[str], set[str], str, str]:
    d = Path(os.environ.get("CLAUDE_CONFIG_DIR", HOME / ".claude")) / "skills"
    found = in_dir(d, names)
    try:  # forks rename the plugin; read its name from the manifest
        plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))["name"]
    except (OSError, ValueError, KeyError):
        plugin = "arete"
    if not found and f"{plugin}@" in run(["claude", "plugin", "list"], timeout=60)[1]:
        return names, names, f"installed as the {plugin} plugin (run /skills in a session)", ""
    return found, names, f"checked {tilde(d)} (run /skills in a session)", "./install.sh claude"


def check_codex(names: set[str], auto: set[str]) -> tuple[set[str], set[str], str, str]:
    rc, out, _ = run(["codex", "debug", "prompt-input", "x"])
    try:  # the rendered prompt is a JSON list of messages; join their text parts
        items = json.loads(out) if rc == 0 else []
        out = "\n".join(c.get("text", "") for it in items for c in (it.get("content") or []) if isinstance(c, dict))
    except (ValueError, AttributeError, TypeError):
        out = ""
    if "<skills_instructions>" not in out:
        found = in_dir(HOME / ".agents" / "skills", names)
        return found, names, "checked ~/.agents/skills (this Codex has no `debug prompt-input`)", "./install.sh agents"
    block = out[out.find("<skills_instructions>"):out.find("</skills_instructions>")]
    listed = {m.split(":")[-1] for m in re.findall(r"^- ([a-z0-9:-]+): ", block, re.M)}
    return listed & names, auto, "listed in the prompt by `codex debug prompt-input`", "./install.sh agents"


def check_gemini(names: set[str], auto: set[str]) -> tuple[set[str], set[str], str, str]:
    # Old releases have no `skills` command and would treat the words as a prompt — check help first.
    rc, out, err = run(["gemini", "skills", "--help"], timeout=60)
    if "skills list" not in out + err:
        return set(), names, "this version has no `gemini skills` command", "upgrade Gemini CLI, then ./install.sh agents"
    rc, out, _ = run(["gemini", "skills", "list", "--all"])
    listed = set(re.findall(r"^(\S+) \[(?:Enabled|Disabled)\]", out, re.M))
    return listed & names, names, "listed by `gemini skills list`", "./install.sh agents"


def check_opencode(names: set[str], auto: set[str]) -> tuple[set[str], set[str], str, str]:
    rc, out, _ = run(["opencode", "debug", "skill"])
    try:
        listed = {s["name"] for s in json.loads(out)}
        return listed & names, names, "listed by `opencode debug skill`", "./install.sh agents"
    except (ValueError, KeyError, TypeError):
        found = in_dir(HOME / ".agents" / "skills", names) | in_dir(HOME / ".claude" / "skills", names)
        return found, names, "checked ~/.agents/skills or ~/.claude/skills", "./install.sh agents"


def check_qwen(names: set[str], auto: set[str]) -> tuple[set[str], set[str], str, str]:
    d = HOME / ".qwen" / "skills"
    return in_dir(d, names), names, f"checked {tilde(d)} (the only place Qwen Code looks)", "./install.sh qwen"


def check_cursor(names: set[str], auto: set[str]) -> tuple[set[str], set[str], str, str]:
    found = in_dir(HOME / ".agents" / "skills", names) | in_dir(HOME / ".cursor" / "skills", names)
    return found, names, "checked ~/.agents/skills or ~/.cursor/skills", "./install.sh agents"


AGENTS = [
    ("Claude Code", "claude", check_claude),
    ("Codex CLI", "codex", check_codex),
    ("Gemini CLI", "gemini", check_gemini),
    ("OpenCode", "opencode", check_opencode),
    ("Qwen Code", "qwen", check_qwen),
    ("Cursor agent", "cursor-agent", check_cursor),
]


def main() -> int:
    names, auto = library()
    print(f"{len(names)} skills in this library ({len(auto)} that an agent may load on its own).\n")
    if "--lint" in sys.argv[1:]:
        print("Checks available for: " + ", ".join(label for label, _, _ in AGENTS))
        return 0 if names and auto else 1
    rows, short = [("AGENT", "VERSION", "SEES", "HOW IT WAS CHECKED", "")], False
    for label, binary, check in AGENTS:
        if not shutil.which(binary):
            rows.append((label, "-", "not installed", "", ""))
            continue
        found, expected, how, fix = check(names, auto)
        ok = expected <= found
        short = short or not ok
        rows.append((label, version(binary), f"{len(found & expected)}/{len(expected)}", how, "" if ok else f"-> {fix}"))
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    for r in rows:
        print("  ".join(r[i].ljust(widths[i]) for i in range(3)) + "  " + " ".join(x for x in r[3:] if x))
    print("\nAnything not listed: npx skills add <this repo> knows 70+ other agents' directories.")
    return 1 if short else 0


if __name__ == "__main__":
    sys.exit(main())
