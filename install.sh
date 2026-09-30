#!/usr/bin/env bash
# Install the skill library into your coding agents. Symlinks, so `git pull` updates every agent.
#
# Usage:
#   ./install.sh all             Link every skill into every agent on this machine: Claude Code,
#                                ~/.agents/skills (Codex CLI, Gemini CLI, OpenCode, Cursor), and
#                                Qwen Code if it is installed.
#   ./install.sh claude [dest]   Link every skill into a Claude Code skills dir
#                                (default: ~/.claude/skills, or $CLAUDE_CONFIG_DIR/skills).
#   ./install.sh agents [dest]   Link every skill into the cross-agent user skills dir
#                                (default: ~/.agents/skills — read by Codex CLI, Gemini CLI, OpenCode,
#                                and Cursor).
#   ./install.sh qwen [dest]     Link every skill into Qwen Code's skills dir (default:
#                                ~/.qwen/skills) — Qwen Code reads only its own directory.
#   ./install.sh link <dir>      Link every skill into any other agent's skills directory.
#   ./install.sh rules <file>    Link ONE rules file into each installed agent's global instructions
#                                (~/.claude/CLAUDE.md, ~/.codex/AGENTS.md, ~/.gemini/GEMINI.md,
#                                ~/.qwen/QWEN.md), so every agent follows the same rules. Never
#                                overwrites a real file. Starter: rules/AGENTS.md — replace it with
#                                your own. (OpenCode reads ~/.claude/CLAUDE.md too.)
#   ./install.sh flat <dest>     Copy the flat self-contained bundle/*.md into <dest>, for loaders that
#                                read plain markdown files (no SKILL.md discovery).
#   ./install.sh uninstall [dir...]  Remove the links this repo created (skills and rules), plus any
#                                in the extra <dir>s you linked. Leaves everything else alone.
#   ./install.sh list            List available skills.
#   ./install.sh help            Show this help.
#
# Linking is idempotent: re-run after `git pull` to pick up new skills; links to skills that no longer
# exist are pruned. A real (non-symlink) directory with a skill's name is never touched.
#
# Env:
#   SKILLS_DEST   default destination for `flat` if <dest> omitted.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_HOME="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
QWEN_HOME="$HOME/.qwen"
RULE_TARGETS=("$CLAUDE_HOME/CLAUDE.md" "${CODEX_HOME:-$HOME/.codex}/AGENTS.md" "$HOME/.gemini/GEMINI.md"
              "$QWEN_HOME/QWEN.md")

skill_dirs() {
  # First-party: skills/<name>/SKILL.md (depth 2) · Vendored: skills/vendored/<upstream>/<name>/SKILL.md (depth 4)
  find "$REPO/skills" -mindepth 2 -maxdepth 4 -name SKILL.md -exec dirname {} \; 2>/dev/null | sort
}

# Symlinks in <dir> that point into this repo.
our_links() {
  local l
  for l in "$1"/*; do
    [ -L "$l" ] || continue
    case "$(readlink "$l")" in "$REPO"/*) echo "$l";; esac
  done
}

link_skills() {
  local dest="$1" n=0 skipped=0 pruned=0 p name target
  mkdir -p "$dest"
  while IFS= read -r p; do
    name="$(basename "$p")"
    target="$dest/$name"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
      # `ln -sfn` onto a real directory would nest the link inside it; leave the user's copy alone.
      echo "  skip $name: $target is a real directory, not a link (remove it to let arete manage it)"
      skipped=$((skipped+1))
      continue
    fi
    ln -sfn "$p" "$target"
    n=$((n+1))
  done < <(skill_dirs)
  while IFS= read -r l; do
    [ -e "$l" ] && continue
    rm -f "$l"
    pruned=$((pruned+1))
  done < <(our_links "$dest")
  echo "Linked $n skills into $dest ($skipped skipped, $pruned stale links pruned)"
}

cmd_list() {
  while IFS= read -r p; do basename "$p"; done < <(skill_dirs)
  echo
  echo "$(skill_dirs | wc -l | tr -d ' ') skills."
}

cmd_claude() {
  link_skills "${1:-$CLAUDE_HOME/skills}"
  echo "Claude Code loads each skill on demand from its SKILL.md description (/skills lists them)."
}

cmd_agents() {
  link_skills "${1:-$HOME/.agents/skills}"
  echo "Codex CLI, Gemini CLI, OpenCode, and Cursor read this directory; restart them to pick up new skills."
}

cmd_qwen() {
  link_skills "${1:-$QWEN_HOME/skills}"
  echo "Qwen Code reads only this directory; restart it to pick up new skills."
}

cmd_link() {
  if [ -z "${1:-}" ]; then echo "usage: ./install.sh link <skills-dir>"; exit 1; fi
  link_skills "$1"
}

cmd_all() {
  cmd_claude
  cmd_agents
  if [ -d "$QWEN_HOME" ]; then cmd_qwen; fi
  echo "Check what each installed agent sees: python3 scripts/agent_check.py"
}

cmd_rules() {
  local src="${1:-}" t linked=0
  if [ -z "$src" ] || [ ! -f "$src" ]; then
    echo "usage: ./install.sh rules <file>    e.g. ./install.sh rules rules/AGENTS.md"
    exit 1
  fi
  src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
  for t in "${RULE_TARGETS[@]}"; do
    if [ ! -d "$(dirname "$t")" ]; then
      echo "  skip $t (agent not installed)"
    elif [ -e "$t" ] && [ ! -L "$t" ]; then
      echo "  skip $t: a real file is already there — merge it into $src, move it aside, re-run"
    else
      ln -sfn "$src" "$t"
      echo "  linked $t -> $src"
      linked=$((linked+1))
    fi
  done
  echo "Linked your rules into $linked agent(s). Edit $src once; every linked agent reads it."
}

cmd_uninstall() {
  local d l removed=0
  for d in "$CLAUDE_HOME/skills" "$HOME/.agents/skills" "$QWEN_HOME/skills" "$@"; do
    [ -d "$d" ] || continue
    while IFS= read -r l; do rm -f "$l"; removed=$((removed+1)); done < <(our_links "$d")
  done
  for l in "${RULE_TARGETS[@]}"; do
    if [ -L "$l" ]; then
      case "$(readlink "$l")" in "$REPO"/*) rm -f "$l"; removed=$((removed+1));; esac
    fi
  done
  echo "Removed $removed links that pointed into $REPO."
}

cmd_flat() {
  local dest="${1:-${SKILLS_DEST:-}}"
  if [ -z "$dest" ]; then echo "error: provide a destination dir (or set SKILLS_DEST)"; exit 1; fi
  # Regenerate the flat bundle if python is available, so it's current.
  if command -v python3 >/dev/null 2>&1; then python3 "$REPO/scripts/build_bundle.py" >/dev/null; fi
  if [ ! -d "$REPO/bundle" ]; then echo "error: $REPO/bundle not found — run scripts/build_bundle.py"; exit 1; fi
  mkdir -p "$dest"
  cp -f "$REPO"/bundle/*.md "$dest"/
  echo "Copied $(ls "$REPO"/bundle/*.md | wc -l | tr -d ' ') flat skill files into $dest"
  echo "Point your markdown-file skill loader at: $dest"
}

case "${1:-help}" in
  all)       cmd_all;;
  claude)    shift; cmd_claude "$@";;
  agents)    shift; cmd_agents "$@";;
  qwen)      shift; cmd_qwen "$@";;
  link)      shift; cmd_link "$@";;
  rules)     shift; cmd_rules "$@";;
  uninstall) shift; cmd_uninstall "$@";;
  flat)      shift; cmd_flat "$@";;
  list)      cmd_list;;
  help|--help|-h) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0";;
  *) echo "unknown command: $1"; echo "run: ./install.sh help"; exit 1;;
esac
