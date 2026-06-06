#!/usr/bin/env bash
# Install the skill library into a coding-agent skills directory.
#
# Usage:
#   ./install.sh claude [dest]   Symlink each skill directory into a Claude Code skills dir
#                                (default: ~/.claude/skills). On-demand discovery via SKILL.md.
#   ./install.sh flat <dest>     Copy the flat self-contained skills/<name>.md bundle into <dest>,
#                                for loaders that read markdown files from a skills/ directory
#                                (e.g. a Gemini-style loader). Set dest to your loader's skills path.
#   ./install.sh list            List available skills.
#   ./install.sh help            Show this help.
#
# Env:
#   SKILLS_DEST   default destination for `flat` if <dest> omitted.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"

skill_dirs() {
  for d in "$REPO"/*/; do
    name="$(basename "$d")"
    case "$name" in scripts|skills|.git|.github) continue;; esac
    [ -f "$d/SKILL.md" ] && echo "$name"
  done
}

cmd_list() { skill_dirs; echo; echo "$(skill_dirs | wc -l | tr -d ' ') skills."; }

cmd_claude() {
  local dest="${1:-$HOME/.claude/skills}"
  mkdir -p "$dest"
  local n=0
  while IFS= read -r name; do
    ln -sfn "$REPO/$name" "$dest/$name"; n=$((n+1))
  done < <(skill_dirs)
  echo "Linked $n skills into $dest"
  echo "Claude Code discovers them on demand via each SKILL.md description."
}

cmd_flat() {
  local dest="${1:-${SKILLS_DEST:-}}"
  if [ -z "$dest" ]; then echo "error: provide a destination dir (or set SKILLS_DEST)"; exit 1; fi
  # Regenerate the flat bundle if python is available, so it's current.
  if command -v python3 >/dev/null 2>&1; then python3 "$REPO/scripts/build_bundle.py" >/dev/null; fi
  if [ ! -d "$REPO/skills" ]; then echo "error: $REPO/skills not found"; exit 1; fi
  mkdir -p "$dest"
  cp -f "$REPO"/skills/*.md "$dest"/
  echo "Copied $(ls "$REPO"/skills/*.md | wc -l | tr -d ' ') flat skill files into $dest"
  echo "Point your markdown-file skill loader at: $dest"
}

case "${1:-help}" in
  claude) shift; cmd_claude "$@";;
  flat)   shift; cmd_flat "$@";;
  list)   cmd_list;;
  help|--help|-h) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//';;
  *) echo "unknown command: $1"; echo "run: ./install.sh help"; exit 1;;
esac
