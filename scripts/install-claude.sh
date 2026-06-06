#!/usr/bin/env bash
# Symlink every skill into the Claude Code personal skills dir (on-demand discovery).
set -euo pipefail
src="$(cd "$(dirname "$0")/.." && pwd)"
dest="${HOME}/.claude/skills"
mkdir -p "$dest"
for d in "$src"/*/; do
  [ -f "${d}SKILL.md" ] || continue
  ln -sfn "$d" "$dest/$(basename "$d")"
  echo "linked $(basename "$d")"
done
