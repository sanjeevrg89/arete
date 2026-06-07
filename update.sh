#!/usr/bin/env bash
# Update the skill library and refresh whatever you installed.
#
# Usage:
#   ./update.sh                 git pull + refresh the Claude Code install (if ~/.claude/skills exists)
#   ./update.sh <flat-dest>     ...also refresh the flat-bundle copy at <flat-dest>
#   SKILLS_DEST=<dir> ./update.sh   same as passing <flat-dest>
#
# Symlink (Claude) installs only need a re-link to pick up NEW skills; flat installs are copies and
# must be re-copied. This script does the right thing for both.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo "==> git pull"
git pull --ff-only

did_something=0

if [ -d "$HOME/.claude/skills" ]; then
  echo "==> refreshing Claude Code install (~/.claude/skills)"
  "$REPO/install.sh" claude
  did_something=1
fi

dest="${1:-${SKILLS_DEST:-}}"
if [ -n "$dest" ]; then
  echo "==> refreshing flat bundle at $dest"
  "$REPO/install.sh" flat "$dest"
  did_something=1
fi

if [ "$did_something" -eq 0 ]; then
  echo "Pulled latest, but found no install to refresh."
  echo "Run ./install.sh claude  (Claude Code)  or  ./install.sh flat <dest>  (flat loader)."
fi

echo "==> done: $(git log --oneline -1)"
