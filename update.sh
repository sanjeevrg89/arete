#!/usr/bin/env bash
# Update the skill library and refresh whatever you installed.
#
# Usage:
#   ./update.sh                 git pull + refresh every symlink install found (Claude Code's
#                               ~/.claude/skills; ~/.agents/skills if arete is linked there)
#   ./update.sh <flat-dest>     ...also refresh the flat-bundle copy at <flat-dest>
#   SKILLS_DEST=<dir> ./update.sh   same as passing <flat-dest>
#
# Symlink installs only need a re-link to pick up NEW skills (and prune removed ones); flat installs
# are copies and must be re-copied. This script does the right thing for both.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo "==> git pull"
git pull --ff-only

did_something=0

claude_skills="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills"
if [ -d "$claude_skills" ]; then
  echo "==> refreshing Claude Code install ($claude_skills)"
  "$REPO/install.sh" claude
  did_something=1
fi

# Only touch ~/.agents/skills if arete was linked there before (./install.sh agents|all).
if [ -n "$(find "$HOME/.agents/skills" -maxdepth 1 -type l -lname "$REPO/*" 2>/dev/null | head -1)" ]; then
  echo "==> refreshing cross-agent install (~/.agents/skills: Codex CLI, Gemini CLI)"
  "$REPO/install.sh" agents
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
  echo "Run ./install.sh all  (Claude Code + Codex/Gemini CLI)  or  ./install.sh flat <dest>  (flat loader)."
fi

echo "==> done: $(git log --oneline -1)"
