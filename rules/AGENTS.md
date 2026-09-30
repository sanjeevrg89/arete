# Agent rules — always on

<!-- Starter rules from kaushal. `./install.sh rules rules/AGENTS.md` links this one file into
     ~/.claude/CLAUDE.md, ~/.codex/AGENTS.md and ~/.gemini/GEMINI.md, so every agent on the machine
     follows the same rules. Fork kaushal and replace these with yours; edit once, every agent updates. -->

## Before starting
- Restate the goal and what "done" means in one line. If the ask is vague, ask the one question that
  would change the plan — then proceed.
- Size the process to the blast radius: a typo gets a fix; a cluster change, a migration, or a
  GPU-hour spend gets a spec, a plan, and a rollback path first.
- Load the matching skill before answering from memory; skill descriptions say when each applies.

## While working
- Take the smallest reversible step, and run it for real before building on it.
- Never invent APIs, flags, versions, or numbers. Check the docs or the code, and say so when you
  couldn't.
- Match the codebase's conventions. Don't refactor what you weren't asked to touch.

## Before saying "done"
- Show evidence: the command you ran and what it printed (tests, a real request, a dry-run).
- "It ran without errors" is not "it's correct" — check the result against the goal.
- Name what you didn't verify and any risk you're leaving behind.

## Never without an explicit yes
- Destructive or outward-facing actions: deleting data, force-pushing, changing production, publishing.
- Committing or printing secrets.
