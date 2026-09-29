# Contributing

Thanks for improving this skill library. It's a cross-agent set of engineering reference skills,
consumable by Claude Code (`SKILL.md`), Gemini CLI (`GEMINI.md`), and AGENTS.md-compatible agents
(`AGENTS.md`). Everything here is generic, open content — usable by anyone.

## Add or edit a skill

1. **Read [`SKILL-AUTHORING-SPEC.md`](SKILL-AUTHORING-SPEC.md).** It defines the required files,
   frontmatter, router rules, voice, and quality bar. Mirror the `skills/go-best-practices/` exemplar.
2. **Create `skills/<slug>/`** with the exact kebab-case slug. It must contain:
   - `SKILL.md` — YAML frontmatter (`name` matching the directory, a one-line router `description`
     of <= 250 chars) + a `## Scope and triggers` section + a concise body.
   - `<slug>-guide.md` — the deep reference (the single source of truth).
   - `AGENTS.md` — condensed always-on summary that points to the guide.
   - `GEMINI.md` — imports the guide via `@./<slug>-guide.md`.
   - `examples.md` — worked examples (recommended).
3. **Add a row** to [`REGISTRY.md`](REGISTRY.md) and slot the skill into the cluster map.
4. **Add a routing case** to [`tests/skill-routing-checklist.md`](tests/skill-routing-checklist.md):
   one discriminating prompt a user would actually type (CI fails if a skill has none).
5. **Cross-link** related skills with `[[slug]]`.

## Content rules

- **Accuracy over completeness.** Never fabricate APIs, flags, version numbers, citations, or
  benchmark figures. Flag fast-moving items with "verify against current docs."
- **Generic and vendor-neutral.** Name public products/projects only as technically needed. Do not
  frame content as private/organization-specific.
- Write as a top-tier practitioner: dense, concrete, opinionated, correct. Signal over volume.
- Cite real, authoritative sources (project docs, papers, standards).

## Before you open a PR

```bash
pip install pyyaml                     # strict frontmatter parsing, same as CI
python scripts/validate.py             # must report 0 errors
python scripts/routing_eval.py --lint  # every skill has a routing case
python scripts/build_bundle.py         # regenerate the flat bundle/ files

# Changed a description? Check it still routes (any agent CLI that reads stdin works):
AGENT_CMD='claude -p --model haiku --disable-slash-commands --tools "" --no-session-persistence' \
  python scripts/routing_eval.py --only <slug>
```

`validate.py` parses frontmatter strictly (as agents do), enforces the router rules, and checks unique
names, required files, resolvable `[[cross-links]]`, and the content rules above. CI
(`.github/workflows/ci.yml`) also runs the official Agent Skills validator (`skills-ref`) on every
first-party skill, lints the routing cases, and fails if `bundle/` is stale — **errors fail the
build; warnings don't.** Commit the regenerated `bundle/` alongside your changes.

## Commit conventions

- Keep commits focused; describe what changed and why.
- Do not add tool-generated co-author trailers.
