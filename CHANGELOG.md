# Changelog

All notable changes to arete are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: semver.

## [Unreleased]

## [1.1.0] — 2026-09-29

Every skill is now visible to, and routable by, every agent. Measured on Claude Code's real skill
listing (30,000-character budget, arete installed alone): skills with a usable description 29/70 →
70/70; routing top-1 on 59 prompts 75% → 100% (Haiku) and 85% → 98% (Sonnet); `npx skills add`
installs 84/84 skills (was 59/84).

### Fixed
- **25 of 59 first-party `SKILL.md` files had invalid YAML frontmatter** — an unquoted `: ` inside a
  multi-line `description`. Claude Code loaded them with no description (at best the page title), and
  `npx skills add` skipped them outright, so Codex, Cursor, and Gemini CLI users never received them.
  The old `validate.py` used a lenient hand-rolled parser and passed them.
- **Descriptions totaled 60,424 characters** — double Claude Code's whole listing budget, which every
  installed skill shares; past it, skills are listed by name only and can't route. 11 more descriptions
  broke the Agent Skills spec's 1,024-character limit.
- `install.sh` nested a symlink *inside* an existing real directory with a skill's name (`ln -sfn`
  onto a directory); it now skips that directory and prunes links to removed skills.
- Removed a committed iCloud duplicate, `scripts/validate 2.py`, and ignored the pattern.

### Changed
- **Every first-party description rewritten as a one-line router**, ≤ 250 characters, trigger terms
  first (60,424 → 13,427 characters in total). The original long text is kept, word for word, as a
  new `## Scope and triggers` section in each `SKILL.md`.
- **`validate.py` parses frontmatter strictly** (PyYAML; a heuristic fallback without it) and enforces
  spec-legal names, the 250-character router limit, and the spec limit for vendored skills; it reports
  the total router budget and rejects Finder/iCloud duplicate files.
- **CI** installs PyYAML, runs the official Agent Skills validator (`skills-ref`) on every first-party
  skill, lints routing cases, and fails when `bundle/` is stale.
- `SKILL-AUTHORING-SPEC.md` documents the router rules and why they exist; its stale 14-skill slug list
  now points to `REGISTRY.md`.
- README rewritten around the problems the library fixes (docs-knowledge vs production scar tissue),
  with real excerpts from `kubernetes-expert` and `gpu-performance-engineering` as evidence; GitHub
  repo description replaced with a single clear hook.

### Added
- **`scripts/routing_eval.py`** — replays one discriminating prompt per skill against the catalog an
  agent sees and scores top-1 routing (`AGENT_CMD`, `--jobs`, `--only`, `--listing` for a listing
  captured from a real session; `--lint` in CI). 13 new routing cases, so all 59 first-party skills
  have one.
- **`./install.sh all | agents | rules | uninstall`** — link every skill into Claude Code *and*
  `~/.agents/skills` (Codex CLI; Gemini CLI via its alias); link one rules file into
  `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, and `~/.gemini/GEMINI.md` (never overwriting a real
  file); remove only what arete linked. `update.sh` refreshes `~/.agents/skills` too.
- **`rules/AGENTS.md`** — a short starter rules file for the one-rules-file-every-agent pattern
  (after [steipete/agent-scripts](https://github.com/steipete/agent-scripts)). Replace it with yours.

## [1.0.0] — 2026-08-24

The "world-class packaging" release: same distinguished-bar content, now installable everywhere.

### Added
- **One-command installs on every channel:**
  - Claude Code plugin (`.claude-plugin/plugin.json` + `marketplace.json`): `/plugin marketplace add sanjeevrg89/arete` → `/plugin install arete@arete`.
  - skills.sh / `npx skills add sanjeevrg89/arete` compatibility — all skill directories now live under `skills/` where the installer's discovery walk finds them.
- **25 vendored process skills** from [mattpocock/skills](https://github.com/mattpocock/skills) (MIT, commit `6654f6b6`) under `skills/vendored/mattpocock/`, with provenance + license in [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). Process (grilling → spec → TDD → review) now ships alongside domain depth.
- **Routing disambiguation:** the 4 first-party skills overlapping vendored ones (`test-driven-development`, `code-review-discipline`, `spec-driven-development`, `verification-and-debugging`) now scope themselves in their router descriptions so both can coexist.
- `CHANGELOG.md` (this file) and this release marks the repo public-first: README rewritten install-first.

### Changed
- **Layout:** skill sources moved from repo root to `skills/<name>/`; the generated flat bundle moved from `skills/*.md` to `bundle/<name>.md`. `validate.py`, `build_bundle.py`, `install.sh`, and `functional_test.py` all updated; run `./install.sh claude` once after pulling to re-link symlinks.
- **README** rewritten: install channels up front, layout map, attribution, stale claims removed.
- `REGISTRY.md` links updated for the new layout + a full index of the vendored skills.

### Fixed
- `kubernetes-expert-guide.md`: native sidecars described as "GA in 1.29" — they are beta/default-on since 1.29; wording corrected.
- `kubernetes-expert-guide.md`: `kubectl apply` claimed to use server-side apply by default — it is client-side unless `--server-side`; now explained correctly with when to opt in.
- `kubernetes-expert-guide.md`: deduplicated ~35 lines of repeated rules (Red flags merged into Anti-patterns; Checklist merged into the Verification gate with its command block).

[Unreleased]: https://github.com/sanjeevrg89/arete/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/sanjeevrg89/arete/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/sanjeevrg89/arete/releases/tag/v1.0.0
