# Changelog

All notable changes to arete are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: semver.

## [Unreleased]

### Changed
- **README rewritten** around the problems the library fixes (docs-knowledge vs production scar
  tissue), with real excerpts from `kubernetes-expert` and `gpu-performance-engineering` as evidence.
- GitHub repo description replaced with a single clear hook.

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

[Unreleased]: https://github.com/sanjeevrg89/arete/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/sanjeevrg89/arete/releases/tag/v1.0.0
