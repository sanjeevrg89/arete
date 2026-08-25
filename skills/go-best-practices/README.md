# go-best-practices

A portable, distinguished-engineer Go ruleset that plugs into multiple AI coding agents from a single
source of truth.

## What's here

| File | Role | Consumed by |
|------|------|-------------|
| `go-guidelines.md` | **The full ruleset — single source of truth.** Edit this. | everything (read directly or imported) |
| `examples.md` | Before/after Go snippets for the highest-impact rules | agents + humans |
| `SKILL.md` | Claude Code skill entry point (YAML frontmatter + summary; defers to the guide) | Claude Code |
| `AGENTS.md` | Always-on summary + pointer to the guide; the emerging cross-tool standard | Codex, Cursor, Jules, Amp, others reading `AGENTS.md` |
| `GEMINI.md` | Gemini CLI entry point; `@`-imports the full guide | Gemini CLI |
| `Makefile` | Quality-gate starter — `make check` runs fmt/vet/lint/race-tests/vuln | any Go module |
| `.golangci.yml` | golangci-lint v2 starter config, curated to match the guide | golangci-lint / `make lint` |
| `README.md` | This file | humans |

**Maintenance:** change `go-guidelines.md` only. The three entry files are thin and rarely need edits.

---

## Install per tool

### Claude Code (skill)
Copy or symlink the whole folder into a skills directory:

```bash
# Personal (all projects):
mkdir -p ~/.claude/skills
ln -s ~/Documents/arete/go-best-practices ~/.claude/skills/go-best-practices

# Or per-project:
mkdir -p .claude/skills
cp -R ~/Documents/arete/go-best-practices .claude/skills/
```

Claude Code auto-discovers it via `SKILL.md` frontmatter and loads it when you work on Go.
Verify with `/skills` (or by asking it to follow the go-best-practices skill).

### Codex / Cursor / Amp / Jules (AGENTS.md)
These tools read `AGENTS.md` from the project root (and merge nested ones). Easiest:

```bash
# From your repo root — symlink the guide + an AGENTS.md that points at it:
ln -s ~/Documents/arete/go-best-practices/go-guidelines.md ./go-guidelines.md
ln -s ~/Documents/arete/go-best-practices/AGENTS.md ./AGENTS.md
```

If you already have an `AGENTS.md`, instead append one line to it:

```md
For Go code, follow the standards in ./go-guidelines.md (read it before writing or reviewing Go).
```

Then drop `go-guidelines.md` into the repo (copy or symlink). Codex picks up `AGENTS.md` automatically.

### Gemini CLI (GEMINI.md)
Gemini CLI reads `GEMINI.md` and supports `@`-file imports.

```bash
# Project-level:
cp ~/Documents/arete/go-best-practices/GEMINI.md ./GEMINI.md
cp ~/Documents/arete/go-best-practices/go-guidelines.md ./go-guidelines.md
# (GEMINI.md imports ./go-guidelines.md)

# Or global, in your home Gemini config dir:
mkdir -p ~/.gemini
cp ~/Documents/arete/go-best-practices/GEMINI.md ~/.gemini/GEMINI.md
cp ~/Documents/arete/go-best-practices/go-guidelines.md ~/.gemini/go-guidelines.md
```

### Project quality gate (Makefile + .golangci.yml)
Copy both into your Go module root so the agent (and humans/CI) share one definition of done:

```bash
cp ~/Documents/arete/go-best-practices/Makefile      ./Makefile
cp ~/Documents/arete/go-best-practices/.golangci.yml ./.golangci.yml
make tools   # installs golangci-lint, goimports, govulncheck
make check   # fmt-check + vet + lint + race tests + govulncheck
```

Before committing the config, set two project-specific values:
- `.golangci.yml` → `formatters.settings.goimports.local-prefixes`: your module path.
- These are starters — relax/tighten `linters.enable` as the team agrees; adopt incrementally on
  large existing codebases (lint a subdir first, or run `golangci-lint run --new-from-rev=origin/main`).

Requires **golangci-lint v2.x**. On v1.x, see the migration note at the bottom of `.golangci.yml`.

### Any other agent (Continue, Aider, Windsurf, Zed, etc.)
Point the tool's rules/instructions file at `go-guidelines.md`, or paste its contents into whatever
"custom instructions / rules" mechanism the tool exposes. The guide is plain Markdown with no
tool-specific syntax, so it drops in anywhere.

---

## Design notes

- **One source of truth.** All entry files defer to `go-guidelines.md` so guidance never drifts
  between tools. Update once, every agent benefits.
- **Symlink > copy** when you want repos to track the canonical guide automatically; **copy** when you
  want a repo pinned to a reviewed snapshot.
- The guidelines are defaults, not dogma. They tell agents to match an existing codebase's conventions
  on stylistic conflicts while still enforcing correctness, error-handling, and concurrency discipline.
