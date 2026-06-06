# skills

A cross-agent skill library: distinguished-engineer reference skills for Go, Kubernetes (use/control/
operator/internals), and the ML-infra stack (Kueue, JobSet/LWS, ML & serving & training frameworks,
Slurm/HPC, GKE, autoscaling). Each skill is **one self-contained directory** that works across
Claude Code, Gemini CLI, and AGENTS.md-compatible agents/IDEs from a single source of truth.

See **[REGISTRY.md](REGISTRY.md)** for the full index and **[SKILL-AUTHORING-SPEC.md](SKILL-AUTHORING-SPEC.md)**
for how to add one.

## What each skill directory contains
| File | Consumed by | Role |
|------|-------------|------|
| `SKILL.md` | Claude Code | Frontmatter (`name` + router `description`) + essentials; loads on demand. |
| `<slug>-guide.md` | everything | The full deep reference (single source of truth). |
| `AGENTS.md` | Codex / Cursor / agentic IDEs (AGENTS.md standard) | Always-on condensed summary + pointer to the guide. |
| `GEMINI.md` | Gemini CLI | Imports the guide via `@./<slug>-guide.md`. |
| `examples.md` | everything | Canonical worked examples / manifests (most skills). |

## Install

### Claude Code (skills load on demand — all of them can coexist)
```bash
mkdir -p ~/.claude/skills
for d in ~/Documents/skills/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$d" ~/.claude/skills/"$(basename "$d")"
done
```

### Gemini CLI (per project or global)
Gemini loads one `GEMINI.md` from the working dir / config dir. Copy the skill(s) relevant to a repo:
```bash
cp -R ~/Documents/skills/gke-master ./            # then GEMINI.md imports the guide
# or globally:  cp -R ~/Documents/skills/<skill> ~/.gemini/
```

### AGENTS.md-compatible agents / agentic IDEs (e.g. Antigravity, Codex, Cursor)
Drop the relevant skill's `AGENTS.md` + guide into the repo root, or reference the guide from your
existing `AGENTS.md`. The content is vendor-neutral Markdown, so it ports to any "custom rules" surface.

## Two consumable layouts
This repo ships skills in **two forms** so it works with any loader:

1. **Directory-per-skill** (source of truth) — `<name>/SKILL.md` + `<name>/<name>-guide.md` +
   `AGENTS.md` + `GEMINI.md` + `examples.md`. Best for Claude Code (`SKILL.md` discovery) and tools
   that read a skill folder.
2. **Flat self-contained bundle** — `skills/<name>.md`, one file per skill with the frontmatter, the
   **full guide, and examples inlined**. Best for loaders that read **markdown files from a `skills/`
   directory** (the deep content travels in the file, not a separate guide). Regenerate with
   `python scripts/build_bundle.py` after editing any skill.

Point a flat-markdown loader at the `skills/` directory; point a folder-based/Claude loader at the
repo root (skill directories).

## Validation / CI
`scripts/validate.py` (stdlib only) checks every skill: valid `SKILL.md` frontmatter, `name` matches
the directory and is unique, a router-grade `description`, the required files (`SKILL.md` / guide /
`AGENTS.md` / `GEMINI.md`), resolvable `[[cross-links]]`, and no vendor-locked framing or co-author
trailers. Run it locally with `python scripts/validate.py`; `.github/workflows/ci.yml` runs it on every
push and PR. Errors fail the build; warnings don't.

## Design notes
- **On-demand loading is what makes a large library viable.** Claude Code discovers skills by their
  `description` and loads only what's relevant — never all guides at once. Keep `AGENTS.md`/`GEMINI.md`
  small (they're always-on); the depth lives in the guide.
- **One source of truth per skill:** the `<slug>-guide.md`. The entry files defer to it.
- The ecosystem (K8s, GKE, ML/serving/training frameworks) moves fast — guides flag version-sensitive
  details and tell the reader to verify against current upstream docs.
