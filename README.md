# skills

[![validate-skills](https://github.com/sanjeevrg89/skills/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjeevrg89/skills/actions/workflows/ci.yml)

A cross-agent skill library: distinguished-engineer reference skills for Go, Kubernetes (use/control/
operator/internals), and the ML-infra stack (Kueue, JobSet/LWS, ML & serving & training frameworks,
Slurm/HPC, GKE, autoscaling). Each skill is **one self-contained directory** that works across
Claude Code, Gemini CLI, and AGENTS.md-compatible agents/IDEs from a single source of truth.

See **[REGISTRY.md](REGISTRY.md)** for the full index and **[SKILL-AUTHORING-SPEC.md](SKILL-AUTHORING-SPEC.md)**
for how to add one (see also [CONTRIBUTING.md](CONTRIBUTING.md)).

## Get 10–100x from it → [USAGE.md](USAGE.md)

Installing the skills is 10% of the value. The leverage is **encoded senior judgment that ports across
every agent (Claude Code, Gemini CLI, Codex) and gets sharper from your usage.** The operating manual —
install everywhere, drive every agent like a senior ([`senior-operating-modes`](senior-operating-modes/)),
and run a loop that improves the skills from feedback ([`skill-self-improvement`](skill-self-improvement/))
— is in **[USAGE.md](USAGE.md)**.

## Architecture

How the repo works: **one source of truth per skill** → consumed by many agents, gated by CI, and
improved by a feedback loop. The diagram below renders inline on GitHub; an **editable
[Excalidraw version is here](https://excalidraw.com/#json=GfOHF9IXI3Njzr2PbvMTF,RtD7am1R9kr-tiTLmY3vFw)**
(open it, then *Export → PNG* into `docs/` if you want to embed the hand-drawn version inline).

```mermaid
flowchart TB
  A([Author or agent writes a skill]) -->|writes| S
  CI["CI gate: validate.py - functional_test.py - VALIDATION 5-layer"] -->|must pass| S
  subgraph S["Source of truth: one directory per skill"]
    direction TB
    S1["SKILL.md -> Claude Code router"]
    S2["slug-guide.md -> deep reference"]
    S3["AGENTS.md -> Codex / Cursor"]
    S4["GEMINI.md -> Gemini import"]
    S5["examples.md -> worked examples"]
  end
  S -->|SKILL.md, on-demand| CC["Claude Code"]
  S -->|AGENTS.md, always-on| CX["Codex / Cursor / IDEs"]
  S -->|GEMINI.md @import| GM["Gemini CLI"]
  S -->|build_bundle.py| FL["Flat-md loaders: skills/name.md"]
  subgraph LOOP["Self-improvement loop: the library improves itself"]
    direction LR
    W["Skill applied to real work"] --> FB["feedback/log.jsonl (signal)"]
    FB --> RANK["skill_feedback.py (rank)"]
    RANK --> RV["Reviser cron: diff -> PR"]
    RV --> VG["Verify gate + human merge"]
  end
  CC -.usage.-> W
  VG -->|merged PR updates the skill| S
```

Install/update wires the source into each agent (`install.sh` symlinks for Claude Code; flat copies for
markdown loaders). The loop is documented in [`skill-self-improvement`](skill-self-improvement/).

## What each skill directory contains
| File | Consumed by | Role |
|------|-------------|------|
| `SKILL.md` | Claude Code | Frontmatter (`name` + router `description`) + essentials; loads on demand. |
| `<slug>-guide.md` | everything | The full deep reference (single source of truth). |
| `AGENTS.md` | Codex / Cursor / agentic IDEs (AGENTS.md standard) | Always-on condensed summary + pointer to the guide. |
| `GEMINI.md` | Gemini CLI | Imports the guide via `@./<slug>-guide.md`. |
| `examples.md` | everything | Canonical worked examples / manifests (most skills). |

## Install

### Quick start (clone + `install.sh`)
The repo is private — clone with your GitHub auth (SSH key or `gh auth login` / a PAT over HTTPS):
```bash
git clone git@github.com:sanjeevrg89/skills.git
cd skills

./install.sh list                 # list the skills
./install.sh claude               # symlink all skills into ~/.claude/skills (Claude Code)
./install.sh flat ~/.gemini/skills  # copy the flat <name>.md bundle for a markdown-file loader
./install.sh help
```
`install.sh flat <dest>` regenerates the flat bundle and copies it to whatever directory your loader
scans (set `<dest>` to your tool's skills path). Manual equivalents below.

### Updating
One command — pulls latest and refreshes your install:
```bash
cd skills
./update.sh                 # git pull + refresh Claude Code install (~/.claude/skills)
./update.sh ~/.gemini/<codename>/skills   # ...also refresh a flat-bundle copy at that path
```
Under the hood: **symlink (Claude) installs** just need a re-link to pick up newly added skills;
**flat installs are copies** and get re-copied. `update.sh` does the right thing for both (manual
equivalent: `git pull` then re-run `./install.sh ...`).

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

## Testing skill routing
[`tests/skill-routing-checklist.md`](tests/skill-routing-checklist.md) has one discriminating prompt
per skill (46) with the skill it should route to — use it to confirm each skill installs and routes
correctly in Claude Code (`/skills` should list every skill).

**Functional checks** (`tests/functional/checks.json` + `scripts/functional_test.py`) go beyond routing:
each is "given this prompt, the output must satisfy X" (regex assertions). CI lints the specs
(`functional_test.py --lint`); to actually run them against an agent: `AGENT_CMD='claude -p' python
scripts/functional_test.py`.

**Validating a skill for real** — structure/routing/functional checks are necessary but not sufficient;
real confidence comes from applying a skill to an actual repo/task and judging the output against an
expert bar. See [`tests/VALIDATION.md`](tests/VALIDATION.md) for the 5-layer procedure (incl. A/B vs a
no-skill baseline and a content-accuracy pass) and record results in
[`tests/validation-log.md`](tests/validation-log.md). Green CI ≠ validated.

## Validation / CI
`scripts/validate.py` (stdlib only) checks every skill: valid `SKILL.md` frontmatter, `name` matches
the directory and is unique, a router-grade `description`, the required files (`SKILL.md` / guide /
`AGENTS.md` / `GEMINI.md`), resolvable `[[cross-links]]`, and no vendor-locked framing or co-author
trailers. Run it locally with `python scripts/validate.py`; `.github/workflows/ci.yml` runs it on every
push and PR. Errors fail the build; warnings don't.

## Self-improvement loop (the library improving itself)

The library doesn't just get edited by hand — it has a loop that improves skills from real-world
feedback. The [`skill-self-improvement`](skill-self-improvement/) skill is the pattern; the repo ships
the substrate:

- **Signal** — append a record to [`feedback/log.jsonl`](feedback/README.md) when a skill underperforms
  (what skill, what was wrong, the correct answer). Failing functional checks are signal too.
- **Rank** — `python scripts/skill_feedback.py` lists the skills with the most negative signal (the
  improvement candidates); `--lint` validates the records in CI.
- **Reviser** — [`.github/workflows/skill-self-improvement.yml`](.github/workflows/skill-self-improvement.yml)
  runs on a weekly cron. The committed job is **dry-run** (lints the signal + prints candidates, no
  secrets, writes nothing); the real reviser step — an agent that diffs an underperforming skill's guide,
  adds a regression check, and **opens a PR** — is documented there, gated, and **never auto-merges**.
- **Gate & distill** — the PR passes the same `validate.py` + functional checks (`ci.yml`) plus review,
  then the lesson is distilled into the guide's anti-patterns + a `must_*` check so it can't regress.

This is `tests/VALIDATION.md` Layer 5 step 4 ("feed failures back"), automated.

## Design notes
- **On-demand loading is what makes a large library viable.** Claude Code discovers skills by their
  `description` and loads only what's relevant — never all guides at once. Keep `AGENTS.md`/`GEMINI.md`
  small (they're always-on); the depth lives in the guide.
- **One source of truth per skill:** the `<slug>-guide.md`. The entry files defer to it.
- The ecosystem (K8s, GKE, ML/serving/training frameworks) moves fast — guides flag version-sensitive
  details and tell the reader to verify against current upstream docs.
