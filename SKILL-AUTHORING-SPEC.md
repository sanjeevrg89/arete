# Skill Authoring Spec (read before writing any skill)

This is a cross-agent skill library, open for anyone to use. Each skill is a self-contained directory consumed by
coding assistants (Claude Code via `SKILL.md`, Codex/Cursor/etc. via `AGENTS.md`, Gemini via imports).
Author every skill to this exact standard so the library is consistent.

The reference exemplar is `skills/go-best-practices/` (see its `SKILL.md`, `AGENTS.md`,
`go-guidelines.md`). Mirror its structure and voice.

## Files every skill directory MUST contain

1. **`SKILL.md`** — Claude Code entry point. YAML frontmatter + concise body.
   ```
   ---
   name: <exact-kebab-slug-matching-the-directory-name>
   description: <THE ROUTER, one line, <= 250 chars — see "The router" below.>
   ---

   # <Skill Title>

   <1–2 line statement of the expertise and the bar (e.g. "Apply the judgment of an engineer who has
   run this in production at scale for years.")>

   ## Scope and triggers
   <The long form the router can't hold: everything the skill covers, every trigger term and symptom,
   and where the boundary with sibling skills sits (`[[other-slug]]`). Agents read it after loading.>

   ## How to use this skill
   1. Read `<slug>-guide.md` in this directory — the full reference. Apply it to the task.
   2. <any examples.md / sub-references>
   3. Match the surrounding codebase/cluster conventions; apply correctness/safety rules regardless.

   ## Essentials (full detail in `<slug>-guide.md`)
   - <8–15 of the highest-value, most load-bearing bullets a top engineer would insist on>

   ## Related skills
   - `[[other-slug]]` — when to reach for it instead / in addition.
   ```

2. **`<slug>-guide.md`** — THE deep reference. This is the meat. **250–450 lines.** Sectioned with
   `##` headings. Must include, adapted to the topic:
   - **Mental model / architecture** — how the thing really works, not a feature list.
   - **Core concepts** — the objects/APIs/abstractions, with precise definitions.
   - **Hands-on** — real, correct artifacts: `kubectl`/YAML/Go/Python/CLI. No pseudo-code where real
     code is possible. Show the canonical idiom.
   - **Best practices** — opinionated, with the *why*.
   - **Anti-patterns / gotchas** — the traps that bite people in production. Be specific.
   - **Performance / scale** — where relevant (throughput, latency, memory, large clusters, big models).
   - **Troubleshooting** — concrete symptoms → diagnosis → fix.
   - **Security / multi-tenancy** — where relevant.
   - **Version awareness** — note that the ecosystem moves fast (it is 2026); flag where APIs/versions
     matter and tell the reader to verify current docs. Don't invent version numbers you're unsure of.
   - **Rationalizations & rebuttals** — a short list of the excuses an engineer or agent uses to skip
     the right thing ("it's just a quick fix, skip the test"), each with a one-line rebuttal. This is
     what makes a guide *agent-actionable* rather than merely informative. Include it where the skill
     has real practices/decisions; **omit for pure-reference/mechanism topics** where it would be
     artificial (don't force it).
   - **Red flags** — a concise "stop and reconsider" list: signals that the current approach is wrong.
   - **Verification gate (definition of done)** — the explicit checklist of what must be true/tested
     before the work counts as complete (commands to run, properties to confirm, evidence to show).
   - **Canonical references** — authoritative links (project docs, KEPs, papers, source). Real URLs only.

   **Process / workflow skills** (lifecycle skills like spec-driven dev, TDD, review, debugging,
   shipping) are structured differently — lead with **Overview**, **When to use**, and **The process**
   (numbered steps with explicit checkpoints/gates), then the Rationalizations / Red flags /
   Verification sections above. They encode *how to work*, not *what to know*.

3. **`AGENTS.md`** — cross-tool always-on summary. Short. Header pointing to `<slug>-guide.md` as the
   authoritative source, then a condensed always-on checklist of the highest-value rules. Keep it small
   (it is loaded into context every turn for tools that use it) — point to the guide for depth.

4. **`examples.md`** (optional but encouraged where patterns help) — before/after or canonical
   worked examples (YAML/code) the agent can imitate.

## The router (`description`)

The description is the only thing an agent sees before deciding whether to load a skill, and every
installed skill's description shares one listing budget. Claude Code cuts the skill listing off at a
fixed character budget (30,000 in the session this rule was measured in) and lists the rest by name
only; Codex caps its list at 2% of the context window (8,000 characters when unknown) and shortens
descriptions to fit. A 1,000-character router doesn't make a skill easier to find — it pushes other
skills, including your own, out of view. And a description that isn't valid YAML is worse than
short: Claude Code loads the skill with no description, and `npx skills add` skips it entirely.

Rules (enforced by `scripts/validate.py` and CI):

- **One line, <= 250 characters.** What it is, then the 5–10 most distinctive trigger terms (tools,
  APIs, file types, error symptoms), then "Use when/for …". Trigger terms first: truncation eats the
  end.
- **Plain YAML that every parser accepts.** No `: ` or ` #` inside an unquoted value; if you need
  one, double-quote the whole value. `name` and `description` only — first-party skills stay
  spec-pure ([Agent Skills spec](https://agentskills.io/specification)) so every agent can load them.
- **Name the sibling only when routing is genuinely ambiguous** ("…; profiling is
  gpu-performance-engineering"). Full boundaries belong in `## Scope and triggers`.
- **Every skill has a routing case** in [`tests/skill-routing-checklist.md`](tests/skill-routing-checklist.md):
  one discriminating prompt a user would actually type, plus acceptable neighbors. After editing a
  description, re-run it: `python scripts/routing_eval.py --only <slug>` (needs `AGENT_CMD`; see
  [`tests/VALIDATION.md`](tests/VALIDATION.md)).

## Quality bar

- Write as a **top-5-in-the-world practitioner with ~10 years of production experience** in the topic.
  Dense, concrete, opinionated, correct. Signal over volume.
- **Accuracy over completeness.** If you are unsure whether a detail is current/correct, say so or omit
  it — never fabricate API fields, flags, version numbers, or benchmark figures.
- Prefer the canonical/idiomatic approach; call out common-but-wrong patterns explicitly.
- Real commands and manifests must be runnable-in-spirit and correct (right apiVersion/kind/fields).
- Cross-link related skills by slug using `[[slug]]` so the library forms a graph.

## Slugs for cross-links

Use the exact directory names listed in [`REGISTRY.md`](REGISTRY.md). `scripts/validate.py` warns on
any `[[slug]]` that doesn't resolve to a skill directory.

## Voice / formatting

- Markdown. `##`/`###` headings, tight bullets, fenced code with language tags.
- No marketing language, no "in today's fast-paced world" filler. Engineer-to-engineer.
- Tables for comparisons. Keep line length readable (~100 cols, wrap prose).
