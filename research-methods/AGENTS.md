# AGENTS.md — Research Methods (multi-perspective, fast, self-critiqued)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full method lives in **`research-methods-guide.md`** next to this file; the four copy-paste prompts
> are in **`examples.md`**. This is the always-on summary.

## The method (run in order; keep perspectives independent)

1. **Multi-perspective scan.** Simulate 4–6 genuinely-different experts (default: practitioner,
   academic, skeptic, economist/incentives, historian). For each: core position (2 sentences),
   strongest evidence, and the one thing only they would say. Generate each lens *before* it sees the
   others — independence is the whole point.
2. **Contradiction map.** Where do lenses directly clash (and which has the stronger evidence)? What do
   **all** agree on (likely true)? What did **none** address (the blind spot — often the best finding)?
3. **Synthesis.** A briefing: nuanced one-paragraph summary; findings **ranked by reliability** (note
   which lenses support/challenge each); the non-obvious cross-finding connection; a **specific action**.
4. **Peer-review gate (NEVER skip).** Make the model grade its own briefing: per-finding confidence
   (1–10 + why), weakest link + what would verify it, bias check (did one voice dominate?), a missing
   6th perspective, overall grade + what to fix.

## Hard rules

- **Pick perspectives that actually differ for this topic.** Swap the defaults per domain; identical
  lenses = single-prompt research with extra steps.
- **Always end on the peer-review gate.** The model is confidently wrong and misattributes sources
  (STORM's known weakness). Phase 4 mitigates, not eliminates.
- **Organize reasoning ≠ establish fact.** Verify load-bearing / high-stakes claims against primary
  sources or tool-backed retrieval (the `deep-research` harness, web search) before acting on them.
- **Match output to the decision.** A briefing exists to drive an action or a recommendation — land on
  the specific "so what," not a tidy summary.

## Definition of done

Four phases complete · perspectives were independent and genuinely distinct · contradictions/agreements/
silence mapped · findings ranked by reliability with a concrete action · a self-critique pass with
confidence scores and a named missing angle · load-bearing claims flagged for (or checked against) real
sources. If any is missing, say so — the briefing is provisional.
