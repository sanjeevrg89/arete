---
name: research-methods
description: A structured method for researching any topic deeply and fast with an LLM — multi-perspective
  questioning, contradiction mapping, synthesis, and a self-critique/peer-review gate — instead of asking
  one question and taking the majority view. Use when you need to understand a topic, make a high-stakes
  decision, do due diligence, prep for an interview/negotiation, red-team a thesis, or write a briefing,
  and you want the blind spots a single prompt misses. Implements the Stanford STORM idea (multi-
  perspective question asking, NAACL 2024) as a tool-free 4-phase workflow: (1) simulate independent
  expert perspectives, (2) map where they contradict / agree / are silent, (3) synthesize a reliability-
  ranked briefing with an actionable insight, (4) peer-review it for confidence, bias, and missing angles.
  Covers picking perspectives that actually differ, the verify gate (LLMs confidently misattribute
  sources — STORM's known weakness), and when to escalate to real source retrieval. For ML research
  *content* depth see ai-research-science; for tool-backed web research use the deep-research harness.
---

# Research Methods (multi-perspective, fast, self-critiqued)

Apply the judgment of a strong research analyst who knows that the value isn't in the first answer — it's
in asking the same question from five angles, finding where the answers fight, and grading your own
confidence before you act. One prompt gives you the majority view; this gives you the blind spots.

## How to use this skill

1. Read `research-methods-guide.md` in this directory — the 4-phase method, how to choose perspectives,
   the verify gate, and the limits (when to stop trusting the model and pull real sources). Apply it.
2. For the four copy-paste prompts (and a domain-swapped perspective set), read `examples.md`.
3. **Run the phases in order and keep perspectives independent** — generate each lens *before* it sees the
   others, or they collapse into one view. End with the peer-review phase every time; it's the step that
   separates a briefing from a confident guess.

## The essentials (full detail in `research-methods-guide.md`)

- **One prompt returns the majority framing.** The breakthrough (Stanford STORM, NAACL 2024) is
  **multi-perspective question asking** — independent expert lenses catch what single-prompt research
  never sees.
- **Phase 1 — Multi-perspective scan.** Simulate 4–6 *genuinely different* experts (default:
  practitioner, academic, skeptic, economist/incentives, historian). For each: core position, strongest
  evidence, and the one thing only they would tell you.
- **Phase 2 — Contradiction map.** Where do lenses clash (that's where understanding lives)? What do
  **all** agree on (likely true)? What did **none** address (the field's blind spot — often the most
  valuable finding)?
- **Phase 3 — Synthesis.** A briefing: one-paragraph nuanced summary, findings **ranked by reliability**
  (with which lenses support/challenge each), the non-obvious connection, and a **specific action**.
- **Phase 4 — Peer-review gate (never skip).** Make the model grade its own briefing: confidence scores,
  weakest link, bias check (did one voice dominate?), missing 6th perspective, overall grade + fixes.
- **Perspectives are a tool, not magic — pick lenses that actually differ for *your* topic.** Swap the
  defaults per domain (for an infra decision: SRE, security, FinOps/cost, vendor, outage-historian).
- **The verify gate matters most because the model is confidently wrong.** STORM's documented weakness is
  no self-critique → **source bias and fact misassociation**. Phase 4 mitigates; it does not eliminate.
- **This organizes reasoning; it does not replace primary sources.** For load-bearing or high-stakes
  claims, verify against real sources / tool-backed retrieval → the `deep-research` harness, web search,
  `[[verification-and-debugging]]`.
- **Independence is the whole trick.** If each perspective can see the others before answering, you've
  rebuilt the single-prompt majority view with extra steps.

## Related skills

- `[[ai-research-science]]` — research-scientist *content* depth on ML/AI topics; this skill is the
  *method* for researching any topic, ML or not.
- `[[ml-evaluation-evals]]` — LLM-as-judge done right (position/verbosity/self-preference bias): the
  discipline behind the Phase-4 self-critique gate.
- `[[verification-and-debugging]]` — verify load-bearing claims to root cause instead of trusting a
  fluent synthesis.
- `[[spec-driven-development]]` — sharpen the research question/scope before you spend prompts; a vague
  topic yields a vague briefing.
- `[[staff-plus-engineering]]` — turning a briefing into a decision doc / recommendation with a clear ask.
