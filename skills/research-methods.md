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

---

# Reference — research-methods

# Research Methods — Full Reference (multi-perspective, fast, self-critiqued)

Most people use an LLM as a search box: ask, read the first answer, close the tab. That answer is the
**majority framing** — the most common way the topic is discussed in the training data. It is exactly
what everyone else also gets. The value in research is not the first answer; it is **asking the same
question from several independent angles, finding where the answers conflict, and grading your own
confidence before acting.**

This skill distills the Stanford **STORM** method (Synthesis of Topic Outlines through Retrieval and
Multi-perspective Question Asking; Stanford OVAL Lab, NAACL 2024) into a **tool-free, four-prompt
workflow** you can run inside any capable LLM. STORM's research contribution is the *pre-writing* stage:
instead of one query, it researches a topic by **simulating multiple expert perspectives asking
questions**, which the paper reports produces more **organized** and **broader-coverage** results than
single-perspective research. You don't need the codebase to get most of that benefit — you need the way
of thinking.

> Distinction: `[[ai-research-science]]` is depth on ML/AI *content*; this skill is the *method* for
> researching anything. For real, tool-backed web research with live citations, use a retrieval harness
> (e.g. the `deep-research` harness or web search) — this method organizes reasoning, it does not fetch
> ground truth.

---

## When to use this skill

- **Before writing** an article, report, design doc, or briefing — so it covers angles others miss.
- **Before a high-stakes decision** — build/buy, vendor choice, architecture bet, investment, hire.
- **Due diligence / prep** — a company before an interview, a counterparty before a negotiation, a field
  before you start learning it.
- **Red-teaming a thesis** — you have a belief and want the strongest case against it before you commit.

**When not to:** a simple lookup with one right answer (just ask); a topic where you already hold deep
expertise (the lenses tell you what you know); or anything where being *fluently wrong* is dangerous and
you haven't got primary sources to verify against — there, lead with retrieval, not simulation.

---

## The method — four phases, in order

Run them as four prompts. The output of each feeds the next. **Keep perspectives independent** (Phase 1
must generate each lens before it sees the others) — that independence is the entire mechanism.

### Phase 1 — Multi-perspective scan (the heart of the method)

Simulate 4–6 **genuinely different** experts on the topic. The default set, because these lenses reliably
see different things:

- **Practitioner** — works with it daily. What do academics miss? What practical realities get ignored?
- **Academic** — studied it for years. What does the peer-reviewed evidence actually say, and where does
  it contradict popular belief?
- **Skeptic** — thinks the mainstream is wrong. Strongest counterargument? What do proponents ignore?
- **Economist / incentives** — follows the money. Who profits from the current narrative? What financial
  incentives shape what gets said?
- **Historian** — has seen the pattern before. What parallels exist, and how did they play out?

For **each** lens, force three things: core position (2 sentences), the strongest evidence for it, and
**the one thing only this lens would tell you**. That last item is where the blind-spot value lives.

*Gate:* the lenses genuinely disagree. If they read like one voice in five hats, your perspectives aren't
distinct enough (see "Choosing perspectives").

### Phase 2 — Contradiction map

Now make the model find where the voices fight — the conflicts are where real understanding lives.

- **Direct contradictions** — list each, with the specific clashing claims. Which side has the stronger
  evidence, which the weaker, and why?
- **The resolving question** — the single question that, if answered, would settle the biggest conflict.
- **Universal agreement** — what does *every* lens agree on? Even opponents confirming it makes it
  **likely true**; this is your firm ground.
- **Collective silence** — what did **none** of the lenses address? That's the blind spot in how the
  topic is usually discussed — frequently the most valuable finding in the whole exercise.

Most people skip this phase. It is the step that separates surface understanding from expertise.

### Phase 3 — Synthesis

Pull it together into a briefing no single lens could write:

- **One-paragraph summary** for someone with 60 seconds who needs *nuance*, not the headline.
- **Key findings ranked by reliability** — and for each, which lenses support it and which challenge it.
  Reliability ranking is what makes this actionable: you know what to lean on and what to hedge.
- **The hidden connection** — one non-obvious link between findings that only appears across all lenses.
- **The actionable insight** — given the evidence, what should *someone in your role* do differently? Be
  specific. A briefing that doesn't land on a "so what" failed.

### Phase 4 — Peer-review gate (never skip)

The model is **sycophantic and confidently wrong**; STORM's own documented weakness is that it does not
self-critique, so **source bias and fact misassociation** slip in. Make the model grade its own work:

- **Confidence scores** — rate each key finding 1–10 for reliability, with the reason for the score.
- **Weakest link** — which claim are you least sure of, and what specific information would verify it?
- **Bias check** — which perspective is overrepresented? Did one voice dominate the synthesis?
- **Missing perspective** — is there a 6th lens that would change the conclusions?
- **Overall grade** — if a domain expert reviewed this, what grade and what would they tell you to fix?

*Gate:* you now know what's solid, what's shaky, and what to verify before you rely on any of it.

---

## Choosing perspectives (the part people get wrong)

- **Pick lenses that actually differ *for this topic*.** The practitioner/academic/skeptic/economist/
  historian default is a strong general set, but swap it when the domain has sharper natural divisions:
  - *Infra/architecture decision:* SRE/on-call, security, FinOps/cost, the vendor, the
    outage-historian (what broke last time we did this).
  - *Product bet:* user, the skeptic PM, growth/economics, support (sees the failures), a competitor.
  - *Scientific claim:* the original author, a replication skeptic, a statistician, a practitioner who
    tried to use it.
- **4–6 lenses.** Fewer than 4 and you're back near single-prompt; more than 6 and they blur and repeat.
- **Independence beats quantity.** Two truly opposed lenses beat five that nod along. If two lenses keep
  agreeing, replace one with a sharper opponent.

---

## The verify gate matters most — and why

This method's biggest risk is the same as its appeal: it produces a **fluent, organized, confident**
briefing. Organization is not correctness. Two failure modes to design against:

- **Source misattribution / fabricated citations.** The model will attach a real-sounding source to a
  claim it didn't make, or invent a citation. **Never trust a citation you haven't seen.** For any
  load-bearing claim, pull the primary source (tool-backed retrieval, web search, the actual paper).
- **Sycophantic self-grading.** A model asked to critique itself often rates its own work highly. Make
  the critique **adversarial** (Phase 4 framed as "what would a tough expert reject?"), and where stakes
  are high, run the critique in a **fresh context** or a **different model** so it isn't defending its
  own prior tokens — the cross-model verify-gate pattern from `[[ml-evaluation-evals]]`.

The honest framing: Phases 1–3 *organize reasoning*; Phase 4 *flags risk*; **only real sources establish
fact.** Escalate to retrieval the moment a decision rides on a specific number, quote, or citation.

---

## Putting it to work

Same four prompts, different intent:

- **Research/writing** — run all four; your draft answers objections before they're raised.
- **Decision** — the contradiction map shows where the real risk lives; the synthesis names the action.
- **Interview/negotiation prep** — the practitioner lens gives you insider language; the skeptic lens
  gives you the sharp questions; the economist lens gives you their incentives.
- **Learning a field** — the practitioner says what to learn first; the skeptic says what's overhyped;
  you skip the noise.

Promote it to a habit: a saved prompt set you run before any briefing or decision. (To run it on a
schedule over changing inputs, wrap it in a loop → `[[skill-self-improvement]]` / a research harness.)

---

## Rationalizations & rebuttals

- *"One good prompt is enough."* → One prompt returns the majority view — the same thing everyone else
  gets. The edge is in the lenses that disagree with it.
- *"The perspectives basically agreed, so I'll skip the contradiction map."* → Then either your lenses
  weren't distinct (fix Phase 1) or you found genuine consensus (valuable — but confirm it wasn't one
  voice wearing five hats).
- *"It cited sources, so it's grounded."* → LLMs fabricate and misattribute citations. A citation you
  haven't opened is a guess. Verify load-bearing ones.
- *"I asked it to check itself and it said it's solid."* → Self-grading is sycophantic. Make the critique
  adversarial and, for stakes, run it in a fresh context or different model.
- *"This replaces reading the sources."* → It replaces the *disorganization* of reading, not the reading.
  For facts that carry a decision, go to the primary source.

---

## Red flags — stop and reconsider

- Your five "perspectives" read like one voice — no real disagreement surfaced.
- You skipped the contradiction map and went straight to a tidy summary.
- A decision is riding on a specific figure/quote/citation you took from the synthesis **unverified**.
- The peer-review phase rated everything 8–10 with no weak link named — that's sycophancy, not rigor.
- One lens (usually the practitioner or the mainstream view) dominated the synthesis.
- The briefing has no specific action — it's a summary pretending to be research.

---

## Verification gate (definition of done)

- [ ] **Four phases ran**, in order, perspectives generated independently.
- [ ] **Perspectives were genuinely distinct** and chosen to fit the topic (not five identical voices).
- [ ] **Contradictions, universal agreements, and collective silence** are all mapped.
- [ ] **Findings ranked by reliability**, each tagged with supporting/challenging lenses, landing on a
      **specific action**.
- [ ] **Peer-review pass done:** confidence scores, a named weakest link + how to verify it, a bias
      check, and a missing-perspective check.
- [ ] **Load-bearing claims verified against primary sources** (or explicitly flagged as unverified).
      Organization is not fact.

If any box is unchecked, the briefing is provisional — say which.

---

## Version awareness & canonical references

It is 2026; verify currency. STORM and its collaborative successor **Co-STORM** are active research — the
exact prompts, demo, and reported metrics evolve. Treat the specific gains as *reported by the paper*,
not gospel, and re-check before you cite numbers.

- STORM paper — *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models*,
  Shao et al., NAACL 2024 (Stanford OVAL).
- Code — github.com/stanford-oval/storm (MIT). Live demo — storm.genie.stanford.edu.
- Related: `[[ml-evaluation-evals]]` (LLM-as-judge biases — the basis for an honest Phase-4 gate),
  `[[ai-research-science]]` (ML content depth), `[[verification-and-debugging]]` (verify to root cause),
  `[[spec-driven-development]]` (scope the question first), `[[staff-plus-engineering]]` (briefing →
  decision doc). For tool-backed retrieval with real citations, use the `deep-research` harness.

---

# Research Methods — the four prompts (copy-paste)

Replace the bracketed parts. Run in order; paste each into the same conversation so later prompts see the
earlier output. Keep Phase 1 independent — don't reveal the lenses to each other.

---

## Prompt 1 — Multi-perspective scan

```
I need to research [YOUR TOPIC].

Simulate 5 different expert perspectives. Answer each INDEPENDENTLY before considering the others:

1. THE PRACTITIONER (works with this daily): What do they know that academics miss? What practical
   realities are usually ignored?
2. THE ACADEMIC (has studied this for years): What does the peer-reviewed evidence actually say? Where
   does the evidence contradict popular belief?
3. THE SKEPTIC (thinks the mainstream view is wrong): What is the strongest counterargument? What
   evidence do proponents conveniently ignore?
4. THE ECONOMIST (follows the money): Who profits from the current narrative? What financial incentives
   shape the research/discourse?
5. THE HISTORIAN (has seen similar patterns): What historical parallels exist? How did those play out?

For each perspective give me:
- Their core position in 2 sentences
- The strongest evidence supporting their view
- The one thing they would tell me that no other perspective would
```

## Prompt 2 — Contradiction map

```
Based on the 5 perspectives above, map the contradictions:

1. Where do two or more perspectives directly contradict each other? List each conflict with the
   specific claims that clash.
2. Which perspective has the strongest evidence? Which the weakest? Why?
3. What single question, if answered, would resolve the biggest contradiction?
4. What does EVERY perspective agree on? (Likely true — even opponents confirm it.)
5. What did NONE of the perspectives address? (The blind spot in the whole field — often the most
   valuable finding.)
```

## Prompt 3 — Synthesis

```
Synthesize the 5 perspectives and the contradiction map into a research briefing:

1. ONE-PARAGRAPH SUMMARY: brief a decision-maker who has 60 seconds and needs nuance, not the headline.
2. KEY FINDINGS (ranked by reliability): the most important things I now know. For each, note which
   perspectives support it and which challenge it.
3. THE HIDDEN CONNECTION: one non-obvious link between findings that only shows up across all 5 lenses.
4. THE ACTIONABLE INSIGHT: given the evidence, what should someone in [YOUR ROLE] do differently? Be
   specific.
5. THE FRONTIER QUESTION: the one question that, if answered, would change how we understand this topic.
```

## Prompt 4 — Peer-review gate (never skip)

```
Now peer-review your own briefing:

1. CONFIDENCE SCORES: rate each key finding 1-10 for reliability. Explain each score.
2. WEAKEST LINK: which claim are you least confident in? What specific information would verify it?
3. BIAS CHECK: which perspective is overrepresented in the synthesis? Did one voice dominate?
4. MISSING PERSPECTIVE: is there a 6th angle that would change the conclusions?
5. OVERALL GRADE: if a tough domain expert reviewed this, what grade would they give, and what would
   they tell me to fix?

Then list which claims I must verify against a primary source before acting on them.
```

---

## Swap the lenses to fit the topic

The 5 defaults are a general set. For a technical/infra decision, sharper lenses disagree more usefully:

```
Simulate 5 perspectives on [DECISION, e.g. "adopt service mesh X for our platform"]:
1. THE SRE / ON-CALL: what breaks at 3am? what's the operational burden no slide deck mentions?
2. THE SECURITY ENGINEER: what's the new attack surface / blast radius / supply-chain risk?
3. THE FinOps / COST OWNER: what does this actually cost at our scale (compute, licenses, headcount)?
4. THE VENDOR / PROPONENT: the strongest honest case for adopting it now.
5. THE OUTAGE-HISTORIAN: when we (or peers) did something like this before, how did it go?
[...same per-lens questions and the same Prompts 2-4 as above...]
```

Other ready sets: *product bet* → user / skeptic-PM / growth-economics / support / competitor;
*scientific claim* → original-author / replication-skeptic / statistician / practitioner-who-tried-it.

---

## Mini worked example (shape, not content)

**Topic:** "Should we standardize the team on a single LLM serving engine?"

- **Phase 1** surfaces five reads: the SRE wants one engine (less on-call surface); the practitioner
  notes different models serve best on different engines; the cost owner sees GPU-hour differences; the
  vendor pitches lock-in-free portability; the outage-historian recalls a past forced-migration.
- **Phase 2** finds the clash: *"one engine"* (ops simplicity) vs *"right engine per model"* (perf/cost).
  Universal agreement: a thin abstraction over the engine is worth it. Nobody addressed: who owns the
  upgrade treadmill — the real silent cost.
- **Phase 3** ranks: "standardize the *interface*, not the *engine*" (high reliability — every lens
  supports it); action: adopt one OpenAI-compatible serving interface, allow 2 engines behind it.
- **Phase 4** flags the weak link: the GPU-hour cost delta is an unverified estimate → pull real
  benchmark numbers before committing → `[[serving-frameworks]]`, `[[inference-optimization]]`.

The briefing's value isn't the summary — it's that it named the silent cost (the upgrade treadmill) and
landed on a specific, defensible action that no single lens produced.
