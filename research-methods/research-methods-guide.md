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
