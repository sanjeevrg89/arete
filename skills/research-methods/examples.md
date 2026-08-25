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
