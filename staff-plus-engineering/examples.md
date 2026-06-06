# Staff+ Engineering — Worked Examples

Canonical artifacts to imitate. Copy the structure; replace the content. These are *templates and
framings*, not filled-in essays — the point is the skeleton a Staff+ engineer reaches for.

---

## 1. Technical design doc / RFC template (one-pager-friendly)

A design doc is written *to be reviewed* — the comment thread is where alignment actually happens.
Keep it the shortest doc that lets a smart skeptic agree or push back. Fill the first three sections
before you write any code.

```markdown
# RFC: <short imperative title, e.g. "Unify training and serving model formats">

Author(s): <name> · Reviewers: <named, not "the team"> · Status: Draft | In review | Approved
Last updated: <date> · Decision needed by: <date>

## TL;DR
<2–4 sentences. The problem, the proposal, and the ask. A reader who stops here knows what you
want and why. (For an exec audience, this is the whole doc — lead with the decision and the impact.)>

## Context & problem
<What's true today, and why it's a problem now. Be concrete and, where you can, quantify it
(latency, $ spend, incident rate, engineer-hours, time-to-market). State the root cause, not the
symptom — this is your "diagnosis.">

## Goals / Non-goals
- Goals: <the specific outcomes this must achieve — measurable where possible>
- Non-goals: <what this explicitly does NOT do — scope discipline; prevents the comment thread
  from sprawling>

## Proposed design
<The actual design. Diagram if it helps. One level of detail down from the abstraction — enough to
review, not a code dump. Call out the load-bearing decisions and why they go this way.>

## Alternatives considered
<For each: the option, and the honest reason it lost. This is the highest-signal section — a doc
with no real alternatives reads as "I already decided." Show your work.>
| Option | Pros | Cons | Why not chosen |
|--------|------|------|----------------|

## Risks & mitigations
<The 2–4 things most likely to go wrong, and how you de-risk each. Name the scariest one first.>

## Rollout / migration plan
<How this ships without a big-bang: phases, feature-flag/dark-launch, backfill, the rollback path,
and what "done" means. Migrations are where Staff+ docs earn trust.>

## Cost & operational impact
<Compute/$ delta, on-call burden, new failure modes, what other teams must change.>

## Open questions
<The genuinely unresolved things you want reviewers to weigh in on. Inviting disagreement here is
how you get real alignment instead of silent nods.>
```

**How to drive it:** share the draft, pre-socialize the contentious parts 1:1 with likely detractors
*before* the review meeting, give a fixed feedback window, resolve disagreements in the doc (or
escalate to a named decider), and record the decision. The meeting should ratify, not litigate.

---

## 2. Engineering strategy outline (diagnosis → guiding policy → coherent action)

Rumelt's three-part kernel, the way Larson uses it. A strategy that says *no* to nothing isn't a
strategy. Write it yourself — don't delegate the thinking.

```markdown
# Engineering Strategy: <domain, e.g. "AI Inference Platform, FY26">

Owner: <Staff+ name> · Sponsor: <exec backing it> · Working group: <3–5 named stakeholders>
Horizon: <~1 year> · Revisit on: <date — strategy is a living doc>

## 1. Diagnosis — what is actually going on
<A clear-eyed theory of the core challenge. The honest root constraint, not a wish list. Example:
"Every model team builds its own serving path, so we run 6 incompatible inference stacks; this
triples on-call load, blocks org-wide latency/cost wins, and makes new-model launches slow.">

## 2. Guiding policy — our approach, and what we're NOT doing
<The overall direction and the trade-offs we are deliberately choosing. Example: "Converge on one
paved-path inference platform with a pluggable backend. We optimize for launch velocity and unit
cost over per-team flexibility. We will NOT support bespoke serving stacks for new launches.">

## 3. Coherent actions — what follows, sequenced
<3–6 mutually reinforcing actions, in order, with the de-risking/foundation bet first. Each should
obviously follow from the guiding policy.>
1. <Foundation/de-risking bet first — e.g. ship the paved-path platform behind a flag for one
   pilot team and prove the latency/cost numbers.>
2. <...>
3. <Migration of existing teams, sequenced by leverage/risk.>
4. <Deprecation: turn off the old paths on a dated timeline.>

## What this is worth (the "so what")
<Quantified outcome — $ saved, latency won, launches unblocked, on-call reduced. This is the
exec-facing payoff and the promotion-case evidence.>

## Assumptions & what would change this
<Date your assumptions. Name what, if it changed (a new model architecture, a vendor option, a cost
shift), would make us revisit. Especially important in fast-moving AI infra (2026).>
```

> **Vision vs. strategy (Larson):** to write a *strategy*, write ~5 design docs and pull out the
> similarities. To write a *vision*, write ~5 strategies and forecast their implications ~2 years out.
> Strategy addresses today's constraints; vision describes where they lead.

---

## 3. "Staff project" framing — tied to AI-platform work

A "staff project" is a real, org-level, ambiguous effort chosen *because* succeeding at it requires —
and therefore demonstrates — the Staff+ competencies. Promotions are retroactive: the project
manufactures the evidence. Frame it before you start so the impact is legible.

```markdown
# Staff Project Framing: "Org-wide AI Inference Platform"

## The problem (why it's Staff-sized, not team-sized)
6 model teams each maintain their own serving stack. No single team owns the org-level cost,
latency, and reliability of inference. Leadership cares (GPU spend up 3x YoY). No one is positioned
to fix it because it spans teams none of them control — exactly the gap a Staff+ Architect fills.

## Why it demonstrates the competencies (the evidence it will produce)
- **Scope & impact:** org-level, multi-team, leadership-visible — not a team deliverable.
- **Technical leadership & direction:** I set the platform architecture and the standards every
  model team inherits (deploy path, eval gates, autoscaling policy).
- **Influence without authority:** success = getting 6 teams I don't manage to migrate. The lever
  is a written strategy + RFC + 1:1 pre-socialization + an exec sponsor — not a mandate.
- **Cross-stack judgment (AI-architect):** the design has to reason across serving, the model/
  quantization trade-off, training/fine-tuning constraints, the K8s GPU/TPU footprint, cost, and
  safety/eval coverage at once — see [[ml-system-design]], [[aiml-on-kubernetes]],
  [[mlops-lifecycle]], [[responsible-ai-governance]].
- **Execution of an ambiguous, large effort:** multi-quarter, sequenced migration that has to
  survive reprioritization and finish.
- **Multiplying others:** the paved path makes every future launch faster; I sponsor an engineer
  from a pilot team to co-lead a migration wave.

## How impact stays legible (avoiding invisible glue work)
- A strategy doc + RFC, both reviewed in the open (artifacts the promotion committee can read).
- A dated rollout with a quantified "so what" (GPU $ saved, p99 latency won, launches unblocked).
- A short written retro after each migration wave, in outcome terms — not "I helped coordinate."
- Manager + sponsor kept supplied with the narrative so they can represent it in calibration.

## Anti-patterns I will avoid
- Building the platform alone (hero-coding) instead of getting teams to adopt it (leverage).
- Driving migrations by mandate instead of by a doc + sponsorship.
- A strategy deck no team acts on — every guiding policy maps to a sequenced, owned action.
- Doing the glue work silently — every unblock gets a one-line written narrative of what it enabled.
```

---

### Cross-references
- Design discipline at org scale: [[ml-system-design]]
- Platform substrate and make-vs-buy: [[aiml-on-kubernetes]]
- Where platform standards and tech debt accrue: [[mlops-lifecycle]]
- Org-level technical-health concern an AI Staff+ owns: [[responsible-ai-governance]]
- The code-level taste this scales into org impact: [[go-best-practices]]
