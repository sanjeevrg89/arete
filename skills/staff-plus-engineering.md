---
name: staff-plus-engineering
description: The non-technical and technical-leadership competencies that distinguish Staff, Senior Staff, Principal, and Distinguished engineers from Senior — the things that actually earn promotion past the senior "terminal" level. Use when working on technical leadership, technical strategy/vision, architecture direction, a design doc / RFC / one-pager / Amazon-style narrative, influence without authority, alignment/sponsorship/consensus, leading large ambiguous multi-team efforts, mentorship & sponsorship, the Staff archetypes (Tech Lead, Architect, Solver, Right Hand), glue work, a "staff project" or promotion case, make-vs-buy and tech-debt-as-portfolio, or org-level AI-platform strategy across training/serving/data/infra/cost/safety. Grounded in StaffEng archetypes, Will Larson's "Staff Engineer", and Tanya Reilly's "The Staff Engineer's Path". Ladders/titles vary by company.
---

# Staff+ Engineering

Apply the judgment of a Staff/Principal/Distinguished engineer who has reached org-level impact
through scope, leverage, and influence — not by coding more. The rest of this library is the
*technical* mastery; this skill is how that mastery becomes **world-class impact through others.**
Exact ladders, titles, and level boundaries vary by company — the role's *shape* is universal.

## How to use this skill

1. **Read `staff-plus-engineering-guide.md`** in this directory — the full reference (the
   Senior→Staff discontinuity, the four archetypes, the six core competencies, writing/strategy, how
   to operate, the AI-architect dimension, and anti-patterns). Apply it to the task.
2. For artifacts to imitate — a technical design-doc / RFC template, an engineering-strategy outline,
   and a "staff project" framing tied to AI-platform work — read **`examples.md`**.
3. Adapt to the specific company's ladder, rubric, and doc culture; the competencies are
   industry-general, the mechanics (titles, packet format, who decides) are local — verify them.

## The essentials (full rationale in `staff-plus-engineering-guide.md`)

- **Senior→Staff is a discontinuity, not "more coding."** Senior is judged on *your* output; Staff+
  on what the *org* produces because you were there. Three words: **scope, leverage, impact through
  others.** Your hours are your least important output.
- **Know your archetype** — Tech Lead (a team), Architect (a domain), Solver (the current fire),
  Right Hand (an exec's technical proxy). For AI infra, **Architect** is the default shape.
- **Lead with leverage, not heroics.** The highest-value move is usually a doc that aligns ten teams,
  a strategy that kills a doomed project early, or a mentee who becomes Staff — not your own commits.
- **Pick the right problem.** Staff impact is org-level. The skill is *problem selection* — the
  highest-leverage problem in reach — and the discipline to skip the interesting ones that don't matter.
- **Influence without authority is the whole game.** You have responsibility without command. Currency
  is trust/credibility/social capital: bank it with good calls and real work; spend it on alignment,
  sponsorship, and consensus. Influence by mandate corrodes; being demonstrably right compounds.
- **Writing is the primary tool of leverage.** Design docs / RFCs / one-pagers / Amazon-style
  narratives scale and persist where meetings don't. Write for the audience — execs (BLUF, one page,
  business framing) vs. engineers (depth, trade-offs). If you can't write clearly, your influence is
  capped at the meetings you can attend.
- **Strategy = diagnosis → guiding policy → coherent action** (Rumelt/Larson). Write it yourself;
  pre-socialize with detractors 1:1; sequence the bets; treat it as a living doc. Vision forecasts
  ~2 years out. A strategy that doesn't say *no* to something isn't a strategy.
- **Manage tech debt as a portfolio**, make-vs-buy on TCO + differentiation (not sticker price), and
  build only where it's a genuine edge — most infra is buy/adopt.
- **Multiply others.** Mentorship gives time; **sponsorship** spends *your* capital on someone's
  opportunity — scarcer and worth more. You're a role model whether you want to be or not.
- **Make glue work legible.** The invisible coordinating/unblocking is often the highest-leverage work
  — and gets you passed over if you never build the narrative. Write up what you enabled, in outcomes.
- **Stay technical.** The Staff+ track exists so you *don't* drift to management. Stale judgment makes
  you an expensive manager without a team. Stay close enough to the work that your opinions are earned.
- **Promotions are retroactive.** Staff+ recognizes impact you're already having, evidenced by
  artifacts and testimony — not tenure. The "staff project" manufactures that evidence.
- **AI-architect:** be the one person who reasons across training ↔ serving ↔ data ↔ infra ↔ cost ↔
  safety and makes *defensible* (written-down, alternatives-considered) org-level calls. Favor
  reversible decisions where the ground is moving (it is 2026).

## Related skills

- `[[ml-system-design]]` — the technical design discipline a Staff+ AI-architect drives at org scale.
- `[[aiml-on-kubernetes]]` — the platform substrate; Staff+ owns its direction and make-vs-buy.
- `[[mlops-lifecycle]]` — where platform standards and technical debt accrue across the lifecycle.
- `[[responsible-ai-governance]]` — safety/eval/governance as an org-level technical-health concern.
- `[[go-best-practices]]` — the code-level taste this skill turns into org-level impact.

---

# Reference — staff-plus-engineering

# Staff+ Engineering Guide

The reference for operating as a Staff, Senior Staff, Principal, or Distinguished engineer — the
non-technical and technical-leadership competencies that distinguish Staff+ from Senior. The hard
technical skills in the rest of this library are necessary but not sufficient; this guide is how they
turn into **org-level impact**. Titles, ladders, and exact level boundaries vary by company — treat
the *shape* of the role as universal and verify the specifics against your own ladder.

## Mental model: the Senior→Staff discontinuity

Senior is, at most companies, the **terminal level** — the point at which you can stay indefinitely
with good performance and no further promotion. Every level up to Senior is mostly "do harder
technical work, more independently, on a larger surface." Staff is a **discontinuity, not a
continuation**. It is not "Senior but codes more." The promotion criteria change kind, not degree:

- **Senior** is judged on the work *you* produce and the team-level outcomes you drive.
- **Staff+** is judged on the work *the organization* produces because you were there — impact
  through others, through systems, through decisions, through raising the technical bar.

Three words capture the shift:

- **Scope** — from a team's problems to an org's or a domain's problems. The blast radius of your
  decisions, and the time horizon you're responsible for, both grow.
- **Leverage** — your hours are now your *least* important output. A design doc that aligns ten
  teams, a strategy that kills a doomed project early, a mentee who becomes Staff — each is worth
  more than anything you could type yourself.
- **Impact through others** — you succeed by making other people and teams more effective, by being
  right about hard ambiguous calls, and by getting an organization to *act* on being right.

### Why the "Staff Engineer" role exists

Companies created the Staff+ track because forcing senior technical talent into management to keep
growing was destroying both: you lost your best engineers from the keyboard and got mediocre
managers. The IC leadership ladder (Staff → Senior Staff → Principal → Senior Principal →
Distinguished → Fellow, names vary) is the answer — a way to grow scope, compensation, and authority
**without managing people**, for those whose comparative advantage is technical judgment and
technical leadership. Roughly ~10% of engineers reach Staff; ~1–2% reach Principal or above. The
levels are not a strict ladder of "more of the same" — Distinguished/Fellow is often a *director-
level IC* setting technical direction for an entire org or company.

The defining tension of the role: **you have leadership responsibility without management authority.**
You can't assign work, do performance reviews, or order anyone to do anything. Everything you achieve
above your own two hands is achieved through influence. That is the whole game.

## The four Staff archetypes

The StaffEng project (Will Larson) identifies four common shapes of the Staff role. Most people are
a blend, and the blend shifts with the company and the moment. Knowing your archetype tells you where
to spend time and what "good" looks like.

| Archetype | Scope | Day-to-day | Relationship style |
|-----------|-------|------------|--------------------|
| **Tech Lead** | One team or a cluster of teams | Guides execution of complex work, owns technical vision for the team, carries cross-team/cross-functional context, partners tightly with the EM and PM | Deep, long-term, with a consistent team |
| **Architect** | A critical technical domain (e.g. API surface, data platform, inference stack) | Maintains intimate knowledge of business needs + user goals + technical constraints; sets direction and standards for the domain; reviews and unblocks | Deep, long-term, domain-anchored |
| **Solver** | Wherever the current fire is | Parachutes into a leadership-identified critical problem, drives it to resolution, then moves on. Less org "chiropractics" — the problem is pre-blessed as important | Transactional, problem-by-problem |
| **Right Hand** | Org-wide, on behalf of a senior exec | Operates as a senior leader without reports; borrows the exec's authority; attends leadership meetings; handles the gnarliest cross-cutting business/tech/people/process problems. Emerges only in large orgs (hundreds of engineers) | Transactional, leadership-adjacent |

Tech Leads and Architects build durable relationships with a stable group; Solvers and Right Hands
move between problems. None is more "senior" than another — they're different bets on where you
create value.

### Matching archetypes to AI-platform / infrastructure work

- **Architect** is the most common Staff+ shape for an AI-infrastructure leader: you own the
  inference platform, the training stack, or the data/feature platform as a domain, and you set the
  technical direction that the rest of this library implements. You are the person who can reason
  across [[ml-system-design]], [[aiml-on-kubernetes]], serving, training, cost, and safety and make a
  *defensible* org-level call.
- **Tech Lead** fits when one platform team (e.g. the team building the inference gateway or the
  training operator) needs technical direction and cross-team glue more than it needs another coder.
- **Solver** fits the "our largest model launch is on fire / our GPU spend tripled / latency regressed
  org-wide" moment — go in, fix it, write up what changed, leave it better instrumented.
- **Right Hand** fits a CTO/VP who needs a technical peer to own "what is our AI platform strategy and
  why" across many teams — make-vs-buy, build-vs-adopt, where to place the next bet.

## Core competencies

### 1. Technical leadership & direction

Set where the technology is going, not just build the next thing. This means owning architecture and
technical vision for a scope larger than one team: picking the abstractions, the boundaries, the
"north star" the platform converges on, and the sequence of bets to get there. The output is often a
**vision** (where we want to be in ~2 years and why it's worth it) plus the **strategy** to get there
(see Technical Strategy below). The bar is *judgment under uncertainty*: being right about
consequential, ambiguous, hard-to-reverse decisions more often than chance, and being able to explain
*why* so others can act on it.

### 2. Scope & impact

The single biggest Senior→Staff lever. Staff+ impact is **org-level, not team-level.** Two failure
modes: working on team-sized problems with Staff-sized talent (under-leveraged), or working on
genuinely big problems that *don't matter* (busy, not impactful). The skill is **problem selection**:
finding the highest-leverage problem in your reach — the one where your judgment changes the outcome
for many teams — and having the discipline to *not* do the ten interesting things that aren't it.
Ask: if I'm wildly successful at this, who is better off, and by how much?

### 3. Influence without authority

You have responsibility without command. The currency is **trust / credibility / social capital** —
you bank it through a track record of good calls, doing real work, and helping peers; you spend it to
move decisions. The concrete skills:

- **Alignment** — getting many stakeholders to *genuinely* agree on a direction, not just nod in a
  meeting. Disagreement surfaced early and resolved is alignment; silence is not.
- **Sponsorship** — an exec or senior leader publicly backing the direction (and backing *you*). Get
  it before you need it; pre-socialize 1:1 with potential detractors before any big meeting so the
  meeting ratifies a decision rather than litigating it.
- **Building consensus** — driving toward a decision the group will actually execute, often via the
  written-doc-plus-review loop below.
- **Navigating disagreement** — "disagree and commit," steelmanning the other side, knowing when to
  escalate to a decider vs. keep building consensus, and *changing your own mind in public* when the
  data says so (this builds credibility, it doesn't spend it).

Influence by mandate ("leadership said so") is the weakest form and erodes fast. Influence by being
consistently, demonstrably right — and generous with context — compounds.

### 4. Execution of ambiguous / large efforts

Staff+ work arrives under-specified: "our inference platform won't scale to next year's models" is
the whole brief. You turn ambiguity into a sequenced plan, define the milestones, identify the load-
bearing risks and de-risk them first, keep multiple teams moving in the same direction, and *finish*.
Leading a multi-quarter, multi-team effort to a real outcome — surviving reorgs, re-prioritization,
and the boring middle — is itself a Staff-defining skill. Stalled/blocked projects are where Staff+
earn their keep: diagnose *why* it's stuck (technical? alignment? resourcing? motivation?) and
unblock at the right layer.

### 5. Mentorship, sponsorship & multiplying others

Your job is to make the engineers around you better, because that's how leverage actually works.

- **Mentorship** = advice and growth (you give your time). **Sponsorship** = spending your own
  capital and reputation to put someone on a visible project, recommend them for promotion, or hand
  them an opportunity. Sponsorship is scarcer and worth more; deliberately sponsor people who don't
  look like you.
- You are a **role model whether you want to be or not** — your habits (how you write, review, argue,
  handle being wrong) set the engineering culture for people who never report to you. Code review
  comments, design-doc feedback, and how you behave in an incident are all teaching, at scale.
- Multiply: a good standard, template, or paved path makes *every* future project better; teaching
  ten engineers to make a class of decision is worth more than making it yourself ten times.

### 6. Engineering judgment & taste

The hardest to name and the most valuable. Taste is the accumulated, often-implicit sense of which
designs will age well, which abstractions will leak, which "simple" thing is actually simple, when to
build vs. buy, when "good enough" is correct and when it's a trap. It's why people bring you the hard
call. You build it by shipping and *living with* systems over years (especially your own mistakes),
and you make it transferable by *explaining the why* — turning taste into reusable principles other
engineers can apply without you in the room.

## Writing & communication: the core tool

At Staff+, **writing is the primary instrument of leverage.** A document scales to readers you'll
never meet, persists after the meeting, forces your own thinking into rigor, and lets dozens of people
align asynchronously across time zones. If you can't write clearly, your influence is capped at the
number of meetings you can attend. The strongest Staff+ engineers are, almost without exception,
strong writers.

### The artifacts

- **Technical design doc / RFC** — the workhorse. A proposal for a non-trivial technical change:
  context and problem, goals/non-goals, the proposed design, alternatives considered (with honest
  trade-offs), risks, and a rollout/migration plan. Written *to be reviewed* — the comment thread is
  where alignment happens. (Template in `examples.md`.)
- **One-pager / brief** — the smallest doc that makes a decision. Forces you to find the actual crux.
- **Engineering strategy / vision** — see below.
- **The "written-doc, read-in-the-room" culture** (popularized by Amazon's six-pager / narrative
  memos and PR/FAQ): banning slide decks in favor of a written narrative read silently at the start of
  the meeting. The point isn't the page count — it's that a prose argument has to be *complete and
  logically sound* in a way bullet points let you fake. Adopt the spirit even where it's not mandated:
  prose narrative over decks for anything consequential.

### Writing for executives vs. engineers

Different audiences need different docs — the *same* doc rarely serves both.

| | For engineers | For executives |
|---|---|---|
| Lead with | Context, then design | The decision/ask and the "so what," up top (BLUF) |
| Detail level | Deep; correctness matters | Ruthlessly compressed; one level of detail down, no more |
| Frame around | Technical correctness, trade-offs | Business outcome, cost, risk, what you need from them |
| Length | As long as needed | One page; respect their time |
| Jargon | Fine, shared vocabulary | Translate; assume zero context on your system |

For execs: state the recommendation and the ask in the first three sentences, quantify the impact
(latency, dollars, risk, time-to-market), and make the decision you want them to make easy to make.

### Tech talks & spoken communication

Talks, brown-bags, and architecture reviews scale a message and build the credibility that funds
influence. The skill mirrors writing: know your audience, lead with the point, one main idea, make it
memorable and repeatable so it propagates without you.

## Technical strategy

Strategy is how a vision becomes a sequence of decisions an organization can actually execute.

### Structure (Rumelt, via Larson): diagnosis → guiding policy → coherent action

1. **Diagnosis** — a clear-eyed theory of *what the actual challenge is*. Most bad strategies skip
   this and jump to solutions. Name the root constraint honestly (e.g. "we can't ship new model
   architectures because serving and training have diverged into two incompatible stacks").
2. **Guiding policy** — the overall approach to the challenge, including the trade-offs you're
   *choosing* (a strategy that doesn't say no to something isn't a strategy).
3. **Coherent action** — specific, mutually-reinforcing actions that follow from the policy.

### How to actually write and drive one (Larson's process, condensed)

- **Write it yourself** — don't delegate the thinking. A practical shortcut: *to write a strategy,
  write ~five design docs and pull out the similarities; to write a vision, write ~five strategies and
  forecast their implications ~2 years out.* Strategy addresses today's constraints; vision forecasts
  where they lead.
- **Small working group (3–5)**, draft the diagnosis first, iterate, get executive review on the
  guiding policy, pre-socialize with extended stakeholders and detractors 1:1, then share broadly with
  a fixed feedback window, finalize, and **commit to revisiting impact** (e.g. in two months). Strategy
  is a living document, not a stone tablet.

### Strategic decisions you'll own

- **Sequencing bets** — order matters; do the de-risking, foundation-laying, or unblocking bet first.
  Don't start three big migrations at once.
- **Make-vs-buy at org scale** — building a platform component vs. adopting OSS vs. buying a vendor.
  Weigh total cost of ownership, differentiation (is this our edge or just plumbing?), talent, lock-in,
  and opportunity cost — not just the sticker price. Most infra is buy/adopt; build only where it's a
  genuine differentiator.
- **Technical debt as a portfolio** — debt isn't uniformly bad; it's leverage with interest. Manage it
  like a portfolio: know which debt is high-interest (slowing every team, raising incident rate) and
  pay that down; tolerate cheap debt in stable corners. Tie paydown to a narrative ("this unblocks X"),
  never "we should clean this up." See [[mlops-lifecycle]] for where platform debt accrues fastest.

## Operating as Staff+

### Glue work and its (in)visibility

"Glue work" — the unglamorous alignment, unblocking, doc-writing, cross-team coordination, and
context-carrying that holds a large effort together — is often *the* highest-leverage thing a Staff+
person does, and it is frequently **invisible** in a system that rewards shipped code. Two
consequences: (1) do the glue work because it's the job, but (2) **build the narrative** so the impact
is legible — write up what you unblocked and what it enabled, in outcome terms. Invisible glue work
with no narrative is how strong Staff+ engineers get passed over. Also watch the trap where glue work
defaults disproportionately to underrepresented engineers; sponsor it being recognized.

### Staying technical vs. drifting to management

The Staff+ track exists *so you stay technical*. The pull toward becoming a shadow-manager (all
meetings, no engineering) is constant. You don't have to be the one typing, but you must stay close
enough to the work to keep your judgment sharp — read the code, prototype the risky part, review the
hard designs. A Staff engineer whose technical judgment has gone stale is just an expensive manager
without a team. Stay in the codebase enough that your opinions are still *earned*.

### The promotion / "staff project" reality

At most companies, **Staff+ promotions are retroactive recognition of impact you're already having**,
demonstrated through artifacts and the testimony of people you've influenced — not a reward for tenure
or a checklist. The pragmatic implication: the "staff project" (a high-visibility, org-level,
ambiguous effort where you can demonstrably exercise the competencies above) is how you *manufacture
the evidence*. Pick a real problem that matters to leadership, where success requires influence and
judgment, not just heroic coding — then make the impact legible in writing. Your manager and sponsor
are your partners here; they translate your impact to the promotion committee, so keep them supplied
with the narrative.

### Working with managers, PMs, and leadership

- **Your manager** is your partner, not your boss-as-obstacle: they handle people/process, you handle
  technical direction; the best pairs operate as a duo. Keep them informed; surface risks early.
- **PMs** own the "what/why for the business," you own the "how/what's-possible technically" — the
  healthy relationship is a negotiation, not a handoff.
- **Leadership** needs translation: turn technical reality into business consequence, give them the
  decision framed for *their* altitude, and never make them feel ambushed in a meeting.

### On-call for the org's technical health

Staff+ engineers carry a standing, informal responsibility for the technical health of their scope:
spotting the slow-motion problems no single team owns (the architecture that won't scale, the
reliability rot, the security posture, the cost curve), and raising them *before* they become
incidents. You're the smoke detector for class-of-problem risks. This is also where
[[responsible-ai-governance]] lives for an AI platform — safety, eval coverage, and governance are
org-level technical-health concerns that need a Staff+ owner, not a team-level afterthought.

## The AI-architect dimension

For a Staff+ engineer in AI infrastructure, the defining capability is **reasoning across the whole
stack and making defensible org-level technical decisions** spanning the rest of this library:

- **Cross-stack judgment** — connect training ↔ serving ↔ data ↔ infra ↔ cost ↔ safety. A serving
  latency target ([[ml-system-design]]) constrains the model and quantization, which constrains
  training and fine-tuning, which constrains data and the GPU/TPU footprint on Kubernetes
  ([[aiml-on-kubernetes]]), which sets the cost curve, which leadership cares about. The Staff+
  AI-architect is the one person who can hold *all* of that at once and find the global optimum
  instead of a local one.
- **AI platform strategy** — make-vs-buy across the stack (managed inference vs. self-hosted
  serving-frameworks; build a training platform vs. adopt one), sequencing the platform roadmap,
  setting standards (eval gates, checkpoint formats, deployment paths) that every model team inherits,
  and managing platform debt across [[mlops-lifecycle]].
- **Defensible decisions** — "defensible" means written down, with the alternatives considered, the
  trade-offs explicit, and the reasoning surviving a smart skeptic's review. The AI space moves fast
  (it is 2026; frameworks, model architectures, and hardware shift quarterly) — so favor reversible
  decisions, keep optionality where the ground is moving, and *date your assumptions* so the strategy
  can be revisited when reality changes.

## Anti-patterns

- **Hero-coding instead of leverage.** Doing all the hard work yourself feels productive and is the
  classic failure mode. If the org's output didn't scale beyond your hands, you operated as a very
  senior Senior, not as Staff.
- **Invisible glue work with no narrative.** Doing the essential coordinating/unblocking but never
  making it legible. The impact is real; the *evidence* isn't — and you get passed over.
- **Strategy decks no one executes.** A beautiful strategy/vision deck that doesn't change what any
  team does on Monday. Strategy that isn't sequenced into coherent action that people actually take is
  theater. (And: a deck where a narrative memo was needed.)
- **Influence by mandate.** Leaning on "leadership decided" or your title instead of earning agreement.
  It works once and corrodes your credibility — the thing your whole influence depends on.
- **Hoarding context.** Being the indispensable single point of knowledge. It feels like job security;
  it's actually a bottleneck and a failure to multiply. Spread context aggressively — write it down.
- **Never writing things down.** Operating purely through meetings and hallway influence. It doesn't
  scale, doesn't persist, can't be reviewed asynchronously, and leaves no artifact for a promotion
  case. The non-writing Staff engineer has a hard ceiling.
- **Chasing the title, not the impact.** Optimizing for the promotion instead of doing work that
  matters. Promotions are retroactive; impact is the only thing that reliably produces them.
- **Drifting out of the technical.** Becoming all-meetings, judgment going stale. Then you're neither
  a good IC nor a manager.

## Version / context awareness

Ladders, titles, level boundaries, and promotion processes **differ substantially by company** — some
have Senior Staff, some don't; "Principal" can sit above or below "Staff" depending on the company;
Distinguished/Fellow may be company-wide or org-wide. Treat the *competencies* in this guide as
industry-general and verify the specific mechanics (rubrics, calibration, packet format, who decides)
against your own organization's ladder. The literature below is the canonical, current grounding.

## Canonical references

- StaffEng — Staff Archetypes: https://staffeng.com/guides/staff-archetypes/
- StaffEng — Overview / Introduction: https://staffeng.com/guides/overview-overview/
- Will Larson, *Staff Engineer: Leadership beyond the management track* (book) and lethain.com
- Will Larson — How to write an engineering strategy / vision: https://lethain.com/eng-strategies/
- Tanya Reilly, *The Staff Engineer's Path* (O'Reilly): three pillars — big-picture thinking,
  execution, leveling up others. https://www.oreilly.com/library/view/the-staff-engineers/9781098118723/
- ShiftMag — Staff, Principal, Distinguished levels explained:
  https://shiftmag.dev/staff-principal-distinguished-engineering-career-levels-explained-3565/
- Gergely Orosz (Pragmatic Engineer) — What is a Staff+ Engineer:
  https://newsletter.pragmaticengineer.com/p/what-is-a-staff-engineer
- LeadDev — Who are staff, principal, and distinguished engineers:
  https://leaddev.com/career-development/who-are-staff-principal-and-distinguished-engineers
- Richard Rumelt, *Good Strategy / Bad Strategy* (diagnosis → guiding policy → coherent action).

---

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
