# Responsible AI Governance — Full Reference

The discipline of building AI systems that are **fair, safe, accountable, transparent, and auditable**,
and of standing up the org-level governance that keeps them that way through their lifecycle. This is the
*governance/safety/ethics* layer. Adversarial and runtime security (prompt injection, model theft,
infra hardening) is the sibling concern [[ai-security-on-gke]]; the two interlock but are not the same.

The bar: think like a head of responsible AI / ML risk who has shipped governed systems, sat on a review
board, run red-team exercises, and answered to auditors and regulators. Governance is not a document you
write once at launch — it is a control loop that runs for the life of the system.

> **Regulation and standards move fast.** Laws (EU AI Act, US state and federal rules, sectoral
> regulation), enforcement timelines, and harmonized standards are evolving in 2026. Treat every legal or
> numeric claim below as *directional* and **verify current** primary sources before relying on it.
> Engineers are not lawyers — loop in counsel for anything with regulatory teeth.

## Mental model: governance as a control loop, not a gate

Bad responsible-AI programs bolt an ethics review onto the end of a project, ship, and forget. Good ones
treat responsibility as a **continuous control loop** aligned to a recognized framework, embedded in the
ML lifecycle ([[mlops-lifecycle]]) rather than parallel to it:

1. **Govern** — policies, roles, accountability, culture, risk appetite. The umbrella that makes the
   other three repeatable rather than heroic.
2. **Map** — context: what is the system for, who is affected, what could go wrong, what does the law
   require? Produce an entry in the **AI inventory** and a **risk profile**.
3. **Measure** — quantify the risks you mapped: fairness metrics, safety evals, robustness, privacy,
   groundedness. Slice-based, not aggregate-only ([[ml-evaluation-evals]]).
4. **Manage** — prioritize, mitigate, decide go/no-go, monitor in production, respond to incidents,
   and feed findings back into Map/Measure.

This Govern/Map/Measure/Manage decomposition is the spine of the **NIST AI Risk Management Framework
(AI RMF 1.0)**. Use it as your organizing skeleton even if your legal obligations come from elsewhere —
it maps cleanly onto ISO/IEC 42001 and the EU AI Act's risk-management-system requirements.

## Frameworks & governance

### NIST AI RMF (voluntary, influential, free)
The **NIST AI Risk Management Framework 1.0** (Jan 2023) is a voluntary, rights- and
safety-oriented framework structured around four functions — **Govern, Map, Measure, Manage** — each
broken into categories and subcategories. It is paired with a **Playbook** of suggested actions and a
**Generative AI Profile** (NIST AI 600-1, 2024) that enumerates GenAI-specific risks (confabulation/
hallucination, dangerous/violent content, data privacy, info integrity, CBRN uplift, etc.). It is not
law and not certifiable, but it is the lingua franca US regulators, auditors, and procurement teams
expect you to speak.
- Primary: `https://airc.nist.gov/airmf-resources/airmf` and `https://www.nist.gov/itl/ai-risk-management-framework`
- Use it to *structure* your program and to map your controls; don't treat "we follow NIST" as a
  compliance claim.

### EU AI Act (law — verify current)
The EU AI Act is a **risk-tiered** regulation. At a high level the tiers are:
- **Unacceptable risk** — prohibited practices (e.g., certain social scoring, manipulative or
  exploitative systems, some biometric categorization). Banned.
- **High risk** — systems in regulated/safety-critical domains and listed use cases; subject to the
  heaviest obligations: risk management system, data governance, technical documentation, logging,
  human oversight, accuracy/robustness/cybersecurity, conformity assessment, registration.
- **Limited risk** — transparency obligations (e.g., disclosing AI interaction, labeling synthetic
  media / deepfakes).
- **Minimal risk** — the bulk of systems; no mandatory obligations.
There are additional obligations for **general-purpose AI (GPAI) models**, with extra duties for models
deemed to carry **systemic risk**.
> The **specific** classifications, obligation lists, applicability dates, fine ceilings, and the GPAI
> systemic-risk compute threshold have phased timelines and are subject to amendment, guidance, and
> harmonized standards still being finalized. **Do not quote dates, thresholds, or penalty figures from
> memory — verify the current consolidated text and Commission guidance.** Primary entry point:
> `https://artificial-intelligence-act.eu/` and the EUR-Lex consolidated text.

### ISO/IEC 42001 (certifiable management system)
**ISO/IEC 42001:2023** is the AI Management System (AIMS) standard — a Plan-Do-Check-Act management
system you can be **certified** against, analogous to ISO 27001 for infosec. It is process-oriented
(policies, objectives, roles, risk treatment, continual improvement) rather than prescribing technical
metrics. Pair it with **ISO/IEC 23894** (AI risk management guidance) and **ISO/IEC 42005** (AI system
impact assessment) for the technical/risk content.

### Standing up org-level governance
The minimum viable governance program:
- **AI policy & risk appetite** — what uses are sanctioned, what is prohibited, who can approve
  exceptions. Tie to a recognized framework.
- **AI inventory** — a living registry of every model/AI system: owner, purpose, data sources, risk
  tier, deployment status, last review. You cannot govern what you cannot enumerate; the inventory is
  the single most under-built and most valuable artifact.
- **Risk register** — per-system enumerated risks with likelihood/impact, owner, mitigation, residual
  risk, and review cadence (see `examples.md` for a template mapped to NIST functions).
- **Review board / approval gates** — a cross-functional body (eng, legal, privacy, security, domain,
  and ideally affected-community representation) that reviews high-risk systems at defined gates
  (design, pre-launch, major change). Make the gate proportional to risk — a low-risk back-office tool
  should not need the same scrutiny as a credit-decisioning model.
- **Roles** — name an accountable owner per system (RACI). "Everyone is responsible" means no one is.
- **Impact / conformity assessments** — for high-risk systems, a documented assessment (algorithmic
  impact assessment / DPIA / EU AI Act conformity assessment) before launch.

## Documentation & transparency artifacts

Documentation is how governance becomes auditable. Each artifact answers a different question; produce
them as code/data alongside the model, version them, and regenerate them on retrain — link lineage to
your pipeline ([[mlops-lifecycle]]).

| Artifact | Question it answers | Canonical reference |
|---|---|---|
| **Model card** | What is this model, how does it perform, where should it (not) be used? | Mitchell et al., arXiv:1810.03993 |
| **Datasheet for datasets** | How was this dataset collected, composed, intended, and what are its biases? | Gebru et al., arXiv:1803.09010 |
| **Data statement** | (NLP) Who produced the language data, in what context, for whom? | Bender & Friedman, TACL 2018 |
| **System card** | How does the whole *system* (model + guardrails + UX + policy) behave and fail? | OpenAI/Meta system cards |

**Model cards** (the load-bearing artifact) should cover: model details (owner, version, date,
architecture, license); intended use and **out-of-scope/misuse**; factors (relevant groups, instruments,
environments); metrics **disaggregated by slice**; evaluation and training data; ethical considerations;
caveats and recommendations. The point of the disaggregated metrics section is to force the question
"for *whom* does this work?" rather than reporting a single headline number.

**Datasheets for datasets** document motivation, composition, collection process, preprocessing/cleaning,
recommended uses, distribution, and maintenance — so downstream users can judge fit and inherited bias.

Provenance/lineage is itself a transparency control: record data sources, transformations, model
versions, and the eval run that gated each release. The EU AI Act's logging and technical-documentation
duties for high-risk systems are, in practice, this discipline written into law. Tie lineage to your
feature store and pipelines ([[data-engineering-feature-stores]], [[mlops-lifecycle]]).

## Fairness & bias

### Where bias comes from
Bias is not one thing. Name the source before you reach for a metric:
- **Historical/societal bias** — the world the data reflects is already unequal; a perfectly accurate
  model can still perpetuate harm.
- **Representation/sampling bias** — groups under- or over-sampled relative to the deployment population.
- **Measurement/label bias** — the label is a flawed proxy (e.g., "arrested" ≠ "committed crime";
  "clicked" ≠ "satisfied"). The most insidious and most common.
- **Aggregation bias** — one model forced onto subpopulations with genuinely different relationships.
- **Feedback-loop bias** — the model's outputs shape future training data (predictive policing,
  recommender lock-in), amplifying initial skew over time. Monitor for this explicitly.
- **Deployment/use bias** — the system is used differently, or on a different population, than intended.

### Fairness definitions are mutually incompatible
There is **no single "fair."** The major group-fairness criteria:
- **Demographic / statistical parity** — positive-prediction rate equal across groups. Ignores ground
  truth; can force unequal error rates.
- **Equalized odds** — equal TPR *and* FPR across groups. **Equal opportunity** is the relaxation
  requiring only equal TPR.
- **Calibration / predictive parity** — a given score means the same probability across groups.

**Impossibility results** (Kleinberg et al. 2016; Chouldechova 2017): when base rates differ across
groups, you generally **cannot** simultaneously satisfy calibration and equalized odds (except in
degenerate cases). This is mathematics, not a tooling gap. Therefore: **choose the fairness criterion
deliberately from the harm and the legal/ethical context — do not let a library pick for you**, and
document the choice and its trade-offs in the model card. Also distinguish **group** fairness from
**individual** fairness ("similar individuals treated similarly") and **counterfactual** fairness;
they can conflict.

### Measure and mitigate
- **Measure**: slice-based evaluation across protected and intersectional groups (not just marginal
  groups — harm often hides at intersections). Report disaggregated metrics with confidence intervals;
  small slices are noisy. Toolkits: Fairlearn, AIF360, What-If Tool, TFMA. ([[ml-evaluation-evals]])
- **Mitigate** at one of three stages:
  - **Pre-processing** — reweight/resample/transform data to reduce skew before training.
  - **In-processing** — fairness constraints/regularizers in the objective (e.g., exponentiated
    gradient reductions in Fairlearn).
  - **Post-processing** — adjust thresholds per group, or calibrate outputs, after training.
  Each has trade-offs; post-processing per-group thresholds may itself be legally fraught (disparate
  treatment). There is almost always a **fairness/accuracy trade-off** — surface it, don't hide it.
- Fairness is **not measured once**. Distributions drift; re-measure on a schedule and monitor in prod
  for **fairness drift** ([[ml-observability-monitoring]]).

## LLM safety

Generative systems add failure modes that classical ML evaluation does not catch. Treat safety as a
first-class, measurable property, not a vibe.

### Harms taxonomy
Build evals against an explicit taxonomy so coverage is auditable. Common categories: hate/harassment/
toxicity; violence and dangerous instructions; sexual content and CSAM (zero-tolerance, dedicated
pipelines); self-harm; illegal activity; privacy violations (PII regurgitation, doxxing); deception/
fraud/manipulation; **CBRN and cyber-offense uplift**; bias/stereotyping; misinformation; and
representational harms. NIST's GenAI Profile (AI 600-1) and the MLCommons **AILuminate** taxonomy are
useful starting structures.

### Red-teaming
Adversarial probing of the *deployed system* for harmful outputs and policy violations. Combine:
- **Manual** expert red-teaming (domain specialists for high-stakes harms like CBRN/bio).
- **Automated** red-teaming (attacker models / fuzzers generating adversarial prompts at scale).
- **Crowd/structured** exercises with documented scope, severity rubric, and tracked remediation.
Red-teaming is governance evidence: log findings, severities, and fixes. Distinguish safety red-teaming
(harmful content/misuse) from **security** red-teaming (prompt injection, exfiltration, jailbreak-as-
attack) — the latter is [[ai-security-on-gke]]; they overlap and should share findings.

### Guardrails & content safety
Defense in depth around the model: **input** filters (prompt/PII classifiers), **output** filters
(toxicity/safety classifiers, regex/PII scrubbers), retrieval and tool-use constraints, and a system
prompt / policy layer. Open tooling: Llama Guard / Prompt Guard, NeMo Guardrails, Guardrails AI,
provider moderation APIs. Guardrails are **probabilistic** — they have false positives and negatives;
they complement, never replace, model-level alignment and human oversight. Runtime enforcement and
threat-model coverage live in [[ai-security-on-gke]].

### Hallucination / groundedness as a safety issue
A confident fabrication in a medical, legal, or financial context is a safety harm, not just a quality
bug. Mitigate with retrieval grounding ([[rag-vector-databases]]), citation/attribution, "I don't
know" calibration, and **groundedness/faithfulness evals** that check outputs against provided context.
Measure factuality and groundedness as named safety metrics ([[ml-evaluation-evals]]).

### Safety evals & the refusal balance
- Run against safety benchmarks/datasets as one signal — but **benchmarks leak and saturate**; treat
  public scores skeptically and maintain private, scenario-specific eval sets. (Relevant suites evolve
  fast — e.g., HarmBench, AILuminate, TruthfulQA, ToxiGen; **verify current** and check for
  contamination.)
- Balance **refusal vs over-refusal**: a model that refuses benign requests ("how do I kill a Python
  process?") is also a failure. Track a helpfulness/harmlessness frontier; measure over-refusal
  explicitly (e.g., XSTune-style benign-but-sensitive prompts).

### Alignment techniques
- **RLHF** — fit a reward model to human preference data, then RL-optimize the policy against it.
- **Constitutional AI / RLAIF** — use a written set of principles ("constitution") and model-generated
  critiques to provide alignment signal with less direct human labeling.
- **DPO** and related preference-optimization methods skip the explicit reward model.
Alignment reduces but does not eliminate harmful behavior; it is one layer among evals, guardrails, and
oversight. Mechanics and training frameworks: [[rl-rlhf-frameworks]].

## Privacy & data governance

- **PII handling & minimization** — collect only what you need, for a stated purpose, retained only as
  long as justified. Inventory where PII enters training data and prompts/logs (LLM logs are a common
  leak path — scrub before storage).
- **Consent & lawful basis** — document the basis for using personal data in training and inference;
  honor purpose limitation. (Specifics are jurisdictional — **verify current**.)
- **Differential privacy** (concept) — provides a formal, tunable privacy guarantee (the **ε** budget)
  by adding calibrated noise; DP-SGD is the standard training mechanism. Smaller ε = stronger privacy,
  usually lower utility. Use when individual-level leakage is a real risk; it is not free accuracy-wise.
- **Federated learning** (concept) — train across decentralized data that never leaves the device/silo,
  sharing model updates instead of raw data. Reduces (does not eliminate) leakage; combine with DP and
  secure aggregation because updates can still leak.
- **Right-to-be-forgotten / machine unlearning** — deleting a record from the training set does not
  remove its influence from trained weights. Removing that influence (retrain, or approximate
  unlearning) is an **open, hard problem** with no robust, verifiable general solution today; design for
  it (data lineage, scoped retraining, retention limits) rather than assuming you can "untrain."
- **Training-data provenance & copyright** — track sources and licenses of training data; the legal
  status of training on copyrighted/scraped/PII data and the resulting outputs is **actively litigated
  and unsettled** (jurisdiction-dependent). **Verify current** law and loop in counsel; engineering's
  job is provenance records that make the question answerable.
- **Memorization & extraction** — LLMs can regurgitate training data verbatim; test for it, especially
  with PII or copyrighted text.

## Accountability & operations

- **Human oversight** — define where a human is in/on the loop and ensure it is **meaningful**, not
  rubber-stamping. Calibrate against **automation bias** (over-trusting outputs) and give humans real
  ability to override. High-risk decisions affecting people generally require it (and the EU AI Act
  mandates it for high-risk — verify current).
- **Incident response for AI harms** — you need an AI incident plan *before* an incident: detection,
  triage/severity, containment (kill switch / rollback / route to fallback), notification (users,
  regulators, affected parties), root cause, and remediation. Maintain a postmortem culture; consider
  contributing to the AI Incident Database. Treat a fairness regression or a harmful-output spike as a
  Sev, not a backlog ticket.
- **Auditability & logging** — log inputs, outputs, model version, decision rationale where feasible,
  and overrides, with retention that supports audit and dispute resolution — **without** creating a new
  PII liability (scrub, access-control, retention-limit the logs). This is also an EU AI Act high-risk
  requirement (verify current).
- **Explainability / interpretability** — feature-attribution methods (**SHAP**, **LIME**, integrated
  gradients) give per-prediction local explanations for tabular/classical models. Know the limits:
  these are *approximations*, can be unstable/manipulable, and **explainability ≠ a correct
  causal account**. For LLMs, post-hoc attributions are far weaker; chain-of-thought is a
  rationalization, not a faithful trace; mechanistic interpretability is promising but immature. Don't
  oversell an explanation as ground truth.
- **Monitoring for fairness/safety drift** — production monitoring must include disaggregated
  performance, safety-classifier rates, refusal rates, groundedness, and data drift — with alerting and
  ownership. Governance that stops at launch is theater. ([[ml-observability-monitoring]])

## Anti-patterns

- **Ethics as a checkbox at the end.** A review the day before launch can only rubber-stamp. Embed
  governance from design (Map) onward.
- **No model card / no datasheet.** Undocumented models are ungovernable and unauditable.
- **Fairness measured once.** A single pre-launch fairness check that is never repeated; distributions
  drift and so does fairness.
- **Aggregate-only metrics.** A single headline accuracy hides slice failures and intersectional harm.
- **No red-team for a generative system.** Shipping an LLM feature without adversarial safety testing.
- **Guardrails as the whole strategy.** Probabilistic filters bolted on, with no model-level alignment,
  evals, or oversight behind them.
- **No incident plan.** Discovering you have no kill switch, rollback, or notification path *during* an
  incident.
- **Ignoring evolving regulation.** Treating a one-time legal review as permanent compliance while the
  law shifts under you.
- **"Safety is someone else's job."** Responsibility outsourced to a lone ethics team with no authority,
  while product/eng treat it as friction. It must be owned across the org with real gates.
- **Fairwashing / ethics theater.** Publishing principles and a model card while the metrics that
  matter go unmeasured or unenforced.
- **Confusing governance with security.** Assuming a pen-test covers fairness, or that a fairness audit
  covers prompt injection. Both are required; see [[ai-security-on-gke]].

## Version & currency awareness

It is 2026 and this domain is moving faster than almost any other in tech. **Verify current** before
relying on:
- Any EU AI Act classification, obligation, date, threshold, or penalty figure.
- US federal/state AI law and executive actions, and sectoral regulation (finance, health, employment).
- ISO/IEC 42001 and companion standards' status and any certification scheme details.
- The contents of the NIST AI RMF Playbook and GenAI Profile (they get updated).
- Specific safety benchmarks/leaderboards (they saturate, leak, and are superseded).
- Copyright/training-data case law (actively litigated, jurisdiction-specific).
Cite primary sources, date your claims, and prefer "as of <date>, verify current" over false precision.

## Canonical references (verify current)

- **NIST AI RMF 1.0** + Playbook + crosswalks: `https://airc.nist.gov/airmf-resources/airmf` ·
  `https://www.nist.gov/itl/ai-risk-management-framework`
- **NIST AI 600-1, Generative AI Profile**: `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf`
- **Model Cards for Model Reporting** — Mitchell et al., 2019: `https://arxiv.org/abs/1810.03993`
- **Datasheets for Datasets** — Gebru et al., 2018/2021: `https://arxiv.org/abs/1803.09010`
- **Data Statements for NLP** — Bender & Friedman, TACL 2018:
  `https://aclanthology.org/Q18-1041/`
- **Inherent Trade-Offs in Fair Determination of Risk Scores** — Kleinberg, Mullainathan, Raghavan,
  2016: `https://arxiv.org/abs/1609.05807`
- **Fair prediction with disparate impact** — Chouldechova, 2017: `https://arxiv.org/abs/1610.07524`
- **Constitutional AI** — Bai et al., 2022: `https://arxiv.org/abs/2212.08073`
- **EU AI Act** (verify current consolidated text): `https://artificial-intelligence-act.eu/` and EUR-Lex
- **ISO/IEC 42001:2023** (AI management system): `https://www.iso.org/standard/81230.html`
- **Fairlearn**: `https://fairlearn.org/` · **AIF360**: `https://aif360.res.ibm.com/`
- **MLCommons AILuminate / AI Safety**: `https://mlcommons.org/`
- **AI Incident Database**: `https://incidentdatabase.ai/`
