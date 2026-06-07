---
name: adversarial-ml-robustness
description: Model-level adversarial machine learning and robustness — at the bar of a researcher who breaks
  and defends models for a living. Use when threat-modeling an ML/LLM model, evaluating or claiming
  robustness, or defending against evasion / adversarial examples (FGSM, PGD, C&W, transferable & patch/
  physical attacks), data poisoning, backdoors/trojans (clean-label, trigger-based), model extraction/
  stealing, model inversion, membership inference, or LLM jailbreaks / training-data extraction. Covers the
  NIST Adversarial ML taxonomy (AI 100-2e2025) and MITRE ATLAS; threat models (white/black/gray-box, Lp
  budgets); defenses and their limits (adversarial training, certified/randomized smoothing, why defensive
  distillation & gradient masking are false security); and the core skill — honest robustness evaluation
  with adaptive attacks, AutoAttack, and RobustBench. Defensive, not an attack playbook.
---

# Adversarial ML & Robustness

Apply the judgment of a researcher who breaks defenses for a living and therefore knows how to build and —
above all — **honestly evaluate** robust ones. This is a **defensive** skill: you understand attacks in
order to defend models and to refuse inflated robustness claims. The hardest and most valuable part is
**evaluation** — most published "robust" defenses were broken because they were tested against weak,
non-adaptive attacks. Robustness is **always relative to a stated threat model**; a claim without one is
meaningless.

Boundary: this is **model-level adversarial ML** (perturbations, poisoning, extraction/inversion, membership
inference, jailbreaks-as-evasion, robustness evaluation). Cloud/runtime/infra security, guardrails,
sandboxing, and supply-chain *enforcement* live in `[[ai-security-on-gke]]`. They compose: this skill says
*what* to defend against; that one says *where in the platform* to enforce it.

## How to use this skill

1. **Read `adversarial-ml-robustness-guide.md`** in this directory — the full reference (taxonomy/frameworks,
   attack categories, defenses and their limits, evaluation methodology, production, anti-patterns). Apply it
   to the task.
2. For an **adaptive robustness-evaluation checklist (with AutoAttack)**, a **backdoor/poisoning detection
   note**, and a **threat-model template**, read **`examples.md`**.
3. Match the surrounding codebase/eval-harness conventions; apply the correctness and evaluation-integrity
   rules regardless. The field moves fast (2026): treat attack/defense names, library APIs, leaderboard
   numbers, and arXiv IDs as **verify against current docs**.

## Essentials (full detail in `adversarial-ml-robustness-guide.md`)

- **No threat model, no robustness claim.** Pin down attacker knowledge (white/gray/black-box), goal
  (targeted/untargeted), capability/budget (`Lp` norm + `ε`, or semantic/physical), and access (query rate,
  pipeline/weight access) *before* evaluating. Evaluate the white-box worst case.
- **Anchor on the standard taxonomies.** Use **NIST AI 100-2e2025** vocabulary and map adversary activity to
  **MITRE ATLAS** tactics/techniques so reports are unambiguous and threat-modeling is systematic.
- **Know the attack families to defend them.** Evasion/adversarial examples (FGSM → PGD → C&W,
  transferability, patch/physical); poisoning & **backdoors/trojans** (clean-label, trigger-based);
  **model extraction**, **model inversion**, **membership inference** (`[[privacy-preserving-ml]]`); LLM
  **jailbreaks-as-evasion**, prompt injection (`[[ai-security-on-gke]]`, `[[llm-app-agent-frameworks]]`),
  training-data extraction/memorization (`[[pretraining-data-tokenizers]]`).
- **The only durable empirical defense is adversarial training** (Madry/PGD) — and it costs 3–30× training
  time and clean accuracy, and is budget-specific. State the budget.
- **Certified robustness gives a guarantee — but only inside its ball.** Randomized smoothing (`L2`
  probabilistic certificate, scales) and IBP/verifiers (deterministic, conservative). Report certified
  radius/accuracy; outside the ball you have *no* guarantee.
- **Gradient masking is false security.** Defensive distillation and most input-transform/detection defenses
  lower the *measured* attack, not the real one, and fall to adaptive attacks (BPDA/EOT/transfer). Run the
  gradient-masking sanity checks; treat transforms/detectors as defense-in-depth only.
- **Adaptive attacks are non-negotiable.** The attacker knows your defense. The minimum bar for any empirical
  robustness claim is **AutoAttack** (Croce & Hein, parameter-free ensemble) **plus a defense-aware adaptive
  attack** — AutoAttack alone can still over-estimate a quirky defense.
- **Use RobustBench** for a strong pretrained baseline and to sanity-check your number against SOTA for the
  *same* threat model. Beating SOTA by a wide margin = suspect your evaluation first.
- **Report clean AND robust accuracy** with the exact threat model; surface the **robustness–accuracy
  tradeoff** and the operating point. Distinguish *empirical* ("couldn't break it") from *certified*
  ("provably unbreakable in this ball").
- **Defend the whole pipeline, not just the model.** Poisoning/backdoors enter via **third-party datasets and
  pretrained weights** — verify provenance, dedup, scan, prefer `safetensors`, and trigger-probe before
  trusting a checkpoint (`[[pretraining-data-tokenizers]]`, enforcement in `[[ai-security-on-gke]]`).
- **Monitor and plan IR.** Watch for query/input/output attack signatures (`[[ml-observability-monitoring]]`);
  rate-limit and coarsen outputs to raise extraction/inversion cost; keep model/dataset lineage so you can
  scope and roll back a poisoning incident.
- **Red-team, but don't mistake it for proof.** Structured adversarial probing feeds defense; "no findings"
  is not robustness. Coordinate with `[[ml-evaluation-evals]]` and `[[responsible-ai-governance]]`.
- **Anti-patterns:** no threat model; weak/non-adaptive eval; claiming robustness without AutoAttack +
  adaptive attack; gradient-masking defenses; ignoring poisoning in third-party data/weights; `Lp`-only
  thinking for patch/physical/text threats; reporting only one of clean/robust accuracy.

## Related skills

- `[[ai-security-on-gke]]` — the platform/runtime side: prompt-injection & jailbreak filtering, sandboxing,
  guardrails, supply-chain *enforcement* (signing, `safetensors`). This skill = the model-level adversary.
- `[[privacy-preserving-ml]]` — DP-SGD and the principled defense against membership inference / model
  inversion / training-data extraction.
- `[[ml-evaluation-evals]]` — eval harness and methodology that robustness/red-team evaluation plugs into.
- `[[responsible-ai-governance]]` — policy, model cards/risk docs, and sign-off for red-teaming results.
- `[[llm-app-agent-frameworks]]` — building the agentic apps where jailbreaks/prompt injection land.
- `[[pretraining-data-tokenizers]]` — data dedup/curation; the poisoning/memorization surface of web corpora.
- `[[ml-observability-monitoring]]` — wiring up attack-signature and distribution-shift monitoring.
</content>

---

# Reference — adversarial-ml-robustness

# Adversarial ML & Robustness — Guide

The authoritative reference for this skill. This is a **defensive** discipline: you defend models by
understanding how they fail under an adaptive adversary, then **evaluate robustness honestly** and put
the right mitigation at the right place in the pipeline. The single most important skill here is
**evaluation** — most published "robust" defenses were later broken because they were tested against
weak, non-adaptive attacks. Read this whole file before claiming a model is robust.

Boundary: this skill is **model-level adversarial ML** (perturbations, poisoning, extraction, inversion,
membership inference, jailbreaks-as-evasion, robustness evaluation). Cloud/runtime/infra security,
guardrails, sandboxing, and supply-chain *controls* live in `[[ai-security-on-gke]]`. The two compose —
adversarial-ML knowledge tells you *what* to defend against; that skill tells you *where in the platform*
to enforce it.

The field moves fast (it is 2026). Treat specific attack/defense names, library APIs, and leaderboard
numbers as **verify against current docs**; treat arXiv IDs below as **verify** unless you can confirm them.

---

## 1. Mental model

A model is a function `f(x) → y` learned from data. An adversary manipulates **one of three surfaces**:

1. **Inference-time input** (`x`): craft an input that causes a wrong/targeted output → *evasion /
   adversarial examples*, and for LLMs *jailbreaks* and *prompt injection*.
2. **Training data** (`D`): corrupt the data so the *learned* `f` is wrong or contains a hidden behavior →
   *data poisoning*, *backdoors/trojans*.
3. **The model itself** (`f`, its parameters, or query access): steal it, recover its training data, or
   recover whether a record was in training → *model extraction*, *model inversion*, *membership inference*.

Robustness is **always relative to a threat model**. "Robust" with no threat model is meaningless. Before
any claim, pin down: attacker **goal**, attacker **knowledge** (white/black/gray-box), attacker
**capability** (what they can perturb and by how much — the *budget*), and attacker **access** (query rate,
training-pipeline access, ability to ship weights). A defense that holds under one threat model can be
trivially broken under another.

### Two frameworks to anchor on
- **NIST AI 100-2e2025 — *Adversarial Machine Learning: A Taxonomy and Terminology***
  (csrc.nist.gov/pubs/ai/100/2/e2025/final). The reference taxonomy. It organizes attacks by *stage*
  (training-time vs. deployment), *attacker goal* (availability, integrity, privacy, and for GenAI
  *misuse*), *knowledge*, and covers both predictive (classical) and generative (LLM) AI. Use its
  vocabulary in reports so everyone means the same thing. **Verify the exact section structure against
  the current PDF** — NIST revises this.
- **MITRE ATLAS** (atlas.mitre.org) — an ATT&CK-style knowledge base of real-world adversary
  **tactics and techniques** against ML systems, with case studies. Use it to *threat-model* and to map
  observed activity to named techniques (reconnaissance, ML model access, poisoning, evasion,
  exfiltration, impact). ATLAS is operational/lifecycle-oriented; NIST is the formal taxonomy. Cite both.

### Threat-model axes
| Axis | Options | Why it matters |
|------|---------|----------------|
| Knowledge | **White-box** (full weights + gradients), **gray-box** (architecture/some info, no weights), **black-box** (query-only: scores or labels) | Determines which attacks apply; white-box is the *worst case* you should evaluate against. |
| Goal | **Untargeted** (any wrong answer) vs. **targeted** (a specific wrong answer) | Targeted is harder for the attacker but higher impact. |
| Capability / budget | `L∞`, `L2`, `L0`, `L1` norm bound `ε`; or semantic/physical constraints | Defines the allowed perturbation set. No budget = no meaningful robustness claim. |
| Access | query rate, label-only vs. score, train-pipeline access, weight distribution | Governs extraction/poisoning/inversion feasibility. |

**Perturbation budgets (Lp norms).** For images, the canonical settings are `L∞` (every pixel may change by
at most `ε`, e.g. `8/255` on CIFAR-10, `4/255` on ImageNet) and `L2` (bounded Euclidean norm). `L0` bounds
the *number* of changed features (sparse/patch-like); `L1` is the convex relaxation. The norm is not the
threat — it is a *tractable proxy* for "looks the same to a human / is the same semantically." Real threats
(patches, rotations, weather, text paraphrase, audio) are often outside any `Lp` ball, so `Lp` robustness is
necessary-ish but never sufficient.

---

## 2. Attack categories (you defend what you understand)

### 2.1 Evasion / adversarial examples (inference-time, integrity)
Find a perturbation `δ` within the budget such that `f(x+δ)` is wrong. Foundational methods:

- **FGSM** (Fast Gradient Sign Method) — Goodfellow, Shlens, Szegedy, *Explaining and Harnessing
  Adversarial Examples*, arXiv:1412.6572. One step: `x' = x + ε·sign(∇ₓ L(f(x), y))`. Fast, weak; a
  defense that only survives FGSM is **not** robust.
- **PGD** (Projected Gradient Descent) — Madry et al., *Towards Deep Learning Models Resistant to
  Adversarial Attacks*, arXiv:1706.06083. Multi-step FGSM with random start, projecting back into the
  `ε`-ball each step. The de-facto **standard strong first-order attack** and the basis of adversarial
  training. Evaluate with enough steps and **multiple random restarts**.
- **C&W** (Carlini & Wagner) — *Towards Evaluating the Robustness of Neural Networks*, arXiv:1608.04644.
  Optimization-based, minimizes perturbation size subject to misclassification; broke many early defenses.
  Strong, especially in `L2`.
- **Transferability** — adversarial examples crafted on a *surrogate* model often fool a different,
  unseen model. This enables **black-box** attacks without gradients: train/obtain a surrogate, attack it,
  transfer. Why query-only access is still dangerous.
- **Physical / patch attacks** — perturbations that survive printing, camera capture, and viewing angle
  (adversarial patches, stickers on stop signs, eyeglass frames). These break the `Lp`-near assumption:
  the patch is large in pixel norm but semantically "an object in the scene." Defend with the *semantic*
  threat model, not just `Lp`.
- **Query-based black-box** — score-based (estimate gradients from confidence outputs) and decision-based
  (only hard labels, e.g. Boundary Attack / HopSkipJump). Rate-limiting and returning **coarse outputs**
  (top-1 label, no logits) raises the query cost — see Production (§5).

### 2.2 Data poisoning & backdoors/trojans (training-time)
The attacker influences training data (or a fine-tuning set, or RLHF preference data) to corrupt `f`.

- **Availability poisoning** — degrade overall accuracy (denial-of-quality).
- **Targeted poisoning** — cause specific inputs to be misclassified while overall accuracy looks fine,
  so it passes a naive eval.
- **Backdoors / trojans** — implant a hidden **trigger** (a pixel pattern, a watermark, a rare token
  phrase) such that any input containing the trigger is classified to the attacker's target, while
  clean-input accuracy is normal. Extremely stealthy by design.
- **Clean-label poisoning** — poison samples whose labels look *correct* to a human reviewer, so manual
  data audit doesn't catch them (e.g. feature-collision attacks). Harder to detect than mislabeled data.
- **For LLMs**: poisoning of pretraining/instruction/RLHF data, and **sleeper-agent / backdoored**
  behaviors triggered by a phrase. Web-scale scraped corpora are a real poisoning surface — see
  `[[pretraining-data-tokenizers]]`. Supply-chain poisoning (a tampered public dataset or a backdoored
  pretrained checkpoint you fine-tune from) is the most realistic enterprise vector — see §5.

### 2.3 Model extraction / stealing (confidentiality of the model)
Query the model enough to train a **functionally equivalent** copy, or recover hyperparameters/architecture.
Motivations: steal IP, then use the copy as a white-box surrogate to craft transferable evasion attacks, or
to mount membership-inference. Defenses: query rate limits, output truncation (top-1 not full softmax),
watermarking, anomaly detection on query distributions. Note these *raise cost*, they don't make extraction
impossible.

### 2.4 Model inversion (privacy of training data — reconstruction)
Use model access to **reconstruct representative inputs** of a class or features of training records
(e.g. reconstructing a recognizable face from a face-recognition model). High-capacity/overfit models leak
more. Mitigate with output coarsening, regularization, and **DP training** — see `[[privacy-preserving-ml]]`.

### 2.5 Membership inference (privacy of training data — membership)
Determine whether a specific record was in the training set. The canonical privacy attack and the empirical
yardstick for "did my model memorize?" Strong evaluation uses **likelihood-ratio / shadow-model** attacks
(e.g. LiRA — Carlini et al., *Membership Inference Attacks From First Principles*, arXiv:2112.03570) and
reports the **true-positive rate at low false-positive rate**, not just average AUC. Overfitting and
duplicated/rare records raise risk. The principled defense is **differentially private training (DP-SGD)**,
which gives a provable bound — see `[[privacy-preserving-ml]]`.

### 2.6 LLM-specific attacks
LLMs collapse several categories into the text/token surface:

- **Jailbreaks = evasion** on an aligned model — adversarial prompts (role-play, obfuscation, suffix
  attacks like GCG / *Universal and Transferable Adversarial Attacks on Aligned Language Models*,
  arXiv:2307.15043 — **verify**) that bypass safety training. Treat as the LLM analogue of adversarial
  examples; they **transfer** across models.
- **Prompt injection** — untrusted content (a retrieved doc, web page, tool output) carries instructions
  that hijack the model's behavior. **Indirect** prompt injection (instructions ride in via RAG/tools) is
  the dominant agent threat. This is largely an *application/architecture* problem: the **control/enforcement**
  belongs in `[[ai-security-on-gke]]` and `[[llm-app-agent-frameworks]]`; here we classify it as evasion of
  the instruction-following contract and red-team for it.
- **Training-data extraction / memorization** — prompting a model to regurgitate verbatim training data
  (PII, secrets, copyrighted text). Carlini et al., *Extracting Training Data from Large Language Models*,
  arXiv:2012.07805 (**verify**). Driven by memorization of duplicated/rare strings; mitigate with
  dedup (`[[pretraining-data-tokenizers]]`), DP, and output filtering.

---

## 3. Defenses and their limits

There is **no general-purpose solution** to adversarial examples. Every defense is a tradeoff and most are
specific to a threat model. Know what actually holds up.

### 3.1 Adversarial training (the strongest empirical defense)
Train on adversarial examples generated on-the-fly (Madry's PGD-based min-max formulation). This is the
**only widely-reproduced** empirical defense that survives strong adaptive evaluation. Costs and caveats:

- **Expensive**: 3–30× training cost (you run an inner PGD loop per batch). Cheaper variants exist
  (free/fast adversarial training) but can suffer *catastrophic overfitting* — verify they still hold under
  AutoAttack, not just PGD.
- **Robustness is budget-specific**: training at `L∞ ε=8/255` does not guarantee robustness at `L2` or at a
  larger `ε`. State the budget.
- **Accuracy cost**: lowers clean accuracy (the robustness–accuracy tradeoff, §4.4).
- Variants that help: TRADES (trades off clean vs. robust loss, arXiv:1901.08573 — **verify**), using
  extra/synthetic data, and weight averaging. Check **RobustBench** for the current top entries.

### 3.2 Certified / provable robustness (a guarantee, within its ball)
Instead of "we couldn't break it," produce a **mathematical certificate** that no perturbation within a
radius can change the prediction:

- **Randomized smoothing** — Cohen, Rosenfeld, Kolter, *Certified Adversarial Robustness via Randomized
  Smoothing*, arXiv:1902.02918. Add Gaussian noise at inference, classify by majority vote; yields a
  **probabilistic `L2` certificate** per input. Scales to ImageNet; cost is many forward passes per
  prediction. The most practical certified method today.
- **Interval Bound Propagation (IBP)** and other deterministic verifiers — propagate input intervals (or
  tighter convex relaxations) through the network to bound outputs. Sound but conservative; harder to scale
  to large nets. Active research; verify the state of the art.

Certified accuracy is **lower** than empirical robust accuracy and is **only** valid inside the certified
ball — outside it (patches, semantic shifts) you have no guarantee. Report certified radius and the fraction
of inputs certified.

### 3.3 Input transformation / detection (weak; use as defense-in-depth only)
JPEG compression, bit-depth reduction, denoising, feature squeezing, randomized resizing; or a separate
detector that flags adversarial inputs. **Most of these were broken** once attacked adaptively (the attacker
optimizes *through* the transformation, e.g. via BPDA — Backward Pass Differentiable Approximation, or EOT —
Expectation Over Transformations for randomized defenses). Treat detection/transforms as cheap
defense-in-depth that raises attacker cost, **never** as your robustness claim.

### 3.4 What gives FALSE security — gradient masking
This is the single biggest cause of broken "defenses" in the literature. A defense can lower the *measured*
attack success **not** by being robust but by **hiding/obscuring gradients** so first-order attacks fail to
find the adversarial example that still exists.

- **Defensive distillation** (train a second model on softened logits) — historically presented as a defense;
  **broken** by C&W. It masked gradients; it did not add robustness. Do not use it as a defense.
- **Obfuscated gradients** — Athalye, Carlini, Wagner, *Obfuscated Gradients Give a False Sense of Security*,
  arXiv:1802.00420 — showed most ICLR-2018 defenses relied on shattered/stochastic/exploding-or-vanishing
  gradients and fell to adaptive attacks (BPDA/EOT/transfer).
- **Tell-tale signs of gradient masking** (Carlini et al., *On Evaluating Adversarial Robustness*,
  arXiv:1902.06705 — **verify**): black-box attacks beat white-box; unbounded attacks don't reach ~100%
  success; increasing the perturbation budget doesn't increase attack success; random sampling finds
  adversarial examples the gradient attack missed; PGD with more steps doesn't help. **If you see these,
  your robustness number is fake.**

### 3.5 The arms race
Defense → broken → stronger defense → broken. Defenses are routinely defeated within months by adaptive
attacks designed *against that specific defense*. The practical consequence: **never trust a robustness claim
that wasn't evaluated by an adaptive attack and an ensemble like AutoAttack**, and re-evaluate when the
threat model changes. The durable wins so far are adversarial training and certified methods — everything
else is supplementary.

---

## 4. Robustness evaluation methodology — the real skill

A robustness number is a *claim about the strongest attack you tried*. Weak evaluation produces inflated
numbers and false confidence. This is where most teams (and many papers) go wrong.

### 4.1 Adaptive attacks (non-negotiable)
The attacker **knows your defense** and tailors the attack to it (white-box worst case). If your defense
masks gradients, use BPDA (replace the non-differentiable step with a differentiable approximation on the
backward pass); if it is randomized, use EOT (average gradients over the randomness); if gradients are
unreliable, add transfer and query-based attacks. A defense evaluated only against a fixed, defense-unaware
attack tells you nothing. Carlini et al., *On Evaluating Adversarial Robustness* (arXiv:1902.06705 —
**verify**) is the checklist; follow it.

### 4.2 AutoAttack (the standard automated baseline)
**AutoAttack** — Croce & Hein, *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse
Parameter-free Attacks*, arXiv:2003.01690. A **parameter-free** ensemble (APGD on CE loss, APGD on the
DLR loss, FAB, and the black-box Square Attack) that removes the "I tuned PGD badly" failure mode. It is the
**minimum bar** for an empirical robustness claim — but it is a *baseline*, not a substitute for a
defense-specific adaptive attack. A defense can still be over-estimated by AutoAttack if it has a quirk that
needs a custom adaptive attack. Run AutoAttack **and** a hand-crafted adaptive attack.

### 4.3 RobustBench (the leaderboard / reference)
**RobustBench** (robustbench.github.io) — a standardized benchmark and leaderboard of robust models,
evaluated with AutoAttack under fixed threat models (CIFAR-10/100, ImageNet; `L∞`/`L2`/common corruptions).
Use it to (a) get a strong **pretrained robust baseline** via its Model Zoo, (b) sanity-check your number
against the SOTA for the same threat model, and (c) avoid reinventing a broken eval. If your defense claims
to beat RobustBench SOTA by a wide margin, suspect your evaluation before you celebrate.

### 4.4 Report clean vs. robust accuracy + the tradeoff
Always report **both**: clean accuracy (no attack) and robust accuracy (under the strongest attack), with
the **exact threat model** (norm, `ε`, dataset, attack + steps + restarts). Robustness usually **costs**
clean accuracy — the **robustness–accuracy tradeoff** (Tsipras et al., *Robustness May Be at Odds with
Accuracy*, arXiv:1805.12152 — **verify**). A defense that improves robust accuracy while tanking clean
accuracy may be useless in production; state the operating point.

### 4.5 What a credible evaluation reports
- Threat model in full (knowledge, goal, norm, `ε`, query budget).
- Clean accuracy and robust accuracy under: AutoAttack **and** a defense-aware adaptive attack.
- PGD with multiple restarts and enough steps; show robust accuracy vs. number of steps **plateaus**.
- The gradient-masking sanity checks (§3.4) all pass.
- For certified defenses: certified radius / certified accuracy, not just empirical.
- For LLMs: which red-team suite, attack success rate, and that you tried *adaptive* jailbreaks.

### 4.6 Red-teaming for ML
Structured adversarial probing of a model/system for failures (harmful outputs, jailbreaks, prompt injection,
PII leakage, bias). Use automated attack suites **and** human red-teamers; track attack-success-rate over
time; feed findings back into training/filters. This sits at the intersection of evaluation and governance —
coordinate with `[[ml-evaluation-evals]]` for the eval harness/methodology and `[[responsible-ai-governance]]`
for policy, documentation (model cards / risk reports), and sign-off. Red-teaming is necessary but **not a
robustness proof** — absence of found failures is not robustness.

---

## 5. Production — defending the real pipeline

Robustness is a **system property**, not just a trained-model property. Place defenses across the lifecycle;
the *enforcement mechanisms* (gateways, sandboxes, network policy, signing) are detailed in
`[[ai-security-on-gke]]` — here is the adversarial-ML view of where and what.

### 5.1 Where in the pipeline to defend
| Stage | Threat | Defense (model-level) |
|-------|--------|------------------------|
| Data ingest / labeling | poisoning, backdoors, clean-label | provenance + integrity (hash/sign datasets), dedup, anomaly/outlier filtering, trusted-curation, spectral-signature / activation-clustering backdoor scans |
| Pretrained-weight reuse | backdoored/trojaned checkpoint | verify provenance + signature; prefer `safetensors` over pickle; backdoor scan; fine-tune+evaluate on a trusted held-out set with trigger probes |
| Training | poisoning, privacy leakage | robust training, DP-SGD where privacy matters (`[[privacy-preserving-ml]]`), adversarial training for robustness |
| Pre-deploy eval | inflated robustness, residual backdoor | AutoAttack + adaptive eval, trigger/backdoor scan, membership-inference audit, red-team |
| Inference | evasion, extraction, inversion, jailbreak | input detection (defense-in-depth), rate limits, output coarsening, monitoring; guardrails at the gateway (`[[ai-security-on-gke]]`) |
| Monitoring | ongoing attack | attack-signature & distribution-shift monitoring (`[[ml-observability-monitoring]]`) |

### 5.2 Monitoring for attack signatures
- **Query-pattern anomalies** — bursts of near-duplicate or boundary-probing queries (extraction /
  decision-based attacks); uniform sweeps of input space; sudden spikes in low-confidence predictions.
- **Input anomalies** — out-of-distribution / high-frequency-noise inputs (possible evasion); known
  trigger patterns.
- **Output anomalies** — confidence collapse, sudden class-distribution shift, jailbreak/PII patterns in
  LLM outputs.
- Wire these into your ML observability stack and alerting — see `[[ml-observability-monitoring]]`. Coarse
  outputs (top-1 label, no logits) and per-caller rate limits both *raise the attacker's cost*.

### 5.3 Supply chain (the most realistic enterprise vector)
You rarely train from scratch; you fine-tune a public checkpoint on a public/mixed dataset. Both can be
poisoned **before you ever see them**.
- **Datasets**: verify source and integrity (signed hashes), dedup, and scan; web-scraped corpora are an
  open poisoning surface (`[[pretraining-data-tokenizers]]`).
- **Weights**: pull from trusted registries, **verify signatures/provenance** (SLSA, Sigstore — enforced via
  `[[ai-security-on-gke]]`), prefer `safetensors` (pickle/`torch.load` executes arbitrary code on load),
  and **backdoor-scan + trigger-probe** before trusting a third-party checkpoint.

### 5.4 Incident response
When an attack is detected/suspected: capture the offending inputs/queries; classify against **MITRE ATLAS**
techniques; estimate blast radius (which model versions/datasets are affected); contain (rate-limit/block the
source, roll back to a clean model/dataset version, rotate exposed credentials); remediate (retrain on
cleaned data, re-evaluate with adaptive attacks before redeploy); document for governance
(`[[responsible-ai-governance]]`). Versioned datasets and models make rollback and root-cause possible —
without lineage you cannot scope a poisoning incident.

---

## 6. Anti-patterns (these are the ones that bite)

- **No threat model.** Claiming "robust" with no specified knowledge/goal/budget. The claim is unfalsifiable
  and almost certainly wrong.
- **Evaluating against weak or non-adaptive attacks.** FGSM-only, a single low-step PGD, or any attack that
  doesn't *know* your defense. The number will be inflated.
- **Claiming robustness without AutoAttack + an adaptive attack.** The minimum bar is both; AutoAttack alone
  can still over-estimate a quirky defense.
- **Gradient-masking defenses.** Defensive distillation, non-differentiable preprocessing, randomization that
  obscures gradients — they lower the *measured* attack, not the real one. Run the §3.4 sanity checks.
- **Trusting input transforms/detectors as the defense.** They fall to BPDA/EOT; use only as defense-in-depth.
- **Ignoring poisoning in third-party data/weights.** Fine-tuning from an unverified checkpoint or training on
  an unscanned scraped dataset; no integrity, no backdoor scan, loading pickle weights.
- **Reporting only robust accuracy (or only clean).** Hiding the robustness–accuracy tradeoff.
- **Conflating empirical and certified robustness.** "We couldn't break it" ≠ "provably unbreakable in this
  ball." State which one you have.
- **`Lp`-only thinking for real-world threats.** Patches, physical, and semantic/text attacks live outside the
  `Lp` ball; an `L∞`-robust model can still fail to a sticker.
- **Treating red-team "no findings" as a robustness proof.** Absence of evidence is not evidence of robustness.
- **No monitoring / no IR plan.** Robustness is not one-and-done; attacks evolve and so must evaluation.

---

## Rationalizations & rebuttals

Excuses used to skip honest robustness work, each rebutted:

- **"We evaluated it and it held up."** Against what? A non-adaptive or FGSM/low-step PGD attack proves
  nothing — those numbers are inflated by construction. A robustness number is only a claim about the
  *strongest attack you tried*; the attacker will tailor theirs to your defense (§4.1). No adaptive attack →
  no claim.
- **"The attack success rate is basically zero, so it's robust."** That is the classic gradient-masking
  signature, not robustness. Run the §3.4 sanity checks: if black-box beats white-box, unbounded attacks
  don't reach ~100%, or more PGD steps don't help, your number is fake and the adversarial examples still
  exist.
- **"Poisoning won't happen to us — our data is fine."** You almost never train from scratch; you fine-tune a
  public checkpoint on a public/mixed corpus, and both can be poisoned *before you ever see them* (§5.3).
  Clean-label and backdoor poisoning are designed to leave clean-input accuracy normal, so "it trains fine"
  is exactly what a successful attack looks like.
- **"Clean accuracy is what the product needs; robust accuracy is academic."** Report both or you are hiding
  the robustness–accuracy tradeoff (§4.4). A model with great clean accuracy can be flipped by a sticker
  (physical/patch attacks, §2.1) or a trigger phrase (backdoor) — that is a production failure, not an
  academic one.
- **"The checkpoint is from a reputable hub, so the weights are safe."** Provenance ≠ integrity. A trojaned
  checkpoint behaves normally until the trigger fires, and `pickle`/`torch.load` executes arbitrary code on
  load. Verify signatures, prefer `safetensors`, and backdoor-scan + trigger-probe before trusting it (§5.3).
- **"Our input filter / detector / JPEG step stops the attack."** These fall to adaptive attacks (BPDA for
  non-differentiable steps, EOT for randomized ones, §3.3). They raise attacker cost as defense-in-depth;
  they are never the robustness claim.
- **"Red-teaming found nothing, so we're good."** Absence of found failures is not robustness (§4.6). A
  red-team result is a lower bound on vulnerability, and it ages the moment a new adaptive jailbreak or
  transfer attack appears.

## Red flags

Stop and reconsider if you see any of these:

- **No threat model.** "Robust" with no stated knowledge (white/black-box), goal, norm + budget `ε`, or
  query/pipeline access. The claim is unfalsifiable (§1).
- **No adaptive attack and no AutoAttack.** Evaluation is FGSM-only or a single low-step PGD; the defense was
  never attacked by something that *knows* the defense (§4.1–4.2).
- **Gradient-masking signatures present.** Black-box beats white-box, unbounded attacks don't reach ~100%
  success, larger `ε` or more PGD steps don't increase attack success, random sampling finds examples the
  gradient attack missed (§3.4).
- **A known gradient-masking defense is the centerpiece.** Defensive distillation, non-differentiable
  preprocessing, or stochastic obfuscation presented as *the* defense rather than defense-in-depth.
- **No RobustBench-style sanity check.** A robustness number reported in isolation, especially one that beats
  RobustBench SOTA for the same threat model by a wide margin — suspect the evaluation first (§4.3).
- **Data/weight supply chain ignored.** Fine-tuning from an unverified checkpoint, training on an unscanned
  scraped dataset, loading pickle weights, no signed hashes, no backdoor scan (§5.3).
- **Only one accuracy reported.** Robust accuracy without clean (or vice versa) — the tradeoff is being
  hidden (§4.4).
- **No monitoring or IR plan.** No attack-signature/distribution-shift monitoring, no dataset/model lineage
  to scope a poisoning incident or roll back (§5.2, §5.4).

## Verification gate (definition of done)

Robustness work is done only when all of these are true and shown:

- [ ] **Threat model written in full** — attacker knowledge (white/gray/black-box), goal (targeted vs.
  untargeted), capability/budget (norm + `ε`, or the semantic/physical constraint), and access (query rate,
  pipeline/weight access). Mapped to NIST AI 100-2e2025 / MITRE ATLAS vocabulary.
- [ ] **Adaptive, defense-aware attack run** — tailored to the defense (BPDA for non-differentiable steps,
  EOT for randomized ones, transfer/query for unreliable gradients), per the §4.1 checklist.
- [ ] **AutoAttack run** as the parameter-free baseline, in addition to the adaptive attack (§4.2).
- [ ] **Gradient-masking sanity checks pass** — PGD robust accuracy plateaus with more steps/restarts,
  white-box ≥ black-box, unbounded attack ≈ 100%, larger `ε` increases attack success (§3.4).
- [ ] **Clean and robust accuracy both reported** with the exact threat model (norm, `ε`, dataset, attack +
  steps + restarts); for certified defenses, certified radius / certified accuracy too (§4.4–4.5).
- [ ] **Result sanity-checked against RobustBench** SOTA for the same threat model (§4.3).
- [ ] **Poisoning/backdoor checks on third-party data and weights** — provenance + signed-hash integrity,
  dedup, backdoor scan (spectral-signature / activation-clustering) and trigger probes on a trusted held-out
  set; `safetensors` over pickle (§5.1, §5.3).
- [ ] **Privacy audit where relevant** — membership-inference (LiRA-style, TPR @ low FPR) and DP where
  required (§2.5).
- [ ] **Monitoring + IR in place** — attack-signature and distribution-shift monitoring wired to alerting;
  versioned datasets/models for rollback and root-cause; an incident-response path (§5.2, §5.4).

## 7. Canonical references (verify against current versions)

- **NIST AI 100-2e2025** — Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and
  Mitigations. csrc.nist.gov/pubs/ai/100/2/e2025/final  *(verify edition/section structure)*
- **MITRE ATLAS** — atlas.mitre.org (tactics, techniques, case studies; ATLAS Navigator).
- **RobustBench** — robustbench.github.io (leaderboard + Model Zoo of robust models).
- **AutoAttack** — Croce & Hein 2020, arXiv:2003.01690; code: github.com/fra31/auto-attack.
- **FGSM** — Goodfellow, Shlens, Szegedy 2014, arXiv:1412.6572.
- **PGD / adversarial training** — Madry et al. 2017, arXiv:1706.06083.
- **C&W** — Carlini & Wagner 2016, arXiv:1608.04644.
- **Obfuscated gradients** — Athalye, Carlini, Wagner 2018, arXiv:1802.00420.
- **On Evaluating Adversarial Robustness** — Carlini et al. 2019, arXiv:1902.06705 *(verify)*.
- **Randomized smoothing** — Cohen, Rosenfeld, Kolter 2019, arXiv:1902.02918.
- **TRADES** — Zhang et al. 2019, arXiv:1901.08573 *(verify)*.
- **Robustness vs. accuracy** — Tsipras et al. 2018, arXiv:1805.12152 *(verify)*.
- **Membership inference (LiRA)** — Carlini et al. 2021, arXiv:2112.03570 *(verify)*.
- **Training-data extraction (LLMs)** — Carlini et al. 2020, arXiv:2012.07805 *(verify)*.
- **GCG / transferable LLM attacks** — Zou et al. 2023, arXiv:2307.15043 *(verify)*.
- **Tooling** — Adversarial Robustness Toolbox (ART), CleverHans, Foolbox, TextAttack (LLM/NLP). Verify the
  current maintained version and API before use.
</content>
</invoke>

---

# Adversarial ML & Robustness — Worked Examples

Three artifacts to imitate: an **adaptive robustness-evaluation checklist** (with AutoAttack), a
**backdoor/poisoning detection note**, and a **threat-model template**. These are *defensive* — you run them
to honestly assess and harden your own model. Verify library APIs, attack/defense names, leaderboard numbers,
and arXiv IDs against **current docs** (it is 2026; the field moves fast).

---

## 1. Adaptive robustness-evaluation checklist (with AutoAttack)

The goal is an **honest** robust-accuracy number. AutoAttack is the minimum automated baseline; a
defense-aware **adaptive** attack is still required. Work top-to-bottom; do not report a number until every
sanity check passes.

### Step 0 — State the threat model (see §3 below). No threat model, no claim.

### Step 1 — Establish baselines
- Clean accuracy (no attack) on the test set.
- A strong **non-robust** reference and, if defending images, a **RobustBench** model for the same threat
  model — so you know what SOTA robust accuracy looks like before trusting your own.

### Step 2 — Run AutoAttack (parameter-free ensemble)
```python
# pip install autoattack   (verify the current package/version & API)
# AutoAttack = APGD-CE + APGD-DLR (targeted) + FAB + Square (black-box). Parameter-free by design.
from autoattack import AutoAttack

# IMPORTANT: model must output LOGITS and include any inference-time preprocessing that is part of
# the defense (smoothing noise, transforms) INSIDE forward(), or you are evaluating the wrong function.
model.eval()
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')   # state norm + eps explicitly
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)
# robust accuracy = accuracy of model on x_adv
```
- Use the **`standard`** version for a headline number. For randomized/EOT defenses use the **`rand`**
  version (averages over randomness) — the `standard` version is unsound on stochastic models.
- Report per-attack success so you can see which component dominates.

### Step 3 — Run a defense-AWARE adaptive attack (the part papers skip)
AutoAttack is generic. If your defense has any of the following, hand-craft the attack:
- **Non-differentiable / quantized / preprocessing step** → **BPDA**: replace it with a differentiable
  approximation (often identity) on the backward pass so PGD gets a usable gradient.
- **Randomization** (noise, random resize/crop) → **EOT**: average the gradient over many samples of the
  randomness each step.
- **Detector / reject option** → attack the *combined* (classifier + detector) objective, not the classifier
  alone.
- **Multiple components** → attack them jointly; an attacker who knows the defense optimizes end-to-end.

### Step 4 — PGD diligence
- Multiple **random restarts** (e.g. 5–20) and enough **steps** that robust accuracy **plateaus** (plot it).
- Try both CE and a margin/DLR loss; gradient-free **Square Attack** as a non-gradient cross-check.

### Step 5 — Gradient-masking sanity checks (ALL must pass, else the number is fake)
| Check | Expected (no masking) | Red flag (masking → fake robustness) |
|-------|----------------------|--------------------------------------|
| Black-box vs. white-box | white-box ≥ black-box success | black-box **beats** white-box |
| Unbounded attack (`ε`→large) | success → ~100% | success stalls below 100% |
| Increasing `ε` | success monotonically rises | success flat as `ε` grows |
| More PGD steps | success rises then plateaus | more steps don't help at all |
| Random/brute-force sampling | finds ≤ what gradient attack finds | finds adversarials the gradient attack missed |
| Transfer from a surrogate | ≤ direct white-box | transfer **beats** direct attack |

(Sanity checks per Carlini et al., *On Evaluating Adversarial Robustness*, arXiv:1902.06705 — **verify**.)

### Step 6 — Report (the honest record)
- Threat model in full (knowledge, goal, **norm + `ε`**, dataset, query budget).
- **Clean accuracy AND robust accuracy** under AutoAttack **and** the adaptive attack (lowest wins).
- PGD steps/restarts and the plateau plot; which sanity checks were run.
- For **certified** defenses: report **certified accuracy / radius** separately — empirical ≠ certified.
- The **robustness–accuracy tradeoff**: state the clean-accuracy cost and the chosen operating point.

> Rule of thumb: if your robust accuracy beats RobustBench SOTA for the same threat model by a wide margin,
> **suspect your evaluation first** (almost always gradient masking or a mis-specified attack), not a
> breakthrough.

---

## 2. Backdoor / poisoning detection note (defensive)

You usually inherit risk from **third-party datasets and pretrained checkpoints**. Assume both *could* be
poisoned; verify before you trust. A **backdoor/trojan** keeps clean accuracy normal while a hidden **trigger**
forces a target output — so standard validation accuracy will **not** catch it.

### When to run
- Before fine-tuning **from a third-party checkpoint**.
- Before training on a **scraped / mixed / externally-sourced dataset**.
- After any **data-pipeline change** or supplier change.

### Provenance & integrity first (cheapest, catches most supply-chain tampering)
- Pull weights/data from trusted registries; **verify signatures / SLSA provenance** (enforced via
  `[[ai-security-on-gke]]`).
- Prefer **`safetensors`** over pickle — `torch.load`/pickle executes arbitrary code on load.
- Pin dataset **hashes**; reject on mismatch. Track lineage so an incident is scope-able (§ guide 5.4).

### Data-side detection
- **Dedup** and outlier/anomaly filtering on inputs and labels (`[[pretraining-data-tokenizers]]`).
- **Clean-label** poisons look correctly labeled to a human — manual review alone is insufficient; combine
  with feature-space outlier detection.
- Inspect class-conditional feature clusters for off-distribution sub-clusters.

### Model-side detection (does this checkpoint have a hidden trigger?)
- **Activation clustering** — cluster penultimate-layer activations *per class*; a backdoored class often
  splits into two clusters (clean vs. triggered).
- **Spectral signatures** — poisoned examples leave a detectable signature in the covariance spectrum of a
  class's representations.
- **Trigger reverse-engineering** (Neural-Cleanse–style) — for each target class, optimize the *smallest*
  perturbation that flips *all* inputs to that class; an anomalously tiny trigger for one class indicates a
  backdoor. (Names/methods — **verify current tooling**, e.g. ART has implementations.)
- **Trigger probing** — evaluate on a held-out **trusted** set and on inputs stamped with candidate triggers
  (known watermark patterns / rare token phrases for LLMs); compare target-class rates.

### On detection
Quarantine the dataset/checkpoint; classify against **MITRE ATLAS**; retrain on cleaned data or switch to a
verified source; **re-run the §1 evaluation** before redeploy; document for `[[responsible-ai-governance]]`.
Detection methods are imperfect — adaptive poisoning can evade them — so layer provenance + multiple
detectors; do not rely on any single scan.

---

## 3. Threat-model template (fill in before any robustness work)

A robustness claim is only meaningful relative to a written threat model. Copy and complete:

```yaml
threat_model:
  system:               # what is defended, e.g. "image classifier f, ResNet-50, CIFAR-10, deployed as API"
  asset:                # what we protect: prediction integrity | training-data privacy | model IP | availability

  attacker_goal:        # untargeted (any wrong output) | targeted (specific output) | extract | invert | infer-membership | jailbreak
  attacker_knowledge:   # white-box (weights+gradients) | gray-box (architecture/partial) | black-box (query-only)
  attacker_access:
    query_interface:    # none | hard-label only | top-k labels | full logits/softmax
    query_budget:       # max queries (e.g. 10k) or rate limit; or "unbounded"
    train_pipeline:     # none | can inject training/fine-tune/RLHF data | can ship a pretrained checkpoint
    weight_access:      # none | can read | can modify
  attacker_capability:  # the allowed perturbation set
    surface:            # inference input | training data | model parameters
    constraint:         # Lp ball (give norm) | semantic | physical (patch/print) | text paraphrase
    norm:               # Linf | L2 | L0 | L1 | n/a
    epsilon:            # e.g. 8/255 (Linf, CIFAR-10) | 0.5 (L2) | n/a
    notes:              # e.g. "patch up to 5% of pixels, any value" if outside an Lp ball

  out_of_scope:         # explicitly excluded threats (e.g. "physical attacks", "white-box not assumed")

  evaluation:
    attacks:            # MUST include AutoAttack + a defense-aware adaptive attack (BPDA/EOT as needed)
    pgd_steps_restarts: # e.g. "100 steps, 10 restarts, plateau confirmed"
    reference:          # RobustBench entry for the same threat model
    metrics:            # clean accuracy; robust accuracy (lowest over all attacks); certified radius if any
    sanity_checks:      # gradient-masking checks from §1 step 5 — list results

  atlas_techniques:     # MITRE ATLAS technique IDs this model maps to (for monitoring + IR)
  references:           # NIST AI 100-2e2025 section(s); papers; tooling versions used
```

Worked example (one line per field, abbreviated):
- system: ResNet-50 image classifier, ImageNet, served behind an API returning top-1 label only.
- asset: prediction integrity. goal: untargeted evasion. knowledge: black-box (label-only), but evaluate
  white-box worst case too. access: 10k queries/day, no pipeline/weight access.
- capability: `Linf`, `ε = 4/255`; **plus** an out-of-`Lp` adversarial-patch case (≤ 5% pixels, any value).
- out_of_scope: poisoning (separate supply-chain review per §2).
- evaluation: AutoAttack (`Linf`, `4/255`, standard) + BPDA-adaptive on the JPEG-preprocessing step; PGD
  100×10, plateau confirmed; compared to RobustBench ImageNet `Linf` entry; all gradient-masking checks pass.

---

### References (verify against current versions)
NIST AI 100-2e2025 (csrc.nist.gov/pubs/ai/100/2/e2025/final) · MITRE ATLAS (atlas.mitre.org) · RobustBench
(robustbench.github.io) · AutoAttack — Croce & Hein, arXiv:2003.01690 (github.com/fra31/auto-attack) ·
On Evaluating Adversarial Robustness — Carlini et al., arXiv:1902.06705 *(verify)* · PGD/adv. training —
Madry et al., arXiv:1706.06083 · Adversarial Robustness Toolbox (ART) for backdoor-detection implementations
*(verify current version)*.
</content>
