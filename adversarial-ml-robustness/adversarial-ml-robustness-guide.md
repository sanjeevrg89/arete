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
