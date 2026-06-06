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
