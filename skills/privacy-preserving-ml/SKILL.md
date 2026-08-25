---
name: privacy-preserving-ml
description: Privacy-enhancing technologies (PETs) for machine learning — the engineering techniques to
  train and serve models without leaking private training data. Use when you must train/fine-tune/serve
  on sensitive data (PII, PHI, financial, on-device), when facing membership-inference / model-inversion
  / training-data-extraction (memorization) attacks, or when a requirement says "private", "anonymized",
  "GDPR/CCPA", "right-to-be-forgotten", "data can't leave the device/silo", or "no raw data sharing".
  Covers differential privacy (the (ε,δ) definition, DP-SGD gradient clipping + noise, privacy accounting
  via RDP/moments accountant/PRV, DP-FTRL, DP fine-tuning & synthetic data), federated learning (FedAvg,
  cross-device vs cross-silo, non-IID, secure aggregation, FL+DP composition, TFF/Flower/FedML/PySyft),
  cryptographic PETs (homomorphic encryption, secure multiparty computation, private set intersection,
  trusted execution environments / confidential computing), and machine unlearning (SISA, exact vs
  approximate, verification). The engineering layer — distinct from policy ([[responsible-ai-governance]])
  and infra ([[ai-security-on-gke]]).
---

# Privacy-Preserving Machine Learning

Apply the judgment of an ML privacy engineer who has shipped models trained with **formal** privacy
guarantees in production — not someone who removed a name column and called it "anonymized." A privacy
claim is only as good as its **threat model + a number you can defend**: an ε, a (ε,δ), a zCDP ρ, a
cryptographic assumption, or a hardware root of trust. "We don't store raw data" is an architecture, not
a guarantee.

> **This field moves fast (it is 2026).** DP-SGD tooling, accountants (RDP → PRV), FL frameworks, and
> *especially* machine-unlearning methods and benchmarks change quickly and unlearning verification is
> still **immature and unreliable**. Verify accountant math, library APIs, and any unlearning claim
> against **current** docs and a fresh threat model before you trust them.

This is the **engineering** skill: the PET mechanisms themselves. The policy/fairness/model-card layer is
[[responsible-ai-governance]]; the runtime/infra hardening (TEEs on a cluster, model-theft defense) is
[[ai-security-on-gke]]; the attacks as adversarial robustness live in [[adversarial-ml-robustness]].

## How to use this skill

1. **Read `privacy-preserving-ml-guide.md`** in this directory — the full reference (the attacks that
   motivate PETs, DP & DP-SGD, federated learning, cryptographic PETs, unlearning, production guidance,
   anti-patterns). Apply it to the task.
2. For artifacts to imitate, read **`examples.md`**: a DP-SGD config sketch with an ε-accounting note,
   an FL + secure-aggregation round outline, and a PET-selection decision table.
3. Pin the **threat model first** (who is the adversary, what do they observe — the model, gradients,
   APIs?), then pick the PET, then commit to a defensible number. Match existing framework/infra
   conventions; apply the privacy-accounting discipline regardless.

## Essentials (full detail in `privacy-preserving-ml-guide.md`)

- **PETs exist because models leak.** Membership inference (was this record in training?), model
  inversion (reconstruct features/faces), and training-data extraction — LLMs **verbatim-memorize and
  regurgitate** rare training strings. De-identification ≠ privacy; only a formal guarantee bounds the leak.
- **Differential privacy is the gold standard.** (ε,δ)-DP bounds how much any *one* record can change the
  output distribution. Smaller ε = more private, less useful. DP **composes** and is **immune to
  post-processing** — the two properties that make it engineering-grade. Pick ε deliberately; ε in the
  single digits is meaningfully private, ε in the tens-to-hundreds is mostly a marketing number.
- **DP-SGD = per-example gradient clipping (bound sensitivity) + calibrated Gaussian noise + accounting.**
  You MUST track the spent budget with a real accountant (**RDP / moments accountant / PRV**) — noise
  without accounting is not DP. Opacus (PyTorch) / TF Privacy / JAX. Expect a utility hit and big compute
  overhead (per-sample grads); large batches + pretraining + DP fine-tuning recover most of it.
- **Central vs local DP.** Central DP trusts a curator to add noise once (best utility); local DP adds
  noise on-device before anything leaves (no trust, much worse utility). **DP-FTRL** gives formal DP in
  FL **without** Poisson client sampling — it underpins production DP language models (mobile keyboard
  next-word prediction shipped with formal zCDP guarantees; *verify current numbers*).
- **Federated learning keeps raw data on the device/silo; only updates move.** **FedAvg** = local SGD +
  server averaging. Cross-device (millions of unreliable phones, **non-IID**, stragglers) vs cross-silo
  (few trusted orgs, e.g. hospitals). FL is **not** private by itself — updates leak; the server still
  sees them.
- **FL needs secure aggregation + DP to actually be private.** **Secure aggregation** lets the server
  learn only the *sum* of updates, never any individual one (masks cancel on summation). Compose with DP
  for a formal guarantee. SecAgg adds rounds/communication and tolerates dropouts by design.
- **Cryptographic PETs trade huge cost for strong guarantees.** **HE** computes on ciphertext (great for
  linear/inference, brutal for deep training); **MPC** splits computation across non-colluding parties
  (communication-bound); **PSI** intersects datasets without revealing non-members. Use only when the
  threat model demands it — they are orders of magnitude slower.
- **TEEs / confidential computing** (SGX/TDX, confidential GPUs) protect data **in use** behind a hardware
  root of trust + remote attestation — cheap relative to HE/MPC, but trusts the vendor and has a real
  side-channel history. Infra specifics: [[gke-master]], [[ai-security-on-gke]].
- **Machine unlearning is driven by the right-to-be-forgotten (GDPR/CCPA) — and is immature.** Exact
  unlearning (**SISA**: shard/isolate/slice so you retrain only one shard) is sound but costly; approximate
  unlearning is faster but its **verification/benchmarks are unreliable** — never trust an unlearning claim
  without an audit, and prefer DP/retention design where you can. **Flag as fast-moving.**
- **Compose deliberately and budget over time.** FL + SecAgg + DP is the canonical production stack. A
  privacy budget is **spent** across releases/queries/retrains — track it; **unbounded reuse of the same ε
  destroys the guarantee.** Choose the PET from the constraint (regulatory / multi-party / on-device).
- **Anti-patterns:** calling data "anonymized" with no DP; training on sensitive data with **no privacy
  accounting**; FL without secure aggregation; reusing/refreshing the privacy budget without bounding it;
  trusting unlearning without verification; reaching for HE/MPC where the cost isn't justified.

## Related skills

- `[[responsible-ai-governance]]` — the policy/fairness/compliance layer (model cards, DP/FL/unlearning at
  concept level, consent, provenance). This skill is the engineering mechanism behind those controls.
- `[[ai-security-on-gke]]` — runtime/infra hardening, model theft, confidential computing on a cluster.
- `[[adversarial-ml-robustness]]` — the membership-inference / model-inversion / extraction attacks as an
  adversarial discipline (and DP as a defense).
- `[[edge-on-device-ml]]` — on-device training/inference where local DP and cross-device FL live.
- `[[fine-tuning-peft]]` — DP fine-tuning and DP synthetic-data generation build on PEFT mechanics.
- `[[gke-master]]` — provisioning confidential VMs / TEE node pools and attestation for FL servers.
