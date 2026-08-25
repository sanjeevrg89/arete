# AGENTS.md — Privacy-Preserving ML (PETs) Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`privacy-preserving-ml-guide.md`** next to this file —
> read it before designing or reviewing any privacy-preserving training/serving, and apply it. Concrete
> artifacts to imitate (DP-SGD config, FL + secure-aggregation round, PET-selection table) are in
> **`examples.md`**. This file is the always-on summary.
>
> This is the **engineering** layer (the PET mechanisms). Policy/fairness/model-cards = [[responsible-ai-
> governance]]; runtime/infra hardening + confidential infra = [[ai-security-on-gke]]; the leakage attacks
> as an adversarial discipline = [[adversarial-ml-robustness]].
>
> **It is 2026 and this field moves fast.** Verify accountant math, library APIs, production ε/ρ numbers,
> and *especially* machine-unlearning methods/benchmarks against current docs. Never quote a production
> privacy number from memory; never fabricate an arXiv ID.

## When doing privacy-preserving ML, apply these by default:

- **A privacy claim = a threat model + a defensible number.** State *what* is private (record vs user),
  *who* the adversary is and *what they see* (final model / white-box weights / per-round gradients /
  the server), and *the guarantee* (formal (ε,δ)/zCDP ρ, a crypto assumption, or a hardware root of
  trust). "We don't store raw data" is an architecture, not a guarantee.
- **De-identification ≠ privacy.** Never call data "anonymized" as a guarantee without DP. Linkage and
  reconstruction defeat identifier-dropping; only a formal mechanism bounds the leak.
- **Models leak — that's why PETs exist.** Membership inference, model/gradient inversion, and
  verbatim memorization/extraction (LLMs) are real. DP is the principled, provable defense.
- **DP-SGD = per-example gradient clipping (bound sensitivity) + calibrated Gaussian noise + accounting.**
  You MUST track spent (ε,δ) with a real accountant (RDP / moments / PRV). **Noise without accounting is
  not DP.** Use Opacus / TF Privacy / JAX DP libs. Expect a per-sample-gradient compute tax.
- **Recover DP utility by:** pretrain non-privately on public data → **DP fine-tune** (biggest lever);
  large batch sizes; tune clip norm `C`; PEFT under DP ([[fine-tuning-peft]]). Choose ε deliberately —
  single-digit ε is meaningful; tens-to-hundreds is mostly marketing.
- **Central vs local DP:** central = trusted curator, best utility; local = on-device noise, no trust,
  much worse utility. **DP-FTRL** gives formal DP in FL without Poisson client sampling — the basis for
  shipped production DP keyboard models (reported in ρ-zCDP; verify exact numbers).
- **Federated learning keeps raw data on device/silo; only updates move (FedAvg).** Cross-device
  (millions of unreliable phones, **non-IID**, stragglers) vs cross-silo (few trusted orgs). Handle
  non-IID with FedProx/SCAFFOLD/FedAdam; compress updates (quantize/sparsify) for communication.
- **FL is NOT private by itself.** The server sees individual updates and **gradient inversion**
  reconstructs client data. Add **secure aggregation** (server learns only the *sum*, dropout-tolerant)
  **and** client-level DP. The production stack is **FedAvg + SecAgg + DP (DP-FTRL accounting)**.
- **Cryptographic PETs cost a lot — use only when the threat model demands it.** HE = compute on
  ciphertext (good for inference, brutal for deep training); MPC = split across non-colluding parties
  (interactive, communication-bound); PSI = intersect sets without revealing non-members. **TEEs /
  confidential computing** protect data in use behind hardware + attestation — near-native speed, but
  trusts the vendor and has a side-channel history ([[gke-master]], [[ai-security-on-gke]]).
- **Machine unlearning is GDPR/CCPA-driven and immature.** Exact (SISA: shard + retrain one shard) is
  sound but costly; approximate is fast but its **verification/benchmarks are unreliable**. **Never trust
  an unlearning claim without an audit;** design for forgetting via lineage + retention + DP. Fast-moving.
- **Budget privacy over time.** ε is spent across releases/queries/retrains and **composes** — keep a
  privacy ledger, cap it, account at the right unit. **Unbounded reuse of the same ε destroys the
  guarantee.**

## Anti-patterns (reject these)
- "Anonymized" with no DP · training on sensitive data with **no privacy accounting** · FL without secure
  aggregation · unbounded privacy-budget reuse · trusting unlearning without verification · HE/MPC where
  the cost isn't justified · tiny ε but white-box weights shared widely · treating synthetic data as
  automatically private.

## Definition of done for a privacy-preserving ML change
A written threat model (what/who/guarantee) · a **formal number** (ε,δ / ρ) with the **accountant named**
and the budget ledger updated · the PET justified against the constraint (and cheaper options ruled out
explicitly) · SecAgg present if FL · an unlearning/deletion path if RTBF applies (with a verification plan)
· fast-moving claims flagged "verify current." Report honestly if any are missing.
