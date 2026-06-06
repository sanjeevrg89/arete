# Privacy-Preserving ML — The Reference

The engineering techniques (privacy-enhancing technologies, PETs) to train and serve models without
leaking the private data they learned from. Scope: the **mechanisms** — differential privacy, federated
learning, cryptographic PETs, machine unlearning. Out of scope (covered by siblings): governance/policy
and model cards ([[responsible-ai-governance]]); cluster/runtime hardening and confidential infra
([[ai-security-on-gke]]); the attacks as an adversarial discipline ([[adversarial-ml-robustness]]).

> **Version awareness (2026).** Accountants (RDP → PRV), DP-SGD libraries, FL frameworks, and *especially*
> machine-unlearning methods/benchmarks move fast. Verify accountant math, library APIs, and unlearning
> claims against current docs. Don't quote a production ε/ρ from memory — verify the source.

---

## 1. Mental model: a privacy claim is a threat model + a defensible number

Privacy is not a feature you bolt on; it is a **quantified bound under a stated adversary**. Before any
PET, answer:

- **What is private?** A record? A user (many records)? A feature? "User-level" DP is much stronger and
  harder than "record-level."
- **Who is the adversary and what do they see?** Only the final model (black-box API)? White-box weights?
  Per-round gradients/updates? The server? Other participants? Each observation surface is a leak channel.
- **What's the guarantee?** A formal one — (ε,δ)-DP, zCDP ρ, a cryptographic hardness assumption, or a
  hardware root of trust — or an *informal* architectural property ("raw data never leaves the device"),
  which is weaker and must not be sold as a guarantee.

The cardinal error in industry: **de-identification ≡ privacy.** Removing names/IDs ("anonymization") is
defeated by linkage and reconstruction; only a formal mechanism bounds what an adversary can infer. If you
can't state the number and the threat model, you don't have a privacy claim — you have a hope.

---

## 2. The motivating attacks (why PETs exist)

Models memorize, and that memory leaks. The defenses below exist to bound these.

- **Membership inference (MIA)** — given a record and access to the model, decide whether it was in the
  training set. The canonical baseline trains shadow models (Shokri et al., arXiv:1610.05820); strong
  modern attacks use per-example loss/likelihood-ratio (LiRA). High MIA accuracy = the model leaks who it
  trained on — a direct privacy harm (e.g., "this patient was in the cancer-cohort training set").
- **Model inversion / reconstruction** — recover representative or actual input features from model access
  (e.g., reconstructing a recognizable face from a face-recognition model). In FL, **gradient inversion**
  reconstructs a client's training batch from a single shared gradient — the reason raw gradients are *not*
  safe to share.
- **Training-data extraction / memorization** — large models, especially LLMs, **verbatim-memorize** rare
  sequences and regurgitate them when prompted (Carlini et al., "Extracting Training Data from Large
  Language Models," arXiv:2012.07805; quantified scaling in "Quantifying Memorization," arXiv:2202.07646).
  This is how secrets, PII, and copyrighted text escape. Memorization grows with model size, duplication,
  and context length.

Relationship to DP: a model trained with a meaningful ε **provably bounds** MIA advantage — DP is the
principled defense, not a heuristic. The attack-as-adversarial-discipline view (and empirical auditing of
DP by *running* MIAs) is [[adversarial-ml-robustness]].

---

## 3. Differential privacy (DP)

### 3.1 Definition and intuition

A randomized mechanism `M` is **(ε, δ)-differentially private** if for all datasets `D`, `D'` differing in
one record, and all output sets `S`:

```
Pr[M(D) ∈ S]  ≤  e^ε · Pr[M(D') ∈ S]  +  δ
```

Intuition: **adding or removing any single individual barely changes the output distribution**, so an
adversary observing the output learns almost nothing about whether you were in the data. ε is the
**privacy loss** (smaller = more private); δ is a small slack probability (the chance of a worse leak) —
keep δ ≪ 1/N (often δ ≈ 1e-5..1e-7 for N records; *the right δ depends on N — verify*).

Two properties make DP **engineering-grade**, unlike ad-hoc scrubbing:

- **Composition** — running k mechanisms degrades privacy predictably (naive: ε's add; advanced/RDP
  composition is much tighter). This lets you *budget* across queries, releases, and retrains.
- **Post-processing immunity** — anything you compute from a DP output is still DP. You can't "un-private"
  a result by analyzing it further.

Other accounting flavors you will meet: **Rényi DP (RDP)** and **zero-concentrated DP (zCDP, ρ)** —
tighter, composition-friendly relaxations that convert back to (ε,δ). Production DP language models are
often reported in **ρ-zCDP**.

**Choosing ε (rule of thumb, not law):** single-digit ε is meaningfully private; ε ~ 1–10 is the common
useful range for ML; ε in the tens-to-hundreds buys little real privacy and is often a number to look
skeptical at. ε is a *risk dial*, set with the data owner — not a default.

### 3.2 DP-SGD — the workhorse for training

Abadi et al., "Deep Learning with Differential Privacy" (arXiv:1607.00133). Make each SGD step DP by
bounding any one example's influence, then masking it with noise:

1. **Per-example gradient clipping** — compute the gradient *per example*, clip its L2 norm to a bound `C`.
   This **bounds sensitivity** (the most any one record can move the update).
2. **Add calibrated Gaussian noise** — add noise `N(0, σ²C²I)` to the *summed* clipped gradients. `σ` is
   the **noise multiplier**; it (with the sampling rate and steps) determines ε.
3. **Average and step** — divide by batch size (or lot size), apply the optimizer step.
4. **Account** — track spent (ε,δ) across all steps with a **moments accountant / RDP accountant / PRV
   accountant**. *Noise without accounting is not DP* — the accountant is what turns the recipe into a
   guarantee.

Knobs and their effect:

| Knob | Effect on privacy | Effect on utility / cost |
|---|---|---|
| Clip norm `C` | smaller `C` → less noise needed for same σ-relative scale, but more bias | too-small `C` clips real signal |
| Noise multiplier `σ` | larger σ → smaller ε (more private) | larger σ → worse accuracy |
| Sampling rate `q` (batch/N) | smaller q → amplification, smaller ε | tiny batches → noisier training |
| Steps / epochs | more steps → more budget spent, larger ε | more steps → better fit |
| Batch size | **large** batches average out noise | large batches help DP a lot (and need more memory) |

Tooling: **Opacus** (PyTorch), **TensorFlow Privacy**, DP libraries in JAX. They implement per-sample
gradients (vmap / functorch) and the accountant for you — but you still own the threat model and the
budget. Per-sample gradients are the **compute/memory tax** of DP-SGD (no free per-batch fusion).

Utility recovery, in priority order: **(1) pretrain non-privately on public data, then DP fine-tune** —
DP fine-tuning of a strong pretrained model is the single biggest lever; (2) very large batch sizes;
(3) tune `C` (often via per-layer or adaptive clipping); (4) more public/in-distribution data; (5) PEFT
adapters under DP to reduce trainable-parameter noise ([[fine-tuning-peft]]).

### 3.3 Central vs local DP

- **Central (curator) DP** — a trusted aggregator holds raw data and adds noise once to the released
  statistic/model. **Best utility**; requires trusting the curator with raw data.
- **Local DP (LDP)** — each user randomizes their data *on-device* before it ever leaves (randomized
  response, RAPPOR-style). **No trusted curator**, but utility is far worse (noise per user, not per
  population). Good for telemetry-scale aggregates; rarely good enough for high-fidelity model training.
- **Shuffle / aggregation models** sit in between: a shuffler or secure-aggregation step amplifies local
  privacy, recovering much of the central-model utility without a fully trusted curator.

### 3.4 DP-FTRL and production DP

Standard DP-SGD privacy amplification relies on **uniform Poisson sampling** of examples — which you can't
guarantee when clients (phones) show up unpredictably. **DP-FTRL** (DP Follow-The-Regularized-Leader, with
a tree-aggregation noise mechanism) gives a **formal DP guarantee without** that sampling assumption,
making it the right fit for cross-device FL.

Production: mobile keyboard **next-word-prediction models have been trained and deployed with formal DP
guarantees** using DP-FTRL in federated learning — Google, "Federated Learning with Formal Differential
Privacy Guarantees" (research.google/blog/federated-learning-with-formal-differential-privacy-guarantees/)
and the follow-up "Advances in private training for production on-device language models." Reported as
**ρ-zCDP** (the first announcement was a Spanish next-word model; later many models with ρ in a small
range, some additionally using secure aggregation). **Verify the exact ρ/ε and model details against the
current posts — do not quote from memory.**

### 3.5 DP for fine-tuning and synthetic data

- **DP fine-tuning** — the dominant practical recipe: a non-private pretrained base + DP-SGD (or DP-PEFT)
  on the sensitive set. Bounds leakage of the fine-tuning data while keeping most utility.
- **DP synthetic data** — train a generator (or use a private statistical model) under DP, then sample a
  synthetic dataset. By post-processing immunity, the synthetic data inherits the DP guarantee, so
  downstream use is unconstrained. Caveat: utility/fidelity is hard, and "looks realistic" is **not** a
  privacy property — only the ε is. Marginal/query-release methods can beat GANs on tabular data.

---

## 4. Federated learning (FL)

### 4.1 FedAvg and the mental model

McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
(arXiv:1602.05629). **Raw data never leaves the device/silo; only model updates move.** One **FedAvg**
round:

1. Server broadcasts the current global model to a cohort of clients.
2. Each client runs **several local SGD epochs** on its own data.
3. Each client sends back its **update** (delta or weights).
4. Server **averages** updates (weighted by local dataset size) → new global model. Repeat.

Local epochs > 1 is the trick that cuts communication rounds by orders of magnitude vs. naive distributed
SGD — at the cost of **client drift** on non-IID data.

### 4.2 Cross-device vs cross-silo

| | Cross-device | Cross-silo |
|---|---|---|
| Clients | millions of phones/IoT | a few orgs (hospitals, banks) |
| Availability | unreliable, intermittent, **stragglers** | reliable, persistent |
| Per-client data | small, **highly non-IID** | large, structured |
| Trust | clients untrusted; need SecAgg+DP | parties semi-trusted but contractual |
| Stateful? | mostly stateless (clients seen once) | stateful, can keep optimizer state |
| Framework fit | TFF, production FL stacks | Flower, FedML, NVFlare |

### 4.3 The hard problems

- **Non-IID data** — clients have skewed/disjoint label distributions; FedAvg can diverge or stall.
  Mitigations: **FedProx** (proximal term limiting drift), **SCAFFOLD** (control variates correcting
  drift), server-side adaptive optimizers (**FedAdam/FedYogi**), fewer local steps.
- **Stragglers / dropouts** — slow/absent clients stall a round. Handle with deadlines, partial
  aggregation, and dropout-tolerant secure aggregation.
- **Communication efficiency** — the bottleneck is the network, not compute. Compress updates: **gradient
  quantization, sparsification (top-k), structured/low-rank updates**, and fewer rounds via more local
  work. PEFT adapters shrink the payload ([[fine-tuning-peft]]).
- **Personalization** — one global model fits no client perfectly. Options: local fine-tuning, **meta-
  learning (Per-FedAvg)**, partial model personalization (shared backbone + private heads), clustered FL.

### 4.4 FL is NOT private by itself

A naive FL server **sees every client's individual update**, and **gradient inversion** can reconstruct a
client's data from it. FL gives *data minimization* (raw data stays put) — not a privacy *guarantee*. To
make FL private you add the two layers below.

### 4.5 Secure aggregation

A cryptographic protocol so the **server learns only the sum** of client updates, never any individual
one. Pairwise masks (Diffie-Hellman-derived) are added to each client's update such that **masks cancel
exactly when summed**; a secret-sharing/recovery step keeps it correct even when some clients **drop out**
mid-round (Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning,"
arXiv:1611.04482). Cost: extra communication rounds and per-round key setup; scales to large cohorts with
careful protocol design.

### 4.6 FL + DP composition

The production-grade stack: **FedAvg + secure aggregation + DP**. Where the noise/clipping lives matters:

- **Client-level (user-level) DP** is what you usually want in cross-device — clip and add noise so that
  *any one client's entire contribution* is masked, protecting the user, not just a record.
- **DP-FTRL** (§3.4) provides the formal accounting without Poisson client sampling.
- **SecAgg + central-style DP** lets you add aggregate noise after a sum the server can't decompose,
  approaching central-DP utility without a trusted curator (the shuffle/aggregation amplification idea).

Track the **client-level** budget across rounds and across model releases — see §7.

### 4.7 Frameworks

- **TensorFlow Federated (TFF)** — research-to-production, strong DP-FTRL/SecAgg story; the lineage behind
  shipped production DP keyboard models.
- **Flower** — framework-agnostic (PyTorch/TF/JAX), strong for cross-silo and experimentation; large
  community.
- **FedML / NVFlare** — cross-silo and enterprise/medical deployments (NVFlare common in healthcare).
- **PySyft (OpenMined)** — FL + MPC/HE research; broader PET toolkit.

Pick by deployment shape (cross-device vs cross-silo) and which DP/SecAgg primitives are first-class.
**Verify current** API surface — these libraries iterate quickly.

---

## 5. Cryptographic PETs and confidential computing

Strong guarantees, large costs. Reach for these only when the threat model genuinely forbids the cheaper
options (e.g., parties cannot see each other's data and DP/FL alone don't satisfy the contract).

### 5.1 Homomorphic encryption (HE)

Compute directly on **ciphertext**; decrypting the result equals computing on plaintext. Schemes: **CKKS**
(approximate real arithmetic — the ML favorite), **BFV/BGV** (exact integer), **TFHE** (fast boolean/
bootstrapping). Libraries: Microsoft SEAL, OpenFHE, Lattigo, Google's HE compiler tooling.

- **Great for:** privacy-preserving *inference* of shallow/linear models, and as a building block.
- **Brutal for:** deep training and non-linearities (activations need polynomial approximation; depth
  needs bootstrapping). Expect **orders-of-magnitude** slowdown and large ciphertext blowup.
- Use when one party must compute on another's data without ever seeing the plaintext, and latency budget
  allows it (often offline/batch inference).

### 5.2 Secure multiparty computation (MPC)

Multiple parties jointly compute a function over their private inputs, each learning **only the output**.
Built on secret sharing / garbled circuits. Frameworks: CrypTen, MP-SPDZ, TF-Encrypted.

- **Communication-bound**, not compute-bound — many rounds of interaction; LAN-fast, WAN-painful.
- Security assumes a bound on colluding parties (e.g., honest-majority or two non-colluding servers).
- Fits cross-silo "no one sees the others' data" inference/training among a small, fixed set of parties.

### 5.3 Private set intersection (PSI)

Two parties compute the **intersection** of their sets (e.g., overlapping users) **without revealing
non-members**. Workhorse for privacy-preserving join / ads-measurement / record linkage before any joint
modeling. Cheap relative to general MPC; many efficient protocols exist.

### 5.4 Trusted execution environments (TEEs) / confidential computing

Hardware-isolated **enclaves** (Intel SGX/TDX, AMD SEV-SNP, ARM CCA, and **confidential GPUs**) protect
data **in use** — encrypted in memory, isolated from the host/hypervisor — with **remote attestation**
proving the right code runs on genuine hardware before any secret is released.

- **Much cheaper** than HE/MPC (near-native compute for confidential GPUs) — often the pragmatic choice
  for "process sensitive data in an untrusted cloud."
- **Costs/risks:** you **trust the hardware vendor**; real **side-channel** history; attestation and key
  management are operationally non-trivial; enclave memory limits (older SGX). Infra/cluster specifics —
  confidential node pools, attestation, GPU support — are [[gke-master]] and [[ai-security-on-gke]].

**HE vs MPC vs TEE, one line each:** HE = compute on ciphertext (no extra parties, slow); MPC = split
across non-colluding parties (interactive, communication-bound); TEE = trust hardware + attestation
(fast, vendor-trust + side channels).

---

## 6. Machine unlearning (fast-moving, treat claims skeptically)

**Driver:** GDPR/CCPA **right-to-be-forgotten** / data-deletion requests — when a user's data must leave
the *model*, not just the database. Retraining from scratch on every deletion is the correct-but-
expensive baseline; unlearning tries to do better.

- **Exact unlearning** — the model behaves *as if* the data were never trained on. **SISA** (Sharded,
  Isolated, Sliced, Aggregated; Bourtoule et al., arXiv:1912.03817): partition data into shards, train an
  isolated submodel per shard, ensemble. To forget a record, **retrain only its shard** (and only from the
  affected slice checkpoint) — bounded, provable, but costs accuracy (ensembling shards) and storage
  (checkpoints).
- **Approximate unlearning** — cheaper parameter edits (influence-function removal, gradient ascent on the
  forget set, fine-tuning) that *approximately* remove influence. Faster, but the guarantee is weak and
  often **unverifiable**.
- **Verification is the hard, immature part.** "We unlearned it" is hard to *prove*: membership-inference
  on the forgotten records is a common audit but is noisy; benchmarks (and even the definition of success)
  are **unsettled and unreliable**. **Never trust an approximate-unlearning claim without an audit**, and
  treat published methods as research-grade.
- **Engineering stance:** design for forgetting up front — **data lineage + retention + sharding**, and
  prefer **DP** (which bounds any single record's influence, easing forgetting) where feasible. **This is
  a fast-moving area — verify current methods/benchmarks before relying on any.**

---

## 7. Production guidance: choosing and composing PETs

### 7.1 Pick the PET from the constraint

- **"Outputs/model must not leak individuals" (any single trusted trainer)** → **DP (DP-SGD / DP fine-
  tuning)** with a defended ε. The default for "train privately on sensitive data."
- **"Data cannot leave the device" (mobile, IoT)** → **cross-device FL** (FedAvg/DP-FTRL) + secure
  aggregation + client-level DP. See [[edge-on-device-ml]].
- **"Several orgs want a joint model, none can share raw data"** → **cross-silo FL** (semi-trusted) or
  **MPC** (no trust); **PSI** first if you must align records.
- **"One party computes on another's data, plaintext must never be exposed"** → **HE** (if latency
  allows) or **TEE/confidential computing** (if vendor-trust is acceptable and you want near-native speed).
- **"Must honor deletion requests against the model"** → design **lineage + retention + SISA**, prefer DP;
  verify any unlearning.

### 7.2 The cost axes (there is no free privacy)

| PET | Privacy guarantee | Accuracy cost | Compute/latency | Communication | Trust assumption |
|---|---|---|---|---|---|
| DP-SGD / DP-FT | formal (ε,δ)/ρ | medium (recoverable via pretraining + big batch) | high (per-sample grads) | n/a | trusted trainer (central) |
| Local DP | formal, per-user | **high** | low | low | none |
| FedAvg (alone) | **none** (data minimization only) | low | client-side | medium–high | server sees updates |
| + Secure aggregation | hides individual updates | none added | moderate | higher (rounds) | non-collusion |
| + Client-level DP | formal, user-level | medium | client-side | medium | curator/aggregator |
| HE | cryptographic | none (approx. error in CKKS) | **very high** | low–medium | none (compute) |
| MPC | cryptographic | none | high | **very high** (interactive) | bounded collusion |
| TEE | hardware + attestation | none | **near-native** | low | hardware vendor |

Numbers vary wildly by model/setup — **measure on your workload**; don't trust a generic slowdown figure.

### 7.3 Composing DP with FL — the canonical stack

`FedAvg (local training) → compressed update → secure aggregation (server sees only the sum) → DP noise +
clipping at user level → DP-FTRL accounting across rounds`. This is the shipped pattern behind production
DP on-device language models. SecAgg without DP hides *individual* updates but the *aggregate* can still
leak; DP without SecAgg makes the server trustworthy with raw updates — you generally want **both**.

### 7.4 Privacy budget over time

ε is a **finite resource spent across the system's life** — every release, query, retrain, and additional
model on the same data **composes** and increases total ε. Maintain a **privacy ledger**: track cumulative
(ε,δ)/ρ per protected unit, set a hard cap, and account at the right granularity (record vs user). The
silent killer is **reuse**: rerunning the "same ε=2" mechanism 50 times is not ε=2. Coordinate the budget
with the governance owner ([[responsible-ai-governance]]).

---

## 8. Anti-patterns (the traps that bite in production)

- **"Anonymized" with no DP.** Dropping identifiers and calling it private. Linkage/reconstruction defeats
  it; only a formal mechanism (a defended ε) bounds the leak. **Stop saying "anonymized" for a guarantee.**
- **Training on sensitive data with no privacy accounting.** Adding "some noise" or "a privacy layer"
  without an accountant. Noise without (ε,δ) accounting is **not** DP — it's just a worse model.
- **FL without secure aggregation.** Trusting that "data stays on device" is enough while the server reads
  every individual gradient — directly invertible. FL ≠ privacy.
- **Unbounded privacy-budget reuse.** Refreshing/reusing ε across releases, queries, and retrains without
  a ledger or cap — silently composes to a meaningless guarantee.
- **Trusting unlearning without verification.** Shipping approximate unlearning and claiming compliance
  with no audit, on immature/unreliable benchmarks. Verify, or design for retrain/SISA.
- **HE/MPC where the cost isn't justified.** Reaching for heavy crypto when DP, FL+SecAgg, or a TEE meets
  the threat model at a fraction of the cost. Match the mechanism to the *actual* adversary, not the
  scariest imaginable one.
- **Optimizing ε to a small number, then sharing white-box weights to anyone.** The threat model must
  include *who sees the model*; a tiny ε plus an open weight dump can still over-share.
- **Treating synthetic data as automatically private.** "Looks realistic" is not a privacy property — only
  a DP guarantee on the generator is. Memorized real records can hide in synthetic samples.

---

## 9. Canonical references (verify current)

Foundational (stable):

- **DP-SGD** — Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang, "Deep Learning with Differential
  Privacy," CCS 2016. arXiv:1607.00133.
- **DP textbook** — Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (the definitions,
  composition, RDP precursors).
- **FedAvg** — McMahan, Moore, Ramage, Hampson, Agüera y Arcas, "Communication-Efficient Learning of Deep
  Networks from Decentralized Data," AISTATS 2017. arXiv:1602.05629.
- **Secure aggregation** — Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine
  Learning," CCS 2017. arXiv:1611.04482.
- **Membership inference** — Shokri, Stronati, Song, Shmatikov, S&P 2017. arXiv:1610.05820.
- **Training-data extraction** — Carlini et al., "Extracting Training Data from Large Language Models,"
  USENIX Security 2021. arXiv:2012.07805; "Quantifying Memorization Across Neural Language Models,"
  arXiv:2202.07646. *(arXiv IDs believed correct — verify before citing.)*
- **SISA unlearning** — Bourtoule et al., "Machine Unlearning," S&P 2021. arXiv:1912.03817. *(verify ID.)*

Production / fast-moving (verify against current docs):

- Google Research, **"Federated Learning with Formal Differential Privacy Guarantees"** —
  research.google/blog/federated-learning-with-formal-differential-privacy-guarantees/ (DP-FTRL; production
  ρ-zCDP keyboard models). Follow-up: **"Advances in private training for production on-device language
  models."** Verify the exact ρ/ε and model list.
- **Gboard DP LMs paper** — "Federated Learning of Gboard Language Models with Differential Privacy,"
  ACL 2023 (Industry). arXiv:2305.18465.
- **2025 PETs/FL survey** — *Artificial Intelligence Review* (Springer), doi:10.1007/s10462-025-11376-7.
  *(DOI is in the correct journal; confirm exact title/scope at the source — verify current.)*

Tooling docs (verify API — these iterate fast): **Opacus** (PyTorch DP-SGD), **TensorFlow Privacy**,
**TensorFlow Federated**, **Flower**, **FedML**, **NVFlare**, **PySyft/OpenMined**; **Microsoft SEAL**,
**OpenFHE**, **Lattigo** (HE); **CrypTen**, **MP-SPDZ** (MPC); the **Google DP** library and DP accounting
(RDP/PRV) libraries.
