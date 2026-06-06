# Privacy-Preserving ML — Worked Examples

Three artifacts to imitate: (1) a **DP-SGD config sketch** with an ε-accounting note, (2) an **FL +
secure-aggregation round** outline, and (3) a **PET-selection decision table**. All numbers below are
**illustrative placeholders** — the realized ε depends on `(noise multiplier, sampling rate, steps, δ)`
and must come from a real **accountant on your run**, not from this file. Library APIs move fast: **verify
against current docs.**

---

## 1. DP-SGD config sketch (with ε accounting note)

The recipe that matters: **pretrain non-privately on public data → DP fine-tune the sensitive set.** This
recovers most of the utility DP costs. Pseudocode-as-config (Opacus-flavored; verify the current API):

```python
# DP fine-tuning of a pretrained model with DP-SGD.
# Privacy unit here = ONE TRAINING RECORD. (User-level DP requires grouping by user — different setup.)

from opacus import PrivacyEngine          # verify current API surface
import torch

model      = load_pretrained()            # non-private pretraining on PUBLIC data already done
optimizer  = torch.optim.SGD(model.parameters(), lr=0.5)   # DP often likes higher LR + big batch
data_loader = sensitive_loader            # large LOGICAL batch is the single biggest utility lever

# --- DP knobs (the whole guarantee lives here) ---
MAX_GRAD_NORM    = 1.0     # C: per-example L2 clip → bounds sensitivity. Tune (often 0.1–1.0).
NOISE_MULTIPLIER = 1.1     # σ: noise scale relative to C. Larger σ → smaller ε, worse accuracy.
TARGET_DELTA     = 1e-6    # δ ≪ 1/N. With N≈1e6 records, 1e-6..1e-7 is typical. (verify for your N)
EPOCHS           = 3       # more steps spend more budget → larger ε
# Effective batch via gradient accumulation; physical micro-batch bounded by per-sample-grad memory.

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=data_loader,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
    # (Poisson sampling is what enables amplification; the engine handles it. In FL use DP-FTRL instead,
    #  which does NOT require Poisson client sampling — see example 2.)
)

# ... standard training loop: forward, loss, backward; the engine clips PER-EXAMPLE grads then adds
#     N(0, σ²C²) to the summed grads before optimizer.step(). PEFT/LoRA under DP shrinks the noised
#     parameter count and usually helps. ...

# --- ε ACCOUNTING (mandatory; this is what makes it DP, not just "noisy SGD") ---
epsilon = privacy_engine.get_epsilon(delta=TARGET_DELTA)   # RDP/PRV accountant under the hood
print(f"Spent (ε, δ) = ({epsilon:.2f}, {TARGET_DELTA}) after {EPOCHS} epochs")  # e.g. ε ≈ 6–8 (ILLUSTRATIVE)
```

**Accounting note (read this):**
- The printed ε is computed by an **accountant** (RDP / moments / PRV) from `(σ, sampling rate q, steps,
  δ)`. There is no shortcut: change any knob and you must re-account. **Adding noise without running the
  accountant is not differential privacy.**
- ε is a **risk dial**: single-digit ε is meaningfully private; ε in the tens-to-hundreds buys little real
  privacy. Set it *with the data owner*, and record it in the model card ([[responsible-ai-governance]]).
- The spent ε is **per this training run on this data**. It **composes** with every other release/retrain
  on the same data — log it to the privacy ledger (example 3 / guide §7.4). Reusing "ε=6" five times is
  **not** ε=6.
- Sanity-check empirically: run a membership-inference / DP-auditing attack and confirm the empirical
  privacy loss is consistent with the claimed ε ([[adversarial-ml-robustness]]).

---

## 2. FL + secure-aggregation round outline (with DP)

The production-grade stack: **FedAvg + secure aggregation + client-level DP (DP-FTRL accounting).** One
round, cross-device (millions of intermittent clients, non-IID):

```
ROUND r  (server orchestrates; raw data NEVER leaves a device)
─────────────────────────────────────────────────────────────
1. SELECT      Server samples a cohort of available clients (over-provision for stragglers/dropouts).
               (Client availability is unpredictable → use DP-FTRL, which gives a formal DP guarantee
                WITHOUT relying on Poisson sampling. Do not assume uniform sampling here.)

2. BROADCAST   Server sends the current global model w_r to the cohort.

3. LOCAL TRAIN Each client k runs E local SGD epochs on its own data → update Δ_k = w_k - w_r.
               (Handle non-IID drift with FedProx/SCAFFOLD or fewer local steps. Compress Δ_k via
                quantization / top-k sparsification to cut communication.)

4. CLIP        Each client CLIPS its update to L2 norm C  (client-/user-level sensitivity bound).
               (This is the per-user analogue of DP-SGD's per-example clip.)

5. MASK        SECURE AGGREGATION: each client adds pairwise masks (DH-derived) to its clipped Δ_k so
               that the masks CANCEL when all updates are summed. Secret-shared so the sum is still
               recoverable if some clients DROP OUT mid-round.  → server can compute ONLY  Σ_k Δ_k,
               never any individual Δ_k. (Bonawitz et al., arXiv:1611.04482)

6. AGGREGATE   Server reconstructs the SUM of (clipped, masked) updates — individual updates are hidden.

7. ADD NOISE   DP noise is added at the AGGREGATE (server-side or distributed), calibrated to C, to
               give a formal CLIENT-LEVEL DP guarantee. (Tree-aggregation noise = DP-FTRL.)

8. UPDATE      w_{r+1} = w_r + (noised aggregate). Update the DP-FTRL privacy accountant for the round.

9. ACCOUNT     Track CUMULATIVE client-level (ε,δ)/ρ across rounds AND across model releases. Stop /
               re-key when the budget cap is hit. (Reported in production as ρ-zCDP — verify numbers.)
```

**Why every step:** drop SecAgg and the server reads individual updates → **gradient inversion**
reconstructs a client's data. Drop DP and the *aggregate* still leaks. Drop clipping and DP noise can't be
calibrated. You want **all three**. (Google, "Federated Learning with Formal Differential Privacy
Guarantees" — the shipped pattern; verify the current ρ.)

---

## 3. PET-selection decision table

Pick from the **constraint**, then commit to a defensible number. (Costs are directional — **measure on
your workload**; do not trust generic slowdown figures.)

```
CONSTRAINT (what the requirement actually forbids)            →  PET CHOICE                         COST / NOTE
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
Model/outputs must not leak any individual; ONE trusted       →  DP-SGD / DP fine-tuning            accuracy hit (recover via
  trainer holds the data                                          + named accountant (RDP/PRV)        public pretrain + big batch);
                                                                                                       per-sample-grad compute tax

Telemetry-scale aggregates, NO trusted curator at all         →  Local DP (on-device randomization)  high utility cost; aggregates
                                                                                                       only, not high-fidelity models

Raw data cannot leave the device (mobile / IoT)               →  Cross-device FL (FedAvg/DP-FTRL)    non-IID + stragglers; comms-bound;
                                                                   + SecAgg + client-level DP           see [[edge-on-device-ml]]

Few orgs want a joint model; none can share raw data;         →  Cross-silo FL (semi-trusted)        contractual trust; reliable clients;
  parties are semi-trusted/contractual                                                                 use [[gke-master]] confidential nodes

Joint computation among parties with NO mutual trust          →  MPC (secret sharing / GC)           interactive, communication-bound;
  (small fixed set, learn only the output)                                                            WAN-painful; assumes bounded collusion

Must align/JOIN records across parties first, privately       →  PSI (private set intersection)      cheap vs general MPC; reveals only
                                                                                                       the intersection

One party computes on another's data; plaintext must          →  HE (CKKS for ML)                    great for shallow/linear inference;
  NEVER be exposed; latency-tolerant                                                                  orders-of-magnitude slow for deep training

Process sensitive data in an untrusted cloud; want            →  TEE / confidential computing        near-native speed; TRUSTS the hardware
  near-native speed; vendor-trust acceptable                      (SGX/TDX/SEV, confidential GPUs)     vendor; side-channel history; attestation
                                                                                                       ops → [[ai-security-on-gke]]

Must honor deletion / right-to-be-forgotten against the       →  Design lineage + retention + SISA;  exact = costly (shard retrain);
  MODEL (GDPR/CCPA)                                                prefer DP; VERIFY any unlearning     approximate = unverifiable (FAST-MOVING)
```

**Composition rule of thumb:** the strongest common production stack is **FL + SecAgg + DP** — each layer
closes a leak the others leave open (guide §7.3). Heavy crypto (HE/MPC) is justified **only** when the
threat model genuinely forbids the cheaper options; otherwise DP, FL+SecAgg, or a TEE meets the bar at a
fraction of the cost.
