# Worked Examples — Accelerator Memory Estimator

Three fully worked estimates: **inputs → formula → number → recommendation**. All arithmetic uses
bytes, then converts to **GiB (÷ 1.074e9)**. Constants from the guide: bf16 = 2 bytes, fp32 = 4,
int4 = 0.5; full training model state = `16·P`; KV cache `= 2·layers·kv_heads·head_dim·seq·batch·dtype_bytes`.

> **Every number below is an estimate (±10–30%). Verify against a real run** (small-slice OOM test +
> `torch.cuda.max_memory_allocated()` / `nvidia-smi`, or the serving engine's reported KV budget)
> before committing hardware. The activation term in particular is the least certain.

---

## (a) Full fine-tune of a 7B in bf16 + AdamW on one 80GB H100 — does it fit?

**Inputs (stated):** P = 7e9; dtype bf16 (mixed precision, fp32 optimizer master); optimizer AdamW;
micro-batch = 4; seq = 4096; hidden = 4096; layers = 32; FlashAttention on; 1 × H100 (80 GiB);
no checkpointing initially.

**Model state (16·P):**
```
weights   2·P = 2 × 7e9 = 1.40e10 bytes = 13.0 GiB
gradients 2·P = 1.40e10                 = 13.0 GiB
optimizer 12·P (fp32 master 4·P + m 4·P + v 4·P) = 8.40e10 = 78.2 GiB
model state = 16·P = 1.12e11 bytes      = 104.3 GiB
```

**Activations** (`batch·seq·hidden·layers·A_bytes`, A_bytes ≈ 50):
```
4 · 4096 · 4096 · 32 · 50 = 1.07e11 bytes = 100.0 GiB   (no checkpointing)
with activation checkpointing (÷ R≈7): ≈ 14.3 GiB
```

**Per-device total (× ~1.15 overhead):**
```
no checkpointing: (104.3 + 100.0) × 1.15 ≈ 235 GiB
with checkpointing: (104.3 + 14.3) × 1.15 ≈ 136 GiB
```

**Compare to capacity:** one H100 = 80 GiB; usable ≈ 68 GiB (15% headroom).
**104.3 GiB of model state alone already exceeds 80 GiB** — it does **not** fit, even before
activations, even with checkpointing. **Estimate, verify.**

**Recommendation:**
- Full FT on a single 80GB GPU is impossible (need ≥ ~136 GiB with checkpointing). Options:
  1. **FSDP/ZeRO-3 across ≥ 4 H100s** → model state 104.3/4 ≈ 26 GiB/GPU; with activation
     checkpointing and overhead it fits comfortably on 4× H100 (verify N).
  2. **QLoRA on a single 24GB GPU:** 4-bit base weights `0.5·P` = 3.5e9 bytes ≈ **3.3 GiB** + small
     adapters + adapter optimizer → fits a single 24GB card with room for activations. This is the
     cheapest path if full-parameter updates aren't required.
  3. 8-bit Adam (optimizer 12·P→~6·P) drops model state to ~10·P ≈ 65 GiB — still > 80 GiB once
     activations/overhead are added on one GPU; helps but doesn't rescue single-GPU full FT.
- **Verify:** run 5 training steps on a 2-layer slice or batch=1 and read peak memory before scaling.

---

## (b) Full pretrain/FT of a 70B with FSDP / ZeRO-3 across 8 × H100 — per-GPU math

**Inputs (stated):** P = 70e9; bf16 + AdamW; ZeRO-3 / FSDP full shard, N = 8; local micro-batch = 1;
seq = 4096; hidden = 8192; layers = 80; FlashAttention; activation checkpointing on; 8 × H100 (80 GiB).

**Model state, then sharded by N (ZeRO-3 divides weights + grads + optimizer):**
```
16·P = 16 × 70e9 = 1.12e12 bytes = 1042.8 GiB   (total, all ranks)
per GPU = 16·P / N = 1042.8 / 8 = 130.4 GiB/GPU   ← still > 80 GiB!
```

**Activations per GPU** (local micro-batch = 1, with checkpointing ÷ R≈8):
```
full: 1 · 4096 · 8192 · 80 · 50 = 1.34e11 bytes = 125.0 GiB
checkpointed (÷8): ≈ 15.6 GiB/GPU
```

**Per-GPU total at N = 8:** (130.4 + 15.6) × 1.15 ≈ **168 GiB/GPU** → does **not** fit 80 GiB.
Plus an FSDP all-gather buffer (~one layer's weights). **Estimate, verify.**

**Scale out — re-run the divide-by-N:**
```
N = 16: model state/GPU = 1042.8/16 = 65.2 GiB ; total w/ ckpt+overhead ≈ 93 GiB → still > 80
N = 32: model state/GPU = 1042.8/32 = 32.6 GiB ; total w/ ckpt+overhead ≈ 55 GiB → fits (with headroom)
```

**Recommendation:**
- **8 × H100 is not enough** for full FT of 70B with AdamW: ~168 GiB/GPU needed vs 80 available.
- Reach a fit by **more devices + checkpointing**: ZeRO-3 across **~32 H100s** lands ~55 GiB/GPU
  (verify). Or **2D parallelism**: TP=8 within a node × FSDP across nodes to also shard activations.
- Cheaper alternatives on 8 GPUs: **8-bit Adam** (model state 10·P → 875 GiB total → 109 GiB/GPU on
  N=8, still tight) or **LoRA/QLoRA** (freeze the 70B base; only adapters get grads/optimizer) — LoRA
  on 8× H100 fits easily since the base is just `2·P` ≈ 130 GiB / 8 ≈ 16 GiB/GPU when sharded.
- **Verify:** launch on the real N with checkpointing enabled and watch `nvidia-smi` peak before
  committing the full run.

---

## (c) KV-cache for a 70B at seq = 8k, batch = 32 with GQA

**Inputs (stated):** P = 70e9; serving in bf16; layers = 80; **GQA with 8 kv_heads** (64 query heads);
head_dim = 128; seq_len = 8192; batch = 32; KV dtype = bf16 (2 bytes); target = 8 × H100 (80 GiB).

**KV cache** (`2 · layers · kv_heads · head_dim · seq · batch · dtype_bytes`):
```
2 · 80 · 8 · 128 · 8192 · 32 · 2 = 8.59e10 bytes = 80.0 GiB   (total KV for the batch)
```

**KV per token** (the rate that governs concurrency):
```
2 · layers · kv_heads · head_dim · dtype_bytes
= 2 · 80 · 8 · 128 · 2 = 327,680 bytes/token ≈ 320 KiB/token
```
Check: 320 KiB × (8192 × 32 tokens) = 80.0 GiB. ✓

**Contrast — if this model used MHA (kv_heads = 64 instead of 8):** KV would be **8× larger ≈ 640 GiB**.
GQA is what makes long-context, high-batch serving viable. **Estimate, verify.**

**Weights:**
```
bf16: 2·P = 130.4 GiB    int8: 1·P = 65.2 GiB    int4: 0.5·P = 32.6 GiB
```

**Fit on 8 × H100 (640 GiB total), tensor-parallel TP = 8 (weights & KV ÷ 8):**
```
weights/GPU = 130.4 / 8 = 16.3 GiB
KV/GPU      = 80.0  / 8 = 10.0 GiB
+ ~3 GiB workspace/GPU  → ~29 GiB/GPU used of 80 → fits with large headroom
```

**Recommendation:**
- 70B bf16 with TP=8 on 8× H100 serves seq=8k × batch=32 comfortably (~29 GiB/GPU). The cluster has
  lots of spare KV budget.
- **How much more concurrency/context fits?** Total KV budget ≈ `8·80 − 130.4(weights) − 8·3(workspace)`
  ≈ **486 GiB** → `max_tokens_in_flight ≈ 486 GiB / 320 KiB ≈ 1.6M tokens` across all sequences
  (e.g. ~200 concurrent 8k-context sequences). Use this to set the serving engine's max-num-seqs /
  max-num-batched-tokens.
- To pack even more: **int8/fp8 KV cache** halves KV per token (160 KiB → ~2× the sequences);
  **int4 weights** free ~98 GiB more for KV. Verify accuracy after quantizing.
- **Verify:** start the serving engine (e.g. vLLM) and read its reported KV-cache blocks / GPU memory
  utilization — it computes the real free-KV budget for your exact build.

---

## Pattern to imitate

For any request: **state inputs → list terms → compute each in bytes with the formula → sum per device
(after sharding ÷N/÷k) → add ~15% overhead → convert to GiB → compare to capacity − headroom → give the
cheapest fitting recommendation → label "estimate (±), verify" and name the verification.**
