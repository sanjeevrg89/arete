---
name: accelerator-memory-estimator
description: >
  Use this to ESTIMATE GPU/TPU memory for an ML workload and DECIDE what fits and what parallelism is
  needed. Trigger whenever someone asks "will this model fit", "do I have enough VRAM/HBM", "OOM /
  CUDA out of memory", "how many H100s / how much TPU HBM do I need", "what batch size fits", "can I
  fine-tune a 7B/70B on one GPU", "how big is the KV cache". Performs the actual arithmetic: training
  memory (weights + gradients + optimizer states + activations + overhead), sharding math
  (FSDP/ZeRO-1/2/3, tensor parallel, pipeline parallel), and inference KV-cache memory
  (with GQA/MQA). Given a model + config, it computes per-device memory, compares to capacity
  (H100 80GB, A100 40/80GB, TPU v5e/v5p HBM), and RECOMMENDS a fitting strategy (fits / FSDP / TP=k /
  activation checkpointing / quantization / QLoRA). All outputs are ESTIMATES (±) — verify against a
  real run.
---

# Accelerator Memory Estimator

This is a **doer**: given a model and a config, you run the formulas in `accelerator-memory-estimator-guide.md`
to produce a **numeric per-device memory estimate** and a concrete fitting recommendation. Apply the
judgment of an engineer who has sized hundreds of training and serving jobs on GPU and TPU clusters.

**Every number you produce is an estimate (±).** Real memory depends on framework, kernels,
fragmentation, and version. Always tell the reader to confirm with `nvidia-smi` / a profiler / an OOM
test on a small slice before committing hardware.

## How to use this skill

1. **Read `accelerator-memory-estimator-guide.md`** — the formulas and the step-by-step procedure.
   Apply it end to end: collect inputs → compute each memory term → sum per device → compare to
   capacity → recommend.
2. For fully worked numeric estimates to imitate (full FT of a 7B on one 80GB GPU; 70B with ZeRO-3 on
   8×H100; KV cache for a 70B with GQA), read **`examples.md`**.
3. State your assumptions explicitly (dtype, optimizer, batch, seq, device, count). Show
   inputs → formula → number → recommendation. Label every result **estimate, verify**.

## Essentials (full detail in `accelerator-memory-estimator-guide.md`)

- **Full training ≈ 16·P bytes + activations + overhead.** For P params with bf16 weights and
  mixed-precision AdamW: weights 2·P + gradients 2·P + optimizer 12·P (fp32 master 4·P + m 4·P +
  v 4·P) = **16·P** before activations. Add ~5–15% framework/fragmentation overhead.
- **8-bit Adam** cuts optimizer state ~12·P → ~6·P; **fp8/8-bit** weights/grads cut further. Note them
  as levers, don't assume them silently.
- **Activations dominate at large batch×seq.** Roughly ∝ batch · seq · hidden · layers · (bytes).
  **Activation checkpointing** trades compute for memory — store only layer boundaries, recompute the
  rest: activation memory drops by roughly the per-layer factor (often 5–10×).
- **Sharding divides by world size.** FSDP / ZeRO-3 shard **weights + gradients + optimizer** by the
  data-parallel size N (each ≈ 16·P/N). ZeRO-1 shards optimizer only; ZeRO-2 optimizer + grads.
  **Tensor parallel (TP=k)** divides per-layer weights/activations by k (high comms, keep intra-node).
  **Pipeline parallel** splits layers across stages (mind the bubble + in-flight microbatches).
- **Inference ≈ weights + KV cache + workspace.** Weights ≈ 2·P bf16 (≈1·P int8, ≈0.5·P int4).
- **KV cache = 2 · layers · kv_heads · head_dim · seq_len · batch · dtype_bytes.** GQA/MQA shrink
  `kv_heads` (often 8 vs 64 query heads → ~8× smaller cache). KV cache, not weights, usually caps
  serving concurrency and context length.
- **Rules of thumb:** full FT ≈ 16·P + activations; LoRA ≈ weights (frozen) + tiny adapter state;
  **QLoRA ≈ 4-bit weights (~0.5·P) + adapters** → 7B fine-tunes on a single 24GB card; inference ≈
  weights + KV.
- **The procedure:** inputs (P or dims, dtype, optimizer, batch, seq, devices, device HBM) → per-device
  bytes → compare to capacity (leave ~10–20% headroom) → recommend the cheapest strategy that fits.
- **Always state ± and verify.** These estimates get you to the right hardware/parallelism class; the
  exact fit is confirmed by a real run.

## Related skills

- `[[training-frameworks]]` — implement FSDP/ZeRO/TP/PP (DeepSpeed, Megatron, FSDP, NeMo) once you know
  the strategy.
- `[[fine-tuning-peft]]` — LoRA/QLoRA mechanics when the estimate says full FT won't fit.
- `[[ml-frameworks]]` — PyTorch/JAX/XLA memory behavior, dtypes, activation checkpointing APIs.
- `[[serving-frameworks]]` — vLLM/SGLang/TensorRT-LLM that turn the KV-cache budget into real concurrency.
- `[[inference-optimization]]` — paged/quantized KV cache, chunked prefill, batching to use the budget.
- `[[ai-networking-collectives]]` — interconnect (NVLink/IB) that decides whether TP/FSDP are viable.
- `[[gke-master]]` — picking TPU/GPU node pools and accelerator SKUs once you know the memory class.
- `[[ai-research-science]]` — scaling laws / architecture choices that set P, layers, hidden, heads.

---

# Reference — accelerator-memory-estimator

# Accelerator Memory Estimator — Guide

This guide is a **procedure plus the formulas you apply** to produce a numeric estimate of GPU/TPU
memory for a given model and config, and to recommend a fitting strategy. It is not a survey.

**Read this first:** every number this guide produces is an **estimate (±)**. It deliberately ignores
exact kernel buffers, allocator fragmentation, cuDNN/cuBLAS workspaces, NCCL buffers, and
framework-version quirks, which together commonly add **10–30%**. Use the estimate to choose the right
*class* of hardware and parallelism, then **confirm with a real run** (`nvidia-smi`,
`torch.cuda.max_memory_allocated()`, a JAX memory profile, or an OOM test on a small slice). When you
report, always write the number as an estimate and tell the reader to verify.

Units: 1 GiB = 1024³ bytes ≈ 1.074e9 bytes. Param counts use 1e9 = 1 "B". When you divide bytes by
1.074e9 you get GiB; people say "GB" loosely — be explicit which you mean. This guide reports GiB.

---

## Mental model

A model occupies accelerator memory as a sum of a few independent terms. **Compute each term, then add
them, then compare to device capacity minus headroom.** Nothing here is mysterious; OOMs come from
forgetting a term (optimizer state, activations, KV cache) or from per-device math after sharding.

**Two regimes, different terms:**

- **Training** per device = weights + gradients + optimizer state + activations + overhead. The first
  three scale with **params P**; activations scale with **batch × seq × hidden × layers**.
- **Inference / serving** per device = weights + **KV cache** + activation/workspace. Weights scale
  with P; KV cache scales with **batch × seq × layers × kv_heads**. At long context / high concurrency
  the **KV cache, not the weights, is the binding constraint**.

Bytes-per-element by dtype: **fp32 = 4, bf16/fp16 = 2, fp8 = 1, int8 = 1, int4 = 0.5.**

---

## Part 1 — Training memory

Let **P** = number of parameters (e.g. 7B = 7e9). For a standard mixed-precision setup (bf16 compute,
fp32 optimizer master copy), the per-step resident memory for the model state is:

| Term | Formula (bytes) | 7B value | Why |
|------|-----------------|----------|-----|
| Weights (bf16) | `2·P` | 14.0 GiB | forward/backward copy in compute dtype |
| Gradients (bf16) | `2·P` | 14.0 GiB | one grad per weight |
| Optimizer: fp32 master | `4·P` | 28.0 GiB | high-precision master weights |
| Optimizer: Adam m | `4·P` | 28.0 GiB | first moment (fp32) |
| Optimizer: Adam v | `4·P` | 28.0 GiB | second moment (fp32) |
| **Model-state subtotal** | **`16·P`** | **~104 GiB** (97.7 GiB at 1024³) | the famous "16 bytes/param" |

So a naive **AdamW + bf16** training step needs **≈ 16·P bytes before activations**. For 7e9 params
that is 16 × 7e9 = 1.12e11 bytes ≈ **104.3 GiB** (÷1.074e9). Already larger than one 80GB H100 — see
Examples (a).

> If you train in pure fp32 (no mixed precision), weights and grads become 4·P each, optimizer stays
> 12·P → 20·P. Most large-model training uses bf16/mixed precision; assume 16·P unless told otherwise.

### Optimizer variants (levers that change the 12·P term)

| Optimizer | State bytes | Note |
|-----------|-------------|------|
| AdamW (fp32 master + m + v) | `12·P` | default; the 16·P total |
| 8-bit Adam (bitsandbytes) | `~6·P` (≈2·P master held bf16 + 2·P + 2·P 8-bit moments) | ~halves optimizer state; tiny accuracy cost |
| SGD + momentum | `~8·P` (master 4·P + momentum 4·P) | no second moment |
| SGD (no momentum) | `~4·P` | rare for LLMs |
| Adafactor | `≪ 12·P` (factored second moment) | popular on TPU/large models |

State these explicitly — silently assuming 8-bit Adam can make a "fits" verdict wrong.

### Activation memory

Activations are the tensors saved during the forward pass for use in the backward pass. They scale
with the **batch and sequence**, unlike the model-state terms. A useful estimate for a transformer:

```
activations ≈ batch · seq · hidden · layers · A_bytes
```

where `A_bytes` is a per-element multiplier capturing how many activation tensors per layer are
retained (attention scores, MLP intermediates, layernorm inputs, residuals). It depends heavily on
implementation (FlashAttention removes the O(seq²) attention-score buffer). Practical ranges:

- **No activation checkpointing, bf16, FlashAttention:** `A_bytes` ≈ **30–70 bytes/element** of
  `batch·seq·hidden·layers`. Use ~50 as a central estimate; this is the *least* certain term — flag it.
- The O(seq²) attention term (`batch · heads · seq² · layers · 2`) is only resident **without**
  FlashAttention. Assume FlashAttention unless told otherwise; if not, add it.

**Activation checkpointing (gradient checkpointing / recompute):** store only the inputs at layer
boundaries and **recompute** intermediates during backward. This trades ~33% extra compute for a large
memory cut.

- Full per-layer checkpointing: activation memory drops roughly to the boundary tensors —
  `≈ batch · seq · hidden · layers · (small bytes)` plus one layer's worth of live recompute. In
  practice a **~5–10× reduction**; a common rough model is the activation term falling toward
  `O(√layers)` of the original when checkpointing every √layers blocks (selective checkpointing).
- Treat checkpointing as: **activations_checkpointed ≈ activations_full / R**, with **R ≈ 5–10**
  (state which R you assumed).

8-bit/fp8 activations and selective recompute reduce this further; note them but don't assume.

### Overhead

Add **framework + fragmentation + CUDA context + collective buffers**:

```
overhead ≈ 0.10–0.20 · (model_state + activations)   plus ~1–2 GiB CUDA/driver context per GPU
```

Use **+15%** as a central estimate. NCCL/communication buffers for FSDP/TP add more (often 1–3 GiB).

### Training per-device total

```
mem_train_per_device =
    ( weights + gradients + optimizer )           # model state, see sharding below
  + activations (× 1/R if checkpointing)          # per local microbatch
  + overhead (~15% + ~1–2 GiB context)
```

---

## Part 2 — Sharding math (parallelism)

Parallelism reduces **per-device** memory by dividing one or more terms across devices. Let **N** =
data-parallel world size, **k** = tensor-parallel size, **s** = pipeline stages.

### FSDP / ZeRO — what each stage shards (divide by N)

| Strategy | Weights | Gradients | Optimizer | Per-device model state |
|----------|---------|-----------|-----------|------------------------|
| DDP (ZeRO-0) | full | full | full | `16·P` (replicated) |
| **ZeRO-1** | full | full | **/N** | `2·P + 2·P + 12·P/N` |
| **ZeRO-2** | full | **/N** | **/N** | `2·P + (2·P + 12·P)/N` |
| **ZeRO-3 / FSDP (full shard)** | **/N** | **/N** | **/N** | **`16·P / N`** |

ZeRO-3/FSDP all-gathers each layer's weights just-in-time, so add a **transient all-gather buffer**
(~the largest layer's weights, a few GiB) on top of `16·P/N`. Activations are **not** sharded by
FSDP/ZeRO — they scale with the *local* microbatch on each device.

- **ZeRO-3/FSDP** is the workhorse for fitting large models on many GPUs: per-device model state ≈
  `16·P/N`. For 70B on N=8: 16 × 70e9 / 8 = 1.4e11 bytes ≈ **130 GiB/GPU model state** — still too big
  for 80GB, so combine with more devices or checkpointing/TP (see Examples b).
- **ZeRO-2** is cheaper in comms (no weight gather) but keeps full weights resident — only use when
  weights alone fit per device.

### Tensor parallel (TP=k) — divide per-layer tensors by k

Splits each layer's weight matrices and the corresponding activations across k devices:
`weights/k`, `activations/k` (the MLP/attention intermediates). High communication (all-reduce every
layer) → **keep TP within a node / NVLink domain** (k ≤ 8 typically). Combine with FSDP across nodes
(2D parallelism): per-device model state ≈ `16·P / (N · k)` when both shard the relevant terms.

### Pipeline parallel (PP=s) — split layers across stages

Each stage holds `layers/s` layers → model state per stage ≈ `16·P / s`. Costs:
- **Pipeline bubble** (idle time ∝ (s−1)/microbatches) — wastes compute, not memory.
- **In-flight microbatches:** stage memory must hold activations for several microbatches at once
  (1F1B schedule keeps ~`s` microbatches live on stage 0) — add that activation multiple.

### Putting parallelism together (3D)

```
model_state_per_device ≈ 16·P / (N_fsdp · k_tp · s_pp)   # for the terms each axis actually shards
activations_per_device ≈ activations(local microbatch) / k_tp / R   # TP shards activations; PP adds in-flight copies
```

Recommend the **simplest axis that fits**: prefer FSDP/ZeRO-3 (data-parallel, easy) → add activation
checkpointing → add TP within node → add PP across nodes only when one node can't hold a pipeline stage.

---

## Part 3 — Inference / serving memory

```
mem_infer_per_device = weights + KV_cache + activation/workspace
```

### Weights

```
weights ≈ P · weight_bytes
```
`weight_bytes` = 2 (bf16), 1 (int8/fp8), 0.5 (int4/NF4). Quantization-aware serving (AWQ/GPTQ/int4)
roughly **quarters** weight memory vs bf16 (plus small scales/zeros overhead, ~3–5%).

### KV cache — usually the binding constraint

For each token, each layer caches a Key and a Value vector per KV head:

```
KV_cache_bytes = 2 · layers · kv_heads · head_dim · seq_len · batch · dtype_bytes
```

- The leading **2** = K and V.
- **kv_heads** is the number of *key/value* heads. With **MHA**, kv_heads = num query heads. With
  **GQA**, kv_heads = num query heads / group size (e.g. Llama-3-70B: 64 query heads, **8 kv heads** →
  8× smaller cache). With **MQA**, kv_heads = 1.
- `layers · kv_heads · head_dim` is the per-token KV "width". A handy equivalent form:
  `KV_per_token = 2 · layers · kv_heads · head_dim · dtype_bytes` bytes/token, then multiply by
  `seq_len · batch`.
- **dtype_bytes** = 2 for bf16/fp16 KV; 1 for fp8/int8 KV cache (common in vLLM/TensorRT-LLM to double
  capacity).

This grows **linearly in both context length and concurrency** — it is what limits how many concurrent
sequences and how long a context you can serve. PagedAttention (vLLM) removes fragmentation waste but
does not change this total.

### Activation / workspace

Decode steps are tiny (batch×1 token). **Prefill** of a long prompt creates a large transient
activation/attention workspace (∝ batch · prompt_len · hidden). Reserve a few GiB, more for big prefill
batches; chunked prefill bounds it. Use **+2–4 GiB** as a central workspace estimate unless prefill is
huge.

### Serving total and the free-KV question

```
KV_budget = device_HBM · num_devices − weights − workspace − headroom(~10–15%)
max_tokens_in_flight ≈ KV_budget / KV_per_token
```

That `max_tokens_in_flight` (≈ Σ over sequences of their lengths) is the real concurrency limit.
Report it — it answers "how many users / how long a context fits".

---

## Part 4 — Device capacities (per accelerator, approximate — verify the SKU)

| Accelerator | HBM (approx) | Notes |
|-------------|--------------|-------|
| NVIDIA H100 SXM / H200 | 80 GiB / 141 GiB | H200 ~141 GiB HBM3e |
| NVIDIA A100 | 40 or 80 GiB | two SKUs — confirm which |
| NVIDIA L4 / L40S | 24 / 48 GiB | inference-oriented |
| NVIDIA RTX 4090 / consumer 24GB | 24 GiB | common for QLoRA |
| TPU v5e | 16 GiB HBM / chip | many chips per host |
| TPU v5p | 95 GiB HBM / chip | training-class |
| TPU v6e (Trillium) | 32 GiB HBM / chip | verify current spec |

Always leave **~10–20% headroom** for fragmentation/spikes; never plan to 100% capacity. Newer SKUs
appear constantly (it is 2026) — **confirm the exact HBM of the SKU you target**; do not trust a
remembered number.

---

## The procedure (apply this end to end)

1. **Collect inputs.** Either param count **P**, or architecture dims (layers L, hidden H, heads, kv
   heads, head_dim, vocab) — then estimate P (for a transformer, `P ≈ 12 · L · H²` plus embeddings, a
   rough check). Also: dtype, optimizer, batch, seq_len, # devices, per-device HBM, whether
   training or inference, and any quantization/LoRA/checkpointing.
2. **Pick the regime** (training vs inference) and write the term list.
3. **Compute each term in bytes** using the formulas above. Show the arithmetic.
4. **Apply sharding** (÷N for FSDP/ZeRO-3 model state; ÷k for TP; etc.) to get **per-device** bytes.
5. **Add overhead** (~15% + context/workspace).
6. **Convert to GiB** (÷1.074e9) and **compare to capacity − headroom**.
7. **Recommend the cheapest strategy that fits** (decision tree below). State the fit margin.
8. **Label it "estimate, verify"** and name the verification (small-slice OOM test / `nvidia-smi` /
   profiler).

### Recommendation decision tree

```
Does model state + activations + overhead fit on ONE device?
├─ yes  → fits as-is (single device). Suggest largest batch/seq that still fits.
└─ no   → TRAINING:
          1. Enable activation checkpointing (÷R≈5–10 on activations). Re-check.
          2. Shard with FSDP/ZeRO-3 across N devices (model state → 16·P/N). Re-check.
          3. Add tensor parallel TP=k within a node (per-layer ÷k). Re-check.
          4. Add pipeline parallel across nodes if one node can't hold a stage.
          5. Reduce optimizer cost: 8-bit Adam (12·P→~6·P) / Adafactor.
          6. Still too big or too few GPUs → PEFT: LoRA (freeze weights) or QLoRA
             (4-bit base weights + adapters) — drops optimizer/grad terms to the adapter only.
       →  INFERENCE:
          1. Quantize weights (int8 → ~1·P, int4 → ~0.5·P). Re-check.
          2. Shrink KV: fp8/int8 KV cache (÷2), shorter max context, GQA model.
          3. Tensor-parallel across k devices (weights & KV ÷k).
          4. Compute max_tokens_in_flight for the chosen config; set max-num-seqs accordingly.
```

---

## Rules of thumb (quick table)

| Scenario | Per-device memory estimate | Notes |
|----------|----------------------------|-------|
| Full fine-tune / pretrain (bf16 + AdamW) | **≈ 16·P bytes + activations + ~15%** | the default; activations can dominate at large batch×seq |
| + activation checkpointing | activations ÷ ~5–10 | ~33% more compute |
| ZeRO-3 / FSDP across N | model state **÷N** (+ all-gather buffer) | activations not sharded |
| 8-bit Adam | optimizer 12·P → ~6·P (total ~10·P) | small accuracy cost |
| **LoRA** | frozen weights (2·P bf16) + tiny adapter + adapter optimizer | no full grads/optimizer for base |
| **QLoRA** | **~0.5·P (4-bit base) + adapters** | 7B fine-tunes on a single 24GB GPU |
| Inference (bf16) | **weights (2·P) + KV cache + ~2–4 GiB** | KV often the limiter |
| Inference (int4 weights) | **~0.5·P + KV + workspace** | AWQ/GPTQ; verify accuracy |
| KV cache | `2 · L · kv_heads · head_dim · seq · batch · dtype_bytes` | GQA/MQA shrink kv_heads |

Memorize: **training ≈ 16·P + activations; inference ≈ weights + KV.**

---

## Anti-patterns / gotchas

- **Forgetting optimizer state.** "It's a 7B, bf16 is 14 GB, fits on a 24GB card" — wrong for training:
  16·P ≈ 104 GiB. The 14 GB number is *inference weights only*.
- **Forgetting activations.** Model state can fit while activations OOM at high batch×seq. Always add
  the activation term; it's the one that scales with your batch.
- **Forgetting the all-gather buffer in FSDP.** `16·P/N` is the *sharded resident* state; gathering a
  layer adds a transient buffer. Leave headroom.
- **Confusing GB and GiB**, and counting "16-byte" optimizer as "16 GB" instead of `16·P` bytes.
- **Using MHA kv_heads for a GQA model** → KV cache overestimated ~8×. Check the actual kv_heads.
- **Planning to 100% of HBM.** Fragmentation and transient spikes OOM you near the top; keep 10–20%.
- **Assuming 8-bit/Adafactor/checkpointing without saying so** — it changes the verdict; state every
  assumption.
- **Treating the estimate as exact.** It's ±; kernels/workspaces/NCCL buffers add 10–30%. Verify.

## Rationalizations & rebuttals

| Excuse | Rebuttal |
|--------|----------|
| "bf16 weights fit, so training fits." | Training is 16·P, not 2·P. Add grads + optimizer + activations. |
| "I'll just lower batch size if it OOMs." | Estimate first; you may need FSDP/TP regardless of batch. |
| "FSDP makes it 8× smaller, done." | Only the *model state* divides by N; activations don't, and there's an all-gather buffer. |
| "KV cache is negligible." | At seq=8k, batch=32 it can exceed the weights. Compute it. |
| "Estimate is close enough, skip the test run." | ±10–30% can be the difference between fit and OOM. A 5-minute small-slice run is cheap insurance. |
| "We'll quantize later if needed." | Quantization changes whether it fits *now*; fold it into the estimate before buying GPUs. |

## Red flags (stop and reconsider)

- A "fits" verdict with **<10% headroom** → treat as does-not-fit; one fragmentation spike OOMs.
- You quoted memory without naming **dtype, optimizer, batch, and seq** → the number is meaningless.
- You divided activations by world size → wrong; FSDP/ZeRO don't shard activations.
- You sized serving on weights alone and ignored KV at the target context/concurrency.
- Your P came from a guess, not the config/checkpoint → re-derive from dims or read the param count.

## Verification gate (definition of done for an estimate)

Before reporting, confirm all of these:

1. **Inputs stated:** P (or dims → P), dtype, optimizer, batch, seq, #devices, per-device HBM, regime.
2. **Every term shown** with its formula and byte value; per-device sum after sharding; overhead added.
3. **Converted to GiB** and compared to **capacity − ~10–20% headroom**; fit margin stated.
4. **A concrete recommendation** (fits / FSDP / TP=k / checkpointing / quantization / QLoRA) with the
   reasoning.
5. **Labeled "estimate (±), verify"** with the named verification: run a **small-slice OOM test** and
   read peak memory (`torch.cuda.max_memory_allocated()` / `nvidia-smi` / JAX memory profiler), or check
   the serving engine's reported KV budget, before committing hardware.

## Version awareness

Accelerator SKUs, HBM sizes, optimizer implementations (8-bit Adam, Adam-mini, fused optimizers), KV
quantization, and framework memory behavior change quickly — it is 2026. Treat the capacity table and
the activation/overhead multipliers as **starting estimates** and confirm against current vendor specs
and your framework version. Never invent a SKU's HBM or a benchmark number; if unsure, say so.

## Canonical references

- ZeRO / memory math: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models" — https://arxiv.org/abs/1910.02054
- Activation recomputation: Chen et al., "Training Deep Nets with Sublinear Memory Cost" —
  https://arxiv.org/abs/1604.06174 ; Korthikanti et al., "Reducing Activation Recomputation in Large
  Transformer Models" — https://arxiv.org/abs/2205.05198
- Megatron tensor/pipeline parallelism — https://arxiv.org/abs/1909.08053 ,
  https://arxiv.org/abs/2104.04473
- GQA — Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" —
  https://arxiv.org/abs/2305.13245
- PagedAttention / vLLM KV cache — https://arxiv.org/abs/2309.06180 and https://docs.vllm.ai
- 8-bit optimizers — Dettmers et al. — https://arxiv.org/abs/2110.02861 ; QLoRA —
  https://arxiv.org/abs/2305.14314
- PyTorch FSDP docs — https://pytorch.org/docs/stable/fsdp.html ; DeepSpeed ZeRO —
  https://www.deepspeed.ai/tutorials/zero/
- Verify accelerator HBM against current NVIDIA / Google Cloud TPU documentation.

---

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
