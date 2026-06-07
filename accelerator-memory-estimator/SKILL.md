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
