# AGENTS.md — Accelerator Memory Estimator

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full formulas + procedure live in **`accelerator-memory-estimator-guide.md`** next to this file —
> read it and apply it. Fully worked numeric estimates to imitate are in **`examples.md`**.
>
> This is a **doer**: given a model + config, COMPUTE per-device GPU/TPU memory and RECOMMEND a fitting
> strategy. **Every output is an estimate (±10–30%) — always tell the user to verify against a real run.**

## When asked "will it fit / how much VRAM / how many GPUs / how big is the KV cache", do this:

1. **Collect inputs:** params P (or dims → P), dtype, optimizer, batch, seq_len, #devices, per-device
   HBM, training vs inference, any quantization/LoRA/checkpointing. State assumptions explicitly.
2. **Compute terms in bytes, show the arithmetic, sum per device, compare to capacity − ~15% headroom.**

## Core formulas (memorize)

- **Training model state ≈ 16·P bytes** (bf16 + AdamW): weights `2·P` + grads `2·P` + optimizer `12·P`
  (fp32 master `4·P` + Adam m `4·P` + v `4·P`). Add activations + ~15% overhead.
- **Activations ≈ batch · seq · hidden · layers · A_bytes** (A_bytes ~50, least-certain term). Activation
  **checkpointing** ÷ R≈5–10 (costs ~33% compute). Not sharded by FSDP/ZeRO.
- **8-bit Adam:** optimizer 12·P → ~6·P. **Adafactor:** ≪12·P. fp8/8-bit weights/grads cut further.
- **Sharding (per-device model state):** ZeRO-1 = `2·P+2·P+12·P/N`; ZeRO-2 = `2·P+(14·P)/N`;
  **ZeRO-3/FSDP = `16·P/N`** (+ transient all-gather buffer). **TP=k** divides per-layer weights &
  activations by k (keep within a node). **PP=s** splits layers (÷s), adds in-flight microbatch activations.
- **Inference ≈ weights + KV cache + ~2–4 GiB workspace.** weights = P·{2 bf16, 1 int8, 0.5 int4}.
- **KV cache = 2 · layers · kv_heads · head_dim · seq_len · batch · dtype_bytes.** GQA/MQA shrink
  kv_heads (e.g. 8 vs 64 → ~8×). KV usually caps serving concurrency/context, not weights.
- **dtype bytes:** fp32=4, bf16/fp16=2, fp8/int8=1, int4=0.5. **GiB = bytes ÷ 1.074e9.**

## Rules of thumb

- Full FT ≈ 16·P + activations. LoRA ≈ frozen weights + tiny adapter. **QLoRA ≈ ~0.5·P + adapters**
  (7B on a 24GB card). Inference ≈ weights + KV.
- Quick check: 7B full FT ≈ 16×7e9 ≈ 104 GiB → does NOT fit one 80GB GPU. 7B inference bf16 ≈ 14 GiB +
  KV → fits.
- Leave **10–20% HBM headroom**; never plan to 100%.

## Recommend the cheapest strategy that fits

Single device fits? → done. Else (training): activation checkpointing → FSDP/ZeRO-3 (÷N) → TP within
node → PP across nodes → 8-bit Adam/Adafactor → LoRA/QLoRA. Inference: quantize weights → fp8/int8 KV /
shorter context → tensor parallel → compute `max_tokens_in_flight = KV_budget / KV_per_token`.

## Definition of done for an estimate

Inputs stated · every term shown with formula + bytes · per-device sum after sharding + overhead ·
converted to GiB and compared to capacity − headroom with fit margin · concrete recommendation ·
**labeled "estimate (±), verify"** with a named check (small-slice OOM test, `nvidia-smi`,
`torch.cuda.max_memory_allocated()`, or the serving engine's reported KV budget).

## Gotchas

Don't quote bf16 weights as the training cost (that's 2·P, not 16·P). Don't shard activations by N.
Don't forget the FSDP all-gather buffer. Use the model's real **kv_heads** (GQA), not query heads.
Confirm the exact SKU HBM (it is 2026; SKUs change). The estimate is ± — say so.
