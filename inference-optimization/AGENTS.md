# AGENTS.md — Inference Optimization (Model Compression & Decode Acceleration)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`inference-optimization-guide.md`** next to this file —
> read it before quantizing, pruning, distilling, or wiring up speculative decoding, and apply it.
> Concrete flows to imitate are in **`examples.md`**. This file is the always-on summary.
>
> Scope: making a model **smaller/faster/cheaper to serve at the model level**. The engine that *runs*
> the result (vLLM/SGLang/TensorRT-LLM/Triton, paged KV, continuous batching) is `[[serving-frameworks]]`.
>
> **This field moves quarterly (it is 2026).** Name techniques; verify every flag, dtype, format, and
> hardware-support claim against current docs. Never fabricate an API, a format, or a benchmark number.

## Apply these by default

- **Find the bound before optimizing.** Decode = **memory-bandwidth-bound** (rereads all weights + KV
  per token); prefill and large-batch decode = **compute-bound**. Compute arithmetic intensity vs the
  roofline; profile at your real operating point (batch, context, SLO). Optimizing the wrong bound buys
  nothing — this is the #1 wasted effort.
- **Be explicit about what each lever buys** — {latency, throughput, memory, $/token, quality}. Weight
  quant → memory + bandwidth-bound decode. Weight+activation quant (FP8/INT8) → also compute/prefill.
  Spec-decode → latency at fixed batch (spends spare compute; can hurt at max throughput). Distillation
  → a permanently smaller model. KV-cache quant → concurrency/throughput.
- **Quantize first; default PTQ.** PTQ needs a small **calibration set (~128–512 real samples,
  including the chat template/languages you serve)**. Escalate to **QAT** only when PTQ recovery is
  insufficient (sub-4-bit / W4A4). On FP8-capable HW, **FP8** weights+activations is the near-lossless
  default; on older HW or tight memory, **INT4 weight-only (AWQ/GPTQ)**. **SmoothQuant** for W8A8.
  **GGUF** = CPU/edge; **NF4** = QLoRA fine-tuning.
- **Quantize the KV cache (FP8/INT8)** — for long context / large batch it caps concurrency more than
  weights do; often the bigger serving win. Keys more sensitive than values.
- **Sparsity that pays on GPU is `2:4` semi-structured** (Ampere+ Sparse Tensor Cores), then recover by
  fine-tune/distill. Unstructured sparsity compresses storage, not dense-GEMM time. **Structured
  pruning** (heads/width/layers) → smaller dense model but **needs distillation to recover**.
- **Distillation needs data.** Student quality ∝ diversity/volume of teacher-labeled data on your
  distribution. Prefer **on-policy/sequence-level** distillation for generative/reasoning students.
  Never distill on thin or off-distribution data.
- **Speculative decoding only pays with high acceptance and a cheap draft.** Speedup ≈ accepted tokens
  per target pass ÷ draft cost. Output is *exact* with correct verification. EAGLE/Medusa/self-spec
  reuse the target's features; **n-gram/prompt-lookup** is near-free when output echoes input. Always
  measure realized acceptance + end-to-end tokens/s on real traffic; draft must share the tokenizer.
- **Architecture levers (often baked in):** **GQA/MQA/MLA** shrink the KV cache; **FlashAttention** is
  mandatory IO-aware exact attention (no quality change); **MoE** cuts active FLOPs but raises memory.
- **Compile and fuse (quality-neutral — take them):** `torch.compile`/Inductor (watch graph breaks),
  **CUDA graphs** (decode launch-overhead win), TensorRT-LLM engine build (arch/shape-specific —
  **rebuild per GPU gen / shape profile**), ONNX Runtime, XLA, custom **Triton/CUTLASS** kernels.
- **Stack levels (data/model/system) deliberately; eval after each.** Typical: FP8/INT4 weights + FP8
  KV + FlashAttention + compile/engine + CUDA graphs + spec-decode. Add one lever, measure, keep/revert.

## Definition of done for an optimization change
- Profiled the bound; chose levers that target it.
- **Eval on a representative task suite (not just perplexity), before and after** — report quality delta
  and the target metric (TTFT/ITL/tokens-per-s/memory/$).
- For spec-decode: realized acceptance rate measured on real traffic.
- For TensorRT-LLM/engine builds: built for the actual GPU arch + shape profile.
- No fabricated flags/formats/numbers; fast-moving specifics flagged "verify against current docs".

## Anti-patterns (reject these)
quantize-without-eval · optimizing the wrong bound · distill-without-data · spec-decode with low
acceptance · calibration-set mismatch · unstructured sparsity expecting GPU speedup · low-bit on
softmax/layernorm/embeddings · reusing a TensorRT engine across GPU gens/shapes · trusting a leaderboard
number that isn't your workload.
