---
name: inference-optimization
description: Model-level LLM inference optimization and efficiency — making models smaller, faster, and
  cheaper to serve at the level of an engineer who squeezes frontier models onto fewer accelerators.
  Use when quantizing (PTQ vs QAT, INT8/INT4, FP8, GPTQ, AWQ, SmoothQuant, GGUF, NF4, KV-cache quant),
  pruning/sparsifying (structured, 2:4 semi-structured, layer/width pruning), distilling
  (teacher→student, sequence-level/on-policy), or speeding up decode with speculative decoding (draft
  models, Medusa, EAGLE, lookahead, n-gram, self-speculative). Also covers low-rank/structural
  efficiency (LoRA, MoE, GQA/MQA/MLA, FlashAttention) and compilation/kernels (torch.compile/Inductor,
  TensorRT-LLM engine build, ONNX Runtime, XLA, Triton/CUTLASS, CUDA graphs). Reach for it to reason
  about the memory-bandwidth vs compute bound, arithmetic intensity, accuracy recovery, acceptance-rate
  economics, and which technique buys latency vs throughput vs memory vs $/token vs quality. The serving
  *engine* that runs the result is `[[serving-frameworks]]`; this is the compression/decode-acceleration
  layer underneath it.
---

# Inference Optimization (Model Compression & Decode Acceleration)

Apply the judgment of an engineer who has shipped frontier models onto a fraction of the accelerators
they were trained on: who knows that **decode is memory-bandwidth-bound and prefill is compute-bound**,
that every optimization is a trade against quality, and that the only honest way to ship one is to
**eval before and after on the workload that matters**.

## How to use this skill

1. **Read `inference-optimization-guide.md`** in this directory — the full reference. It builds the
   roofline/arithmetic-intensity mental model first (this is *why* every technique works), then covers
   quantization, pruning/sparsity, distillation, speculative decoding, structural/low-rank efficiency,
   and compilation/kernels, with a decision guide and anti-patterns.
2. For concrete flows to imitate (INT4/AWQ quantization, TensorRT-LLM build-then-serve, speculative
   decoding config), read **`examples.md`**.
3. Match the surrounding stack's conventions (framework, accelerator class, serving engine). Apply the
   correctness rules — **always eval the optimized model on a representative set** — regardless.
4. **This field moves very fast (it is 2026).** Methods, kernel names, format names, and hardware
   support change quarterly. Name techniques, but verify current APIs, flags, supported dtypes, and
   "does my hardware/engine support this" against the project's own docs before committing. Never
   fabricate a flag, a format, or a benchmark number.

## Essentials (full detail in `inference-optimization-guide.md`)

- **Know your bound before you optimize.** Decode is **memory-bandwidth-bound** (each step rereads all
  weights + KV to emit one token); prefill and large-batch decode are **compute-bound**. Compute
  arithmetic intensity (FLOPs ÷ bytes moved) vs the hardware roofline. Optimizing the wrong bound
  (e.g. faster kernels on a bandwidth-bound decode) buys nothing.
- **Each technique buys a different thing.** Weight-only quant → less memory + faster bandwidth-bound
  decode. Weight+activation quant (FP8/INT8) → also faster compute-bound prefill. Speculative decoding
  → lower latency at *fixed* batch, costs extra compute. Distillation → a permanently smaller/cheaper
  model. Be explicit about which of {latency, throughput, memory, $/token, quality} you are buying.
- **Quantization is the highest-leverage lever.** **Weight-only INT4** (GPTQ/AWQ) roughly halves model
  memory vs FP8 and speeds bandwidth-bound decode; **FP8** (weights+activations) on Hopper/Blackwell-class
  GPUs accelerates compute too with near-lossless quality. **SmoothQuant** migrates activation outliers
  into weights to make W8A8 viable. **GGUF** is the llama.cpp/CPU/edge format; **NF4** is QLoRA's
  4-bit weight format for fine-tuning.
- **KV-cache quantization** (FP8/INT8 KV) is often a bigger serving win than weight quant for long
  contexts and large batches — the KV cache, not the weights, is what caps your concurrency.
- **PTQ vs QAT.** Start with **post-training quantization** (cheap, calibration set of ~128–512
  samples). Drop to **quantization-aware training** only when PTQ accuracy recovery is insufficient
  (typically INT4 weight+activation, or aggressive ≤4-bit). QAT costs a training run.
- **Sparsity that hardware accelerates is `2:4` semi-structured** (2 of every 4 weights zero) on
  Ampere+; unstructured sparsity rarely speeds dense GEMMs. **Structured pruning** (drop heads/layers/
  width) shrinks the model for real but needs distillation/fine-tuning to recover.
- **Speculative decoding only pays when acceptance is high.** Speedup ≈ accepted tokens per draft step;
  a draft model that is too weak (low acceptance) or too slow erases the win. EAGLE/Medusa/self-spec
  reuse the target model's features to lift acceptance without a separate well-aligned draft.
- **GQA/MQA/MLA** shrink the KV cache by sharing K/V across query heads (architecture-level, usually
  baked in at pretraining). **FlashAttention** is the must-have IO-aware attention kernel. **MoE**
  cuts active FLOPs per token but raises total memory.
- **Compile and fuse.** `torch.compile` (Inductor), TensorRT-LLM engine builds, ONNX Runtime, and XLA
  fuse kernels and cut launch overhead; **CUDA graphs** kill per-step launch latency in decode. Custom
  **Triton/CUTLASS** kernels are the last mile for quantized/fused ops.
- **Stack techniques deliberately, eval after each.** A common production recipe is *FP8 (or INT4)
  weights + FP8 KV cache + FlashAttention + CUDA graphs + speculative decode*. Combining is normal;
  combining blind is how quality silently regresses.
- **Anti-patterns:** quantize-without-eval, distill-without-(enough)-data, spec-decode with low
  acceptance, optimizing the wrong bound, and chasing a benchmark number that isn't your workload.

## Related skills

- `[[serving-frameworks]]` — the engines (vLLM, SGLang, TensorRT-LLM, Triton, Dynamo) that *run* the
  optimized model and supply paged KV/continuous batching. This skill produces the artifact they serve.
- `[[ml-frameworks]]` — PyTorch/JAX/XLA, GPU & TPU, where `torch.compile`, kernels, and dtypes live.
- `[[fine-tuning-peft]]` — LoRA/QLoRA and PEFT; the training-time counterpart to low-rank inference
  efficiency and the QAT/distillation training loops.
- `[[gke-inference-gateway]]` — routing/load-balancing the optimized model behind an inference gateway.
- `[[aiml-on-kubernetes]]` — the umbrella for training/inference/fine-tuning on K8s & GKE.
- `[[gke-master]]` — GKE accelerator node pools (GPU/TPU classes) you size *after* optimizing.
