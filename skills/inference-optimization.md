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

---

# Reference — inference-optimization

# Inference Optimization — Deep Reference (Model Compression & Decode Acceleration)

The full reference for making LLMs **smaller, faster, and cheaper to serve at the model level** — the
techniques an engineer uses to fit a frontier model onto fewer accelerators without giving up the
quality that mattered. Read the roofline mental model first; it is the backbone that explains *why*
every technique below works and *when* it does nothing. The serving *engine* that runs the result
(paged KV, continuous batching, scheduler) is `[[serving-frameworks]]`; this guide is the layer
underneath — the model artifact and the decode algorithm.

This field moves quarterly. Method names, format names, kernel names, and hardware support change.
Treat every specific flag, dtype, and "supported?" claim as something to **verify against current
docs** before you commit it to a config.

---

## 1. Mental model: the roofline, and which bound you are on

One forward pass over a transformer does two arithmetically different jobs:

1. **Prefill** — process the whole prompt at once. Many tokens × big matmuls → high **arithmetic
   intensity** (FLOPs per byte moved from memory). The GPU's tensor cores (FLOPs) are the limit →
   **compute-bound**. Latency here = TTFT.
2. **Decode** — generate one token per step. Each step must reread *all the weights* and the *entire
   KV cache* from HBM to produce a single token. Tiny matmuls, enormous memory traffic, low arithmetic
   intensity → **memory-bandwidth-bound**, and inherently sequential. Latency here = ITL / TPOT.

**Arithmetic intensity** `I = FLOPs / bytes_moved`. The **roofline**: achievable FLOP/s =
`min(peak_FLOPs, I × peak_bandwidth)`. If `I` is below the ridge point (`peak_FLOPs / peak_bandwidth`,
often a few hundred FLOP/byte on modern accelerators — verify per device), you are bandwidth-bound and
*adding compute does nothing*; the only levers are **move fewer bytes** (quantize weights/KV, smaller
model) or **reuse bytes across more work** (bigger batch).

Decode at batch size 1 is deeply bandwidth-bound: you read ~`2 × num_params` bytes (FP16) of weights to
emit one token. This is why:

- **Weight-only quantization** speeds decode almost linearly with the byte reduction (INT4 ≈ 4× fewer
  weight bytes than FP16) even though it does no real compute math at lower precision.
- **Batching raises arithmetic intensity** — one weight read now serves N sequences — and eventually
  pushes large-batch decode toward compute-bound. So the *same model* can be bandwidth-bound for a
  latency-sensitive single user and compute-bound for a throughput-maximizing batch. The optimization
  that helps depends on the operating point.
- **Speculative decoding** attacks the sequential dependency, not the bound: it verifies several draft
  tokens in one target forward pass, turning low-intensity per-token decode into a higher-intensity
  batched verify.

**Internalize this:** before any optimization, identify the bound at your real operating point (batch,
context length, SLO). Profile it (e.g. with the framework profiler / Nsight) rather than guessing.
Optimizing the wrong bound is the #1 wasted-effort anti-pattern.

### What each family actually buys

| Technique | Memory | Decode latency | Throughput | $/token | Quality risk |
|---|---|---|---|---|---|
| Weight-only quant (INT4/INT8) | ↓↓ | ↓↓ (bw-bound) | ↑ | ↓↓ | low–med |
| Weight+activation quant (FP8/INT8) | ↓↓ | ↓ | ↑↑ (also prefill) | ↓↓ | med |
| KV-cache quant (FP8/INT8) | ↓↓ (long ctx) | ↓ | ↑↑ (more concurrency) | ↓↓ | low–med |
| 2:4 structured sparsity | ↓ | ↓ (on supported HW) | ↑ | ↓ | med |
| Structured pruning (layers/width) | ↓↓ | ↓↓ | ↑↑ | ↓↓ | high (needs recovery) |
| Distillation (smaller student) | ↓↓↓ | ↓↓↓ | ↑↑↑ | ↓↓↓ | high (needs data) |
| Speculative decoding | ≈ (extra) | ↓↓ (fixed batch) | ≈/↓ at max batch | ≈ | none if verified |
| GQA/MQA/MLA | ↓↓ (KV) | ↓ | ↑↑ | ↓ | baked in pretrain |
| Compile/kernels/CUDA graphs | ≈ | ↓ (launch ovhd) | ↑ | ↓ | none |

`≈` = roughly unchanged. Note speculative decoding trades *spare compute* for latency and can *reduce*
peak throughput when the batch is already compute-saturated — it shines at low-to-medium load.

---

## 2. Quantization — the highest-leverage lever

Quantization maps high-precision tensors (FP16/BF16) to low-bit representations. It is the first thing
to reach for because memory and bandwidth are usually what bind you.

### Axes that actually matter

- **What is quantized:** *weight-only* (W) vs *weights + activations* (WxAy, e.g. W8A8, W4A16). Weights
  are static (easy, quantize once); activations are dynamic, input-dependent, and have **outlier
  channels** that make low-bit activation quant hard. KV cache is a third, separate target.
- **PTQ vs QAT:** *Post-training quantization* needs only a small **calibration set** (~128–512
  representative samples) to set scales — minutes to hours, no gradients. *Quantization-aware training*
  simulates quantization in the forward pass and trains through it (straight-through estimator),
  recovering accuracy at the cost of a training run. **Default to PTQ; escalate to QAT only when PTQ
  recovery is insufficient** (typically sub-4-bit or W4A4).
- **Granularity:** per-tensor (one scale) < per-channel (per output channel) < per-group (e.g. groups
  of 64/128 along the input dim). Finer granularity recovers accuracy at small overhead; group-wise is
  standard for INT4 weights.
- **Symmetric vs asymmetric, static vs dynamic** (activation scales fixed from calibration vs computed
  at runtime). Dynamic is more accurate, slightly slower.

### Formats and methods (verify support per framework/hardware)

- **INT8 weight+activation (W8A8):** classic, broadly supported, ~2× memory cut, good quality with
  per-channel weights + per-token activations. **SmoothQuant** (arXiv 2211.10438) makes W8A8 robust by
  migrating activation outlier magnitude into the weights via a per-channel scale, so neither side has
  to absorb the full dynamic range.
- **INT4 weight-only (W4A16):** ~4× weight memory cut, big bandwidth-bound decode win. Two dominant PTQ
  methods:
  - **GPTQ** (arXiv 2210.17323) — second-order, layer-wise error compensation using approximate
    Hessian information; quantizes weights column-by-column and updates the rest to compensate.
  - **AWQ** (Activation-aware Weight Quantization, arXiv 2306.00978) — protects the ~1% of salient
    weight channels (identified by activation magnitude) by per-channel scaling before quantizing.
    Tends to be fast to apply and robust; very common for INT4 serving.
- **FP8 (E4M3 / E5M2):** floating-point 8-bit, native on Hopper/Blackwell-class GPUs. Wider dynamic
  range than INT8 → handles activation outliers better, often **near-lossless** for weights+activations,
  and accelerates *compute-bound* prefill (not just bandwidth). The default "first quantization" on
  FP8-capable hardware. E4M3 for weights/activations, E5M2 where more range is needed (verify per stack).
- **GGUF:** the llama.cpp container format (successor to GGML) with a family of k-quant schemes
  (Q4_K_M, Q5_K_M, Q6_K, etc.) for CPU/edge/Apple-silicon and mixed CPU-GPU offload. The standard for
  local/desktop inference; not what you'd use on a datacenter GPU fleet.
- **NF4 (4-bit NormalFloat):** information-theoretically motivated 4-bit weight type from **QLoRA**
  (arXiv 2305.14314), designed for normally-distributed weights; primarily a *fine-tuning* memory trick
  (frozen NF4 base + LoRA adapters) — see `[[fine-tuning-peft]]`.
- **Sub-4-bit (3-bit, 2-bit, ternary/binary):** research-active; quality cliffs are steep and hardware
  support thin. Treat as experimental and verify before relying on it.

### KV-cache quantization — often the bigger serving win

For long contexts and large batches, the **KV cache, not the weights, caps concurrency**. Quantizing it
to **FP8 or INT8** (sometimes INT4 with care) roughly halves/quarters KV memory → more concurrent
sequences → higher throughput, and less KV bandwidth per decode step → lower ITL. Keys are usually more
sensitive than values; per-channel/per-token schemes and keeping a small recent window in higher
precision help. This is a distinct decision from weight quant and frequently worth more.

### Accuracy recovery and where quant breaks

- **Always have a calibration set drawn from the real distribution** (including the chat template,
  languages, and domains you serve). Bad calibration data is the most common cause of a "quantization
  hurt quality" surprise.
- **Where it breaks:** activation outliers (motivates SmoothQuant/FP8); attention softmax and layernorm
  are precision-sensitive — keep them higher precision; the first and last layers and embeddings often
  warrant higher precision; very low bit-width (≤3) hits sharp cliffs; long-context and math/code
  reasoning degrade before perplexity does, so **don't trust perplexity alone**.
- **Recovery ladder:** finer granularity → keep sensitive layers in higher precision (mixed precision)
  → better method (AWQ/GPTQ over RTN) → SmoothQuant for W8A8 → QAT as the last resort.
- **Mandatory:** eval the quantized model on task metrics (not just perplexity) before shipping. See
  `[[ml-frameworks]]` for harnesses; lm-eval-harness-style task suites are the norm.

**Tooling:** NVIDIA TensorRT Model Optimizer (github.com/NVIDIA/TensorRT-Model-Optimizer) provides
PTQ/QAT, FP8/INT4/INT8, AWQ, SmoothQuant, sparsity, and speculative-decoding helpers, exporting to
TensorRT-LLM; `llm-compressor`/`compressed-tensors` (vLLM ecosystem) and `autoawq`/`auto-gptq` are
common open paths. Verify current method/format support per tool.

---

## 3. Pruning and sparsity

Remove parameters to shrink and speed the model. The catch: **only sparsity the hardware can exploit
speeds dense matmuls.**

- **Unstructured sparsity** (zero out individual weights by magnitude, e.g. via SparseGPT, arXiv
  2301.00774) gives high zero ratios but irregular patterns; a dense GEMM still touches the zeros, so it
  rarely speeds GPU inference and mainly helps compressed storage.
- **2:4 semi-structured ("fine-grained structured") sparsity** — exactly 2 of every 4 contiguous
  weights are zero. Ampere and later Tensor Cores have a **Sparse Tensor Core** path that skips the
  zeros for up to ~2× GEMM throughput. This is the sparsity that actually pays on NVIDIA GPUs. Apply,
  then fine-tune/distill to recover. Verify kernel/engine support (TensorRT/TensorRT-LLM, cuSPARSELt).
- **Structured pruning** — remove whole units: attention **heads**, MLP **width/channels**, or entire
  **layers** (depth pruning). Yields a smaller *dense* model that runs fast everywhere, but is the most
  destructive and **requires distillation/fine-tuning to recover** (the "prune then distill" recipe,
  e.g. as used to derive smaller models from larger ones in the Minitron line of work, arXiv 2407.14679).
- **Width vs depth:** depth (drop layers) tends to hit latency hardest (fewer sequential steps); width
  (narrow MLPs) tends to be gentler on quality per FLOP saved. Profile both.

Rule of thumb: pruning is rarely the *first* lever for LLMs (quant gives more for less risk); it earns
its place when you need a permanently smaller dense model and can afford a recovery run — at which
point it blends into distillation.

---

## 4. Knowledge distillation

Train a smaller/cheaper **student** to mimic a larger **teacher**. Unlike quant/pruning, the output is
a genuinely different, smaller model — the deepest memory and $/token win, at the highest data/compute
cost.

- **Logit / response distillation:** student matches teacher's softened output distribution
  (temperature-scaled), classic Hinton-style KL objective (arXiv 1503.02531). Cheap signal: just run the
  teacher over a corpus and train on its logits/top-k.
- **Feature / intermediate distillation:** also match hidden states/attention maps — more signal,
  more coupling to teacher architecture.
- **Sequence-level distillation:** train on teacher-*generated* sequences (the teacher labels the data
  by sampling completions), not just per-token logits — strong for generative tasks.
- **On-policy / generalized KD (GKD):** distill on sequences the *student* generates, scored by the
  teacher, to fix the train/inference distribution mismatch (exposure bias) of static teacher data
  (arXiv 2306.13649). On-policy distillation is the current best practice for instruction/reasoning
  students because the student learns to recover from its own trajectories.
- **Combine with pruning:** "prune-then-distill" (initialize the student from a pruned teacher) reaches
  a target size far cheaper than training a small model from scratch.

**Anti-pattern: distill-without-(enough)-data.** The student is only as good as the diversity and volume
of teacher-labeled data covering your real distribution. Thin or off-distribution data yields a student
that looks fine on benchmarks and fails in production. Budget the data generation, not just the training.

The training machinery (FSDP/DeepSpeed, RL-style on-policy loops, LoRA for the student) lives in
`[[fine-tuning-peft]]` and the training skills.

---

## 5. Speculative decoding and friends

Attack decode's **sequential bottleneck**: instead of one token per expensive target forward pass,
*propose* several cheap candidate tokens and *verify* them in a single target pass. With a correct
verification (rejection sampling) the output distribution is **provably identical** to plain decoding —
it is exact, not approximate (arXiv 2211.17192, "Fast Inference from Transformers via Speculative
Decoding"; also arXiv 2302.01318).

### The economics — acceptance rate is everything

Per target step you propose `k` draft tokens; the target verifies all `k+1` positions in one pass and
accepts a prefix. Expected accepted tokens per target pass ≈ a function of the per-token acceptance
rate `α`. **Speedup ≈ (accepted tokens per target pass) ÷ (1 + draft cost ratio).** Two ways to lose:

1. **Acceptance too low** (`α` small): the draft disagrees with the target often, so you waste target
   compute verifying rejected tokens. Driven by how well-aligned the draft is to the target.
2. **Draft too expensive:** if drafting `k` tokens costs nearly as much as the target step, even high
   acceptance doesn't pay. The draft must be much cheaper than the target.

So the design space is "maximize acceptance × minimize draft cost." Crucially, speculative decoding
spends **spare compute** to cut latency — it helps most when the GPU is *not* already compute-saturated
(low/medium batch, latency-sensitive). At max throughput batch sizes it can *hurt*. Tune it per
operating point.

### Variants (verify engine support; this list churns)

- **Draft-model (vanilla) speculative decoding:** a small separate model of the same family/tokenizer
  drafts; the big target verifies. Simple, but you must ship and align a second model.
- **Self-speculative:** the target model drafts using a subset of its own layers (layer skipping), no
  separate model (e.g. LayerSkip / draft-and-verify approaches).
- **Medusa** (arXiv 2401.10774): add a few lightweight **extra decoding heads** to the target that
  predict tokens at future positions in parallel; verify with a tree of candidates. No separate draft
  model; cheap to train the heads.
- **EAGLE** (arXiv 2401.15077, EAGLE-2 arXiv 2406.16858, and the EAGLE-3 line): autoregress at the
  **feature level** (predict the next hidden state) rather than the token level, with a tree draft.
  Among the highest acceptance / best speedups in the draft-head family; a default to evaluate.
- **Lookahead decoding** (arXiv 2402.02057): generate n-grams in parallel via a Jacobi-iteration scheme
  and verify them — no draft model, no extra training.
- **N-gram / prompt-lookup decoding:** draft by copying n-grams from the prompt/context (great when
  output echoes input — summarization, code edit, RAG with quotation). Essentially free; surprisingly
  effective on the right workloads.
- **Multi-token prediction (MTP)** heads trained at pretraining time double as a draft mechanism in
  some recent models.

Most serving engines (`[[serving-frameworks]]`) implement one or more of these; the *model-side* work
is producing/aligning the draft heads or draft model and measuring acceptance on your traffic.

**Anti-pattern: spec-decode with low acceptance.** Always measure realized acceptance rate and
end-to-end tokens/s on your traffic; a config that looks good on one dataset can be net-negative on
another. If acceptance is low, fix alignment (train the heads on your distribution) or turn it off.

---

## 6. Low-rank and structural efficiency

Architectural choices that change the FLOP/byte equation. Many are baked in at pretraining (you inherit
them), but they dictate what's possible at inference.

- **Attention KV sharing — MQA / GQA / MLA:**
  - **MQA** (Multi-Query Attention, arXiv 1911.02150): all query heads share *one* K/V head → smallest
    KV cache, some quality loss.
  - **GQA** (Grouped-Query Attention, arXiv 2305.13245): query heads share K/V in *groups* — the modern
    default, a tunable midpoint between MHA quality and MQA cache size.
  - **MLA** (Multi-head Latent Attention, popularized by the DeepSeek-V2 line, arXiv 2405.04434):
    compress K/V into a low-rank **latent** that's cached instead of full K/V → large KV-cache savings
    while preserving quality. Verify engine support before assuming you can serve it efficiently.
  KV-cache size directly caps concurrency, so these are first-order for throughput.
- **FlashAttention** (arXiv 2205.14135; FlashAttention-2 arXiv 2307.08691; FlashAttention-3 for Hopper,
  arXiv 2407.08608): IO-aware exact attention that tiles the computation in SRAM and never materializes
  the full N×N attention matrix in HBM — large speedup and memory reduction, *no* quality change. This
  is table stakes; ensure your stack uses it (or FlashInfer-style kernels for serving).
- **MoE (Mixture of Experts) as efficiency:** route each token to a few of many expert MLPs, so *active*
  FLOPs per token stay small while total parameters (capacity) grow. Cuts compute/$ per token but
  **raises total memory** (all experts must be resident or fetched) and adds routing/load-balancing and
  expert-parallel serving complexity. An efficiency win on the *compute/quality* axis, a cost on memory.
- **Low-rank factorization (LoRA-style):** decompose weight updates/weights into `A·B` of rank `r`.
  Inference-relevant uses: serve **many LoRA adapters** over one shared base (multi-tenant adapter
  serving) cheaply; merge an adapter into the base for zero-overhead inference. Training-side detail in
  `[[fine-tuning-peft]]`.

---

## 7. Compilation and kernels

Once the math is fixed, cut overhead and fuse work. These are (mostly) **quality-neutral** wins — take
them.

- **`torch.compile` / TorchInductor:** traces the model and generates fused (often Triton) kernels,
  removing Python/eager overhead and fusing pointwise/reduction ops. Watch for graph breaks
  (dynamic shapes, data-dependent control flow) — they fragment the graph and erode the gain. Use
  `dynamic=True`/`mark_dynamic` for variable sequence lengths; verify current modes/flags.
- **CUDA graphs:** capture the fixed sequence of kernel launches for a decode step once and replay it,
  eliminating per-step CPU launch overhead — a real ITL win at small batch where launch latency is a
  big fraction of step time. Serving engines often combine CUDA graphs with padding to a set of fixed
  batch sizes. (`torch.compile` mode `reduce-overhead` uses CUDA graphs under the hood.)
- **TensorRT / TensorRT-LLM:** ahead-of-time **engine build** that fuses layers, selects tactics,
  applies the chosen precision (FP8/INT4/INT8), and emits a hardware-specific optimized engine. Highest
  peak performance on NVIDIA, at the cost of an opaque build step that is **GPU-arch- and shape-specific**
  (rebuild per GPU generation / batch-seq profile). See `examples.md`. Verify the build API — it changes.
- **ONNX Runtime:** portable graph optimization + execution providers (CUDA/TensorRT/CPU/others); useful
  for cross-hardware portability and non-NVIDIA targets.
- **XLA (JAX / `torch_xla`):** whole-program compilation/fusion; the default path on **TPU** and a
  strong CPU/GPU option. See `[[ml-frameworks]]`.
- **Custom kernels — Triton / CUTLASS:** the last mile. Write fused dequant-GEMM, paged-attention,
  MoE-grouped-GEMM, and quantized kernels when no library op exists. Triton for fast portable kernels;
  CUTLASS for peak GEMM templates. This is where W4A16 dequant-fused matmuls and custom KV layouts live.
- **Paged / continuous batching:** the engine-boundary techniques (PagedAttention, in-flight batching,
  chunked prefill) belong to `[[serving-frameworks]]`, but they are *why* a well-quantized model
  actually translates into throughput — quantization frees memory that the engine turns into more
  concurrent sequences. Optimize the model with the serving engine's behavior in mind.

---

## 8. The efficiency taxonomy and how to combine

Organize techniques by the level they act on:

- **Data level:** prompt/context compression, KV reuse via prefix caching, retrieval to shorten context.
  (Mostly engine/application side — `[[serving-frameworks]]`, `[[rag-vector-databases]]`.)
- **Model level (this skill):** quantization, pruning/sparsity, distillation, efficient architecture
  (GQA/MLA/MoE/low-rank). Changes the artifact.
- **System level:** compilation, kernels, CUDA graphs, batching, parallelism, scheduling. Changes how
  the artifact runs.

(This data/model/system framing follows the efficient-LLM and inference surveys — arXiv 2312.03863 and
arXiv 2402.09748.) The levels are **multiplicative and largely orthogonal**, so a real production stack
combines them. A common high-performance recipe:

> **FP8 (or INT4) weights + FP8/INT8 KV cache + FlashAttention + `torch.compile`/engine build +
> CUDA graphs + speculative decoding (EAGLE/Medusa/n-gram)**, served by a paged/continuous-batching
> engine.

Combine deliberately, **eval after each addition**: interactions exist (e.g. aggressive weight quant +
KV quant + spec-decode can compound quality loss; a draft model must be quantized/aligned to the
quantized target). Add one lever, measure quality and the target metric, keep or revert.

---

## 9. Decision guide — pick the lever for your bound

Identify what binds you at your real operating point, then:

- **Latency-bound (single user / small batch, low TTFT/ITL SLO):** you are bandwidth-bound in decode.
  → weight-only INT4/FP8, KV-cache quant, FlashAttention, **CUDA graphs**, and **speculative decoding**
  (the biggest latency lever when GPU has spare compute). MLA/GQA if you control the architecture.
- **Throughput-bound ($/token at high load, batch saturated):** you are (or want to be) compute-bound.
  → FP8 weights+activations (accelerates compute), KV-cache quant (more concurrency = bigger batches),
  engine continuous batching, MoE for FLOP/quality. Spec-decode may *not* help here — measure.
- **Memory-bound (model won't fit / KV won't fit / too many accelerators):**
  → weight quant (INT4 first for footprint), KV-cache quant (long context), structured pruning +
  distillation for a permanently smaller model, MLA/GQA for KV. Offload only as a last resort (it
  reintroduces bandwidth limits).
- **Cost-bound (fixed quality target, minimize total $):** combine — usually weight + KV quant + a
  distilled/right-sized base model + an engine that maximizes goodput. The cheapest token is the one
  served by the smallest model that still passes your evals.

**First move, almost always:** profile to find the bound, then **FP8 (FP8-capable HW) or INT4
weight-only (older HW / tight memory) + KV-cache quant**, eval, and only then reach for pruning,
distillation, or spec-decode.

---

## 10. Anti-patterns and gotchas

- **Quantize-without-eval.** Perplexity barely moves while code/math/long-context/multilingual quality
  falls off a cliff. Always run task evals on your distribution, before and after. Non-negotiable.
- **Optimizing the wrong bound.** Faster kernels on a bandwidth-bound decode, or weight quant when the
  *KV cache* is what's overflowing. Profile first.
- **Distill-without-(enough)-data.** A student trained on thin/off-distribution teacher data benchmarks
  fine and fails in production.
- **Spec-decode with low acceptance** (or too-heavy a draft). Net-negative latency. Measure realized
  acceptance and end-to-end tokens/s on real traffic.
- **Calibration-set mismatch.** PTQ scales set from the wrong data (no chat template, wrong languages)
  silently degrade quality.
- **Trusting a leaderboard number that isn't your workload.** "2× faster" / "lossless" claims are
  workload-, batch-, context-, and hardware-specific. Reproduce on yours.
- **Unstructured sparsity expecting GPU speedup.** It compresses storage, not dense-GEMM time. Use 2:4.
- **Quantizing precision-sensitive ops** (softmax, layernorm, embeddings, first/last layers) to low bit.
  Keep them higher precision.
- **TensorRT engine reuse across GPU generations or shape profiles.** Engines are arch- and
  shape-specific; rebuild. A stale engine silently underperforms or fails.
- **Ignoring the tokenizer/template in the draft model.** Draft and target must share tokenizer and
  prompt formatting for speculative decoding to verify correctly.
- **Merging quality regressions across layers of optimization** without isolating which one caused it.
  Add and eval one lever at a time.

---

## 11. Rationalizations & rebuttals

The excuses that precede a regression in production. Each is a real thing engineers and agents say.

- *"INT4/FP8 is basically lossless, ship it without re-evaluating."* — Perplexity barely moves while
  code, math, long-context, and multilingual quality fall off a cliff. "Lossless" is workload-specific.
  Run task evals on **your** distribution before and after. Non-negotiable.
- *"Speculative decoding always helps, turn it on everywhere."* — It spends *spare* compute to cut
  latency; at throughput-saturated batch sizes it can *reduce* peak throughput, and at low acceptance
  it's net-negative latency. Measure realized acceptance and end-to-end tokens/s at your operating point.
- *"Let's optimize first, profile later."* — Optimizing the wrong bound is the #1 wasted-effort
  anti-pattern. Faster kernels do nothing for a bandwidth-bound decode; weight quant does nothing when
  the *KV cache* is what overflows. Identify the bound at your real batch/context/SLO first.
- *"Quantize the whole model uniformly — fewer special cases."* — Softmax, layernorm, embeddings, and
  the first/last layers are precision-sensitive; pushing them to low bit silently degrades quality. Use
  mixed precision and keep sensitive ops higher.
- *"Distillation just needs the training run set up."* — The student is only as good as the diversity
  and volume of teacher-labeled data covering your real distribution. Thin or off-distribution data
  yields a student that benchmarks fine and fails in production. Budget the data generation, not just the
  training.
- *"Calibration data doesn't matter much, any text will do."* — PTQ scales set from the wrong data (no
  chat template, wrong languages/domains) are the most common cause of a "quantization hurt quality"
  surprise. Draw the calibration set from the real serving distribution.
- *"The paper / leaderboard says 2× faster, so we'll get 2×."* — Speedup and "lossless" claims are
  workload-, batch-, context-, and hardware-specific. Reproduce on your traffic and hardware before
  committing it to a config.
- *"Unstructured pruning hit 60% sparsity, that's a big speedup."* — A dense GEMM still touches the
  zeros; unstructured sparsity compresses storage, not GPU matmul time. Use 2:4 on supported Tensor
  Cores if you want speed.

---

## 12. Red flags

Stop and reconsider if any of these are true:

- **No task eval (only perplexity, or nothing) after quantization.** You have no evidence the artifact
  is shippable; perplexity hides code/math/long-context/multilingual cliffs.
- **An optimization was applied without first identifying the bound.** Tuning kernels on a
  bandwidth-bound decode, or weight-quantizing when the KV cache caps concurrency.
- **Speculative decoding is on but realized acceptance rate is unmeasured or low**, or the draft is
  nearly as expensive as the target — likely net-negative latency.
- **The draft and target don't share tokenizer / prompt template** (or the draft wasn't aligned to the
  *quantized* target) — verification breaks or acceptance collapses.
- **The memory-vs-compute bound is being ignored:** the same model is treated identically for a
  latency-sensitive single user (bandwidth-bound) and a throughput-maximizing batch (compute-bound).
- **Calibration set doesn't match production** (missing chat template, wrong languages/domains).
- **Multiple optimizations stacked at once with a single end-to-end eval** — you can't attribute a
  quality regression to a lever. Add and eval one at a time.
- **A TensorRT engine is reused across a different GPU generation or shape profile** — engines are
  arch- and shape-specific; a stale engine silently underperforms or fails.
- **Precision-sensitive ops (softmax, layernorm, embeddings, first/last layers) quantized to low bit**,
  or sub-4-bit weights relied on without escalating the recovery ladder.

---

## 13. Verification gate (definition of done)

An optimization is not "done" until all of these hold. Show the evidence, don't assert it.

- [ ] **Bound identified at the real operating point.** Profiled (framework profiler / Nsight) at the
      actual batch, context length, and SLO — not guessed. You can state: bandwidth-, compute-, memory-,
      or cost-bound.
- [ ] **Technique matched to the bound.** The lever attacks the bound you found (e.g. weight/KV quant
      for bandwidth-bound decode; FP8 W+A or MoE for compute-bound throughput; quant/prune+distill for
      memory-bound footprint; spec-decode only where there's spare compute).
- [ ] **Quality eval before and after, on your distribution.** Task metrics (lm-eval-harness-style
      suites), not perplexity alone, including the regression-prone slices (code, math, long-context,
      multilingual). Calibration set drawn from real traffic.
- [ ] **Latency, throughput, and $/token measured** at the operating point: TTFT, ITL/TPOT, tokens/s,
      and cost per token. For speculative decoding, also realized **acceptance rate** on real traffic.
- [ ] **Accuracy delta is acceptable** against an agreed quality bar — the cheapest token is the one
      served by the smallest model that still passes your evals.
- [ ] **One lever at a time.** Each addition was added and evaluated in isolation; any combined-stack
      regression can be attributed to a specific lever.
- [ ] **Artifact reproducibility recorded:** exact format/dtype, method (AWQ/GPTQ/SmoothQuant/FP8),
      granularity, sensitive-layer overrides, draft/target pairing, and (if built) the TensorRT engine's
      GPU arch and shape profile — so the result rebuilds and isn't reused out of scope.

---

## 14. Version awareness

It is 2026 and this is one of the fastest-moving areas in ML systems. Specifically verify against
current docs before relying on:

- Which **dtypes/formats** your GPU generation and serving engine support (FP8 variants, INT4 schemes,
  microscaling/MX formats, FP4-class types on newest hardware — confirm reality, don't assume).
- The **API surface** of TensorRT Model Optimizer, TensorRT-LLM, `llm-compressor`, `autoawq`,
  `torch.compile` modes — these change across releases.
- Which **speculative-decoding** variants your engine implements and how to configure them.
- Current **method names and SOTA** (the EAGLE/Medusa/spec-decode and quantization literature publishes
  new methods constantly). Treat named methods here as the durable concepts; check for newer variants.

Do not fabricate flags, format names, dtype support, or benchmark numbers. When unsure, say "verify
against current docs" and link the source.

---

## 15. Canonical references

- LLM inference / efficiency surveys: **arXiv 2402.09748** (LLM inference survey) · **arXiv 2312.03863**
  (efficient LLMs survey) · arXiv 2312.15234 (efficient LLM inference).
- Quantization: GPTQ **2210.17323** · AWQ **2306.00978** · SmoothQuant **2211.10438** · QLoRA/NF4
  **2305.14314** · LLM.int8() **2208.07339**.
- Sparsity/pruning: SparseGPT **2301.00774** · Wanda **2306.11695** · Minitron prune+distill **2407.14679**.
- Distillation: Hinton KD **1503.02531** · sequence-level KD **1606.07947** · GKD/on-policy **2306.13649**.
- Speculative decoding: **2211.17192** · **2302.01318** · Medusa **2401.10774** · EAGLE **2401.15077**,
  EAGLE-2 **2406.16858** · Lookahead **2402.02057**.
- Attention/architecture: FlashAttention **2205.14135** / FA-2 **2307.08691** / FA-3 **2407.08608** ·
  MQA **1911.02150** · GQA **2305.13245** · MLA (DeepSeek-V2) **2405.04434**.
- Tooling: **NVIDIA TensorRT Model Optimizer** — github.com/NVIDIA/TensorRT-Model-Optimizer ·
  TensorRT-LLM — github.com/NVIDIA/TensorRT-LLM · `llm-compressor`/`compressed-tensors` (vLLM) ·
  `autoawq`, `auto-gptq`, llama.cpp/GGUF. Verify URLs and current APIs.

Cross-links: `[[serving-frameworks]]` (engines) · `[[ml-frameworks]]` (PyTorch/JAX/XLA, GPU/TPU) ·
`[[fine-tuning-peft]]` (LoRA/QLoRA, QAT/distillation training) · `[[gke-inference-gateway]]` (routing) ·
`[[aiml-on-kubernetes]]` (umbrella) · `[[gke-master]]` (accelerator node pools).

---

# Inference Optimization — Worked Examples

Canonical, correct-in-shape flows to imitate. **APIs and flags in this area change quarterly (2026)** —
every command below is illustrative of the *workflow*; **verify exact argument names, dtypes, and
hardware support against current docs** (TensorRT Model Optimizer, TensorRT-LLM, your AWQ/GPTQ library,
your serving engine) before running. Do not treat any flag here as guaranteed-current.

---

## 1. INT4 / AWQ weight-only quantization flow

Goal: take an FP16 base model to **INT4 weight-only (W4A16)** for a bandwidth-bound decode win, with a
real eval gate. Shape of the flow with `autoawq` (one common open path; `llm-compressor` and TensorRT
Model Optimizer follow the same *shape*).

```python
# pip install autoawq  (verify package name/version + GPU support before running)
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "org/base-model-fp16"
quant_path = "org/base-model-awq-int4"

# Quant config: 4-bit, group size 128, GEMM kernel. zero_point/version names change — verify.
quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path)

# CALIBRATION DATA IS THE WHOLE GAME: draw ~128-512 samples from YOUR real distribution,
# formatted with the SAME chat template / languages / domains you serve. A generic web
# corpus here is the usual cause of a "quantization hurt quality" surprise.
calib = load_my_representative_samples()   # list[str], already chat-templated

model.quantize(tok, quant_config=quant_config, calib_data=calib)
model.save_quantized(quant_path)
tok.save_pretrained(quant_path)
```

**Mandatory eval gate** — perplexity is NOT sufficient; INT4 can leave perplexity flat while code/math/
long-context/multilingual quality drops:

```bash
# Run a task suite (lm-eval-harness-style) on BOTH fp16 and the int4 export, compare per task.
# Verify current harness name/flags.
lm_eval --model hf --model_args pretrained=org/base-model-fp16 \
        --tasks gsm8k,arc_challenge,<your-domain-task> --batch_size auto
lm_eval --model hf --model_args pretrained=org/base-model-awq-int4 \
        --tasks gsm8k,arc_challenge,<your-domain-task> --batch_size auto
```

Ship only if the quality delta on *your* tasks is within tolerance. If not: recovery ladder →
finer group size → keep sensitive layers (first/last, embeddings) higher precision → try GPTQ →
escalate to QAT. The AWQ checkpoint is then loadable by a serving engine that supports AWQ
(`[[serving-frameworks]]`); pass the engine's quantization flag (verify name, e.g. `quantization=awq`).

---

## 2. TensorRT-LLM: build-then-serve note

TensorRT-LLM is an **ahead-of-time engine build** — it fuses layers, applies precision, and emits a
**GPU-arch- and shape-specific** engine. The two-phase shape (quantize/convert → build engine → serve):

```bash
# Phase A — produce a quantized checkpoint (e.g. FP8 or INT4-AWQ) with TensorRT Model Optimizer
#   (github.com/NVIDIA/TensorRT-Model-Optimizer). The exact CLI/script names and the FP8/INT4
#   support matrix change across releases — VERIFY against the repo's current examples.
#   Conceptually: load base model -> run PTQ (FP8 or AWQ-INT4, with YOUR calibration set) ->
#   export a TensorRT-LLM checkpoint directory.

# Phase B — build the engine for the TARGET GPU and a chosen shape profile (max batch, max input/
#   output len). The engine is specific to this GPU generation + these shapes.
trtllm-build --checkpoint_dir ./ckpt_fp8 \
             --output_dir ./engine_fp8 \
             --gemm_plugin auto \
             --max_batch_size 64 \
             --max_input_len 8192 \
             --max_seq_len 9216
#   (flag names/values illustrative — verify with `trtllm-build --help` for your version.)

# Phase C — serve the engine (Triton with the TensorRT-LLM backend, or trtllm-serve, or via an
#   engine that consumes TRT-LLM engines). Routing/batching is the serving layer's job.
```

Key correctness points:

- **Rebuild per GPU generation and per shape profile.** An engine built for one arch (or a different
  `max_batch_size`/`max_seq_len`) will underperform or fail to load elsewhere. Do not reuse blindly.
- Choose precision at build time to match the bound: **FP8** (weights+activations) on FP8-capable GPUs
  for compute *and* bandwidth; **INT4-AWQ** weight-only when memory/bandwidth dominate.
- Eval the *built engine's* outputs, not just the pre-build checkpoint — the build (fusion, precision,
  plugins) is part of what can move quality.
- The engine pairs with the serving engine's paged-KV / in-flight batching — see `[[serving-frameworks]]`.

---

## 3. Speculative decoding config note

Speculative decoding is **exact** (rejection-sampling verification reproduces the target's
distribution) and trades **spare compute for latency**. It pays only when **acceptance is high** and the
**draft is cheap** relative to the target.

Illustrative serving config (shape only — the actual keys differ per engine and change; **verify**):

```yaml
# Conceptual — map to your engine's real speculative-decoding options.
target_model: org/base-model-fp8        # the big, accurate model
speculative:
  method: eagle                          # or: draft_model | medusa | ngram | lookahead
  # draft_model: org/base-model-draft    # for vanilla draft-model spec: MUST share the target's
  #                                       #   tokenizer; quantize/align it to the (quantized) target.
  num_speculative_tokens: 5              # k candidates proposed per target verification step
```

Operate it correctly:

- **Pick the variant for the workload.** `ngram`/prompt-lookup is near-free and excellent when output
  echoes input (summarization, code edit, RAG quotation). `eagle`/`medusa`/self-speculative need no
  separate aligned draft model and tend to give high acceptance. A separate `draft_model` must share
  the tokenizer and be aligned (ideally quantized to match the target).
- **Measure realized acceptance rate AND end-to-end tokens/s on real traffic**, not a demo prompt. A
  config that wins on one dataset can be net-negative on another.
- **It helps most with spare compute** (low/medium batch, latency-sensitive). At max-throughput batch
  sizes the GPU is already compute-saturated and spec-decode can *reduce* throughput — benchmark at
  your real load and turn it off where it loses.
- Tune `num_speculative_tokens` (`k`): too high wastes verification compute on rejected tails; sweep it.

---

## Stacking note

These compose. A typical high-performance stack: **FP8 (or INT4) weights + FP8/INT8 KV cache +
FlashAttention + engine build / `torch.compile` + CUDA graphs + speculative decoding**, served by a
paged/continuous-batching engine (`[[serving-frameworks]]`). **Add one lever at a time and re-run the
eval gate** — interactions (aggressive weight quant + KV quant + spec-decode) can compound quality loss
that no single lever caused.
