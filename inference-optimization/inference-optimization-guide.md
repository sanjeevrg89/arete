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

## 11. Version awareness

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

## 12. Canonical references

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
