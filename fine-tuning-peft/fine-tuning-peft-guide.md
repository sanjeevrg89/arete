# Fine-Tuning & PEFT — Practitioner Guide

The reference for adapting pretrained LLMs to tasks/domains cost-effectively. This is opinionated and
ships-models oriented. The ecosystem moves fast (it is 2026): library APIs, default hyperparameters, and
"what fits on one GPU" numbers change — **verify specifics against current docs** (links at the end).

---

## 1. Mental model: when to fine-tune at all

Fine-tuning is one of four levers for changing model behavior. Reach for the cheapest one that works:

| Lever | Changes | Cost / maintenance | Best for |
|---|---|---|---|
| **Prompt + few-shot** | Behavior at inference time | Near-zero; iterate in minutes | Format nudges, simple tasks, prototyping |
| **RAG** | The *knowledge* the model sees | Index/retrieval infra, ongoing | Up-to-date or proprietary facts, citations |
| **Fine-tuning (SFT/PEFT)** | The model's *weights* → behavior/style/skill | Data + training + eval + a model artifact to maintain | Consistent output shape, tone, narrow task, tool protocols, latency (smaller model), distillation |
| **Full pretraining** | The model from scratch | Enormous; rarely justified | New tokenizer/architecture/language at scale → [[training-frameworks]] |

**Rules of thumb:**

- Fine-tuning teaches **behavior and skills**, not **facts**. If the problem is "the model doesn't know X,"
  that's usually RAG (or a tool), not fine-tuning. Trying to inject a knowledge base via SFT is expensive,
  goes stale, and tends to produce confident hallucinations.
- Fine-tune when you need: a reliably-structured output (JSON/DSL/tool calls), a specific tone/persona,
  strong performance on one narrow task, lower latency/cost by moving work to a smaller model, or
  **distillation** of a large teacher's outputs into a small student.
- Fine-tuning has a **maintenance tax**: every base-model upgrade means re-running the pipeline, and you own
  the data/eval forever. Prompting and RAG have none of that. Factor this in before committing.
- The honest default order: prompt → few-shot → RAG → PEFT SFT → (if needed) preference alignment → (rarely)
  full fine-tuning. Most teams stop well before the end.

### Data requirements

Quality and *distribution match* dominate quantity.

- **Quality > quantity.** A few thousand clean, correctly-formatted, diverse examples routinely beat tens of
  thousands of noisy ones. Style/format tasks can work with hundreds–low-thousands; broad instruction tuning
  wants more diversity. There is no magic number — measure on a holdout.
- **Match the inference distribution.** Train on inputs that look like production inputs, with outputs in the
  exact shape you want at inference.
- **Hygiene is non-negotiable:** dedupe (exact + near-dup), remove truncated/garbled samples, balance classes
  / task types, and **decontaminate** — remove any training example that overlaps your eval/holdout or known
  public benchmarks. Eval leakage is the most common way teams fool themselves.
- For instruction data, diversity of *instructions* matters more than raw count. For distillation, filter the
  teacher outputs (rejection sampling / quality scoring) rather than training on everything.

---

## 2. The fine-tuning spectrum: full FT vs PEFT

### Full fine-tuning

Update every weight. Maximum capacity, but:

- **Memory:** you hold the model + gradients + optimizer state. With Adam in mixed precision the rough rule
  is roughly ~16 bytes/param for weights+grads+optimizer moments (fp16/bf16 weights + fp32 master +
  two fp32 Adam moments), before activations. A 7B model is well beyond a single 24GB GPU for full FT; you
  need FSDP/ZeRO sharding across GPUs ([[training-frameworks]]).
- **Catastrophic forgetting:** narrow data can overwrite general capability. Mitigate with lower LR, fewer
  epochs, and mixing in general/instruction data.
- **Artifacts:** a full-size checkpoint per task. Expensive to store and serve at any task count.
- **When it's worth it:** large/high-quality datasets, when PEFT demonstrably underperforms, or deep domain
  shift. Use FSDP/DeepSpeed ZeRO; that's [[training-frameworks]] territory.

### PEFT (parameter-efficient fine-tuning)

Freeze the pretrained weights; train a tiny set of new/added parameters (typically **<1%**). Benefits:
small memory footprint, fast, tiny adapter artifacts (MBs), and—because the base is frozen—**much less
catastrophic forgetting**. On most task-adaptation work, well-tuned LoRA/QLoRA **matches full fine-tuning
quality**. PEFT is the correct default. See the survey (arXiv 2403.14608) for the taxonomy.

---

## 3. PEFT methods

### LoRA (Low-Rank Adaptation) — arXiv 2106.09685

For a frozen weight `W ∈ R^{d×k}`, learn a low-rank update `ΔW = B·A` where `A ∈ R^{r×k}`, `B ∈ R^{d×r}`,
`r ≪ min(d,k)`. The forward pass becomes `h = Wx + (alpha/r)·B·A·x`; `B` is zero-initialized so training
starts at the base model. Only `A` and `B` train.

Key hyperparameters:

- **`r` (rank):** adapter capacity. Common range 8–64; higher `r` = more capacity and more params. Start
  small (8–16) and raise only if the task underfits.
- **`lora_alpha`:** scaling; the effective scale applied is `alpha/r`. A common convention is `alpha = r`
  or `alpha = 2r`. If you change `r`, keep the `alpha/r` ratio in mind rather than blindly copying `alpha`.
- **`target_modules`:** *which* linear layers get adapters. This is the single highest-leverage and most
  commonly-botched setting. The original paper adapted attention projections, but **adapting the MLP/feed-
  forward projections too generally helps**. For Llama-style models that means
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Many libraries support
  `target_modules="all-linear"` to cover every linear layer automatically — a safe strong default. Adapting
  only `q_proj,v_proj` is a frequent quiet underperformer.
- **`lora_dropout`:** regularization on the adapter path (e.g., 0.0–0.1).
- **`modules_to_save`:** fully train specific modules (e.g., a resized embedding / new LM head) when you add
  tokens — those can't be expressed as a low-rank delta.

LoRA adds **no inference latency once merged** (`W' = W + (alpha/r)·B·A`).

### QLoRA — arXiv 2305.14314

LoRA on top of a **4-bit quantized frozen base**. Three pieces:

1. **NF4 (4-bit NormalFloat):** an information-theoretically-motivated 4-bit datatype for normally-
   distributed weights. The base model is stored in NF4 and dequantized on the fly during the forward/
   backward pass; gradients flow only into the bf16 LoRA adapters.
2. **Double quantization:** quantize the quantization constants too, shaving additional memory.
3. **Paged optimizers:** use NVIDIA unified memory to page optimizer state to CPU on memory spikes, avoiding
   OOM during long sequences.

QLoRA is what lets you fine-tune a 7–13B model on a single 24GB consumer GPU and (per the paper) a ~65–70B
model on a single 48–80GB GPU, at quality close to 16-bit LoRA. **Verify current single-GPU fits against
docs** — they depend on seq length, batch size, and library version. In `transformers`/`peft` this is
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_use_double_quant=True)` plus `prepare_model_for_kbit_training`.

### DoRA (Weight-Decomposed LoRA) — arXiv 2402.09353

Decomposes the weight into **magnitude** and **direction**, applying LoRA to the direction and training the
magnitude separately. Often closes the gap to full FT better than plain LoRA at low ranks, for modest extra
cost. In PEFT it's a flag on `LoraConfig` (`use_dora=True`). Try it when LoRA underfits at a given rank.

### Other PEFT families (know when each applies)

- **(IA)³** — arXiv 2205.05638: learns elementwise rescaling vectors for key/value/FFN activations. Even
  fewer params than LoRA; strong in few-shot regimes.
- **Prefix / Prompt tuning / P-tuning (v2)** — prepend trainable continuous "soft prompt" vectors to the
  input or to each layer's keys/values; the model weights stay frozen. Extremely few params; generally
  weaker and trickier to tune than LoRA for general SFT, but cheap to store per task.
- **Bottleneck adapters** (Houlsby/Pfeiffer) — insert small trainable MLP modules between transformer
  sublayers. The original adapter family; LoRA largely superseded them for LLMs but they remain relevant in
  the adapter-fusion / multi-task literature.

**Default:** LoRA or QLoRA for almost everything. Reach for DoRA when LoRA underfits, (IA)³ for ultra-low-
param/few-shot, soft prompts only for niche multi-task adapter-swapping setups.

### Adapter lifecycle: merge vs keep separate

- **Merge** (`merge_and_unload` in PEFT): fold `ΔW` back into `W` → a standalone model, zero adapter
  overhead, simplest to serve. Do this for a single dedicated fine-tune. Note merging a LoRA *into a 4-bit
  base* needs care — typically dequantize/merge into a 16-bit base, then optionally re-quantize for serving.
- **Keep separate**: ship the small adapter (MBs) and serve **one base + many LoRAs**, swapping/batching
  adapters per request. vLLM, SGLang, and the GKE Inference Gateway support multi-LoRA serving so you host
  one set of base weights and many cheap adapters. See [[serving-frameworks]] / [[gke-inference-gateway]].

---

## 4. The SFT pipeline (do these right or quality silently drops)

Supervised fine-tuning / instruction tuning: train on `(prompt, desired_completion)` pairs.

1. **Choose the base.** Instruction-tune from a **base** model; further-tune from an **instruct** model only
   when you want to keep its chat/safety behavior. Smaller is fine if it clears your eval. Respect the
   license and acceptable-use terms.
2. **Format with the model's chat template.** Use `tokenizer.apply_chat_template(messages, tokenize=False,
   add_generation_prompt=...)`. **Never hand-roll role markers** — a template mismatch (wrong special
   tokens, missing BOS/EOS) silently destroys instruction-following and is brutal to debug.
3. **Mask the loss to completions only.** The model should learn to *produce the answer*, not to reproduce
   the instruction. TRL's `SFTTrainer` supports completion-only masking (e.g., a
   `DataCollatorForCompletionOnlyLM` / `completion_only_loss` / `assistant_loss_only`-style option depending
   on version) — set the prompt tokens' labels to `-100`. Verify the exact knob in your TRL version.
4. **Packing for throughput.** Concatenate multiple short examples into one max-length sequence to avoid
   wasting compute on padding. **Caveat:** ensure attention doesn't cross sample boundaries and that
   completion-only masking is preserved per-sample — naive packing can leak context across examples or
   break masking. Modern TRL handles this; confirm before trusting it.
5. **Hyperparameters (sane starting points, then tune):** LR ~1e-4–3e-4 for LoRA (higher than full FT, which
   is ~1e-5–2e-5); cosine schedule with warmup; 1–3 epochs (more overfits fast on small data); effective
   batch size via gradient accumulation; bf16; gradient checkpointing to trade compute for memory.
6. **Then (optionally) preference alignment.** Once SFT is good, DPO/ORPO/KTO (offline) or PPO/GRPO (online)
   can sharpen helpfulness/format/safety. ORPO can even fold preference signal into one stage. **All of this
   is [[rl-rlhf-frameworks]]** — hand off there; don't reinvent it here.

---

## 5. The training stack

| Tool | What it is | Use it when |
|---|---|---|
| **HF `peft` + `transformers` + `trl`** | The reference Python stack: `LoraConfig`/`get_peft_model`, `SFTTrainer`/`DPOTrainer` | You want full programmatic control / custom loops; the lingua franca |
| **bitsandbytes** | 4-bit/8-bit quantization + paged optimizers (the engine behind QLoRA) | Any k-bit training; pass via `BitsAndBytesConfig` |
| **Axolotl** | YAML-config wrapper over the HF stack | Reproducible config-driven runs; LoRA/QLoRA/full, packing, DeepSpeed/FSDP integration |
| **Unsloth** | Hand-optimized Triton kernels for LoRA/QLoRA | Single-GPU speed/memory wins; faster training, longer context on the same card |
| **Llama-Factory** | Broad LoRA/QLoRA/full + many methods, CLI/Web UI | Fast experimentation across many models/methods |
| **torchtune** | Native PyTorch recipes (LoRA/QLoRA/full, distributed) | A clean PyTorch-idiomatic path without the HF-trainer layer |
| **NVIDIA NeMo** | Large-scale framework incl. PEFT | Already in the NeMo/Megatron ecosystem at scale → also [[training-frameworks]] |

There is no single "best" — pick by control vs convenience and by what your platform already runs. Library
defaults and feature flags shift between releases; **pin versions and read that version's docs.**

### Memory math: what fits on one GPU

Order-of-magnitude reasoning (per-GPU VRAM ≈ weights + grads + optimizer + activations):

- **Base weights:** fp16/bf16 ≈ 2 bytes/param (≈ 14GB for 7B); **NF4 4-bit ≈ 0.5 byte/param** (≈ 3.5GB for
  7B). This 4× cut on the *frozen* base is QLoRA's core win.
- **LoRA adapters:** tiny — `r·(d+k)` per target matrix, totaling well under 1% of params; their grads and
  optimizer state are correspondingly tiny.
- **Full FT (Adam):** ≈ 16 bytes/param (bf16 weight + fp32 master + 2 fp32 moments) before activations → a
  7B is ~112GB+ → needs multi-GPU sharding (FSDP/ZeRO).
- **Activations:** scale with batch size × sequence length; **gradient checkpointing** trades recompute to
  slash this and is usually on for fine-tuning.

Net: **QLoRA puts 7–13B on a single 24GB card and ~70B on a single 80GB card** (paper-scale, seq/batch
dependent — verify current). Full FT of the same models needs a cluster. Knobs to fit a run: lower batch
size (raise grad-accum to keep effective batch), shorter `max_seq_len`, gradient checkpointing, 4-bit base,
paged optimizer, and Unsloth kernels.

---

## 6. Evaluation (the part teams skip and regret)

A fine-tune without an eval is a vibe. Build the harness *before* you train. See [[ml-evaluation-evals]].

- **Held-out task metric.** Split a true holdout *before* any training and never touch it for tuning. Report
  the task's real metric (exact-match/F1/pass@k/structured-validity/etc.).
- **LLM-as-judge / human spot-check** for open-ended quality, with a fixed rubric. Sanity-check the judge.
- **Base-model regression suite.** Run the *same* general-capability evals on base and fine-tuned model to
  catch **catastrophic forgetting** — a task win that tanks general ability is often a net loss.
- **Format/constraint checks.** If you fine-tuned for JSON/DSL/tool calls, measure schema-valid rate
  directly.
- **No leakage, ever.** Decontaminate train against eval/holdout and against public benchmarks you report.
  Track contamination as a first-class risk.
- **Track the base/template/data version** alongside metrics so results are reproducible across base-model
  upgrades.

---

## 7. Anti-patterns & gotchas

- **Fine-tuning when prompting/RAG would do.** The most expensive mistake. Exhaust cheaper levers first;
  fine-tuning for *facts* is almost always the wrong tool.
- **Tiny or dirty datasets.** A few hundred noisy/duplicated/mislabeled examples → an overfit, brittle model.
  Clean and dedupe before you scale up.
- **No eval / no holdout.** Training loss going down tells you nothing about generalization. Build the
  holdout and regression eval first.
- **Leaking eval into train.** Inflated metrics, production disappointment. Decontaminate.
- **Wrong `target_modules`.** Adapting only `q_proj,v_proj` (or missing the MLP projections) caps quality.
  Prefer covering attention + MLP, or `all-linear`.
- **Chat-template mismatch.** Hand-rolled role markers / wrong special tokens silently break instruction-
  following. Always use `apply_chat_template` and inspect a few tokenized samples.
- **Training loss on the prompt.** Forgetting completion-only masking teaches the model to echo instructions.
- **Catastrophic forgetting unmeasured.** Narrow data degrades general skills; PEFT + a data mix help, but
  you must run the base regression eval to know.
- **Too many epochs / LR too high.** Overfits small datasets fast; outputs get repetitive/memorized. Fewer
  epochs, early-stop on the holdout.
- **Adding tokens without `modules_to_save`.** Resized embeddings/LM head can't be a low-rank delta; train
  them fully or the new tokens never learn.
- **Merging a LoRA into a 4-bit base carelessly.** Merge into a 16-bit base, then re-quantize for serving;
  in-place 4-bit merges can degrade quality. Verify your library's supported path.
- **Picking the wrong base.** Tuning *from an instruct model* and clobbering its chat behavior, or expecting
  a tiny base to learn a task it lacks the capacity for.

---

## 8. Troubleshooting (symptom → likely cause → fix)

- **Loss is flat / model won't learn.** LR too low, wrong/empty `target_modules`, or labels all masked →
  raise LR, broaden target modules (`all-linear`), verify some labels ≠ -100 in a batch.
- **Loss → 0, garbage or repetitive generations.** Overfitting / too many epochs / template mismatch →
  fewer epochs, more/cleaner data, confirm chat template, check EOS is learned.
- **Great train metric, poor production behavior.** Eval leakage or train/inference distribution mismatch →
  decontaminate, make training inputs look like production.
- **Model forgot general abilities.** Catastrophic forgetting → mix in general data, lower LR/epochs, prefer
  PEFT over full FT, confirm via the base regression eval.
- **OOM during training.** Lower batch size (raise grad-accum), shorten `max_seq_len`, enable gradient
  checkpointing, switch to 4-bit base + paged optimizer, or use Unsloth kernels.
- **Merged model behaves differently from the adapter.** Wrong merge precision (4-bit in-place) or template
  drift between training and serving → merge into 16-bit, then re-quantize; pin the template.
- **Multi-LoRA serving picks the wrong/blended output.** Adapter not loaded/routed correctly, or rank/target
  mismatch with the served base → verify adapter ID routing and that the base matches training.

---

## 9. Version awareness

It is 2026 and this space churns. Treat as **subject to change, verify current docs**: exact single-GPU
memory fits; TRL's completion-only-loss / packing API names; `peft` config flags (DoRA, target-module
auto-detection); default hyperparameters; quantization formats beyond NF4; which serving stacks support
multi-LoRA and how. Pin library versions, read the release notes, and confirm flags before relying on them.
Don't trust hyperparameters copied from an old blog post.

---

## 10. Canonical references (real URLs; verify for updates)

- **PEFT survey (2024):** "Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey" —
  https://arxiv.org/abs/2403.14608
- **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" —
  https://arxiv.org/abs/2106.09685
- **QLoRA:** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" —
  https://arxiv.org/abs/2305.14314
- **DoRA:** Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" — https://arxiv.org/abs/2402.09353
- **(IA)³ / T-Few:** Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning..." —
  https://arxiv.org/abs/2205.05638
- **Hugging Face PEFT docs:** https://huggingface.co/docs/peft
- **TRL (SFTTrainer / alignment trainers):** https://huggingface.co/docs/trl
- **bitsandbytes:** https://huggingface.co/docs/bitsandbytes
- **Axolotl:** https://github.com/axolotl-ai-cloud/axolotl ·
  **Unsloth:** https://github.com/unslothai/unsloth ·
  **Llama-Factory:** https://github.com/hiyouga/LLaMA-Factory ·
  **torchtune:** https://github.com/pytorch/torchtune ·
  **NVIDIA NeMo:** https://github.com/NVIDIA/NeMo
