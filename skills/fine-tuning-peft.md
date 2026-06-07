---
name: fine-tuning-peft
description: Adapting pretrained LLMs to tasks/domains cost-effectively via supervised fine-tuning (SFT) and
  parameter-efficient fine-tuning (PEFT). Use when deciding fine-tune vs prompt/RAG, or when running LoRA,
  QLoRA, DoRA, (IA)³, prefix/prompt/P-tuning; building SFT/instruction-tuning data; setting rank/alpha/target
  modules; doing 4-bit NF4 + paged-optimizer training; chat templates, packing, completion-only loss masking;
  merging adapters and multi-LoRA serving; and the memory math for what fits on one GPU. Covers the HF
  PEFT + Transformers/TRL stack, Axolotl, Unsloth, Llama-Factory, torchtune, NeMo, and bitsandbytes. For
  distributed/full pretraining see [[training-frameworks]]; for preference/RL post-training (DPO/PPO/GRPO)
  see [[rl-rlhf-frameworks]]; for serving adapters see [[serving-frameworks]]/[[gke-inference-gateway]].
---

# Fine-Tuning & PEFT

Apply the judgment of someone who ships fine-tuned LLMs cost-effectively in production: fine-tune only when
prompting/RAG can't get there, spend the effort on **data quality over quantity**, default to **PEFT
(LoRA/QLoRA)** over full fine-tuning, and never declare done without a holdout eval and a regression check
against the base model.

## How to use this skill

1. **Read `fine-tuning-peft-guide.md`** in this directory — the full reference (decision framework, PEFT
   methods, the SFT pipeline, the training stack, memory math, evaluation, troubleshooting). Apply it.
2. For a concrete LoRA/QLoRA SFT config to imitate (TRL `SFTTrainer` + HF PEFT, and Axolotl-style YAML),
   plus the memory-budget and adapter-merge/serve notes, read **`examples.md`**.
3. Match the surrounding codebase/training stack; apply the data-hygiene, masking, and eval rules regardless.

## The essentials (full detail in `fine-tuning-peft-guide.md`)

- **Don't fine-tune by reflex.** Prompting + few-shot + RAG covers most "wrong knowledge / wrong facts"
  problems and is cheaper to maintain. Fine-tune for *behavior/format/style/latency*: a consistent output
  shape, a tone, a tool-calling protocol, a narrow task, or distilling a big model into a small one.
- **Data quality > quantity.** A few thousand clean, diverse, correctly-formatted examples beat 100k noisy
  ones. Dedupe, decontaminate against your eval set, and balance. Garbage in → confidently-wrong out.
- **Default to PEFT.** LoRA/QLoRA update <1% of params and typically match full-FT quality on task adaptation
  at a fraction of the memory/compute, with no separate full-model checkpoint to store per task.
- **QLoRA = 4-bit NF4 frozen base + LoRA adapters in bf16 + paged optimizer.** It's what lets you fine-tune
  a 7–13B model on a single 24GB GPU and a ~70B on a single 80GB GPU. Verify exact fits against current docs.
- **LoRA knobs that matter:** `target_modules` (cover the attention *and* MLP projections, not just q/v),
  `r` (rank, capacity), `lora_alpha` (scaling; effective scale is `alpha/r`), `lora_dropout`. Wrong target
  modules is the #1 quiet quality killer.
- **Use the model's chat template.** Apply `tokenizer.apply_chat_template`; never hand-roll role markers.
  Mismatched templates silently wreck instruction-following.
- **Train loss on completions only.** Mask the prompt/instruction tokens so the model learns to *answer*,
  not to parrot the question. Pack sequences for throughput, but don't let packing cross-contaminate masking.
- **Pick the right base.** Instruction-tune from a *base* model; further-tune from an *instruct* model only
  when you want to preserve its chat behavior. Respect the license.
- **SFT first, then (optionally) preference alignment.** DPO/ORPO/KTO/PPO/GRPO come *after* a good SFT
  checkpoint — that work lives in [[rl-rlhf-frameworks]].
- **Watch catastrophic forgetting.** Narrow fine-tunes degrade general ability; PEFT mitigates it, mixing in
  some general/instruction data helps, and you must measure it with a base-model regression eval.
- **Evaluate like you mean it.** Held-out task metric + LLM-as-judge/human spot-check + a general-capability
  regression suite. Never leak eval examples into training. See [[ml-evaluation-evals]].
- **Serving:** merge the adapter for a single dedicated model, or keep adapters separate and serve **one base
  + many LoRAs** dynamically. See [[serving-frameworks]]/[[gke-inference-gateway]].

## Related skills

- `[[training-frameworks]]` — full/distributed pretraining, FSDP/DeepSpeed/Megatron; reach for it when PEFT
  isn't enough and you need multi-node full fine-tuning or pretraining.
- `[[rl-rlhf-frameworks]]` — preference/RL post-training (DPO/ORPO/KTO/PPO/GRPO, reward models) after SFT.
- `[[serving-frameworks]]` / `[[gke-inference-gateway]]` — serving merged models and multi-LoRA fleets.
- `[[inference-optimization]]` — quantization/throughput/latency once the fine-tune ships.
- `[[ml-evaluation-evals]]` — how to actually measure a fine-tune and catch regressions.
- `[[aiml-on-kubernetes]]` — running the fine-tuning jobs on K8s/GKE with accelerators.

---

# Reference — fine-tuning-peft

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

## 10. Rationalizations & rebuttals

The excuses that lead to a wasted training run or a model that looks good and ships broken:

- *"Just fine-tune it — faster than fiddling with prompts/RAG."* No. A prompt iterates in minutes with
  zero artifact to maintain; a fine-tune costs data + training + eval + a model you own through every base
  upgrade. Exhaust prompt → few-shot → RAG first, and only fine-tune what those genuinely can't do.
- *"Fine-tune so it knows our docs/data."* Fine-tuning teaches behavior, not facts. SFT'd knowledge goes
  stale and produces confident hallucinations. Knowledge is RAG or a tool; reserve fine-tuning for output
  shape, tone, narrow skills, or latency via a smaller model.
- *"More data is better — just scrape more."* Quality and distribution match dominate quantity. A few
  thousand clean, deduped, in-distribution examples beat tens of thousands of noisy ones. Dirty data
  overfits a brittle model. Clean and dedupe before scaling up.
- *"We'll add an eval/holdout later — let's just train first."* Then training loss is your only signal and
  it tells you nothing about generalization. Build the holdout and base-regression suite *before* you
  train, or you have a vibe, not a result.
- *"Decontamination is overkill, the overlap is tiny."* Eval leakage is the most common way teams fool
  themselves — inflated metrics, production disappointment. Decontaminate train against holdout and any
  public benchmark you report; treat contamination as a first-class risk.
- *"LoRA on `q_proj,v_proj` is the standard config."* It's a frequent quiet underperformer. Adapt
  attention + MLP projections (or `all-linear`); raise rank or try DoRA before concluding LoRA can't reach
  the bar.
- *"Merged fine, ship it."* A merge can regress: in-place merges into a 4-bit base degrade quality, and
  template drift between training and serving breaks behavior. Merge into a 16-bit base then re-quantize,
  and run the eval on the *merged* artifact — not just the adapter — before shipping.

---

## 11. Red flags (stop and reconsider)

- **Reaching for fine-tuning before prompt/few-shot/RAG have been seriously tried** — especially to inject
  facts/knowledge.
- **Dataset is a few hundred examples, or undeduped/garbled/mislabeled** — no decontamination step exists.
- **No true holdout split before training**, or the holdout gets touched during tuning.
- **Train/eval (or train/public-benchmark) overlap** — metrics look great, production behavior doesn't match.
- **Hand-rolled role markers / no `apply_chat_template`**, or nobody has inspected tokenized samples for
  correct special tokens / BOS / EOS.
- **No completion-only masking** — the model is being trained to echo instructions, not produce answers.
- **`target_modules` limited to `q_proj,v_proj`** (MLP projections omitted), or wrong modules for the
  architecture; added tokens without `modules_to_save`.
- **No base-model regression eval** — catastrophic forgetting (a task win that tanks general ability) goes
  unmeasured. Bonus flags: LR/epochs copied from an old blog post; merged model never re-evaluated.

---

## 12. Verification gate (definition of done)

The work is not done until every box is checked:

- [ ] **Cheaper levers ruled out.** Prompt, few-shot, and RAG were tried and demonstrably can't meet the
      bar for this task (and the task is behavior/skill/shape, not facts).
- [ ] **Data quality checked.** Deduped (exact + near-dup), garbled/truncated samples removed, classes/task
      types balanced, inputs/outputs match the inference distribution.
- [ ] **Decontaminated.** No training example overlaps the holdout or any reported public benchmark.
- [ ] **Chat template + masking correct.** Formatted via `apply_chat_template`; completion-only masking on
      (prompt labels = `-100`); a few tokenized samples inspected to confirm special tokens and that some
      labels ≠ -100.
- [ ] **Config sane.** `target_modules` cover attention + MLP (or `all-linear`); `alpha/r` ratio
      intentional; `modules_to_save` set if tokens were added; epochs/LR tuned, not copied blindly.
- [ ] **Holdout eval beats base.** Real task metric on the untouched holdout shows the fine-tune wins; for
      structured output, schema-valid rate measured directly.
- [ ] **No regression.** Base-model general-capability eval run on both — no unacceptable catastrophic
      forgetting.
- [ ] **Merged artifact verified.** If merging, merged into 16-bit (then re-quantized as needed) and the
      *merged* model re-evaluated; behavior matches the adapter.
- [ ] **Adapter/model serves.** Loads and routes correctly in the target serving stack (single merged
      model, or correct multi-LoRA routing against the matching base); base/template/data versions recorded.

---

## 13. Canonical references (real URLs; verify for updates)

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

---

# Fine-Tuning & PEFT — Worked Examples

Canonical config sketches to imitate. API names/flags move between library versions (it is 2026) — **verify
against the current `peft`/`trl`/Axolotl docs** before relying on a specific keyword. Imitate the shape and
the choices (target modules, masking, memory knobs), not the version-specific spelling.

---

## 1. QLoRA SFT with HF PEFT + TRL (Python)

A 4-bit NF4 base + LoRA adapters, instruction tuning with completion-only loss. This is the reference idiom.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

BASE = "meta-llama/Llama-3.1-8B"  # a *base* model for instruction tuning; respect its license

# --- 4-bit NF4 quantization for the FROZEN base (the "Q" in QLoRA) ---
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                 # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,     # compute in bf16
    bnb_4bit_use_double_quant=True,            # quantize the quant constants too
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:                # avoid pad==eos surprises during masking
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto",
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# --- LoRA: cover attention AND MLP projections, not just q/v ---
peft_config = LoraConfig(
    r=16,                      # rank = capacity; start 8-16, raise only if it underfits
    lora_alpha=32,            # effective scale = alpha/r = 2.0 here
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[          # or simply target_modules="all-linear"
        "q_proj", "k_proj", "v_proj", "o_proj",      # attention
        "gate_proj", "up_proj", "down_proj",         # MLP / FFN
    ],
    # use_dora=True,          # enable DoRA if plain LoRA underfits at this rank
    # modules_to_save=["embed_tokens", "lm_head"],  # ONLY if you added/resized tokens
)

# --- Data: format with the model's OWN chat template; never hand-roll role markers ---
ds = load_dataset("your-org/your-sft-data", split="train")  # columns -> chat "messages"

def to_text(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(to_text, remove_columns=ds.column_names)

cfg = SFTConfig(
    output_dir="out/llama3.1-8b-qlora",
    num_train_epochs=2,                  # 1-3; small data overfits fast
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch = 2 * 8 * num_gpus
    learning_rate=2e-4,                  # LoRA LR (full FT would be ~1e-5-2e-5)
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=2048,
    packing=True,                        # pack short samples; keep per-sample masking intact
    # completion_only_loss=True,         # mask the prompt -> loss only on the answer.
    #   (exact knob varies by TRL version: DataCollatorForCompletionOnlyLM /
    #    completion_only_loss / assistant-only masking — VERIFY in your version.)
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     peft_config=peft_config, processing_class=tokenizer)
trainer.train()
trainer.save_model("out/llama3.1-8b-qlora")   # saves the small ADAPTER (MBs), not the full model
```

**Before training, sanity-check two things on a real batch:**
1. Print a decoded sample — confirm the chat template, special tokens, and BOS/EOS look right.
2. Confirm some `labels != -100` (completions unmasked) and the prompt tokens *are* `-100` (masked).

---

## 2. Equivalent Axolotl-style YAML (config-driven)

Same run expressed declaratively. Field names track Axolotl's schema — **check current docs**; treat this as
the shape to imitate.

```yaml
base_model: meta-llama/Llama-3.1-8B
load_in_4bit: true            # QLoRA: 4-bit NF4 frozen base
adapter: qlora                # vs `lora` (16-bit base) or unset (full fine-tune)

# LoRA hyperparameters
lora_r: 16
lora_alpha: 32                # effective scale = alpha/r
lora_dropout: 0.05
lora_target_linear: true      # == target all linear layers (attention + MLP)
# lora_target_modules:        # or list them explicitly
#   [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

datasets:
  - path: your-org/your-sft-data
    type: chat_template        # use the model's chat template
chat_template: llama3
train_on_inputs: false         # <-- completion-only loss (mask the prompt). Critical.

sequence_len: 2048
sample_packing: true           # pack short samples for throughput
pad_to_sequence_len: true

# Memory / speed
bf16: true
gradient_checkpointing: true
flash_attention: true

# Schedule
micro_batch_size: 2
gradient_accumulation_steps: 8 # effective batch = micro * accum * num_gpus
num_epochs: 2
learning_rate: 0.0002
lr_scheduler: cosine
warmup_ratio: 0.03
optimizer: paged_adamw_8bit     # paged optimizer (the QLoRA spill-to-CPU trick)
```

Key parity points with the Python version: **4-bit NF4 base, LoRA over attention + MLP, completion-only
loss (`train_on_inputs: false`), packing, paged optimizer, bf16 + gradient checkpointing.**

---

## 3. Memory-budget note (single-GPU, order-of-magnitude)

VRAM ≈ base weights + adapter (grads+optimizer) + activations. Rough per-param weight costs:

| Component | Full FT (Adam, bf16) | LoRA (16-bit base) | QLoRA (NF4 base) |
|---|---|---|---|
| Base weights | ~2 B/param (trainable) | ~2 B/param (frozen) | **~0.5 B/param (frozen)** |
| Grads + Adam optimizer | ~14 B/param (master + 2 moments) | only on adapter (<1% params) | only on adapter (<1% params) |
| **Rough total for 8B (pre-activations)** | **~110GB+ → needs multi-GPU** | ~16GB+ | **~6–10GB** |

So **QLoRA puts a 7–13B fine-tune on a single 24GB GPU and ~70B on a single 80GB GPU** (QLoRA paper scale;
seq-length/batch dependent — **verify against current docs**). Activations scale with `batch × seq_len`;
gradient checkpointing slashes them. If you OOM, in order: lower `micro_batch_size` (raise grad-accum to
keep effective batch), shorten `sequence_len`, ensure gradient checkpointing is on, go 4-bit + paged
optimizer, then try Unsloth kernels. Full fine-tuning the same models needs FSDP/ZeRO across GPUs
([[training-frameworks]]).

---

## 4. Merge & serve note

**Option A — merge for a single dedicated model** (zero adapter overhead at inference):

```python
from peft import AutoPeftModelForCausalLM
# Load the base in 16-bit (NOT 4-bit) for a clean merge, attach the adapter, fold in ΔW:
model = AutoPeftModelForCausalLM.from_pretrained(
    "out/llama3.1-8b-qlora", torch_dtype="bfloat16")
merged = model.merge_and_unload()              # W' = W + (alpha/r)·B·A
merged.save_pretrained("out/llama3.1-8b-merged")
tokenizer.save_pretrained("out/llama3.1-8b-merged")
# Then optionally re-quantize the merged model for serving.
```

**Gotcha:** merging a LoRA *in place into a 4-bit base* can degrade quality. Merge into a 16-bit base, then
re-quantize for deployment. **Pin the chat template** used at serving to exactly the one used in training.

**Option B — keep adapters separate: one base + many LoRAs.** Don't merge. Ship the small adapter (MBs) and
let the serving stack load/route adapters per request, hosting one copy of the base weights for many cheap
fine-tunes. vLLM/SGLang and the GKE Inference Gateway support multi-LoRA serving and LoRA-aware routing —
see [[serving-frameworks]] and [[gke-inference-gateway]]. This is the right pattern when you have many
task-specific adapters over a shared base.
