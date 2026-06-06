# AGENTS.md — Fine-Tuning & PEFT

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`fine-tuning-peft-guide.md`** next to this file — read it
> before designing or running a fine-tune, and apply it. A concrete LoRA/QLoRA SFT config to imitate (TRL
> + HF PEFT and Axolotl-style YAML) plus memory-budget and merge/serve notes are in **`examples.md`**.
> This file is the always-on summary.
>
> Scope: adapting pretrained LLMs to tasks/domains via SFT and PEFT, cost-effectively. Full/distributed
> pretraining → [[training-frameworks]]; preference/RL post-training (DPO/PPO/GRPO) → [[rl-rlhf-frameworks]];
> serving adapters → [[serving-frameworks]]/[[gke-inference-gateway]].

## Apply by default on any fine-tuning task:

- **Don't fine-tune by reflex.** Try prompting → few-shot → RAG first. Fine-tune for behavior/format/style/
  latency/distillation, **not** to inject facts (that's RAG). Account for the maintenance tax: every base
  upgrade means re-running the whole pipeline.
- **Data quality > quantity.** A few thousand clean, diverse, correctly-formatted examples beat 100k noisy
  ones. Dedupe, balance, and **decontaminate against the eval/holdout and public benchmarks.**
- **Default to PEFT (LoRA/QLoRA).** Updates <1% of params, ~matches full-FT quality on task adaptation,
  tiny adapter artifacts, far less catastrophic forgetting. Use full FT only when PEFT demonstrably
  underperforms — and then it's multi-GPU FSDP/ZeRO ([[training-frameworks]]).
- **QLoRA = 4-bit NF4 frozen base + bf16 LoRA + paged optimizer.** Via `BitsAndBytesConfig(load_in_4bit,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bf16, bnb_4bit_use_double_quant=True)` +
  `prepare_model_for_kbit_training`. Single-GPU fits depend on seq/batch/version — **verify current docs.**
- **LoRA knobs:** `target_modules` cover attention **and** MLP (`q,k,v,o,gate,up,down` or `all-linear`) —
  only `q,v` is the #1 quiet underperformer. `r` = capacity (start 8–16). Effective scale = `alpha/r`.
  Use `modules_to_save` for resized embeddings/LM head. Try **DoRA** (`use_dora=True`) when LoRA underfits.
- **Chat template:** always `tokenizer.apply_chat_template`; never hand-roll role markers / special tokens.
  A mismatch silently breaks instruction-following.
- **Loss on completions only:** mask prompt tokens to `-100`; pack for throughput but keep masking per-sample
  and don't let attention cross sample boundaries.
- **Base choice:** instruction-tune from a *base* model; further-tune an *instruct* model only to preserve
  its chat behavior. Respect the license.
- **Hyperparameters (start, then tune):** LoRA LR ~1e-4–3e-4 (full FT ~1e-5–2e-5), cosine + warmup, 1–3
  epochs, bf16, gradient checkpointing, effective batch via grad-accum. Too many epochs/too-high LR overfit
  small data fast.
- **SFT first, alignment after.** DPO/ORPO/KTO/PPO/GRPO come after a good SFT checkpoint → [[rl-rlhf-frameworks]].
- **Evaluate or it didn't happen:** held-out task metric + judge/human spot-check + **base-model regression
  suite** for catastrophic forgetting. Build the holdout before training; never leak it. See
  [[ml-evaluation-evals]].
- **Serving:** merge the adapter for a single dedicated model (merge into a 16-bit base, then re-quantize),
  or keep adapters separate for **one base + many LoRAs** dynamic serving →
  [[serving-frameworks]]/[[gke-inference-gateway]].

## Stack
HF `peft`+`transformers`+`trl` (reference) · `bitsandbytes` (4-bit + paged opt) · Axolotl (YAML) ·
Unsloth (fast kernels) · Llama-Factory · torchtune · NeMo. Pin versions; APIs and defaults move.

## Definition of done for a fine-tune
Clean/decontaminated data · correct chat template + completion-only masking verified on sample batches ·
trained with sane LR/epochs · **holdout task metric reported** · **base regression eval reported** (no
forgetting) · merge/serve path validated end-to-end · base/template/data versions recorded.

## Top anti-patterns to flag
Fine-tuning when prompt/RAG suffices · tiny/dirty data · no eval/holdout · eval leakage into train · wrong
`target_modules` · chat-template mismatch · loss on the prompt · unmeasured catastrophic forgetting ·
too many epochs · 4-bit in-place adapter merge.
