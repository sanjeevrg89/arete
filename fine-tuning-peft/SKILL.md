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
