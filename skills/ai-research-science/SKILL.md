---
name: ai-research-science
description: The research-science layer of frontier LLM development — the WHY behind the methods, the
  theory, and the open problems, across the whole model lifecycle. Use when you need to reason like a
  research scientist rather than run a pipeline: forming and falsifying hypotheses, designing rigorous
  ablations/controls, scaling laws (Kaplan vs Chinchilla, data-constrained, emergent-abilities debate),
  architecture science (attention MHA/MQA/GQA/MLA, RoPE/ALiBi/YaRN, pre-LN/RMSNorm/QK-norm, SwiGLU,
  MoE routing & load-balancing, Mamba/SSMs & hybrids, long-context), training science (Adam/AdamW/Lion/
  Muon, gradient noise scale & batch size, loss spikes/z-loss/stability, bf16-vs-fp8 dynamics, the
  pretraining objective), inference & test-time-compute science (sampling/calibration theory, CoT,
  self-consistency, o1/R1-style long reasoning, process- vs outcome-reward search), fine-tuning science
  (why LoRA works / intrinsic dimensionality, catastrophic forgetting, model merging — task arithmetic/
  TIES/DARE/SLERP/soups, distillation theory), and DEEPEST on RL/RLHF post-training (reward modeling &
  Bradley-Terry, reward hacking/Goodhart, PPO objective & KL-to-reference, the direct-alignment family
  DPO/IPO/KTO/ORPO/SimPO, GRPO & RLVR, PRMs, RLAIF/Constitutional AI, on/off-policy, open problems).
  Reach for it to read/reproduce a paper, critique an experiment, or choose a research direction. Defer
  the how-to-run engineering to the sibling skills.
---

# AI Research Science

Apply the judgment of a research scientist at a frontier lab who has trained and post-trained large
models, published, reviewed, and reproduced others' work — and who knows the difference between a result
that will hold and a result that is a measurement artifact. This skill is the **why / the methods / the
frontier**, not the how-to-run. When the task is implementation (launch flags, framework APIs, cluster
shapes), defer to the engineering siblings below and stay on the science: theory, mechanism, evidence,
open problems.

The bar: **accuracy over completeness, mechanism over recipe, falsifiable claims over vibes.** The field
moves weekly; treat every quantitative claim and citation as provisional and verify against current
papers/docs.

## How to use this skill

1. **Read `ai-research-science-guide.md`** — the deep reference (methodology, architecture, training,
   inference, fine-tuning, RL/RLHF, frontier). Apply its mental models to the task at hand.
2. For worked artifacts — an **ablation-study design template**, an **RLHF pipeline in prose** (SFT →
   reward model → PPO/DPO/GRPO with knobs & failure modes), and a **read-and-reproduce-a-paper
   checklist** — read **`examples.md`**.
3. Ground claims in the named canonical works; for any arXiv ID you are not certain of, name the
   paper/authors and write "(verify the citation)". Never fabricate fields, numbers, or IDs.

## Essentials (full detail in `ai-research-science-guide.md`)

- **The result is a hypothesis until a control kills the confound.** Change one thing; match
  compute/data/tokens; report seeds and variance; an ablation without a matched baseline proves nothing.
  Eval rigor and contamination dominate everything downstream — see [[ml-evaluation-evals]].
- **Scaling laws are the lab's compass.** Kaplan undertrained models; **Chinchilla** (Hoffmann et al.,
  2022) showed compute-optimal is ≈20 tokens/param. Inference cost now pushes far past Chinchilla-optimal
  (smaller model, more tokens). Data-constrained scaling and the "emergent abilities = metric artifact"
  debate are live; predictability of capability is the real prize.
- **Architecture is mostly plumbing above a threshold; a few changes earn their keep.** GQA, RoPE+YaRN,
  RMSNorm/pre-LN, SwiGLU, and FlashAttention are robust wins. MoE buys capacity at fixed FLOPs but lives
  or dies on routing/load-balancing. SSMs/Mamba trade exact recall for linear-time state; hybrids win in
  practice.
- **Optimization is an empirical science.** AdamW is the default; Muon/Lion are credible challengers
  (verify current evidence). The **gradient noise scale** (McCandlish et al., 2018) predicts the
  critical batch size — past it, you waste compute. Loss spikes are a numerics/stability problem:
  z-loss, QK-norm, careful init, bf16 master weights, fp8 with care.
- **Test-time compute is a scaling axis.** CoT, self-consistency, and verifier-guided search trade
  inference FLOPs for accuracy; o1/DeepSeek-R1-style long reasoning is RL-trained, not just prompted.
  Process reward models (PRMs) supervise steps; outcome reward models (ORMs) supervise answers.
- **LoRA works because fine-tuning updates are low intrinsic rank** (Aghajanyan et al.; Hu et al.,
  2021). Merging (task arithmetic, TIES, DARE, SLERP, model soups) composes capabilities without
  retraining. Distillation transfers the teacher's soft targets / on-policy behavior. Practice →
  [[fine-tuning-peft]].
- **RL/RLHF is reward design under Goodhart's law.** Bradley-Terry reward models overfit and get hacked;
  KL-to-reference is the leash; PPO is unstable and expensive; **DPO** reparameterizes the reward away
  and trains on preferences directly; **GRPO/RLVR** drop the critic and use verifiable rewards for
  reasoning. Length bias, sycophancy, and mode collapse are the recurring failures. Engineering →
  [[rl-rlhf-frameworks]].
- **The frontier is reasoning, agents, interpretability, and data/algorithmic efficiency.** "Is it
  scaling or algorithms?" is the central strategic question; both compound. Alignment is an open
  research problem, not a solved checkbox — see [[responsible-ai-governance]].

## Related skills

- `[[training-frameworks]]` — DDP/FSDP/Megatron/DeepSpeed/MaxText: how to actually run distributed training.
- `[[rl-rlhf-frameworks]]` — TRL/veRL/OpenRLHF/NeMo: how to wire and scale the RLHF/RLVR loop.
- `[[fine-tuning-peft]]` — running LoRA/QLoRA/DoRA, SFT data, adapter merging/serving.
- `[[ml-frameworks]]` — PyTorch/JAX/XLA, CUDA/TPU substrate, torch.compile, kernels.
- `[[inference-optimization]]` — quantization/pruning/distillation/speculative decoding mechanics.
- `[[serving-frameworks]]` — vLLM/SGLang/TensorRT-LLM: serving the model.
- `[[pretraining-data-tokenizers]]` — corpus curation, dedup/decontamination, tokenizer design.
- `[[ml-evaluation-evals]]` — metrics, benchmarks, contamination, LLM-as-judge — the measurement backbone.
- `[[multimodal-ml]]` — vision/audio fusion, encoders, cross-modal training.
- `[[responsible-ai-governance]]` — safety/alignment policy, oversight, governance.
- `[[maxtext-jax-llm]]` — reference JAX LLM implementation to read alongside the science.
