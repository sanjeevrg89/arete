# AGENTS.md — AI Research Science

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ai-research-science-guide.md`** next to this file — read
> it before reasoning about training/inference/fine-tuning/RL/RLHF *science*, and apply it. Worked
> artifacts (ablation-design template, RLHF pipeline in prose, paper read/reproduce checklist) are in
> **`examples.md`**. This file is the always-on summary.
>
> **Scope:** this is the research-science layer — the *why / methods / open problems*. For *how to run*
> the thing (frameworks, launch flags, cluster shapes) defer to the engineering siblings:
> `[[training-frameworks]]`, `[[rl-rlhf-frameworks]]`, `[[fine-tuning-peft]]`, `[[ml-frameworks]]`,
> `[[inference-optimization]]`, `[[serving-frameworks]]`, `[[pretraining-data-tokenizers]]`,
> `[[ml-evaluation-evals]]`.

## Apply by default when reasoning about model science

- **Accuracy over completeness. Never fabricate** API fields, hyperparameter values, benchmark numbers,
  or arXiv IDs. The field moves weekly (2026) — flag fast-moving claims "verify against current
  docs/papers". Cite IDs only when confident; otherwise name authors/title + "(verify the citation)".
- **A result is a hypothesis until a control kills the confound.** Change one variable; match
  compute/tokens/data; report seeds + variance; learning curves > single endpoints. An unmatched baseline
  invalidates the comparison. Reproduce the baseline before trusting your delta.
- **Eval rigor dominates.** Watch contamination/leakage, metric-induced "emergence", prompt-format and
  answer-extraction artifacts. A delta inside the noise band is nothing. Defer to `[[ml-evaluation-evals]]`.
- **Scaling laws are the compass.** Chinchilla ≈20 tokens/param is compute-optimal; inference cost
  justifies overtraining smaller models. Data-constrained scaling: repeats decay after a few epochs.
- **Architecture is mostly plumbing above a threshold; a few changes earn their keep** (GQA, RoPE+YaRN,
  pre-LN/RMSNorm, SwiGLU, FlashAttention). MoE lives/dies on routing + load-balancing. SSMs/Mamba trade
  exact recall for linear-time state; hybrids win. Be suspicious of tricks that vanish at scale.
- **Optimization is empirical.** AdamW default (decoupled decay); Muon/Lion are challengers (verify).
  Gradient noise scale → critical batch size (past it, compute is wasted). Loss spikes = numerics:
  QK-norm, z-loss, init, bf16 master weights, fp8 with care.
- **Test-time compute is a scaling axis:** CoT, self-consistency, verifier-guided/best-of-N search,
  o1/R1-style RL-trained long reasoning. ORM scores answers; PRM scores steps (denser, costlier).
- **PEFT works because fine-tuning is low intrinsic rank** (LoRA). Watch catastrophic forgetting (replay,
  low LR, adapters). Merge with task arithmetic/TIES/DARE/SLERP/soups. Distillation transfers soft
  targets / on-policy behavior.
- **RL/RLHF = reward design under Goodhart's law.** Bradley-Terry RMs get hacked (verbosity, format,
  sycophancy); KL-to-reference is the leash; RM ensembles + early stop curb overoptimization. PPO is
  the full (brittle, 4-model) loop; **DPO** reparameterizes the reward away (offline, stable);
  **GRPO/RLVR** drop the critic and use verifiable/rule-based rewards for reasoning. On-policy beats
  offline at the top (verify); offline suffers distribution shift (mitigate with iterative DPO).
- **Frontier:** reasoning/agents, mechanistic interpretability (superposition, SAEs, induction heads),
  data efficiency / model collapse, scaling-vs-algorithms (both compound), scalable-oversight safety.

## Definition of done for a research claim
State the mechanism and what would falsify it · matched baseline + controls · seeds/variance reported ·
contamination ruled out · scale-trend (not single point) where capability is at issue · citations real
(or flagged to verify). Report honestly when evidence is weak or fast-moving.
