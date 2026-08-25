---
name: rl-rlhf-frameworks
description: Expert RL / RLHF / RLAIF post-training for LLMs at production scale — reward modeling, PPO,
  DPO, GRPO, RLOO, KTO, ORPO, RLAIF/Constitutional AI, rejection sampling / best-of-n, on- vs off-policy,
  KL control and the reference model. Use when building or debugging an RLHF/RLAIF pipeline, choosing
  between PPO/DPO/GRPO, picking a framework (TRL, veRL/HybridFlow, OpenRLHF, NeMo-Aligner/NeMo-RL, RLlib,
  TRLX, MaxText-RL), wiring the generate→score→update loop, budgeting memory across policy/reference/
  reward/critic copies, splitting rollout (vLLM/SGLang) from learner (FSDP/Megatron), doing weight
  resharding, placing actors/learners colocated vs disaggregated on Ray/Kubernetes, or fighting reward
  hacking, training instability, and the generation-throughput bottleneck. Covers preference-data
  handling, eval, and a PPO-vs-DPO-vs-GRPO and framework decision guide.
---

# RL / RLHF / RLAIF Frameworks for LLM Post-Training

Apply the judgment of an engineer who has run large post-training pipelines for years: who knows that
RLHF at scale is a **distributed-systems problem first and an RL problem second**, that the loop is
generation-bound, that the win usually comes from rollout throughput and KL discipline rather than a
fancier loss, and who picks the simplest method that achieves the objective.

## How to use this skill

1. **Read `rl-rlhf-frameworks-guide.md`** in this directory — the full reference (algorithms, the
   four-model systems architecture, frameworks, K8s/Ray placement, performance, troubleshooting,
   decision guides). Apply it to the task.
2. For concrete artifacts to imitate — a GRPO/PPO training-config sketch, an actor/learner Ray
   placement sketch, and a reward-model note — read **`examples.md`**. Treat every flag/field as
   shape-correct, not verbatim: **verify against the current docs** of the framework you target.
3. Match the surrounding repo/cluster conventions (framework, scheduler, accelerator type). Apply the
   correctness rules — KL control, reward-hacking guards, memory budgeting, eval discipline —
   regardless of framework.

## Essentials (full detail in `rl-rlhf-frameworks-guide.md`)

- **The loop is generate → score → update.** Generation (autoregressive decode) dominates wall-clock;
  treat it as an inference-serving problem (vLLM/SGLang, KV cache, continuous batching), not a training
  afterthought. Optimize rollout throughput before touching the loss.
- **Know the four model roles and their cost.** Policy/actor (trained), reference (frozen, KL anchor),
  reward (frozen scorer), critic/value (trained, PPO only). PPO holds up to four; DPO/GRPO/RLOO drop the
  critic, GRPO/DPO often drop the reward model too. Memory budget = sum of the copies you keep.
- **KL to the reference model is the safety rail.** It stops the policy from drifting into degenerate,
  reward-hacking, or off-distribution text. Tune `β`/`kl_coef`; watch KL as a first-class metric. A KL
  spike with a reward spike is almost always reward hacking, not progress.
- **On-policy (PPO, GRPO, RLOO) needs fresh rollouts from the current policy** → requires weight sync
  from learner to the inference engine every step. **Off-policy/offline (DPO, KTO, ORPO)** trains on a
  fixed preference dataset → no rollout, no reward model, no resharding; vastly simpler to operate.
- **Weight resharding is the hard systems edge.** Learner shards (FSDP/Megatron, TP/PP) differ from the
  inference engine's layout; you must gather/reshard and push weights to vLLM/SGLang each step. This sync
  is a top latency and correctness hazard. Frameworks that solve it well (veRL/HybridFlow, OpenRLHF) earn
  their keep here.
- **Colocated vs disaggregated placement.** Colocated: actor and learner share GPUs, time-sliced (saves
  hardware, simpler sync, contends for memory). Disaggregated: separate rollout and training pools
  (scales independently, needs network weight transfer). Choose by scale and accelerator supply.
- **Pick the method for the objective, not the hype.** DPO/ORPO/KTO when you have (or can build)
  preference data and want stability + low cost. GRPO/RLOO for reasoning/verifiable-reward tasks at lower
  memory than PPO. PPO when you need a learned value function and maximal control. See the decision guide.
- **Pick the framework for scale + ecosystem.** TRL for single-node/research and DPO-family on PyTorch;
  veRL or OpenRLHF for large multi-node on-policy PPO/GRPO; NeMo-Aligner/NeMo-RL for Megatron/NeMo shops;
  MaxText-RL for JAX/TPU. RLlib for general RL, not LLM-first.
- **Reward hacking is the default failure mode.** The policy games the reward model, not the goal. Guard
  with KL, reward-model ensembles/normalization, length penalties, held-out eval, and human/LLM spot
  checks. A rising reward curve is not success — eval is.
- **Preference data is the product.** Garbage pairs → garbage policy. Dedup, balance, decontaminate
  against eval, and version it. For RLAIF/Constitutional AI, the AI-feedback prompt and critique/revise
  chain are the dataset; treat them with the same rigor.
- **Budget memory across copies before you launch.** Per model: weights + (optimizer + grads for trained
  ones) + activations + KV cache for rollouts. Four-model PPO can need ~4× a plain SFT job; offline DPO is
  close to SFT. Right-size before you OOM at step 1.

## Related skills

- `[[ray-on-kubernetes]]` — Ray/KubeRay is the common substrate for RLHF actor/learner placement; defer
  Ray cluster depth there.
- `[[serving-frameworks]]` — vLLM/SGLang/Dynamo power the rollout/generation half of the loop.
- `[[training-frameworks]]` — FSDP/DeepSpeed/Megatron/NeMo power the learner half.
- `[[ml-frameworks]]` — PyTorch vs JAX/XLA, GPU vs TPU foundations that decide framework choice.
- `[[maxtext-jax-llm]]` — JAX/TPU post-training path (MaxText-RL).
- `[[kueue-advanced]]` — queueing, quota, and gang-scheduling the heterogeneous RLHF job on K8s.
- `[[aiml-on-kubernetes]]` — umbrella for training/inference/RL/RLHF on Kubernetes & GKE.
