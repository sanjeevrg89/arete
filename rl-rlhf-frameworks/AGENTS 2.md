# AGENTS.md — RL / RLHF / RLAIF Post-Training for LLMs

> Cross-tool agent instructions (Codex, Cursor, Gemini, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`rl-rlhf-frameworks-guide.md`** next to this file — read
> it before designing or debugging an RLHF/RLAIF pipeline. Concrete artifacts to imitate (GRPO/PPO
> config sketch, Ray actor/learner placement, reward-model note) are in **`examples.md`**.
>
> Ecosystem moves fast (2026): treat all framework flags, class names, and backend bindings as
> shape-correct and **verify against current docs** of the version in use. Never fabricate flags,
> benchmarks, or APIs.

## Apply by default when working on LLM RL / RLHF / RLAIF post-training:

- **The loop is generate → score → update; it is generation-bound.** Treat rollout as an inference
  problem (vLLM/SGLang, paged KV cache, continuous batching, prefix caching). Optimize rollout throughput
  before the loss — decode is usually 60–80%+ of step time.
- **Know the four model roles and budget their memory:** policy/actor (trained), reference (frozen, KL
  anchor), reward (frozen scorer), critic/value (trained, PPO only). PPO can need ~4× SFT memory; GRPO/
  RLOO drop the critic; DPO/KTO/ORPO are offline and ~SFT cost. Count params + optimizer + grads +
  activations + KV cache for every resident model *before* launch.
- **KL to the reference is the safety rail.** Tune `kl_coef`/`β`, monitor KL as a first-class metric. A
  reward spike with a KL spike is reward hacking, not learning.
- **On-policy (PPO/GRPO/RLOO) needs fresh rollouts → weight sync to the inference engine every step.**
  Off-policy/offline (DPO/KTO/ORPO, rejection sampling) trains on fixed preference data → no rollout, no
  RM, no resharding. The "do we generate during training?" question drives all the systems complexity.
- **Weight resharding (learner FSDP/Megatron layout → vLLM/SGLang layout) is the hardest edge.** It's on
  the critical path and a silent-corruption hazard. Validate sync with a checksum or fixed-prompt sanity
  generation; a layout/precision/name mismatch degrades completions with no crash.
- **Placement:** colocated (shared GPUs, cheap sync, HBM contention, can't scale halves independently)
  vs disaggregated (separate rollout/training pools, scale independently, weights cross the network).
  Choose by scale and accelerator supply.
- **Pick the method for the objective:** DPO/ORPO/KTO for stable cheap offline preference alignment;
  GRPO/RLOO for online reasoning/verifiable-reward tasks at lower memory than PPO; PPO only when you need
  a learned value function. Rejection sampling / best-of-n / RAFT is the stable baseline to try first.
- **Pick the framework for scale + ecosystem:** TRL (PyTorch, ≤few nodes, DPO family); veRL/HybridFlow or
  OpenRLHF (large multi-node on-policy PPO/GRPO); NeMo-Aligner/NeMo-RL (Megatron/NeMo, largest models);
  MaxText-RL (JAX/TPU); RLlib (non-LLM RL). All route rollouts to vLLM/SGLang and training to
  FSDP/DeepSpeed/Megatron.
- **Reward hacking is the default failure mode.** The policy games the reward model, not the goal. Guard
  with KL, RM ensembles/normalization, length/format penalties, and held-out eval. **The reward curve is
  not success — eval is.** Use an independent judge.
- **Preference data and the RLAIF constitution/judge prompt are the product.** Dedup, balance,
  decontaminate against eval, version with the run. Audit any LLM judge for position/verbosity/self-
  preference bias.
- **On Kubernetes:** Ray/KubeRay is the common substrate ([[ray-on-kubernetes]]); gang-schedule the
  heterogeneous job and manage quota via [[kueue-advanced]]; pin rollout vs learner pods to the right
  accelerator pools; keep on-policy weight sync inside one cluster.

## Definition of done for an RLHF/RLAIF change
- Memory budgeted across all resident model copies; no OOM at step 1.
- KL monitored and bounded; reward-hacking guards in place.
- Held-out eval (independent of training reward/judge) shows real improvement and no capability regression.
- Weight sync validated (on-policy); preference/judge data versioned and decontaminated.
- Framework flags verified against current docs, not copied blindly.

Full rationale, tables, troubleshooting, and decision guides: **`rl-rlhf-frameworks-guide.md`**.
