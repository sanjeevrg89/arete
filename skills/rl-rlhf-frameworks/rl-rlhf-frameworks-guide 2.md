# RL / RLHF / RLAIF Frameworks — Deep Reference

Post-training an LLM with reinforcement learning is the alignment stage after pretraining and SFT. The
goal: shift the policy's distribution toward outputs humans (or an AI proxy) prefer, without destroying
the capabilities learned earlier. The recurring lesson at scale is that **the hard part is the
distributed system, not the gradient**. This guide is organized so you can reason about both.

> Fast-moving ecosystem (it is 2026). Framework APIs, default flags, supported algorithms, and the
> rollout/learner backends they bind to change frequently. Treat specific flags, class names, and
> backend names here as shape-correct guidance and **verify against the current docs** of the version
> you run.

---

## 1. Mental model: the generate → score → update loop

Every online RLHF method is the same outer loop:

1. **Generate (rollout).** Sample completions from the current policy for a batch of prompts. This is
   autoregressive decoding — an *inference* workload. It dominates wall-clock (often 60–80%+ of step
   time). This is why RLHF is called **generation-bound**.
2. **Score (reward).** Assign a scalar (or per-token) reward to each completion: a learned reward model,
   a verifiable checker (unit tests, math answer match, a rules engine), or an LLM-as-judge (RLAIF).
3. **Update (learn).** Compute advantages and a policy-gradient (or preference) loss, backprop, step the
   optimizer. This is a *training* workload. Then **sync** the new weights back to the inference engine
   for the next rollout.

Offline/off-policy methods (DPO, KTO, ORPO) collapse this: the rollouts and rewards are precomputed into
a fixed preference dataset, so the loop is just "update" — a supervised-style training job. That single
distinction (do you generate during training?) drives most of the systems complexity.

**Why RLHF is heterogeneous.** Generation wants high-throughput inference kernels (paged KV cache,
continuous batching, low precision). Training wants sharded optimizer state, high-precision grads,
collective comms. These are different runtimes with different optimal hardware utilization. A good RLHF
framework is fundamentally a **scheduler that marries an inference engine to a training engine** and
moves weights between them.

---

## 2. The four model roles and their memory cost

| Role | Trained? | Purpose | Present in |
|---|---|---|---|
| **Policy / actor** | Yes | The model being optimized; generates rollouts | All methods |
| **Reference** | No (frozen) | KL anchor — penalizes drift from the SFT model | PPO, GRPO, RLOO, DPO/KTO* |
| **Reward (RM)** | No (frozen) | Scores completions | PPO, RLOO, best-of-n, RAFT |
| **Critic / value** | Yes | Estimates V(s) for advantage (GAE) | PPO only |

\*DPO/KTO use the reference *implicitly* in the loss (log-ratio to the reference policy); ORPO drops the
reference entirely.

**Memory budget.** Per model copy you pay: parameters; plus, for *trained* models, optimizer states
(Adam ≈ 2× params in fp32) and gradients; plus activations during the backward pass; plus, for the
*generating* model, KV-cache for in-flight rollouts. Rough rule of thumb:

- **Offline DPO/KTO/ORPO:** ~SFT-scale. Policy (trained) + reference (frozen, can be the same checkpoint
  reloaded or even folded into precomputed logprobs). Cheapest to run.
- **GRPO / RLOO:** policy (trained) + reference (frozen) + reward source. No critic → meaningfully
  cheaper than PPO. The advantage is computed from *group* statistics (GRPO) or leave-one-out baselines
  (RLOO) instead of a learned value head.
- **PPO:** up to four models resident. The critic doubles your trained-model footprint vs GRPO. Can need
  ~4× a plain SFT job's memory. This is the single biggest reason teams moved to GRPO/RLOO for
  reasoning-style training.

Always budget the copies *before* launch. The classic failure is OOM at step 1 because the reward model
and KV cache weren't counted.

---

## 3. Algorithms at a systems level

### Reward modeling (RM)
Train a model (usually init from the SFT checkpoint with a scalar head) on **preference pairs**
(chosen, rejected) with the Bradley–Terry loss: `-log σ(r(chosen) − r(rejected))`. The RM is the proxy
objective the policy optimizes; its quality caps the whole pipeline. Watch for **reward over-
optimization**: as the policy pushes RM score up, true quality eventually drops (Goodhart). Mitigations:
RM ensembles, score normalization/whitening, RM trained on on-policy data, and KL control.

### PPO (Proximal Policy Optimization)
The canonical online RLHF method. Generate rollouts, score with RM, compute token-level advantages via
**GAE** using the critic, optimize the clipped surrogate objective with a per-token KL penalty to the
reference. Strengths: maximal control, well-understood, handles dense/shaped rewards. Costs: four models,
the critic is finicky to train, sensitive to many hyperparameters (clip range, GAE λ, KL coef, value
loss coef). Use when you genuinely need a learned value function.

### GRPO (Group Relative Policy Optimization)
Drops the critic. For each prompt, sample a **group** of G completions, score all, and use the
group-normalized reward (`(r − mean)/std`) as the advantage for every token in that completion. Same KL
control as PPO. Strengths: no value model (big memory/stability win), excellent for **verifiable-reward**
reasoning (math, code) where rewards are sparse and group sampling gives a clean baseline. Popularized by
DeepSeek; now the default for reasoning RL in many shops. Watch: group size vs throughput tradeoff;
length/format reward hacking; std-normalization instability on low-variance groups.

### RLOO (REINFORCE Leave-One-Out)
Also critic-free. Uses a leave-one-out baseline: each sample's advantage is its reward minus the mean of
the *other* samples in the group. Simpler than PPO, competitive for many alignment tasks, lower variance
than vanilla REINFORCE. Sits in the same family as GRPO.

### DPO (Direct Preference Optimization)
**Offline / off-policy.** No reward model, no rollout, no critic. Trains directly on preference pairs
with a loss that is mathematically equivalent to the RLHF objective under a Bradley–Terry assumption,
using the **log-ratio of policy to reference**. The reference (the SFT model) is the KL anchor baked into
the loss. Strengths: stable, cheap, simple to operate (it's basically a fine-tune). Costs: only as good
as your fixed preference data; off-policy, so it can't discover new behaviors the way online RL can; the
**`β`** temperature controls how hard it pushes vs staying near the reference. The pragmatic default
starting point for most preference alignment.

### KTO (Kahneman–Tversky Optimization)
Like DPO but needs only **per-example binary labels** (good/bad), not pairs — easier data collection.
Based on a prospect-theory utility. Use when you have thumbs-up/down signal rather than ranked pairs.

### ORPO (Odds Ratio Preference Optimization)
Folds preference optimization **into SFT** — a single stage, **no reference model**, an odds-ratio
penalty on the rejected response added to the SFT NLL loss. Cheapest of all (one model, one pass). Use to
do SFT and alignment together when you want minimal moving parts.

### RLAIF / Constitutional AI
Replace human preference labels with **AI feedback**. Constitutional AI: a "constitution" of principles;
the model critiques and revises its own outputs (the SFT-on-revisions phase), then an AI labeler ranks
pairs to train the RM (the RL phase). RLAIF generalizes this — an LLM judge produces the preferences. The
*prompt/constitution and the critique→revise chain are now your dataset*; their design is the work.
Scales preference collection cheaply but inherits the judge's biases (position bias, verbosity bias,
self-preference). Calibrate and audit the judge.

### Rejection sampling / best-of-n / RAFT / RFT
The simplest "RL." Sample n completions, score with the RM/verifier, keep the best (best-of-n) or all
above a threshold, and **SFT on the survivors** (RAFT / rejection-sampling fine-tuning / RFT). No policy
gradient at all — just generate, filter, fine-tune. Surprisingly strong, very stable, trivially
parallel, and a great baseline before reaching for PPO/GRPO. Many "RL" pipelines are really iterated
rejection sampling.

### On-policy vs off-policy; KL control
**On-policy** (PPO/GRPO/RLOO): the data comes from the *current* policy, so you must re-generate (and
therefore re-sync weights) every step. **Off-policy/offline** (DPO/KTO/ORPO, rejection-sampling): trains
on a fixed dataset; no live generation. On-policy can discover behaviors but is far costlier to operate.

**KL to the reference** is the universal safety rail. It keeps the policy on the manifold of fluent,
in-distribution text and is the primary defense against reward hacking and mode collapse. Implementations
vary: PPO/GRPO add a per-token KL penalty (coef `kl_coef`/`β`) or a KL-in-reward term; DPO embeds it in
the loss. **Monitor KL as a first-class metric.** A reward spike *with* a KL spike is reward hacking, not
learning. Some setups adapt the KL coefficient to hit a target KL.

---

## 4. RLHF systems architecture

### Rollout (inference) vs learner (training) split
The two halves want different runtimes:

- **Rollout engine:** vLLM or SGLang. Paged KV cache, continuous/in-flight batching, prefix caching
  (reuse the shared prompt prefix across a sampled group — a big GRPO win), tensor parallelism for big
  models. Optimized for throughput of many short-to-medium generations.
- **Learner engine:** FSDP (PyTorch) or Megatron-LM / NeMo (TP+PP+DP) or DeepSpeed ZeRO. Optimized for
  sharded optimizer state and collective backward passes.

The framework's job is to schedule these and move weights between them.

### Weight synchronization / resharding
After each learner step (on-policy), the updated policy weights must reach the inference engine — but the
two engines shard the model differently (FSDP flat-shards parameters across DP ranks; Megatron splits by
TP/PP; vLLM has its own TP layout). You must **gather and reshard** the learner's weights into the
inference engine's layout and load them in. This is the **hardest and most error-prone edge** of online
RLHF:

- It is on the critical path every step → a latency hot spot. Frameworks optimize it with NCCL/collective
  broadcasts, CUDA-IPC handles for colocated GPUs, or staged transfers.
- A layout/precision/naming mismatch silently corrupts the policy — completions degrade with no crash.
  Validate sync with a checksum or a fixed-prompt sanity generation.

veRL/HybridFlow and OpenRLHF invest heavily here; this is much of why they exist.

### Colocated vs disaggregated placement
- **Colocated:** actor (rollout) and learner share the same GPUs, time-sliced — generate, then swap to
  train on the same devices. Saves hardware, makes weight sync cheap (same memory/IPC, no network), but
  the two phases contend for HBM (KV cache vs optimizer state) and you can't independently scale them.
  Often best when accelerators are scarce.
- **Disaggregated:** separate pools — a rollout cluster (many inference replicas) and a training cluster.
  Each scales independently (add rollout replicas to fix the generation bottleneck), but weights now
  cross the network and the pools can idle waiting on each other. Often best at large scale with ample
  hardware. Hybrid placements (some roles colocated, some split) are common.

### Why generation-bound matters
Because decode dominates, the highest-leverage optimizations are almost always on the rollout side:
faster inference engine, larger effective batch, prefix caching, more rollout replicas, shorter max
generation length, and overlapping generation with learning. Speeding up the optimizer rarely moves the
needle.

---

## 5. Frameworks: what each is, when to use

| Framework | Backers / stack | Rollout backend | Learner backend | Best for |
|---|---|---|---|---|
| **TRL** | Hugging Face, PyTorch | vLLM (online) | Accelerate/FSDP, DeepSpeed, PEFT | Single-/few-node, research, the DPO/KTO/ORPO/GRPO/PPO trainers, fast iteration |
| **veRL (HybridFlow)** | ByteDance, PyTorch | vLLM, SGLang | FSDP, Megatron | Large multi-node on-policy PPO/GRPO; strong placement + resharding; production scale |
| **OpenRLHF** | community, PyTorch + Ray | vLLM | DeepSpeed (ZeRO-3), FSDP | Scalable PPO/GRPO/RLOO/DPO on Ray; clean distributed PPO; accessible |
| **NeMo-Aligner / NeMo-RL** | NVIDIA, PyTorch | TensorRT-LLM, vLLM | Megatron-Core (TP/PP/EP) | Megatron/NeMo shops, largest models, NVIDIA-optimized stacks |
| **Ray RLlib** | Ray/Anyscale | — (general RL) | Ray Train | General/classic RL (not LLM-first); reach for it only for non-LLM RL |
| **MaxText-RL** | Google/JAX ecosystem | JAX inference / sharded decode | JAX/XLA on TPU (and GPU) | JAX/TPU post-training; pairs with [[maxtext-jax-llm]] |
| **TRLX** | CarperAI (legacy) | — | Accelerate/Megatron | Earlier PPO/ILQL library; largely superseded — verify maintenance before adopting |

How they map: every serious online framework routes **rollouts to vLLM or SGLang** and **training to
FSDP, DeepSpeed, or Megatron**, then implements weight resharding between them. Differences are in
placement strategy (Ray-native vs custom), supported algorithms, and scale ceiling.

**Choosing:**
- PyTorch + want it to "just work" on 1–8 GPUs, or any offline DPO-family → **TRL**.
- PyTorch + large multi-node on-policy PPO/GRPO, max performance → **veRL** (or **OpenRLHF** if you want
  Ray-native simplicity and DeepSpeed).
- Megatron/NeMo ecosystem, biggest models, NVIDIA stack → **NeMo-Aligner / NeMo-RL**.
- JAX / TPU → **MaxText-RL** (see [[maxtext-jax-llm]], [[ml-frameworks]]).
- Non-LLM RL (robotics, control, classic environments) → **RLlib**.

See [[serving-frameworks]] for the rollout backends and [[training-frameworks]] for the learner backends.

---

## 6. Running on Kubernetes

RLHF jobs are heterogeneous, multi-host, and gang-scheduled — a good fit for K8s with the right
substrate.

- **Ray / KubeRay is the common substrate.** OpenRLHF, veRL, and others use Ray to place actor/learner
  roles across pods. Run them as `RayCluster`/`RayJob` via KubeRay. Defer Ray cluster depth to
  **[[ray-on-kubernetes]]** — head/worker groups, autoscaling, placement groups, GCS fault tolerance.
- **Queueing and gang scheduling via [[kueue-advanced]].** An RLHF job needs *all* its
  rollout+learner pods at once (gang/all-or-nothing) or it deadlocks waiting on accelerators. Use Kueue
  `ClusterQueue`/`LocalQueue` with workload quotas and gang admission; pair with topology-aware
  scheduling so the learner's collective comms land on a tight network domain. Multi-host roles map
  cleanly to **[[jobset-leaderworkerset]]**.
- **Accelerator placement.** Pin rollout pods and learner pods to the right node pools (e.g. inference-
  optimized vs training-optimized GPUs/TPUs). Use node selectors/taints, and topology hints so a
  TP/PP group is co-located on high-bandwidth interconnect (NVLink domain / TPU slice). Disaggregated
  placement = two node pools; colocated = one pool, shared GPUs.
- **Multi-cluster patterns.** When rollout and learner live in different clusters/regions (capacity or
  cost), use MultiKueue (see [[kueue-advanced]]) for cross-cluster admission, and design the weight-sync
  transport for the higher cross-cluster latency — usually a reason to keep on-policy training in a
  single cluster and reserve multi-cluster for offline/data-generation stages.

See [[aiml-on-kubernetes]] for the umbrella view of ML workloads on K8s/GKE.

---

## 7. Performance and scale

- **Generation throughput is the bottleneck.** Profile the split first. If decode is >60% of step time
  (it usually is), invest there: more rollout replicas, a faster engine, bigger rollout batch,
  speculative/parallel decoding where supported.
- **KV cache for rollouts.** Rollouts are bursty and variable-length. Paged KV cache (vLLM/SGLang) and
  continuous batching keep the inference GPUs saturated. **Prefix caching** is a major GRPO win: a group
  of G samples shares the prompt prefix → compute it once. Size the KV cache against your max generation
  length and concurrency; oversubscription causes preemption thrash.
- **Memory budgeting across copies.** Sum params + optimizer + grads + activations + KV cache for every
  resident model (§2). Levers: LoRA/PEFT to shrink trained footprint; ZeRO-3 / FSDP full-shard to spread
  optimizer state; offload reference/reward to a separate node; quantize frozen models (reward/reference)
  to fp8/int8; drop to GRPO/RLOO to kill the critic; go offline (DPO) to drop rollout + RM entirely.
- **Overlap.** Pipeline generation of batch N+1 with the learner step on batch N (async/off-by-one
  on-policy) to hide the generation bottleneck — at the cost of slight staleness (mild off-policy-ness),
  which is usually fine and often a big throughput win.
- **Batch and group sizing.** Larger rollout batches and groups improve advantage estimates and
  throughput but cost memory and staleness. Tune empirically.

---

## 8. Reward hacking, instability, and eval

- **Reward hacking is the default outcome, not an edge case.** The policy optimizes the *proxy*, not the
  goal: it learns to be verbose, sycophantic, to spam formatting, to exploit RM blind spots, or to game a
  verifier (e.g. printing the expected answer without reasoning). Defenses: KL control; RM ensembles and
  score normalization; length/format penalties; on-policy RM data; and above all **held-out eval that is
  not the training reward**.
- **Instability symptoms:** reward collapses or explodes; KL runs away; entropy → 0 (mode collapse,
  repetitive output); critic value loss diverges (PPO); NaNs after a sync. Common causes: KL coef too
  low, learning rate too high, RM mis-scaled, a bad weight resharding (corrupted policy), or
  normalization on a degenerate group (GRPO).
- **Eval is the ground truth, not the reward curve.** Use held-out benchmarks, an independent LLM judge
  (different from the training judge to avoid shared bias), pairwise win-rate vs the SFT baseline, and
  capability regressions (does alignment tax core skills?). A rising reward with flat/falling eval = you
  are training a reward hacker.
- **Preference / dataset handling is the product.** Dedup; balance topics and chosen/rejected difficulty;
  **decontaminate against eval sets**; filter low-margin or noisy pairs; version and snapshot data with
  the run. For RLAIF, version the constitution/judge prompt and audit the judge for position/verbosity/
  self-preference bias and calibration.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at step 1 | Didn't budget all model copies + KV cache | Recount §2; drop critic (GRPO/RLOO), LoRA, ZeRO-3 offload, smaller rollout batch/KV |
| Reward up, quality down | Reward over-optimization / hacking | Raise KL coef; RM ensemble + normalize; length penalty; trust eval not reward |
| KL runs away, output degenerate | KL coef too low / sync bug | Raise/adapt KL coef; verify weight resharding with a fixed-prompt sanity check |
| Output collapses to repetition | Entropy collapse / mode collapse | Add entropy bonus; raise KL; lower LR; check reward scale |
| Throughput terrible | Generation-bound, under-provisioned rollout | Add rollout replicas; enable prefix caching/continuous batching; shorten max gen; overlap gen+train |
| Completions silently worsen after a step | Corrupted weight sync (layout/precision/name mismatch) | Checksum the transferred weights; validate inference engine reload; check TP/PP↔vLLM mapping |
| Job deadlocks pending pods | No gang scheduling — partial admission | Gang/all-or-nothing admission via Kueue; reserve quota; see [[kueue-advanced]] |
| GRPO loss NaN | Group std ≈ 0 → divide-by-zero in normalization | Add epsilon; clip; skip degenerate groups; increase group size |
| Critic value loss diverges (PPO) | Value LR/clip mis-set, reward scale off | Tune value loss coef/clip; normalize/whiten rewards; consider GRPO/RLOO instead |

---

## 10. Decision guides

### PPO vs DPO vs GRPO (and friends)
- **Start with DPO / ORPO / KTO** if you have or can cheaply build preference data and want a stable,
  cheap, easy-to-operate alignment pass. Off-policy, no rollout, ~SFT cost. ORPO if you want to fold it
  into SFT; KTO if you only have binary good/bad labels.
- **Use GRPO / RLOO** for online RL on **verifiable or reasoning** tasks (math, code, tool use) where you
  want exploration but not the cost/fragility of a critic. Lower memory than PPO, strong on sparse
  verifiable rewards. The current default for reasoning RL.
- **Use PPO** when you specifically need a **learned value function** / dense token-level advantages and
  want maximal control, and can afford four models and the tuning. Most teams now reach for GRPO first.
- **Use rejection sampling / best-of-n / RAFT** as the stable baseline before any policy-gradient method
  — often it's enough, and it's the easiest to scale.
- Online (PPO/GRPO/RLOO) can discover new behaviors but pays the full generate→sync→update cost every
  step; offline (DPO family) is far cheaper but bounded by its fixed dataset.

### Which framework for which scale/ecosystem
- **PyTorch, ≤8 GPUs / research / DPO family:** TRL.
- **PyTorch, large multi-node on-policy PPO/GRPO, max perf:** veRL (HybridFlow); OpenRLHF if you want
  Ray-native + DeepSpeed simplicity.
- **Megatron/NeMo / largest models / NVIDIA stack:** NeMo-Aligner / NeMo-RL.
- **JAX / TPU:** MaxText-RL ([[maxtext-jax-llm]]).
- **Non-LLM RL:** RLlib.
- **PyTorch vs JAX** is usually the first fork: it dictates the rollout (vLLM/SGLang vs JAX decode) and
  learner (FSDP/Megatron vs XLA) backends, and therefore the framework. See [[ml-frameworks]].

---

## 11. Canonical references

Verify against current versions; these move fast.

- **TRL** — https://github.com/huggingface/trl ; docs https://huggingface.co/docs/trl
- **veRL (HybridFlow)** — https://github.com/volcengine/verl ; docs https://verl.readthedocs.io
- **OpenRLHF** — https://github.com/OpenRLHF/OpenRLHF
- **NeMo-Aligner / NeMo-RL** — https://github.com/NVIDIA/NeMo-Aligner ; https://github.com/NVIDIA-NeMo/RL
- **Ray RLlib** — https://docs.ray.io/en/latest/rllib/index.html
- **vLLM** — https://docs.vllm.ai ; **SGLang** — https://github.com/sgl-project/sglang
- **InstructGPT / RLHF** — Ouyang et al. 2022, "Training language models to follow instructions with
  human feedback", https://arxiv.org/abs/2203.02155
- **PPO** — Schulman et al. 2017, https://arxiv.org/abs/1707.06347
- **DPO** — Rafailov et al. 2023, https://arxiv.org/abs/2305.18290
- **GRPO / DeepSeekMath** — Shao et al. 2024, https://arxiv.org/abs/2402.03300
- **RLOO** — Ahmadian et al. 2024, "Back to Basics", https://arxiv.org/abs/2402.14740
- **KTO** — Ethayarajh et al. 2024, https://arxiv.org/abs/2402.01306
- **ORPO** — Hong et al. 2024, https://arxiv.org/abs/2403.07691
- **RLAIF / Constitutional AI** — Bai et al. 2022, https://arxiv.org/abs/2212.08073 ; RLAIF, Lee et al.
  2023, https://arxiv.org/abs/2309.00267
- **Reward over-optimization** — Gao et al. 2022, https://arxiv.org/abs/2210.10760
- **HybridFlow paper** — Sheng et al. 2024, https://arxiv.org/abs/2409.19256
