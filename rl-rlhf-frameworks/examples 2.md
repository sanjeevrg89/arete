# RL / RLHF Worked Examples

Canonical, shape-correct artifacts to imitate. **None of these flag names/fields are guaranteed
verbatim** — frameworks rename and reorganize config keys frequently. Use them for *structure and
intent*, then **verify every key against the current docs** of the version you run. Never copy a flag you
haven't confirmed exists.

---

## 1. GRPO training-config sketch (TRL-style, PyTorch)

Online GRPO: critic-free, group-sampled, KL-anchored to the reference. The rollout backend is vLLM; the
learner is FSDP/Accelerate. Shape only — confirm key names in the current `trl` `GRPOConfig`.

```python
# pip install trl vllm  (verify versions)
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

dataset = load_dataset("your-org/reasoning-prompts", split="train")  # prompts only; rewards are computed online

def reward_fn(completions, **kwargs):
    # Verifiable reward: e.g. exact-match on a math answer, or unit-test pass rate.
    # Return one float per completion. Keep it cheap and deterministic.
    return [1.0 if is_correct(c) else 0.0 for c in completions]

cfg = GRPOConfig(
    output_dir="grpo-run",
    # --- generation (rollout) ---
    use_vllm=True,                 # route rollouts to vLLM
    num_generations=8,             # group size G — the GRPO baseline comes from these
    max_prompt_length=1024,
    max_completion_length=1024,    # shorter = faster; decode dominates wall-clock
    temperature=1.0,
    # --- optimization (learner) ---
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,            # RL LRs are small; large LR => instability
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,   # trade compute for activation memory
    # --- KL control (the safety rail) ---
    beta=0.04,                     # KL coefficient to the reference model; watch KL as a metric
    # --- logging/eval ---
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,                # eval on a HELD-OUT set, not the training reward
)

trainer = GRPOTrainer(
    model="your-org/sft-checkpoint",   # policy, init from SFT
    reward_funcs=reward_fn,            # or a reward-model id for a learned RM
    args=cfg,
    train_dataset=dataset,
    # reference model is created from the SFT checkpoint automatically for the KL term
)
trainer.train()
```

Notes that generalize to any framework:
- `num_generations` (group size) trades advantage quality + throughput vs memory. Enable vLLM **prefix
  caching** so the shared prompt prefix in a group is computed once.
- Keep the LR small and **watch KL**. A reward spike with a KL spike is reward hacking.
- Eval (`eval_steps`) must use a held-out signal independent of the training reward.

### PPO contrast (what changes)
PPO adds a **value/critic model** (a second trained model → ~2× trained footprint) and uses GAE for
token-level advantages. In `trl` that's `PPOConfig`/`PPOTrainer` taking an explicit `value_model` and a
`reward_model`, with `cliprange`, `cliprange_value`, `gamma`, `lam` (GAE λ), and `kl_coef`. Reach for PPO
only when you genuinely need a learned value function; otherwise GRPO/RLOO save the critic.

### veRL-style config (large multi-node, shape only)
veRL drives the same loop from a hierarchical (Hydra/YAML) config, e.g.:

```yaml
# verl GRPO/PPO sketch — verify keys against current verl docs
algorithm:
  adv_estimator: grpo            # or gae for PPO
  kl_ctrl: { kl_coef: 0.001 }
actor_rollout_ref:
  model: { path: your-org/sft-checkpoint }
  actor:                          # the trained policy (learner)
    strategy: fsdp                # or megatron
    optim: { lr: 1e-6 }
    ppo_mini_batch_size: 256
  rollout:                        # the generation engine
    name: vllm                    # or sglang
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.5   # leave HBM for the colocated learner
    n: 8                          # group size
  ref: { fsdp_config: {} }        # frozen reference for KL
reward_model:
  enable: true
  model: { path: your-org/reward-model }
trainer:
  n_gpus_per_node: 8
  nnodes: 4
  placement: colocate             # actor+rollout share GPUs; vs disaggregated pools
```

---

## 2. Actor / learner placement sketch on Ray

The systems crux of online RLHF: split **rollout (inference)** from **learner (training)** and move
weights between them. This sketch shows the *placement* contract (Ray actors + a placement group), not a
full trainer. Run the `RayCluster` on K8s via KubeRay — defer cluster details to **[[ray-on-kubernetes]]**
and gang-admission/quota to **[[kueue-advanced]]**.

```python
import ray
from ray.util.placement_group import placement_group

ray.init()

# Disaggregated: a pool of rollout (vLLM) actors and a pool of learner (FSDP) actors.
# Colocated alternative: one bundle set with STRICT_PACK so actor+learner share GPUs.
NUM_ROLLOUT, NUM_LEARNER = 4, 4
pg = placement_group(
    bundles=[{"GPU": 1, "CPU": 8} for _ in range(NUM_ROLLOUT + NUM_LEARNER)],
    strategy="PACK",  # PACK/SPREAD for disaggregated; STRICT_PACK to colocate roles on a node
)
ray.get(pg.ready())

@ray.remote(num_gpus=1)
class RolloutWorker:
    def __init__(self, model_path):
        from vllm import LLM
        self.llm = LLM(model=model_path, enforce_eager=False)  # paged KV cache, continuous batching
    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params)
    def update_weights(self, weight_handle):
        # CRITICAL: reshard learner weights into vLLM's layout and load in-place.
        # Validate with a fixed-prompt sanity generation after loading.
        self.llm.llm_engine.model_executor.load_weights(weight_handle)

@ray.remote(num_gpus=1)
class LearnerWorker:
    def __init__(self, model_path):
        ...  # FSDP-wrapped policy (+ critic for PPO), reference, optimizer
    def train_step(self, rollouts_with_rewards):
        ...  # advantages -> policy-gradient/preference loss -> optimizer.step()
        return self.export_weights_for_inference()  # gather/reshard to the rollout layout

rollouts = [RolloutWorker.options(placement_group=pg).remote("sft-ckpt") for _ in range(NUM_ROLLOUT)]
learners = [LearnerWorker.options(placement_group=pg).remote("sft-ckpt") for _ in range(NUM_LEARNER)]

# Outer loop: generate -> score -> update -> SYNC WEIGHTS back to rollout workers.
for step in range(num_steps):
    batches = ray.get([r.generate.remote(p, sp) for r, p in zip(rollouts, prompt_shards)])
    scored  = score_with_reward_model(batches)              # learned RM, verifier, or LLM judge
    weights = ray.get([l.train_step.remote(s) for l, s in zip(learners, scored)])[0]
    ray.get([r.update_weights.remote(weights) for r in rollouts])  # the resharding hot path
```

Placement decisions encoded above:
- **Colocated** → `STRICT_PACK` so a rollout and a learner share a node's GPUs (time-sliced); cheap
  weight sync (same host, CUDA-IPC), but KV cache and optimizer state contend for HBM — hence
  `gpu_memory_utilization` < 1.0 on the rollout side.
- **Disaggregated** → separate node pools (two Kueue `ResourceFlavor`s / node selectors); scale rollout
  replicas independently to fix the generation bottleneck; weights cross the network on sync.
- Gang-schedule the whole set (all rollout + learner pods admit together) or it deadlocks — see
  [[kueue-advanced]] and [[jobset-leaderworkerset]].

---

## 3. Reward-model note

The reward model is the proxy objective the policy optimizes; its quality **caps** the entire pipeline.

```python
# Reward model: SFT-init backbone + a scalar value head, trained on PREFERENCE PAIRS.
# Bradley-Terry loss:  -log σ( r(chosen) - r(rejected) )
import torch, torch.nn.functional as F

def bradley_terry_loss(reward_chosen, reward_rejected):
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()

# Dataset: {"prompt", "chosen", "rejected"} pairs (human-labeled, or AI-labeled for RLAIF/CAI).
# In TRL this is RewardConfig / RewardTrainer; shape only — verify current keys.
```

Practical rules:
- **Init the RM from the SFT checkpoint** (or a close model); add a scalar head. Train on clean,
  deduplicated, **decontaminated** preference pairs.
- **Normalize/whiten reward scores** before they enter the policy update; an un-scaled RM destabilizes
  PPO/GRPO. Consider an **RM ensemble** to reduce over-optimization (Goodhart).
- **On-policy RM data helps:** an RM trained only on off-policy data drifts as the policy moves; refresh
  with completions from the evolving policy when you can.
- **For RLAIF / Constitutional AI**, the labels come from an LLM judge guided by a constitution/principle
  set. The judge prompt *is* the reward spec — version it, and audit for position bias, verbosity bias,
  and self-preference. The judge for *eval* should differ from the judge used for *training* to avoid
  shared bias.
- **The RM is not your eval.** A rising RM score with flat/falling held-out eval means you are training a
  reward hacker. Trust the held-out, independent eval.

For the rollout backends (vLLM/SGLang) see [[serving-frameworks]]; for the learner backends
(FSDP/DeepSpeed/Megatron) see [[training-frameworks]]; for PyTorch-vs-JAX/TPU foundations see
[[ml-frameworks]] and [[maxtext-jax-llm]].
