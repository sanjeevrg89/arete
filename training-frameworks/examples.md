# Distributed Training — Worked Examples

Correct-*in-shape* artifacts to imitate. Exact flag/config-key names drift between releases — verify
against the version you pin (see `training-frameworks-guide.md` §13). The structure, not any specific
benchmark number, is the thing to copy.

---

## 1. FSDP2 PyTorch training launch (single-node, 8 GPUs)

### 1a. `torchrun` invocation

```bash
# One process per GPU on one node. torchrun sets RANK / LOCAL_RANK / WORLD_SIZE for you.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  train.py --model-config configs/llama_8b.yaml
```

Multi-node (2 nodes × 8 GPUs = 16 ranks), elastic so the job survives a node loss:

```bash
# Run on every node; rdzv endpoint = a stable address all nodes can reach (e.g. the leader pod's DNS).
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --max-restarts=3 \
  --rdzv-backend=c10d \
  --rdzv-id=llama8b-run \
  --rdzv-endpoint="$LEADER_ADDR:29500" \
  train.py --model-config configs/llama_8b.yaml
# Elastic: --nnodes=min:max (e.g. 1:2) lets the job continue/rescale instead of dying on node loss.
```

### 1b. FSDP2 sharding setup in `train.py` (shape-correct skeleton)

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 2-D mesh = HYBRID shard: shard within a node, replicate across nodes -> param all-gather stays on
# NVLink, only a smaller reduce crosses the slow fabric. For a pure 1-D full-shard use one axis.
mesh = init_device_mesh("cuda", (NUM_NODES, GPUS_PER_NODE), mesh_dim_names=("replicate", "shard"))

model = build_model(cfg).to("cuda")

# Keep compute in bf16; keep an fp32 master copy + fp32 reduction for stability.
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

# Wrap PER TRANSFORMER BLOCK first (each becomes an all-gather/reduce-scatter unit), then the root.
# Whole-model-as-one-unit = no memory savings; every tiny module = comm overhead.
for block in model.transformer_blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)

# Selective activation checkpointing on the blocks (recompute cheap ops, keep flash-attn output).
# from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

for step, batch in enumerate(loader):           # loader must be checkpointable/deterministic
    for i, micro in enumerate(split_microbatches(batch, cfg.grad_accum)):
        is_last = (i == cfg.grad_accum - 1)
        model.set_requires_gradient_sync(is_last)   # skip grad reduce-scatter until the last micro-batch
        loss = model(micro).loss / cfg.grad_accum
        loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)

    if step % cfg.ckpt_every == 0:               # SHARDED checkpoint via DCP — never gather to rank 0
        sd_model, sd_opt = get_state_dict(model, opt)
        dcp.save({"model": sd_model, "optim": sd_opt, "step": step}, checkpoint_id=ckpt_path(step))
        # dcp.async_save(...) to overlap the flush with the next steps.
```

Notes: `fully_shard`/`MixedPrecisionPolicy`/`set_requires_gradient_sync` are FSDP2 API — confirm exact
signatures against your PyTorch version. Effective global batch = micro-batch × `grad_accum` ×
DP-degree; re-tune LR when you change DP. To add **tensor parallel**, build a `(dp, tp)` mesh and
`parallelize_module(block, mesh["tp"], plan)` *before* `fully_shard` on the `dp` axis.

---

## 2. Megatron-Core / Megatron-LM 3D-parallel launch (sketch)

Megatron is launched via `torchrun` with parallelism degrees passed as flags. Shape only — **verify
every flag against your Megatron-LM/Megatron-Core release**, they change.

```bash
# Example: 16 nodes × 8 = 128 GPUs.  TP=8 (inside one node, on NVLink) · PP=4 (across nodes) ·
# DP = 128 / (TP*PP) = 4.  SP on.  Optional CP for long context, EP for MoE.
torchrun --nnodes=16 --nproc-per-node=8 \
         --rdzv-backend=c10d --rdzv-endpoint="$LEADER_ADDR:29500" \
  pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --sequence-parallel \
  --num-layers-per-virtual-pipeline-stage 2 \   # interleaved 1F1B -> smaller bubble
  --context-parallel-size 1 \                   # raise for very long context
  --use-distributed-optimizer \                 # ZeRO-1 optimizer-state sharding over the DP group
  --recompute-granularity selective \           # selective activation recomputation
  --bf16 \
  --micro-batch-size 1 \
  --global-batch-size 512 \                     # = micro-batch × grad-accum × DP, set by Megatron
  --num-layers 80 --hidden-size 8192 --num-attention-heads 64 --seq-length 8192 \
  --lr 1.5e-4 --min-lr 1.5e-5 --lr-warmup-iters 2000 --clip-grad 1.0 \
  --save /ckpts/run1 --load /ckpts/run1 --ckpt-format torch_dist   # sharded/distributed checkpoint
# MoE adds (verify): --num-experts N --expert-model-parallel-size E --moe-router-topk K
# fp8 via Transformer Engine adds (verify): --fp8-format hybrid (with the right recipe flags)
```

Topology mapping that matters: **TP=8 lands within a node** (NVLink); **PP=4 crosses nodes** (P2P,
bubble-tolerant); **DP=4 is outermost** (one all-reduce/step). `world_size = TP·PP·DP·CP·EP`.

---

## 3. DeepSpeed ZeRO-3 config sketch (`ds_config.json`)

Launched with `deepspeed train.py --deepspeed --deepspeed_config ds_config.json` (or via HF Trainer /
Accelerate which generate/consume this). Shape only — verify keys against your DeepSpeed version.

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param":     { "device": "none" }
  },
  "gradient_clipping": 1.0,
  "zero_allow_untested_optimizer": false
}
```

- `stage: 3` = shard params + grads + optimizer (FSDP-equivalent). `stage: 2` = grads + optimizer;
  `stage: 1` = optimizer only.
- `offload_optimizer.device: "cpu"` = **ZeRO-Offload**; set `offload_param.device: "nvme"` (+ an
  `nvme_path`) for **ZeRO-Infinity** when you're memory-bound and can trade throughput.
- For comm reduction on bandwidth-constrained clusters, ZeRO++ adds quantized-weight / hierarchical-
  partition / quantized-gradient options — verify the exact keys for your version.

---

## 4. `torchrun` → JobSet / Kueue mapping (K8s)

How the launcher above lands on a cluster. Orchestration depth is in `[[jobset-leaderworkerset]]`,
`[[kueue-advanced]]`, `[[aiml-on-kubernetes]]`; this is just the mapping.

```
Process model                         Kubernetes object
------------------------------------  ----------------------------------------------------
1 process per accelerator             1 container process; nproc-per-node = GPUs per pod
1 node = 1 pod (LOCAL_WORLD_SIZE)     1 Pod per node, requesting all node GPUs + RDMA NIC
N nodes = 1 gang                       1 JobSet (or Kubeflow Trainer v2 TrainJob, built on JobSet)
rendezvous endpoint                    leader Pod's stable headless-Service DNS : 29500
gang scheduling (all-or-nothing)       Kueue admission (whole gang or none — no half-start deadlock)
TP/EP on fast links                    Kueue Topology-Aware Scheduling -> one fabric/NVLink domain
survive node loss                      JobSet restart policy + torchrun --max-restarts + DCP + Kueue
```

Each pod runs the same `torchrun --nnodes=$N --nproc-per-node=$G --rdzv-endpoint=$LEADER:29500 ...`;
`$N`, `$LEADER`, and rank/replica identity come from the JobSet/LeaderWorkerSet env. Put TP/EP within
a pod (one node), PP/DP across pods. Kubeflow Trainer v2's `TrainJob`/`TrainingRuntime` builds this
JobSet for you; the v1 `PyTorchJob` (Training Operator) wires the master+worker rendezvous env directly.
Always gang-schedule and place into one topology domain — a multi-node job that half-starts deadlocks,
and a TP group split across nodes stalls on comms.
