---
name: training-frameworks
description: Expert distributed-training knowledge for training large models on hundreds-to-thousands of
  accelerators — the parallelism strategies and the frameworks that implement them. Use when choosing,
  configuring, or debugging multi-GPU/TPU training: data parallel (DDP), ZeRO/FSDP/FSDP2 sharding,
  tensor parallel (Megatron), pipeline parallel (GPipe/1F1B/interleaved, bubble), sequence/context
  parallel, expert parallel (MoE all-to-all), 3D/ND parallelism, activation checkpointing, gradient
  accumulation, bf16/fp8 mixed precision, NCCL/RCCL collectives & comm overlap, distributed/async
  checkpointing, elastic/fault-tolerant training, MFU/goodput. Covers PyTorch DDP/FSDP/TorchTitan/
  torchrun, DeepSpeed (ZeRO/Offload/Infinity/MoE), Megatron-LM/Megatron-Core, NVIDIA NeMo, HF
  Accelerate/Trainer/TRL, PyTorch Lightning, Ray Train, JAX MaxText/Levanter/Paxml/Pathways, and how
  these map onto Kubeflow Trainer/PyTorchJob/MPIJob/JobSet. Triggers: torchrun/deepspeed launch args,
  FSDP/ZeRO config, "model won't fit", low MFU, OOM at scale, checkpoint sharding, MoE training, 3D
  parallelism, picking tp/pp/dp degrees.
---

# Distributed Training Frameworks & Parallelism

Apply the judgment of an engineer who has trained frontier-scale dense and MoE models on thousands of
accelerators for years: pick the **fewest parallelism axes that make the model fit**, map the chattiest
comms onto the fastest interconnect, keep every device doing math (high MFU), and survive failures
without losing the run.

## How to use this skill

1. **Read `training-frameworks-guide.md`** in this directory — the full reference (parallelism
   strategies, framework selection, comms, checkpointing/fault tolerance, decision guide,
   anti-patterns, troubleshooting). Apply it to the task.
2. For concrete artifacts to imitate — an FSDP2 `torchrun` launch, a Megatron/DeepSpeed config sketch,
   and a `torchrun`→JobSet/Kueue mapping — read **`examples.md`**.
3. Match the surrounding repo's framework and conventions; apply the correctness rules
   (topology-aware mesh shape, comm overlap, sharded checkpoints, resumable data loader) regardless.
   Never fabricate a flag/config key — when unsure, verify against the pinned version's docs.

## Essentials (full detail in `training-frameworks-guide.md`)

- **Static state for Adam mixed precision ≈ 16–18 bytes/param** (bf16 params+grads + fp32 master +
  fp32 m,v). That plus activations is your memory budget. If it fits on one GPU → **DDP**; else shard.
- **Parallelism is memory-vs-comms.** DDP replicates everything (limit: model must fit one device).
  **ZeRO/FSDP** shard optimizer(1)→grads(2)→params(3=FSDP `FULL_SHARD`). **TP** splits inside a layer
  (Megatron). **PP** splits across layers (bubble ∝ `(p-1)/m`). **CP/SP** split the sequence. **EP**
  splits MoE experts (all-to-all). Compose into a **3D/ND device mesh**.
- **Topology is the design constraint:** TP/EP must stay inside NVLink/one node (chattiest collectives);
  PP/DP/FSDP can cross nodes; gang-schedule the whole job into one fabric domain.
- **FSDP2** (`fully_shard`, DTensor-based) is the current PyTorch path; use a **HYBRID/2-D mesh**
  multi-node so the param all-gather stays on NVLink. Wrap per transformer block, not whole-model.
- **Always-on memory tools:** selective activation checkpointing, gradient accumulation (skip the
  all-reduce on non-final micro-batches), bf16 (fp8 on the big GEMMs with careful scaling), fused/
  distributed optimizer (ZeRO-1).
- **Comm overlap is MFU.** Overlap grad all-reduce / FSDP all-gather+reduce-scatter / TP comms with
  compute; lost overlap, a CPU-bound data loader, or `m≈p` pipelines are the top MFU killers.
- **Collectives:** all-reduce (DDP/TP), reduce-scatter (FSDP grads), all-gather (FSDP params),
  all-to-all (MoE), P2P (PP). NCCL/RCCL on GPU; XLA collectives on TPU.
- **Checkpoint sharded + async** (PyTorch DCP / Megatron dist / DeepSpeed universal) — never gather to
  rank 0. Use elastic `torchrun`/torchft/Pathways + a **resumable data loader** to recover from node
  loss. Optimize **MFU**, then **goodput**.
- **Framework fit:** TorchTitan/FSDP2 for PyTorch-from-scratch; Megatron-Core/NeMo for frontier
  dense/MoE on NVIDIA; DeepSpeed for ZeRO-3 + offload; HF Accelerate/Trainer/TRL for fine-tune;
  JAX MaxText/Levanter/Paxml+Pathways on TPU; Ray Train to orchestrate. Don't add an axis you don't need.

## Related skills

- `[[ml-frameworks]]` — PyTorch/JAX/XLA compute internals, DTensor, `torch.compile`, GPU/TPU (sibling).
- `[[aiml-on-kubernetes]]` — umbrella for training/inference/RL on K8s & GKE.
- `[[jobset-leaderworkerset]]` — the multi-host gang primitive a `torchrun` rendezvous targets.
- `[[kueue-advanced]]` — gang scheduling, quota, and Topology-Aware Scheduling for training jobs.
- `[[slurm-hpc-on-kubernetes]]` — Slurm/MPI/RDMA, MPIJob, Volcano for HPC-style launches.
- `[[gke-master]]` — TPU/GPU node pools, networking, and placement on GKE.
- `[[serving-frameworks]]` — inference/serving (vLLM, SGLang, TensorRT-LLM), the other side of the stack.
