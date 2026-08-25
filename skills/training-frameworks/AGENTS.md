# AGENTS.md — Distributed Training Frameworks & Parallelism

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`training-frameworks-guide.md`** next to this file —
> read it before configuring or debugging distributed training, and apply it. Concrete artifacts to
> imitate (FSDP2 `torchrun` launch, Megatron/DeepSpeed config sketch, `torchrun`→JobSet/Kueue mapping)
> are in **`examples.md`**. This file is the always-on summary.
>
> Scope: parallelism strategies + the training stacks (PyTorch FSDP/TorchTitan, DeepSpeed,
> Megatron-Core/NeMo, JAX MaxText, Ray Train, Kubeflow Trainer). Compute internals → `[[ml-frameworks]]`;
> K8s orchestration → `[[aiml-on-kubernetes]]`/`[[jobset-leaderworkerset]]`/`[[kueue-advanced]]`/`[[gke-master]]`.

## When working on multi-accelerator training, apply by default:

- **Budget first.** Adam mixed-precision static state ≈ **16–18 bytes/param** (bf16 params+grads, fp32
  master, fp32 m,v) + activations. Fits on one GPU → **DDP**; otherwise shard. Don't add a parallelism
  axis you don't need — each adds comms and a way to mis-shape the mesh.
- **Know the axes:** DDP (replicate, all-reduce grads) · ZeRO/FSDP (shard opt→grad→param) · TP
  (intra-layer, Megatron) · PP (inter-layer, bubble ∝ `(p-1)/m`, use 1F1B/interleaved) · CP/SP
  (sequence) · EP (MoE experts, all-to-all). Compose into an **N-D `DeviceMesh`**.
- **Topology is the constraint:** **TP/EP inside NVLink/one node** (chattiest collectives); **PP/DP/FSDP
  can cross nodes**; gang-schedule the whole job into one fabric domain (Kueue TAS). `tp×pp×cp` must
  divide your per-node/per-island GPU counts.
- **FSDP2** (`fully_shard`, DTensor) is the current PyTorch path; FSDP1 (`FullyShardedDataParallel`) is
  legacy. Use a **HYBRID/2-D mesh** multi-node (all-gather on NVLink, smaller reduce across fabric).
  Wrap **per transformer block**, never whole-model and never every tiny submodule.
- **Memory tools (orthogonal, always available):** selective activation checkpointing; gradient
  accumulation (skip the grad reduce on non-final micro-batches via `no_sync`); bf16 default (fp8 on
  big GEMMs with careful amax/scaling); fused/distributed (ZeRO-1) optimizer. Keep the **fp32 master copy**.
- **Comm overlap = MFU.** Overlap all-reduce / all-gather+reduce-scatter / TP comms with compute
  (FSDP prefetch). Top MFU killers: lost overlap, CPU-bound data loader, `m≈p` pipelines. Profile the
  timeline (Nsight / PyTorch profiler / XLA trace) and look for non-overlapped collectives.
- **Effective global batch = micro-batch × grad-accum × DP degree.** Changing DP changes LR/schedule
  and convergence — re-tune.
- **Checkpoint sharded + async**, never gather to rank 0: PyTorch **DCP** (`torch.distributed.checkpoint`),
  Megatron distributed, DeepSpeed universal. Save cadence balances cost vs work-lost-on-failure.
- **Fault tolerance:** elastic `torchrun` (`--nnodes=min:max`, `--max-restarts`) + torchft / Pathways
  on TPU, plus a **checkpointable/deterministic data loader** so resume doesn't re-see or skip data.
  Optimize **MFU** (FLOP/s ÷ peak), then **goodput** (useful ÷ wall). Only quote MFU you measured.
- **Pick the framework:** TorchTitan/FSDP2 (PyTorch from scratch, `torch.compile`) · Megatron-Core/NeMo
  (frontier dense/MoE on NVIDIA, max MFU) · DeepSpeed (ZeRO-3 + CPU/NVMe offload, minimal code change)
  · HF Accelerate/Trainer/TRL (fine-tune/post-train) · JAX MaxText/Levanter/Paxml+Pathways (TPU/SPMD)
  · Ray Train (orchestrate any of them). On K8s: Kubeflow Trainer v2 / JobSet + Kueue around the launcher.

## Never
- Run **TP or EP across the slow inter-node fabric** (stalls on per-layer all-reduce/all-to-all).
- **Gather-to-rank-0** checkpoints at scale (OOMs the coordinator, serializes I/O).
- Use fp8/fp16 **without proper scaling** or **without the fp32 master copy** (silent divergence/NaN).
- Resume from a checkpoint with a **freshly seeded data loader** (re-sees/skips data → quality regress).
- **Fabricate a flag/config key.** When unsure, describe the concept and say verify against the version.

## Definition of done for a training-config change
Mesh shape divides the topology (TP/EP on-node); comm overlap intact in a profile; effective batch
and LR consistent; sharded+async checkpointing and a resumable loader configured; elastic/restart path
present; MFU measured (not assumed). Report honestly if any is unverified.
