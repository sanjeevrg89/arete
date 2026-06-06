# AGENTS.md — AI/ML on Kubernetes & GKE

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`aiml-on-kubernetes-guide.md`** next to this file — read it
> before designing or debugging ML-on-K8s workloads. Annotated end-to-end manifests are in
> **`examples.md`**. This is the always-on summary.
>
> This is the **umbrella/strategy** skill for the ML lifecycle (training, serving, fine-tuning, RL/RLHF,
> agentic) on Kubernetes and GKE. Its job is to give the end-to-end picture and the K8s orchestration,
> and to **route to the right sibling skill** for depth — don't reimplement framework internals here.

## Apply by default on accelerator/ML workloads on K8s/GKE:

- **Utilization is the top-line metric**, not request latency. Accelerators are scarce and expensive;
  idle = waste. Track **MFU/goodput** (training) and TTFT/TPOT/KV-utilization/tokens-per-sec (serving).
- **Multi-host work is gang-scheduled or it deadlocks.** All N pods or none. Use **Kueue**
  ([[kueue-advanced]]) for queueing/quota/all-or-nothing admission and **JobSet** (training) /
  **LeaderWorkerSet** (inference) ([[jobset-leaderworkerset]]) for the multi-pod group with stable
  identity + headless DNS. Never model multi-host training as a bare Deployment.
- **Placement is correctness-adjacent.** Keep a multi-host slice **contiguous in one network domain**
  (Kueue Topology-Aware Scheduling). TP/EP stay inside the fast NVLink/ICI domain; DP/PP go across it.
- **Request accelerators correctly.** GPUs: `nvidia.com/gpu` (limits), GKE accelerator nodeSelector +
  GPU taint toleration; managed driver installer + device plugin. TPUs: `google.com/tpu` +
  `gke-tpu-accelerator` + `gke-tpu-topology`; per-pod chip count must match topology. Sharing: MIG
  (isolated), time-slicing (no memory isolation — risky), MPS; prefer **DRA** where GKE supports it.
- **Networking makes or breaks training.** GPUDirect-TCPX/TCPXO or RDMA/RoCE need multi-NIC pods
  (GKE multi-networking), the NCCL plugin/sidecar, and tuned `NCCL_*` env; often `hostNetwork: true` +
  `dnsPolicy: ClusterFirstWithHostNet`. Verify the current GKE recipe per machine family. TPUs use the
  ICI mesh (XLA-managed) — keep the slice whole.
- **Checkpoint for fault tolerance.** At 1k+ chips a host *will* fail; design so a failure costs minutes.
  Multi-tier (Local SSD → Parallelstore/Hyperdisk → GCS), async/sharded, restart-from-checkpoint via
  JobSet failure policy; consider elastic training and checkpoint-on-failure. Test restore.
- **Inference: single- vs multi-host.** Fits one node → Deployment + model-aware autoscaling
  ([[autoscaling-kubernetes]]). Doesn't fit → **LeaderWorkerSet** (leader+workers = one replica, scale by
  group). Autoscale on KV-cache utilization / queue depth / TTFT, **never CPU**. Cold start is dominated
  by weight-load time — stage weights on a fast volume / cache; plan node provisioning (DWS/reservations).
- **Storage tiers:** GCS (FUSE CSI, with caching) for data/artifacts; Parallelstore for high-BW shared
  FS / fast checkpoint; Hyperdisk/Local SSD for fast local. Never starve accelerators on the data loader.
- **Fine-tuning ask scales with method.** Full ≈ training (multi-host, FSDP/ZeRO). LoRA/QLoRA often
  single-node, sometimes a MIG slice; many adapters can multiplex on one served base.
- **RL/RLHF is heterogeneous + multi-component:** rollout (inference) + reward model + learner (training)
  + reference, different hardware, independent scaling, periodic **weight sync** to rollouts. Pushes to
  multiple node pools/clusters (Kueue MultiKueue) often on Ray. PPO (most components), GRPO (rollout-
  heavy), DPO (≈ a normal fine-tune). Frameworks: TRL, veRL, NeMo-RL, OpenRLHF.
- **Agentic serving:** long-lived stateful sessions (prefix-cache + session-affinity), separate the cheap
  CPU orchestration tier from the expensive GPU tier, and **sandbox tool execution** (gVisor/Kata,
  NetworkPolicy, Workload Identity, seccomp) as a multi-tenant security boundary.
- **Observability:** DCGM exporter (GPU: SM/HBM/NVLink/**XID errors**), TPU runtime metrics, GKE Managed
  Prometheus with namespace-level accelerator metrics. Alert on XID, NCCL timeouts, stalled all-reduce.
- **Multi-tenancy = quota + isolation:** Kueue ClusterQueues/cohorts for fair-share of scarce
  accelerators; namespaces + ResourceQuota + taints + NetworkPolicy + Workload Identity per tenant.

## Version awareness (2026)
Accelerator SKUs (H100/H200/B200/GB200; TPU v5e/v5p/v6e Trillium), GKE features (DRA, DWS, GPUDirect-
TCPXO/RDMA, Inference Gateway, custom compute classes), and framework versions change every quarter.
Name them, but **verify current region/SKU availability, quotas, and API stability in live GKE docs and
`gcloud`** before committing. Never fabricate benchmark numbers — measure on your hardware.

## Routing (this skill is the integrator)
`[[gke-master]]` node pools/networking · `[[kueue-advanced]]` gang/quota/TAS · `[[jobset-leaderworkerset]]`
multi-host groups · `[[ml-frameworks]]` PyTorch/JAX/XLA · `[[training-frameworks]]` FSDP/Megatron/etc ·
`[[serving-frameworks]]` vLLM/SGLang/etc · `[[autoscaling-kubernetes]]` autoscaling ·
`[[slurm-hpc-on-kubernetes]]` Slurm/HPC · `[[kubernetes-expert]]` core K8s.
See the decision table at the end of `aiml-on-kubernetes-guide.md`.
