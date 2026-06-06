---
name: aiml-on-kubernetes
description: Umbrella/strategy skill for running the full ML lifecycle on Kubernetes and GKE at frontier
  scale — training, inference/serving, fine-tuning, RL/RLHF, and agentic workloads. Use when the task
  involves GPUs (H100/H200/B200/GB200, device plugin, MIG, time-slicing, DRA) or TPUs (v5e/v5p/v6e
  Trillium, single/multi-host slices, topology) on K8s/GKE; gang/queued batch jobs; multi-host distributed
  training (FSDP/ZeRO, tensor/pipeline/sequence parallel); checkpointing to GCS/Parallelstore; multi-host
  vLLM/SGLang serving with LWS; GPUDirect-TCPX/RDMA/NCCL networking; accelerator observability (DCGM,
  Managed Prometheus), MFU/goodput, quota and multi-tenancy. Also covers AI FinOps — accelerator cost and
  capacity planning: MFU/goodput and $/token unit economics, right-sizing, Spot + checkpointing,
  reservations vs on-demand vs committed-use, scale-to-zero, GPU/TPU-hour estimation, and cost attribution/
  showback by team/namespace. Routes to deeper sibling skills for framework internals; owns the end-to-end
  picture and the K8s-specific orchestration.
---

# AI/ML on Kubernetes & GKE

Apply the judgment of a staff ML-infra engineer who runs frontier training and serving on Kubernetes and
GKE: you size accelerators, schedule gang jobs that either run whole or not at all, wire up RDMA/TCPX so
NCCL hits line rate, keep MFU and goodput high across thousands of chips, and serve trillion-parameter
models across hosts. This is the **map of the territory** — it gives the end-to-end lifecycle and the
K8s/GKE orchestration, and routes to the deeper sibling skills for framework internals.

## How to use this skill

1. **Read `aiml-on-kubernetes-guide.md`** in this directory — the full reference: accelerators on
   GKE, distributed training, inference/serving, fine-tuning, RL/RLHF, agentic, cross-cutting infra,
   and a decision guide. Apply it to the task.
2. For annotated end-to-end manifests (multi-host JobSet+Kueue training; multi-host vLLM/LWS serving),
   read **`examples.md`** and imitate them.
3. **Route to the right sibling** rather than going deep here: this skill is the integrator. Match the
   existing cluster's conventions (node-pool labels, taints, queue names); apply the correctness/safety
   and topology rules regardless.

## Essentials (full detail in `aiml-on-kubernetes-guide.md`)

- **Accelerators are scheduled like any resource, but placement is everything.** Request
  `nvidia.com/gpu` or `google.com/tpu`; the real work is node pools, drivers, taints/tolerations, and
  **topology-aware placement** so a multi-host slice lands on one contiguous network domain. On GKE,
  let the GPU/TPU device plugins and the driver installer DaemonSet do the low-level work; verify
  current SKU/region availability — hardware moves fast.
- **Multi-host = gang scheduling, full stop.** A job spanning N hosts must get all N pods or none, or
  you burn accelerators idling. Use [[kueue-advanced]] for queueing/quota/all-or-nothing admission and
  [[jobset-leaderworkerset]] (JobSet for training, LeaderWorkerSet for inference) to model the
  multi-pod group with stable identity and headless networking.
- **Networking makes or breaks training.** GPUDirect-TCPX/TCPXO or RDMA over Converged Ethernet, multi-NIC
  pods, and correct NCCL env are the difference between 30% and 90% of line rate. TPU multi-host slices
  use the ICI mesh — you don't manage it, but you must keep the slice contiguous (topology).
- **Checkpoint for fault tolerance, not just resumption.** At scale a host *will* fail mid-run; design
  for it. Asynchronous/multi-tier checkpointing to local disk → Hyperdisk/Parallelstore → GCS, plus
  elastic/restartable jobs. Goodput = useful compute / wall-clock; checkpoint overhead and restart cost
  are first-order.
- **Inference splits single-host vs multi-host.** If the model + KV cache fit one node, it's a Deployment
  with HPA-style autoscaling ([[autoscaling-kubernetes]]). If it doesn't, it's multi-host with
  LeaderWorkerSet, one logical replica = leader + workers sharing weights over the fabric.
- **Serving infra is shaped by KV cache and batching.** Continuous batching, prefix/KV-cache reuse, and
  prefill/decode disaggregation change your memory, autoscaling signal, and routing. Defer engine
  internals to [[serving-frameworks]]; own the GKE Inference Gateway / model-aware routing and load
  from GCS with fast cold-start.
- **Fine-tuning's infra ask depends on method.** Full fine-tune ≈ training (optimizer+grad sharding,
  multi-host). LoRA/QLoRA fits far smaller footprints — often single-node, sometimes a fraction of a
  GPU via MIG/time-slicing. Size the request to the method.
- **RL/RLHF is heterogeneous and multi-cluster by nature.** Rollout (inference) + reward model +
  learner (training) have different accelerator profiles and lifecycles; orchestrating their data flow
  and weight sync on K8s is the hard part. Name the frameworks (TRL, veRL, NeMo-RL, OpenRLHF); treat
  it as a systems problem.
- **Observability for accelerators is non-negotiable.** DCGM exporter for GPUs, TPU runtime metrics,
  GKE Managed Prometheus with namespace-level accelerator metrics. Watch MFU/goodput, SM/HBM occupancy,
  NCCL all-reduce time, XID/SXid errors, and host/NVLink failures.
- **Multi-tenancy is quota + isolation.** Kueue ClusterQueues/cohorts for fair-share of scarce
  accelerators; per-tenant namespaces, ResourceQuotas, and node taints. Reserved capacity (CUD/flex-start/
  DWS on GKE) vs on-demand changes how you queue.
- **AI FinOps: optimize price-per-result, not price-per-hour.** Accelerators are the dominant cost, so
  utilization *is* cost — the KPIs are MFU/goodput and $/training-run (training) and $/token &
  tokens/$/accelerator-hour at an SLO (serving). Levers: right-size SKUs, Spot/preemptible + checkpointing
  ([[ml-checkpointing-orbax]]), bin-pack (MIG/time-slicing), scale-to-zero ([[autoscaling-kubernetes]]),
  quota/fair-share ([[kueue-advanced]]), and the right reservation/on-demand/Spot/committed-use blend.
- **Plan capacity before you commit budget.** Estimate training accelerator-hours from
  ≈6·params·tokens / (peak-FLOPs·MFU); size inference from peak-QPS·tokens / measured tokens-sec-per-
  accelerator-at-SLO, plus headroom. Attribute cost by team via namespace-scoped accelerator metrics
  (`k8s_container`/`namespace_name`) → showback/chargeback and unit economics. See guide §10.

## Related skills

- `[[gke-master]]` — GKE Standard/Autopilot, TPU/GPU node pools, NAP, networking, security, DWS.
- `[[kueue-advanced]]` — batch queueing, quota/cohorts, gang admission, MultiKueue, Topology-Aware Scheduling.
- `[[jobset-leaderworkerset]]` — JobSet (multi-host training) and LeaderWorkerSet (multi-host inference).
- `[[ml-frameworks]]` — PyTorch, JAX, XLA, GPU/TPU programming internals.
- `[[training-frameworks]]` — DDP/FSDP, DeepSpeed/ZeRO, Megatron, NeMo, Ray Train, Kubeflow Trainer, MaxText.
- `[[serving-frameworks]]` — vLLM, SGLang, Dynamo, Triton, TensorRT-LLM, Ray Serve, KServe.
- `[[autoscaling-kubernetes]]` — HPA/VPA/Cluster Autoscaler/Karpenter/KEDA/NAP for model servers.
- `[[slurm-hpc-on-kubernetes]]` — Slurm/Slinky/Volcano/MPI/RDMA and the Slurm-vs-K8s decision.
- `[[kubernetes-expert]]` — general end-to-end Kubernetes practitioner mastery.
