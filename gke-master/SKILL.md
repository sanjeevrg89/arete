---
name: gke-master
description: Google Kubernetes Engine (GKE) specialist knowledge for designing and operating large
  ML/platform clusters on Google Cloud. Use whenever the work is GKE-specific — Standard vs Autopilot,
  node pools & machine families, Node Auto-Provisioning, GPU node pools (A3/A3-Mega/A3-Ultra/A4 with
  H100/H200/B200/GB200; time-sharing/MPS/MIG/DRA), TPU node pools (v5e/v5p/v6e Trillium, single/multi-host
  slices), Dataplane V2, VPC-native/NEGs/Gateway API, multi-networking & GPUDirect-TCPX/TCPXO/RDMA,
  Hyperdisk ML/Filestore/GCS FUSE CSI/Parallelstore, fast model loading & model streaming (run:ai Model
  Streamer-style streaming, FUSE caching, image streaming), Multi-Tier Checkpointing (MTC) for fast
  training restart, GKE Sandbox/gVisor (Agent Sandbox) for untrusted code & AI-agent tool execution,
  Workload Identity Federation, release channels & upgrades, Fleets/Config Sync, Backup for GKE, Managed
  Service for Prometheus, and GPU/TPU accelerator metrics. For generic Kubernetes use
  `[[kubernetes-expert]]`; this skill is the GKE layer on top.
---

# GKE Master

Apply the judgment of a GKE specialist / Google Cloud architect who has designed and run large
multi-tenant ML platforms (thousands of GPUs/TPUs) on GKE in production for years. Generic Kubernetes
lives in `[[kubernetes-expert]]`; **this skill is only the GKE-specific layer** — modes, node pools,
accelerators, GCP networking/storage/identity, and Google's managed operations.

## How to use this skill

1. **Read `gke-master-guide.md`** in this directory — the full reference. Apply it to the GKE task.
   For concrete, annotated artifacts to imitate (a TPU/GPU `gcloud … node-pools create`, Workload
   Identity Federation setup, a GCS FUSE CSI volume on a Pod), read **`examples.md`**.
2. Match the existing cluster's conventions (Standard vs Autopilot, release channel, Terraform/Config
   Connector vs `gcloud`); apply the correctness/security rules regardless.
3. **GCP moves fast and SKUs/regions/quotas change constantly.** Name machine types, accelerators, and
   flags as starting points but tell the reader to verify against current GKE docs and *their* project's
   regional availability and quota. Never fabricate a machine-type name, flag, or quota number.

## Essentials (full detail in `gke-master-guide.md`)

- **Pick the mode deliberately.** Autopilot = Google manages nodes, you pay per Pod resource request,
  hardened defaults — great for general workloads and increasingly for GPUs. Standard = you own node
  pools, full control of machine types/system config/local SSD/host networking — required for the
  gnarliest ML (custom GPUDirect, specific local-SSD layouts, multi-host TPU topologies). They are
  converging; choose by *how much node control you need*, not habit.
- **Workload Identity Federation for GKE is the only correct way to reach GCP APIs from Pods.** Never
  mount SA JSON keys. Bind a Kubernetes SA to an IAM principal; least-privilege per namespace/workload.
- **Accelerators are node-pool-shaped.** GPU types attach to specific machine families (e.g. H100 on
  A3, H200/B200/GB200 on A3-Ultra/A4 — verify); TPUs are their own machine types where the node pool
  *is* the slice, with single-host vs multi-host topologies. Multi-host slices need all-or-nothing
  placement (compact placement / TPU topology) and pair with `[[jobset-leaderworkerset]]`/`[[kueue-advanced]]`.
- **Right-size GPU sharing:** time-sharing (oversubscribe, no isolation), MPS (concurrent, weak
  isolation), MIG (hardware partitions on A100/H100), and DRA (the emerging GKE-native way to request
  fractional/partitioned accelerators). Don't share GPUs for latency-critical training.
- **VPC-native + Dataplane V2 are the modern defaults.** Alias-IP Pods, eBPF/Cilium dataplane with
  network policy and FQDN/observability. Use container-native load balancing (**NEGs**) so the LB
  targets Pods directly; Gateway API is the strategic ingress, classic Ingress is legacy.
- **High-perf ML networking = multi-networking + GPUDirect (TCPX on A3, TCPXO on A3-Mega, RDMA/RoCE on
  A3-Ultra/A4).** This needs multiple vNICs, the right node image/drivers, and an NCCL plugin —
  treat it as a node-pool design decision, not an afterthought.
- **Storage by access pattern:** Hyperdisk (Balanced/Throughput/Extreme; **Hyperdisk ML** for
  read-heavy, many-reader model loading), Filestore (shared NFS), **GCS FUSE CSI** (datasets/checkpoints
  in object storage), **Parallelstore CSI** (high-throughput scratch), local SSD (ephemeral
  scratch/checkpoints). Match throughput/IOPS to the workload; don't run training data off slow PD.
- **Cut serving cold-start with fast model loading:** stream weights from GCS (run:ai Model
  Streamer-style streaming), GCS FUSE with local-SSD caching, Hyperdisk ML for many-reader weight
  loading, and image streaming for multi-GB images. Cut training restart with **Multi-Tier Checkpointing
  (MTC)** — node-local SSD + GCS + peer recovery. Both are newer/fast-moving — verify current GKE
  availability; see `[[serving-frameworks]]`, `[[ml-checkpointing-orbax]]`.
- **Autoscaling is layered:** HPA/VPA (Pods) sit above Cluster Autoscaler and **Node Auto-Provisioning**
  (NAP creates right-shaped pools on demand, incl. GPU/TPU). For batch/gang ML, drive capacity with
  **Kueue + ProvisioningRequest** so jobs scale up atomically — see `[[autoscaling-kubernetes]]`,
  `[[kueue-advanced]]`.
- **Operate on release channels.** Subscribe (Rapid/Regular/Stable/Extended), set maintenance
  windows/exclusions, choose surge vs blue-green node upgrades, and never let nodes drift. Use Fleets +
  Config Sync/Config Connector for multi-cluster GitOps; Backup for GKE for state.
- **Observability is GCP-native:** Cloud Logging/Monitoring + **Google Cloud Managed Service for
  Prometheus** (PodMonitoring/ClusterPodMonitoring). For accelerators, enable DCGM (GPU) and TPU
  metrics; the `k8s_container` resource carries `namespace_name` for per-team attribution. Watch GPU
  `duty_cycle` vs TPU `tensorcore_utilization` — different metrics, same intent.
- **Harden by default:** Shielded GKE nodes, private clusters, Binary Authorization, **GKE Sandbox
  (gVisor)** — user-space kernel isolation via RuntimeClass for untrusted/multi-tenant code and **AI-agent
  tool execution** (verify GPU/feature compatibility) — Confidential GKE where required, Security Posture
  dashboard, and tight RBAC↔IAM mapping. Secrets via Secret Manager CSI, not baked images. See
  `[[ai-security-on-gke]]`, `[[llm-app-agent-frameworks]]`.
- **Cost:** Spot node pools for fault-tolerant ML, NAP to avoid idle capacity, Autopilot to stop paying
  for unschedulable headroom, committed-use discounts, and right-sized Hyperdisk over Extreme-everywhere.

## Related skills

- `[[kubernetes-expert]]` — generic Kubernetes practitioner mastery; the base layer under this skill.
- `[[autoscaling-kubernetes]]` — HPA/VPA/Cluster Autoscaler/Karpenter/NAP/KEDA mechanics in depth.
- `[[kueue-advanced]]` — Kueue quota/gang/ProvisioningRequest/MultiKueue/TAS for batch ML on GKE.
- `[[jobset-leaderworkerset]]` — multi-host TPU/GPU training & inference topologies on GKE.
- `[[aiml-on-kubernetes]]` — the umbrella ML-on-K8s/GKE skill (training/inference/RL stacks).
- `[[serving-frameworks]]` / `[[ml-frameworks]]` — vLLM/SGLang/Triton serving; PyTorch/JAX/XLA on GPU/TPU.
- `[[ml-checkpointing-orbax]]` — checkpoint sharding/async/format depth beneath Multi-Tier Checkpointing.
- `[[ai-security-on-gke]]` / `[[llm-app-agent-frameworks]]` — agent isolation threat model & app stack
  for GKE Sandbox / Agent Sandbox tool execution.
