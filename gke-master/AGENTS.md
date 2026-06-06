# AGENTS.md — GKE (Google Kubernetes Engine) Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`gke-master-guide.md`** next to this file — read it before
> designing or operating on GKE, and apply it. Annotated worked examples (TPU/GPU node pool, Workload
> Identity Federation, GCS FUSE CSI volume) are in **`examples.md`**. This file is the always-on summary.
>
> Generic Kubernetes lives in `[[kubernetes-expert]]` / `[[autoscaling-kubernetes]]`. **This skill is
> only the GKE-specific layer.** GCP SKUs/regions/quotas change constantly — name them but tell the user
> to verify current GKE docs and their project's regional availability/quota. Never fabricate a
> machine-type name, flag, or quota number.

## When the task is GKE-specific, apply these by default:

- **Choose mode deliberately.** Autopilot = Google-managed nodes, per-Pod-request billing, hardened
  defaults; Standard = you own node pools and get full machine/system-config/local-SSD/host-network
  control (needed for custom GPUDirect, precise local-SSD layouts, multi-host TPU topologies). They are
  converging — pick by node-control needs, not habit. Prefer **Custom Compute Classes** over hand-pinned
  machine types for "give me this accelerator, fall back gracefully."
- **Workload Identity Federation for GKE is the only correct way to reach GCP APIs from Pods.** Never
  mount SA JSON keys. Bind the Kubernetes SA to an IAM principal; least privilege per namespace/workload.
- **Accelerators are node-pool-shaped.** GPU type ∝ machine family (A100→a2, H100→a3-highgpu/megagpu,
  H200→a3-ultragpu, B200→a4, GB200 NVL — verify). TPUs: the node pool *is* the slice — machine type +
  `--tpu-topology`, single-host vs **multi-host (all-or-nothing, ICI, compact placement)**. Multi-host
  pairs with `[[jobset-leaderworkerset]]` + `[[kueue-advanced]]`.
- **GPU sharing by isolation need:** time-sharing (none) < MPS (weak) < MIG (HW partitions, A100/H100) <
  DRA (emerging native fractional). Never share GPUs for latency-critical or training; give them whole
  devices + the right interconnect.
- **High-perf ML networking:** multi-networking + **GPUDirect-TCPX (A3) / TCPXO (A3-Mega) / RDMA-RoCE
  (A3-Ultra/A4/GB200)** + NCCL plugin. It's a node-pool design decision; benchmark NCCL all-reduce/MFU.
  Wrong TPU topology silently tanks MFU too.
- **Networking defaults:** VPC-native (plan Pod/Service CIDRs + `max-pods-per-node` to avoid silent IP
  exhaustion) · **Dataplane V2** (eBPF/Cilium, NetworkPolicy) · **container-native LB via NEGs** (LB
  targets Pod IPs) · **Gateway API** for new ingress (classic Ingress legacy) · private clusters + Cloud
  NAT. Don't force-remove a stuck LB/NEG **finalizer** — fix the orphaned GCP backend first or you leak
  paid resources.
- **Storage by access pattern:** Hyperdisk (Balanced/Throughput/Extreme) for block · **Hyperdisk ML**
  for read-heavy many-reader model loading · Filestore for RWX NFS · **GCS FUSE CSI** for
  datasets/checkpoints (tune caching + Workload Identity) · **Parallelstore CSI** for parallel scratch ·
  **local SSD** for ephemeral in-step scratch/checkpoint staging (non-durable). Don't run hot training
  data off plain PD.
- **Fast model loading / restart (newer, verify availability):** cut **serving cold-start** with weight
  **streaming from GCS** (run:ai Model Streamer-style), **GCS FUSE + local-SSD caching**, **Hyperdisk ML**
  for many-reader loading, and **image streaming** for multi-GB images. Cut **training restart** with
  **Multi-Tier Checkpointing (MTC)** = node-local SSD + GCS + **peer recovery**; depth in
  `[[ml-checkpointing-orbax]]`, serving runtimes in `[[serving-frameworks]]`.
- **Autoscaling is layered:** HPA/VPA (Pods) over Cluster Autoscaler (existing pools) and **NAP** (new
  pools, incl. GPU/TPU). For gang/batch ML use **Kueue + ProvisioningRequest** so capacity scales up
  atomically (no partial-schedule deadlock). Pin accelerator pools to zones with the SKU/quota. See
  `[[autoscaling-kubernetes]]`, `[[kueue-advanced]]`.
- **Security:** Shielded nodes · Binary Authorization (signed images) · **GKE Sandbox (gVisor)** —
  user-space-kernel isolation via **RuntimeClass `gvisor`** for untrusted/multi-tenant code and **AI-agent
  tool execution** (defense-in-depth, not a VM; adds syscall overhead; verify GPU/feature compatibility);
  see `[[ai-security-on-gke]]`, `[[llm-app-agent-frameworks]]` · Confidential GKE where required ·
  Security Posture · **RBAC↔IAM** (need both; map Google Groups → RBAC; don't use broad IAM to dodge
  RBAC) · **Secret Manager CSI** + KMS app-layer Secret encryption (no keys/secrets baked in images).
- **Operations:** subscribe to a **release channel** (Rapid/Regular/Stable/Extended); set **maintenance
  windows + exclusions** (and surge vs **blue-green** node upgrades) so upgrades don't kill long
  training. Use **Fleets + Config Sync/Policy Controller + Config Connector** for multi-cluster GitOps;
  **Backup for GKE** for state/DR. Cost: Spot (fault-tolerant, taint it) · NAP · Autopilot per-Pod
  billing · CUDs · right-sized Hyperdisk · namespace cost allocation.
- **Observability is GCP-native:** Cloud Logging/Monitoring + **Managed Service for Prometheus**
  (`PodMonitoring`/`ClusterPodMonitoring`). Enable **DCGM** (GPU) and **TPU** metrics — GPU
  **`duty_cycle`** vs TPU **`tensorcore_utilization`** (different names, same intent). Accelerator/
  container metrics ride the `k8s_container` resource carrying **`namespace_name`** → per-team attribution.

## ML stack on GKE (reference)
GPU/TPU pools (+ Custom Compute Classes, NAP) → GPUDirect/TPU topology → **Kueue + ProvisioningRequest**
+ **JobSet/LeaderWorkerSet** → **GCS FUSE + Hyperdisk ML + Parallelstore + local SSD** (+ **MTC** for
fast restart, **model streaming/FUSE cache/image streaming** for fast cold-start — verify availability) →
vLLM/SGLang/MaxText serving → **Managed Prometheus + DCGM/TPU metrics**. This is the AI Hypercomputer
pattern. See
`[[aiml-on-kubernetes]]`, `[[serving-frameworks]]`, `[[ml-frameworks]]`, `[[training-frameworks]]`.

## Top anti-patterns to flag
SA JSON keys in Pods · Autopilot expected to behave like Standard · unplanned IP exhaustion · GPUs/TPUs
without the interconnect/topology · gang jobs without ProvisioningRequest (partial-schedule deadlock) ·
Spot without checkpointing/taints · force-removing stuck LB/NEG finalizers (leaks paid GCP resources) ·
static pinned versions / no maintenance exclusions · slow storage for hot data · GPU sharing for
latency-critical/training · IAM granted to dodge RBAC.
