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

---

# Reference — gke-master

# GKE Master — Reference Guide

This is the GKE-specific layer. Generic Kubernetes (objects, scheduling, controllers, HPA mechanics) is
in `[[kubernetes-expert]]` / `[[autoscaling-kubernetes]]`. Everything here is what changes *because it's
GKE on Google Cloud*: modes, node pools, accelerators, GCP networking/storage/identity, managed ops, and
GCP-native observability — with an ML-platform bias.

> **Currency caveat (2026):** Machine families, accelerator SKUs, regional availability, default
> versions, and quota numbers change constantly. Names below are accurate starting points; **always
> verify against current GKE docs and your project's regional availability/quota** before committing to
> a design. Never hardcode a machine-type name or quota you haven't checked.

---

## 1. Mental model: what GKE adds over Kubernetes

GKE is managed Kubernetes where Google operates the control plane (etcd, apiserver, scheduler,
controller-manager) and gives you two operating models for the data plane:

- **Standard** — you create and own **node pools** (groups of identical Compute Engine VMs). You choose
  machine type, node image, disk, local SSD, system config, networking, and you pay for the VMs whether
  or not Pods fill them. Maximum control; you carry node lifecycle responsibility.
- **Autopilot** — Google runs the nodes. You submit Pods; GKE provisions/bin-packs/scales the right
  nodes underneath and you **pay per Pod resource request** (CPU/mem/ephemeral storage, plus accelerator
  pricing for GPU/TPU Pods). Hardened, opinionated defaults (Workload Identity, Shielded nodes, no host
  access, restricted privileged Pods). You give up node-level knobs in exchange for less ops.

The two modes are **converging**: Autopilot now supports many GPU/TPU workloads, custom compute classes,
and more scheduling control; Standard gained Autopilot-style safety features. Choose by **how much node
control the workload genuinely needs**, not by reflex.

| Decision factor | Lean Autopilot | Lean Standard |
|---|---|---|
| General services, web/API, most batch | ✅ | |
| You want to stop paying for idle headroom / per-Pod billing | ✅ | |
| Hardened defaults with minimal effort | ✅ | |
| Custom node **system config** (sysctls, kubelet, hugepages) | (limited) | ✅ |
| Specific **local SSD** layout / RAID for checkpoints | | ✅ |
| Custom **GPUDirect** (TCPX/TCPXO/RDMA) multi-NIC plumbing | | ✅ (most flexibility) |
| Multi-host TPU slices with precise topology | possible | ✅ (full control) |
| DaemonSets needing host networking / privileged access | | ✅ |
| Tight, predictable per-node cost accounting via CUDs | | ✅ |

**Custom Compute Classes** (a prioritized list of node shapes with fallback, e.g. "Spot H100 → on-demand
H100 → A100") work in both modes and are the modern way to express "give me this accelerator, fall back
gracefully" — prefer them over hand-pinned `nodeSelector` machine types.

---

## 2. Node pools & compute (Standard)

A **node pool** = a set of identical VMs with a shared config (machine type, image, disk, autoscaling
bounds, taints/labels, locations). Design principles:

- **One node pool per distinct hardware/role.** Separate CPU system pools from GPU pools from TPU pools;
  separate Spot from on-demand. Don't mix machine types in one pool.
- **Machine families** (verify current): `e2`/`n2`/`n2d`/`c3`/`c3d`/`c4`/`t2d` for general & CPU
  compute; `a2`/`a2-ultragpu` (A100), `a3-highgpu`/`a3-megagpu`/`a3-ultragpu` (H100/H200), `a4`
  (B200), and GB200-class for NVL/rack-scale (verify names/availability per region). Memory-bound →
  `m`-series; storage-bound → local-SSD-attachable families.
- **Node images:** **Container-Optimized OS (COS)** with `containerd` is the default and recommended —
  minimal, locked-down, auto-updated. Ubuntu images exist for specific driver/module needs. Avoid
  Docker-runtime images (gone). For GPUs you generally want COS with the managed driver install.
- **Image streaming** (GKE) lazily streams container image data so large images (common in ML) start
  before the full pull completes — enable it; it dramatically cuts cold-start on multi-GB CUDA images.
- **Ephemeral storage:** boot disk vs **local SSD-backed ephemeral storage**. For checkpoint/scratch and
  GCS-FUSE caches, back ephemeral storage with local SSD (`--ephemeral-storage-local-ssd`) for far
  higher throughput than the boot PD.
- **Node system config** (`--system-config-from-file`): set kubelet flags (e.g. `cpuManagerPolicy`,
  `podPidsLimit`), sysctls (e.g. `net.core.*`, `vm.max_map_count` for some ML runtimes), and hugepages.
  This is Standard-only granularity that ML data planes often need.

### Spot / preemptible
- **Spot VMs** (the modern form; preemptible is the legacy 24h-capped variant) cut cost ~60–90% but can
  be reclaimed anytime with ~30s notice. Use for fault-tolerant training with frequent checkpointing,
  batch, and dev. **Taint Spot pools** and tolerate explicitly so only opted-in workloads land there.
  Pair with Kueue + checkpointing so preemption is just a resume, not a loss.

### Node Auto-Provisioning (NAP)
- NAP lets the cluster **create new node pools on demand** with shapes inferred from pending Pods'
  requests, including GPU and TPU pools — beyond just scaling existing pools (Cluster Autoscaler). Bound
  it with resource limits, allowed machine families, and defaults (image type, disk, Shielded, Spot).
  NAP + Custom Compute Classes is the cleanest "ask for an accelerator, get a right-sized pool" pattern.

---

## 3. Accelerators — GPUs

GPUs attach to specific machine families; **the GPU type is a property of the node pool/machine type**,
not a free-floating request.

| GPU | Typical machine family | GPUDirect | Notes (verify) |
|---|---|---|---|
| A100 40/80GB | `a2`, `a2-ultragpu` | — | Mature; MIG-capable |
| L4 | `g2` | — | Cost-effective inference |
| H100 80GB | `a3-highgpu` (8×) | **TCPX** | Mainstream large training |
| H100 (Mega) | `a3-megagpu` | **TCPXO** | Higher inter-node BW |
| H200 | `a3-ultragpu` | **RDMA/RoCE** | More HBM per GPU |
| B200 | `a4` | **RDMA** | Blackwell |
| GB200 NVL | GB200/`a4x`-class | **RDMA**, NVLink domain | Rack-scale; verify exact names |

- **Driver install:** prefer GKE-managed GPU drivers (`--accelerator type=…,count=…,gpu-driver-version=default|latest`)
  on COS, which runs the NVIDIA driver installer DaemonSet for you. Request GPUs with
  `nvidia.com/gpu: <n>` and (on Autopilot/compute classes) a `cloud.google.com/gke-accelerator` selector.
- **GPU sharing strategies** (set on the node pool):
  - **Time-sharing** — many Pods time-slice one physical GPU; oversubscription, **no memory isolation**.
    Good for bursty, tolerant inference/dev; never for SLO-critical or training.
  - **MPS** (Multi-Process Service) — concurrent execution with limited isolation; better than
    time-sharing for co-located inference.
  - **MIG** (Multi-Instance GPU, A100/H100) — hardware partitions (e.g. 7×1g, 3×2g) with real
    memory/SM isolation. Best for predictable multi-tenant inference. Choose the MIG profile at node-pool
    creation.
  - **DRA (Dynamic Resource Allocation)** — the Kubernetes-native, GKE-emerging way to request
    structured/fractional/partitioned accelerators via ResourceClaims rather than the
    `nvidia.com/gpu` integer extended resource. This is the strategic direction for flexible GPU sharing;
    check current GKE support level before depending on it.
- **Rule:** don't share GPUs for latency-critical training or tightly-coupled multi-GPU jobs — give
  them whole devices and the right interconnect.

---

## 4. Accelerators — TPUs

TPUs are first-class on GKE and behave differently from GPUs: **the node pool models a TPU slice.**

- **Generations (verify availability):** **v5e** (cost/inference & mid training), **v5p** (large
  training, high-BW ICI), **v6e "Trillium"** (current-gen efficiency/throughput). Each has its own
  machine types and supported topologies.
- **The SKU/node-pool model:** you create a TPU node pool with a machine type (e.g. a `ct5lp`/`ct5p`/
  `ct6e`-family type — verify exact names) and a **`--tpu-topology`** (e.g. `2x2x1`, `2x4`, `4x4x4`).
  Topology + machine type determine how many chips and hosts the slice has.
- **Single-host vs multi-host slices:**
  - **Single-host slice** — one VM exposes all the slice's TPU chips; the node pool has one node per
    slice. Simpler; good for inference and small training.
  - **Multi-host slice** — the slice spans multiple VMs connected by **ICI (inter-chip interconnect)**.
    The node pool is created as a set of hosts that must be scheduled and run **all-or-nothing**. GKE
    surfaces this via labels (`cloud.google.com/gke-tpu-topology`, `…-tpu-accelerator`) and the slice's
    host count; Pods are placed with **multi-host placement / compact placement** so the whole slice
    lands on the contiguous hardware.
- **Requesting TPUs:** Pods request `google.com/tpu: <chips-per-host>` and select the
  accelerator/topology via node labels. For multi-host, you run **one Pod per host** coordinated as a
  group — this is exactly where **`[[jobset-leaderworkerset]]`** (JobSet/LWS) and **`[[kueue-advanced]]`**
  (gang scheduling, ProvisioningRequest) come in: the slice scales up atomically and the workers join
  the same JAX/`MaxText`/PyTorch-XLA mesh.
- **Topology matters for collectives.** Pick the topology to match the model's parallelism mapping
  (data/tensor/pipeline) so ICI carries the heavy traffic. Mismatched topology silently tanks MFU.

See `[[ml-frameworks]]` for JAX/XLA & PyTorch-XLA specifics; this skill covers only the *GKE node-pool*
shape of TPUs.

---

## 5. Autoscaling on GKE

Layered, and GKE-specific where it touches infrastructure:

- **Pod layer (generic, see `[[autoscaling-kubernetes]]`):** HPA scales replicas on
  CPU/mem/custom/external metrics (incl. Managed Prometheus metrics via the adapter); VPA right-sizes
  requests. On Autopilot, VPA + per-Pod billing means right-sizing requests is *directly* a cost lever.
- **Node layer:**
  - **Cluster Autoscaler (CA)** scales *existing* node pools between min/max when Pods are pending or
    nodes idle. Per-pool, per-location.
  - **Node Auto-Provisioning (NAP)** creates *new* pools with inferred shapes — including GPU/TPU.
  - **Location/zone policy:** node pools span zones in a region; for accelerators you often **pin to
    specific zones** where the SKU/quota exists. Use `--node-locations`. Multi-host TPU slices and
    GPUDirect pools often must be single-zone for placement/latency.
- **Batch/gang ML capacity → `ProvisioningRequest` + Kueue.** For training that needs N nodes *together*
  or not at all, drive scale-up with a **ProvisioningRequest** (atomic capacity request to the
  autoscaler, including TPU slices) managed by **Kueue**. This avoids the classic failure where CA
  brings up nodes one at a time and a gang job partially schedules then deadlocks. See
  `[[kueue-advanced]]`.
- **Scale-down protection:** annotate Pods that must not be evicted
  (`cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`) — important for long checkpoints and
  stateful training steps.

---

## 6. Networking

### VPC-native & IP planning
- **VPC-native (alias IP) is the default and required for most features.** Pods get real VPC IPs from a
  secondary range; Services from another. **Plan IP ranges early** — Pod-per-node density × max nodes can
  exhaust ranges fast on big ML clusters. Consider larger Pod CIDRs and `--max-pods-per-node` tuning.

### Dataplane V2
- **GKE Dataplane V2** is the eBPF/Cilium-based data plane. It provides **Kubernetes NetworkPolicy** (and
  FQDN/CIDR-aware policy), built-in flow logging/observability, and better scale than the legacy
  iptables/kube-proxy path. Make it the default for new clusters. Network policy enforcement and
  Dataplane V2 metrics are GKE-integrated.

### Load balancing & ingress
- **Container-native load balancing via NEGs (Network Endpoint Groups):** the LB targets **Pod IPs
  directly** instead of bouncing through node ports + kube-proxy. This is the correct default for
  HTTP(S)/external services — better latency, accurate health checks, even traffic distribution. A
  Service/Gateway annotation creates the NEG.
- **Gateway API is the strategic ingress** (GKE Gateway controller: regional/global external/internal
  classes), superseding the older **GKE Ingress** (still supported, still NEG-backed). Use Gateway for
  new designs; multi-cluster Gateway for fleet-wide traffic.
- **Finalizer note (ties to earlier NEG/LB discussion):** GKE LB/NEG controllers attach **finalizers**
  to Services/Ingress/Gateway so backend GCP resources (forwarding rules, backend services, NEGs) are
  cleaned up before the K8s object is deleted. If a Service hangs in `Terminating`, suspect a stuck LB
  finalizer / orphaned GCP backend — don't force-remove the finalizer until the GCP resource is gone, or
  you leak (and keep paying for) load balancer resources.

### High-performance ML networking
- **Multi-networking:** attach **multiple network interfaces** to Pods (via `GKENetworkParamSet` +
  `Network` CRDs and multiple node NICs) so collective traffic rides dedicated high-BW networks separate
  from the default Pod network.
- **GPUDirect** kernel-bypass GPU-to-GPU networking, by family:
  - **GPUDirect-TCPX** on A3 (`a3-highgpu`)
  - **GPUDirect-TCPXO** on A3-Mega (`a3-megagpu`)
  - **GPUDirect-RDMA / RoCE** on A3-Ultra / A4 (H200/B200) and GB200 NVL.
  Each needs the multi-NIC node pool, the right node image/drivers, and the **NCCL plugin/installer
  DaemonSet** so NCCL uses the fast path. This is a node-pool design decision — get it right at creation;
  retrofitting is painful. Validate with NCCL all-reduce benchmarks before declaring victory.

### DNS & private clusters
- **Cloud DNS for GKE** can replace `kube-dns` with managed, scalable DNS (cluster-scoped or VPC-scoped)
  — useful at scale and for cross-cluster name resolution.
- **Private clusters:** nodes have no public IPs; control-plane access via private endpoint +
  authorized networks; egress via Cloud NAT. Standard for production/regulated ML. Pull images via
  Artifact Registry over private Google access.

---

## 7. Storage

Choose by **access pattern**, not habit:

| Need | Use | Why |
|---|---|---|
| RWO block, general PVC | **Persistent Disk / Hyperdisk** (Balanced/Throughput/Extreme) | Standard block; Hyperdisk decouples IOPS/throughput from size |
| Read-heavy model weights, many readers | **Hyperdisk ML** | High aggregate read throughput, attach read-only to many nodes; fast model loading |
| Shared POSIX (RWX) | **Filestore** (Basic/Zonal/Enterprise) | Managed NFS for shared datasets/home dirs |
| Datasets/checkpoints in object storage | **GCS FUSE CSI driver** | Mount a GCS bucket as a volume; great for training data & checkpoints; tune caching |
| High-throughput parallel scratch | **Parallelstore CSI** | DAOS-based parallel FS for data-hungry training/HPC |
| Ephemeral fast scratch/checkpoints | **Local SSD** (ephemeral) | Highest IOPS/throughput, node-local, non-durable |

Guidance:
- **GCS FUSE CSI** is the workhorse for ML data/checkpoints: enable the driver
  (`--addons GcsFuseCsiDriver`), mount via CSI ephemeral volume or PV, and **tune caching**
  (file/metadata/stat cache, optional local-SSD-backed cache) — naive FUSE on small-file datasets is
  slow; caching and parallel/streaming reads fix it. Use Workload Identity for bucket access (no keys).
- **Hyperdisk ML** is the right answer for "100 nodes all need to load the same 200GB checkpoint fast" —
  read-only multi-attach, far better than each node pulling from GCS cold.
- **Local SSD** for in-step scratch and fast local checkpoint staging (then async-flush to GCS). Never
  rely on it for durability — it's lost on node recreation/preemption.
- **Parallelstore** when you need sustained parallel throughput beyond Filestore (large multi-host
  training reading enormous datasets).
- Pick **Hyperdisk over Extreme-everywhere**: provision IOPS/throughput to the workload to avoid
  overpaying.

### Fast model loading / model streaming (cutting inference cold-start)

For large-model serving, **time-to-first-token-ready** is dominated by pulling tens-to-hundreds of GB of
weights onto each replica. Naively pulling from cold GCS on every pod start makes autoscaling slow and
expensive. The techniques below stack; many are newer or partner features, so **verify current GKE
availability, naming, and supported accelerators** before designing around any one of them.

- **Stream weights instead of fully downloading first.** Model-streaming loaders (e.g. the
  **run:ai Model Streamer** and similar tools, often integrated with serving runtimes) read weight
  tensors **directly from Cloud Storage into GPU/host memory concurrently**, overlapping I/O with load so
  the engine starts before the full object set is on local disk. This pairs well with `[[serving-frameworks]]`
  runtimes (vLLM/SGLang/TensorRT-LLM) that support streaming/lazy weight loaders — check the runtime's
  current support and the loader's GKE/GCS integration.
- **GCS FUSE with aggressive caching** so repeated/parallel reads hit a local cache instead of going to
  object storage each time. Mount the bucket via the **GCS FUSE CSI driver** and enable **file/metadata/
  stat caching backed by local SSD** (`--ephemeral-storage-local-ssd`), plus parallel/streaming download
  tuning. This turns "every replica re-downloads from cold GCS" into "warm local reads," which is often
  the single biggest cold-start win. (Tuning knobs evolve — verify current flags.)
- **Hyperdisk ML for read-heavy weight loading.** Stage weights onto a **Hyperdisk ML** volume and
  attach it **read-only to many nodes** — high aggregate read throughput so a large fleet loads the same
  checkpoint fast without each node pulling from GCS. Best when many replicas share one model version and
  you can pre-populate the volume (see the storage table above).
- **Image & data caching to cut cold-start.** Enable GKE **image streaming** (section 2) so multi-GB
  CUDA/serving images start before the full pull completes; and consider **baking small/frequently-loaded
  weights into a cached layer or a preloaded data volume**. For larger evolving features (secondary boot
  disks / preloaded container & model data, image/data caching add-ons), **verify what's currently GA on
  GKE** — this area is moving fast.
- **Rule of thumb:** size the path to the bottleneck — for one model on a big fleet, **Hyperdisk ML**;
  for many models or churny weights, **streaming from GCS + FUSE local-SSD cache**; always combine with
  **image streaming**. Measure cold-start (pod-start → ready) directly; don't assume.

### Multi-Tier Checkpointing (MTC) for fast training restart

At large scale, training-checkpoint **save and restore** are throughput-critical: long save stalls waste
accelerator time, and slow restores after a failure/preemption lengthen every restart. **Multi-Tier
Checkpointing (MTC)** is GKE's tiered approach that combines **fast node-local storage (local SSD / RAM)**
with **durable Cloud Storage**, plus **peer recovery** between nodes:

- **The tiers:** checkpoints are written first to **node-local SSD (or memory)** — very fast, low-stall —
  and **asynchronously backed up to Cloud Storage (GCS)** for durability. Restore prefers the fastest
  tier that has a valid checkpoint.
- **Peer recovery:** when a node or slice is replaced (preemption/failure), a restarting worker can
  **pull the latest checkpoint shard from a healthy peer node's local copy** instead of always reading
  the full checkpoint cold from GCS — dramatically shortening restart time for multi-host jobs.
- **Why it matters:** it decouples checkpoint *frequency* (you can save often, cheaply, to local SSD)
  from *durability* (GCS still holds a copy), so you lose less work on Spot/maintenance interruptions
  while keeping save stalls small. This is the natural complement to **Spot pools + frequent
  checkpointing** and to **JobSet/LeaderWorkerSet** multi-host restart behavior.
- **Caveats:** MTC is a **newer, fast-moving** GKE capability with specific requirements (local SSD on
  the node pool, a supported checkpoint library/integration, and framework support). **Verify current
  GKE availability, supported frameworks, and setup** before designing around it — don't assume flags or
  guarantees. The **checkpointing-library depth** (sharding, async save, format, replication) lives in
  **`[[ml-checkpointing-orbax]]`**; MTC is the GKE storage-tiering substrate underneath it.

---

## 8. Security & identity

- **Workload Identity Federation for GKE** — the *only* correct way to call GCP APIs from Pods. Bind a
  Kubernetes ServiceAccount to an IAM principal (`principal://…/sa/<ns>/<ksa>` or via an IAM SA with
  Workload Identity binding). **Never mount SA JSON keys.** Scope per namespace/workload; least
  privilege. (See `examples.md` for the binding sketch.)
- **Shielded GKE nodes** — secure boot, vTPM, integrity monitoring. On by default in Autopilot; enable
  in Standard.
- **Binary Authorization** — admission-time policy that only allows attested/signed images. Wire into CI
  so only your build pipeline's signed images deploy.
- **GKE Sandbox (gVisor)** — see the dedicated subsection below; user-space kernel isolation for
  untrusted/multi-tenant code and AI-agent tool execution, applied per-Pod via RuntimeClass.
- **Confidential GKE** — memory encryption (AMD SEV / confidential VMs) for sensitive data; some
  accelerator combinations are restricted — verify.
- **Security Posture dashboard** — built-in config & vulnerability scanning surfaced in the console;
  enable it for continuous misconfig/CVE visibility.
- **RBAC ↔ IAM:** GKE authorizes via **both** Cloud IAM (coarse, project/cluster-level) **and**
  Kubernetes RBAC (fine, in-cluster). A user needs IAM to reach the cluster and RBAC for in-cluster
  permissions. Map human/group identities to RBAC via **Google Groups for RBAC**. Don't grant
  `container.admin` as a substitute for proper RBAC.
- **Secrets:** **Secret Manager CSI driver** (or the broader Secret Store CSI) to mount secrets from
  Secret Manager — no secrets baked into images or plain K8s Secrets where avoidable. Also enable
  **application-layer secrets encryption** (Cloud KMS) for etcd-stored Secrets.

### GKE Sandbox (gVisor) for untrusted code & AI-agent tool execution

**GKE Sandbox** runs Pods inside **gVisor**, a user-space application kernel. Instead of letting a
container call the host Linux kernel directly, gVisor intercepts syscalls in user space and re-implements
the kernel ABI, so a workload's syscalls never reach the host kernel directly. This shrinks the attack
surface dramatically for **code you don't trust** — a primary use case in 2026 is **AI agents executing
model-generated or user-supplied code/tools** (code interpreters, sandboxed tool calls, untrusted
notebooks, multi-tenant CI). Google also markets this pattern as an **Agent Sandbox** for agentic
workloads — names and packaging are evolving, so **verify the current product/feature naming and
availability** before depending on a specific entry point.

- **The model is RuntimeClass-based.** GKE Sandbox is exposed as a Kubernetes **`RuntimeClass`** (the
  gVisor handler, `runsc`). You opt a Pod in with `spec.runtimeClassName: gvisor`; un-annotated Pods keep
  the default `runc` runtime. Enable sandboxing on a Standard node pool
  (`gcloud container node-pools create … --sandbox type=gvisor`) — GKE taints sandbox nodes so only
  RuntimeClass-`gvisor` Pods land there. On Autopilot, request it via the supported Pod spec / compute
  class — **verify the current Autopilot path**. You can mix sandboxed and non-sandboxed pools in one
  cluster and reserve sandboxing for the untrusted tier.
- **What it protects against:** container-escape / privilege-escalation via the host kernel — a
  malicious or buggy workload exploiting a kernel vulnerability is contained within gVisor instead of
  reaching the node kernel and other tenants. It is **defense in depth**, layered with (not a replacement
  for) least-privilege Pod security, NetworkPolicy (Dataplane V2), Workload Identity scoping, and
  non-root/seccomp. For AI agents, it bounds the blast radius of arbitrary generated code.
- **What it does *not* do:** it is not a full VM boundary (for the strongest isolation consider separate
  node pools / clusters / Confidential GKE), and it doesn't replace network or IAM controls. Pair it with
  tight egress policy so sandboxed code can't exfiltrate.
- **Performance & compatibility tradeoffs:** the syscall interception adds overhead — syscall-heavy and
  high-I/O workloads see the most; CPU-bound compute sees the least. **Not all features are compatible**:
  historically **GPUs/TPUs, some `/proc`-sysfs access, certain syscalls, hostpath/host networking, and
  privileged features** were unsupported or restricted under gVisor; GPU support has been expanding —
  **verify the current compatibility matrix** for your accelerator/feature set rather than assuming. Test
  the actual workload under `gvisor` before committing; some binaries that touch unusual syscalls fail or
  slow down.
- **When to use:** running untrusted/multi-tenant code, agent tool execution, customer-supplied training
  scripts, or notebook backends where a tenant could be hostile. **When to skip:** trusted first-party
  latency- or syscall-sensitive services, and (subject to current compatibility) tightly-coupled
  accelerator training. See **`[[ai-security-on-gke]]`** for the broader agent-isolation threat model and
  **`[[llm-app-agent-frameworks]]`** for where agent code-execution sandboxes fit in the app stack.

---

## 9. Operations

- **Release channels:** subscribe to **Rapid / Regular / Stable / Extended** rather than pinning static
  versions. Channels deliver tested upgrades on a cadence (Rapid = newest, Stable = conservative,
  Extended = longest support window). Set the channel per cluster; let Google handle control-plane
  upgrades within it.
- **Maintenance windows & exclusions:** define when automatic node upgrades may run
  (`--maintenance-window-*`) and **exclusions** to freeze upgrades during launches/peak ML jobs. Long
  training runs need exclusions or surge config so a node upgrade doesn't kill a multi-day job.
- **Node upgrade strategies:**
  - **Surge upgrades** — spin up extra nodes (`maxSurge`), drain old ones (`maxUnavailable`). Default,
    fast, needs spare capacity/quota.
  - **Blue-green upgrades** — bring up a full new node pool, shift workloads, keep the old pool for fast
    rollback. Safer for sensitive/stateful ML; more capacity required. Choose per node pool.
- **Fleets & multi-cluster:**
  - **Fleet** — the org-level grouping of clusters (and the unit for team scoping, fleet-wide features).
  - **Multi-cluster Services (MCS)** and **Multi-cluster Gateway/Ingress** — cross-cluster service
    discovery and traffic.
  - **Config Sync / Config Management** — GitOps reconciliation of cluster config from a repo across the
    fleet (policy + config drift control). **Policy Controller** (Gatekeeper-based) for guardrails.
  - **Config Connector** — manage **GCP resources** (buckets, IAM, SQL, even other clusters) as
    Kubernetes CRDs from inside the cluster (KRM). Pairs with Config Sync for full GitOps of infra.
- **Backup for GKE** — back up and restore cluster state (Kubernetes objects) **and PV data**; schedule
  backups, restore to another cluster/region for DR or migration.
- **Cost optimization:** Spot pools (fault-tolerant ML) · NAP to eliminate idle pools · Autopilot per-Pod
  billing to stop paying for headroom · **Committed Use Discounts (CUDs)** for steady GPU/TPU/CPU ·
  right-sized Hyperdisk · GKE cost-allocation/usage-metering by namespace/label · bin-pack with proper
  requests + VPA.

---

## 10. Observability

- **Cloud Logging & Cloud Monitoring** are integrated: system/workload logs and metrics flow
  automatically (configurable per cluster). Control-plane logs (apiserver/scheduler/controller-manager)
  are opt-in — enable for production debugging.
- **Google Cloud Managed Service for Prometheus (GMP)** — fully-managed, Prometheus-compatible metrics
  at scale. Scrape with **`PodMonitoring`** (namespaced) / **`ClusterPodMonitoring`** (cluster-wide)
  CRDs; query with PromQL/Cloud Monitoring/Grafana. This is the recommended path for app & ML-framework
  metrics (vLLM, JAX, DCGM exporters) — see `[[serving-frameworks]]`.
- **Accelerator metrics:**
  - **GPU:** enable **DCGM** metrics (managed DCGM integration / DCGM exporter) for utilization,
    memory, ECC, NVLink, etc. Key signal: **`duty_cycle`** (fraction of time the GPU was active).
  - **TPU:** GKE exposes TPU metrics; key signal: **`tensorcore_utilization`** (TensorCore busy
    fraction) plus memory/HBM usage. **GPU `duty_cycle` and TPU `tensorcore_utilization` are different
    metric names with the same intent** — don't look for one on the other accelerator.
- **Per-team attribution:** container/accelerator metrics are reported on the **`k8s_container`**
  monitored-resource, which carries **`namespace_name`** (plus pod/container) — so you can attribute
  GPU/TPU utilization and spend per team by namespace. Design namespaces so this attribution is clean.

---

## 11. The recommended ML stack on GKE (training & inference)

A battle-tested GKE AI/ML platform (Google's **AI Hypercomputer** is the integrated reference
architecture for this) typically composes:

- **Compute:** GPU pools (A3/A3-Mega/A3-Ultra/A4) and/or TPU pools (v5e/v5p/v6e), with **Custom Compute
  Classes** for graceful fallback and **NAP** for on-demand provisioning.
- **Fast interconnect:** GPUDirect-TCPX/TCPXO/RDMA via multi-networking + NCCL plugin for GPUs; correct
  TPU topology + multi-host placement for TPUs.
- **Job orchestration:** **Kueue** for quota/gang/priority/fair-sharing and **ProvisioningRequest**-driven
  scale-up (`[[kueue-advanced]]`); **JobSet / LeaderWorkerSet** for multi-host training and disaggregated
  inference (`[[jobset-leaderworkerset]]`).
- **Data & checkpoints:** **GCS FUSE CSI** (datasets/checkpoints) + **Hyperdisk ML** (fast weight
  loading) + **Parallelstore** (parallel scratch) + **local SSD** (in-step scratch). For fast restart use
  **Multi-Tier Checkpointing (MTC)**; for fast inference cold-start use **model streaming / FUSE
  local-SSD cache / image streaming** (section 7) — verify current availability.
- **Serving:** vLLM/SGLang/Triton/TensorRT-LLM/Dynamo on GPU, JAX/MaxText on TPU — see
  `[[serving-frameworks]]` / `[[ml-frameworks]]`.
- **Observability:** **Managed Prometheus** + DCGM/TPU metrics, attributed by `namespace_name`.
- **Identity/security:** Workload Identity Federation, private cluster, least-privilege RBAC↔IAM.

See `[[aiml-on-kubernetes]]` for the end-to-end ML architecture; this skill is the GKE substrate beneath
it.

---

## 12. Anti-patterns & GKE gotchas

- **SA JSON keys in Pods.** Always Workload Identity Federation. Mounted keys are the #1 GKE security
  finding.
- **Treating Autopilot like Standard.** Don't expect host networking, privileged DaemonSets, arbitrary
  sysctls, or specific local-SSD layouts on Autopilot. If you need those, that's a Standard signal.
- **Ignoring IP exhaustion.** VPC-native secondary ranges run out silently on big clusters; you discover
  it when Pods can't schedule. Plan Pod/Service CIDRs and `max-pods-per-node` up front.
- **GPU/TPU without the interconnect.** Buying H100/TPU then running collectives over the default Pod
  network (no TCPX/TCPXO/RDMA, wrong NCCL config, or wrong TPU topology) wastes most of the FLOPs.
  Always benchmark all-reduce/MFU.
- **Gang jobs without ProvisioningRequest/Kueue.** CA brings up nodes incrementally; a multi-host job
  partially schedules, holds the nodes, and deadlocks. Use atomic ProvisioningRequest + gang scheduling.
- **Spot without checkpointing / taints.** Spot reclamation kills uncheckpointed training; untainted
  Spot pools attract workloads that can't tolerate preemption.
- **Force-deleting a stuck LB/NEG finalizer.** Leaks GCP forwarding rules/backend services you keep
  paying for. Fix the underlying GCP resource first.
- **Pinning static cluster versions / no maintenance exclusions.** Either you drift onto unsupported
  versions, or an auto-upgrade drains a node mid-training. Use release channels + exclusions/blue-green.
- **Slow storage for hot data.** Running training data or 100-node weight loading off plain PD or naive
  GCS FUSE (no caching). Match storage to access pattern (Hyperdisk ML / Parallelstore / FUSE caching).
- **GPU sharing for the wrong workload.** Time-sharing/MPS on latency-critical or training jobs causes
  noisy-neighbor stalls. Use MIG/DRA for isolation, whole devices for training.
- **Confusing IAM and RBAC.** Granting broad IAM (`container.admin`) to dodge RBAC setup; in-cluster
  authorization still needs RBAC. Map groups → RBAC.
- **One giant cluster vs fleet.** Past a point, separate clusters + Fleet/MCS/Config Sync beats one
  mega-cluster (blast radius, version skew, quota). But don't over-shard either.

---

## Rationalizations & rebuttals

| Excuse | Rebuttal |
|---|---|
| "Static SA JSON keys are simpler than Workload Identity Federation." | A mounted key is a long-lived, exfiltratable credential and the #1 GKE security finding. WIF removes the key entirely — bind a KSA to an IAM principal once and every Pod gets short-lived, auto-rotated tokens. It's *less* setup, not more. |
| "We'll skip Dataplane V2 / NEGs and use the legacy path; it works." | Without Dataplane V2 you have no eBPF NetworkPolicy, no FQDN/CIDR policy, no built-in flow observability, and worse scale. Without NEGs the LB bounces through node ports + kube-proxy — extra hop, inaccurate health checks, uneven distribution. Both are the default for a reason; retrofitting Dataplane V2 means a cluster rebuild. |
| "One big node pool is easier to manage than many." | Mixed machine types in one pool break autoscaling, taint/label targeting, and accelerator placement. One pool per hardware/role (CPU vs GPU vs TPU, Spot vs on-demand) is what lets CA, NAP, and Spot taints work at all. |
| "Default timeouts / maintenance windows are fine for our training jobs." | An auto node-upgrade will drain a node mid-run and kill a multi-day job. Use maintenance *exclusions* (or blue-green) to freeze upgrades during long runs, and annotate long-checkpoint Pods `safe-to-evict: "false"`. |
| "Cluster Autoscaler will bring up the nodes for our multi-host job." | CA scales incrementally; a gang job partially schedules, holds the nodes, and deadlocks. Drive atomic scale-up with `ProvisioningRequest` + Kueue so the whole slice arrives all-or-nothing. |
| "GPUDirect/TCPX/RDMA is an optimization we can add later." | Multi-NIC, the right node image/drivers, and the NCCL plugin are node-pool *creation* decisions — retrofitting is painful. Without them, collectives run over the default Pod network and you waste most of the FLOPs you paid for. Validate with an all-reduce benchmark before declaring done. |
| "GCS FUSE / plain PD is fine for loading weights and datasets." | Naive FUSE on small files and cold-GCS re-downloads on every replica make autoscaling slow and expensive. Match storage to access pattern: Hyperdisk ML (read-only multi-attach for one model on a big fleet), FUSE + local-SSD cache (churny/many models), Parallelstore (parallel scratch), always with image streaming. |
| "A public cluster is easier to reach." | Public node IPs widen the attack surface for no benefit. Use a private cluster (no public node IPs, private control-plane endpoint + authorized networks, Cloud NAT egress) — the production/regulated default. |

## Red flags

Stop and reconsider if you see any of these:

- **SA JSON keys mounted in Pods** (or any credential file under `/var/secrets`) instead of Workload
  Identity Federation.
- **Public cluster / nodes with public IPs** for a production or data-sensitive workload — no private
  endpoint, no authorized networks.
- **No release channel** (statically pinned version) and **no maintenance window/exclusion plan** — you
  will drift onto unsupported versions or get drained mid-training.
- **Accelerator nodes without the interconnect configured** — H100/H200/B200 or multi-host TPU running
  collectives over the default Pod network (no GPUDirect TCPX/TCPXO/RDMA, no NCCL plugin, or a TPU
  topology that doesn't match the parallelism mapping). MFU/all-reduce never benchmarked.
- **Gang/multi-host job without `ProvisioningRequest` + Kueue** — relying on plain CA for all-or-nothing
  capacity.
- **A Service/Ingress/Gateway stuck in `Terminating`** — suspect a stuck LB/NEG finalizer / orphaned GCP
  backend. Do **not** force-remove the finalizer (you leak forwarding rules/backend services you keep
  paying for); fix the GCP resource first.
- **GPU sharing (time-sharing/MPS) on latency-critical or training workloads** instead of MIG/DRA or
  whole devices — noisy-neighbor stalls with no memory isolation.
- **Spot pools without taints and without checkpointing** — preemption attracts intolerant workloads and
  destroys uncheckpointed runs.
- **Broad IAM (`container.admin`) granted to dodge RBAC setup** — in-cluster authorization still needs
  RBAC; map Google Groups → RBAC.
- **No VPC-native CIDR / `max-pods-per-node` plan** on a large cluster — secondary ranges exhaust
  silently and Pods stop scheduling.

## Verification gate (definition of done)

Before the cluster/workload counts as production-ready:

- [ ] **Identity:** Workload Identity Federation is the *only* path to GCP APIs — confirm **no** SA JSON
  keys are mounted. KSAs bound least-privilege per namespace/workload. Human access via Google Groups →
  RBAC (not `container.admin`).
- [ ] **Network & security posture:** VPC-native with a CIDR / `max-pods-per-node` plan that survives max
  scale; Dataplane V2 enabled with NetworkPolicy; private cluster (private endpoint + authorized
  networks, Cloud NAT egress); Shielded nodes; container-native LB via NEGs; Security Posture dashboard
  on. Untrusted/agent code runs under a `gvisor` RuntimeClass.
- [ ] **Node pools right-sized for accelerators:** one pool per hardware/role; correct machine family +
  GPU/TPU SKU for the workload; GPUDirect (TCPX/TCPXO/RDMA) or correct TPU topology + multi-host
  placement configured at creation and **validated with an all-reduce / MFU benchmark**; Spot pools
  tainted and paired with checkpointing.
- [ ] **Autoscaling + capacity:** CA bounds set per pool; NAP limits/defaults configured if used; gang/
  multi-host jobs scale via `ProvisioningRequest` + Kueue (atomic); long-checkpoint Pods annotated
  `safe-to-evict: "false"`; accelerator pools pinned to zones with the SKU/quota.
- [ ] **Operations:** subscribed to a release channel with a maintenance window **and exclusions** for
  long runs; node upgrade strategy chosen per pool (surge vs blue-green); Backup for GKE scheduled if
  stateful.
- [ ] **Observability:** Cloud Logging/Monitoring on (control-plane logs enabled for prod); Managed
  Service for Prometheus scraping app/ML metrics via `PodMonitoring`/`ClusterPodMonitoring`; accelerator
  metrics flowing (GPU `duty_cycle` / TPU `tensorcore_utilization`); per-team attribution clean via
  `namespace_name`.
- [ ] **Cost reviewed:** CUDs for steady accelerator/CPU; Spot where fault-tolerant; NAP/Autopilot to
  eliminate idle headroom; Hyperdisk IOPS/throughput right-sized; cost-allocation/usage-metering enabled
  by namespace/label.

---

## Canonical references (verify current)

- GKE docs: https://cloud.google.com/kubernetes-engine/docs
- Autopilot overview: https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview
- About GPUs in GKE: https://cloud.google.com/kubernetes-engine/docs/concepts/gpus
- About TPUs in GKE: https://cloud.google.com/kubernetes-engine/docs/concepts/tpus
- Node Auto-Provisioning: https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning
- Custom compute classes: https://cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes
- Dataplane V2: https://cloud.google.com/kubernetes-engine/docs/concepts/dataplane-v2
- Container-native LB (NEGs): https://cloud.google.com/kubernetes-engine/docs/concepts/container-native-load-balancing
- GKE Gateway API: https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api
- GPUDirect & multi-networking: https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx
- Workload Identity Federation for GKE: https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity
- GCS FUSE CSI driver: https://cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver
- Hyperdisk ML: https://cloud.google.com/kubernetes-engine/docs/how-to/storage-hyperdisk
- GKE Sandbox (gVisor): https://cloud.google.com/kubernetes-engine/docs/how-to/sandbox-pods
- gVisor project: https://gvisor.dev/docs/
- run:ai Model Streamer: https://github.com/run-ai/runai-model-streamer
- Multi-Tier Checkpointing (verify current naming/availability): https://cloud.google.com/kubernetes-engine/docs
- Parallelstore CSI: https://cloud.google.com/parallelstore/docs/gke
- Managed Service for Prometheus: https://cloud.google.com/stackdriver/docs/managed-prometheus
- GPU/TPU metrics & observability: https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics
- Release channels: https://cloud.google.com/kubernetes-engine/docs/concepts/release-channels
- Backup for GKE: https://cloud.google.com/kubernetes-engine/docs/add-on/backup-for-gke/concepts/backup-for-gke
- Fleets / Config Sync / Config Connector: https://cloud.google.com/kubernetes-engine/fleet-management/docs
- AI Hypercomputer: https://cloud.google.com/ai-hypercomputer/docs
- ProvisioningRequest / Kueue on GKE: https://cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest

---

# GKE Worked Examples

Annotated, correct-in-shape artifacts to imitate. **Verify machine types, accelerator SKUs, topologies,
regions, and flags against current GKE docs and your project's quota before running** — names and
defaults change. Placeholders use `<ANGLE_BRACKETS>`.

---

## 1. Create accelerator node pools (`gcloud container node-pools create`)

### 1a. Multi-host TPU v5p slice (Standard)

```bash
# A multi-host TPU v5p slice node pool. The node pool IS the slice: machine type + topology
# determine chips and host count. 2x2x4 (= 16 chips) over multiple hosts connected by ICI.
# All hosts schedule all-or-nothing; pair with JobSet/LWS + Kueue for the workload.  [[kueue-advanced]]
gcloud container node-pools create tpu-v5p-16 \
  --cluster=<CLUSTER> \
  --location=<REGION> \                      # regional cluster
  --node-locations=<ZONE> \                  # pin to the single zone that has the SKU + quota
  --machine-type=ct5p-hightpu-4t \           # v5p host type (4 TPU chips/host) — VERIFY current name
  --tpu-topology=2x2x4 \                      # slice topology -> total chips; must match the machine type
  --num-nodes=4 \                             # hosts in the slice (chips/topology ÷ chips/host)
  --placement-type=COMPACT \                  # contiguous placement for tight ICI
  --node-labels=team=research \
  --enable-gvnic                              # gVNIC for high-throughput networking

# Pods then select the slice via labels and request chips/host:
#   nodeSelector:
#     cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
#     cloud.google.com/gke-tpu-topology: 2x2x4
#   resources: { limits: { google.com/tpu: "4" } }   # chips per host
```

### 1b. H100 GPU node pool with GPUDirect-TCPX (Standard)

```bash
# A3 (8x H100). For collective-heavy training you also wire multi-networking + the NCCL/TCPX
# installer DaemonSet (separate step) — the pool just provides the hardware + extra NICs + driver.
gcloud container node-pools create a3-h100 \
  --cluster=<CLUSTER> \
  --location=<REGION> \
  --node-locations=<ZONE> \                  # accelerator pools are typically single-zone
  --machine-type=a3-highgpu-8g \             # 8x H100 80GB — VERIFY current family/name
  --accelerator=type=nvidia-h100-80gb,count=8,gpu-driver-version=latest \  # GKE-managed driver
  --ephemeral-storage-local-ssd=count=16 \   # local SSD for scratch/checkpoints + FUSE cache
  --num-nodes=2 \
  --enable-autoscaling --min-nodes=0 --max-nodes=8 \
  --node-taints=nvidia.com/gpu=present:NoSchedule \  # only GPU-tolerating Pods land here
  --node-labels=accelerator=h100 \
  --image-type=COS_CONTAINERD                # Container-Optimized OS

# Spot variant for fault-tolerant training: add `--spot` and a Spot taint; checkpoint frequently.
# Prefer Node Auto-Provisioning + a Custom Compute Class for graceful fallback instead of pinning.
```

> On **Autopilot** you don't create node pools — you request the accelerator on the Pod
> (`cloud.google.com/gke-accelerator: nvidia-h100-80gb`, `nvidia.com/gpu`, optional
> `cloud.google.com/gke-spot`) and GKE provisions the node. A **Custom Compute Class** expresses
> priority/fallback across GPU types in both modes.

---

## 2. Workload Identity Federation for GKE (the right way to call GCP APIs)

No SA keys. Bind a Kubernetes ServiceAccount (KSA) to an IAM identity, then grant that identity the GCP
roles it needs.

```bash
# 0) Cluster must have Workload Identity enabled (default on Autopilot):
#    gcloud container clusters update <CLUSTER> --location <REGION> \
#      --workload-pool=<PROJECT_ID>.svc.id.goog

PROJECT_ID=<PROJECT_ID>
NAMESPACE=research
KSA=trainer                 # Kubernetes ServiceAccount the Pods run as
GSA=trainer-gsa             # (optional) IAM service account to impersonate

# 1) Create the KSA
kubectl create serviceaccount $KSA --namespace $NAMESPACE

# 2a) DIRECT binding (no GSA): grant the KSA principal a role directly. Preferred when possible.
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --role=roles/storage.objectViewer \
  --member="principal://iam.googleapis.com/projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA}"

# 2b) OR impersonate an IAM SA (when you need to reuse an existing GSA's grants):
gcloud iam service-accounts create $GSA
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --role=roles/storage.objectViewer \
  --member="serviceAccount:${GSA}@${PROJECT_ID}.iam.gserviceaccount.com"
# Let the KSA impersonate the GSA:
gcloud iam service-accounts add-iam-policy-binding ${GSA}@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA}]"
# And annotate the KSA to point at the GSA:
kubectl annotate serviceaccount $KSA --namespace $NAMESPACE \
  iam.gke.io/gcp-service-account=${GSA}@${PROJECT_ID}.iam.gserviceaccount.com
```

```yaml
# 3) Run the Pod as that KSA — it gets GCP credentials via the metadata server, no keys mounted.
apiVersion: v1
kind: Pod
metadata:
  name: trainer
  namespace: research
spec:
  serviceAccountName: trainer        # <- the KSA bound above
  containers:
    - name: trainer
      image: <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/trainer:latest
      # SDKs/gcloud/gsutil inside here authenticate as the bound identity automatically.
```

---

## 3. GCS FUSE CSI volume on a Pod (datasets / checkpoints)

Mount a GCS bucket as a volume. Requires the **GCS FUSE CSI driver addon**
(`--addons GcsFuseCsiDriver`) and bucket access via **Workload Identity** (section 2) — grant the KSA
`roles/storage.objectAdmin` (or narrower) on the bucket.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fuse-trainer
  namespace: research
  annotations:
    gke-gcsfuse/volumes: "true"                 # opt the Pod into the CSI sidecar injection
    # Optional sidecar sizing for heavy I/O:
    gke-gcsfuse/cpu-limit: "0"                   # "0" = unlimited; tune for throughput
    gke-gcsfuse/memory-limit: "0"
    gke-gcsfuse/ephemeral-storage-limit: "0"     # cache can be large; back ephemeral with local SSD
spec:
  serviceAccountName: trainer                    # Workload Identity → bucket access, no keys
  containers:
    - name: trainer
      image: <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/trainer:latest
      volumeMounts:
        - name: data
          mountPath: /data
          readOnly: true                         # datasets read-only
        - name: ckpt
          mountPath: /checkpoints                 # checkpoints read-write
  volumes:
    - name: data
      csi:
        driver: gcsfuse.csi.storage.gke.io
        readOnly: true
        volumeAttributes:
          bucketName: <DATASET_BUCKET>
          mountOptions: "implicit-dirs"           # plus caching opts for throughput, e.g.:
          # file caching / metadata caching dramatically speed small-file dataset reads —
          # set fileCacheCapacity / metadataCacheTtlSecs (and a local-SSD-backed cache dir) as needed.
          fileCacheCapacity: "100Gi"
          metadataStatCacheCapacity: "-1"         # -1 = unbounded stat cache (tune to dataset)
    - name: ckpt
      csi:
        driver: gcsfuse.csi.storage.gke.io
        volumeAttributes:
          bucketName: <CHECKPOINT_BUCKET>
          mountOptions: "implicit-dirs"
```

> For "100 nodes load the same 200GB checkpoint fast," prefer **Hyperdisk ML** (read-only multi-attach)
> over cold GCS FUSE reads. For high-throughput parallel scratch, use **Parallelstore CSI**. Stage
> in-step scratch and checkpoint writes on **local SSD**, then async-flush to GCS.

---

## 4. PodMonitoring for Managed Service for Prometheus (per-team metrics)

Scrape an ML serving/training Pod's `/metrics` with Managed Prometheus. Metrics land on the
`k8s_container` resource carrying `namespace_name` → per-team attribution.

```yaml
apiVersion: monitoring.googleapis.com/v1
kind: PodMonitoring
metadata:
  name: vllm-metrics
  namespace: serving           # scoped to this namespace -> namespace_name attribution
spec:
  selector:
    matchLabels:
      app: vllm
  endpoints:
    - port: metrics            # named container port exposing Prometheus metrics
      interval: 15s
      path: /metrics
# Use ClusterPodMonitoring (cluster-scoped) for fleet/infra exporters like DCGM.
# Enable DCGM (GPU) / TPU metrics on the cluster for accelerator signals:
#   GPU -> duty_cycle ;  TPU -> tensorcore_utilization  (different names, same intent).
```
