# AI/ML on Kubernetes & GKE — Staff ML-Infra Reference

The full ML lifecycle — pretraining, fine-tuning, RL post-training, inference, and agentic serving — run
on Kubernetes and GKE. This guide is the **integrator**: the end-to-end mental model and the K8s-specific
orchestration. It deliberately stops at the boundary of framework internals and routes to the sibling
skills ([[ml-frameworks]], [[training-frameworks]], [[serving-frameworks]], [[kueue-advanced]],
[[jobset-leaderworkerset]], [[gke-master]], [[autoscaling-kubernetes]], [[slurm-hpc-on-kubernetes]]).

> **Version note (it is 2026).** Accelerator SKUs, GKE features, and framework versions move every
> quarter. Hardware names below (H100/H200/B200/GB200, TPU v5e/v5p/v6e Trillium) and GKE features
> (DRA, custom compute classes, DWS, GPUDirect-TCPXO/RDMA, Inference Gateway) are named so you know what
> to look for — **always verify current region/SKU availability, quotas, and API stability in live GKE
> docs and `gcloud` before committing a design.** Never fabricate benchmark numbers; measure on your
> hardware.

---

## 1. Mental model: ML on K8s is scheduling scarce, tightly-coupled accelerators

Three properties make ML workloads different from ordinary microservices, and every design decision flows
from them:

1. **Accelerators are scarce, expensive, and indivisible-ish.** A node-hour of 8×H100 or a TPU v5p slice
   costs orders of magnitude more than a CPU node. Idle accelerators are pure waste, so utilization
   (MFU/goodput) is the top-line metric, not request latency.
2. **The work is gang-coupled.** A distributed training step is a synchronous collective across every
   participating chip. N-1 of N hosts is useless — you need all-or-nothing scheduling, stable identity,
   and a fast intra-job network. This is the opposite of the embarrassingly-parallel, fungible-replica
   assumption baked into Deployments/HPA.
3. **Topology is correctness-adjacent.** Placement determines whether NCCL/ICI collectives run at line
   rate. A "scheduled" job spread across the wrong network domain can be 3× slower or hang.

So the K8s primitives you reach for are not Deployments and Services — they are **Kueue** (queue, quota,
gang admission), **JobSet/LeaderWorkerSet** (multi-pod groups with identity + headless DNS), the device
plugins / DRA (accelerator allocation), and topology-aware placement (contiguous network domains).

---

## 2. Accelerators on K8s / GKE

### GPUs

- **Requesting them.** GPUs surface as an extended resource `nvidia.com/gpu` you put in
  `resources.limits`. You cannot request fractional GPUs through this resource directly — sharing is via
  MIG, time-slicing, or MPS (below). Pods land on GPU nodes via nodeSelectors/labels (on GKE,
  `cloud.google.com/gke-accelerator=nvidia-h100-80gb` etc.) and tolerate the GPU taint
  (`nvidia.com/gpu=present:NoSchedule`).
- **Drivers & device plugin.** On GKE, the **GPU driver installer DaemonSet** and the **NVIDIA device
  plugin** are managed for you (driver version selectable: default/latest). On self-managed clusters use
  the **NVIDIA GPU Operator** (driver, device plugin, DCGM, MIG manager, node feature discovery in one).
  Don't hand-roll drivers.
- **SKUs (verify availability/region).** H100 (80GB HBM3), H200 (141GB HBM3e), B200 / GB200 NVL72
  (Blackwell, NVLink-connected racks). On GKE these map to specific machine families (e.g. A3/A3-Ultra/
  A3-Mega for Hopper, A4/A4X for Blackwell-class) — exact families and their NIC/NVLink topology change;
  check current docs.
- **Sharing one GPU across pods:**
  - **MIG (Multi-Instance GPU):** hardware partitioning (A100/H100/B200) into isolated instances
    (e.g. `1g.10gb`). Best isolation; fixed profiles. Surfaces as `nvidia.com/mig-1g.10gb`.
  - **Time-slicing:** software round-robin; oversubscribes one GPU to many pods with **no memory
    isolation**. Good for dev/inference of small models; dangerous for anything that can OOM a neighbor.
  - **MPS:** concurrent contexts with some spatial sharing; middle ground.
- **DRA (Dynamic Resource Allocation).** The post-device-plugin future: GPUs/TPUs as first-class
  schedulable resources via `ResourceClaim`/`DeviceClass`, enabling richer requests (specific
  partitions, shared claims, attributes). GA-ish in recent K8s; GKE exposes it progressively. Prefer
  DRA where supported for new designs that need fine-grained or shared allocation — but confirm GKE
  support level for your cluster version.

### TPUs (GKE-specific)

- **Requesting them.** TPUs surface as `google.com/tpu`. You select a TPU type and **topology** via node
  labels/nodeSelectors: `cloud.google.com/gke-tpu-accelerator` (e.g. `tpu-v5p-slice`,
  `tpu-v6e-slice`) and `cloud.google.com/gke-tpu-topology` (e.g. `2x2x2`, `4x4x4`, `16x16`). The
  per-pod TPU request must match the per-host chip count of that topology.
- **Single-host vs multi-host slices.** A **single-host** slice fits in one VM (e.g. v5e `2x4` = 8 chips
  on one host) → one pod, like a GPU node. A **multi-host** slice (e.g. v5p `4x4x4`) spans many VMs
  connected by the **ICI** (inter-chip interconnect) mesh; it must be scheduled as a gang and every host
  must be present and contiguous. GKE creates these as a **multi-host TPU node pool** where the hosts of
  one slice scale as a unit.
- **Generations (verify):** v5e (cost/throughput inference + training), v5p (large-scale training), v6e
  "Trillium" (current high-perf gen). Topology shapes and max slice sizes differ per generation.
- **Why it matters for K8s:** you don't manage the ICI mesh, but you must keep the slice whole. Use
  JobSet for multi-host training and LeaderWorkerSet for multi-host inference so the pods share identity
  and the slice stays contiguous; pair with Kueue for gang admission.

### Networking for collectives

This is where training performance is won or lost.

- **GPUDirect-TCPX / TCPXO (GKE):** GPU-to-GPU transfers that bypass host memory copies over multiple
  NICs (A3/A3-Mega/A3-Ultra-class nodes have many NICs). Requires multi-NIC pods (additional networks
  via GKE multi-networking), an injected sidecar/plugin (the `tcpx`/`tcpxo` NCCL plugin and RxDM
  daemon), and **NCCL env tuning** (`NCCL_*` vars for the plugin, channels, algorithms). Without correct
  config you fall back to slow paths.
- **RDMA / RoCE:** Blackwell-class (A4/A4X) and some H200 configs use RDMA over Converged Ethernet for
  the GPU fabric; exposed via multi-networking + the appropriate NCCL/IB transport. Verify the exact
  GKE recipe for your machine family.
- **Multi-NIC pods:** the data-plane NICs are separate from the default pod NIC; declare them with GKE
  network objects (`GKENetworkParamSet` / `Network`) and the `networking.gke.io/...` pod annotations.
- **NCCL/collectives:** the all-reduce/all-gather/reduce-scatter latency is the training tax. Topology-
  aware placement + the right transport + tuned NCCL = high bus bandwidth. TPUs use ICI/DCN collectives
  managed by the runtime (XLA), not NCCL. Defer collective-algorithm depth to [[ml-frameworks]].

### Storage for data & checkpoints

| Need | GKE option | Notes |
|------|-----------|-------|
| Training data, checkpoints, artifacts | **GCS** (via GCS FUSE CSI driver) | Object store; great for large sequential read + checkpoint write. Use the FUSE caching layer. |
| High-throughput shared FS / fast checkpoint | **Parallelstore** (CSI) | Managed DAOS-based parallel FS; high IOPS/BW for many-host checkpoint and data loading. |
| Fast local-attached / block | **Hyperdisk** (ML/Throughput/Balanced), **Local SSD** | Local SSD for scratch & first-tier checkpoint; Hyperdisk for persistent high-BW volumes. |
| Read-mostly dataset broadcast | GCS FUSE + caching, or Parallelstore | Avoid every pod re-pulling from GCS cold. |

Checkpoint I/O is bursty and huge (a frontier model checkpoint is TBs). Multi-tier — write to Local SSD,
async-replicate to Parallelstore/Hyperdisk, then to GCS — so the training step isn't blocked on object
storage. See §5.

---

## 3. Distributed training on K8s

### Parallelism strategies (systems-level; depth in [[training-frameworks]] / [[ml-frameworks]])

You don't pick these in YAML, but they determine the infra ask (chip count, intra- vs inter-host
bandwidth, memory):

- **Data parallel (DDP):** replicate the model, shard the batch, all-reduce gradients. Bandwidth-bound on
  gradient all-reduce.
- **FSDP / ZeRO (sharded data parallel):** shard params/grads/optimizer state across ranks to fit large
  models; more communication (all-gather params per layer). ZeRO stages 1/2/3 = optimizer/+grad/+param
  sharding. The default for large dense models that don't fit one device's memory.
- **Tensor parallel (TP):** split individual layers across devices; very chatty → keep **within a host /
  NVLink/ICI domain**.
- **Pipeline parallel (PP):** split layers into stages across hosts; tolerates lower bandwidth (point-to-
  point), introduces bubbles.
- **Sequence/context parallel (SP/CP):** shard the sequence dimension for long context.
- **Expert parallel (EP):** route MoE experts across devices.

Real frontier runs compose these (e.g. TP within host × PP across hosts × FSDP/DP across the cluster — a
"3D/4D parallelism" mesh). The infra job is to make the placement match the comm pattern: TP/EP inside
the fast domain, DP/PP across it.

### Orchestration primitives

- **Gang / all-or-nothing scheduling + queueing → [[kueue-advanced]].** Kueue admits a workload only when
  the *whole* gang fits within quota, holding it in a ClusterQueue otherwise. This prevents the classic
  deadlock where two half-scheduled jobs each hold accelerators and neither can start. Kueue's
  **Topology-Aware Scheduling (TAS)** places the gang within a network domain (rack/block/superblock).
- **Multi-host group → [[jobset-leaderworkerset]].** **JobSet** models a training run as a set of
  coordinated Jobs (replicated Jobs with shared headless service, stable pod hostnames, success/failure
  policy, startup ordering). It's the canonical K8s object for multi-host training and integrates with
  Kueue. For TPU multi-host, JobSet + the TPU node-pool auto-creates the slice.
- **Kubeflow Trainer (v2) / Training Operator.** Higher-level CRDs (`TrainJob`, runtime templates) that
  wrap PyTorch/JAX distributed training, generating the underlying pods/services and setting rendezvous
  env. Good when you want a training-specific API instead of raw JobSet. Defer config depth to
  [[training-frameworks]].
- **MPI Operator.** For MPI-launched jobs (Horovod, some HPC/NCCL launchers, DeepSpeed via mpirun): a
  launcher pod + worker pods with `mpirun`/`hostfile` wiring. Overlaps with [[slurm-hpc-on-kubernetes]]
  (Volcano/Slinky) for HPC-style scheduling.
- **Ray / KubeRay.** RayCluster/RayJob for frameworks built on Ray (Ray Train, and many RL stacks). Ray
  manages its own actor placement on top of K8s pods; you still gang-schedule the RayCluster via Kueue.

### Rendezvous, identity, host networking

- Distributed init needs a **stable rendezvous**: a leader address + world size + per-pod rank. JobSet/
  LWS give stable pod hostnames via a headless Service so `MASTER_ADDR`/`torchrun --rdzv` /
  `c10d`/`jax.distributed.initialize` can resolve peers. The "rank-0 / leader" pod is index 0 of the
  replicated Job or the LWS leader.
- **`hostNetwork: true`** is common for max network performance (skip the pod-network overlay) and is
  sometimes required by the GPUDirect/RDMA recipe; it also means you manage port collisions. TPU
  multi-host typically uses host networking. Follow the current GKE recipe for your machine type.
- Set `dnsPolicy: ClusterFirstWithHostNet` when using hostNetwork so DNS still resolves.

### Checkpointing strategy

See §5 — it's a fault-tolerance topic, not just a "save the model" topic.

### Fault tolerance & elastic training

- At 1k+ accelerators, **MTBF is measured in hours** — a host, NIC, or GPU XID error *will* interrupt a
  long run. Design so a single failure costs minutes, not the whole run.
- **Restart policy:** JobSet failure policy / `restartPolicy` + a max-restarts budget; on failure the
  gang restarts and resumes from the last checkpoint. GKE/JobSet can restart the whole replicated Job.
- **Elastic training** (`torchrun --nproc/--max-restarts` with elastic agent, TorchElastic) lets the
  world size shrink/grow across membership changes — powerful but app-side; the infra provides the
  signal and the replacement node. Defer the elastic-agent details to [[training-frameworks]].
- **Health checking & repair:** GKE node auto-repair, plus DCGM/XID-based draining of bad GPUs. Pair
  with Kueue so a restarted gang re-admits cleanly.

---

## 4. Inference / serving on K8s

### Single-host vs multi-host

- **Single-host:** model weights + peak KV cache fit one node (e.g. a 70B in FP8 on 8×H100, or many
  models on one node). This is a **Deployment** (or LWS with one worker) behind a Service; scale replicas
  with autoscaling ([[autoscaling-kubernetes]]).
- **Multi-host:** the model is too big for one node (very large dense models, big MoE, or long-context
  KV) → **LeaderWorkerSet (LWS)**. One logical replica = a **leader + N workers** that together hold one
  model shard-set and communicate over the fabric (NCCL/ICI) per token. LWS gives the group stable
  identity and headless DNS, and scales by *group*, not by pod. This is the standard for multi-host vLLM/
  SGLang and multi-host TPU serving. See [[jobset-leaderworkerset]] and `examples.md`.

### What the serving engine implies for infra (engine internals → [[serving-frameworks]])

- **KV cache dominates memory.** Per-request memory grows with context length × batch; PagedAttention/
  block KV management is why vLLM/SGLang exist. Infra impact: HBM headroom, `--gpu-memory-utilization`
  tuning, and OOM if you over-batch.
- **Continuous batching** means utilization and latency depend on in-flight request count, not just RPS —
  so your autoscaling signal should be queue depth / KV-cache utilization / time-to-first-token, not CPU.
- **Prefill/decode disaggregation** (separate prefill and decode pools, e.g. Dynamo/llm-d patterns)
  splits one model into two differently-shaped deployments connected by a KV transfer — more pools to
  schedule and route between.
- **Prefix/KV-cache reuse** rewards routing requests with shared prefixes to the same replica → drives
  **cache-aware routing** (see Inference Gateway below).

### Autoscaling model servers → [[autoscaling-kubernetes]]

- Scale on **model-aware metrics** (KV-cache utilization, pending queue, TTFT/TPOT, batch size) exported
  by the engine, via Custom/External Metrics → HPA, or KEDA. CPU/memory HPA is wrong for GPU serving.
- **Scale-to-zero & cold start** is dominated by **model load time** (pulling tens of GB of weights and
  loading to HBM). Mitigate: weights on a fast volume (Hyperdisk ML / Parallelstore / Local SSD), GCS
  FUSE caching, image/weight pre-pull and node warm-pools, and container/secondary-boot-disk image
  streaming. Plan node provisioning latency too (NAP / DWS / reservations).

### GPU/TPU sharing for serving

- Small models / low QPS: MIG or time-slicing to pack many models per accelerator; or multiple model
  servers per node. Watch memory isolation (time-slicing has none).
- Large models: dedicate whole nodes / multi-host groups.

### Model loading from GCS & artifacts

- Mount weights via **GCS FUSE CSI** (with caching) or stage to a fast local volume at startup via an
  init container. For repeated loads, a node-local cache or Parallelstore beats cold GCS every time.

### GKE Inference Gateway / model routing (concepts)

- The **GKE Inference Gateway** (Gateway API inference extension / `InferencePool` + `InferenceModel`-style
  CRDs) does **model-aware, load-aware routing**: route by model name, do cache/prefix-aware and
  load-aware balancing across replicas, support LoRA-adapter multiplexing and traffic splitting for
  rollouts. Conceptually it's an L7 router that understands LLM semantics (queue depth, KV utilization)
  rather than round-robin. Confirm current API names/versions; the inference-routing space is evolving
  fast (Gateway API Inference Extension, llm-d, Dynamo router).

---

## 5. Checkpointing (training and post-training)

Treat checkpointing as the backbone of goodput, not an afterthought:

- **Tiers:** (1) in-memory/Local SSD for fastest write, (2) Parallelstore/Hyperdisk for fast shared
  durable, (3) GCS for cheap durable / cross-region. Async copy down the tiers so the training step is
  never blocked on the slow tier.
- **Asynchronous & sharded checkpointing:** each rank writes its shard; libraries (PyTorch Distributed
  Checkpoint / DCP, Orbax for JAX, NeMo/Megatron savers) overlap save with compute. Defer the library
  details to [[training-frameworks]] / [[ml-frameworks]].
- **Frequency vs cost:** checkpoint often enough that a restart loses minutes, not hours, but not so
  often that I/O eats goodput. Tie frequency to MTBF and step time.
- **In-memory / peer checkpointing & emergency checkpoint-on-failure** (write a final checkpoint when a
  node signals failure) cut wasted work dramatically at frontier scale.
- **Restore = the resume path of fault tolerance** — test it. A checkpoint you can't reliably restore
  across a changed world size is a liability.

---

## 6. Fine-tuning — sizing the infra to the method

| Method | Trainable params | Memory profile | Typical infra ask |
|--------|------------------|----------------|-------------------|
| **Full fine-tune** | All weights | params + grads + optimizer state (≈ training) | Multi-host, FSDP/ZeRO, same as pretraining at smaller scale |
| **PEFT / LoRA** | Low-rank adapters only | base model frozen (can be quantized-ish), tiny optimizer state | Often **single node**; sometimes one GPU |
| **QLoRA** | LoRA on a 4-bit-quantized base | base in 4-bit + adapters | Smallest; large model on a single/fewer GPUs |

Implications:

- Full fine-tune is a training job — reach for JobSet + Kueue + the multi-host machinery above.
- LoRA/QLoRA often collapses to a single-node Job, and a **fraction of a GPU** (MIG/time-slicing) for
  small bases — so they're great multi-tenant batch workloads. Many adapters can be **served** off one
  base model via adapter multiplexing (Inference Gateway / engine LoRA support).
- The same checkpoint/data storage story applies; artifacts are much smaller for PEFT.

Defer the algorithm/library specifics (PEFT, bitsandbytes, Unsloth, axolotl, TRL SFT) to
[[training-frameworks]].

---

## 7. RL & RLHF / RLAIF on K8s

RL post-training is the **hardest orchestration problem** in this guide because it is inherently
**heterogeneous and multi-component**, mixing inference and training in a tight loop.

### The architecture

A modern RLHF/RLAIF loop has several distinct roles, each with a different accelerator profile:

- **Rollout / actor (generation):** runs *inference* (a vLLM/SGLang-style engine) to sample responses
  from the current policy. Throughput-bound, benefits from inference-optimized accelerators (v5e, fewer
  big GPUs) and continuous batching.
- **Reward model (and/or judge/critic):** scores responses (a forward pass) — another inference-ish
  service; for RLAIF the "reward" may be an LLM judge.
- **Learner / trainer (policy update):** runs *training* (gradients, optimizer) to update the policy
  weights — training-shaped, FSDP/Megatron, NVLink/ICI-heavy.
- **Reference model:** frozen copy for the KL penalty.

### Why it's multi-cluster / heterogeneous

- The rollout fleet and the learner fleet want **different hardware** and scale **independently** (you
  often need many rollout replicas feeding one learner). They have different lifecycles and failure
  domains.
- After each policy update, **new weights must be broadcast** from the learner to every rollout engine
  (weight sync) — a periodic, large, latency-sensitive data movement that dominates the systems design.
- This pushes toward **multiple node pools / clusters** (training cluster + inference cluster), possibly
  across regions, coordinated by [[kueue-advanced]] MultiKueue and a higher-level orchestrator (Ray is
  common, since it can hold actors of different shapes in one logical app).

### Algorithms at a systems level

- **PPO:** classic actor-critic; needs policy + value + reward + reference models live simultaneously →
  most memory-hungry and most components to schedule.
- **DPO:** offline-ish preference optimization; **no online rollout engine** in the basic form → looks
  much more like a normal (full/PEFT) fine-tune job. Cheapest to orchestrate.
- **GRPO** (and group-relative variants): drops the value model, samples groups of responses → heavy
  rollout, lighter learner; popular for reasoning models. Rollout throughput is the bottleneck.

### Frameworks (named at systems level; details → [[training-frameworks]] / [[serving-frameworks]])

- **TRL** — HF library; SFT/DPO/GRPO/PPO; single-node-friendly, scales with accelerate/DeepSpeed.
- **veRL** (Volcano Engine RL) — Ray-based, hybrid-engine, designed for large-scale RLHF with
  colocated/disaggregated rollout+train.
- **NeMo-RL / NeMo-Aligner** — NVIDIA stack, Megatron-backed learner, scales to large models.
- **OpenRLHF** — Ray + vLLM + DeepSpeed, disaggregated rollout/learner.

### Orchestration challenges on K8s

- **Co-location vs disaggregation:** put rollout and train on the same GPUs (time-share, simpler weight
  sync, idle bubbles) or separate pools (better utilization, harder sync). Both are valid; pick per scale.
- **Weight sync mechanism:** NCCL broadcast, shared storage, or engine-specific update APIs — schedule
  the network for it.
- **Gang the whole app:** the learner + rollout + reward set must come up together; Kueue admits the
  combined workload, Ray/JobSet wires the topology.
- **Statefulness & long runs:** RL runs are long, stateful, and bursty — checkpoint the policy *and* the
  optimizer/replay state.

---

## 8. Agentic workloads

Serving agents (tool-use, multi-step reasoning, long-horizon tasks) at scale adds requirements beyond
single-shot inference:

- **Long-running / stateful inference:** an agent turn fans out into many model calls + tool calls with
  conversation/scratchpad state. Sessions are long-lived and sticky → session-aware routing and KV-cache
  reuse across turns matter a lot (prefix caching pays off). Watch for long-tail latency and timeouts.
- **Tool execution & sandboxing:** agents run arbitrary code / hit external systems. Isolate untrusted
  execution in **sandboxes** — gVisor (GKE Sandbox), Kata/microVMs, or per-session ephemeral pods/
  namespaces with tight NetworkPolicy, no node creds (Workload Identity, minimal scopes), seccomp, and
  resource limits. Treat tool sandboxes as a multi-tenant security boundary.
- **Orchestration:** agent frameworks/graphs are often CPU-bound control planes that *call* GPU model
  servers; separate the (cheap, scalable) orchestration tier from the (expensive) inference tier so they
  scale independently. Long-running stateful agents may use StatefulSets / per-session workspaces.
- **Cost:** agent loops amplify token usage; route to right-sized models, cache aggressively, and cap
  fan-out.

---

## 9. Cross-cutting infrastructure

### Data pipelines

- Ingest/preprocess/tokenize with batch jobs (Ray Data, Spark-on-K8s, Beam/Dataflow, or plain Jobs) →
  write sharded datasets (WebDataset/Parquet/Arrow) to GCS/Parallelstore. Decouple data prep from the
  accelerator job; never let GPUs idle on the data loader. Use streaming dataset loaders + prefetch +
  the storage tiering in §2.

### Experiment & pipeline orchestration

- **Kubeflow Pipelines / Argo Workflows** — DAG orchestration of multi-step ML pipelines (data → train →
  eval → deploy). **Ray** — for apps that want a single distributed runtime (Train/Tune/Serve/RLlib).
  Pick one per org; all sit on top of the scheduling primitives in this guide.

### Observability for accelerators

- **GPUs:** **DCGM exporter** → Prometheus: SM/tensor-core occupancy, HBM used/bandwidth, power, temp,
  NVLink throughput, and **XID/SXid errors** (the canary for failing hardware). GKE surfaces accelerator
  metrics and you can run **Managed Prometheus** with **namespace-level accelerator metrics**.
- **TPUs:** TPU runtime/`libtpu` metrics (duty cycle, HBM, tensorcore util) via Managed Prometheus.
- **What to watch:** **MFU** (model FLOPs utilization — achieved vs peak FLOPs) and **goodput** (fraction
  of wall-clock doing useful, non-wasted compute) are the north-star training metrics; for serving watch
  TTFT/TPOT, KV-cache utilization, queue depth, and tokens/sec/accelerator. Alert on XID errors, NCCL
  timeouts, and stalled all-reduce.

### Cost & utilization

- Frontier accelerators are the dominant cost line, so **utilization is cost.** Track MFU/goodput and
  $/token (training) and $/Mtok (serving). Use **CUDs/reservations** for steady base load, **DWS
  (Dynamic Workload Scheduler) flex-start/calendar** for bursty training, **Spot** only for restartable/
  fault-tolerant work, and **bin-pack** with MIG/time-slicing for small workloads. NAP/custom compute
  classes ([[autoscaling-kubernetes]], [[gke-master]]) right-size node pools.

### Multi-tenancy & quota

- **Kueue** ClusterQueues + cohorts give fair-share and borrowing of scarce accelerators across teams;
  this is the primary multi-tenant lever for batch ([[kueue-advanced]]). Per-team **namespaces** +
  **ResourceQuota** + node **taints/labels** isolate workloads; NetworkPolicy and Workload Identity for
  security isolation. Reserve a slice of capacity for interactive/dev to avoid starving experimentation.

---

## 10. AI FinOps — cost & capacity planning

Accelerators are the dominant cost line of any ML platform, so FinOps for ML is not spreadsheet hygiene —
it is a Staff-level engineering concern that shapes scheduling, autoscaling, storage, and serving
architecture. §9's "Cost & utilization" note is the one-liner; this section is the discipline.

### Why it's a Staff concern: utilization is the real KPI

The price tag is set by **accelerator-hours**, not nodes, requests, or CPU. So the metric that matters is
not spend — it's **spend per unit of useful result**:

- **Training:** **MFU** (model FLOPs utilization — achieved vs theoretical-peak FLOPs) and **goodput**
  (fraction of wall-clock doing non-wasted, recoverable compute). A run at 35% MFU costs ~2× a run at 70%
  MFU for the same model. The cost KPI is **$/useful-training-FLOP** or, pragmatically, **$/training-run**
  and **cost-per-experiment**.
- **Inference:** **tokens/$/accelerator-hour** and **$/token** (or $/Mtok) at a target latency SLO. A
  server that is fast but runs at batch=1 can be far more expensive per token than a slightly slower one
  at high continuous-batch occupancy. Always quote unit cost **at an SLO**, never raw throughput.

The trap a Staff engineer must kill org-wide: **optimizing price-per-hour instead of price-per-result.**
A cheaper GPU/hour that yields lower MFU or fewer tokens/sec can cost *more* per result. Decisions are
made on $/result, with utilization as the lever.

### Cost levers (and where each lives in this library)

| Lever | What it does | Where |
|-------|--------------|-------|
| **Right-size the accelerator** | Match SKU to the workload (don't serve a 7B on B200; don't train on inference-tuned chips). Pick the smallest accelerator that hits the SLO/step-time. | §2; `[[gke-master]]` |
| **Spot / preemptible + checkpointing** | Spot/preemptible capacity is far cheaper but can be reclaimed; only viable with frequent, fast checkpoint + auto-resume. Pair Spot with restartable JobSets. | §5; `[[ml-checkpointing-orbax]]`, `[[kueue-advanced]]` |
| **Bin-packing & sharing** | Pack small models / dev / PEFT onto fractional accelerators via **MIG** (isolated) or **time-slicing** (no memory isolation) so one GPU serves many tenants. | §2 |
| **Scale-to-zero for inference** | Idle model servers burn the most expensive resource doing nothing. Scale replicas (and node pools) to zero off-peak; budget for cold-start = weight-load time. | §4; `[[autoscaling-kubernetes]]` |
| **Quota / fair-share** | Cap and share scarce accelerators across teams so no one hoards; borrowing + reclaim keeps utilization high. | `[[kueue-advanced]]` |
| **Commitment posture** | Match purchase mode to demand shape (below). | this section |
| **Inference optimization** | Quantization (FP8/INT8), speculative decoding, prefix/KV-cache reuse, continuous batching, prefill/decode disaggregation — each raises tokens/$/accelerator. | §4; `[[inference-optimization]]` |

**Right-sizing in practice.** The cheapest run is the one that doesn't waste FLOPs: profile MFU before
scaling out, fix the data loader before adding chips (a starved accelerator is 100% waste), and prefer a
smaller contiguous slice at high MFU over a larger sprawling one at low MFU.

**Spot + checkpoint is the single biggest training cost lever** for fault-tolerant work. With
checkpoint-on-preemption and JobSet auto-restart (§5), preemption costs minutes, not the run — so a large
fraction of training can ride cheaper preemptible capacity. Never run Spot without a tested fast-resume
path; never run steady 24×7 base load on Spot.

**Commitment posture — match purchase mode to demand shape:**

- **Committed-use / reservations** — for **steady, predictable base load** (always-on serving, long
  multi-month pretraining). Lowest unit price in exchange for a commitment; an empty reservation is pure
  burn (see anti-patterns).
- **On-demand** — for **unpredictable or short** needs and headroom above the committed base. Highest unit
  price; pay only for what you use.
- **Spot / preemptible** — for **interruptible, checkpointed** work (most training, batch eval, PEFT,
  offline data prep). Cheapest, but reclaimable.
- **Calendar/queued reservations (e.g. DWS-style flex-start/calendar on GKE)** — for **bursty large
  gangs** you can wait for: get a guaranteed contiguous block at a scheduled time without holding it 24×7.

Blend them: a reserved/committed base for steady load, on-demand for headroom, Spot for the
restartable bulk, and queued reservations for big scheduled gangs. **Verify current pricing, discount
rates, commitment terms, and SKU/region availability against live cloud pricing/docs — these move every
quarter; do not hardcode rate assumptions.**

### Capacity planning

Capacity planning answers "how many accelerators, of which kind, for how long, where?" *before* you
commit budget — and it's an estimate you refine with measured MFU/throughput.

**Training — estimate accelerator-hours from FLOPs.** The textbook approximation for a dense transformer
is **compute ≈ 6 × parameters × training-tokens** FLOPs (forward+backward). Then:

```
accelerator-hours  ≈  (6 × params × tokens)  /  (peak_FLOPs_per_accelerator × MFU × 3600)
wall-clock-hours   ≈  accelerator-hours / number_of_accelerators
estimated cost     ≈  accelerator-hours × blended_$_per_accelerator_hour
```

- Use a **realistic MFU** (measure it on a short pilot — don't assume peak; large dense runs commonly land
  well below peak, and the exact figure is hardware/parallelism-specific — measure, don't fabricate).
- Add **goodput overhead**: restart/recovery, checkpoint I/O, stragglers, and warm-up. A run is *never*
  100% goodput; budget the gap explicitly.
- Re-derive after a pilot run; the pilot's measured tokens/sec is worth more than any a-priori estimate.

**Inference — throughput planning (QPS × tokens → accelerators).**

```
required_tokens_per_sec ≈ peak_QPS × (avg_input + avg_output tokens) × safety_margin
accelerators            ≈ required_tokens_per_sec / (measured tokens/sec/accelerator at target SLO)
```

- The denominator (**tokens/sec/accelerator at the SLO**) must be **measured** under realistic batching and
  context length — it varies enormously with model, quantization, batch, and TTFT/TPOT targets.
- Plan **headroom** for peak-vs-average traffic, failover, and rollout (surge) — a fleet sized to average
  QPS will fall over at peak. Size to peak × margin, then claw back idle cost with scale-to-zero/autoscaling.
- Separate **prefill** and **decode** capacity if you disaggregate; they scale on different signals.

**Cross-cutting capacity constraints:**

- **Quota & region.** Accelerator quota is per-region and often the *real* limiting reserve, not budget.
  Multi-region spreads risk and unlocks capacity but adds data-gravity, egress, and weight-sync cost.
- **Contiguity.** A capacity number is meaningless if it can't be placed as one contiguous topology — a
  multi-host gang needs a contiguous network domain (Topology-Aware Scheduling, `[[kueue-advanced]]`), not
  just N free chips scattered across the fleet.
- **Buy-vs-rent / cloud-vs-on-prem.** Steady, high-utilization, multi-year base load can favor owned/
  on-prem or long commitments; bursty, uncertain, or fast-evolving (chase-the-latest-SKU) demand favors
  cloud on-demand/Spot/reservations. The break-even hinges on **realized utilization** — owned hardware at
  30% utilization is usually worse than rented at 80%. Frame it as a utilization-and-horizon decision, not
  a sticker-price one.

### Measurement & accountability

You cannot manage what you don't attribute. The platform must answer "what did each team/run/experiment
cost, and what did it return?"

- **Cost attribution by team/namespace.** Accelerator utilization metrics (DCGM for GPUs, TPU runtime
  metrics; §9) are exported on the **`k8s_container`** monitored resource, which carries
  **`namespace_name`** (and pod/container labels). Joining accelerator-hours per namespace to the
  accelerator's cost rate gives **per-team cost** without bespoke metering. Enforce one-team-per-namespace
  (or per-label) discipline so the attribution is clean; mirror it in Kueue ClusterQueue/cohort names so
  quota and cost line up.
- **Showback vs chargeback.** *Showback* reports each team's spend for visibility; *chargeback* actually
  bills it back. Showback first to build the culture, chargeback once attribution is trusted — chargeback
  is what makes teams turn off idle reservations.
- **Unit economics.** Track and publish **$/token** (serving), **$/training-run** and
  **cost-per-experiment** (training/research), and **tokens/$/accelerator-hour**. Unit economics — not raw
  spend — is what tells you whether scaling up is healthy or wasteful.
- **Dashboards & alerts.** A FinOps dashboard pairs **utilization** (MFU/goodput, KV-cache utilization,
  tokens/sec/accelerator) with **cost** (per namespace/queue, per model, per run) on the same pane, plus
  **idle-accelerator** and **reservation-utilization** panels. Alert on low utilization of *committed*
  capacity (you're paying for it regardless) and on cost-per-token regressions after a deploy. Defer
  metric pipelines/dashboards depth to `[[ml-observability-monitoring]]`.

### Anti-patterns

- **Idle reserved/committed accelerators.** Paying the commitment price for capacity that sits empty — the
  worst of both worlds (committed *and* unused). Right-size commitments to measured steady demand; fill
  reservations with preemptible batch backfill.
- **No utilization target.** Running with no MFU/goodput or tokens/sec/accelerator SLO means no one owns
  waste. Set a target (and alert below it).
- **Optimizing price-per-hour, not price-per-result.** Chasing the cheapest GPU/hour while ignoring that it
  yields lower MFU or fewer tokens/sec — often a net loss per result.
- **Spot without checkpoint.** Running on preemptible capacity with no fast checkpoint/resume — a
  preemption then throws away hours of work, erasing the discount many times over.
- **No cost attribution.** A single shared namespace / no team labels → no one can see or own their spend,
  so nobody optimizes. The tragedy of the commons on the most expensive resource you have.
- **Over-provisioned inference.** A fleet sized to peak (or worse, to a vanity SLO) that never scales down,
  burning accelerators at low occupancy off-peak. Use scale-to-zero/model-aware autoscaling and right-size
  to measured load + headroom.

---

## 11. Decision guide — which sibling skill / tool for which problem

| You are doing... | Reach for | Sibling skill |
|------------------|-----------|---------------|
| Choosing GKE node pools, GPU/TPU machine families, networking, security, DWS/reservations | GKE node-pool & cluster design | `[[gke-master]]` |
| Gang scheduling, quota/fair-share, queueing, MultiKueue, topology-aware placement | Kueue | `[[kueue-advanced]]` |
| Modeling a multi-host **training** group (stable identity, headless DNS, restart policy) | JobSet | `[[jobset-leaderworkerset]]` |
| Modeling a multi-host **inference** group (leader+workers, scale by group) | LeaderWorkerSet | `[[jobset-leaderworkerset]]` |
| PyTorch/JAX/XLA internals, GPU/TPU programming, NCCL/collective algorithms | Framework internals | `[[ml-frameworks]]` |
| FSDP/ZeRO/DeepSpeed/Megatron/NeMo/Ray Train/Kubeflow Trainer/MaxText config | Training stacks | `[[training-frameworks]]` |
| vLLM/SGLang/Dynamo/Triton/TensorRT-LLM/Ray Serve/KServe engine config | Serving stacks | `[[serving-frameworks]]` |
| HPA/VPA/Cluster Autoscaler/Karpenter/KEDA/NAP for model servers or training | Autoscaling | `[[autoscaling-kubernetes]]` |
| Slurm/Slinky/Volcano/MPI/RDMA, or "Slurm vs K8s for this run?" | HPC scheduling | `[[slurm-hpc-on-kubernetes]]` |
| General K8s mechanics (objects, RBAC, networking, debugging) | Core K8s | `[[kubernetes-expert]]` |

**Quick routing heuristics:**

- *"My multi-host job half-schedules and deadlocks"* → gang admission in [[kueue-advanced]].
- *"NCCL is slow / collectives hang"* → networking §2 + topology (TAS in [[kueue-advanced]]), NCCL/
  transport tuning in [[ml-frameworks]].
- *"Model too big for one node to serve"* → LeaderWorkerSet ([[jobset-leaderworkerset]]) + multi-host
  vLLM ([[serving-frameworks]]).
- *"Serving autoscaling reacts wrong"* → model-aware metrics in [[autoscaling-kubernetes]].
- *"Standing up RLHF"* → §7 here for the architecture; learner config in [[training-frameworks]], rollout
  engine in [[serving-frameworks]], multi-cluster in [[kueue-advanced]] (MultiKueue).
- *"Should this run on Slurm instead?"* → [[slurm-hpc-on-kubernetes]].

---

## 12. Canonical references (verify currency — 2026)

- GKE GPUs: https://cloud.google.com/kubernetes-engine/docs/how-to/gpus
- GKE TPUs: https://cloud.google.com/kubernetes-engine/docs/concepts/tpus
- GKE AI/ML orchestration: https://cloud.google.com/kubernetes-engine/docs/integrations/ai-infra
- GPUDirect / multi-networking on GKE: https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx
- Kueue: https://kueue.sigs.k8s.io/
- JobSet: https://jobset.sigs.k8s.io/
- LeaderWorkerSet: https://lws.sigs.k8s.io/
- Kubeflow Trainer: https://www.kubeflow.org/docs/components/trainer/
- NVIDIA GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/
- DCGM / DCGM-Exporter: https://github.com/NVIDIA/dcgm-exporter
- Kubernetes DRA: https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/
- vLLM: https://docs.vllm.ai/ · SGLang: https://docs.sglang.ai/
- Gateway API Inference Extension: https://gateway-api-inference-extension.sigs.k8s.io/
- GKE Inference Gateway: https://cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway
- Ray on Kubernetes (KubeRay): https://docs.ray.io/en/latest/cluster/kubernetes/index.html
- TRL: https://huggingface.co/docs/trl · veRL: https://github.com/volcengine/verl ·
  OpenRLHF: https://github.com/OpenRLHF/OpenRLHF · NeMo-RL: https://github.com/NVIDIA-NeMo/RL
- PyTorch Distributed Checkpoint: https://pytorch.org/docs/stable/distributed.checkpoint.html ·
  Orbax: https://orbax.readthedocs.io/
