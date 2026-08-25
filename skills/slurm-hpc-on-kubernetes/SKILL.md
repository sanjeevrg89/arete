---
name: slurm-hpc-on-kubernetes
description: Expert knowledge of running Slurm and HPC-style batch/tightly-coupled workloads on or
  alongside Kubernetes, and choosing between the two worlds. Use when the task involves Slurm
  (slurmctld/slurmd/slurmdbd, sbatch/srun/salloc, #SBATCH, partitions, QOS, fair-share, backfill, gang
  scheduling), Slurm-on-K8s operators (Slinky/slurm-operator, SUNK), HPC schedulers run on K8s (Volcano,
  Kueue, coscheduling, YuniKorn, Flux Framework / Flux Operator / MiniCluster for hierarchical graph-based
  scheduling and co-scheduling), MPI on K8s (MPI Operator / MPIJob, Horovod, PMIx, mpirun, NCCL), HPC
  interconnect on cloud (InfiniBand, RDMA/RoCE, GPUDirect, SR-IOV, topology-aware placement), or
  deciding Slurm vs Kubernetes for an org and migrating between them. Covers why gang/all-or-nothing,
  queueing, fair-share and quota matter and where vanilla K8s pod-by-pod scheduling falls short.
---

# Slurm & HPC on Kubernetes

Apply the judgment of an engineer who has run both a production Slurm cluster and a multi-tenant GPU
Kubernetes platform, and has bridged the two — bursting Slurm into K8s, or replacing it with
Kueue/Volcano — for years. The hard part is rarely YAML; it is **gang semantics, fair-share/quota,
and the interconnect**. Get those right and most else follows.

## How to use this skill

1. **Read `slurm-hpc-on-kubernetes-guide.md`** in this directory — the full reference (Slurm
   architecture, why HPC schedulers exist, Slinky/SUNK, Volcano/Kueue/YuniKorn, MPI Operator, RDMA,
   decision guidance, troubleshooting). Apply it to the task.
2. For concrete artifacts to imitate — a multi-node GPU `sbatch` script and its K8s equivalents
   (MPIJob, Kueue + JobSet) annotated with the HPC↔K8s mapping — read **`examples.md`**.
3. Match the surrounding cluster/scheduler conventions (which scheduler, which CRDs are installed,
   how RDMA is exposed); apply the gang-scheduling, topology, and quota correctness rules regardless.
4. Be version-careful: Slinky, Volcano, Kueue and the MPI Operator move fast. Don't assert a flag,
   field, or capability you're unsure of — describe the concept and say to verify against current
   project docs.

## Essentials (full detail in `slurm-hpc-on-kubernetes-guide.md`)

- **Slurm's model is all-or-nothing and queue-first.** A job (`sbatch`) requests N nodes × resources;
  the controller (`slurmctld`) holds it in a partition queue until the *whole* allocation is grantable,
  then launches all tasks together (gang). Vanilla K8s schedules **pod-by-pod**, so a multi-pod job can
  half-start and deadlock holding GPUs. Closing that gap is the entire point of Volcano/Kueue/coscheduling.
- **Four things Slurm gives you that the default K8s scheduler does not:** (1) gang / all-or-nothing
  scheduling, (2) hierarchical **fair-share** so heavy users yield to idle ones over time, (3) a real
  **queue** with backfill (small jobs fill holes ahead of a big reservation without delaying it), and
  (4) **preemption-by-quota/priority**. K8s needs add-ons for all four.
- **On K8s, pick the scheduler to fit the gap.** **Kueue** = quota + queueing + gang admission on top of
  the *default* scheduler (cloud-native, JobSet/MPIJob/RayJob/batch Job aware, MultiKueue, TAS). See
  `[[kueue-advanced]]`. **Volcano** = a *replacement* scheduler with gang, queues, fair-share,
  reclaim/preempt, and a built-in MPI/PyTorch plugin (closest feel to Slurm). **Coscheduling plugin** =
  minimal PodGroup gang on the default scheduler. **YuniKorn** = hierarchical-queue scheduler. Don't run
  two gang schedulers fighting over the same pods.
- **Flux Framework / Flux Operator = HPC-native scheduling *on* K8s.** Flux is a fully hierarchical,
  graph-based scheduler (a modern successor to Slurm for many cases): nested/recursive scheduling and
  fine-grained graph-based resource modeling (`fluxion`), not flat resource counts. The **Flux Operator**
  runs it as a **MiniCluster** — a pod set that boots a Flux instance and co-schedules a tightly-coupled
  MPI ring (all pods must start together). It schedules *inside* the pod set; Kueue/Volcano still admit/
  place that set as a gang, so it's layered with them, not a replacement. Reach for it for hierarchical
  scheduling, complex resource graphs, or HPC-native MPI/co-scheduling; verify specifics against the Flux
  Framework / Flux Operator docs.
- **MPI on K8s = the MPI Operator (`MPIJob`).** It creates a *launcher* + N *worker* pods, wires
  passwordless SSH (or PMIx) and a hostfile, then the launcher runs `mpirun`/`horovodrun` across workers.
  This is the canonical home for Horovod and classic MPI. NCCL is the GPU collective layer underneath;
  MPI/`torchrun` just bootstraps the process group. See `[[training-frameworks]]`, `[[aiml-on-kubernetes]]`.
- **Tightly-coupled GPU jobs live or die on the interconnect.** You need RDMA — **InfiniBand** or
  **RoCEv2** — exposed to pods (RDMA device plugin / SR-IOV / multi-NIC via Multus or GKE's networks),
  **GPUDirect RDMA**, hugepages, and the right NCCL env (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`,
  `NCCL_NET_GDR_LEVEL`). Without it you fall back to TCP and lose most of the cluster's value. On GKE see
  `[[gke-master]]` (GPUDirect-TCPX/TCPXO, gVNIC, compact placement).
- **Topology matters.** Slurm has topology-aware scheduling; the K8s equivalent is Kueue **Topology-Aware
  Scheduling (TAS)** / Volcano network topology / node affinity to a rack/block/superpod. Place all gang
  members close or pay a latency tax on every all-reduce.
- **Slurm-on-K8s: Slinky vs SUNK vs side-by-side.** **Slinky** (SchedMD's `slurm-operator`) runs Slurm
  *as* K8s workloads — `slurmctld`/`slurmd`/`slurmdbd`/login as pods/StatefulSets, with elastic `slurmd`
  for bursting. **SUNK** (CoreWeave) is a similar Slurm-on-K8s stack. **Side-by-side** = both schedulers
  on shared nodes (cgroup/partition fencing) — flexible but the hardest to keep from double-booking.
- **Decision rule of thumb:** keep/choose **Slurm** when researchers already live in `sbatch`, jobs are
  MPI/tightly-coupled, and the scheduler is the product. Choose **K8s + Kueue/Volcano** when you want one
  control plane for training *and* serving *and* services, cloud elasticity, and the ML/operator
  ecosystem. Many shops run **both** and burst. Don't migrate for fashion; migrate for a concrete win.
- **`#SBATCH` ↔ K8s mapping (memorize the spine):** `--nodes` → number of worker pods / JobSet replicas;
  `--ntasks-per-node` → procs per pod (MPI slots) or `nproc_per_node`; `--gpus`/`--gres=gpu:N` →
  `nvidia.com/gpu`; `--partition`/`--qos` → Kueue ClusterQueue / Volcano queue; `--time` →
  `activeDeadlineSeconds`; `--exclusive` → whole-node request / dedicated nodepool; account/fair-share →
  Kueue quotas or Volcano fair-share. `examples.md` shows this concretely.
- **Top failure modes:** *partial gang / deadlock* (no gang admission, or quota lets some pods in and
  starves the rest — fix with PodGroup/`minMember` or Kueue gang); *NCCL hangs* (wrong NIC/HCA, IB not
  in pod, GDR off, MTU/PFC mismatch on RoCE); *jobs queued forever* (quota too small, topology domain
  can't fit the gang, fair-share starvation); *MPI launcher can't reach workers* (SSH/hostfile/DNS,
  headless Service, pods not all Running). Diagnose with `squeue`/`scontrol` on Slurm,
  `kubectl get podgroups`/Workload/Volcano events on K8s.

## Related skills

- `[[kueue-advanced]]` — deep dive on Kueue: ClusterQueue/LocalQueue, cohorts, gang admission, MultiKueue,
  Topology-Aware Scheduling. The cloud-native answer to Slurm queues/quota.
- `[[aiml-on-kubernetes]]` — umbrella for training/inference on K8s/GKE; this skill is its HPC/MPI arm.
- `[[training-frameworks]]` — DDP/FSDP/DeepSpeed/Megatron and how their launchers ride on MPIJob/JobSet.
- `[[jobset-leaderworkerset]]` — JobSet/LWS as the K8s multi-host primitive Kueue admits as a gang.
- `[[autoscaling-kubernetes]]` — provisioning the nodes a queued gang is waiting on (Cluster Autoscaler /
  Karpenter / ProvisioningRequest), and elastic `slurmd` bursting.
- `[[gke-master]]` — GPU/TPU node pools, GPUDirect-TCPX/TCPXO, compact placement, RDMA on GKE.
- `[[kubernetes-internals-expert]]` — the default scheduler's framework (filter/score/permit plugins)
  that gang plugins and Kueue build on.
