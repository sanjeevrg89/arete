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

---

# Reference — slurm-hpc-on-kubernetes

# Slurm & HPC on Kubernetes — Reference Guide

The authoritative reference for this skill. Two scheduling worlds — the HPC batch world (Slurm) and
the cloud-native world (Kubernetes) — are converging on the same hardware: GPU/accelerator clusters
with RDMA fabrics running large, tightly-coupled jobs. This guide gives you enough Slurm to be
dangerous, explains *why* HPC schedulers exist and where K8s falls short, then covers the bridges
(Slinky/SUNK), the K8s-native HPC schedulers (Volcano/Kueue/coscheduling/YuniKorn), MPI on K8s, the
interconnect story, and how to choose.

Ecosystem caveat: it is 2026 and Slinky, Volcano, Kueue, and the MPI Operator all ship frequently.
Treat specific CRD fields, flags, and capabilities below as "verify against current project docs"
unless you've confirmed them on your cluster. Concepts are stable; field names drift.

---

## 1. Slurm fundamentals an infra engineer must know

**Slurm** (Simple Linux Utility for Resource Management) is the dominant open-source HPC scheduler. Its
mental model is the inverse of Kubernetes: Kubernetes is a *declarative, always-on* system where you
post desired state and controllers converge to it; Slurm is a *batch queue* where you submit jobs that
wait, run to completion, and release resources. There is no "deployment that stays up" — there are jobs.

### Architecture (the daemons)

- **`slurmctld`** — the central controller. One active (optionally with a backup for HA). Owns the
  queue, makes all scheduling decisions, tracks node/job state, enforces limits. The single brain. Loose
  analog: the kube-scheduler + a chunk of the API server, fused.
- **`slurmd`** — node daemon, one per compute node. Launches and monitors job steps, sets up cgroups,
  reports node state. Analog: kubelet.
- **`slurmstepd`** — per-step shepherd spawned by `slurmd` to actually run a job step's tasks.
- **`slurmdbd`** — the accounting/database daemon (fronts MySQL/MariaDB). Stores job accounting,
  associations, QOS, fair-share usage, limits. Required for fair-share, QOS, and per-account limits.
- **`slurmrestd`** — optional REST API gateway. Login/submit nodes run the client tools (`sbatch`,
  `srun`, `squeue`, `scontrol`, `sinfo`, `sacct`).

Config is files, classically `slurm.conf` (+ `gres.conf`, `topology.conf`, `cgroup.conf`,
`acct_gather.conf`). Modern Slurm also supports `configless` mode where `slurmd` pulls config from
`slurmctld`. There is no etcd; state is in the controller's memory + a state-save directory + the
accounting DB.

### Partitions and nodes

A **partition** is a named queue mapped to a set of nodes with policies (time limits, allowed accounts/
QOS, priority, preemption, `OverSubscribe`). Think "logical pool with rules" — loosely a K8s
nodepool + namespace-quota + priorityclass rolled together. A node can belong to multiple partitions.
**GRES** (Generic Resources) model GPUs and other devices (`gres=gpu:a100:8`), analogous to K8s
extended resources (`nvidia.com/gpu`).

### The job model

- **`sbatch script.sh`** — submit a *batch* job (the common case). The script's `#SBATCH` comment lines
  carry the resource request; the body runs on the first allocated node and uses `srun` to fan work out.
- **`srun`** — launch a *job step* (the parallel tasks) inside an allocation, or run interactively. `srun`
  is also the MPI/parallel launcher: it starts N tasks across the allocated nodes and provides the
  process-management interface (PMIx) MPI needs.
- **`salloc`** — request an interactive allocation; you get a shell with the nodes reserved and run
  `srun` by hand. Analog: an interactive debugging pod, but multi-node.

Key `#SBATCH` directives (the ones that map to K8s):

```bash
#SBATCH --job-name=train
#SBATCH --partition=gpu          # which queue/pool
#SBATCH --qos=high               # quality-of-service tier
#SBATCH --account=research       # for fair-share/accounting
#SBATCH --nodes=8                # N nodes (gang size)
#SBATCH --ntasks-per-node=8      # tasks (MPI ranks) per node -> often = GPUs/node
#SBATCH --gpus-per-node=8        # or --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0                  # 0 = all memory on the node
#SBATCH --time=04:00:00          # wallclock limit (hard kill)
#SBATCH --exclusive              # whole nodes, no co-tenants
#SBATCH --constraint=a100        # feature/label selector
```

The whole allocation is **gang-granted**: all 8 nodes (64 GPUs) are reserved before the job starts, and
all tasks launch together. That is the property vanilla K8s lacks.

### Scheduling features that make Slurm "HPC"

- **Gang / all-or-nothing.** The job runs only when the *entire* request fits. No half-starts.
- **Backfill.** The default plugin (`sched/backfill`) lets smaller/shorter jobs run in the holes ahead
  of a large job's reservation *as long as they don't delay it*. This needs accurate `--time` limits —
  the backfill scheduler reasons about future free windows. Massively improves utilization.
- **Fair-share.** Priority rises for users/accounts who've used less than their share recently and falls
  for heavy users (classically a half-life-decayed usage model, `priority/multifactor`). Hierarchical by
  account tree. This is the feature people miss most when they move off Slurm.
- **QOS.** Named policy tiers (priority boost, limits, preemption rights, partition access). E.g. a
  `high` QOS preempts `normal`.
- **Preemption.** By partition priority or QOS: a higher-priority job can suspend/requeue/kill lower ones
  to make room.
- **Topology-aware scheduling.** `topology.conf` describes the network tree (switches/leaf/spine); Slurm
  prefers allocations that minimize the number of switches a job spans, so all-reduce traffic stays local.
- **Tightly-coupled / MPI.** Slurm + PMIx gives MPI ranks fast bootstrap and wire-up; `srun --mpi=pmix`
  is the canonical launcher. Tightly-coupled means every rank must run simultaneously and they
  communicate constantly — which is exactly why gang + topology + RDMA all matter together.

---

## 2. Why HPC schedulers exist — and where vanilla K8s falls short

Kubernetes' default scheduler is excellent for *services*: it places **one pod at a time**, optimizing
each placement independently, assuming pods are largely independent and long-lived. Batch/HPC breaks
every one of those assumptions. The four canonical gaps:

1. **No gang scheduling.** A distributed training job is M pods that are useless unless *all* M run. The
   default scheduler will happily admit 6 of 8 pods when only 6 GPUs are free; those 6 sit idle holding
   GPUs, the other 2 can't be placed, and if a second job does the same you get **gang deadlock** — both
   jobs half-scheduled, neither able to complete, GPUs wasted. Slurm never does this; it grants the whole
   allocation or nothing.

2. **No real queue / fair-share.** K8s has no notion of "job A waits behind job B in a fair queue." Pods
   are admitted as resources free up, roughly first-come, with no hierarchical fairness, no backfill, no
   per-user/per-account accounting of historical usage. A few heavy users can monopolize a cluster
   indefinitely.

3. **No preemption-by-quota.** K8s PriorityClass preemption is coarse and pod-level; it has no concept of
   "team A is over its quota, reclaim from it to serve team B who is under theirs."

4. **No first-class accounting/quotas for batch.** ResourceQuota is static per-namespace and doesn't
   borrow/lend or track usage over time.

These gaps are *intentional* — they're outside the scheduler's core job — and are filled by add-ons:
**Kueue** (queue + quota + gang admission on top of the default scheduler) and **Volcano** (a full
replacement scheduler with gang/fair-share/reclaim). The coscheduling plugin and YuniKorn are narrower
points on the same spectrum. Understanding *which gap you have* tells you which tool to reach for.

---

## 3. K8s-native HPC/batch schedulers

| Tool | What it is | Gang | Queue/quota | Fair-share | Preempt/reclaim | Replaces default sched? | Best for |
|------|-----------|------|-------------|-----------|------------------|-------------------------|----------|
| **Kueue** | Job-level admission controller + quota | Yes (all-or-nothing admit) | Yes (ClusterQueue/LocalQueue, cohorts, borrowing) | Yes (fair sharing, weights) | Yes (within/across cohort) | No — uses default scheduler | Cloud-native batch on existing K8s; multi-cluster (MultiKueue); JobSet/MPIJob/RayJob/batch Job |
| **Volcano** | Full batch scheduler | Yes (PodGroup `minMember`) | Yes (queues, capacity/proportion) | Yes (DRF/proportion) | Yes (reclaim, preempt) | Yes — runs as `schedulerName: volcano` | Slurm-like feel; built-in MPI/PyTorch/TF plugins; AI training |
| **Coscheduling plugin** | scheduler-plugins gang plugin (PodGroup) | Yes | No (just gang) | No | Limited | No — a default-scheduler plugin | Minimal gang without adopting a big system |
| **YuniKorn** | Full scheduler, hierarchical queues | Yes (gang via taskgroups) | Yes (hierarchical queues) | Yes | Yes | Yes — replacement scheduler | Mixed batch + service, Spark/Flink heritage, queue hierarchies |
| **Flux** (via Flux Operator) | HPC-native graph scheduler run *as* a K8s workload (MiniCluster) | Yes (co-schedules the whole MiniCluster pod set as one gang) | Yes (Flux's own hierarchical queues *inside* the MiniCluster) | Yes (Flux fair-share, hierarchical) | Yes (within Flux) | No — Flux schedules *inside* a pod set the default scheduler/Kueue placed | HPC-native scheduling on K8s, nested/hierarchical scheduling, complex graph-based resource modeling, tightly-coupled MPI |

Key decision points:

- **Kueue admits *workloads*, not pods.** It sits above the default scheduler: a Job is held `Suspended`
  until its full quota is available in a ClusterQueue, then unsuspended so the *default* scheduler places
  the pods. This means you keep all the default scheduler's behavior, plugins, and ecosystem — Kueue only
  governs *whether/when* a job is allowed to start. It understands batch/v1 Job, JobSet, MPIJob, RayJob,
  and more, and supports **cohorts** (quota borrowing/lending between queues), **MultiKueue** (dispatch
  to multiple clusters), and **Topology-Aware Scheduling**. This is the cloud-native, future-leaning
  answer; see `[[kueue-advanced]]`.
- **Volcano replaces the scheduler.** Pods carry `schedulerName: volcano` and a **PodGroup** with
  `minMember`. Volcano makes the gang decision itself and adds queues, DRF fair-share, reclaim, and
  framework plugins (its `mpi`, `pytorch`, `tensorflow` job plugins inject the env/SSH/services those
  frameworks need). It feels closest to Slurm and is popular for self-managed AI training. Cost: you've
  swapped a core component, so you own its upgrades and its interaction with everything else.
- **Don't run two gang schedulers over the same pods.** Pick Kueue *or* Volcano as the batch authority
  for a given pool. A common clean split is Kueue for admission/quota + default scheduler for placement,
  *or* Volcano end-to-end. Mixing creates fights over who owns the gang decision.
- **Coscheduling plugin** is the right minimal choice if you only need gang and want to stay on the
  default scheduler without Kueue's quota machinery.
- **When to use Flux** (detail in §4d): reach for the **Flux Framework** / **Flux Operator** when you
  want a genuinely *HPC-native* scheduler on K8s rather than a K8s-native one — specifically when you need
  **hierarchical / nested scheduling** (a job that itself sub-schedules its own work), **fine-grained
  graph-based resource modeling** (scheduling against a resource graph, not just counts of
  `nvidia.com/gpu`), or strong **co-scheduling / gang** for tightly-coupled MPI. Flux runs *as* a workload
  inside a pod set (a MiniCluster) — it doesn't replace the K8s scheduler; it schedules *within* the
  allocation that the default scheduler (or Kueue/Volcano) gave it. Treat it as complementary to
  Kueue/Volcano (which admit/place the pod set) rather than a competitor to them.

---

## 4. Slurm on Kubernetes

Three architectures, increasing intimacy:

### 4a. Slinky — SchedMD's `slurm-operator`

**Slinky** is SchedMD's (the Slurm vendor) official project for running Slurm *as a Kubernetes
workload*. A Slurm operator + Helm charts deploy the Slurm control plane and compute as K8s objects:
`slurmctld` and `slurmdbd` as pods/StatefulSets, `slurmd` compute as a set you can scale, login pods,
and the accounting MySQL. The headline value is **elastic compute**: `slurmd` pods can be scaled up/down
(potentially driven by queue depth) so Slurm "bursts" onto K8s-managed capacity, and you get one
substrate (K8s) for both your services and your Slurm cluster. Users still submit with `sbatch`/`srun`
exactly as before — the Slurm UX is preserved on top of K8s plumbing.

When it makes sense: you have a Slurm-native user base and want K8s operational benefits (GitOps, elastic
nodes, shared infra) without retraining researchers. Trade-offs: you now run *both* a Slurm control plane
and Kubernetes — two scheduling systems, two failure domains; networking/RDMA must be plumbed into the
`slurmd` pods just as for any tightly-coupled K8s job. Verify current Slinky version, supported Slurm
versions, and exactly which features (elastic scaling, login, accounting) are GA in the version you
deploy — this project has evolved quickly.

### 4b. SUNK — Slurm on Kubernetes (CoreWeave)

**SUNK** is CoreWeave's Slurm-on-Kubernetes stack with the same core idea: run Slurm components as K8s
workloads on a shared GPU cluster so the same nodes can serve both Slurm batch jobs and native K8s
workloads, with K8s providing the underlying node/lifecycle/networking management. Positioning and
feature set differ from Slinky in the details (and availability/openness may differ); verify against
CoreWeave's current docs rather than assuming parity. The strategic point is the same: let one GPU fleet
serve Slurm users and K8s users without partitioning hardware permanently.

### 4c. Side-by-side (both schedulers, shared nodes)

Run a normal Slurm cluster and a normal Kubernetes cluster that *share* compute nodes — each node runs
both `slurmd` and `kubelet`. Whichever system isn't using a node leaves it to the other. This is the most
flexible (no operator dependency, each system is "real") and the **hardest to keep correct**: two
schedulers can both believe a GPU is free and **double-book** it. You need strict fencing — cgroup/MIG
partitioning, node draining/cordoning handshakes, or time-slicing the fleet (e.g. Slurm owns a node, then
hands it to K8s). Use it when you have strong reasons to keep both stacks independent; otherwise prefer an
operator (Slinky/SUNK) or pick one scheduler.

### Choosing among the three

- Want Slurm UX + K8s ops, single vendor-blessed path → **Slinky**.
- On CoreWeave or want their integrated stack → **SUNK**.
- Must keep two fully independent stacks on shared metal → **side-by-side** (accept the fencing burden).
- Don't actually need Slurm semantics, just gang/quota → skip Slurm-on-K8s entirely; use **Kueue/Volcano**.

### 4d. Flux Framework and the Flux Operator (HPC-native scheduler as a K8s workload)

**Flux** is a modern, fully hierarchical, graph-based resource manager and scheduler from the HPC world —
often framed as a next-generation successor to Slurm for many use cases. Two ideas make it distinctive:

- **Fully hierarchical / nested scheduling.** A Flux instance can launch *another* Flux instance as a job,
  recursively. A large allocation can spin up sub-schedulers that independently schedule their own work
  (e.g. an ensemble / many-task workflow, or a workflow engine that wants its own scheduler), without
  contending on one central controller. This recursion is the headline difference from Slurm's single
  `slurmctld` brain — it removes the central scheduler as a throughput bottleneck for high-throughput and
  workflow-heavy workloads.
- **Graph-based resource modeling.** Flux schedules against a **resource graph** (via its `fluxion`
  scheduler) — nodes, sockets, cores, GPUs, and their interconnect/locality expressed as a graph that
  requests are matched against. This is more expressive than scheduling against flat extended-resource
  counts (`nvidia.com/gpu: 8`), so it can reason about fine-grained, topology-aware placement directly.

**The Flux Operator** runs Flux *as a workload on Kubernetes*. Its core CRD is the **MiniCluster**: the
operator brings up a set of pods (an indexed pod set, broker pod + workers) that boot a **Flux instance**
spanning them, then runs your job (e.g. an MPI program via `flux run`/`flux submit`) inside that instance.
Mental model: the MiniCluster *is* a transient Flux cluster materialized as pods — analogous to a Slurm
allocation, but the scheduler lives inside the pods rather than in a separate control plane.

How it relates to gang scheduling and MPI on K8s:

- **Co-scheduling / gang.** A MiniCluster only does useful work once *all* its pods are up and have joined
  the Flux instance to form the MPI ring — exactly the "many pods must start together" property. The
  MiniCluster pod set must therefore be admitted/placed as a gang. Flux itself co-schedules the tasks
  *within* that set; getting the set placed all-or-nothing is the job of the default scheduler plus a gang
  admitter — i.e. **Kueue** (admit the MiniCluster's pods as one workload; see `[[kueue-advanced]]`) or
  **Volcano** (PodGroup `minMember`). Flux and Kueue/Volcano are layered, not alternatives: Kueue/Volcano
  decide *whether/where* the pod set runs; Flux decides *how work is scheduled inside it*.
- **Relation to the MPI Operator (§5).** The Flux Operator is an alternative way to run tightly-coupled
  MPI on K8s: instead of the MPI Operator's launcher + SSH/PMIx + hostfile wiring (§5), the Flux broker
  bootstraps and wires up the ranks (Flux provides the PMI/process-management interface MPI needs) and you
  launch with `flux run`/`flux submit` instead of `mpirun`. NCCL is still the GPU collective layer
  underneath either way (see §5's MPI-vs-NCCL note), and the same interconnect requirements apply (§6):
  RDMA in the pod, GPUDirect, correct NCCL env, and topology-aware placement of the MiniCluster pods.
- **Mapping.** MiniCluster ≈ a Slurm allocation / a JobSet pod set; `flux submit`/`flux run` ≈ `srun` /
  `mpirun`; the Flux broker ≈ `slurmctld` but co-located in the pods and nestable; a nested Flux instance
  ≈ a sub-allocation. Compare with `[[jobset-leaderworkerset]]` (JobSet/LWS as the generic K8s multi-host
  primitive) — a MiniCluster is the Flux-native equivalent that additionally carries a scheduler inside.

When to choose it: you want HPC-native, hierarchical, graph-aware scheduling *on* K8s (high-throughput
ensembles/workflows, nested scheduling, complex resource graphs, or you already use Flux on bare-metal HPC
and want parity on K8s). When you only need gang + quota for a containerized training job, plain
Kueue/Volcano + JobSet/MPIJob is simpler. The Flux Operator and `fluxion` move quickly — verify current
MiniCluster CRD fields, supported launch/MPI integrations, and Kueue/Volcano interop against the **Flux
Framework / Flux Operator docs** before designing around specifics.

---

## 5. MPI on Kubernetes — the MPI Operator

Tightly-coupled MPI is the HPC workload K8s historically handled worst, and the **MPI Operator**
(Kubeflow) is the answer. It defines an **`MPIJob`** CRD:

- A **launcher** pod and N **worker** pods (`replicas`).
- The operator wires **passwordless SSH** between launcher and workers (or **PMIx**, depending on the
  `mpiImplementation` / version), generates a **hostfile**, and creates the headless Service/DNS so the
  launcher can reach every worker.
- The launcher runs `mpirun`/`mpiexec` (or `horovodrun`) which fans the ranks out across the workers.

This is the canonical home for **Horovod** (data-parallel training originally built on MPI all-reduce)
and for any classic MPI binary. Mapping from Slurm: launcher ≈ the `sbatch` script body; workers ≈ the
allocated compute nodes; `mpirun -np` ≈ `srun -n`; the operator's SSH/hostfile ≈ what Slurm+PMIx provides
natively.

```yaml
apiVersion: kubeflow.org/v2beta1     # verify the apiVersion shipped by your installed MPI Operator
kind: MPIJob
metadata:
  name: horovod-train
spec:
  slotsPerWorker: 8                  # GPUs/ranks per worker  (~ --ntasks-per-node)
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: launcher
            image: my/horovod:latest
            command: ["mpirun"]
            args: ["-np","16","-bind-to","none","-map-by","slot",
                   "python","train.py"]
    Worker:
      replicas: 2                    # ~ --nodes
      template:
        spec:
          containers:
          - name: worker
            image: my/horovod:latest
            resources:
              limits:
                nvidia.com/gpu: 8    # ~ --gpus-per-node
```

**MPI vs NCCL — keep them straight.** MPI (or `torchrun`/PyTorch's c10d, or PMIx) is the *bootstrap/
launch* layer: it starts the ranks, assigns rank IDs, and exchanges the addresses needed to form the
process group. **NCCL** is the *collective communication* layer that actually moves gradients
(all-reduce/all-gather) GPU-to-GPU over NVLink/PCIe/RDMA. So MPIJob launches the job; NCCL (configured by
env like `NCCL_IB_HCA`) does the heavy GPU traffic underneath. Modern PyTorch training on K8s more often
uses **JobSet/LeaderWorkerSet + `torchrun`** (see `[[jobset-leaderworkerset]]`, `[[training-frameworks]]`)
than MPI; reach for MPIJob specifically when you have Horovod or genuine MPI code. Either way Kueue can
admit the job as a gang and Volcano can schedule it.

---

## 6. The interconnect & node story on cloud/K8s

Tightly-coupled jobs are bandwidth- and latency-bound on the network. A multi-node all-reduce moving over
plain TCP/pod-overlay networking can run an order of magnitude slower than over RDMA — often making the
extra nodes worthless. The HPC fabric must reach into the pod:

- **InfiniBand (IB)** — the classic HPC fabric, lowest latency, native RDMA. On cloud, exposed via an
  **RDMA device plugin** (advertises e.g. `rdma/hca` resources) and/or **SR-IOV** virtual functions, often
  with **Multus** for a secondary high-speed NIC alongside the pod's normal eth0.
- **RoCEv2 (RDMA over Converged Ethernet)** — RDMA on Ethernet; needs a lossless fabric (PFC/ECN, correct
  MTU). The common cloud path when there's no IB.
- **GPUDirect RDMA** — NIC DMAs straight to/from GPU memory, bypassing the host CPU/bounce buffers.
  Essential for full bandwidth. On GKE this is **GPUDirect-TCPX / GPUDirect-TCPXO**; see `[[gke-master]]`.
- **SR-IOV / multi-NIC / host networking** — tightly-coupled jobs frequently need a dedicated high-speed
  NIC per GPU exposed into the pod (SR-IOV VFs via the SR-IOV device plugin + Multus, or vendor "networks"
  abstractions like GKE multi-networking). Some setups use `hostNetwork: true` to avoid overlay overhead.
- **Hugepages** are commonly required by the RDMA stack; request them as a K8s resource.
- **NCCL tuning env** (the levers you'll actually set):
  `NCCL_IB_HCA` (which HCAs to use), `NCCL_SOCKET_IFNAME` (control-plane NIC),
  `NCCL_NET_GDR_LEVEL` (enable GPUDirect), `NCCL_IB_GID_INDEX` / GID for RoCE, `NCCL_DEBUG=INFO` to print
  the topology NCCL chose. Wrong values here are the #1 cause of "it runs but is mysteriously slow."

**Topology-aware placement.** All gang members should land in the same network locality domain
(rack/block/superpod) so collectives stay on the fast fabric. The K8s mechanisms:
- **Kueue Topology-Aware Scheduling (TAS)** — declare topology levels via node labels; Kueue admits the
  gang into a domain that fits. See `[[kueue-advanced]]`.
- **Volcano network-topology / HyperNode** features for topology-aware gang placement.
- **Node affinity + compact placement** — pin to a placement group / compact-placement policy (GKE) or
  rack label. See `[[gke-master]]` and `[[autoscaling-kubernetes]]` for provisioning the placed capacity.
Slurm gets this from `topology.conf` automatically; on K8s you assemble it from these pieces.

---

## 7. Decision guidance: Slurm vs Kubernetes for an org

There is no universal answer; there are real trade-offs. Weigh:

| Dimension | Favors **Slurm** | Favors **K8s + Kueue/Volcano** |
|-----------|------------------|-------------------------------|
| User familiarity | Researchers live in `sbatch`/`srun`, modules, shared FS | Teams already fluent in K8s/YAML/CI |
| Workload shape | Classic MPI, tightly-coupled simulation, many small jobs, backfill | Containerized ML, mixed train + serve + services |
| Fair-share / accounting | Mature, hierarchical, battle-tested (`sacct`, associations) | Kueue cohorts/fair-sharing, improving fast but newer |
| Multi-tenancy & isolation | cgroup/partition based | Namespaces, RBAC, NetworkPolicy, quotas |
| Cloud elasticity / autoscale | Bolt-on (cloud plugins, elastic via Slinky) | Native (Cluster Autoscaler/Karpenter/ProvisioningRequest) |
| Ecosystem / tooling | HPC modules, MPI, scientific stacks | Operators, observability, GitOps, ML platforms, serving |
| Inference / online serving | Not its job | First-class (Deployments, KServe, autoscaling) |
| Single control plane for everything | No (separate stack) | Yes |

Heuristics:

- **Keep Slurm** when the scheduler *is the product* — large MPI/simulation HPC center, deep `sbatch`
  muscle memory, backfill-driven utilization, no appetite to retrain. Consider **Slinky/SUNK** to gain
  K8s ops underneath without changing the user UX.
- **Choose K8s + Kueue/Volcano** when you want one platform for training *and* serving *and* services,
  cloud-native elasticity, the operator/ML ecosystem, and your users are (or will be) container-native.
  **Kueue** for the cloud-native, default-scheduler-preserving path (and multi-cluster via MultiKueue);
  **Volcano** if you want the most Slurm-like single-cluster batch scheduler with built-in MPI plugins.
- **Run both / hybrid** is extremely common: Slurm for the established HPC users, K8s for ML platform and
  serving, with bursting between them (Slinky elastic `slurmd`, or shared GPU pools). Don't force a single
  winner if two communities genuinely have different needs.
- **Migration is a multi-quarter program, not a flag.** Hardest parts: replicating *fair-share* and
  *accounting* semantics, reproducing *backfill* utilization, getting *RDMA/topology* parity, and the
  human cost of retraining researchers off `sbatch`. Migrate for a concrete win (unify train+serve, cloud
  elasticity, ecosystem), pilot one team, run side-by-side during transition, and prove fairness/utilization
  parity before you decommission Slurm.

---

## 8. Troubleshooting tightly-coupled jobs

| Symptom | Likely cause | Diagnosis | Fix |
|---------|-------------|-----------|-----|
| **Gang deadlock / partial gang** — some pods Running, rest Pending forever, GPUs idle | No gang admission (default scheduler placed a subset), or quota lets some in | `kubectl get pods` (subset Running); Volcano: `kubectl get podgroup` (`minMember` unmet); Kueue: Workload not Admitted | Enforce gang: Volcano PodGroup `minMember` = all pods; Kueue all-or-nothing admission; coscheduling PodGroup. Never let a subset start. |
| **Job queued forever** | Quota too small, no topology domain fits the gang, or fair-share starvation | Kueue: describe Workload / ClusterQueue for "couldn't admit"; Slurm: `squeue --start`, `scontrol show job` reason | Raise quota / add capacity (autoscale), relax topology requirement, check fair-share/priority |
| **NCCL hang at init / `all-reduce` stalls** | IB not exposed in pod, wrong `NCCL_IB_HCA`/`NCCL_SOCKET_IFNAME`, GDR off, RoCE PFC/MTU mismatch | `NCCL_DEBUG=INFO` shows chosen transport (look for `NET/IB` vs `NET/Socket`); check the RDMA device is present in the pod | Expose RDMA (device plugin/SR-IOV/Multus); set correct HCA/IFNAME; enable GDR (`NCCL_NET_GDR_LEVEL`); fix MTU/PFC on RoCE |
| **Throughput far below expectation** | Falling back to TCP/overlay instead of RDMA; gang spread across racks | `NCCL_DEBUG=INFO` transport; check node topology labels / placement | Fix RDMA path; enforce topology-aware placement (Kueue TAS / Volcano topology / compact placement) |
| **MPI launcher can't reach workers** | SSH/PMIx not wired, hostfile wrong, headless Service/DNS, or workers not all Running | MPIJob: `kubectl logs <launcher>` (SSH refused / host unreachable); check worker pod readiness and the operator-created Service | Ensure all workers Running before launch (gang!), verify operator wired SSH/hostfile, check DNS/headless Service |
| **Job killed at wallclock** | `--time` / `activeDeadlineSeconds` hit | Slurm: `sacct` state TIMEOUT; K8s: pod terminated by deadline | Raise the limit or checkpoint; note accurate `--time` is also what makes Slurm **backfill** work |
| **Slurm + K8s double-booked a GPU (side-by-side)** | Both schedulers think a node is free | Compare `sinfo`/`scontrol show node` against `kubectl describe node` allocations | Strict fencing: cordon/drain handshake, cgroup/MIG partition, or time-slice node ownership |
| **GPU not visible / GRES mismatch** | Device plugin not advertising, or `gres.conf`/labels wrong | `kubectl describe node` capacity (`nvidia.com/gpu`), Slurm `scontrol show node` Gres | Fix device plugin / `gres.conf`; ensure driver + DCGM/feature labels present |

General method: on **Slurm** use `squeue` (queue + reasons), `scontrol show job/node`, `sinfo`,
`sacct`/`sacctmgr` (accounting/fair-share). On **K8s** use `kubectl get/describe` on the Job/JobSet/MPIJob,
the **Workload** object (Kueue) or **PodGroup** + Volcano events, scheduler logs, and `NCCL_DEBUG=INFO`
for the interconnect. Always confirm *the gang is whole and placed in one topology domain* before chasing
deeper bugs — most "mysterious" tightly-coupled failures are gang or interconnect, not the application.

---

## 9. Version awareness

- **Slurm** — `slurm.conf`/`gres.conf`/`topology.conf` semantics are stable; specific plugins
  (`sched/backfill`, `priority/multifactor`, `select/cons_tres`) and `--mpi=pmix` support depend on build/
  version. Confirm your site's Slurm version for QOS/preemption/TRES behavior.
- **Slinky / SUNK** — moving fast; confirm GA features (elastic `slurmd`, login, accounting, supported
  Slurm versions) against current SchedMD/CoreWeave docs before designing around them.
- **Kueue** — pre-1.0-era APIs stabilized but features (MultiKueue, TAS, fair sharing, ProvisioningRequest
  integration) graduate over releases; check the version's feature gates. Deep dive in `[[kueue-advanced]]`.
- **Volcano** — PodGroup/Queue APIs and the action/plugin set (gang, reclaim, preempt, DRF) evolve; verify
  the job-plugin set (mpi/pytorch/tensorflow) for your release.
- **MPI Operator** — the `MPIJob` `apiVersion` and SSH-vs-PMIx behavior differ across versions; check the
  installed CRD's apiVersion.
- **Flux Framework / Flux Operator** — moving fast; the MiniCluster CRD fields, the `flux`/`fluxion`
  command surface, MPI/PMI launch integration, and Kueue/Volcano interop evolve. Verify against the Flux
  Framework / Flux Operator docs before relying on specific fields, flags, or capabilities.
- **NCCL / interconnect** — NCCL env var names and GPUDirect variants change with NCCL and cloud-provider
  releases; verify on the target platform (`[[gke-master]]` for GKE specifics).

Where you are unsure of an exact field/flag/capability: describe the concept and tell the reader to verify
against the project's current documentation rather than guessing.

---

## 10. Canonical references

- Slurm documentation — https://slurm.schedmd.com/documentation.html (and `slurm.conf`, `sbatch`,
  `srun`, `sacctmgr`, multifactor priority, topology man pages).
- Slinky / `slurm-operator` (SchedMD) — https://github.com/SlinkyProject and SchedMD's Slinky docs.
- SUNK (CoreWeave) — CoreWeave docs/blog for current capabilities.
- Volcano — https://volcano.sh and https://github.com/volcano-sh/volcano (PodGroup, queues, plugins).
- Kueue — https://kueue.sigs.k8s.io and https://github.com/kubernetes-sigs/kueue (see `[[kueue-advanced]]`).
- Kubernetes scheduler-plugins (coscheduling) — https://github.com/kubernetes-sigs/scheduler-plugins.
- Apache YuniKorn — https://yunikorn.apache.org.
- Flux Framework — https://flux-framework.org and https://github.com/flux-framework (flux-core, and the
  `fluxion` graph scheduler at https://github.com/flux-framework/flux-sched).
- Flux Operator (MiniCluster on K8s) — https://github.com/flux-framework/flux-operator.
- MPI Operator (Kubeflow) — https://github.com/kubeflow/mpi-operator.
- Horovod — https://github.com/horovod/horovod.
- NVIDIA NCCL — https://docs.nvidia.com/deeplearning/nccl/ (env vars, IB/GDR tuning).
- RDMA / SR-IOV on K8s — k8snetworkplumbingwg (Multus, SR-IOV network device plugin) and the NVIDIA
  Network Operator.
- GKE accelerators/networking — see `[[gke-master]]` (GPUDirect-TCPX/TCPXO, compact placement, gVNIC).

---

# Examples — the same multi-node GPU job in Slurm and on Kubernetes

One intent — **8 nodes × 8 GPUs = 64-GPU data-parallel training, all-or-nothing, on the fast fabric** —
expressed three ways: (1) a Slurm `sbatch` script, (2) the same as an MPI Operator `MPIJob`, (3) the same
as a Kueue-admitted `JobSet` running `torchrun`. Annotations show the HPC↔K8s mapping. Treat CRD
`apiVersion`s and field names as "verify against the version installed on your cluster."

---

## 1. Slurm — `sbatch` (the reference intent)

```bash
#!/bin/bash
#SBATCH --job-name=gpt-train
#SBATCH --partition=gpu          # queue/pool        -> Kueue ClusterQueue / Volcano queue
#SBATCH --qos=high               # QOS tier          -> queue priority / PriorityClass
#SBATCH --account=research       # for fair-share    -> Kueue quota cohort / Volcano fair-share
#SBATCH --nodes=8                # GANG SIZE         -> 8 worker pods / JobSet replicas
#SBATCH --ntasks-per-node=8      # ranks per node    -> 8 GPUs/pod, nproc_per_node=8
#SBATCH --gpus-per-node=8        # GPUs/node         -> resources: nvidia.com/gpu: 8
#SBATCH --cpus-per-task=12       # CPUs/rank         -> cpu requests
#SBATCH --mem=0                  # all node memory   -> memory requests / whole-node
#SBATCH --time=04:00:00          # wallclock (hard)  -> activeDeadlineSeconds: 14400
#SBATCH --exclusive              # whole nodes       -> dedicated nodes / whole-node request
#SBATCH --constraint=a100        # node feature      -> nodeSelector / affinity
#SBATCH --output=%x-%j.out

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5          # use the IB HCAs (site-specific)
# srun launches all 64 tasks together (gang) and provides PMIx wire-up for the
# process group; topology.conf keeps the 8 nodes on as few switches as possible.
srun --mpi=pmix python train.py --global-batch-size 4096
```

Key Slurm properties to reproduce on K8s: **the whole 64-GPU allocation is granted at once (gang)**;
the job **waits in a fair-share queue** until it fits; **backfill** runs smaller jobs in the meantime;
**topology** keeps the 8 nodes network-local; **PMIx** bootstraps the ranks; **RDMA/IB** carries NCCL.

---

## 2. Kubernetes — MPI Operator `MPIJob` (for Horovod / classic MPI)

Use this when the workload is **Horovod** or genuine MPI. The operator creates a launcher + 8 workers,
wires SSH/PMIx + hostfile + headless Service, and the launcher runs `mpirun` across the workers.

```yaml
apiVersion: kubeflow.org/v2beta1     # VERIFY the apiVersion of your installed MPI Operator
kind: MPIJob
metadata:
  name: gpt-train
  labels:
    kueue.x-k8s.io/queue-name: research-lq   # optional: let Kueue gang-admit this MPIJob
spec:
  slotsPerWorker: 8                  # ranks(GPUs) per worker   <- #SBATCH --ntasks-per-node=8
  runPolicy:
    cleanPodPolicy: Running
    activeDeadlineSeconds: 14400     # <- #SBATCH --time=04:00:00
  mpiReplicaSpecs:
    Launcher:                        # ~ the sbatch script body (drives mpirun)
      replicas: 1
      template:
        spec:
          containers:
          - name: launcher
            image: my/horovod-gpu:latest
            command: ["mpirun"]
            args:                    # -np 64 = 8 workers * 8 slots   <- srun -n 64
            - "-np"
            - "64"
            - "-bind-to"
            - "none"
            - "-map-by"
            - "slot"
            - "-x"
            - "NCCL_DEBUG=INFO"
            - "-x"
            - "NCCL_IB_HCA=mlx5"
            - "python"
            - "train.py"
            - "--global-batch-size"
            - "4096"
    Worker:                          # the 8 compute "nodes"          <- #SBATCH --nodes=8
      replicas: 8
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-a100   # <- --constraint=a100 (example)
          containers:
          - name: worker
            image: my/horovod-gpu:latest
            resources:
              limits:
                nvidia.com/gpu: 8                  # <- --gpus-per-node=8
                rdma/hca: 1                        # RDMA/IB into the pod (device-plugin specific)
                hugepages-2Mi: 2Gi                 # RDMA stack often needs hugepages
              requests:
                cpu: "96"                          # ~ cpus-per-task * slots
          # RDMA NIC commonly attached via SR-IOV/Multus annotation or provider multi-networking.
```

Notes:
- **Gang:** ensure all 8 workers + launcher start together. Under Kueue, admission is all-or-nothing;
  with Volcano, attach a PodGroup with `minMember` covering every pod. Don't let the launcher start before
  the workers are Running, or `mpirun` will fail to reach hosts.
- **MPI vs NCCL:** `mpirun` only *launches/wires* the 64 ranks; **NCCL** (the `NCCL_IB_HCA` env) does the
  actual GPU all-reduce over IB. Wrong HCA/IFNAME or missing `rdma/hca` → NCCL falls back to TCP and
  crawls, or hangs at init.

---

## 3. Kubernetes — Kueue + `JobSet` running `torchrun` (modern PyTorch path)

For native PyTorch DDP/FSDP, the idiomatic 2026 path is **JobSet/LeaderWorkerSet + `torchrun`**, admitted
as a gang by **Kueue** (no MPI/SSH needed; `torchrun` + c10d bootstraps the process group). See
`[[jobset-leaderworkerset]]`, `[[training-frameworks]]`, and `[[kueue-advanced]]`.

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2   # VERIFY installed JobSet apiVersion
kind: JobSet
metadata:
  name: gpt-train
  labels:
    kueue.x-k8s.io/queue-name: research-lq   # Kueue LocalQueue -> ClusterQueue quota (the "partition")
spec:
  replicatedJobs:
  - name: workers
    replicas: 1
    template:
      spec:
        parallelism: 8                 # 8 pods            <- #SBATCH --nodes=8
        completions: 8
        activeDeadlineSeconds: 14400   #                   <- #SBATCH --time=04:00:00
        completionMode: Indexed        # stable rank index per pod (JOB_COMPLETION_INDEX)
        template:
          spec:
            nodeSelector:
              cloud.google.com/gke-accelerator: nvidia-a100   # <- --constraint=a100
            containers:
            - name: trainer
              image: my/pytorch-gpu:latest
              command: ["torchrun"]
              args:
              - "--nnodes=8"                       # <- --nodes=8
              - "--nproc_per_node=8"               # <- --ntasks-per-node=8 (8 GPUs/pod)
              - "--rdzv_backend=c10d"
              - "--rdzv_endpoint=gpt-train-workers-0.gpt-train:29500"  # headless-Service DNS rendezvous
              - "train.py"
              - "--global-batch-size"
              - "4096"
              env:
              - { name: NCCL_DEBUG,         value: "INFO" }
              - { name: NCCL_IB_HCA,        value: "mlx5" }
              - { name: NCCL_SOCKET_IFNAME, value: "eth0" }
              resources:
                limits:
                  nvidia.com/gpu: 8                # <- --gpus-per-node=8
                  rdma/hca: 1                      # RDMA/IB into the pod
                  hugepages-2Mi: 2Gi
```

The Kueue side (the "partition + fair-share + accounting" half of Slurm):

```yaml
# ResourceFlavor: ties quota to actual hardware (a100 nodes, optionally a topology for TAS)
apiVersion: kueue.x-k8s.io/v1beta1     # VERIFY installed Kueue apiVersion
kind: ResourceFlavor
metadata: { name: a100 }
spec:
  nodeLabels: { cloud.google.com/gke-accelerator: nvidia-a100 }
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue                     # ~ a Slurm partition + its quota/fair-share
metadata: { name: gpu-cq }
spec:
  namespaceSelector: {}
  cohort: research-cohort              # cohorts borrow/lend quota  ~ fair-share across accounts
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu","cpu","memory"]
    flavors:
    - name: a100
      resources:
      - { name: "nvidia.com/gpu", nominalQuota: 64 }   # this queue's GPU budget
      - { name: "cpu",            nominalQuota: "768" }
      - { name: "memory",         nominalQuota: 6Ti }
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue                       # namespaced handle the job points at
metadata: { name: research-lq, namespace: research }
spec: { clusterQueue: gpu-cq }
```

How this reproduces the Slurm properties:
- **Gang / all-or-nothing:** Kueue holds the JobSet `Suspended` until the *full* 64-GPU request fits the
  ClusterQueue, then unsuspends it — the default scheduler then places all 8 pods. No partial gang.
- **Queue + fair-share + accounting:** ClusterQueue quota = the partition budget; **cohorts** =
  borrow/lend ≈ fair-share across accounts; Kueue fair sharing arbitrates contention.
- **Topology (≈ `topology.conf`):** add Kueue **Topology-Aware Scheduling** (node topology labels +
  `podSetTopologyRequest`) so all 8 pods land in one rack/block; otherwise every all-reduce pays latency.
- **Capacity:** if 64 GPUs aren't free, the job waits; pair with Cluster Autoscaler/Karpenter/
  ProvisioningRequest (`[[autoscaling-kubernetes]]`) to provision the gang's nodes, ideally compact-placed.

---

## Quick mapping table

| Slurm `#SBATCH` / concept | MPIJob | JobSet + torchrun | Kueue / Volcano |
|---------------------------|--------|-------------------|-----------------|
| `--nodes=8` | `Worker.replicas: 8` | `parallelism/completions: 8` | gang `minMember` / admit whole |
| `--ntasks-per-node=8` | `slotsPerWorker: 8` / `-np 64` | `--nproc_per_node=8` | — |
| `--gres=gpu:8` / `--gpus-per-node=8` | `nvidia.com/gpu: 8` | `nvidia.com/gpu: 8` | covered resource quota |
| `--partition` / `--qos` | (label to a queue) | `queue-name` label | ClusterQueue / Volcano Queue |
| `--account` (fair-share) | — | — | cohort / fair sharing |
| `--time=04:00:00` | `activeDeadlineSeconds` | `activeDeadlineSeconds` | — |
| `--exclusive` | whole-node request | whole-node request | dedicated flavor/nodes |
| `--constraint=a100` | `nodeSelector`/affinity | `nodeSelector`/affinity | ResourceFlavor `nodeLabels` |
| `srun --mpi=pmix` (launch) | `mpirun` (SSH/PMIx) | `torchrun --rdzv c10d` | — |
| `topology.conf` | — | — | Kueue TAS / Volcano topology |
| IB/RDMA + NCCL env | `rdma/hca` + `NCCL_IB_HCA` | `rdma/hca` + `NCCL_IB_HCA` | — |
