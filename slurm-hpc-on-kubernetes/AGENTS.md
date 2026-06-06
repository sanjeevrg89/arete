# AGENTS.md — Slurm & HPC on Kubernetes

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`slurm-hpc-on-kubernetes-guide.md`** next to this file — read it
> before designing or debugging Slurm/HPC-on-K8s work. Annotated worked artifacts (an `sbatch` script and
> its MPIJob / Kueue+JobSet equivalents) are in **`examples.md`**. This file is the always-on summary.
>
> The library moves fast (it is 2026). Never assert a CRD field, flag, or project capability you're
> unsure of — describe the concept and say to verify against current Slinky/Volcano/Kueue/MPI-Operator
> docs.

## Apply by default when the task touches Slurm or HPC-style batch on Kubernetes

- **Slurm is a queue-first, all-or-nothing batch system.** `slurmctld` (controller) + `slurmd` (per node)
  + `slurmdbd` (accounting). Jobs (`sbatch`) request N nodes × resources, wait in a **partition** queue
  until the *whole* allocation fits, then launch all tasks (gang). `srun` is the parallel/MPI launcher;
  `salloc` is interactive.
- **Vanilla K8s schedules pod-by-pod → it has no gang, no fair-share queue, no backfill, no
  preempt-by-quota.** A multi-pod job can half-start and **deadlock** holding GPUs. Closing this gap is
  the whole reason for Kueue/Volcano/coscheduling. Always verify the gang is whole before debugging deeper.
- **Pick the K8s scheduler to fit the gap:** **Kueue** = quota + queueing + gang admission *on top of the
  default scheduler* (cloud-native, JobSet/MPIJob/RayJob aware, cohorts, MultiKueue, Topology-Aware
  Scheduling) → see `[[kueue-advanced]]`. **Volcano** = *replacement* scheduler with gang (PodGroup
  `minMember`), queues, DRF fair-share, reclaim/preempt, built-in MPI/PyTorch plugins (most Slurm-like).
  **Coscheduling plugin** = minimal PodGroup gang on the default scheduler. **YuniKorn** = hierarchical
  queues. **Never run two gang schedulers over the same pods.**
- **Flux Framework / Flux Operator = HPC-native scheduling on K8s (layered, not a replacement):** Flux is
  a fully hierarchical, graph-based scheduler (nested/recursive scheduling, `fluxion` resource graph). The
  **Flux Operator** runs it as a **MiniCluster** pod set that boots a Flux instance and co-schedules a
  tightly-coupled MPI ring (all pods start together). Flux schedules *inside* the pod set; Kueue/Volcano
  still admit/place that set as a gang. Use for hierarchical scheduling, complex resource graphs, or
  HPC-native MPI/co-scheduling; verify specifics against the Flux Framework / Flux Operator docs.
- **MPI on K8s = the MPI Operator (`MPIJob`):** launcher + N worker pods, operator wires SSH/PMIx +
  hostfile + headless Service, launcher runs `mpirun`/`horovodrun`. Canonical home for Horovod/classic
  MPI. **MPI/torchrun = launch/bootstrap; NCCL = the GPU collective layer underneath.** Modern PyTorch
  more often uses JobSet/LWS + `torchrun` (`[[jobset-leaderworkerset]]`, `[[training-frameworks]]`).
- **Tightly-coupled GPU jobs need the HPC fabric in the pod:** InfiniBand or RoCEv2 RDMA (RDMA device
  plugin / SR-IOV / Multus / multi-NIC), **GPUDirect RDMA**, hugepages, and correct NCCL env
  (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `NCCL_NET_GDR_LEVEL`; `NCCL_DEBUG=INFO` to confirm transport).
  Without RDMA you fall back to TCP and lose most of the value. On GKE → `[[gke-master]]`
  (GPUDirect-TCPX/TCPXO, compact placement).
- **Topology matters:** place all gang members in one locality domain (rack/block/superpod) via Kueue TAS
  / Volcano topology / compact placement, or every all-reduce pays a latency tax.
- **Slurm-on-K8s options:** **Slinky** (SchedMD `slurm-operator`: Slurm control plane + elastic `slurmd`
  as K8s workloads, bursting), **SUNK** (CoreWeave), or **side-by-side** (both schedulers on shared nodes
  — most flexible, but you must fence against double-booking). If you don't need Slurm semantics, skip
  these and use Kueue/Volcano.
- **`#SBATCH` ↔ K8s mapping:** `--nodes` → worker pods / JobSet replicas; `--ntasks-per-node` → MPI slots
  / `nproc_per_node`; `--gres=gpu:N`/`--gpus-per-node` → `nvidia.com/gpu`; `--partition`/`--qos` → Kueue
  ClusterQueue / Volcano queue; `--time` → `activeDeadlineSeconds`; `--exclusive` → whole-node request;
  account/fair-share → Kueue quota/fair-sharing or Volcano fair-share.
- **Decision rule:** keep/choose **Slurm** when researchers live in `sbatch`, jobs are MPI/tightly-coupled,
  and the scheduler is the product (add Slinky/SUNK for K8s ops). Choose **K8s + Kueue/Volcano** for one
  control plane across train+serve+services, cloud elasticity, and the ML/operator ecosystem. Hybrid/burst
  is common. Migrate for a concrete win, not fashion — fair-share, backfill utilization, and RDMA/topology
  parity are the hard parts.
- **Top failure modes:** partial gang / deadlock (no gang admission → enforce `minMember`/Kueue gang);
  NCCL hang or slow (wrong NIC/HCA, IB not in pod, GDR off, RoCE MTU/PFC); queued forever (quota too small,
  topology can't fit, fair-share starvation); launcher can't reach workers (SSH/hostfile/DNS, pods not all
  Running). Diagnose: Slurm `squeue`/`scontrol`/`sacct`; K8s `kubectl get podgroup`/Workload + scheduler
  events + `NCCL_DEBUG=INFO`.

## Related skills
`[[kueue-advanced]]` · `[[aiml-on-kubernetes]]` · `[[training-frameworks]]` · `[[jobset-leaderworkerset]]`
· `[[autoscaling-kubernetes]]` · `[[gke-master]]` · `[[kubernetes-internals-expert]]`
