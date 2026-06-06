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
