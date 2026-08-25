# Kueue — Worked Examples

Annotated, runnable-in-spirit manifests. `apiVersion`s reflect the Kueue v1beta1 line (~2026); **verify
the version path against the installed CRDs** before applying — Kueue is pre-1.0 and field/version
details change. Apply top-to-bottom: cluster-scoped quota objects first, then namespaced LocalQueue, then
the workload.

## 1. The minimal complete stack: ResourceFlavor + ClusterQueue + LocalQueue

```yaml
# A typed slice of capacity. Quota numbers do NOT live here — only how to recognize the nodes.
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: a100-ondemand
spec:
  nodeLabels:                          # Pods of admitted workloads must land on nodes with these labels
    cloud.google.com/gke-accelerator: nvidia-tesla-a100
    kueue.x-k8s.io/capacity: on-demand
  nodeTaints:                          # the flavor tolerates (and thus targets) this taint
    - key: nvidia.com/gpu
      value: present
      effect: NoSchedule
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: a100-spot                      # cheaper, preemptible; listed first for fungibility below
spec:
  nodeLabels:
    cloud.google.com/gke-accelerator: nvidia-tesla-a100
    kueue.x-k8s.io/capacity: spot
---
# The quota holder. Belongs to cohort "gpu-pool" so it can borrow/lend with sibling ClusterQueues.
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: team-research
spec:
  cohort: gpu-pool                     # borrowing/lending pool shared with other teams' CQs
  namespaceSelector: {}                # which namespaces may use this CQ; {} = all (scope in prod)
  queueingStrategy: BestEffortFIFO     # skip-ahead for utilization; use StrictFIFO for strict ordering
  preemption:
    withinClusterQueue: LowerPriority         # higher-prio workloads evict lower-prio in this CQ
    reclaimWithinCohort: LowerPriority        # reclaim our nominal by preempting cohort borrowers
  flavorFungibility:
    whenCanBorrow: TryNextFlavor       # if spot is full, try on-demand before borrowing on spot
    whenCanPreempt: TryNextFlavor
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: a100-spot              # try cheap/preemptible first
          resources:
            - name: "cpu"
              nominalQuota: 480
            - name: "memory"
              nominalQuota: 3840Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 64
              borrowingLimit: 32       # may run up to 64+32=96 spot GPUs if cohort has idle quota
              lendingLimit: 32         # keep 32 GPUs of guaranteed floor even when we are idle
        - name: a100-ondemand
          resources:
            - name: "cpu"
              nominalQuota: 240
            - name: "memory"
              nominalQuota: 1920Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 16
              borrowingLimit: 0        # never borrow on the expensive flavor
---
# Namespaced pointer that tenants submit to. Holds NO quota.
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: research-lq
  namespace: team-research
spec:
  clusterQueue: team-research
```

Quota intuition for `nvidia.com/gpu` on the spot flavor: this CQ is guaranteed 64, can burst to 96 by
borrowing the cohort's idle GPUs, and will always lend at most 32 of its own idle 64 (so 32 stays
reclaimable on demand).

## 2. A batch/v1 Job that opts into the queue

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-shard
  namespace: team-research
  labels:
    kueue.x-k8s.io/queue-name: research-lq          # THE opt-in. No label => Kueue ignores the job.
    kueue.x-k8s.io/priority-class: high-priority    # WorkloadPriorityClass (queue ordering/preemption)
spec:
  parallelism: 8
  completions: 8
  # NOTE: do not set spec.suspend yourself — Kueue's webhook forces suspend:true until admission.
  template:
    spec:
      restartPolicy: Never
      tolerations:
        - key: nvidia.com/gpu
          value: present
          effect: NoSchedule
      containers:
        - name: trainer
          image: gcr.io/example/trainer:latest
          resources:
            requests:                                # drives the Workload's PodSet demand
              cpu: "8"
              memory: 64Gi
              nvidia.com/gpu: "1"
            limits:
              nvidia.com/gpu: "1"
```

```yaml
# The WorkloadPriorityClass referenced above (Kueue-specific; distinct from Pod PriorityClass).
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: high-priority
value: 1000
description: "High-priority research training"
```

Inspect admission:

```bash
kubectl get workloads -n team-research                 # find the auto-created Workload
kubectl describe workload <name> -n team-research       # QuotaReserved / Admitted / Pending reason
kubectl describe clusterqueue team-research             # flavorsUsage, pending/admitted
```

## 3. A JobSet (multi-host training) referencing the LocalQueue

JobSet is usually the right shape for tightly-coupled multi-host training; each ReplicatedJob becomes a
PodSet, and the whole JobSet is admitted gang-style.

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: llm-pretrain
  namespace: team-research
  labels:
    kueue.x-k8s.io/queue-name: research-lq
spec:
  replicatedJobs:
    - name: workers
      replicas: 1
      template:
        spec:
          parallelism: 16
          completions: 16
          backoffLimit: 0
          template:
            spec:
              restartPolicy: Never
              tolerations:
                - key: nvidia.com/gpu
                  value: present
                  effect: NoSchedule
              containers:
                - name: worker
                  image: gcr.io/example/megatron:latest
                  resources:
                    requests:
                      cpu: "12"
                      memory: 200Gi
                      nvidia.com/gpu: "8"
                    limits:
                      nvidia.com/gpu: "8"
```

Kueue reserves quota for all 16 × 8 = 128 GPUs atomically or leaves the JobSet pending — no partial gang.
See `[[jobset-leaderworkerset]]` for JobSet/LWS depth.

## 4. Topology-Aware Scheduling (TAS): keep a gang in one rack/block

```yaml
# Describe the node-label hierarchy of the cluster's network topology.
apiVersion: kueue.x-k8s.io/v1alpha1            # VERIFY: TAS API group/version evolves
kind: Topology
metadata:
  name: gpu-topology
spec:
  levels:
    - nodeLabel: cloud.provider.com/topology-block
    - nodeLabel: cloud.provider.com/topology-rack
    - nodeLabel: kubernetes.io/hostname
---
# Bind a flavor to the topology so TAS applies to workloads using it.
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: a100-tas
spec:
  topologyName: gpu-topology
  nodeLabels:
    cloud.google.com/gke-accelerator: nvidia-tesla-a100
---
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: tightly-coupled-train
  namespace: team-research
  labels:
    kueue.x-k8s.io/queue-name: research-lq
spec:
  replicatedJobs:
    - name: workers
      replicas: 1
      template:
        spec:
          parallelism: 32
          completions: 32
          template:
            metadata:
              annotations:
                # HARD: all 32 Pods must fit within ONE rack domain. Use ...preferred-topology for soft.
                kueue.x-k8s.io/podset-required-topology: cloud.provider.com/topology-rack
            spec:
              restartPolicy: Never
              containers:
                - name: worker
                  image: gcr.io/example/trainer:latest
                  resources:
                    requests: { cpu: "12", memory: 200Gi, nvidia.com/gpu: "8" }
                    limits:   { nvidia.com/gpu: "8" }
```

TAS computes a fitting rack at admission and injects nodeAffinity so kube-scheduler co-locates the gang.
**Verify the annotation keys, API version, and feature gate** for your installed Kueue.

## 5. ProvisioningRequest AdmissionCheck (just-in-time capacity)

Have Kueue ask the autoscaler/GKE for nodes *before* unsuspend, so a large gang never starts onto
nonexistent capacity.

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ProvisioningRequestConfig
metadata:
  name: gpu-provisioning
spec:
  provisioningClassName: queued-provisioning.gke.io   # e.g. GKE DWS class — VERIFY for your platform
  managedResources:
    - nvidia.com/gpu
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: AdmissionCheck
metadata:
  name: gpu-provisioning-check
spec:
  controllerName: kueue.x-k8s.io/provisioning-request
  parameters:
    apiGroup: kueue.x-k8s.io
    kind: ProvisioningRequestConfig
    name: gpu-provisioning
---
# Attach the check to the ClusterQueue (admission completes only after the check passes).
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: team-research-jit
spec:
  cohort: gpu-pool
  namespaceSelector: {}
  admissionChecks:
    - gpu-provisioning-check
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: a100-ondemand
          resources:
            - { name: "cpu", nominalQuota: 240 }
            - { name: "memory", nominalQuota: 1920Gi }
            - { name: "nvidia.com/gpu", nominalQuota: 64 }
```

Pair with `[[autoscaling-kubernetes]]` and `[[gke-master]]`.

## 6. MultiKueue (dispatch to worker clusters) — shape only

```yaml
# On the MANAGER cluster: a kubeconfig Secret per worker, a MultiKueueCluster per worker,
# a MultiKueueConfig listing them, and an AdmissionCheck wiring it onto the ClusterQueue.
apiVersion: kueue.x-k8s.io/v1beta1
kind: MultiKueueCluster
metadata:
  name: worker-us-central1
spec:
  kubeConfig:
    locationType: Secret
    location: worker-us-central1-kubeconfig      # Secret holding the worker's kubeconfig
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: MultiKueueConfig
metadata:
  name: multi-region
spec:
  clusters:
    - worker-us-central1
    - worker-europe-west4
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: AdmissionCheck
metadata:
  name: multikueue-check
spec:
  controllerName: kueue.x-k8s.io/multikueue
  parameters:
    apiGroup: kueue.x-k8s.io
    kind: MultiKueueConfig
    name: multi-region
# ...then list `multikueue-check` under the manager ClusterQueue's admissionChecks.
# The manager holds a mirror Workload; the real Job runs on a chosen worker, with ABA-style failover.
```

**Verify** the supported-integration matrix and whether your version sources cluster connection info via
`ClusterProfiles` vs Secret kubeconfigs.

## 7. Plain-Pod gang (bare Pods as one Workload)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gang-member-0
  namespace: team-research
  labels:
    kueue.x-k8s.io/queue-name: research-lq
    kueue.x-k8s.io/pod-group-name: my-gang          # all members share this group name
  annotations:
    kueue.x-k8s.io/pod-group-total-count: "4"       # group is admitted all-or-nothing at 4
spec:
  restartPolicy: Never
  containers:
    - name: worker
      image: gcr.io/example/worker:latest
      resources:
        requests: { cpu: "8", memory: 64Gi, nvidia.com/gpu: "1" }
        limits:   { nvidia.com/gpu: "1" }
# Create 4 Pods with the same pod-group-name; Kueue treats them as one gang Workload.
```

Requires the `pod` integration enabled in Configuration. **Do not remove the Kueue-added finalizers** on
these Pods — they gate quota release.
