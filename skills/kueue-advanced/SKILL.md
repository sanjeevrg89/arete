---
name: kueue-advanced
description: Advanced mastery of Kueue (sigs.k8s.io/kueue), the Kubernetes-native job queueing and quota
  manager for batch/ML fleets. Use when working with Kueue objects — Workload, ResourceFlavor,
  ClusterQueue, LocalQueue, Cohort, AdmissionCheck — or when jobs are stuck Suspended/Pending/Inadmissible,
  when designing quota/borrowing/lending and cohorts, fair sharing, preemption, gang/all-or-nothing
  admission, waitForPodsReady, workload priority classes, Topology-Aware Scheduling (TAS),
  ProvisioningRequest, MultiKueue multi-cluster dispatch, or wiring Kueue to batch/Job, JobSet, RayJob,
  MPIJob, PyTorchJob/Kubeflow Trainer, LeaderWorkerSet, plain Pods, or AppWrapper. Covers the
  suspend/resume admission mechanism, kueue.x-k8s.io labels/finalizers, the Configuration API,
  troubleshooting, and sizing. Mentions feature maturity; verify current API against kueue.sigs.k8s.io.
---

# Kueue Advanced

Apply the judgment of a Kueue maintainer / power user who has run it for large multi-tenant ML batch
fleets for years. Kueue's job is **admission control**: decide *when* and *where* (which ResourceFlavor)
a batch workload may start, by checking quota in a borrowing/lending economy, then unsuspend it. It does
**not** schedule Pods to nodes — kube-scheduler still does that. Keep that boundary crisp.

## How to use this skill

1. **Read `kueue-advanced-guide.md`** in this directory — the full reference (architecture, quota model,
   scheduling/fairness, gang & TAS & ProvisioningRequest, MultiKueue, integrations, troubleshooting,
   anti-patterns, version awareness). Apply it to the task.
2. For complete annotated manifests to imitate (ResourceFlavor + ClusterQueue + LocalQueue + a
   Job/JobSet + a TAS and ProvisioningRequest example), read **`examples.md`**.
3. Match the cluster's existing Kueue topology and naming; apply correctness/safety rules regardless.
   Kueue moves fast (beta features graduate often) — when an API field, feature gate, or default is
   load-bearing, verify it against the docs for the *installed* version before relying on it.

## The essentials (full detail in `kueue-advanced-guide.md`)

- **The Workload is the atom of admission**, not the Pod. Every managed Job creates a suspended Workload;
  Kueue admits the Workload (assigns a flavor + reserves quota), then sets `spec.suspend: false`. A Job
  with `suspend: true` and no Pods is the normal, healthy pre-admission state — not a bug.
- **A Job is only managed if it opts in:** `kueue.x-k8s.io/queue-name: <localqueue>` label (or
  `manageJobsWithoutQueueName: true`, which you almost never want fleet-wide). No queue label → Kueue
  ignores it and it runs unqueued.
- **Quota lives on the ClusterQueue**, expressed per ResourceFlavor: `nominalQuota` (guaranteed),
  `borrowingLimit` (max you can pull *from the cohort* above nominal), `lendingLimit` (max you let
  others borrow from your unused nominal). Cohorts are the borrowing pool. LocalQueue is the
  namespaced pointer tenants submit to; it holds no quota.
- **Two queueing strategies:** `StrictFIFO` (head-of-line blocking — older workload blocks newer even if
  newer fits) vs `BestEffortFIFO` (skip-ahead when head can't be admitted). Default for throughput is
  BestEffortFIFO.
- **Preemption has two axes:** `withinClusterQueue` (priority-based, plus reclaim) and
  `reclaimWithinCohort` (take back quota you lent / lower-priority borrowers). Tune with
  `borrowWithinCohort` and `flavorFungibility` (`whenCanBorrow`/`whenCanPreempt`). **Fair Sharing**
  (weighted, DRF-style) is the alternative to classic priority preemption — don't mix mental models.
- **Gang/all-or-nothing:** Kueue admits a Workload only if the *whole* PodSet quota fits (prevents
  partial-gang deadlock). `waitForPodsReady` adds job-level readiness with a timeout + requeue/backoff so
  a half-started gang doesn't hold quota forever. Partial admission exists for elastic jobs that declare
  a min count.
- **TAS (Topology-Aware Scheduling)** places a gang within a topology domain (block/rack/host) for
  tight-coupling bandwidth — opt in per flavor + per workload annotation. **ProvisioningRequest** is an
  AdmissionCheck that asks the cluster autoscaler/GKE for just-in-time capacity *before* unsuspend, so
  gangs don't start onto nodes that don't exist yet.
- **MultiKueue** dispatches a Workload from a manager cluster to one of several worker clusters
  (multi-region/multi-cloud capacity, ABA failover). The manager holds a mirror Workload; the real Job
  runs on the worker. Wired via AdmissionCheck + MultiKueueConfig/MultiKueueCluster (and ClusterProfiles).
- **Finalizers/labels are load-bearing.** `kueue.x-k8s.io/...` finalizers let Kueue release quota on
  completion. **Never strip them manually** — you leak quota and wedge the ClusterQueue. To unmanage,
  delete the Job through normal means or remove the queue-name label *before* creation.
- **Diagnose admission via the Workload, not the Job:** `kubectl describe workload` → conditions
  (`QuotaReserved`, `Admitted`, `Pending` reason). Inadmissible almost always = quota exhausted,
  no matching flavor (taints/labels), or a failing AdmissionCheck.
- **Sizing:** one Cohort per fungible capacity pool; ClusterQueue per team/tenant; keep borrowing
  topology shallow and intentional; size `borrowingLimit`/`lendingLimit` to your fairness SLA, not "max".

## Related skills

- `[[jobset-leaderworkerset]]` — the multi-host Job/inference workloads Kueue most often admits.
- `[[aiml-on-kubernetes]]` — umbrella for training/inference/RL on K8s; Kueue is its admission layer.
- `[[autoscaling-kubernetes]]` — Cluster Autoscaler/Karpenter/NAP behind ProvisioningRequest.
- `[[training-frameworks]]` — PyTorch/JAX/Kubeflow Trainer jobs that become Kueue Workloads.
- `[[slurm-hpc-on-kubernetes]]` — Slurm/Volcano comparison; Kueue is the K8s-native queueing answer.
- `[[gke-master]]` — GKE specifics: ProvisioningRequest, TPU/GPU node pools, DWS.
- `[[kubernetes-controller-expert]]` — the reconcile/Workload-controller mechanics underneath Kueue.
