# AGENTS.md — Kueue (sigs.k8s.io/kueue) Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`kueue-advanced-guide.md`** next to this file — read it before
> designing quota, debugging admission, or wiring an integration. Annotated manifests to imitate are in
> **`examples.md`**. This file is the always-on summary.
>
> **Core boundary:** Kueue does **admission control** (when + which ResourceFlavor a Workload may start),
> not Pod scheduling. It admits a Workload (assigns flavor + reserves quota) and unsuspends the Job;
> kube-scheduler binds Pods to nodes. Kueue is pre-1.0 and moves fast — verify load-bearing fields,
> feature gates, and defaults against the installed version at kueue.sigs.k8s.io.

## When working with Kueue, apply these by default:

- **The Workload is the unit of admission, not the Pod.** Each managed Job → one suspended Workload with
  PodSets. Kueue admits the Workload, records `status.admission`, then sets `spec.suspend: false`. A Job
  at `suspend: true` with zero Pods pre-admission is **healthy**, not broken.
- **Opt-in is the `kueue.x-k8s.io/queue-name: <localqueue>` label.** No label → Kueue ignores it. Avoid
  `manageJobsWithoutQueueName: true` fleet-wide (it suspends system/operator jobs); scope it if used.
- **Quota lives on the ClusterQueue, per ResourceFlavor:** `nominalQuota` (guaranteed floor),
  `borrowingLimit` (max above nominal from the cohort), `lendingLimit` (max own idle quota lent out).
  Cohort = borrowing pool. LocalQueue is a namespaced pointer with **no** quota. ResourceFlavor carries
  `nodeLabels`/`nodeTaints`/`tolerations`, not quota numbers.
- **Diagnose at the Workload, never the Job:** `kubectl describe workload` → conditions
  `QuotaReserved` / `Admitted` / Pending reason. Inadmissible ≈ quota exhausted, no matching flavor
  (taints/labels), resourceGroups doesn't cover the resource, or a failing AdmissionCheck.
- **Never strip `kueue.x-k8s.io/...` finalizers or labels by hand.** Finalizers gate quota release;
  removing them leaks quota and wedges the ClusterQueue. To unmanage, remove the queue-name label
  *before* creation, not mid-flight.
- **Queueing strategy:** `StrictFIFO` (head-of-line blocking) vs `BestEffortFIFO` (skip-ahead, higher
  utilization). **Preemption** has two axes: `withinClusterQueue` (priority) and `reclaimWithinCohort`
  (reclaim lent quota). **Fair Sharing** (weighted DRF) is an alternative — don't mix it with strict
  priority in one cohort. Tune flavor selection with `flavorFungibility` (`whenCanBorrow`/`whenCanPreempt`).
- **Gang is intrinsic:** quota for the whole Workload is reserved atomically (no partial-gang deadlock).
  Add **`waitForPodsReady`** so a half-started gang re-suspends/requeues instead of holding quota.
  Partial admission (min count) is for elastic jobs only and integration-dependent.
- **Two priority axes:** `WorkloadPriorityClass` (via `kueue.x-k8s.io/priority-class`) drives queue
  ordering/preemption in Kueue; Pod `PriorityClass` drives in-node kube-scheduler preemption. Different
  mechanisms — don't conflate.
- **TAS** places a gang within a topology domain (block/rack/host) via Topology + flavor `topologyName` +
  `kueue.x-k8s.io/podset-required-topology` / `...preferred-topology` annotations. It injects affinity;
  it doesn't bind Pods. Young feature — verify keys/gate.
- **ProvisioningRequest** AdmissionCheck gets just-in-time autoscaler/GKE capacity *before* unsuspend, so
  gangs don't start onto nonexistent nodes. **MultiKueue** dispatches the Workload to one of several
  worker clusters (multi-region/cloud, ABA failover); manager holds a mirror Workload, Job runs on the
  worker. Both are wired as AdmissionChecks on the ClusterQueue.
- **Integrations** must be enabled in Configuration `integrations.frameworks` AND carry the queue label:
  Job, JobSet, RayJob/RayCluster, MPIJob, PyTorchJob/TFJob (Kubeflow), LeaderWorkerSet, plain Pods/pod
  groups, AppWrapper. Each maps to PodSets per replica role.

## Sizing defaults
- Cohort per fungible capacity pool; ClusterQueue per team/tenant; ResourceFlavor per hardware/cost class
  (order cheap→expensive, accurate `nodeLabels`/`nodeTaints`).
- nominal = guaranteed floor; borrowingLimit = burst ceiling; lendingLimit = floor kept while idle.
- Keep borrowing topology shallow; bound `borrowingLimit` (don't leave unlimited everywhere).
- Right-size `waitForPodsReady` timeouts to real image-pull + autoscale times.

## Before declaring Kueue work done
Confirm: the integration is enabled and the Job carries the queue-name label; the Workload reaches
`Admitted` (or the Pending reason is understood); flavors match real node labels/taints; quota math
(nominal + borrowable vs request) holds; finalizers intact; and any feature-gated bits (TAS, partial
admission, hierarchical cohorts, MultiKueue) are actually enabled in the installed version.
