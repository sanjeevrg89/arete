# Kueue Advanced — Full Reference

Kueue (`sigs.k8s.io/kueue`) is a Kubernetes-native **job queueing and quota** controller. It decides
*whether*, *when*, and *on which ResourceFlavor* a batch workload is allowed to start, enforcing quota
across teams in a borrowing/lending economy. It is the admission layer for batch/ML fleets; it is **not**
a Pod scheduler — once Kueue admits a Workload and unsuspends the Job, ordinary kube-scheduler binds the
Pods to nodes. Internalizing that split prevents most confusion.

This guide is current to the Kueue v0.x line through ~2026. Kueue moves fast and graduates beta features
frequently. Where an exact field name, default, or feature-gate name is load-bearing, **verify against
the docs for the installed version** at https://kueue.sigs.k8s.io. Do not assume a gate is on by default.

## Mental model: how a Job becomes admitted

1. A user creates a batch object (e.g. `batch/v1` Job) with the label
   `kueue.x-k8s.io/queue-name: <localqueue>`. Kueue's mutating webhook forces the Job to start
   **suspended** (`spec.suspend: true`) so no Pods are created yet.
2. Kueue's Job controller creates a **Workload** object (kind `Workload`,
   `kueue.x-k8s.io/v1beta1`) representing the resource *demand*: one or more **PodSets** (count +
   per-Pod resource requests + node selectors/tolerations/affinities/topology requests).
3. The Workload lands in a **LocalQueue** (namespaced) which points at a **ClusterQueue**
   (cluster-scoped, holds quota). The ClusterQueue's queueing strategy orders pending Workloads.
4. Kueue's scheduler picks the head (or next admissible) Workload, runs **flavor assignment** (find a
   ResourceFlavor whose nominal/borrowable quota covers every PodSet), and if any **AdmissionChecks**
   are configured (ProvisioningRequest, MultiKueue, custom), waits for them to pass.
5. On success Kueue sets `QuotaReserved`, then `Admitted`, records the assignment in
   `workload.status.admission`, and flips the Job's `spec.suspend` to `false`. kube-scheduler now runs
   the Pods.
6. On completion/deletion, the Workload's **finalizer** lets Kueue release the reserved quota back to
   the ClusterQueue/cohort. A leaked finalizer = leaked quota.

A Job sitting at `suspend: true` with zero Pods is the **normal pre-admission state**, not a failure.

## Core objects

- **Workload** — the unit of admission. Holds PodSets, priority, the target queue, and (once admitted)
  `status.admission` (flavors + assigned quota per PodSet) and conditions. Usually created *for* you by
  an integration; you rarely author it by hand.
- **ResourceFlavor** — a *typed* slice of cluster capacity (e.g. `gpu-a100`, `spot`, `on-demand-h100`).
  Carries `nodeLabels` (matched against nodes), `nodeTaints` (the flavor *tolerates* / requires), and
  `tolerations`. Flavors are how Kueue distinguishes heterogeneous hardware. Quota numbers are *not* on
  the flavor — they live on the ClusterQueue, keyed by flavor.
- **ClusterQueue** — cluster-scoped quota holder. Defines `resourceGroups` (sets of covered resources)
  → `flavors` → per-resource `nominalQuota` / `borrowingLimit` / `lendingLimit`. Also
  `cohort`, `queueingStrategy`, `preemption`, `flavorFungibility`, `admissionChecks`, `namespaceSelector`
  (which namespaces may use it), and `stopPolicy`.
- **LocalQueue** — namespaced pointer to one ClusterQueue. Tenants submit to LocalQueues; holds **no**
  quota of its own. Provides per-namespace visibility/metrics.
- **Cohort** — the borrowing/lending pool. ClusterQueues sharing a cohort can borrow each other's unused
  nominal quota (subject to limits). Cohorts can be implicit (string field on the ClusterQueue) or, in
  newer versions, a first-class **Cohort** object supporting **hierarchical cohorts** (a tree of quota
  pools). Verify whether the Cohort API is enabled in your version.
- **AdmissionCheck** — a pluggable gate that must pass before admission completes. Built-in controllers:
  **ProvisioningRequest** (autoscaler capacity) and **MultiKueue** (cross-cluster dispatch). You can
  write custom ones. A ClusterQueue references checks by name; they run after quota reservation.
- **WorkloadPriorityClass** — a Kueue-specific priority for *queue ordering and preemption* that is
  independent of the Pod `PriorityClass` used for in-node kube-scheduler preemption. Reference it on the
  Job via `kueue.x-k8s.io/priority-class`. Don't conflate the two.

## Quota model

Quota is declared **per ClusterQueue, per ResourceFlavor, per resource**:

- **`nominalQuota`** — guaranteed capacity for this ClusterQueue on this flavor. Always available to it.
- **`borrowingLimit`** — the maximum it may consume *above nominal* by borrowing other ClusterQueues'
  idle nominal quota within the cohort. Omitted/unset commonly means "unlimited within what the cohort
  has"; set it to bound a noisy tenant. `0` means no borrowing.
- **`lendingLimit`** — the maximum of its *own* unused nominal quota it will let the cohort borrow. Use
  to reserve guaranteed headroom for a tenant even when idle (it won't lend everything). Verify the
  feature gate / availability for your version.

**Guaranteed vs best-effort capacity:** nominal quota is guaranteed (a ClusterQueue can always reclaim
its nominal via cohort preemption). Anything borrowed above nominal is best-effort — it can be reclaimed
when the lender needs it back. Design so each tenant's *floor* = its nominal, and bursts ride on borrowing.

**Cohort arithmetic:** for a resource/flavor, a ClusterQueue's admissible amount ≈ its `nominalQuota`
plus the cohort's currently-idle lendable quota, capped by its `borrowingLimit`. Hierarchical cohorts
nest these pools so you can model org → team → sub-team guarantees.

**Flavor fungibility** (`flavorFungibility`) controls how flavor assignment traverses the flavor list
when the preferred flavor is full:

- `whenCanBorrow: Borrow | TryNextFlavor` — if the current flavor needs borrowing to fit, should we
  borrow here or try the next (cheaper-to-fit) flavor first?
- `whenCanPreempt: Preempt | TryNextFlavor` — if fitting the current flavor would require preemption,
  preempt here or try the next flavor first?

Typical config: prefer cheaper/spot flavors and only borrow/preempt on the on-demand flavor — encode
that by ordering flavors and setting fungibility to `TryNextFlavor` on the early ones.

## Scheduling, ordering, and fairness

**Queueing strategy** (per ClusterQueue):

- **`StrictFIFO`** — strict order by priority then timestamp. The head-of-line Workload blocks all
  others until it can be admitted (head-of-line blocking). Use when fairness/ordering matters more than
  utilization.
- **`BestEffortFIFO`** — same ordering, but if the head can't be admitted Kueue skips ahead to the next
  admissible Workload. Higher utilization; large jobs can be starved without preemption help. Common
  default for throughput-oriented fleets.

**Preemption** (`clusterQueue.spec.preemption`):

- **`withinClusterQueue`**: `Never` | `LowerPriority` | `LowerOrNewerEqualPriority`. Lets a higher-prio
  (or equal-prio-but-older, depending on setting) Workload evict lower-prio ones *in the same CQ*.
- **`reclaimWithinCohort`**: `Never` | `LowerPriority` | `Any`. Lets a CQ reclaim its nominal quota by
  preempting *borrowers* in the cohort (those running above their own nominal on borrowed quota).
- **`borrowWithinCohort`**: allows preempting cohort workloads in order to *borrow* (not just reclaim),
  gated by a priority threshold. Use sparingly — it can cause churn.
- Preemption evicts by **deleting the Workload's Pods / suspending the Job** (re-suspend), not killing
  Pods out of band; the evicted Workload requeues. Combine with `waitForPodsReady` so evicted gangs
  don't thrash.

**Fair Sharing** is the alternative to classic priority preemption: instead of strict priority, Kueue
balances *dominant resource share* (DRF-style) across ClusterQueues in a cohort, weighted by a per-CQ
`fairSharing.weight`. A CQ exceeding its fair share becomes a preemption target for an under-share CQ.
Enable cohort-wide via the Configuration `fairSharing` block. **Don't mix** the strict-priority and
fair-sharing mental models in one cohort — pick one fairness policy and design quotas around it.

## Gang, partial admission, and waitForPodsReady

- **Gang / all-or-nothing admission** is intrinsic: Kueue reserves quota for the *entire* Workload (all
  PodSets, full counts) atomically, or not at all. This is what prevents the classic partial-gang
  deadlock where two half-started distributed jobs each hold quota and neither can finish.
- **Partial admission** lets an *elastic* Workload declare a minimum count per PodSet (e.g. a job that
  can run with anywhere from 4 to 16 workers). Kueue admits at the largest count that fits, down to the
  minimum; below the minimum it stays pending. The integration/job must support shrinking. Verify
  support per integration.
- **`waitForPodsReady`** (Configuration-level, opt-in) adds **job-level readiness**: after unsuspend,
  Kueue waits up to a timeout for all Pods of the Workload to be Ready/Running. If they don't become
  ready (image pull failure, capacity vanished, node not ready), Kueue **re-suspends and requeues** with
  backoff (`requeuingStrategy`: backoff limit, base/cap delay, and timestamp ordering). Essential so a
  stuck gang releases quota instead of holding it forever. Tune `timeout`, `recoveryTimeout`, and
  `backoffLimitCount` to your image-pull/scale-up realities.

## Topology-Aware Scheduling (TAS)

For tightly-coupled training (all-reduce/all-gather bandwidth sensitive), placing a gang *spread across
racks/blocks* tanks throughput. TAS lets Kueue place a Workload's Pods within a **topology domain**.

- Define topology via the **Topology** object describing the hierarchy of node label keys
  (e.g. `cloud.provider.com/topology-block` → `...-rack` → `kubernetes.io/hostname`).
- Reference the Topology from a ResourceFlavor (`topologyName`).
- Workloads request placement with annotations on the Job's Pod template:
  `kueue.x-k8s.io/podset-required-topology: <level>` (hard — must fit in one domain at that level) or
  `kueue.x-k8s.io/podset-preferred-topology: <level>` (best-effort, fall back to a higher level).
- Kueue computes a fitting domain at admission and injects nodeAffinity so kube-scheduler lands Pods in
  it. TAS is comparatively young and evolving — **verify the annotation keys, feature gate, and whether
  TAS-only ClusterQueues are required** for your version before relying on it in production.

## ProvisioningRequest (just-in-time capacity)

`ProvisioningRequest` is a built-in **AdmissionCheck** that, before unsuspend, asks the
**cluster autoscaler** (notably on GKE — including DWS/flex-start and reserved capacity classes) to
provision the nodes the gang needs. Only when capacity is confirmed does admission complete and the Job
unsuspend — so a 64-Pod gang never starts onto nodes that don't exist yet (avoiding partial start +
re-suspend churn).

- Wire it with a `ProvisioningRequestConfig` (provisioningClassName, parameters, managedResources) and
  an `AdmissionCheck` of `controllerName: kueue.x-k8s.io/provisioning-request` referencing it; list that
  check on the ClusterQueue.
- Use it specifically for workloads that need *new* capacity (large gangs, scarce accelerators) rather
  than steady-state. Pair with `[[autoscaling-kubernetes]]` and `[[gke-master]]`.

## MultiKueue (multi-cluster dispatch)

MultiKueue spreads admission across clusters for multi-region / multi-cloud capacity and failover.

- A **manager** cluster runs Kueue and holds the LocalQueues/ClusterQueues users submit to. **Worker**
  clusters run the actual Jobs.
- Configure with a `MultiKueueCluster` (a kubeconfig Secret per worker) + a `MultiKueueConfig` (the list
  of worker clusters), exposed as an `AdmissionCheck` (`controllerName: kueue.x-k8s.io/multikueue`) on
  the ClusterQueue. (`ClusterProfiles`, the upstream multi-cluster API, may be used to source cluster
  connection info depending on version — verify.)
- On admission, the manager creates a **mirror Workload + the real Job** on a chosen worker; status is
  synced back. If a worker can't admit, MultiKueue tries another — giving **ABA-style failover** across
  regions/clouds. The job's Pods only ever run on the worker.
- The job integration must be MultiKueue-supported (Job, JobSet, and several others are). Verify the
  supported-integration matrix for your version.

## Integrations: how each maps to a Workload

Kueue manages an object only when (a) its integration is **enabled** in the Configuration
(`integrations.frameworks`) and (b) the object carries `kueue.x-k8s.io/queue-name` (or
`manageJobsWithoutQueueName` is on for matching namespaces). Each integration translates its object into
one Workload with appropriate PodSets:

| Object | PodSets | Notes |
|---|---|---|
| `batch/v1` Job | 1 PodSet (parallelism × template) | The canonical case; `suspend` toggled by Kueue. |
| JobSet (`jobset.x-k8s.io`) | one PodSet per ReplicatedJob | Best fit for multi-host training; see `[[jobset-leaderworkerset]]`. |
| RayJob / RayCluster (KubeRay) | head + worker PodSets | Worker groups become PodSets. |
| MPIJob (MPI Operator) | launcher + worker PodSets | Classic HPC/all-reduce. |
| PyTorchJob / TFJob / others (Kubeflow Trainer / Training Operator) | per replica role | Master/Worker/PS roles → PodSets. |
| LeaderWorkerSet | leader + worker PodSets | Multi-host inference/training; see `[[jobset-leaderworkerset]]`. |
| Plain Pods / **pod groups** | one PodSet (group = gang) | Use `kueue.x-k8s.io/pod-group-name` + `pod-group-total-count` for gangs of bare Pods. |
| AppWrapper | wraps arbitrary objects into one Workload | For composite/legacy workloads. |

Common labels/annotations: `kueue.x-k8s.io/queue-name` (opt-in), `kueue.x-k8s.io/priority-class`
(WorkloadPriorityClass), `kueue.x-k8s.io/pod-group-name` / `...pod-group-total-count` (plain-Pod gangs),
TAS annotations above. Kueue also adds `kueue.x-k8s.io/managed: "true"` and protective **finalizers** on
managed objects/Pods so it can release quota on terminal state.

## Configuration API

Kueue is configured via a `Configuration` (`config.kueue.x-k8s.io`) passed to the manager (ConfigMap).
The fields that bite people:

- **`integrations.frameworks`** — the allowlist of object kinds Kueue manages. If your Jobs aren't being
  suspended, the integration is probably not enabled here (or the CRD/operator isn't installed).
- **`manageJobsWithoutQueueName`** — if `true`, Kueue manages matching jobs *even without* a queue label.
  Fleet-wide this is dangerous: it suspends everything (including system/operator jobs) until queued. Use
  `managedJobsNamespaceSelector` to scope it. Default off; usually keep it off.
- **`waitForPodsReady`** — readiness/requeue behavior (above).
- **`fairSharing`** — enable + preemption strategy for fair sharing.
- Feature gates (`--feature-gates` on the manager) toggle alpha/beta features (TAS, partial admission,
  hierarchical cohorts, etc.). **Verify gate names and default state per version.**

## Operations & troubleshooting

Diagnose at the **Workload**, not the Job:

```bash
kubectl get workloads -n <ns>                       # ADMITTED, AGE
kubectl describe workload <wl> -n <ns>              # conditions: QuotaReserved / Admitted / reasons
kubectl get clusterqueue <cq> -o yaml               # status.flavorsUsage / pendingWorkloads / admitted
kubectl describe clusterqueue <cq>                  # events, borrowing
kubectl get localqueue -n <ns>                      # pending/admitted counts per namespace
```

Symptom → diagnosis → fix:

- **Job stays Suspended, no Pods** → check the Workload's `Admitted`/`QuotaReserved` conditions. If
  Pending, read the message: usually quota exhausted, no matching flavor, or a pending AdmissionCheck.
- **`Inadmissible` / "couldn't assign flavors"** → (1) no flavor's `nodeLabels` match any node, or
  `nodeTaints` aren't tolerated by the PodSet; (2) request exceeds nominal+borrowable quota;
  (3) `resourceGroups` doesn't cover a requested resource (e.g. you request `nvidia.com/gpu` but the CQ
  only covers `cpu`/`memory`). Fix the flavor labels/taints, the quota, or the resourceGroups coverage.
- **Stuck pending AdmissionCheck** → ProvisioningRequest can't get capacity (quota/region exhausted), or
  MultiKueue worker unreachable. Inspect the ProvisioningRequest/MultiKueueCluster objects.
- **Admitted but Pods never Ready, then re-suspended on a loop** → `waitForPodsReady` requeuing on a
  real failure (bad image, missing volume, node pressure). Look at the actual Pods/events; fix the root
  cause or the requeue backoff is masking it.
- **Quota "leaked" — ClusterQueue shows usage but no running jobs** → orphaned Workloads with finalizers,
  often because someone stripped the finalizer or force-deleted Jobs. List Workloads, find terminal ones
  still holding admission, and let Kueue reconcile (don't hand-edit finalizers unless recovering).
- **Everything suspended cluster-wide after install** → `manageJobsWithoutQueueName: true` without a
  namespace selector. Scope or disable it.

**Metrics & dashboards:** Kueue exposes Prometheus metrics — pending/admitted/evicted workloads per
ClusterQueue, quota usage vs nominal, admission latency, preemptions. Build per-cohort utilization and
pending-by-reason dashboards; alert on sustained pending + low utilization (misconfigured flavors) and on
borrowing pinned at limit (under-provisioned tenant). Verify exact metric names for your version.

## Anti-patterns & gotchas

- **Stripping `kueue.x-k8s.io/...` finalizers or labels by hand.** Finalizers gate quota release;
  removing them leaks quota and wedges the CQ. The queue-name label is the opt-in — remove it *before*
  creation to unmanage, never mid-flight.
- **Treating Kueue as a Pod scheduler.** It assigns flavors and quota, then unsuspends; node placement is
  kube-scheduler's job. TAS only *injects affinity*; it doesn't bind Pods.
- **`manageJobsWithoutQueueName: true` fleet-wide.** Suspends operator/system jobs. Scope it.
- **Mixing fair-sharing and strict-priority** semantics in one cohort. Pick one.
- **Unbounded `borrowingLimit` everywhere** → one tenant starves the cohort; no tenant has a real floor.
  Set lending/borrowing limits to encode the fairness SLA.
- **Deep/overlapping cohort topologies.** Hard to reason about; keep borrowing shallow and intentional.
- **Forgetting `waitForPodsReady`** on accelerator gangs — a half-scheduled gang holds quota until manual
  intervention.
- **Wrong PriorityClass axis** — using Pod `PriorityClass` expecting queue ordering, or
  `WorkloadPriorityClass` expecting in-node preemption. They are different mechanisms.
- **One giant ClusterQueue for all teams** — no isolation, no per-tenant guarantee. Use a CQ per tenant
  in a shared cohort.

## Sizing & design guidance

- **One Cohort per fungible capacity pool** (e.g. all clusters' A100s that can substitute for each
  other). ClusterQueues *within* it borrow/lend.
- **ClusterQueue per team/tenant.** Nominal = the team's guaranteed floor; borrowingLimit = how far it
  may burst; lendingLimit = how much floor it keeps even when idle.
- **ResourceFlavor per distinct hardware/cost class** (`a100-ondemand`, `a100-spot`, `h100`, `tpu-v5e`)
  with accurate `nodeLabels`/`nodeTaints`. Order them cheap→expensive and use `flavorFungibility` to try
  cheap first.
- **Reserve headroom with lendingLimit** rather than zero borrowing — lets idle quota be useful while
  guaranteeing the owner can reclaim.
- **Choose strategy per CQ:** BestEffortFIFO for throughput pools; StrictFIFO where ordering/fairness to
  large jobs matters (with preemption to avoid starving them).
- **Right-size waitForPodsReady** timeouts to real image-pull + autoscale times; too short = thrash, too
  long = slow quota recovery.

## Version awareness

Kueue is pre-1.0 and graduates features quickly: TAS, hierarchical cohorts, partial admission,
MultiKueue, fair sharing, and various AdmissionChecks have moved across alpha/beta and changed defaults.
Before relying on a specific field, feature gate, default, or annotation key: check the installed
chart/version, the Configuration in use, and the docs/CRD for that version. Prefer general statements
over invented specifics when unsure.

## Rationalizations & rebuttals

- *"The Workload's wedged — just strip the finalizer to unstick it."* The finalizer is what releases
  reserved quota back to the ClusterQueue/cohort. Hand-removing it leaks quota and silently shrinks the
  cohort. Diagnose the terminal Workload and let Kueue reconcile; edit finalizers only as last-resort
  recovery, then fix the root cause that orphaned it.
- *"One giant ClusterQueue for everyone is simpler."* It removes the only thing Kueue gives you:
  per-tenant guarantees and isolation. No nominal floor, no borrowing SLA, one noisy job starves all.
  A CQ per tenant in a shared cohort is barely more config and is the actual design.
- *"Skip TAS for this tightly-coupled job — it'll schedule fine."* Without topology constraints
  kube-scheduler can spread a gang across racks/blocks and tank all-reduce/all-gather bandwidth. The job
  "runs" but at a fraction of throughput. Use `podset-required/preferred-topology` for bandwidth-sensitive
  gangs.
- *"Borrowing limits don't matter, there's plenty of capacity."* Unbounded `borrowingLimit` means the
  first tenant to burst can consume the whole cohort, and no tenant has a real floor until contention
  hits — exactly when it's too late. Set borrowing/lending limits up front to encode the fairness SLA.
- *"Don't bother with `waitForPodsReady` — Pods come up fine."* Until the day an image pull fails or
  capacity vanishes mid-scale-up; then a half-scheduled accelerator gang holds quota indefinitely and
  blocks every other Workload. Enable it so stuck gangs re-suspend and release quota.
- *"I'll just set Pod `PriorityClass` for queue ordering."* That axis drives in-node kube-scheduler
  preemption, not Kueue queue ordering or preemption. Use `WorkloadPriorityClass` for queueing; they are
  separate mechanisms and conflating them silently does nothing you intended.
- *"`manageJobsWithoutQueueName: true` is convenient — auto-manage everything."* Fleet-wide it suspends
  operator/system jobs until they're queued, which they never will be. Keep it off, or scope it tightly
  with `managedJobsNamespaceSelector`.

## Red flags

- Workloads sitting `Inadmissible` / Pending for long periods and nobody reading the condition message
  (flavor mismatch vs quota exhausted vs pending AdmissionCheck are different fixes).
- Anyone hand-editing `kueue.x-k8s.io/...` finalizers or labels on live Workloads/Jobs/Pods.
- A ClusterQueue showing quota usage with no corresponding running jobs (leaked/orphaned Workloads).
- No `preemption` policy configured on a contended ClusterQueue/cohort — large jobs silently starve and
  nominal quota can't be reclaimed.
- `borrowingLimit` unset (effectively unbounded) on every CQ, so no tenant has an enforceable floor.
- Quota/cohort numbers that don't reconcile with real node capacity (nominal sums far exceed installed
  accelerators, or a flavor's `nodeLabels` match zero nodes).
- Accelerator gangs running without `waitForPodsReady`, or with timeouts shorter than real image-pull +
  autoscale times (admission thrash) or far longer (slow quota recovery).
- Fair-sharing and strict-priority preemption semantics mixed in one cohort.
- Borrowing pinned at the limit for a tenant for sustained periods (under-provisioned floor) while another
  cohort sits idle.

## Verification gate (definition of done)

Before calling a Kueue setup complete, confirm each:

- **Plumbing wired:** ResourceFlavor(s) with accurate `nodeLabels`/`nodeTaints`, a ClusterQueue with
  `resourceGroups` covering every requested resource (CPU/mem *and* `nvidia.com/gpu`/TPU), and a
  LocalQueue in each tenant namespace pointing at it.
  `kubectl get resourceflavor,clusterqueue,localqueue -A`.
- **Integration enabled:** the job kind is in `integrations.frameworks` and the test Job carries
  `kueue.x-k8s.io/queue-name`; Kueue forces `spec.suspend: true` and creates a Workload.
- **Admission works:** submit a real job; the Workload reaches `QuotaReserved` then `Admitted`,
  `status.admission` shows the assigned flavor/quota, and the Job unsuspends (`spec.suspend: false`) with
  Pods scheduling. `kubectl describe workload <wl> -n <ns>`.
- **Gang admission verified:** submit a multi-PodSet job larger than free quota; confirm it is admitted
  all-or-nothing (no partial start) and that a fitting one admits atomically.
- **Quota & borrowing correct:** usage on the CQ matches expectation; borrowing respects `borrowingLimit`,
  a lender keeps its `lendingLimit` floor, and a higher-priority job reclaims nominal via preemption (if
  configured). `kubectl get clusterqueue <cq> -o yaml` → `status.flavorsUsage`.
- **Resilience:** with `waitForPodsReady` on, a job that can't become Ready re-suspends and requeues
  (quota released), not wedged forever.
- **TAS (if used):** a `required/preferred-topology` job lands within the intended domain (nodeAffinity
  injected) — verify against current annotation keys/feature gate at https://kueue.sigs.k8s.io.
- **Observability:** Kueue Prometheus metrics scraped; pending-by-reason and per-cohort utilization
  visible; alerts on sustained pending + low utilization and on borrowing pinned at limit. Verify exact
  metric names for the installed version.

## Canonical references

- Project docs: https://kueue.sigs.k8s.io (Concepts, Tasks, Admission, TAS, MultiKueue, Configuration).
- Source: https://github.com/kubernetes-sigs/kueue
- KEPs: https://github.com/kubernetes-sigs/kueue/tree/main/keps (design docs for cohorts, fair sharing,
  TAS, MultiKueue, ProvisioningRequest, partial admission).
- API reference: https://kueue.sigs.k8s.io/docs/reference/kueue.v1beta1/ (verify the version path).
- ProvisioningRequest / autoscaler: cluster-autoscaler and GKE DWS docs.
