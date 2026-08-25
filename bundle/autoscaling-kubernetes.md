---
name: autoscaling-kubernetes
description: Expert mastery of autoscaling on Kubernetes across all layers — pod-horizontal (HPA),
  pod-vertical (VPA, in-place resize), node (Cluster Autoscaler, Karpenter, GKE Node Auto-Provisioning),
  and event-driven (KEDA). Use when designing or debugging HPA control loops (desiredReplicas, behavior,
  stabilization windows), custom/external metrics (custom.metrics.k8s.io, external.metrics.k8s.io,
  Prometheus Adapter), VPA modes and HPA-vs-VPA conflicts, Cluster Autoscaler vs Karpenter (NodePools,
  consolidation, disruption), KEDA ScaledObjects/ScaledJobs and scale-to-zero, Kueue ProvisioningRequest,
  or ML/GPU/LLM-inference autoscaling (GPU utilization, queue depth, TTFT/concurrency, scale-to-zero for
  accelerators, multi-host LWS). Covers tuning, metric-pipeline reliability, and autoscaler fights.
---

# Autoscaling on Kubernetes

Apply the judgment of a platform engineer who has tuned autoscaling for large multi-tenant and
ML/GPU/TPU fleets in production for years — where a bad stabilization window costs money on idle GPUs
and a metric-pipeline outage silently freezes scaling during an incident.

## How to use this skill

1. **Read `autoscaling-kubernetes-guide.md`** in this directory — the full reference (three axes, the
   HPA algorithm, VPA, Cluster Autoscaler, Karpenter, KEDA, GKE NAP/Kueue, ML/GPU scaling, tuning).
   Apply it to the task. For concrete, imitate-ready manifests (HPA with external metrics + `behavior`,
   a KEDA queue-driven `ScaledObject` with scale-to-zero, a Karpenter `NodePool`), read **`examples.md`**.
2. Match the surrounding cluster's conventions (existing metric pipeline, node-provisioner choice,
   namespace quotas); apply the correctness/safety rules regardless.
3. Before declaring autoscaling work done: confirm the metric pipeline actually serves the metric
   (`kubectl get --raw /apis/external.metrics.k8s.io/v1beta1/...`), that HPA and VPA never target the
   same resource, that PodDisruptionBudgets won't deadlock scale-down, and that scale-to-zero paths
   have an activation source.

## Essentials (full detail in `autoscaling-kubernetes-guide.md`)

- **Three independent axes, composed deliberately.** HPA changes replica *count*, VPA changes per-pod
  *requests*, node autoscalers change *capacity*. HPA + VPA must **not** drive the same resource
  (CPU/mem) — they fight; use VPA for the dimension HPA doesn't scale on, or use GKE Multidim Pod
  Autoscaling. Node scaling reacts to what pod scaling produces (Pending pods).
- **HPA algorithm:** `desiredReplicas = ceil(currentReplicas × currentMetric / targetMetric)`, gated by
  a tolerance (default 10%) and recomputed every `--horizontal-pod-autoscaler-sync-period` (default 15s).
  With multiple metrics, HPA takes the **max** desired across them.
- **Resource metrics need metrics-server; custom/external need an adapter.** CPU/mem flow via
  `metrics.k8s.io` (metrics-server). Pod/object metrics via `custom.metrics.k8s.io`, queue depth and
  other cluster-external signals via `external.metrics.k8s.io` — typically the **Prometheus Adapter** or
  KEDA's metrics adapter. No adapter, no scaling.
- **`behavior` is how you stop thrash.** Tune `scaleUp`/`scaleDown` policies (Pods/Percent per period),
  `stabilizationWindowSeconds` (scale-down defaults to 300s, scale-up 0s), and `selectPolicy`. For
  expensive/slow-to-warm workloads, scale up fast and down slow.
- **Set `Ready`/readiness correctly or HPA misreads load.** During cold start, unready pods are excluded
  and their metrics ignored; with bad probes HPA over- or under-scales. Use the right `--metric-resolution`
  and account for metric lag.
- **VPA modes:** `Off` (recommend only — the safe default), `Initial` (set at admission), `Recreate`/`Auto`
  (evict to resize). Classic `Auto` **evicts** pods to apply new requests. **In-place pod resize**
  (resizing without restart) is the modern path — note its maturity per your K8s version and **verify**
  before relying on it for production VPA.
- **Cluster Autoscaler** scales up when pods are `Pending` for lack of capacity and scales down nodes
  under a utilization threshold after a cooldown; it respects PDBs, `cluster-autoscaler.kubernetes.io/
  safe-to-evict: "false"`, and local-storage pods. Expanders: `random`/`most-pods`/`least-waste`/`priority`.
- **Karpenter** provisions right-sized nodes just-in-time from a flexible instance-type set, then
  **consolidates** (bin-packs, replaces, removes empty/underutilized nodes) and handles **drift**. Mark
  pods `karpenter.sh/do-not-disrupt: "true"` to protect them. Prefer it over CA where node-group sprawl
  or bin-packing efficiency hurts.
- **KEDA** does event-driven scaling and **scale-to-zero**: a `ScaledObject` wraps your Deployment and
  generates an HPA under the hood; KEDA's scalers (Kafka, SQS/queues, Prometheus, cron, etc.) feed
  `external.metrics`. The **activation** threshold (0→1) is separate from the scaling metric (1→N).
- **ML/GPU scaling:** CPU is the wrong signal for LLM serving — scale on **queue depth / pending
  requests / concurrency / TTFT**, exposed via Prometheus + KEDA/HPA. Scale-to-zero saves expensive
  accelerators but cold start (image pull, weight load) is brutal — mitigate or keep warm. Multi-host
  serving scales by replica-group (LWS), not by pod. Batch capacity flows through Kueue, not HPA.
- **The classic failure modes:** autoscaler fights (HPA vs VPA, two controllers on one Deployment),
  metric-pipeline outages freezing scaling, PDBs blocking scale-down, slow node provisioning under load,
  and accelerator scarcity making "scale up" mean "wait."

## Related skills

- `[[gke-master]]` — GKE Node Auto-Provisioning, Autopilot, cluster autoscaler profiles, TPU/GPU pools.
- `[[kueue-advanced]]` — queue/quota-aware capacity and `ProvisioningRequest` for gang-scheduled batch.
- `[[serving-frameworks]]` — vLLM/SGLang/Triton/KServe metrics (queue, TTFT) that drive inference scaling.
- `[[aiml-on-kubernetes]]` — umbrella for training/inference on K8s; where autoscaling fits the ML stack.
- `[[jobset-leaderworkerset]]` — LWS/JobSet multi-host groups that scale as a unit.
- `[[kubernetes-expert]]` — general K8s practitioner context (workloads, scheduling, resources).
- `[[kubernetes-internals-expert]]` — scheduler/kubelet internals behind Pending pods and in-place resize.

---

# Reference — autoscaling-kubernetes

# Autoscaling on Kubernetes — Deep Reference

Autoscaling on Kubernetes is not one feature; it is **four control loops on three orthogonal axes** that
you compose into a system. Getting it right means each loop has a clean signal, the loops don't fight,
and capacity arrives before SLOs break — without paying for idle GPUs. This guide is the source of
truth; `examples.md` has imitate-ready manifests.

It is 2026 and this ecosystem moves fast. Treat every version-, flag-, and maturity-claim here as
"verify against current upstream docs for your cluster version." Where a feature's stability is in
flux (in-place pod resize, MPA, some KEDA scalers), this guide says so explicitly.

## Mental model: three axes + an event layer

| Axis | What it changes | Controller(s) | Reacts to |
|------|-----------------|---------------|-----------|
| Horizontal pods | replica **count** | HPA (and KEDA, which drives an HPA) | a target metric vs current |
| Vertical pods | per-pod **requests/limits** | VPA (recommender/updater/admission) | historical + live usage |
| Nodes | cluster **capacity** | Cluster Autoscaler / Karpenter / GKE NAP | unschedulable Pending pods; node utilization |

The flow is a pipeline, not a hierarchy:

```
metric ↑  →  HPA/KEDA add replicas  →  new pods Pending (no room)  →  node autoscaler adds nodes
                                                                   ↘ pods schedule, serve traffic
metric ↓  →  HPA/KEDA remove replicas →  nodes underutilized        →  node autoscaler removes nodes
```

Two design rules fall out immediately:

1. **Pod autoscaling and node autoscaling are decoupled by the scheduler.** Node autoscalers never
   look at your metric; they look at *Pending pods* and *node utilization*. So your HPA/VPA must produce
   schedulable demand (correct requests, correct affinities) or the node layer can't help.
2. **Never point two loops at the same dimension.** HPA and VPA both targeting CPU is the canonical
   self-inflicted outage: HPA adds replicas to lower per-pod CPU, VPA lowers requests to match usage,
   HPA's percentage-of-request jumps, it scales again — oscillation. Choose one loop per resource.

## HPA — horizontal pod autoscaler

### Control loop and algorithm

The HPA controller (in `kube-controller-manager`) reconciles every
`--horizontal-pod-autoscaler-sync-period` (**default 15s**). Each tick it reads the current metric for
the target's pods and computes:

```
desiredReplicas = ceil( currentReplicas × ( currentMetricValue / desiredMetricValue ) )
```

For **utilization** targets the metric is `sum(usage)/sum(requests)` across ready pods (so *requests
must be set* or utilization HPA is undefined). Guards on top of the raw formula:

- **Tolerance** — if the ratio is within `±` the tolerance of 1.0, HPA does nothing. Default **10%**
  globally (`--horizontal-pod-autoscaler-tolerance=0.1`); on newer clusters this is configurable
  per-direction via `behavior.scaleUp/scaleDown.tolerance` (alpha/beta-gated — **verify** for your
  version). This deadband is the first line of defense against thrash.
- **Readiness / missing metrics** — pods that are not `Ready`, are still in their initial readiness
  window, or have missing metrics are handled conservatively: unready pods are assumed at 0% when
  scaling up and at 100% when scaling down, so HPA won't over-react to cold-starting pods.
- **Multiple metrics** — HPA computes `desiredReplicas` for each metric independently and takes the
  **max**. This means *any one hot metric scales you up*; design metrics so that's the intent.

### Metric sources — and their adapters

| API group | Examples | Source you must install |
|-----------|----------|--------------------------|
| `metrics.k8s.io` (Resource) | CPU, memory | **metrics-server** |
| `custom.metrics.k8s.io` (Pods/Object) | requests/sec per pod, queue length on an Object | an adapter, usually **Prometheus Adapter**, or KEDA |
| `external.metrics.k8s.io` (External) | SQS depth, Kafka lag, PubSub backlog, anything off-cluster | **Prometheus Adapter** or **KEDA** |

The HPA `metric.type` is `Resource`, `Pods`, `Object`, `External`, or `ContainerResource` (per-container
utilization — prefer it over `Resource` for multi-container pods where one container dominates).
**No adapter ⇒ no metric ⇒ HPA reports `unknown` and won't scale.** Verify the pipeline directly:

```bash
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | jq .
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/<metric>" | jq .
kubectl describe hpa <name>   # look at the conditions: AbleToScale, ScalingActive, ScalingLimited
```

### `behavior` — stabilization and rate policies

`spec.behavior` is where you encode "scale up fast, scale down slow" and kill thrash:

```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 0        # react immediately to load spikes
    selectPolicy: Max
    policies:
      - type: Percent
        value: 100                       # at most double per period
        periodSeconds: 60
      - type: Pods
        value: 4                          # ...or +4 pods, whichever Max selects
        periodSeconds: 60
  scaleDown:
    stabilizationWindowSeconds: 300      # default; consider the max metric over last 5 min
    selectPolicy: Max
    policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

- **`stabilizationWindowSeconds`** makes HPA use the *most-scale-out-favorable* recommendation over the
  window: for scale-down it takes the **max** desired over the window (prevents premature shrink on a
  metric dip). Default **300s for scale-down, 0s for scale-up**.
- **Policies** cap the *rate* of change (Pods or Percent per `periodSeconds`); `selectPolicy: Max/Min/
  Disabled`. `Disabled` on `scaleDown` pins the direction off entirely (useful for stateful or
  expensive pods you only ever scale up automatically).

### HPA pitfalls

- **Cold start / warm-up.** If pods take 60s to be useful, a 0s scale-up window plus aggressive policy
  floods you with pods that aren't serving yet, metric stays high, you over-scale. Fix with readiness
  gates, `initialReadinessDelay`/`cpuInitializationPeriod` awareness, and modest scale-up policies.
- **Metric lag.** Prometheus scrape interval + adapter rate window + HPA sync period stack up; a 30s
  scrape and a 5-min rate() makes HPA blind to fast spikes. Match windows to your latency budget.
- **Thrash from a noisy metric.** Use tolerance + stabilization; smooth the metric in the adapter
  (`rate`/`avg_over_time`) rather than feeding raw gauges.
- **`ScalingLimited: true`** in `describe hpa` means min/max replicas is binding — you're not actually
  autoscaling, you're pinned. Check it before trusting the loop.
- **HPA can't scale a workload with no `requests`** on a utilization target, and won't manage a
  Deployment that another controller also writes `.spec.replicas` on (last writer wins → fight).

## VPA — vertical pod autoscaler

VPA right-sizes **requests** (and optionally limits) from observed usage. It is three components:

- **Recommender** — watches usage history (and metrics-server), produces `target`/`lowerBound`/
  `upperBound`/`uncappedTarget` recommendations.
- **Updater** — decides which running pods are mis-sized and, in evicting modes, **evicts** them so they
  come back resized.
- **Admission controller** — a mutating webhook that rewrites pod requests at creation time.

### Modes (`updatePolicy.updateMode`)

| Mode | Behavior | Use it for |
|------|----------|-----------|
| `Off` | Recommend only; never mutates pods | **The safe default** — read recommendations, set requests yourself / in CI |
| `Initial` | Sets requests only at pod creation; never disrupts running pods | Set-and-forget sizing without eviction churn |
| `Recreate` | Evicts and recreates pods to apply new requests | Legacy; disruptive |
| `Auto` | Currently = `Recreate` (evicts) until in-place is the default | Non-critical workloads tolerant of restarts |

**Why `Auto` evicts:** historically the only way to change a pod's requests was to recreate it, so the
updater deletes the pod (respecting PDBs) and the admission controller resizes the replacement. This is
why naive VPA `Auto` on a singleton or a tight-PDB service causes availability surprises.

### In-place pod resize (the modern vertical path)

Kubernetes added **in-place resource resize** (`resize` subresource; `resizePolicy` per container with
`RestartPolicy: NotRequired`/`RestartContainer`) so requests/limits can change **without recreating the
pod**. VPA is moving toward using this so vertical scaling stops meaning "evict." **Maturity caveat:**
the in-place resize feature graduated through alpha/beta over recent releases and VPA's in-place support
is itself evolving — **do not assume it's GA or default for your cluster; verify the feature-gate state
and VPA version before relying on it in production.** Until then, treat `Auto`/`Recreate` as disruptive.

### VPA × HPA interaction

- **Never run VPA and HPA on the same metric** (both on CPU, both on memory) — they oscillate.
- The supported pattern: **HPA on a custom/external metric (e.g. RPS, queue depth), VPA on CPU+memory**
  — each owns a different dimension.
- **GKE Multidimensional Pod Autoscaling (MPA)** scales horizontally on CPU *and* vertically on memory
  in one object, sidestepping the conflict. MPA is GKE-specific; **verify its current availability/limits**
  on your GKE version (see `[[gke-master]]`).

## Cluster Autoscaler (CA)

CA adds and removes **nodes** by reasoning about the scheduler, not metrics.

- **Scale up:** when pods are `Pending` because no node can fit them, CA simulates which **node group**
  (an ASG / MIG / equivalent) would make them schedulable and increases that group's size. It only
  considers groups whose template would actually satisfy the pods' requests, affinities, taints, and
  topology.
- **Scale down:** a node is a candidate when its utilization is below
  `--scale-down-utilization-threshold` (default 0.5) and all its pods can move elsewhere. After
  `--scale-down-unneeded-time` (default 10m) it's drained and removed. Scan loop runs every
  `--scan-interval` (default 10s).

**Expanders** decide *which* node group to grow on scale-up:

| Expander | Picks the group that… |
|----------|------------------------|
| `random` | any (default-ish, cheap) |
| `most-pods` | schedules the most pending pods |
| `least-waste` | leaves the least idle CPU/mem after the pods land (bin-packing) |
| `priority` | matches your configured priority list (a ConfigMap) — common for spot-then-on-demand |

You can chain them (`--expander=priority,least-waste`).

**Scale-down protections** (the things that block node removal):

- A **PodDisruptionBudget** that would be violated by the eviction.
- `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"` on a pod.
- Pods with **local storage** (`emptyDir`) unless annotated `safe-to-evict: "true"`.
- kube-system pods without a PDB, bare pods (not backed by a controller), pods with restrictive
  affinity. Annotate a whole node un-removable with `cluster-autoscaler.kubernetes.io/scale-down-disabled`.

**Limitations:** CA is **node-group-shaped** — every group is one instance type/zone template, so
heterogeneous needs mean many groups and the expander does coarse choices. It does not bin-pack across
groups, can be slow to converge, and assumes group instances are interchangeable.

## Karpenter — just-in-time node provisioning

Karpenter replaces node-group plumbing with **direct, right-sized provisioning** and continuous
**consolidation**. Two CRDs:

- **`NodePool`** — the scheduling constraints and disruption policy: allowed instance types/families,
  capacity type (spot/on-demand), arch/zones via `requirements`, `limits` (cap total CPU/mem/GPU),
  `weight`, and a `disruption` block.
- **`EC2NodeClass`/`NodeClass`** (provider-specific) — the node's infra: AMI/image family, subnets,
  security groups, IAM/instance profile, block devices, user data.

When pods are Pending, Karpenter looks at *all* allowed instance types at once and launches a node (or
a few) that fits them **and** is cost-efficient — no pre-defined group needed. It can pick across many
shapes in one decision, which is why bin-packing beats CA on mixed fleets.

**Disruption / consolidation** (`spec.disruption`):

- **`consolidationPolicy: WhenEmptyOrUnderutilized`** — remove empty nodes and *replace/merge*
  underutilized ones with cheaper or fewer nodes. `WhenEmpty` only removes fully empty nodes.
- **`consolidateAfter`** — how long a node must be a candidate before action (debounce).
- **Drift** — when a node no longer matches its NodePool/NodeClass (e.g. AMI bumped, requirements
  changed), Karpenter replaces it. Powerful and dangerous: a NodeClass edit can roll your whole fleet.
- **`budgets`** — cap how many nodes Karpenter may disrupt at once (by count or %), per the schedule —
  the safety valve against mass churn.
- **Protect a pod** with `karpenter.sh/do-not-disrupt: "true"` (annotation) so consolidation/drift won't
  evict it mid-flight; set it on long batch jobs, stateful pods, and anything restart-hostile.

**Karpenter vs Cluster Autoscaler:**

| | Cluster Autoscaler | Karpenter |
|---|---|---|
| Capacity unit | pre-defined node groups | any allowed instance type, on demand |
| Bin-packing | per-group, coarse | global, fine — picks the cheapest fitting shape |
| Consolidation | basic scale-down | active replace/merge + drift |
| Ops overhead | manage N groups | manage NodePools/NodeClasses |
| Maturity/portability | very mature, all clouds | strongest on AWS; other providers vary — **verify** |

Prefer Karpenter where instance diversity and cost matter; CA where you need its maturity, a provider
without solid Karpenter support, or tight node-group control. Don't run both on the same nodes.

## KEDA — event-driven autoscaling

KEDA (`keda.sh`) adds **event-driven** scaling and **scale-to-zero** on top of HPA. You declare a
`ScaledObject` (for Deployments/StatefulSets) or `ScaledJob` (for Jobs); KEDA's operator **creates and
manages an HPA** for you, and its metrics adapter serves the scaler's value over `external.metrics.k8s.io`.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
spec:
  scaleTargetRef: { name: worker }
  minReplicaCount: 0          # scale to zero when idle
  maxReplicaCount: 50
  cooldownPeriod: 300         # wait before scaling back to zero
  triggers:
    - type: aws-sqs-queue
      metadata: { queueURL: ..., queueLength: "5" }
```

- **Scalers ecosystem** (60+): Kafka lag, SQS/RabbitMQ/Azure Service Bus/PubSub queue depth, Prometheus
  (any PromQL), `cron` (time-of-day capacity), Redis lists, GCP/Azure metrics, and more.
- **Scale-to-zero** is KEDA's headline capability — HPA alone can't go to 0 replicas. KEDA scales the
  workload to/from zero and only hands off to the HPA for the `1..N` range.
- **Activation vs scaling — the distinction that trips everyone up.** The trigger's main threshold drives
  `1→N`. Going from `0→1` is governed by the **activation** threshold
  (`activationQueueLength`/`activation...` per scaler, default 0). The metric must cross *activation* to
  wake the workload, then *scaling* governs how many replicas. Set activation deliberately or a single
  stray message wakes an expensive deployment.
- **`ScaledJob`** is for queue-driven *batch*: KEDA creates Jobs (not long-running replicas) per
  backlog — the right tool when each item is a discrete unit of work rather than a stream into a service.

Because KEDA drives an HPA, all the HPA `behavior`/stabilization knowledge still applies (set
`advanced.horizontalPodAutoscalerConfig.behavior`).

## GKE specifics

See `[[gke-master]]` for depth. The autoscaling-relevant pieces:

- **Node Auto-Provisioning (NAP)** — GKE creates and deletes **whole node pools** automatically based on
  Pending pods' needs (including the right GPU/TPU shape), instead of you predefining every pool. It's
  CA-driven NAP on Standard; on **Autopilot** node management is fully implicit (you only set pod
  requests). Cluster autoscaler **profiles** (`balanced` vs `optimize-utilization`) tune scale-down
  aggressiveness.
- **Kueue `ProvisioningRequest`** — queue/quota-aware **capacity provisioning** for batch. Kueue (see
  `[[kueue-advanced]]`) holds a workload until capacity can be *atomically* provisioned via a
  `ProvisioningRequest` (backed by NAP), so a gang-scheduled job doesn't partially start and waste
  accelerators waiting for the rest. This is the correct path for **batch/training capacity** —
  fundamentally different from HPA, which reacts to live metrics on already-running services.

## ML / GPU / inference autoscaling

This is where generic autoscaling advice fails and you need workload-aware signals.

- **CPU utilization is the wrong signal for LLM/GPU serving.** A vLLM/SGLang replica can be CPU-idle
  while the GPU is saturated and the request queue is growing. Scale on **load metrics the server
  exposes**: pending/queued requests, running-vs-waiting sequences, **KV-cache utilization**,
  concurrency, or **TTFT/p95 latency**. Pull these via Prometheus and feed HPA (custom/external) or
  KEDA. See `[[serving-frameworks]]` for which metrics vLLM/SGLang/Triton/Dynamo/KServe expose.
- **Why request-rate / concurrency / queue beats CPU:** they correlate directly with the SLO (latency)
  and with the bottleneck (GPU compute + KV cache), and they react *before* CPU would (CPU may never
  move). Target a concurrency or queue-depth that keeps p95 under budget.
- **Scale-to-zero for accelerators** (via KEDA) saves real money on expensive/scarce GPUs and TPUs, but
  **cold start is the tax**: pulling a multi-GB image and loading tens of GB of weights can take minutes.
  Mitigate with image streaming/preloaded images, model caching on fast local disk, keeping a warm
  `minReplicaCount: 1` for latency-sensitive paths, or a small always-on tier + KEDA burst tier.
- **Multi-host inference scales by group, not by pod.** A model sharded across N hosts is one logical
  replica — use **LeaderWorkerSet** (`[[jobset-leaderworkerset]]`) so the autoscaler adds/removes an
  entire N-host group atomically. Plain HPA on a Deployment would add single pods that can't serve alone.
- **Batch vs serving split:** **serving** scales reactively (HPA/KEDA on queue/latency); **batch/training
  capacity** is provisioned through **Kueue** quotas + `ProvisioningRequest` (gang, all-or-nothing),
  *not* HPA. Mixing them — e.g. HPA-ing a training job — wastes accelerators on partial gangs.
- **Accelerator scarcity** means "scale up" often equals "wait for capacity." Plan for it: spot+on-demand
  fallback (Karpenter NodePool weights / CA priority expander), reservations, multi-zone/region spread,
  and Kueue to queue rather than fail. See `[[aiml-on-kubernetes]]` for the end-to-end picture.

## Tuning, reliability, and the autoscaler-fights catalog

- **Stabilization windows are your thrash control.** Scale up fast (short window, generous policy), scale
  down slow (≥300s, conservative percent) — especially for warm-up-heavy or expensive pods.
- **The metric pipeline is a hard dependency.** If metrics-server / Prometheus / the adapter is down, HPA
  reports `unknown` and **silently stops scaling** — exactly when you may need it most during an incident.
  Alert on HPA `ScalingActive: false`/`AbleToScale: false`, monitor the adapter, and prefer `behavior`
  that fails safe (e.g. don't auto-scale-down to nothing on missing metrics).
- **Autoscaler fights to watch for:** HPA + VPA on the same resource; two controllers writing
  `.spec.replicas` (HPA + a GitOps tool that pins replicas — exclude `replicas` from sync); CA/Karpenter
  scaling down a node whose pods an HPA is about to need (use stabilization + don't over-tight scale-down).
- **PDBs are double-edged.** They protect availability but a too-strict PDB (`minAvailable: 100%`)
  **deadlocks scale-down** — CA/Karpenter can never drain the node. Set realistic budgets.
- **Slow node provisioning** is the silent SLO killer: pod Pending → CA/Karpenter decision → cloud
  launch → kubelet ready → image pull can be minutes. Keep headroom (overprovisioning pause-pods /
  low-priority placeholder Deployments) for latency-critical scale-up.
- **Right-size requests first.** Every autoscaler reasons about *requests*. Wrong requests poison HPA
  utilization, VPA recommendations, CA bin-packing, and Karpenter shape selection simultaneously.

## Version awareness

This page reflects the state of the ecosystem as of 2026 and intentionally avoids pinning exact version
numbers for fast-moving features. Re-verify before relying on: **in-place pod resize** and **VPA in-place
support** (feature-gate/maturity), **GKE Multidimensional Pod Autoscaling** availability, **HPA
configurable tolerance** (`behavior.*.tolerance`), and **Karpenter's** non-AWS provider maturity. When in
doubt, check the workload's metrics with `kubectl get --raw` and the autoscaler's own conditions/logs
rather than trusting documentation alone.

## Rationalizations & rebuttals

| Excuse | Rebuttal |
|--------|----------|
| "Just scale on CPU — it's the default and simplest." | For LLM/GPU serving CPU can sit idle while the GPU saturates and the queue grows; CPU never moves, so HPA never fires. Scale on the server's load metrics (queued/running requests, KV-cache utilization, concurrency, p95/TTFT). |
| "HPA and VPA on the same resource is fine, they'll settle." | They oscillate: HPA adds replicas to cut per-pod CPU, VPA cuts requests to match usage, HPA's percent-of-request jumps, it scales again. One loop per dimension — HPA on RPS/queue, VPA on CPU+mem, or use MPA. |
| "Skip the stabilization window — I want it responsive." | A 0s scale-*down* window shrinks you on every metric dip and thrashes warm-up-heavy pods. Scale up fast, scale down slow (≥300s, conservative percent). 0s belongs only on `scaleUp`. |
| "Scale-to-zero is too risky to bother with." | For expensive/scarce GPUs/TPUs idle cost is the bigger risk. The real tax is cold start; pay it down with image streaming/preloaded images, weight caching on fast local disk, or a warm `minReplicaCount: 1` tier + KEDA burst tier — then scale the rest to zero. |
| "I'll set requests later — get it scaling first." | Every autoscaler reasons about *requests*. Wrong/absent requests poison HPA utilization (undefined without requests), VPA recommendations, CA bin-packing, and Karpenter shape selection at once. Right-size requests first. |
| "One HPA per metric and take whichever — more signals, safer." | HPA takes the **max** across metrics, so any one hot/noisy metric scales you up. That's only safe if every metric genuinely *should* trigger scale-up; otherwise a flaky signal pins you high. |
| "Run both Cluster Autoscaler and Karpenter for belt-and-suspenders." | They make conflicting node decisions on the same nodes and fight over scale-down. Pick one per node set. |
| "Let my GitOps tool keep `.spec.replicas` in sync with HPA." | Two controllers writing `.spec.replicas` is last-writer-wins — they fight every reconcile. Exclude `replicas` from GitOps sync when an HPA owns it. |

## Red flags

Stop and reconsider if you see any of these:

- **Replica or request count oscillating** on a steady workload — HPA×VPA on the same resource, two
  controllers writing `.spec.replicas`, or a noisy metric with no tolerance/stabilization.
- **`kubectl describe hpa` shows `ScalingActive: false` / `AbleToScale: false` or metric `unknown`** —
  the metric pipeline (metrics-server / Prometheus / adapter / KEDA) is down and HPA has *silently
  stopped scaling*, often mid-incident.
- **`ScalingLimited: true`** — min/max replicas is binding; you're pinned, not autoscaling.
- **Nodes stuck and never draining on scale-down** — a too-strict PDB (e.g. `minAvailable: 100%`),
  `safe-to-evict: "false"`, bare/`emptyDir` pods, or kube-system pods without a PDB deadlocking CA/Karpenter.
- **CPU near zero on a GPU serving workload that's still queueing/missing latency SLO** — you're scaling
  on the wrong signal entirely.
- **Pending GPU/TPU pods that never schedule** — accelerator scarcity isn't handled (no spot+on-demand
  fallback, no reservations, no multi-zone spread, no Kueue queueing); "scale up" silently means "wait
  forever."
- **CA/Karpenter removing a node whose pods an HPA is about to need** — scale-down too aggressive / no
  stabilization, causing churn against imminent demand.
- **Partial gangs of training pods running and burning accelerators** while waiting for the rest — HPA
  was used where Kueue + `ProvisioningRequest` (all-or-nothing) belongs; or single pods added for a
  multi-host replica that can't serve alone (needs LeaderWorkerSet).
- **A Karpenter NodeClass/NodePool edit rolling the whole fleet** unexpectedly — drift replacement with
  no `budgets` cap.

## Verification gate (definition of done)

Before calling autoscaling "done," confirm:

- [ ] **Right scaling signal chosen** per workload — utilization only where it tracks the bottleneck;
  for GPU/LLM serving, a load metric (queue depth / running-vs-waiting / KV-cache util / concurrency /
  p95-TTFT), not CPU. Verify the metric is actually served:
  `kubectl get --raw "/apis/{custom,external}.metrics.k8s.io/..."` returns a value.
- [ ] **Requests set and right-sized** on every autoscaled pod (utilization HPA is undefined without them).
- [ ] **`behavior` / stabilization tuned** — scale up fast, scale down slow (≥300s `scaleDown`
  stabilization, conservative percent); tolerance/policies sized to the latency budget and warm-up time.
- [ ] **No HPA×VPA conflict on the same resource** — distinct dimensions (HPA on RPS/queue, VPA on
  CPU+mem) or MPA; and exactly one controller writes `.spec.replicas` (GitOps excludes it).
- [ ] **Node provisioning verified end-to-end** — a Pending pod actually triggers CA/Karpenter/NAP and
  schedules on a node that satisfies its requests/affinity/taints/topology; headroom (pause-pods /
  low-priority placeholders) in place for latency-critical scale-up.
- [ ] **Scale-down safety verified** — PDBs realistic (not `100%`), drain succeeds; only one node
  autoscaler per node set; Karpenter `disruption.budgets` set against mass churn/drift.
- [ ] **Scale-to-zero path validated** (if used) — activation vs scaling thresholds set deliberately,
  cold-start time measured and within budget (or a warm tier kept).
- [ ] **Load-tested** — drive a realistic spike and a sustained ramp; confirm capacity arrives before
  SLOs break, the loop converges without thrash, and scale-down returns to baseline.
- [ ] **Failure mode checked** — metric pipeline down ⇒ alerts fire (`ScalingActive: false`) and the
  system fails safe (no runaway scale-down on missing metrics).
- [ ] **Maturity caveats verified for your cluster** — if relying on in-place pod resize / VPA in-place
  or GKE MPA, confirm feature-gate state / availability rather than assuming GA.

## Canonical references

- HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/ and the walkthrough
  https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/
- HPA algorithm details: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#algorithm-details
- VPA: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler
- In-place pod resize: https://kubernetes.io/docs/tasks/configure-pod-container/resize-container-resources/
- Cluster Autoscaler FAQ (expanders, scale-down, annotations):
  https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md
- Karpenter: https://karpenter.sh/docs/ (concepts: NodePools, NodeClasses, Disruption)
- KEDA: https://keda.sh/docs/ (Concepts, Scalers, ScaledObject/ScaledJob)
- GKE NAP: https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning
- Kueue ProvisioningRequest: https://kueue.sigs.k8s.io/docs/admission-check-controllers/provisioning/
- LeaderWorkerSet: https://github.com/kubernetes-sigs/lws
- metrics-server: https://github.com/kubernetes-sigs/metrics-server ·
  Prometheus Adapter: https://github.com/kubernetes-sigs/prometheus-adapter

---

# Autoscaling — Worked Examples

Canonical, imitate-ready manifests. Shapes are correct; **verify apiVersions and provider-specific
fields against your cluster version and provider docs** before applying (this ecosystem moves fast).

---

## 1. HPA with custom + external metrics and `behavior`

A web service scaled on **requests-per-second per pod** (custom metric via Prometheus Adapter) *and*
**SQS backlog** (external metric), with scale-up-fast / scale-down-slow tuning. HPA takes the **max**
desired across the two metrics — either hot signal scales you out.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web
  namespace: shop
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web
  minReplicas: 3
  maxReplicas: 60
  metrics:
    # (a) custom Pods metric — avg requests/sec per pod, served via custom.metrics.k8s.io
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "50"          # target 50 rps/pod
    # (b) external metric — total messages waiting in the order queue
    - type: External
      external:
        metric:
          name: sqs_approx_messages_visible
          selector:
            matchLabels: { queue: orders }
        target:
          type: AverageValue
          averageValue: "30"          # ~30 backlog msgs per pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0   # react immediately to spikes
      selectPolicy: Max
      policies:
        - { type: Percent, value: 100, periodSeconds: 60 }   # ≤ double / min
        - { type: Pods,    value: 4,   periodSeconds: 60 }    # ...or +4 / min
    scaleDown:
      stabilizationWindowSeconds: 300 # consider the max recommendation over 5 min
      selectPolicy: Max
      policies:
        - { type: Percent, value: 25, periodSeconds: 120 }    # shrink gently
```

Notes:
- The custom metric must be registered: `kubectl get --raw
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/shop/pods/*/http_requests_per_second"` should return
  data. Likewise the external one under `/apis/external.metrics.k8s.io/...`.
- `AverageValue` divides the metric by replica count under the hood — `desired = ceil(currentReplicas ×
  current/target)`. Use `Value` (not `Average`) only when the target is an absolute, not per-pod, figure.
- Watch `kubectl describe hpa web` → conditions. `ScalingActive: false`/`unknown` = broken pipeline.

---

## 2. KEDA `ScaledObject` — queue-driven, scale-to-zero

A worker Deployment that sits at **0 replicas** when the queue is empty and bursts on backlog. KEDA
generates the HPA; you still tune HPA `behavior` through `advanced`.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: image-worker
  namespace: media
spec:
  scaleTargetRef:
    name: image-worker            # the Deployment
  minReplicaCount: 0              # scale to zero when idle (KEDA-only capability)
  maxReplicaCount: 100
  idleReplicaCount: 0            # explicit idle floor (optional; must be < minReplicaCount range)
  cooldownPeriod: 300           # seconds idle before scaling back to 0
  pollingInterval: 15           # how often KEDA queries the scaler
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
            - { type: Percent, value: 50, periodSeconds: 60 }
  triggers:
    - type: aws-sqs-queue
      authenticationRef: { name: keda-sqs-auth }
      metadata:
        queueURL: https://sqs.us-east-1.amazonaws.com/123456789012/image-jobs
        awsRegion: us-east-1
        queueLength: "10"            # SCALING target: ~10 msgs per replica (drives 1→N)
        activationQueueLength: "5"   # ACTIVATION: need ≥5 msgs to wake 0→1
```

Notes:
- **Activation vs scaling:** `activationQueueLength` gates `0→1` (so a single stray message doesn't wake
  the fleet); `queueLength` governs `1→N`. They are independent — set activation deliberately.
- For **discrete batch items**, prefer `kind: ScaledJob` (KEDA spawns Jobs sized to the backlog) over a
  long-running Deployment.
- KEDA owns the HPA — don't also create an HPA targeting the same Deployment, and don't pin
  `.spec.replicas` from GitOps (it'll fight KEDA).

---

## 3. Karpenter `NodePool` (+ `EC2NodeClass`) sketch

Just-in-time, cost-aware nodes for a general workload with **spot-first, on-demand fallback** and
consolidation. Provider here is AWS; the `NodeClass` kind/fields are provider-specific.

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: general
spec:
  template:
    metadata:
      labels: { team: platform }
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: default
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]      # try spot, fall back to on-demand
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["m6i.large", "m6i.xlarge", "m6a.large", "m6a.xlarge", "c6i.large"]
      expireAfter: 720h                        # recycle nodes (drift/patching)
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 1m                       # debounce before merging/removing
    budgets:
      - nodes: "10%"                           # never disrupt > 10% of nodes at once
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: default
spec:
  amiFamily: AL2023
  amiSelectorTerms:
    - alias: al2023@latest                     # editing this triggers DRIFT → fleet replace
  subnetSelectorTerms:
    - tags: { karpenter.sh/discovery: my-cluster }
  securityGroupSelectorTerms:
    - tags: { karpenter.sh/discovery: my-cluster }
  role: KarpenterNodeRole-my-cluster
```

Protect a restart-hostile pod from consolidation/drift:

```yaml
metadata:
  annotations:
    karpenter.sh/do-not-disrupt: "true"        # on the Pod / pod template
```

Notes:
- Give Karpenter a **wide instance-type set** — that's the whole point; over-constraining it back to one
  shape recreates Cluster Autoscaler's limitations.
- A `NodeClass` edit (AMI alias, etc.) causes **drift** and can roll the entire fleet — gate it behind
  `budgets` and a maintenance window.
- For GPU/accelerator pools, add the GPU instance families and a corresponding NodePool `weight`/taint;
  pair with `do-not-disrupt` on long-running inference/training pods so consolidation can't evict them.

---

## 4. Anti-pattern: HPA and VPA fighting on CPU (do not do this)

```yaml
# HPA scaling on CPU utilization ...
kind: HorizontalPodAutoscaler
spec:
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
---
# ... AND VPA Auto adjusting CPU requests on the SAME Deployment  → oscillation
kind: VerticalPodAutoscaler
spec:
  updatePolicy: { updateMode: "Auto" }
  resourcePolicy:
    containerPolicies:
      - containerName: '*'
        controlledResources: ["cpu", "memory"]   # CPU here collides with the HPA above
```

**Fix:** let HPA own the horizontal signal (ideally a custom/external metric, not CPU) and restrict VPA
to the dimension HPA doesn't touch — or use GKE Multidimensional Pod Autoscaling (verify availability):

```yaml
# VPA limited to memory only; HPA scales horizontally on RPS / queue depth
spec:
  resourcePolicy:
    containerPolicies:
      - containerName: '*'
        controlledResources: ["memory"]
  updatePolicy:
    updateMode: "Off"        # start in Off, read recommendations, then graduate deliberately
```
