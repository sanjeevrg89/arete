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
