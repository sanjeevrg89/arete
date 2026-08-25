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
