# AGENTS.md — Kubernetes Autoscaling Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`autoscaling-kubernetes-guide.md`** next to this file —
> read it before designing or debugging autoscaling, and apply it. Imitate-ready manifests (HPA with
> external metrics + `behavior`, a KEDA scale-to-zero `ScaledObject`, a Karpenter `NodePool`) are in
> **`examples.md`**. This file is the always-on summary.

## When working on Kubernetes autoscaling, apply these by default:

- **Three axes, composed not stacked.** HPA = replica count, VPA = per-pod requests, node autoscaler
  (Cluster Autoscaler / Karpenter / GKE NAP) = capacity. Node autoscalers react to **Pending pods and
  node utilization**, never to your metric — so pod autoscaling must produce schedulable demand.
- **Never point two loops at one dimension.** HPA + VPA on the same resource (both CPU or both memory)
  oscillates — canonical self-inflicted outage. Supported split: HPA on a custom/external metric (RPS,
  queue), VPA on CPU+memory. On GKE consider Multidimensional Pod Autoscaling (verify availability).
- **HPA algorithm:** `desiredReplicas = ceil(current × currentMetric / targetMetric)`, gated by tolerance
  (default 10%), recomputed each sync period (default 15s); with multiple metrics HPA takes the **max**.
  Utilization targets require `requests` to be set.
- **Metrics need adapters.** CPU/mem via metrics-server (`metrics.k8s.io`); pod/object metrics via
  `custom.metrics.k8s.io`; off-cluster signals (queue depth, Kafka lag) via `external.metrics.k8s.io`
  — Prometheus Adapter or KEDA. No adapter ⇒ HPA shows `unknown` and won't scale. Verify with
  `kubectl get --raw /apis/external.metrics.k8s.io/...` and `kubectl describe hpa`.
- **Tune `behavior` to stop thrash:** scale **up fast** (short/zero `stabilizationWindowSeconds`,
  generous Pods/Percent policy), scale **down slow** (≥300s window, conservative percent). `selectPolicy:
  Disabled` on `scaleDown` pins direction off for expensive/stateful pods.
- **Readiness matters:** unready/cold-starting pods are excluded from HPA math; bad probes ⇒ over/under
  scale. Account for metric lag (scrape interval + rate window + sync period).
- **VPA modes:** `Off` = recommend only (**safe default**); `Initial` = set at admission; `Recreate`/
  `Auto` = **evict to resize**. `Auto` evicts because changing requests historically required recreating
  the pod. **In-place pod resize** (no restart) is the modern path — **verify feature-gate/VPA maturity
  for your cluster version** before relying on it.
- **Cluster Autoscaler:** scales up on `Pending` pods that no node fits; scales down nodes under the
  utilization threshold after a cooldown. Respects PDBs, `safe-to-evict: "false"`, local-storage pods.
  Expanders: `random`/`most-pods`/`least-waste`/`priority` (chainable). Node-group-shaped → coarse.
- **Karpenter:** provisions right-sized nodes JIT from a flexible instance set (`NodePool` + `NodeClass`),
  then **consolidates** (empty/underutilized) and handles **drift** (NodeClass edits can roll the fleet).
  Protect pods with `karpenter.sh/do-not-disrupt: "true"`; cap churn with disruption `budgets`. Prefer
  over CA for mixed/cost-sensitive fleets; don't run both on the same nodes.
- **KEDA:** event-driven scaling + **scale-to-zero** (HPA alone can't reach 0). A `ScaledObject` generates
  an HPA under the hood and serves the scaler over `external.metrics`. The **activation** threshold (0→1)
  is separate from the scaling threshold (1→N) — set it deliberately. `ScaledJob` for queue-driven batch.
- **ML/GPU serving:** CPU is the wrong signal — scale on **queue depth / pending requests / concurrency /
  KV-cache util / TTFT** from the serving framework via Prometheus. Scale-to-zero saves accelerators but
  cold start (image + weights) is minutes — mitigate or keep a warm replica. Multi-host serving scales by
  **LeaderWorkerSet group**, not per pod. Batch/training capacity goes through **Kueue + ProvisioningRequest**
  (gang, all-or-nothing), not HPA.
- **Right-size `requests` first** — every autoscaler reasons about requests; wrong requests poison HPA
  utilization, VPA, CA bin-packing, and Karpenter shape selection at once.

## Definition of done for autoscaling changes
- The metric pipeline actually serves the metric (`kubectl get --raw ...`); `kubectl describe hpa` shows
  `ScalingActive: true` / `AbleToScale: true`, not `unknown`/`ScalingLimited` unexpectedly.
- HPA and VPA do **not** target the same resource; only one controller writes `.spec.replicas`.
- PDBs allow eventual scale-down (no `minAvailable: 100%` deadlock); `do-not-disrupt`/`safe-to-evict`
  set on restart-hostile pods.
- Scale-to-zero paths have an activation source; expensive cold-start paths have a warm tier or
  mitigation. Alerts exist for a frozen metric pipeline.

When unsure about a fast-moving feature (in-place resize, MPA, configurable tolerance, Karpenter on
non-AWS), state the maturity and verify against current upstream docs — never fabricate fields/flags.
