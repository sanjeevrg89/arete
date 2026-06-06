---
name: kubernetes-controller-expert
description: World-class guidance for writing correct, production-grade Kubernetes controllers with
  controller-runtime (kubebuilder) and client-go. Use when authoring or reviewing a Reconciler,
  reconcile loop, informer/lister/workqueue code, finalizers, owner references, status conditions,
  server-side apply, leader election, or envtest/Ginkgo controller tests — anytime there are `.go`
  files importing `sigs.k8s.io/controller-runtime`, `k8s.io/client-go`, or `k8s.io/apimachinery`, a
  `Reconcile(ctx, req)` method, a `SetupWithManager`, a `main.go` wiring a `manager.Manager`, or
  symptoms like hot-loop reconciles, stuck finalizers, cache staleness races, requeue storms, or
  controllers fighting each other. Covers level-triggered reconciliation, the
  reflector→DeltaFIFO→indexer→workqueue flow, idempotent reconcile, `RequeueAfter`, GC via
  ownerReferences, `metav1.Condition` + observedGeneration, and scaling caches with selectors/indexes.
---

# Kubernetes Controller Expert

Write controllers the way a maintainer of a widely-used controller (Deployment/Job-class) would: a
**level-triggered, idempotent, eventually-consistent** reconcile that compares desired vs observed
state and converges, never trusts event ordering, and never leaks external resources or hot-loops.

## How to use this skill

1. **Read `kubernetes-controller-expert-guide.md`** in this directory — the full reference. Apply it
   to the controller at hand. For a canonical controller-runtime `Reconciler` (owner refs, status
   conditions, finalizer, server-side apply) and a minimal `main.go` wiring a `Manager`, read
   **`examples.md`**.
2. Go code defers to **[[go-best-practices]]** for idiom (errors, context, concurrency, tests).
3. Match the surrounding project's conventions (kubebuilder layout, existing API types); apply the
   correctness/safety rules (idempotency, finalizers, GC, no hot loops) regardless.

## Essentials (full detail in `kubernetes-controller-expert-guide.md`)

- **Reconcile is level-triggered, not edge-triggered.** A `reconcile.Request` is only a *hint* that
  *something* changed for that key — never which field, never the old value, never the event type.
  Re-read current state from the cache every time; converge to desired. Drop all events and you must
  still self-heal on the next resync.
- **Reconcile must be idempotent.** Running it twice (or 100×) on unchanged state makes zero writes.
  Build desired state from spec, diff against observed, apply only the delta.
- **Read from the cache (lister/cached client), write to the API server.** The cached read is
  eventually consistent and can be **stale** — never assume your last write is visible on the next
  reconcile. Treat `AlreadyExists` / conflict / `NotFound` as normal and requeue.
- **Fetch-or-return on NotFound:** `client.Get` → if `apierrors.IsNotFound(err)`, the object is gone;
  return `nil` (no requeue). Owned children are cleaned up by GC via `ownerReferences`, not by you.
- **Owner references + controller GC** are how you delete children — set `controllerutil.SetControllerReference`
  (or SSA owner refs) so the API server garbage-collects them. Cross-namespace owner refs do not work.
- **Finalizers** for external cleanup: add the finalizer, and on `deletionTimestamp != nil` run cleanup,
  then **remove the finalizer**. A finalizer you never remove = an object stuck `Terminating` forever.
  Cleanup must be idempotent and tolerate the resource already being gone.
- **Status: `observedGeneration` + `metav1.Condition`.** Update status via the **status subresource**
  (`Status().Update/Patch`). Set `condition.ObservedGeneration = obj.Generation`. Use
  `meta.SetStatusCondition`. Don't write status when nothing changed — that triggers a watch event and
  can hot-loop.
- **Requeue correctly:** return an `error` for transient failures (rate-limited exponential backoff);
  `RequeueAfter` for "check again later" (polling external state, TTLs); `ctrl.Result{}` (empty, nil
  err) when done — no busy-loop. Never `RequeueAfter: 0` to "retry now."
- **Predicates and selectors to cut load:** `GenerationChangedPredicate` to skip status-only updates;
  label/field selectors + indexed caches to avoid full-namespace lists at scale.
- **`Owns()` for children, `Watches()` + a mapping function** for related objects you don't own.
  Tune `MaxConcurrentReconciles`; the workqueue dedupes and rate-limits per key.
- **Avoid the classic bugs:** status writes that re-trigger your own watch (hot loop); two controllers
  fighting over the same field (use SSA field ownership / single owner); non-idempotent side effects;
  unbounded goroutines; missing RBAC; relying on cache freshness right after a write.
- **Test with envtest** (real apiserver + etcd, no kubelet) for reconcile behavior; the **fake client**
  is fine for pure logic but does **not** enforce SSA, admission, defaulting, or GC — don't trust it for
  those. Table-driven where possible.

## Related skills

- `[[kubernetes-operator-expert]]` — the operator/CRD packaging layer: API design, CRD schema,
  webhooks, conversion, kubebuilder scaffolding, OLM. Reach for it for the *operator* around the controller.
- `[[kubernetes-internals-expert]]` — apiserver/etcd/watch-cache/GC internals when you need to know
  *why* the machinery behaves as it does.
- `[[go-best-practices]]` — all Go idiom the controller code must follow.
- `[[kubernetes-expert]]` — operating clusters and the objects your controller manages.
- `[[kueue-advanced]]`, `[[jobset-leaderworkerset]]` — large real-world controllers to study as
  reference implementations of these patterns.
