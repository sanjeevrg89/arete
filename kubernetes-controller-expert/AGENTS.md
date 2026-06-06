# AGENTS.md — Kubernetes Controller Engineering

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`kubernetes-controller-expert-guide.md`** next to this file —
> read it before writing or reviewing a controller. A canonical Reconciler + `main.go` to imitate are
> in **`examples.md`**. All Go idiom defers to **[[go-best-practices]]**. This file is the always-on
> summary.
>
> Scope: **building controllers** (controller-runtime, client-go, reconcile). The operator/CRD layer is
> **[[kubernetes-operator-expert]]**; apiserver internals are **[[kubernetes-internals-expert]]**.

## When writing/reviewing a controller (`Reconcile`, `SetupWithManager`, informers, finalizers), apply by default:

- **Level-triggered, not edge-triggered.** A `reconcile.Request` is only "namespace/name might have
  changed — go look." It carries no event type, no old value, no changed field. **Re-read current state
  every reconcile** and converge. Never depend on event ordering or on seeing every event.
- **Reconcile must be idempotent.** A no-op reconcile on unchanged state makes **zero** writes. Build
  desired state from `.spec`, diff against observed, apply only the delta. It will run repeatedly.
- **Read from the cache, write to the API server.** Cached reads (lister/cached `client.Client`) are
  eventually consistent and can be **stale** — don't trust read-after-write; requeue and re-observe.
  Writes go straight to the apiserver.
- **Fetch-or-return on NotFound:** `if err := r.Get(...); err != nil { return ctrl.Result{}, client.IgnoreNotFound(err) }`.
  Parent gone → do nothing about children; GC handles them.
- **Owner references** on every created child (`controllerutil.SetControllerReference`) so GC collects
  them and `Owns()` maps child events to the owner. One `controller:true` owner per object. **No
  cross-namespace ownerRefs** (they silently don't GC).
- **Finalizers for external cleanup:** add the finalizer *before* creating external resources; on
  `deletionTimestamp != nil` run **idempotent** cleanup, then **always remove the finalizer**. A
  finalizer you fail to remove = object stuck `Terminating` forever and namespace deletion hangs.
- **Status via the subresource** (`r.Status().Update/Patch`). Set `status.observedGeneration =
  obj.Generation`. Use `metav1.Condition` + `meta.SetStatusCondition`/`meta.IsStatusConditionTrue`.
  **Write status only when it actually changed** — a no-op status write fires your own watch and hot-loops.
- **Requeue correctly:** return `error` → exponential backoff (transient failures); `RequeueAfter: d`
  → poll later (external state/TTL); `ctrl.Result{}, nil` → done. **Never** `RequeueAfter: 0`, a tiny
  delay, or a blocking `time.Sleep` in Reconcile — that's a busy-loop that holds a worker slot.
- **Prefer Server-Side Apply** for owned objects: `r.Patch(obj, client.Apply, client.FieldOwner("..."),
  client.ForceOwnership)` — declarative, idempotent, no read, conflict-free with proper field ownership.
  Use a stable unique `FieldOwner`; don't mix SSA and client-side `Update` on the same object.
- **Cut load with predicates & selectors:** `predicate.GenerationChangedPredicate{}` so status/metadata
  updates don't re-enqueue (generation bumps on spec only); label/field selectors + `GetFieldIndexer`
  indexes instead of full-namespace LISTs; keep mapping functions cache-only and bounded.
- **Wire `Owns()` for children, `Watches()` + `EnqueueRequestsFromMapFunc` for related objects you
  don't own.** Tune `MaxConcurrentReconciles`; the workqueue dedupes per key and never reconciles one
  key concurrently.
- **Manager hygiene:** leader election on for HA (else replicas fight); `ctrl.SetupSignalHandler()` ctx
  for graceful shutdown; `AddHealthzCheck`/`AddReadyzCheck`; metrics exposed; structured `logr` via
  `log.FromContext(ctx)`.
- **RBAC must match the verbs used** — incl. `<resource>/status`, `<resource>/finalizers`,
  `coordination.k8s.io/leases` (leader election), `events`. Keep `// +kubebuilder:rbac:` markers in
  sync. Symptom of drift: `Forbidden` reconcile loops.

## Bugs to reject in review

Status-write hot loops (fix: write-only-on-change **and** `GenerationChangedPredicate`); two controllers
fighting over one field (one owner per field; SSA makes conflicts explicit); non-idempotent side effects
(converge, don't blindly create); trusting the cache right after a write; goroutines/sleeps outliving
Reconcile; missing RBAC; requeue storms / unbounded mapping fan-out; full-namespace LISTs in hot paths;
mutating cached objects without `DeepCopy`.

## Testing

- **envtest** (real apiserver + etcd, no kubelet) for API-mediated behavior: admission, status
  subresource, finalizers, owner refs, optimistic concurrency. Pods never run — assert on the objects
  your controller creates; use `Eventually(...)` for eventual consistency.
- **fake client** only for pure reconcile logic. It does **not** faithfully emulate SSA, admission,
  defaulting, validation, or GC; enable `WithStatusSubresource`. Don't trust it for those paths.
- Table-driven tests; push logic into pure `buildDesired*` helpers; run `-race`; inject a clock for timing.

## Definition of done

Reconcile idempotent + level-triggered; NotFound handled; ownerRefs + finalizers + status(observedGeneration,
conditions) correct; requeue semantics right with no busy-loop; RBAC matches; leader election + graceful
shutdown + health + metrics + structured logging wired; no hot loop / fight / requeue storm; tests pass
(`go test -race ./...`) with envtest for API behavior. Use the checklist at the end of the guide.
