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

---

# Reference — kubernetes-controller-expert

# Kubernetes Controller Engineering Guide

Production-grade guidance for writing Kubernetes controllers, written from the perspective of a
maintainer of a widely-used controller. This is the single source of truth; `SKILL.md`, `AGENTS.md`,
and `examples.md` defer to it.

> Scope: **building controllers** — the reconcile loop, `client-go` informer machinery, and
> `controller-runtime`. The operator/CRD packaging layer (API design, CRD schema, webhooks,
> kubebuilder scaffolding, OLM) is **[[kubernetes-operator-expert]]**; deep apiserver/etcd/watch-cache
> internals are **[[kubernetes-internals-expert]]**. All Go idiom defers to **[[go-best-practices]]**.

> Version note (2026): controller-runtime and client-go track Kubernetes minor releases and move
> fast. APIs like `client.Client`, `manager.Manager`, `builder.Builder`, and `metav1.Condition` are
> stable in spirit, but exact signatures, default flags, and metrics endpoints change across minors.
> **Verify against the controller-runtime version in your `go.mod`** and the godoc for that tag. Do
> not trust a version number unless you checked it.

---

## 1. Mental model: the controller pattern

A controller runs a **control loop**: observe actual state, compare to desired state, take actions to
drive actual toward desired, repeat — forever. This is the same loop a thermostat runs. Get this model
right and most controller bugs disappear; get it wrong and you write edge-triggered, ordering-dependent
code that breaks the first time an event is missed.

**Level-triggered, not edge-triggered.** A thermostat acts on the *current* temperature (the level),
not on "the temperature just rose by 1°" (the edge). A correct controller acts on the *current
observed state* of the world, not on the specific event that woke it.

- A `reconcile.Request` (or a workqueue key) tells you **only** "namespace/name *might* have changed —
  go look." It does **not** carry the event type (add/update/delete), the old object, which field
  changed, or any payload. If you find yourself wanting "what was the previous value," your design is
  edge-triggered and wrong.
- Therefore: on every reconcile, **re-read current state** from the cache and recompute. Never rely on
  event ordering, never assume you'll see every event, never assume events arrive at most once.
  Informers coalesce, drop, and replay; a resync delivers synthetic "updates" with no real change.
- **The strong guarantee you get:** if the real state ever differs from what your last reconcile saw,
  you will be told to reconcile that key again (eventually). The **weak guarantee:** you will *not*
  necessarily see every intermediate state, and you may be told to reconcile when nothing changed.
  Design for both: converge from *any* observed state, and make a no-op reconcile cost nothing.

**Desired vs observed → idempotent convergence.**

- **Desired state** = a pure function of the object's `.spec` (and referenced inputs). Compute it the
  same way every time.
- **Observed state** = what's actually in the cluster right now (child objects, external resources).
- **Reconcile** = `diff(desired, observed)` → apply only the delta. Run it twice on unchanged state
  and it makes **zero writes**. This idempotency is non-negotiable: the loop *will* run repeatedly.

**Eventual consistency.** You operate on a cache that lags the API server, against an API server that
lags etcd commit visibility, against a world (Deployments → ReplicaSets → Pods → kubelet) that takes
time to settle. Never assume "I just created X, so X exists in my cache." Reconcile, observe it's not
there yet, requeue, converge. Reconciling is cheap; assuming freshness is a bug.

---

## 2. client-go machinery (what controller-runtime wraps)

Even when you use controller-runtime, understand the `client-go` layer underneath — it explains cache
staleness, resync cost, and where load comes from.

```
API server (watch) ──► Reflector ──► DeltaFIFO ──► Indexer (thread-safe store/cache)
                                          │                    ▲
                                          └──► informer ───────┘ updates the indexer,
                                                 handlers        fans out to event handlers
                                                    │
                                                    ▼
                                              Workqueue (rate-limited, dedup'd by key)
                                                    │
                                                    ▼
                                              your worker → Reconcile / sync
```

- **Reflector** — runs a `LIST` then a long-lived `WATCH` against one resource type, turning the
  stream into deltas. On watch errors it **re-LISTs** (relist) from the last known resourceVersion or
  from scratch — another reason you can't depend on seeing every event.
- **DeltaFIFO** — an ordered, deduplicated queue of changes per object key. Compresses multiple
  pending deltas for the same key.
- **Indexer** — the in-memory **store** (thread-safe `cache.Store` with indexes). This is the **cache**
  you read from. It is **eventually consistent** — it reflects the last watch event processed, which
  lags the API server.
- **Informer** — drives the reflector→DeltaFIFO→indexer pipeline and invokes registered event handlers
  (Add/Update/Delete). A **SharedInformerFactory** (controller-runtime's cache is a shared informer
  per GVK) ensures **one** informer/watch per type is shared across all controllers — critical: don't
  spin up a second informer for a type you already watch.
- **Lister** — a typed, read-only accessor over the indexer (`Get`, `List`, label-selector `List`).
  Reads are served from local memory — fast, but **stale**. In controller-runtime this is the cached
  `client.Client` read path (`mgr.GetClient().Get/List`).
- **Workqueue** — a rate-limited, **deduplicating** queue. Multiple events for the same key collapse to
  one item; a key being processed is not re-queued until done (prevents concurrent reconciles of the
  same object). The default is an exponential-backoff rate limiter.

**Resync period.** An informer resync re-delivers every object in the cache to the handlers as a
synthetic "update" (NOT a fresh LIST from the API server — it replays the local cache). It exists to
re-trigger reconciles that may have been missed and to catch drift you can only detect by re-running
logic. Cost: at resync, **every** object you watch gets re-enqueued → a reconcile spike proportional to
object count. **Default to a long resync or none** (controller-runtime: `SyncPeriod`, default ~10h);
only shorten it if you have real reason, and know it multiplies your steady-state reconcile load.

**The golden rule: read cache, write API.**

- **Reads** go through the lister/cached client — fast, but may be stale or even absent right after you
  wrote. Tolerate `NotFound`-of-a-thing-you-just-created and conflicts.
- **Writes** go to the **API server** directly. After a write, **do not** read your own write back from
  the cache expecting to see it — the cache updates asynchronously via the watch. If you need the
  server's view immediately (rare), use a direct (uncached) reader, but prefer to just requeue and
  re-observe next reconcile.

**Watch cache (server side).** The API server keeps its own watch cache; your watch is usually served
from it, not etcd. Relevant to you mainly because it bounds how fresh watches are and why a `List`
with `ResourceVersion=0` may be served from cache. Deep detail lives in [[kubernetes-internals-expert]].

---

## 3. controller-runtime: the objects you actually use

controller-runtime (kubebuilder's runtime) is the standard framework. The pieces:

- **`manager.Manager`** — owns the shared cache (informers), the clients, leader election, metrics
  server, health probes, and the lifecycle of all controllers. You create one Manager per binary and
  `mgr.Start(ctx)` it. It blocks until `ctx` is cancelled, then drains.
- **`client.Client`** — the unified read/write client. **Reads (`Get`/`List`) are served from the
  Manager's cache** (informers) by default — fast and stale. **Writes (`Create`/`Update`/`Patch`/
  `Delete`) go straight to the API server.** Need an uncached read? Use `mgr.GetAPIReader()` or
  configure `client.Options{Cache: &client.CacheOptions{...}}` to bypass — but that's a direct LIST/GET
  and you should rarely need it.
- **`reconcile.Reconciler`** — your interface: `Reconcile(ctx, reconcile.Request) (reconcile.Result, error)`.
  The `Request` is just `NamespacedName`. That's the whole input. (See §1: it's a hint, not a payload.)
- **`builder.Builder`** (`ctrl.NewControllerManagedBy(mgr)`) — declarative controller wiring:
  - **`For(&v1.Foo{})`** — the primary type this controller reconciles; enqueues the object's own key.
  - **`Owns(&appsv1.Deployment{})`** — watch a child type; map child events back to the **owner's**
    key via its controller `ownerReference` (`EnqueueRequestForOwner`). This is how a change to a Pod
    re-reconciles its owning Foo.
  - **`Watches(&v1.Bar{}, handler.EnqueueRequestsFromMapFunc(fn))`** — watch a type you don't own and
    map its events to one or more primary keys with your own function (e.g. a referenced ConfigMap →
    all Foos that reference it).
  - **`WithEventFilter(pred)` / per-source predicates** — drop events before they hit the queue.
  - **`WithOptions(controller.Options{MaxConcurrentReconciles: N, RateLimiter: ...})`**.
- **EventHandlers** — translate a watch event into workqueue keys:
  `EnqueueRequestForObject` (the object itself), `EnqueueRequestForOwner` (the owner; used by `Owns`),
  `EnqueueRequestsFromMapFunc` (arbitrary mapping). Mapping functions run on **every** matching event —
  keep them cheap and cache-only (no API calls).
- **Predicates** — pure filters on events: `GenerationChangedPredicate` (skip updates that didn't bump
  `.metadata.generation` — i.e. **status-only and metadata-only changes**, the #1 hot-loop fix),
  `LabelChangedPredicate`, `AnnotationChangedPredicate`, `ResourceVersionChangedPredicate`, and custom
  `predicate.Funcs`. Predicates run before enqueue → cheap way to shed load.
- **Rate-limited workqueue** — same dedup + exponential backoff as client-go. Returning an `error`
  re-enqueues with backoff; `RequeueAfter` enqueues after a fixed delay (no backoff growth).
- **`MaxConcurrentReconciles`** — how many objects reconcile in parallel. The queue still guarantees a
  single key is never reconciled concurrently, so raising this is safe for throughput **as long as your
  Reconcile is idempotent and doesn't share mutable state across keys**.

---

## 4. The reconcile loop, step by step (the canonical shape)

```go
func (r *FooReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    // 1. Fetch the primary object from the cache. Gone? Nothing to do.
    var foo appv1.Foo
    if err := r.Get(ctx, req.NamespacedName, &foo); err != nil {
        // IsNotFound: object deleted; owned children GC'd via ownerRefs. Don't requeue.
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // 2. Handle deletion via finalizer (see §6) before doing any normal work.
    if !foo.DeletionTimestamp.IsZero() {
        return r.reconcileDelete(ctx, &foo)
    }
    if added, err := r.ensureFinalizer(ctx, &foo); err != nil || added {
        return ctrl.Result{}, err // re-reconcile with the finalizer present
    }

    // 3. Build DESIRED state purely from foo.Spec.
    desired := buildDesiredDeployment(&foo)
    if err := controllerutil.SetControllerReference(&foo, desired, r.Scheme); err != nil {
        return ctrl.Result{}, err
    }

    // 4. Apply desired → observed. Prefer server-side apply (idempotent, no read needed).
    if err := r.Patch(ctx, desired, client.Apply,
        client.FieldOwner("foo-controller"), client.ForceOwnership); err != nil {
        return ctrl.Result{}, fmt.Errorf("apply deployment: %w", err)
    }

    // 5. Observe children and compute status.
    var dep appsv1.Deployment
    if err := r.Get(ctx, client.ObjectKeyFromObject(desired), &dep); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err) // not in cache yet → requeue
    }

    // 6. Update status (subresource) only if it changed. Set observedGeneration.
    return r.updateStatus(ctx, &foo, &dep)
}
```

**Return-value contract** (`ctrl.Result`, `error`):

| Return | Meaning | When |
| --- | --- | --- |
| `ctrl.Result{}, nil` | Done. Don't requeue (until a watch fires). | Steady state reached. |
| `ctrl.Result{}, err` | Requeue with **exponential backoff**. | Transient failure (conflict, API error). |
| `ctrl.Result{RequeueAfter: d}, nil` | Requeue after fixed `d`. | Poll external state / TTL / "check later". |

- **`Requeue: true`** (bare) is legacy; prefer returning an error (for backoff) or `RequeueAfter` (for
  a known delay). Newer controller-runtime deprecates the bare `Requeue` field — verify in your version.
- **Never** return `RequeueAfter: 0` or a tiny delay to "retry immediately" — that's a busy-loop.
- **Conflicts** (`apierrors.IsConflict`) on `Update` are normal under concurrency: re-`Get` and retry,
  or just `return ctrl.Result{}, err` to requeue. SSA (`client.Apply`) largely avoids them.

**Build then apply — never read-modify-write blindly.** Compute the full desired object from spec,
then reconcile it onto the cluster. Two write strategies:

| Strategy | How | Trade-off |
| --- | --- | --- |
| **Create-or-Update** | `controllerutil.CreateOrUpdate` / `CreateOrPatch` (Get, mutate in a closure, Create/Update) | Familiar; but you own the read-modify-write and conflict retries; you can clobber fields others set. |
| **Server-Side Apply (SSA)** | `client.Patch(obj, client.Apply, FieldOwner("..."))` | Declarative, idempotent, **no read needed**, conflict-free with proper field ownership. Preferred for objects this controller owns. Requires a unique, stable `FieldOwner`. |

**SSA caveats:** you must send a *fully-specified intent* for the fields you own (omit fields you don't
manage; don't send zero-values you don't mean). `ForceOwnership` resolves conflicts in your favor —
use it deliberately. Mixing SSA and client-side `Update` on the same object from the same controller
causes field-ownership churn; pick one per object.

---

## 5. Owner references & garbage collection

The API server's garbage collector deletes objects whose owners are gone — this is how you avoid
writing manual child-deletion code.

- Set a **controller owner reference** on every child you create:
  `controllerutil.SetControllerReference(owner, child, scheme)`. This sets `ownerReferences[].controller=true`
  and `blockOwnerDeletion=true`, and is also what `Owns()`/`EnqueueRequestForOwner` use to map child
  events back to the owner.
- **One controller owner per object.** Two `controller: true` owner refs is an error — it's the
  on-disk signal for "who is the single managing controller." Multiple *non-controller* owner refs are
  allowed (object survives until all owners are gone).
- **Owner references are namespace-scoped.** A namespaced owner **cannot** own an object in another
  namespace, and a namespaced object **cannot** own a cluster-scoped object. Cross-namespace "ownership"
  must be implemented with finalizers + a label/annotation back-reference and explicit cleanup, not
  ownerRefs. The GC silently won't collect a cross-namespace ref.
- **Deletion propagation:** GC default is **background**; `Foreground` deletes children before the
  owner finishes (owner stays until children gone, gated by `blockOwnerDeletion`); `Orphan` strips the
  ownerRef and leaves children. Know which your deletes use.
- When the primary object is `NotFound` in Reconcile, **do nothing** about children — GC handles them.
  Don't write child-cleanup-on-not-found code; you can't, the parent is already gone.

---

## 6. Finalizers — deletion and external cleanup

A finalizer is a string in `.metadata.finalizers`. While any finalizer is present, a `DELETE` only
sets `.metadata.deletionTimestamp` — the object is **not** removed from etcd until the **last**
finalizer is gone. This is your hook to clean up things the GC can't: external resources (cloud LBs,
DNS records, S3 buckets, database rows, entries in another cluster).

**The pattern (idempotent, both directions):**

```go
const finalizer = "foo.example.com/cleanup"

// Not being deleted: ensure our finalizer is present.
if foo.DeletionTimestamp.IsZero() {
    if !controllerutil.ContainsFinalizer(&foo, finalizer) {
        controllerutil.AddFinalizer(&foo, finalizer)
        return ctrl.Result{}, r.Update(ctx, &foo) // requeue; finalizer now persisted
    }
} else {
    // Being deleted: run external cleanup, then drop the finalizer.
    if controllerutil.ContainsFinalizer(&foo, finalizer) {
        if err := r.deleteExternalResources(ctx, &foo); err != nil {
            return ctrl.Result{}, err // stay; retry with backoff
        }
        controllerutil.RemoveFinalizer(&foo, finalizer)
        return ctrl.Result{}, r.Update(ctx, &foo) // object now deletable
    }
    return ctrl.Result{}, nil
}
```

**Rules that prevent stuck-`Terminating` objects and leaked resources:**

- **Cleanup must be idempotent and tolerate "already gone."** Deleting the external resource twice, or
  when it never existed, must succeed. The reconcile *will* re-run after a crash mid-cleanup.
- **Always remove the finalizer after successful cleanup.** A bug, a permanent error, or a panic
  before `RemoveFinalizer` leaves the object stuck `Terminating` **forever**. This is the single most
  common controller production incident: a controller is uninstalled (or its RBAC revoked) while its
  finalizer is still on live objects → those objects can never be deleted, and namespace deletion hangs.
- **Don't block deletion on a dependency that can't be satisfied.** If cleanup can permanently fail
  (e.g. external system is gone for good), you need an escape hatch — surface it in status/events, and
  consider giving up after bounded retries rather than wedging deletion. Decide this deliberately.
- **Recovering a stuck object** (ops knowledge to encode in runbooks): `kubectl patch foo NAME -p
  '{"metadata":{"finalizers":[]}}' --type=merge` removes the finalizer manually — but this **leaks**
  whatever the finalizer was protecting. It's a break-glass, not a fix.
- **Add the finalizer *before* you create external resources**, so a crash between "created the cloud
  LB" and "added the finalizer" can't orphan the LB. Equivalently: never create an external resource
  you couldn't find again from the object's own spec/status.
- Finalizer names are **domain-qualified** (`foo.example.com/name`) by convention to avoid collisions.

---

## 7. Status: conditions & observedGeneration

Status is your controller's report on observed reality. It is read by users, other controllers, and
`kubectl wait`. Get the conventions right.

- **Always update status via the status subresource:** `r.Status().Update(ctx, &foo)` or
  `r.Status().Patch(...)`. This requires `// +kubebuilder:subresource:status` on the type (and the CRD
  to declare it). Writing `.status` through a normal `Update` is wrong when the subresource is enabled —
  it's silently ignored.
- **`status.observedGeneration`** = the `.metadata.generation` your controller has acted on. Set it
  each reconcile: `foo.Status.ObservedGeneration = foo.Generation`. Consumers compare
  `observedGeneration == generation` to know whether status reflects the **current** spec or a stale one.
- **`metav1.Condition`** is the standard status primitive. Each condition has `Type` (e.g. `Ready`,
  `Available`, `Progressing`), `Status` (`True`/`False`/`Unknown`), `Reason` (a CamelCase machine
  token), `Message` (human text), `LastTransitionTime`, and crucially **`ObservedGeneration`**. Manage
  them with the `meta` helpers — they handle transition-time and dedup:

  ```go
  meta.SetStatusCondition(&foo.Status.Conditions, metav1.Condition{
      Type:               "Ready",
      Status:             metav1.ConditionTrue,
      Reason:             "DeploymentAvailable",
      Message:            "all replicas ready",
      ObservedGeneration: foo.Generation,
  })
  // and meta.IsStatusConditionTrue / FindStatusCondition to read.
  ```
  Conventionally a positive-polarity `Ready`/`Available` plus `Progressing`; pick a vocabulary and be
  consistent. (The Kubernetes API conventions describe the standard condition semantics.)

- **Only write status when it actually changed.** A no-op status `Update` still bumps `resourceVersion`
  and fires a watch event → which (if you don't filter) re-enqueues you → **hot loop**. Compare before
  writing, or rely on `meta.SetStatusCondition` not changing `LastTransitionTime` when the status is
  unchanged. Even so, gate the write: `if !apiequality.Semantic.DeepEqual(old.Status, foo.Status) { Status().Update(...) }`.
- **Spec and status writes are separate calls.** Don't try to update both in one `Update` when the
  subresource is on. Reconcile order: reconcile real resources first, then write status last.

---

## 8. Manager wiring: leader election, shutdown, metrics, logging, health

(See `examples.md` for a full `main.go`.)

- **Leader election** — run multiple replicas for availability, but only the leader reconciles.
  `manager.Options{LeaderElection: true, LeaderElectionID: "<stable-id>", LeaderElectionNamespace: ...}`.
  Uses a `Lease` object. Without it, two replicas both reconcile and **fight** (double writes, races).
  Tune `LeaseDuration`/`RenewDeadline`/`RetryPeriod` only if you understand the failover trade-off.
- **Graceful shutdown** — `mgr.Start(ctx)` returns when `ctx` is cancelled (wire `SIGTERM`/`SIGINT`
  via `ctrl.SetupSignalHandler()`). The Manager stops sources, lets in-flight reconciles finish (up to
  a grace period), and releases the leader lease so failover is fast. Don't `os.Exit` out from under it.
- **Metrics** — the Manager serves controller-runtime metrics (reconcile count/latency/errors, workqueue
  depth/latency, per-controller) on a metrics endpoint (commonly `:8080`; recent versions default to a
  protected `:8443` — **verify your version's default and auth**). Watch `workqueue_depth`,
  `controller_runtime_reconcile_errors_total`, and reconcile latency in prod.
- **Structured logging** — use **`logr`** (`sigs.k8s.io/controller-runtime/pkg/log`). Get the
  request-scoped logger with `log.FromContext(ctx)` inside Reconcile (it's pre-populated with the
  reconciled object's GVK/namespace/name). Set the global logger once at startup
  (`ctrl.SetLogger(zap...)`). Log **key/value structured**, not formatted strings. Defer to
  [[go-best-practices]] on logging discipline (log at boundaries, never log-and-return the same error).
- **Health & readiness** — register `mgr.AddHealthzCheck("healthz", healthz.Ping)` and
  `mgr.AddReadyzCheck("readyz", healthz.Ping)`; serve on the health probe bind address. Readiness
  should reflect cache sync (controller-runtime can gate readiness on informer sync) so the pod isn't
  marked ready before its caches are warm.

---

## 9. Anti-patterns & gotchas (the production traps)

- **Status-write hot loop.** Your reconcile writes status → status change fires your own watch →
  re-enqueue → write status again. Fix: (a) only write when status truly changed, and (b) filter
  primary-object updates with `predicate.GenerationChangedPredicate{}` so status/metadata-only updates
  don't re-enqueue (generation only bumps on **spec** changes). Both together.
- **Controllers fighting over a field.** Two controllers `Update` the same field with different values
  → infinite write war, each undoing the other. Fix: one owner per field. SSA field ownership makes the
  conflict explicit (you'll get a conflict error instead of silent clobbering); use distinct
  `FieldOwner`s and don't both manage the same path.
- **Non-idempotent side effects.** "Create a cloud resource" on every reconcile without checking it
  exists → duplicates / quota exhaustion / cost. Make the operation a converge: look it up (by a
  deterministic name derived from the object, or an ID stored in status), create only if absent.
- **Trusting the cache right after a write.** `Create(x)` then `Get(x)` from the cache → `NotFound`,
  because the cache updates asynchronously. Don't treat that as an error; requeue and re-observe. Don't
  paper over it with an uncached read unless you genuinely need read-after-write.
- **Unbounded goroutines / background work in Reconcile.** Reconcile must be synchronous and return.
  Don't spawn a goroutine that outlives the call, don't `time.Sleep` to "wait" for a resource — return
  `RequeueAfter` and let the queue bring you back. Long sleeps hold a worker slot and stall the queue.
- **Missing or wrong RBAC.** The controller's ServiceAccount needs exactly the verbs it uses
  (`get;list;watch` for cached types, `create;update;patch;delete` for managed types, `update` on
  `<resource>/status` and `<resource>/finalizers`, `create;patch` on `events`, plus
  `coordination.k8s.io/leases` for leader election). Symptoms: reconcile loops with `Forbidden`
  errors, or watches that never establish. Keep RBAC markers (`// +kubebuilder:rbac:...`) in sync with
  the code; regenerate and review. Least privilege — scope to namespaces where possible.
- **Requeue storms.** A bad error path that re-enqueues immediately, a resync set too short, or a map
  function enqueuing thousands of keys per event → workqueue saturates, reconcile latency explodes.
  Watch queue depth; use backoff (return error, not bare requeue); cap mapping fan-out.
- **Full-namespace LISTs in hot paths.** `List` without a selector over a huge namespace on every
  reconcile is O(objects). Use field indexes (§10) and selectors.
- **Reconciling cluster state you don't own.** Don't manage objects another controller owns; you'll
  fight. Own the GVKs you create; merely *read* the ones you depend on.
- **Mutating a cached object in place.** Objects returned by `Get`/`List` are **pointers into the
  shared cache** (controller-runtime gives you a copy on `Get` into your struct, but be careful with
  list items and any deep references). Never mutate-then-not-write assuming it's local; never write back
  a cached object without `DeepCopy` if you've been holding it. When in doubt, `obj.DeepCopy()`.
- **Ignoring `Generation` semantics.** `metadata.generation` increments on **spec** writes only (for
  resources with a status subresource), not on status/metadata. That's exactly why
  `GenerationChangedPredicate` and `observedGeneration` work — don't reinvent change-detection.

---

## 10. Performance & scale

- **Workqueue depth is your primary saturation signal.** Sustained nonzero depth → reconciles can't
  keep up. Raise `MaxConcurrentReconciles`, make Reconcile cheaper, or cut event volume with predicates.
- **Resync cost scales with object count.** At each resync every watched object re-enqueues. Long
  resync (or none) for large fleets; never set a short resync "just in case."
- **Scope the cache.** By default the Manager caches **all** objects of every watched GVK
  cluster-wide — memory grows with cluster size. Reduce it:
  - **Namespace scoping:** `cache.Options{DefaultNamespaces: map[string]cache.Config{ns: {}}}` to watch
    only specific namespaces.
  - **Label/field selectors per GVK:** `cache.Options{ByObject: map[client.Object]cache.ByObject{
    &v1.Pod{}: {Label: labels.SelectorFromSet(...)}}}` so the informer only caches objects matching the
    selector — fewer objects, smaller watch, fewer reconciles. (Verify the exact `cache.Options` shape
    for your controller-runtime version; this API has evolved.)
- **Field indexes for cheap lookups.** Register an index at startup so you can `List` by a field
  without scanning:
  ```go
  mgr.GetFieldIndexer().IndexField(ctx, &v1.Pod{}, "spec.nodeName",
      func(o client.Object) []string { return []string{o.(*v1.Pod).Spec.NodeName} })
  // then: r.List(ctx, &pods, client.MatchingFields{"spec.nodeName": node})
  ```
  Use this for the "find all children referencing X" pattern in mapping functions instead of LISTing
  everything and filtering in Go.
- **Cheap mapping functions.** `EnqueueRequestsFromMapFunc` runs on every event for the watched type;
  it must be cache-only and O(1)/indexed, never do API calls or full LISTs.
- **Avoid reconcile amplification.** One config change fanning out to N objects (each reconcile writing
  status, re-triggering...) can melt a cluster. Bound fan-out; rate-limit; consider whether the work
  belongs in one reconcile pass.
- **Client-side rate limiting.** The REST client has QPS/Burst limits (`rest.Config.QPS`/`Burst`).
  Defaults are low for high-throughput controllers; raise deliberately and watch apiserver load (and
  priority-and-fairness). Too-low QPS shows up as `client-side throttling` log lines and latency.

---

## 11. Testing controllers

- **envtest** (`sigs.k8s.io/controller-runtime/pkg/envtest`) spins up a **real apiserver + etcd** (no
  kubelet, no controller-manager, no scheduler) so you can test Reconcile against genuine API
  semantics: admission, validation, defaulting, status subresource, finalizers, owner refs, optimistic
  concurrency. This is the **right** level for controller behavior tests. You install your CRDs
  (`CRDDirectoryPaths`), start the env, build a Manager or call Reconcile directly, and assert on
  cluster state via the client. Note: **no kubelet means Pods never become Running** and built-in
  controllers don't run — so e.g. a Deployment won't actually create Pods; assert on the objects your
  controller creates, and simulate downstream status if needed.
- **fake client** (`sigs.k8s.io/controller-runtime/pkg/client/fake`) is an in-memory client for **pure
  reconcile logic**. Fast, no binaries. **Caveats — it does NOT faithfully emulate the apiserver:**
  - **Server-Side Apply** support has been limited/incorrect across versions — don't trust fake-client
    SSA; test SSA paths with envtest. (Verify your version's current SSA fidelity.)
  - **No admission, no defaulting, no validation, no GC.** Owner-reference garbage collection does not
    happen; finalizer deletion semantics are emulated only partially.
  - The status subresource must be enabled explicitly (`WithStatusSubresource(&v1.Foo{})`) or status
    writes via `Status().Update` won't behave like the real server.
  - Use it for "given this input object, does Reconcile compute the right desired object / status",
    not for testing API-server-mediated behavior.
- **Table-driven reconcile tests** are the default for logic (see [[go-best-practices]] §11): a slice of
  named cases, each with seed objects → run Reconcile → assert resulting objects/status/result. Pure
  helper functions (`buildDesiredDeployment(spec)`) are the easiest, fastest things to unit-test — push
  logic into them.
- **Ginkgo/Gomega vs std `testing`.** Kubebuilder scaffolds **Ginkgo** suites with envtest, and they're
  idiomatic for async, eventually-consistent assertions (`Eventually(...).Should(...)`). Plain `testing`
  + table tests is perfectly fine and often clearer for deterministic logic. Use Ginkgo where you need
  `Eventually` polling against a live envtest apiserver; use std `testing` for unit logic. Don't fight
  the team's existing choice.
- **Assert with `Eventually`** for anything the apiserver/cache makes eventually consistent — never a
  bare `Get` immediately after a write in envtest (same staleness as prod). Poll with a timeout.
- Run with `-race`; controllers are concurrent. Inject a fake clock for TTL/requeue-timing logic rather
  than sleeping.

---

## 12. Review checklist (paste into controller PRs)

- [ ] Reconcile is **level-triggered**: re-reads current state, never depends on event type/order/old value.
- [ ] Reconcile is **idempotent**: a no-op reconcile makes zero writes; running twice is safe.
- [ ] **Fetch-or-return on NotFound** (`client.IgnoreNotFound`); no child cleanup on parent-not-found.
- [ ] **Owner references** set on all created children (`SetControllerReference`); one controller owner;
      no cross-namespace ownerRefs.
- [ ] **Finalizer** added before external resources exist; cleanup idempotent; finalizer **always**
      removed after success; can't wedge deletion forever.
- [ ] **Status** via subresource; `observedGeneration` set; `metav1.Condition` via `meta` helpers;
      written **only when changed**.
- [ ] **Requeue** semantics correct: error→backoff, `RequeueAfter`→poll, empty→done; no busy-loop, no
      blocking sleeps in Reconcile, no `RequeueAfter: 0`.
- [ ] **Predicates**: `GenerationChangedPredicate` (or equivalent) so status/metadata updates don't
      re-enqueue; mapping functions cache-only and bounded.
- [ ] **Writes go to API, reads from cache**; no trusting read-after-write; conflicts handled (or SSA).
- [ ] **SSA**: stable unique `FieldOwner`; not mixed with client-side `Update` on the same object.
- [ ] **RBAC** markers match actual verbs (incl. `/status`, `/finalizers`, `leases`, `events`); least privilege.
- [ ] **No hot loop / requeue storm / controller fight**; `MaxConcurrentReconciles` and cache scope
      sized for the workload; selectors/field indexes instead of full LISTs.
- [ ] **Manager**: leader election on for HA; signal-handler ctx for graceful shutdown; health/ready
      checks; metrics exposed; structured `logr` logging.
- [ ] **Tests**: envtest for API-mediated behavior, fake client only for pure logic (SSA/GC caveats
      respected), `Eventually` for async, `-race` clean.

---

## Canonical references

- Kubebuilder Book — https://book.kubebuilder.io (architecture, reconcile, finalizers, webhooks)
- controller-runtime godoc — https://pkg.go.dev/sigs.k8s.io/controller-runtime *(read the tag in your go.mod)*
- client-go architecture — https://github.com/kubernetes/sample-controller (the canonical informer/workqueue example)
- "A Deep Dive into Kubernetes Controllers" / controller patterns — https://kubernetes.io/docs/concepts/architecture/controller/
- Kubernetes API conventions (conditions, observedGeneration, status) —
  https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md
- Server-Side Apply — https://kubernetes.io/docs/reference/using-api/server-side-apply/
- Garbage collection & owner references — https://kubernetes.io/docs/concepts/architecture/garbage-collection/
- Finalizers — https://kubernetes.io/docs/concepts/overview/working-with-objects/finalizers/
- Kueue and JobSet/LeaderWorkerSet sources are excellent real-world controllers to study —
  see [[kueue-advanced]] and [[jobset-leaderworkerset]].

---

# Kubernetes Controller Examples

Canonical, imitate-this artifacts for a controller-runtime controller. Go idiom follows
**[[go-best-practices]]**; controller semantics follow `kubernetes-controller-expert-guide.md`.

These compile in spirit against a recent controller-runtime; **verify exact imports/signatures against
the version in your `go.mod`** (the framework moves with each Kubernetes minor). Import paths and the
`cache.Options`/metrics defaults are the most version-sensitive parts.

Assume an API type `Foo` (group `apps.example.com/v1`) with a `.spec.replicas` and `.spec.image`, a
status subresource, and these kubebuilder markers on the type:

```go
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`
```

---

## 1. A canonical Reconciler

Demonstrates: fetch-or-return-on-NotFound, finalizer add/remove with idempotent external cleanup,
desired-state construction, owner references, **server-side apply**, observation, and a status update
with `observedGeneration` + `metav1.Condition` written only when changed.

```go
package controller

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"github.com/go-logr/logr"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	appv1 "example.com/foo/api/v1"
)

const (
	finalizer  = "apps.example.com/cleanup"
	fieldOwner = client.FieldOwner("foo-controller")
)

// FooReconciler reconciles a Foo object by managing a backing Deployment.
type FooReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// RBAC — keep these in sync with the verbs the code actually uses.
// +kubebuilder:rbac:groups=apps.example.com,resources=foos,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=apps.example.com,resources=foos/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=apps.example.com,resources=foos/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

func (r *FooReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// 1. Fetch the primary object from the cache. Gone? Nothing to do — GC handles children.
	var foo appv1.Foo
	if err := r.Get(ctx, req.NamespacedName, &foo); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// 2. Deletion path: run idempotent external cleanup, then drop the finalizer.
	if !foo.DeletionTimestamp.IsZero() {
		if controllerutil.ContainsFinalizer(&foo, finalizer) {
			if err := r.deleteExternalResources(ctx, &foo); err != nil {
				return ctrl.Result{}, fmt.Errorf("external cleanup: %w", err) // retry w/ backoff
			}
			controllerutil.RemoveFinalizer(&foo, finalizer)
			if err := r.Update(ctx, &foo); err != nil {
				return ctrl.Result{}, fmt.Errorf("remove finalizer: %w", err)
			}
		}
		return ctrl.Result{}, nil // object is now deletable
	}

	// 3. Ensure our finalizer is present BEFORE creating external resources.
	if !controllerutil.ContainsFinalizer(&foo, finalizer) {
		controllerutil.AddFinalizer(&foo, finalizer)
		if err := r.Update(ctx, &foo); err != nil {
			return ctrl.Result{}, fmt.Errorf("add finalizer: %w", err)
		}
		return ctrl.Result{}, nil // requeue with the finalizer persisted
	}

	// 4. Build DESIRED state purely from spec, stamp the owner reference.
	desired := desiredDeployment(&foo)
	if err := controllerutil.SetControllerReference(&foo, desired, r.Scheme); err != nil {
		return ctrl.Result{}, fmt.Errorf("set owner ref: %w", err)
	}

	// 5. Apply desired → observed via Server-Side Apply: idempotent, no read, conflict-free.
	if err := r.Patch(ctx, desired, client.Apply, fieldOwner, client.ForceOwnership); err != nil {
		return ctrl.Result{}, fmt.Errorf("apply deployment: %w", err)
	}

	// 6. Observe the child to compute status. Not in cache yet → requeue and re-observe.
	var dep appsv1.Deployment
	if err := r.Get(ctx, client.ObjectKeyFromObject(desired), &dep); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// 7. Status: observedGeneration + Ready condition, written ONLY if it changed.
	return r.reconcileStatus(ctx, &foo, &dep, logger)
}

func (r *FooReconciler) reconcileStatus(
	ctx context.Context, foo *appv1.Foo, dep *appsv1.Deployment, logger logr.Logger,
) (ctrl.Result, error) {
	base := foo.DeepCopy() // compare against this to avoid no-op status writes

	ready := dep.Status.ReadyReplicas == *foo.Spec.Replicas
	cond := metav1.Condition{
		Type:               "Ready",
		ObservedGeneration: foo.Generation,
		Status:             metav1.ConditionFalse,
		Reason:             "DeploymentProgressing",
		Message:            fmt.Sprintf("%d/%d replicas ready", dep.Status.ReadyReplicas, *foo.Spec.Replicas),
	}
	if ready {
		cond.Status = metav1.ConditionTrue
		cond.Reason = "DeploymentAvailable"
		cond.Message = "all replicas ready"
	}
	foo.Status.ObservedGeneration = foo.Generation
	foo.Status.ReadyReplicas = dep.Status.ReadyReplicas
	meta.SetStatusCondition(&foo.Status.Conditions, cond)

	if apiequality.Semantic.DeepEqual(base.Status, foo.Status) {
		return ctrl.Result{}, nil // nothing changed — DON'T write (avoids a self-triggered watch)
	}
	if err := r.Status().Update(ctx, foo); err != nil {
		if apierrors.IsConflict(err) {
			return ctrl.Result{}, nil // someone else won; we'll be re-enqueued
		}
		return ctrl.Result{}, fmt.Errorf("update status: %w", err)
	}
	return ctrl.Result{}, nil
}

// desiredDeployment is a PURE function of spec — easy to unit-test, deterministic name for idempotency.
func desiredDeployment(foo *appv1.Foo) *appsv1.Deployment {
	labels := map[string]string{"app.kubernetes.io/managed-by": "foo-controller", "foo": foo.Name}
	return &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"}, // required for SSA
		ObjectMeta: metav1.ObjectMeta{
			Name:      foo.Name,
			Namespace: foo.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: foo.Spec.Replicas,
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "app",
						Image: foo.Spec.Image,
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("100m"),
								corev1.ResourceMemory: resource.MustParse("128Mi"),
							},
						},
					}},
				},
			},
		},
	}
}

// deleteExternalResources MUST be idempotent and tolerate "already gone".
func (r *FooReconciler) deleteExternalResources(ctx context.Context, foo *appv1.Foo) error {
	// e.g. delete a cloud LB / DNS record keyed deterministically off foo. Deleting twice, or when it
	// was never created, must succeed. The Deployment itself is GC'd via the owner ref — not here.
	_ = ctx
	_ = foo
	return nil
}

// SetupWithManager wires the controller: reconcile Foo, own its Deployment, skip status-only updates.
func (r *FooReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&appv1.Foo{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.Deployment{}). // Deployment events → owner Foo via the controller ownerRef
		WithOptions(controller.Options{MaxConcurrentReconciles: 4}).
		Named("foo").
		Complete(r)
}
```

Notes on what makes this correct:

- **`For(...).WithPredicates(GenerationChangedPredicate{})`** stops the controller's *own* status writes
  (which don't bump `.metadata.generation`) from re-enqueuing it — the canonical hot-loop fix.
- **SSA (`client.Apply`)** needs `TypeMeta` populated and a stable `FieldOwner`; it requires no prior
  read and is conflict-free for the fields this controller owns.
- **Status written only when changed**, via the **subresource**, with `ObservedGeneration` set on both
  the status and the condition.
- **Finalizer added before** any external resource exists; cleanup idempotent; finalizer **always**
  removed on success → no stuck `Terminating`.
- `desiredDeployment` is a **pure function** → trivially table-testable without a cluster.

---

## 2. A minimal `main.go` wiring a Manager

Demonstrates: scheme registration, leader election (HA), graceful-shutdown signal context, health/ready
probes, metrics, structured logging.

```go
package main

import (
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	appv1 "example.com/foo/api/v1"
	"example.com/foo/internal/controller"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme)) // built-in types (Deployment, etc.)
	utilruntime.Must(appv1.AddToScheme(scheme))          // our CRD types
}

func main() {
	var metricsAddr, probeAddr string
	var enableLeaderElection bool
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8443", "metrics endpoint")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "health probe endpoint")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false, "enable leader election for HA")
	opts := zap.Options{Development: false}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts))) // structured logr backend, set ONCE

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsserver.Options{BindAddress: metricsAddr},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "foo-controller.apps.example.com", // stable, unique per controller
		// SyncPeriod / Cache scoping go here. Default cache is cluster-wide; scope it for scale:
		//   Cache: cache.Options{DefaultNamespaces: map[string]cache.Config{"my-ns": {}}},
		// (verify cache.Options shape for your controller-runtime version)
	})
	if err != nil {
		setupLog.Error(err, "unable to create manager")
		os.Exit(1)
	}

	if err := (&controller.FooReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to set up controller", "controller", "Foo")
		os.Exit(1)
	}

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	// SetupSignalHandler returns a ctx cancelled on SIGTERM/SIGINT → graceful drain + lease release.
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
```

Notes:

- **`ctrl.GetConfigOrDie()`** loads in-cluster config (or `$KUBECONFIG` locally).
- **Leader election** ON in production: only the leader reconciles; replicas stand by for fast
  failover. Without it, every replica reconciles and they **fight**.
- **`mgr.Start(ctrl.SetupSignalHandler())`** blocks until SIGTERM; the Manager then stops sources, lets
  in-flight reconciles finish, and releases the leader `Lease`. Never `os.Exit` out from under it.
- The metrics bind address default has shifted across controller-runtime versions (and recent versions
  protect it with auth) — **confirm the default and protection for your version** before exposing it.

---

## 3. Watching a non-owned related object (mapping function)

When `Foo` references a `ConfigMap` it does **not** own, watch the ConfigMap and map its events back to
every `Foo` that references it — using a **field index** so the lookup is cheap, not a full LIST+filter.

```go
// In SetupWithManager: register an index on the referenced ConfigMap name, once.
const cmRefField = ".spec.configMapRef"

if err := mgr.GetFieldIndexer().IndexField(ctx, &appv1.Foo{}, cmRefField,
	func(o client.Object) []string {
		ref := o.(*appv1.Foo).Spec.ConfigMapRef
		if ref == "" {
			return nil
		}
		return []string{ref}
	}); err != nil {
	return err
}

// Then on the builder:
//   .Watches(&corev1.ConfigMap{}, handler.EnqueueRequestsFromMapFunc(r.foosForConfigMap))

// foosForConfigMap maps one ConfigMap event → the Foos that reference it. Cache-only, bounded.
func (r *FooReconciler) foosForConfigMap(ctx context.Context, cm client.Object) []ctrl.Request {
	var foos appv1.FooList
	if err := r.List(ctx, &foos,
		client.InNamespace(cm.GetNamespace()),
		client.MatchingFields{cmRefField: cm.GetName()},
	); err != nil {
		return nil
	}
	reqs := make([]ctrl.Request, 0, len(foos.Items))
	for i := range foos.Items {
		reqs = append(reqs, ctrl.Request{
			NamespacedName: client.ObjectKeyFromObject(&foos.Items[i]),
		})
	}
	return reqs
}
```

This is the pattern Kueue, JobSet, and LeaderWorkerSet use heavily — see [[kueue-advanced]] and
[[jobset-leaderworkerset]] for production-scale examples of indexed watches and mapping functions.
