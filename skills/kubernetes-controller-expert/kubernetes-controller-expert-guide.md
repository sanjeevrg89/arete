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

## Rationalizations & rebuttals

The excuses that show up in controller PRs and reconcile code, each with the one-line rebuttal.

- **"I'll trust event ordering / the old object — I know an update fired."** A `reconcile.Request`
  carries only `namespace/name`. Informers coalesce, drop, replay, and resync delivers synthetic
  updates. Re-read current state and recompute every time; if you want "the previous value," your
  design is edge-triggered and wrong (§1).
- **"Skip the finalizer — GC will clean it up."** GC only deletes objects via ownerRefs, in-cluster,
  same-namespace. It cannot delete external resources (cloud LBs, DNS, S3, DB rows) or cross-namespace
  things. Without a finalizer those leak silently on delete (§5, §6).
- **"Status writes are cheap — just write it every reconcile."** A no-op status `Update` still bumps
  `resourceVersion`, fires your own watch, and re-enqueues you → hot loop. Gate the write on an actual
  change and filter with `GenerationChangedPredicate` (§7, §9).
- **"No need for idempotency — it only runs once per change."** It does not. The queue resyncs,
  retries with backoff, and re-enqueues on any watch event. A non-idempotent "create cloud resource"
  becomes duplicates, quota exhaustion, and cost. Make every action a converge: look up, create only
  if absent (§1, §9).
- **"I'll just read my own write back from the cache to confirm it."** The cache updates
  asynchronously via the watch; a `Get` right after `Create` returns `NotFound`. Don't treat that as
  an error and don't paper over it with an uncached read — requeue and re-observe (§2, §9).
- **"One replica is fine, I don't need leader election."** The moment a rollout runs two replicas,
  both reconcile and fight: double writes, races, write wars. Turn on leader election for any
  multi-replica deployment (§8).
- **"I'll spawn a goroutine / sleep to wait for the resource to settle."** Reconcile must be
  synchronous and return. A background goroutine outlives the call and escapes the queue's
  guarantees; a sleep holds a worker slot and stalls the queue. Return `RequeueAfter` (§9).
- **"`MaxConcurrentReconciles` is high, so I can share state across keys for speed."** The queue only
  guarantees a single key isn't reconciled concurrently; different keys run in parallel. Shared
  mutable state across reconciles is a data race. Keep Reconcile stateless (§3).

## Red flags

Signals that the current approach is wrong — stop and reconsider.

- **Hot-loop requeues.** Reconcile count/latency climbing on objects nobody changed, or
  `controller_runtime_reconcile_total` ticking with no spec edits → status-write loop or missing
  `GenerationChangedPredicate` (§7, §9).
- **Requeue storms / saturated workqueue.** Sustained nonzero `workqueue_depth`, a map function
  enqueuing thousands of keys per event, a resync set too short, or an error path that re-enqueues
  immediately (§9, §10).
- **Unbounded goroutines or sleeps in Reconcile.** Any `go func()` that outlives the call or
  `time.Sleep` to "wait" — workers leak or stall instead of returning `RequeueAfter` (§9).
- **Missing or wrong RBAC.** `Forbidden` errors in logs, watches that never establish, or RBAC
  markers out of sync with the verbs used (esp. `/status`, `/finalizers`, `leases`, `events`) (§9).
- **Cache-staleness races.** Code that `Get`s its own write back, or assumes "I just created X so X
  exists" — reads are from a cache that lags the API server (§2, §9).
- **Edge-triggered assumptions.** Branching on event type, reaching for the "old" object, or relying
  on seeing every intermediate state — the request is a hint, not a payload (§1).
- **Objects stuck `Terminating`.** A finalizer that isn't removed after cleanup (bug, permanent
  error, panic before `RemoveFinalizer`, or controller uninstalled with finalizers live) wedges
  deletion forever and hangs namespace deletion (§6).
- **Controllers fighting over a field.** Two writers flip the same field back and forth (infinite
  write war), or two `controller: true` owner refs on one object → no single managing owner (§5, §9).
- **Full-namespace LISTs in hot paths.** `List` without a selector on every reconcile, or a mapping
  function doing API calls / O(objects) work instead of using a field index (§10).
- **Trusting `fake` client for API-mediated behavior.** SSA, GC, admission, defaulting, and finalizer
  semantics are not faithfully emulated — passing fake-client tests prove nothing about those (§11).

## Verification gate (definition of done)

The work is not done until all of these are true and demonstrated.

- [ ] **Idempotent reconcile**: running Reconcile twice on unchanged state makes **zero writes**;
      verified by an envtest/table test asserting no diff on the second pass.
- [ ] **Level-triggered**: re-reads current state each call; no dependency on event type/order/old value.
- [ ] **Finalizer add/remove correct**: added before any external resource exists; cleanup idempotent
      and tolerates "already gone"; finalizer **always** removed after success; can't wedge deletion.
- [ ] **Owner refs**: `SetControllerReference` on every created child; exactly one `controller: true`
      owner; no cross-namespace / namespaced→cluster-scoped ownerRefs.
- [ ] **Status conditions**: written via the status subresource; `observedGeneration` set;
      `metav1.Condition` managed with `meta` helpers; written **only when changed**.
- [ ] **Requeue semantics**: error→backoff, `RequeueAfter`→poll, empty→done; no `RequeueAfter: 0`, no
      blocking sleeps, no goroutines outliving the call.
- [ ] **Writes→API, reads→cache**: no read-after-write trust; conflicts handled or avoided via SSA
      with a stable unique `FieldOwner` (not mixed with client-side `Update` on the same object).
- [ ] **RBAC** markers match actual verbs (incl. `/status`, `/finalizers`, `leases`, `events`); least
      privilege; controller starts and watches establish without `Forbidden`.
- [ ] **No hot loop / requeue storm**: `GenerationChangedPredicate` (or equivalent) in place; mapping
      functions cache-only and bounded; `workqueue_depth` returns to zero at steady state.
- [ ] **envtest passing** for API-mediated behavior (status subresource, finalizers, owner refs,
      optimistic concurrency); fake client used only for pure logic with SSA/GC caveats respected.
- [ ] **Race-clean**: full suite passes under `go test -race`.

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
