---
name: kubernetes-internals-expert
description: Source-level Kubernetes internals for debugging control-plane and node problems others can't. Use when investigating WHY the cluster behaves as it does, not just how to use it — apiserver request lifecycle (auth/authz/admission/CEL/etcd), API machinery (scheme, GVK/GVR, codecs, conversion, defaulting), the watch cache, resourceVersion/watch/bookmarks/pagination, APF/FlowSchema priority & fairness, the aggregation layer/APIServices, admission webhook plumbing; etcd (Raft, MVCC/revisions, keyspace, compaction/defrag, quota/NOSPACE, quorum); the scheduler framework (PreFilter→Bind, preemption, nominatedNodeName, scheduling queues); controller-manager (shared informers, garbage collector + owner refs/finalizers, node lifecycle/Leases, lease-based leader election); kubelet (syncLoop, PLEG, CRI/containerd/CRI-O, cgroups v2, QoS & eviction manager, device plugins, DRA, CSI, static pods); networking dataplane (kube-proxy iptables/IPVS/nftables, Service VIPs, EndpointSlices, CNI, conntrack); and cross-cutting mechanics (level-triggered reconciliation, finalizers, field/label selectors & indexes, server-side apply/managedFields). Reach for it on symptoms like Pods stuck Pending/Terminating/ContainerCreating, NotReady nodes, slow apiserver, etcd NOSPACE, 410 Gone, 429/APF throttling, stuck namespaces, OOMKill-vs-eviction, PLEG unhealthy, and when reading apiserver/etcd/scheduler/kubelet logs & metrics or using `kubectl get --raw`/`etcdctl`/`crictl`.
---

# Kubernetes Internals Expert

Apply the judgment of a SIG contributor who reads the source: someone who debugs apiserver, etcd,
scheduler, controller-manager, kubelet, and dataplane problems by reasoning about how each component
*actually works*, then confirms against source/KEPs/metrics rather than guessing.

## How to use this skill

1. **Read `kubernetes-internals-expert-guide.md`** in this directory — the full reference, organized
   by component (apiserver → etcd → scheduler → controller-manager → kubelet → networking →
   cross-cutting → debugging). Apply it to the task.
2. For any failure, **localize to the component and its loop**: which controller/runtime owns the
   field or action, what did it observe, and did its loop run? Internals are conceptual — reason about
   the mechanism, then verify with the right metric/log/raw query (§8 of the guide).
3. **Verify version-sensitive claims** against the cluster's release notes / the KEP / source before
   relying on specifics. It is 2026 — DRA, the nftables proxy, CEL admission policies, and APF are all
   moving. Never fabricate a feature-gate status, API field, flag, or version number; if unsure, say so.

## Essentials (full detail in `kubernetes-internals-expert-guide.md`)

- **Everything is level-triggered reconciliation over a consistent store.** Components write desired
  state (spec) and report observed state (status); independent loops converge. Watches+resync are an
  optimization — act on *current state*, not the event. There is no global transaction; eventual
  consistency is the contract.
- **Apiserver write path is a fixed pipeline:** auth → authz → mutating admission → schema validation
  → validating admission → CEL policy → etcd write. Reads skip admission and are served from the
  **watch cache**. Know this order to explain any reject/mutate/not-persisted.
- **`resourceVersion` is an opaque etcd-revision token**, not a counter. `410 Gone` = relist needed;
  bookmarks advance RV cheaply; LIST paginates with `limit`/`continue`. `rv=0` = possibly-stale cache.
- **APF (FlowSchema → PriorityLevelConfiguration)** isolates misbehaving clients; `429` with PF
  headers tells you which flow. A **down aggregated APIService** (e.g. `metrics.k8s.io`) makes
  discovery/`kubectl` feel broken cluster-wide.
- **etcd is Raft + MVCC.** Odd member count for quorum; **compaction** drops history, **defrag**
  reclaims disk (etcd never shrinks itself); hitting the DB quota → **NOSPACE alarm**, writes refused.
  Large objects / fat `managedFields` / hot status-write loops hurt via Raft fsync — watch
  `etcd_disk_wal_fsync_duration_seconds`.
- **Scheduler framework:** PreFilter/**Filter** (hard feasibility) → PreScore/**Score** (soft ranking)
  → Reserve/**Permit** (gang hook) → bind. **Preemption** lives in PostFilter and sets
  `status.nominatedNodeName`. Stuck-Pending → check events + which queue (active/backoff/unschedulable).
- **Controller-manager = shared informers + workqueues.** GC uses **owner references**; deletion
  policy is foreground (finalizer-blocked) / background / orphan. Node liveness via **Leases**
  (`kube-node-lease`); HA via **lease-based leader election** (cooperative, not fencing — be idempotent).
- **kubelet syncLoop + PLEG drive the CRI** (containerd/CRI-O; dockershim gone since 1.24). cgroup v2
  enforces resources; **OOMKill (memory limit, exit 137) ≠ eviction (kubelet reclaiming node
  pressure)**. Device plugins advertise GPUs as countable resources; **DRA** is the modern accelerator
  path (verify GA/shape). Static pods bootstrap the control plane (mirror pods, no scheduler).
- **Dataplane:** Service ClusterIP is a virtual IP that **kube-proxy** programs (iptables vs IPVS vs
  **nftables** — verify default) using **EndpointSlices**; **CNI** gives every pod a NAT-free routable
  IP; **conntrack** staleness (esp. UDP/DNS) and table-full are classic intermittent-failure causes.
- **Finalizers** keep an object `Terminating` until every finalizer clears; #1 "stuck Terminating"
  cause is a finalizer whose controller is dead. **Server-side apply** tracks field ownership in
  `managedFields` (conflicts on contested fields; bloats objects).
- **Debug with the right lens:** `kubectl get --raw /metrics|/readyz?verbose`, `etcdctl endpoint
  status`/`alarm list`/`defrag`, `crictl`, `journalctl -u kubelet`, and the per-component metrics in
  §8. Match each symptom to its component before changing anything.

## Related skills

- `[[kubernetes-controller-expert]]` — writing controllers to this reconciliation model (client-go,
  informers, workqueues, leader election, finalizers in your own code).
- `[[kubernetes-expert]]` — end-to-end practitioner mastery (using Kubernetes) when the question isn't
  about internals.
- `[[autoscaling-kubernetes]]` — HPA/VPA/Cluster Autoscaler/Karpenter reacting to scheduler Pending
  pods and node pressure.
- `[[kueue-advanced]]` — gang/coscheduling, batch quota, and topology-aware scheduling above the
  default scheduler for ML/batch.
- `[[gke-master]]` — managed control plane, node pools, and how a provider operates these internals.

---

# Reference — kubernetes-internals-expert

# Kubernetes Internals Engineering Guide

Source-level Kubernetes internals, written for the engineer who debugs control-plane and node
problems others can't — the apiserver request lifecycle, etcd, the scheduler framework, the
controller-manager, the kubelet, and the networking dataplane. This is the single source of truth;
`SKILL.md` and `AGENTS.md` defer to it.

> **Accuracy bar.** This is the skill where fabrication is most damaging. Mechanism names, struct
> names, and especially feature-gate maturity drift between releases. It is **2026**: DRA, the
> nftables kube-proxy backend, ValidatingAdmissionPolicy/CEL, and APF have all moved. When a precise
> version/GA status matters, **verify against the current release notes, the KEP, and source** rather
> than trusting a number from memory. Source lives under `k8s.io/kubernetes` (staging repos like
> `k8s.io/apiserver`, `k8s.io/client-go`, `k8s.io/api`, `k8s.io/apimachinery` are mirrored there).

---

## 0. The one mental model: level-triggered reconciliation over a consistent store

Everything in Kubernetes is a loop: *observe desired + actual state from a shared store, compute a
diff, act to converge, repeat.* The store is etcd, fronted by the apiserver. Components don't command
each other; they **write desired state** (spec) and **report observed state** (status) as objects,
and independent controllers reconcile. This is **level-triggered, not edge-triggered**: a controller
acts on the *current observed state*, not on the event that woke it. Events (watches) are an
optimization to avoid polling — a missed or duplicated event must never cause incorrect behavior,
which is why every informer does a periodic **resync** (re-deliver current cache state) and every
reconcile must be idempotent. Internalize this and most "why didn't X happen" questions answer
themselves: find the controller that owns that field, read its observed inputs, and check whether its
loop ran and what it saw.

Corollaries:
- **Eventual consistency is the contract.** There is no global transaction across objects. A Pod
  referencing a missing ConfigMap is a legal intermediate state.
- **Status is a cache of reality, not a command.** Never drive logic off another controller's status
  without expecting it to be stale or absent.
- **Optimistic concurrency** (`resourceVersion` + CAS in etcd) is how writers avoid clobbering each
  other — there are no locks.

See [[kubernetes-controller-expert]] for writing controllers to this model; this guide is the
server/runtime side that makes it work.

---

## 1. kube-apiserver: the request lifecycle

A write request (`POST`/`PUT`/`PATCH`) passes through a fixed pipeline. Knowing the order is the key
to debugging "my object was rejected/mutated/not persisted":

1. **Auth (authentication)** — who are you? Chained authenticators: client cert, bearer token
   (ServiceAccount JWT, OIDC), webhook, request header. First to succeed wins; output is a
   `user.Info` (name, UID, groups, extra). Failure → `401`.
2. **Authz (authorization)** — may you do this verb on this resource? Chained authorizers, usually
   Node, RBAC, then webhook; **first explicit allow/deny wins**, default deny. Failure → `403`.
3. **Mutating admission** — webhooks (`MutatingWebhookConfiguration`) and built-in plugins can
   *change* the object (inject sidecars, defaults, labels). Runs before validation so mutations are
   themselves validated.
4. **Schema validation / decoding** — the body is decoded into a typed object via the **scheme** and
   **codecs**; structural-schema validation for CRDs; required fields, types, formats checked.
5. **Validating admission** — webhooks (`ValidatingWebhookConfiguration`) and built-in validators can
   *reject* but not change. 
6. **CEL policy admission** — `ValidatingAdmissionPolicy` (and the mutating-policy line of work, which
   has been maturing — **verify GA status for your release**) evaluate **CEL** expressions in-process,
   no webhook round-trip. Faster and safer than webhooks for many policies.
7. **Persist to etcd** — the storage layer encodes the object (protobuf under the hood) and does a
   guarded write (transaction conditioned on current revision for updates).

Reads (`GET`/`LIST`/`WATCH`) skip admission; they go auth → authz → storage/watch-cache.

**Built-in admission plugins** (run alongside webhooks, ordered by the apiserver) include
`NamespaceLifecycle`, `LimitRanger`, `ServiceAccount`, `ResourceQuota`, `PodSecurity` (Pod Security
Admission), `DefaultStorageClass`, `Priority`, `MutatingAdmissionWebhook`,
`ValidatingAdmissionWebhook`, `ValidatingAdmissionPolicy`. Enabled/ordered set is fixed in code; the
`--enable-admission-plugins`/`--disable-admission-plugins` flags toggle within that.

### Admission webhook plumbing (the part that causes outages)
- A webhook is an HTTPS endpoint receiving an `AdmissionReview` and returning one (with `allowed`,
  optional `patch` for mutating). `failurePolicy: Fail` means an unreachable webhook **blocks all
  matching writes** — a webhook serving Pods with `Fail` and a crashed backend can wedge the whole
  cluster (you can't even create the Pod that runs the webhook). Use `failurePolicy: Ignore` for
  non-critical webhooks, scope `namespaceSelector`/`objectSelector` tightly, and **always exclude the
  webhook's own namespace and kube-system**. Set a low `timeoutSeconds`.
- `reinvocationPolicy: IfNeeded` re-runs mutating webhooks if a later webhook changed the object.
- Ordering among webhooks is **not guaranteed**; design them order-independent.

### API machinery: scheme, GVK/GVR, codecs, conversion, defaulting
- **Group/Version/Kind (GVK)** identifies a type (`apps/v1`, `Deployment`); **Group/Version/Resource
  (GVR)** is the REST path (`apps/v1`, `deployments`). The **RESTMapper** maps between them.
- The **Scheme** (`runtime.Scheme`) is the registry: it knows every Go type↔GVK, conversion
  functions between versions, and defaulting functions. Each apiserver keeps an **internal
  (hub) version** ("`__internal`"); every external version converts to/from the hub, so you write
  N converters not N². 
- **Codecs/serializers**: JSON, YAML, and **protobuf** (the wire format between core components and
  etcd for built-in types; CRDs are JSON). The negotiated serializer is chosen from the `Accept`/
  `Content-Type` headers.
- **Defaulting** happens on decode (setting unset fields to defaults); **conversion** happens when a
  client requests a different version than stored. This is why an object created via `v1beta1` reads
  back correctly via `v1` — the storage version is converted on the fly.
- **Storage version**: exactly one version per resource is written to etcd. Changing it requires
  rewriting objects (`StorageVersionMigration`) so old encodings don't linger.

### resourceVersion, watch, list, bookmarks, pagination
- **`resourceVersion` (RV)** is an opaque token backed by the etcd revision. It is **not** a
  per-object counter you can do arithmetic on; treat it as opaque and monotonic *per resource type*.
  An object's RV is the etcd revision at which it last changed.
- **WATCH** streams changes since a given RV: `ADDED/MODIFIED/DELETED` events. The apiserver serves
  watches from the **watch cache** (see below), not etcd directly.
- **Bookmarks** (`BOOKMARK` events) periodically advance the client's observed RV without sending
  object data, so that on reconnect the client can resume from a recent RV and avoid an expensive
  relist (`410 Gone`). Enabled by default; client-go relies on them.
- **`410 Gone` ("too old resource version")** means the requested RV has been compacted out of the
  watch cache/etcd history. The client must **relist** to get a fresh RV, then re-watch. A flood of
  `410`s is a symptom of watch-cache pressure or etcd compaction lag.
- **LIST consistency**: a default `LIST` is served from the watch cache and is *consistent as of some
  recent RV*. `resourceVersion=0` means "any cached version, possibly stale, cheapest." A specific RV
  means "at least this fresh." **Consistent reads from cache** (serving a quorum-fresh list from the
  watch cache instead of hitting etcd) is a maturing optimization — verify its status/behavior in your
  release.
- **Pagination**: `limit` + `continue` token chunk large lists so a single `LIST` doesn't load all
  objects into apiserver memory and blow it up. `kubectl` and informers paginate. A `continue` token
  encodes the RV + key; if it expires you get `410` and restart.

### Priority & Fairness (APF / FlowSchema / PriorityLevelConfiguration)
APF replaced the old max-inflight flags. It classifies every request into a **FlowSchema** (by user,
verb, resource) which routes to a **PriorityLevelConfiguration**; each priority level has concurrency
shares and queues, so a misbehaving client (e.g. a controller hot-looping LISTs) is throttled into
its own queue instead of starving leader election or node heartbeats. Built-in levels protect
system traffic (`system`, `leader-election`, `node-high`, `workload-high`, `workload-low`,
`catch-all`, plus `exempt` which bypasses limits). Debug with `apiserver_flowcontrol_*` metrics and
the `/debug/api_priority_and_fairness/dump_*` endpoints; `429` with
`X-Kubernetes-PF-FlowSchema-UID` headers tells you which flow throttled you.

### Aggregation layer & APIServices
The aggregator (`kube-aggregator`) lets a separate server own an API group/version. An `APIService`
object maps a GVR to a backing Service (e.g. `metrics.k8s.io` → metrics-server, or a custom
extension apiserver). The main apiserver proxies matching requests there. A **down aggregated API
makes `kubectl get`/discovery slow or partially failing** for *all* groups because discovery
aggregates; a stuck `metrics.k8s.io` is a classic cause of "the whole API feels broken." Built-in
types and CRDs do **not** go through the aggregator — only registered APIServices do.

CRDs vs aggregation: CRDs are the default extension mechanism (declarative, structural schema, CEL
validation, served by the main apiserver). Reach for an aggregated apiserver only when you need
custom storage, protobuf, or behavior CRDs can't express.

---

## 2. etcd: the consistent store underneath everything

etcd is a **strongly consistent, replicated key-value store** using the **Raft** consensus protocol.
The apiserver is the *only* component that talks to it; everything else goes through the apiserver.

- **Quorum.** A cluster of `N` members tolerates `floor((N-1)/2)` failures. Run **odd numbers** (3 or
  5). Lose quorum → the cluster is read-only/unavailable for writes; the apiserver returns errors and
  the control plane stalls. Adding a 4th member *lowers* fault tolerance relative to its cost — don't.
- **MVCC & revisions.** etcd is multi-version: every write bumps a global **revision**; the apiserver's
  `resourceVersion` is derived from it. Old revisions are retained until **compaction**. Each key also
  has create/mod/version metadata. Watches and ranged reads can target a revision.
- **Keyspace.** Objects are stored under a registry prefix, roughly
  `/registry/<resource>/<namespace>/<name>` (e.g. `/registry/pods/default/web-0`), value = encoded
  object (protobuf for built-ins, JSON for CRDs). You can inspect with `etcdctl get --prefix /registry`
  — useful for "is the object even in etcd?" and for spotting an enormous key.
- **Compaction** discards revision history older than a point (the apiserver runs periodic
  compaction by default). Without it the keyspace and revision history grow unboundedly.
  **Defragmentation** (`etcdctl defrag`) reclaims the on-disk space freed by compaction — etcd does
  *not* shrink its file automatically. A common outage: keyspace hits the **DB quota**
  (`--quota-backend-bytes`, default 2 GiB historically — verify for your version), etcd goes into a
  **`NOSPACE` alarm** and refuses writes until you compact, defrag, **and disarm the alarm**.
- **Watch.** etcd watches stream key changes from a revision; the apiserver multiplexes these into
  the watch cache. A slow/overloaded etcd causes watch lag → stale caches → controllers acting on old
  state.
- **Why large objects and high write rates hurt.** etcd writes go through Raft (replicated, fsync'd to
  disk on a quorum). Large objects (huge ConfigMaps/Secrets, fat CRD status, giant
  `managedFields`/annotations) and high churn amplify Raft traffic, disk fsync latency, and memory.
  etcd is acutely sensitive to **disk write latency (fsync) and network RTT between members** — put
  it on fast local SSD/NVMe with low-latency, dedicated networking. `etcd_disk_wal_fsync_duration_*`
  and `etcd_disk_backend_commit_duration_*` are the metrics to watch; sustained high fsync = doom.
- **Common etcd outages:** quota/NOSPACE; defrag needed after a churn spike; one slow member dragging
  the quorum (Raft proposal latency); clock skew / cert expiry breaking peer mTLS; a runaway
  controller hot-writing an object thousands of times/sec (often a status-update loop) blowing up
  revisions and compaction load.

---

## 3. kube-scheduler: the scheduling framework

The scheduler watches for **unscheduled Pods** (`spec.nodeName == ""`) and binds each to a node. Since
v1.19 it is built on the **Scheduling Framework**: a pipeline of **extension points** each backed by
**plugins**. A scheduling cycle (per Pod, serial) then an async binding cycle:

| Phase | Extension points | Purpose |
|---|---|---|
| Scheduling (sync) | **PreFilter** | precompute/validate; can reject early |
| | **Filter** | per-node predicate — is this node *feasible*? (taints, affinity, resources, volume zone) |
| | **PostFilter** | runs only if no node is feasible — this is where **preemption** lives |
| | **PreScore / Score** | rank feasible nodes (0–100), e.g. `NodeResourcesFit`, `InterPodAffinity`, `ImageLocality`, `PodTopologySpread` |
| | **Reserve / Unreserve** | tentatively claim resources on the picked node (stateful plugins) |
| | **Permit** | allow / deny / **wait** — the hook **gang/coscheduling** uses to hold a Pod until siblings are ready |
| Binding (async) | **PreBind / Bind / PostBind** | e.g. provision/attach volumes, then write the binding |

- **Filter** is a hard gate (feasibility); **Score** is soft ranking. After scoring, the scheduler
  picks the highest score (ties broken with some randomization for spread) and **Reserves**, then runs
  the binding cycle which issues a **Binding** (`pods/binding` subresource sets `nodeName`). The
  scheduler does **not** start the container — it only assigns the node; the kubelet on that node
  takes over (level-triggered: kubelet watches Pods bound to it).
- **Scheduling queues**: **activeQ** (ready to schedule, heap ordered by priority), **backoffQ**
  (recently failed, waiting out exponential backoff before retry), **unschedulableQ** (no feasible
  node; moved back to active when a cluster event that *might* help occurs — e.g. a node added, a Pod
  deleted). Understanding these explains "why is my Pod stuck Pending" — check events and which queue
  it's parked in.
- **Preemption** (PostFilter): when a higher-`PriorityClass` Pod can't fit, the scheduler finds a
  node where evicting lower-priority Pods would make room, picks the minimal victim set respecting
  PDBs where possible, sets the pending Pod's **`status.nominatedNodeName`** (pod nomination — a hint
  that it's expected to land there once victims drain), and deletes the victims. Nomination prevents
  other Pods from grabbing the freed space and lets the scheduler track intent across cycles.
- **Performance**: `percentageOfNodesToScore` lets the scheduler stop filtering once "enough" feasible
  nodes are found in large clusters (it doesn't score all 5000 nodes). Throughput metrics:
  `scheduler_pod_scheduling_duration_seconds`, `scheduler_schedule_attempts_total`,
  `scheduler_pending_pods`.
- **Gang / co-scheduling / batch quota** is *not* in the default scheduler — it's done via a `Permit`-
  plugin (coscheduling) or, in batch/ML contexts, by **Kueue** (admission-level gang + quota) and
  **Volcano**. For ML/batch gang semantics, MultiKueue, and topology-aware scheduling, see
  [[kueue-advanced]]; for autoscaling interplay (scaling up nodes for Pending Pods) see
  [[autoscaling-kubernetes]].

---

## 4. kube-controller-manager: shared informers, GC, node lifecycle, leader election

The KCM runs dozens of built-in controllers in one process (Deployment, ReplicaSet, Job, Node,
EndpointSlice, ServiceAccount-token, ResourceQuota, GC, …). They share infrastructure:

- **Shared informer architecture.** A `SharedInformerFactory` keeps **one watch + one in-memory cache
  (the informer's `Indexer`/store) per resource type**, shared by all controllers that need it — so
  20 controllers watching Pods cost one watch, not twenty. Each controller registers event handlers
  that enqueue keys (`namespace/name`) into a **workqueue**; workers pop keys and reconcile by reading
  the **current** object from the cache (level-triggered — the event is just a wake-up). The workqueue
  **dedupes** (a key enqueued 5× while a worker is busy is processed once) and supports **rate-limited
  retries** with exponential backoff. **Resync** periodically re-enqueues everything to recover from
  missed events. This is the same client-go machinery controllers use ([[kubernetes-controller-expert]]).
- **Garbage collector.** Tracks **owner references** (`metadata.ownerReferences`) to build a dependency
  graph and delete dependents when owners go. Three **propagation policies** on delete:
  - **Background** (default for most): delete the owner immediately; GC reaps dependents async.
  - **Foreground**: owner gets a `foregroundDeletion` **finalizer** and stays in `Terminating` until
    all dependents with `blockOwnerDeletion: true` are gone, *then* the owner is removed.
  - **Orphan**: dependents survive, their owner ref is stripped (an `orphan` finalizer drives this).
  A wrong/missing owner ref → orphaned objects that never get cleaned up; a stuck dependent
  (finalizer that never clears) → owner stuck in foreground deletion. `ownerReferences` must point
  within the same namespace (cluster-scoped owners can own namespaced deps, not the reverse).
- **Node lifecycle controller.** Consumes the kubelet's **node heartbeat** — historically the Node's
  `status` plus the modern **`Lease`** objects in `kube-node-lease` (a tiny per-node Lease updated
  every few seconds is far cheaper on etcd than rewriting full node status). On missed heartbeats it
  marks the Node `NotReady`, and after a grace period taints it
  (`node.kubernetes.io/unreachable:NoExecute`), which triggers **taint-based eviction** of Pods
  (respecting their `tolerationSeconds`). Tuning these timers trades failover speed against flapping.
- **Leader election (lease-based).** HA control-plane components (KCM, scheduler, and many operators)
  run multiple replicas but only one acts. They contend on a **`coordination.k8s.io/Lease`** object:
  the holder renews `renewTime` before `leaseDurationSeconds` expires; if it fails to renew, another
  candidate acquires the lease and takes over. This is **cooperative, not fencing** — it relies on the
  old leader noticing it lost the lease and stopping (`OnStoppedLeading`). A leader that hangs while
  technically holding the lease, or clock skew, can cause split-brain; design controllers to be
  idempotent and to exit promptly on lost leadership. ([[kubernetes-controller-expert]] covers wiring
  `leaderelection` in controller-runtime.)

---

## 5. kubelet: syncLoop, PLEG, CRI, cgroups, eviction, devices, volumes

The kubelet is the node agent. It watches the apiserver for **Pods bound to its node** and drives the
container runtime to make reality match. It is the boundary between the declarative control plane and
the imperative OS.

- **syncLoop.** The kubelet's central loop multiplexes several channels: pod updates from the
  apiserver, the **PLEG** channel, periodic sync, housekeeping, and liveness/readiness results. Each
  event triggers `syncPod` for the affected Pod(s): compute the desired vs actual container set and
  reconcile (create/kill/restart containers, set up the sandbox, mount volumes, report status). It is
  level-triggered like everything else.
- **PLEG (Pod Lifecycle Event Generator).** The kubelet must learn when containers actually
  start/die. Classic PLEG **relists** the runtime's containers periodically and diffs to synthesize
  lifecycle events. **`PLEG is not healthy`** in node logs (a watchdog firing because relist took too
  long) is a classic symptom of an overloaded or wedged container runtime, disk I/O starvation, or too
  many pods/node. **Evented PLEG** (runtime pushes events via the CRI instead of pure polling) reduces
  this load — **verify its maturity/default in your release.**
- **CRI (Container Runtime Interface).** kubelet → runtime is a gRPC API (`RuntimeService` +
  `ImageService`). Dockershim was **removed in v1.24**; runtimes are **containerd** and **CRI-O**.
  The kubelet creates a **pod sandbox** (the pause container / network+IPC namespaces) first, then
  containers inside it. `crictl` is the node-level debug tool that speaks CRI directly (bypassing the
  apiserver) — use it to see what the runtime actually thinks is running when the apiserver view and
  reality diverge.
- **cgroups v2 & resource enforcement.** Modern nodes use the **cgroup v2** unified hierarchy.
  - **CPU `requests`** → cgroup `cpu.weight` (proportional share under contention; not a cap).
  - **CPU `limits`** → CFS bandwidth quota (`cpu.max`) — hard throttling; over-tight limits cause
    latency spikes from throttling even when the node is idle.
  - **Memory `limits`** → `memory.max`; exceeding it = **OOMKill** of the container (exit 137), *not*
    eviction. **Memory has no "throttle"** — it's a hard wall.
  - The kubelet lays out a **cgroup tree** by QoS: `Guaranteed`, `Burstable`, `BestEffort` slices,
    plus reservations for system/kube daemons (`--system-reserved`, `--kube-reserved`).
- **QoS classes & the eviction manager.** QoS is derived, not set: **Guaranteed** (every container has
  requests==limits for cpu+mem), **Burstable** (some requests but not Guaranteed), **BestEffort** (no
  requests/limits). The **eviction manager** watches node-level signals (`memory.available`,
  `nodefs.available`, `imagefs.available`, `pid.available`) against **soft/hard eviction thresholds**;
  under pressure it **evicts pods** (deletes them so they reschedule elsewhere) in QoS+priority+usage
  order — BestEffort first, then Burstable over-requests, Guaranteed last. **Eviction ≠ OOMKill**:
  eviction is the kubelet proactively reclaiming node resources; OOMKill is the kernel killing a
  cgroup that hit its memory limit. Confusing the two misdiagnoses incidents.
- **Device Plugin API.** Vendors (NVIDIA, etc.) run a device-plugin DaemonSet that registers with the
  kubelet over a Unix socket and advertises **extended resources** (e.g. `nvidia.com/gpu`). Pods
  request whole devices as countable resources; the plugin returns the device handles/mounts/env on
  allocation. Limitations: whole-device granularity, no rich constraints, awkward sharing.
- **DRA (Dynamic Resource Allocation)** is the modern accelerator path that addresses those limits:
  resources are modeled as `ResourceClaim`/`ResourceClaimTemplate`/`DeviceClass`/`ResourceSlice`
  objects, allocated by a vendor **DRA driver** with structured parameters (partitioning, sharing,
  topology). The scheduler participates in allocation. DRA has been advancing through the gates across
  recent releases — **its API shape and GA status change; verify against the current KEPs (sig-node /
  DRA, KEP-4381 and related) and release notes for your version** before relying on specifics. For
  GPU/TPU accelerator workloads end-to-end see [[aiml-on-kubernetes]] and [[gke-master]].
- **Volume manager / CSI on the node.** The **attach/detach** controller (control plane) attaches a
  volume to the node (`VolumeAttachment`); the kubelet's volume manager then calls the node-local
  **CSI driver** (`NodeStageVolume` → `NodePublishVolume`) to mount it into the pod. Stuck mounts,
  `Multi-Attach` errors (a volume still attached to a dead node), and CSI driver crashes are common
  "Pod stuck in `ContainerCreating`" causes — check the kubelet log and the CSI node-plugin pod.
- **Static pods.** Pods defined by **files** in `--pod-manifest-path` (e.g. `/etc/kubernetes/manifests`),
  run directly by the kubelet with **no scheduler or apiserver involvement** — this is how kubeadm
  bootstraps the control plane itself (apiserver/etcd/controller-manager/scheduler are static pods).
  The kubelet creates a read-only **mirror pod** in the apiserver so they're visible to `kubectl`, but
  you cannot delete them via the API — you edit/remove the manifest file on the node.

---

## 6. Networking dataplane: kube-proxy, Services, EndpointSlices, CNI, conntrack

- **The CNI contract.** The kubelet (via the runtime) calls a **CNI plugin** to set up the pod's
  network namespace: allocate an IP (IPAM), wire a veth into the pod, and program routes so every pod
  gets a routable IP and pods can reach each other **without NAT** (the flat pod network is a core
  K8s assumption). CNI is `ADD`/`DEL`/`CHECK` on a netns; the chosen implementation (Calico, Cilium,
  flannel, cloud CNIs) decides overlay vs native routing, NetworkPolicy enforcement, and eBPF vs
  iptables. A failed CNI `ADD` → Pod stuck `ContainerCreating` with a sandbox/network error.
- **Service VIP implementation.** A `ClusterIP` Service has a **virtual IP** that exists nowhere as a
  real interface — it's a load-balancing rule. **kube-proxy** watches Services + **EndpointSlices** and
  programs the node's dataplane to DNAT traffic destined for the VIP to one of the backend pod IPs.
  - **iptables mode** (long-time default): a chain of rules per Service with probabilistic
    (`statistic random`) selection across endpoints. Simple and ubiquitous, but **rule count scales
    O(Services × endpoints)** and large clusters see slow rule reprogramming and latency.
  - **IPVS mode**: uses the kernel **IPVS** load balancer (hash tables) — scales far better for many
    Services/endpoints, supports real LB algorithms (rr, lc, …). Still uses iptables for some
    auxiliary rules.
  - **nftables mode**: the modern backend replacing iptables, with much better update/lookup
    performance at scale. **Verify its GA/default status for your release** — it has been graduating
    and is the strategic direction as the kernel deprecates legacy iptables. (Separately, eBPF
    dataplanes like Cilium can **replace kube-proxy entirely**.)
- **EndpointSlices** replaced the monolithic `Endpoints` object: backends are sharded into slices
  (default cap ~100 endpoints each) so a single large Service doesn't produce one giant object that
  every node rewatches on every pod change — critical for scale. Each slice carries addresses,
  ports, topology hints, and readiness/terminating conditions. `Endpoints` still exists for
  compatibility but EndpointSlice is the source of truth.
- **conntrack pitfalls.** DNAT relies on the kernel **conntrack** table to remember the
  VIP→pod mapping for a flow. Pitfalls: the table **fills up** (`nf_conntrack: table full, dropping
  packet` — tune `nf_conntrack_max`); **stale entries to a deleted pod** cause connections to a VIP to
  hit a dead backend until the entry expires (long-lived idle TCP/UDP especially); **UDP** has no
  connection teardown so stale UDP conntrack entries (classic with DNS) black-hole traffic — many
  setups add rules to flush conntrack on endpoint change. `terminating` endpoint conditions and graceful
  termination exist to drain flows before a pod's rules are removed.

---

## 7. Cross-cutting mechanisms

### Finalizers & deletion at the apiserver level
A `DELETE` doesn't immediately remove an object that has `metadata.finalizers`. The apiserver sets
**`metadata.deletionTimestamp`** and the object enters **`Terminating`**, remaining in etcd until
**every finalizer string is removed**. Controllers owning a finalizer do their cleanup (drain
external resources, detach volumes, decrement quota) and then **patch off their finalizer**; when the
list is empty the apiserver hard-deletes. **The #1 "stuck Terminating" cause** is a finalizer whose
controller is gone/broken — the object can never be reaped. Force-removing a finalizer (`kubectl
patch ... -p '{"metadata":{"finalizers":null}}'` or `--force --grace-period=0`) **skips the cleanup
the finalizer existed to do** — understand the consequence (leaked cloud LB, orphaned volume) before
doing it. Namespaces are the canonical case: a `Terminating` namespace is usually blocked on a
finalizer for some API in it (often a broken aggregated API / `metrics.k8s.io`).

### Field/label selectors and indexes
- **Label selectors** filter on `metadata.labels` and are general-purpose. **Field selectors** filter
  on a *small fixed set* of fields the apiserver indexes for that resource (e.g. Pod's
  `spec.nodeName`, `status.phase`; not arbitrary fields). The kubelet's per-node Pod watch uses
  `spec.nodeName` field selector so it only watches *its* pods — this index is load-bearing for
  scale.
- Informer-side, client-go `Indexer`s let controllers build their own in-memory indexes (e.g. pods by
  node) for O(1) lookup instead of scanning the cache.

### Server-side apply (SSA) & managedFields
**SSA** (`PATCH` with `Content-Type: application/apply-patch+yaml`) makes the apiserver track **field
ownership**: each field is recorded in **`metadata.managedFields`** against the **field manager** that
set it. Multiple actors (a controller, a human, an operator) can each own different fields of the same
object; a manager updating a field it doesn't own causes a **conflict** (forceable with
`?force=true`). This replaces fragile client-side strategic-merge `kubectl apply` 3-way merges with
server-authoritative merging — essential when a controller and a GitOps tool both touch one object.
Gotchas: `managedFields` **bloats objects** (etcd cost — relevant to §2) and is noisy in `get -o yaml`
(`kubectl ... --show-managed-fields=false` to hide); list-typed fields need correct merge keys
(`listType`/`listMapKey` in the schema) or owners collide.

### How level-triggered reconciliation falls out of watch + resync
Tying §0 to the machinery: an informer maintains a cache via an initial **LIST** (snapshot at an RV)
then a **WATCH** from that RV (incremental). Periodic **resync** re-delivers cache state so a logic
bug or missed event can't permanently desync a controller. Because the reconciler always reads
*current* cache state (not the triggering event), duplicate or out-of-order events are harmless — the
loop is **idempotent and convergent**. Bookmarks/`410` handling (§1) keep the watch cheap and correct
across reconnects. This is why "edge-triggered" thinking ("I'll act on the create event") is the
classic controller bug: act on **state**, not **events**.

---

## 8. Debugging internals: logs, metrics, raw API, etcd

- **Raw API access.** `kubectl get --raw /healthz`, `/livez?verbose`, `/readyz?verbose` (per-check
  health of the apiserver), `/metrics` (Prometheus), `/version`, `/openapi/v3`, and discovery
  (`/apis`). `kubectl get --raw '/api/v1/nodes?limit=1'` to test storage paths.
  `--profile`/`/debug/pprof` (gated) for CPU/heap profiles of the apiserver.
- **Key apiserver metrics:** `apiserver_request_duration_seconds` (latency by verb/resource/code — your
  first stop for "the API is slow"), `apiserver_request_total` (watch the `code` label for `429`/`5xx`),
  `etcd_request_duration_seconds` (apiserver→etcd latency), `apiserver_storage_objects` (object counts
  per resource — find the resource that's exploding), `apiserver_flowcontrol_*` (APF rejects/queueing),
  `apiserver_watch_cache_*`, `apiserver_current_inflight_requests`.
- **etcd metrics:** `etcd_disk_wal_fsync_duration_seconds` and
  `etcd_disk_backend_commit_duration_seconds` (disk — the usual culprit),
  `etcd_server_leader_changes_seen_total` (leader churn = instability),
  `etcd_mvcc_db_total_size_in_bytes` vs `..._in_use_bytes` (gap = needs **defrag**),
  `etcd_server_proposals_failed_total`, `etcd_network_peer_round_trip_time_seconds`.
  Inspect directly with `etcdctl endpoint status -w table`, `endpoint health`,
  `etcdctl get --prefix --keys-only /registry/<resource>` (find big/numerous keys), `etcdctl
  alarm list` / `alarm disarm`, `etcdctl compact <rev>`, `etcdctl defrag`. Always use the right
  certs/endpoints; **never** write to etcd directly under the apiserver.
- **Scheduler:** `scheduler_pending_pods`, `scheduler_schedule_attempts_total{result}`,
  `scheduler_pod_scheduling_duration_seconds`, `scheduler_preemption_*`. `-v=4..10` logs show
  per-plugin filter/score decisions for a Pod (why a node was rejected). Pod `Events`
  (`FailedScheduling`) name the failing predicates ("0/12 nodes available: 3 Insufficient cpu, …").
- **kubelet:** `/metrics`, `/metrics/cadvisor` (container resource usage), `/metrics/resource`,
  `/stats/summary` (node/pod stats incl. eviction signals), `/healthz`. `crictl ps`/`inspect`/`logs`/
  `images` for the runtime's view. `journalctl -u kubelet` for syncLoop/PLEG/CSI/eviction errors.
  `kubectl get --raw /api/v1/nodes/<node>/proxy/metrics/...` to reach a node's kubelet through the API.
- **Common control-plane failure modes → first move:**
  - *Whole API slow/timeouts* → apiserver request-duration & inflight metrics; check **etcd fsync**
    latency and **APF** rejections; check a flooding client (object counts, `apiserver_request_total`
    by user-agent).
  - *`kubectl` discovery slow / `metrics` errors* → a **down aggregated APIService** (§1).
  - *Writes failing cluster-wide* → etcd **quota/NOSPACE alarm** or **lost quorum** (§2); or a
    `failurePolicy: Fail` **admission webhook** down (§1).
  - *Pods stuck Pending* → scheduler events + which **queue** (§3); resources/taints/affinity/PVC
    binding; quota.
  - *Pods stuck Terminating* → a **finalizer** with a dead controller (§7).
  - *Pods stuck ContainerCreating* → **CNI** failure or **CSI** mount/attach failure on the node (§5/§6).
  - *Node NotReady* → kubelet→apiserver **heartbeat/Lease** failure (§4), or kubelet/runtime down
    (`PLEG` unhealthy, §5).
  - *Intermittent Service connection failures* → **conntrack** staleness or kube-proxy not
    reprogramming (§6).

---

## 9. Version awareness (it is 2026 — verify, don't trust memory)

Kubernetes ships ~3 minor releases/year and graduates features through **Alpha → Beta → GA** via
**feature gates** and **KEPs**. The following are *actively evolving* — confirm GA status, default
on/off, and exact API shape against the **current release notes, the KEP, and source** for the
cluster's version before you rely on specifics:

- **DRA (Dynamic Resource Allocation)** — API objects and allocation model have changed across
  releases; the structured-parameters model is the direction. (sig-node DRA KEPs.)
- **nftables kube-proxy backend** — graduating to replace iptables; check default.
- **ValidatingAdmissionPolicy / CEL admission** and the **mutating-policy (CEL-based admission
  mutation)** line — CEL in admission is expanding; verify which policy kinds are GA.
- **APF** is GA but priority levels/defaults and **consistent reads from cache** keep tuning.
- **Evented PLEG**, **in-place pod vertical scaling** (resize requests/limits without restart),
  **sidecar containers** (restartable init), **user-namespaces**, and **cgroup v2**-only assumptions —
  all have moved recently; verify.

When you don't know the current status, **say so and tell the reader to check the KEP/release notes**
rather than asserting a version. A confidently wrong feature-gate claim is worse than "verify this."

---

## 10. Canonical references (real URLs only)

- Kubernetes source — https://github.com/kubernetes/kubernetes (apiserver/etcd storage:
  `staging/src/k8s.io/apiserver`; apimachinery: `staging/src/k8s.io/apimachinery`; client-go informers:
  `staging/src/k8s.io/client-go/tools/cache`; scheduler: `pkg/scheduler`; kubelet: `pkg/kubelet`;
  kube-proxy: `pkg/proxy`).
- KEPs — https://github.com/kubernetes/enhancements (the design-of-record for every feature; the
  fastest way to learn a mechanism precisely).
- API conventions — https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md
- Admission controllers reference — https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/
- Validating Admission Policy / CEL — https://kubernetes.io/docs/reference/access-authn-authz/validating-admission-policy/
- API Priority and Fairness — https://kubernetes.io/docs/concepts/cluster-administration/flow-control/
- Scheduling Framework — https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/
- Scheduler config / plugins — https://kubernetes.io/docs/reference/scheduling/config/
- Garbage collection — https://kubernetes.io/docs/concepts/architecture/garbage-collection/
- Node heartbeats / leases — https://kubernetes.io/docs/concepts/architecture/nodes/
- Container Runtime Interface — https://kubernetes.io/docs/concepts/architecture/cri/
- Dynamic Resource Allocation — https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/
- Device plugins — https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/
- Node eviction (kubelet) — https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/
- kube-proxy / Service proxy — https://kubernetes.io/docs/reference/networking/virtual-ips/
- EndpointSlices — https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/
- Server-Side Apply — https://kubernetes.io/docs/reference/using-api/server-side-apply/
- Finalizers — https://kubernetes.io/docs/concepts/overview/working-with-objects/finalizers/
- etcd docs (Raft, MVCC, maintenance/defrag, tuning) — https://etcd.io/docs/
- Operating etcd for Kubernetes — https://kubernetes.io/docs/tasks/administer-cluster/configure-upgrade-etcd/
- Kubernetes release notes / changelog — https://github.com/kubernetes/kubernetes/tree/master/CHANGELOG
