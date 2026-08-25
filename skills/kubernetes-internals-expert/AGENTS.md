# AGENTS.md — Kubernetes Internals

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`kubernetes-internals-expert-guide.md`** next to this
> file — read it before reasoning about apiserver/etcd/scheduler/kubelet/dataplane internals. This is
> the always-on summary.
>
> **Bar: a SIG contributor who reads the source.** Reason about how the mechanism *actually* works,
> localize a failure to the owning component and its loop, then confirm with the right metric/log/raw
> query. **It is 2026 — never fabricate a feature-gate status, API field, flag, or version number.**
> DRA, the nftables kube-proxy backend, CEL admission policies, and APF are all evolving; if unsure,
> say so and point to the KEP / release notes / source.

## When debugging or explaining Kubernetes internals, apply these by default:

- **Level-triggered reconciliation over a consistent store** is the whole model. Spec = desired,
  status = observed; independent loops converge; eventual consistency, no cross-object transactions.
  Watches+resync are an optimization — controllers act on **current state**, not the triggering event,
  so reconciles are idempotent. "Why didn't X happen" → find the owning controller, its inputs, and
  whether its loop ran.
- **Apiserver write path (fixed order):** auth → authz → mutating admission → schema validation →
  validating admission → CEL policy (`ValidatingAdmissionPolicy`) → etcd write. Reads skip admission,
  served from the **watch cache**. A `failurePolicy: Fail` webhook that's down blocks all matching
  writes — exclude kube-system/its own ns, scope selectors, low timeout.
- **API machinery:** GVK (type) vs GVR (REST path) via RESTMapper; the **scheme** registers
  types/conversion/defaulting; one **internal hub version** + per-version converters; protobuf wire
  for built-ins, JSON for CRDs; defaulting on decode, conversion on version mismatch; one **storage
  version** per resource.
- **resourceVersion is an opaque etcd-revision token** (no arithmetic). `410 Gone` → relist;
  **bookmarks** advance RV cheaply for resumable watches; LIST paginates with `limit`/`continue`;
  `rv=0` = cheap possibly-stale cache read.
- **APF** (FlowSchema → PriorityLevelConfiguration) isolates noisy clients into queues; `429` + PF
  headers name the flow. A **down aggregated APIService** (e.g. `metrics.k8s.io`) makes discovery and
  `kubectl get` feel broken cluster-wide. CRDs/built-ins don't go through the aggregator.
- **etcd = Raft + MVCC.** Odd member count (3/5) for quorum; lost quorum → control plane stalls.
  **Compaction** drops revision history; **defrag** reclaims disk (etcd never shrinks itself); DB
  quota exceeded → **NOSPACE alarm**, writes refused until compact+defrag+disarm. Large objects, fat
  `managedFields`, and hot status-write loops hurt via Raft fsync — watch
  `etcd_disk_wal_fsync_duration_seconds`, `..._backend_commit_...`, leader changes.
- **Scheduler framework:** PreFilter/**Filter** = hard feasibility; PreScore/**Score** = soft ranking;
  Reserve/**Permit** (gang/coscheduling hook); PreBind/**Bind** writes `nodeName` (kubelet then runs
  it). **Preemption** is in PostFilter and sets `status.nominatedNodeName`. Stuck-Pending → events +
  which queue (active/backoff/unschedulable). Gang/batch quota lives above the default scheduler
  (Kueue/Volcano → `[[kueue-advanced]]`).
- **Controller-manager = shared informers + workqueues.** One watch/cache per type shared by all
  controllers; events enqueue keys, workers read current state; dedupe + rate-limited retry + resync.
  **GC** uses **owner references**; delete policy = foreground (finalizer-blocked) / background /
  orphan. Node liveness via **Leases** in `kube-node-lease`; HA via **lease-based leader election**
  (cooperative, not fencing — exit promptly on lost leadership, stay idempotent).
- **kubelet:** syncLoop + **PLEG** (relist diff; "PLEG not healthy" = wedged/overloaded runtime or
  disk) drive the **CRI** (containerd/CRI-O; dockershim removed in 1.24). cgroup v2 enforces:
  CPU limit = CFS throttle, memory limit = hard wall → **OOMKill (exit 137)**. **OOMKill ≠ eviction**
  — eviction is the kubelet reclaiming node pressure (memory/disk/pid signals), QoS+priority ordered
  (BestEffort first). **Device plugins** advertise GPUs as countable extended resources; **DRA** is
  the modern structured-parameter accelerator path (verify GA/API shape). CSI node mount failures and
  **static pods** (control-plane bootstrap, mirror pods, no scheduler) are common knowledge gaps.
- **Dataplane:** Service ClusterIP is a virtual IP **kube-proxy** programs (iptables O(n) vs IPVS
  hash vs **nftables** — verify default; eBPF/Cilium can replace kube-proxy) from **EndpointSlices**
  (sharded, replacing monolithic Endpoints). **CNI** gives every pod a NAT-free routable IP. **conntrack**
  staleness (esp. UDP/DNS) and table-full cause intermittent Service failures.
- **Cross-cutting:** **finalizers** hold an object `Terminating` until all clear — #1 stuck-Terminating
  cause is a dead finalizer controller; force-removing skips its cleanup (leaked LB/volume).
  **Field selectors** index a fixed small set (e.g. Pod `spec.nodeName` — load-bearing for the
  per-node kubelet watch). **Server-side apply** tracks field ownership in `managedFields` (conflicts
  on contested fields; bloats objects → etcd cost).

## First moves by symptom
- API slow/timeouts → `apiserver_request_duration_seconds`, inflight, **etcd fsync**, **APF** rejects,
  flooding client (object counts / user-agent).
- `kubectl`/discovery slow, metrics errors → **down aggregated APIService**.
- Cluster-wide write failures → etcd **NOSPACE/lost quorum**, or a `Fail` **admission webhook** down.
- Pending → scheduler events + queue. Terminating → **finalizer**. ContainerCreating → **CNI/CSI**.
  NotReady → kubelet **Lease/heartbeat** or runtime/**PLEG**. Intermittent Service drops → **conntrack**.

## Debug tooling
`kubectl get --raw /readyz?verbose|/metrics|/livez?verbose`; per-component Prometheus metrics (§8 of
guide); `etcdctl endpoint status -w table` / `alarm list` / `compact` / `defrag` /
`get --prefix --keys-only /registry/...`; `crictl ps|inspect|logs`; `journalctl -u kubelet`. Never
write to etcd directly under the apiserver. Confirm version-gated behavior against current docs/KEPs.
