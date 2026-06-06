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
