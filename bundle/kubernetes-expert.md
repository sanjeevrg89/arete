---
name: kubernetes-expert
description: End-to-end Kubernetes practitioner mastery for using K8s well in production — authoring,
  reviewing, debugging, and operating workloads on real clusters. Use when working with Pods,
  Deployments, StatefulSets, DaemonSets, Jobs/CronJobs, Services, Ingress/Gateway API, NetworkPolicy,
  PV/PVC/StorageClass/CSI, ConfigMaps/Secrets, RBAC/ServiceAccounts, Pod Security Admission,
  securityContext, ResourceQuota/LimitRange, HPA targets, probes, PodDisruptionBudgets, taints/
  tolerations, affinity, topology spread; writing or reviewing K8s manifests/Helm/Kustomize; or
  debugging CrashLoopBackOff, ImagePullBackOff, Pending, OOMKilled, evictions, and rollout failures
  with kubectl. Covers requests/limits, QoS, scheduling, zero-downtime rollouts, graceful termination,
  multi-tenancy, and a production-readiness checklist. NOT for control-plane source internals or
  writing controllers/operators (see related skills).
---

# Kubernetes Expert (using Kubernetes well)

Apply the judgment of an engineer who has run large multi-tenant Kubernetes clusters in production for
~10 years. Kubernetes is a **declarative, level-triggered reconciliation engine**: you describe desired
state, controllers continuously drive actual state toward it. Almost every good decision and every
debugging session flows from that one idea.

## How to use this skill

1. **Read `kubernetes-expert-guide.md`** in this directory — the full reference (mental model,
   workloads, scheduling, networking, storage, config/secrets, security, multi-tenancy, ops,
   troubleshooting, anti-patterns, readiness checklist). Apply it to the task.
2. For canonical correct manifests to imitate (production Deployment, StatefulSet, NetworkPolicy),
   read **`examples.md`**.
3. Match the cluster's existing conventions (labels, namespaces, ingress class, CNI, storage classes,
   GitOps tooling). Apply the correctness and safety rules regardless of local convention.

## The essentials (full detail in `kubernetes-expert-guide.md`)

- **Declarative, level-triggered.** Apply manifests; never imperatively mutate live objects in prod
  (`kubectl edit`/`patch` outside of break-glass). The manifest in Git is the source of truth.
- **Never run naked Pods.** Use a controller (Deployment / StatefulSet / DaemonSet / Job). A bare Pod
  isn't rescheduled when its node dies.
- **Always set resource `requests` (and usually `limits`).** Requests drive scheduling and QoS; no
  request = `BestEffort` = first evicted. Set memory `limit == request` (memory is incompressible);
  be deliberate about CPU limits (throttling vs. burst).
- **Every long-running workload needs a `readinessProbe`; add `liveness`/`startup` only with care.** A
  bad liveness probe causes self-inflicted CrashLoopBackOff. Readiness gates traffic; liveness restarts.
- **Zero-downtime = readiness + PodDisruptionBudget + `preStop`/grace + RollingUpdate `maxUnavailable`.**
  Handle SIGTERM and the de-registration race; the guide has the exact pattern.
- **`latest` tag is banned.** Pin immutable tags or digests; set `imagePullPolicy` accordingly. Mutable
  tags break rollbacks and reproducibility.
- **Secrets are base64, not encrypted by default.** Don't commit them; prefer file mounts over env;
  enable encryption-at-rest / external secret stores; lock down RBAC on `secrets`.
- **Least-privilege RBAC and ServiceAccounts.** No `cluster-admin` for apps; never use the `default` SA
  with `automountServiceAccountToken: true` unless the pod actually calls the API.
- **Pod Security Admission `restricted`** as the default namespace baseline: non-root, drop ALL caps,
  `readOnlyRootFilesystem`, `seccompProfile: RuntimeDefault`, no privilege escalation.
- **Spread for availability:** `topologySpreadConstraints` across zones/nodes + anti-affinity + PDB.
  One replica or all replicas on one node/zone is not HA.
- **Namespaces are the tenancy boundary:** `ResourceQuota` + `LimitRange` + `NetworkPolicy` (default
  deny) per tenant. Without a default-deny NetworkPolicy, all pods can talk to all pods.
- **Debug in order:** `kubectl get` → `describe` (Events!) → `logs`/`--previous` → `events` →
  `debug` ephemeral container. Learn the failure signatures: CrashLoopBackOff, ImagePullBackOff,
  Pending, OOMKilled, evicted.
- **Label everything** with the `app.kubernetes.io/*` recommended labels; manifests should be
  GitOps-friendly (no live mutation, no generated names you can't reconcile).

## Related skills

- `[[kubernetes-internals-expert]]` — how the apiserver/etcd/scheduler/kubelet/kube-proxy work under
  the hood when you need to reason past the API surface.
- `[[kubernetes-controller-expert]]` / `[[kubernetes-operator-expert]]` — when you're *extending*
  Kubernetes (controllers, CRDs, webhooks) rather than using it.
- `[[autoscaling-kubernetes]]` — HPA/VPA/Cluster Autoscaler/Karpenter/KEDA when you need to scale.
- `[[gke-master]]` — GKE-specific (Autopilot, node pools, Workload Identity, networking).
- `[[aiml-on-kubernetes]]` — running training/inference (GPU/TPU, gang scheduling) on K8s.

---

# Reference — kubernetes-expert

# Kubernetes Practitioner Guidelines

Production-grade guidance for **using Kubernetes well** — authoring workloads, operating multi-tenant
clusters, and debugging them — from the perspective of an engineer who has done this at scale for
~10 years. This is the single source of truth; `SKILL.md` and `AGENTS.md` defer to it.

> Scope: using Kubernetes. NOT control-plane source internals (`[[kubernetes-internals-expert]]`), NOT
> writing controllers/operators (`[[kubernetes-controller-expert]]`, `[[kubernetes-operator-expert]]`).
> Kubernetes moves fast: an API graduates or a default flips every release. Where a field, default, or
> version matters, **verify against the docs for your cluster's exact version** (`kubectl version`,
> `kubectl explain <kind>.spec...`, `kubectl api-resources`). Do not assume a beta API is enabled.

---

## 0. The one mental model

Kubernetes is a **declarative, level-triggered reconciliation system**, not a script runner.

- **Declarative:** you submit *desired state* (a manifest). You don't tell K8s *how* to get there.
- **Level-triggered, not edge-triggered:** controllers continuously observe *actual state*, compare it
  to *desired state*, and act to close the gap — a reconcile loop. Missing an event is fine; the next
  reconcile re-reads the whole world. This is why K8s is self-healing and why "I deleted the pod and it
  came back" is correct behavior, not a bug.
- **Control plane vs. nodes.** The **control plane** holds desired+actual state and runs controllers:
  - **kube-apiserver** — the only thing that talks to etcd; the front door for all reads/writes.
  - **etcd** — the consistent key-value store of record (all cluster state).
  - **kube-controller-manager** — the built-in controllers (Deployment→ReplicaSet→Pod, Node, Job, …).
  - **kube-scheduler** — assigns Pending pods to nodes based on requests, affinity, taints, spread.
  - **cloud-controller-manager** — cloud integration (LoadBalancers, node lifecycle, routes).
  - **Nodes** run **kubelet** (turns pod specs into running containers via the CRI, reports status),
    **kube-proxy** or an eBPF dataplane (Service VIP routing), and a **CNI** plugin (pod networking).
- **Everything is an object** addressed by `apiVersion`/`kind`/`namespace`/`name`, with `spec` (desired)
  and `status` (actual, written by controllers). You own `spec`; controllers own `status`.
- **Labels and selectors are the glue.** Controllers and Services find their members by **label
  selector**, not by name. Getting labels/selectors right is half of using K8s correctly.

Consequence for you: write the desired state, let controllers converge, and **debug by comparing
desired vs. actual** (`kubectl describe`, Events, `status`). Don't fight the reconciler with imperative
commands.

---

## 1. Workloads — pick the right controller

| Controller | Use it for | Identity | Update strategy |
|---|---|---|---|
| **Deployment** | Stateless services (web, API, workers) | Interchangeable pods | RollingUpdate / Recreate |
| **StatefulSet** | Stable identity/storage (DBs, queues, quorum) | Stable name + PVC per pod | RollingUpdate (ordered) / OnDelete |
| **DaemonSet** | One pod per (matching) node (log/metric agents, CNI, CSI) | Per-node | RollingUpdate / OnDelete |
| **Job** | Run-to-completion batch | Ephemeral | n/a (completions/parallelism) |
| **CronJob** | Scheduled Jobs | Ephemeral | schedule + concurrencyPolicy |
| **ReplicaSet** | Low-level pod count; you rarely create directly | Interchangeable | (managed by Deployment) |

- **Never create naked Pods or bare ReplicaSets** in prod. A naked Pod is not rescheduled if its node
  dies; a Deployment owns a ReplicaSet that maintains the desired replica count.
- **Deployment update strategies:**
  - `RollingUpdate` (default): `maxUnavailable` and `maxSurge` control the rollout. For HA, keep
    `maxUnavailable: 0` (or a small number) and rely on `maxSurge` to add new pods before removing old.
  - `Recreate`: kills all old pods before creating new — causes downtime; use only when two versions
    cannot coexist (e.g., exclusive lock, incompatible schema mid-migration).
  - **A rollout only progresses as pods become Ready** — so your readiness probe *is* your rollout
    safety. `kubectl rollout status deploy/x` waits; `kubectl rollout undo deploy/x` rolls back to the
    previous ReplicaSet (kept per `revisionHistoryLimit`).
- **StatefulSet** gives each pod a stable ordinal name (`web-0`, `web-1`), stable network identity via a
  **headless Service** (`clusterIP: None`), and a dedicated PVC via **`volumeClaimTemplates`** that
  survives pod rescheduling. Updates and scale-down are **ordered** (highest ordinal first). Use it only
  when you need that identity/storage — it is heavier and slower than a Deployment. Deleting a
  StatefulSet does **not** delete its PVCs by default (data safety); `persistentVolumeClaimRetentionPolicy`
  tunes this.
- **DaemonSet** schedules one pod per node matching its `nodeSelector`/affinity/tolerations. Tolerate the
  taints you need (e.g., control-plane, not-ready) for true cluster-wide agents.
- **Job:** set `completions`, `parallelism`, `backoffLimit` (retry cap), and `activeDeadlineSeconds`
  (wall-clock cap). Use the **Indexed** completion mode for partitioned work (each pod gets
  `JOB_COMPLETION_INDEX`). Always set `ttlSecondsAfterFinished` so finished Jobs are garbage-collected.
  For batch/quota/gang scheduling, reach for `[[kueue-advanced]]` and `[[jobset-leaderworkerset]]`.
- **CronJob:** set `concurrencyPolicy` (`Forbid` is usually what you want), `startingDeadlineSeconds`,
  and history limits (`successfulJobsHistoryLimit`/`failedJobsHistoryLimit`). Schedules run in the
  cluster's timezone unless you set `spec.timeZone`.

### Probes — readiness, liveness, startup

Three distinct jobs; conflating them is a top-3 outage cause.
- **`readinessProbe`** — "should this pod receive traffic?" Failure removes the pod from Service
  endpoints but does **not** restart it. **Almost every serving workload should have one.**
- **`livenessProbe`** — "is this pod wedged and needs a restart?" Failure **kills and restarts** the
  container. Use sparingly and make it cheap and reliable. A liveness probe that checks a dependency
  (DB) turns a downstream blip into a cluster-wide restart storm and **self-inflicted CrashLoopBackOff**.
- **`startupProbe`** — protects slow-starting apps: liveness/readiness are suspended until startup
  succeeds, so you can give a long boot a generous `failureThreshold * periodSeconds` without weakening
  the steady-state liveness check.
- Probe types: `httpGet`, `tcpSocket`, `grpc` (GA), `exec` (most expensive — forks a process each time).
- Tune `initialDelaySeconds`/`periodSeconds`/`timeoutSeconds`/`failureThreshold`/`successThreshold`.
  Default liveness to *generous* thresholds; readiness can be tighter.

### Init and sidecar containers

- **`initContainers`** run to completion, in order, before app containers start. Use for setup that must
  finish first (schema migration gate, fetching config, waiting on a dependency).
- **Native sidecars** (beta, enabled by default since 1.29 — verify for your version): a sidecar is an
  `initContainer` with `restartPolicy: Always`. It
  **starts before** the main containers, **stays running** alongside them, and importantly **does not
  block Job completion** and is terminated after the main containers on shutdown. This fixes the old
  "sidecar keeps the Job pod alive forever" and "proxy dies before app drains" problems. Prefer native
  sidecars over the legacy plain-container sidecar pattern for proxies/log shippers/agents.

---

## 2. Scheduling & placement

The scheduler places a Pending pod on a feasible node with the best score. You influence it with:

### Requests, limits, and QoS

- **`requests`** = what the scheduler reserves; sum of requests on a node can't exceed allocatable.
  **`limits`** = hard cap the kubelet/runtime enforces.
- **CPU is compressible** (throttled when over limit). **Memory is incompressible** (over limit →
  **OOMKilled**). Therefore: **set `memory request == memory limit`**. For CPU, set a request always;
  a CPU *limit* causes CFS throttling and can hurt tail latency — many shops set CPU requests only and
  omit CPU limits for latency-sensitive services (with a node-level safeguard). Be deliberate.
- **QoS classes** (derived, not set directly):
  - **Guaranteed** — every container has `requests == limits` for both CPU and memory. Last to be
    evicted; eligible for CPU pinning. Use for latency-critical / stateful workloads.
  - **Burstable** — has requests but not the Guaranteed condition. The common case.
  - **BestEffort** — no requests/limits at all. **First to be evicted; first OOM-killed.** Never run
    anything you care about as BestEffort.
- Right-size with real data (`kubectl top`, metrics, VPA in recommend mode — see
  `[[autoscaling-kubernetes]]`). Over-requesting wastes money; under-requesting causes evictions/OOM.

### Placement primitives

- **`nodeSelector`** — simplest: schedule only on nodes with matching labels.
- **Node affinity** — `requiredDuringScheduling...` (hard) and `preferredDuringScheduling...` (soft)
  rules over node labels; richer than nodeSelector.
- **Pod affinity / anti-affinity** — co-locate or spread relative to *other pods* by label, scoped by
  `topologyKey` (e.g., `kubernetes.io/hostname` for per-node, `topology.kubernetes.io/zone` for
  per-zone). **Prefer `topologySpreadConstraints`** for spreading — it's cheaper and more expressive
  than `preferred` anti-affinity, which is O(pods²) and slows scheduling on big clusters.
- **`topologySpreadConstraints`** — the modern way to spread replicas evenly across zones/nodes. Set
  `maxSkew`, `topologyKey`, `whenUnsatisfiable` (`DoNotSchedule` hard vs `ScheduleAnyway` soft), and a
  `labelSelector`. Spread across `topology.kubernetes.io/zone` *and* `kubernetes.io/hostname` for real HA.
- **Taints & tolerations** — a **taint** on a node repels pods; a **toleration** on a pod lets it land
  there anyway. Use for dedicated node pools (GPU, special hardware), control-plane isolation, and
  node-condition taints (`NoSchedule`/`PreferNoSchedule`/`NoExecute`). Tolerations don't *attract* —
  pair with node affinity/`nodeSelector` to actually target the tainted pool.
- **PriorityClass & preemption** — higher-priority pods can **preempt** (evict) lower-priority pods to
  schedule. Define a few classes (e.g., `system-critical`, `high`, `default`, `best-effort`). Use
  `preemptionPolicy: Never` for jobs that should wait rather than evict others.

---

## 3. Networking

- **Pod networking (CNI):** every pod gets its own routable IP in a flat network; pods reach each other
  by IP without NAT. The **CNI plugin** (Calico, Cilium, AWS VPC CNI, etc.) implements this. Cilium
  (eBPF) is the common modern choice and can replace kube-proxy. You usually consume, not configure, the
  CNI — but it determines NetworkPolicy support and performance.
- **Service types:**
  - **ClusterIP** (default) — stable virtual IP + DNS name, load-balanced across endpoints; in-cluster
    only.
  - **NodePort** — exposes the Service on a port on every node; rarely used directly in prod.
  - **LoadBalancer** — provisions a cloud L4 LB; one per Service can get expensive (front with Ingress).
  - **ExternalName** — DNS CNAME to an external host, no proxying.
  - **Headless** (`clusterIP: None`) — no VIP; DNS returns pod IPs directly. Required for StatefulSets
    and client-side load balancing / discovery.
- **EndpointSlices** are the scalable backing for Service membership (replaced the monolithic
  `Endpoints` object). The Service controller and kube-proxy/eBPF consume them; you rarely touch them
  directly, but they're what `kubectl get endpointslices` shows for "who is this Service actually
  routing to" — the first thing to check when a Service returns nothing (usually a selector mismatch).
- **DNS (CoreDNS):** services resolve as `name.namespace.svc.cluster.local`. Pods get a `ndots:5`
  resolv.conf by default, which means **unqualified external lookups try several search-domain
  suffixes first** — a classic latency/error source. Use **FQDNs with a trailing dot** for external
  hosts (`api.example.com.`) or tune `dnsConfig` to avoid the search-path tax. CoreDNS itself can become
  a bottleneck at scale — enable NodeLocal DNSCache.
- **Ingress vs. Gateway API:**
  - **Ingress** — the older L7 HTTP routing API; needs an ingress controller (nginx, etc.). Limited
    expressiveness; vendors pile features into annotations (non-portable).
  - **Gateway API** (GA) — the **successor**: role-oriented (`GatewayClass`/`Gateway`/`HTTPRoute`/
    `GRPCRoute`/`TLSRoute`), portable, supports header/weight routing, traffic splitting, and cross-
    namespace delegation. **Prefer Gateway API for new clusters**; it's the direction the ecosystem is
    moving. Ingress is in maintenance mode.
- **NetworkPolicy:** namespaced firewall rules over pods by label, for ingress and egress.
  **Default is allow-all** — until you apply a policy that selects a pod, it accepts all traffic.
  **Establish a default-deny per namespace**, then explicitly allow needed flows (see `examples.md`).
  NetworkPolicy is only enforced if your CNI supports it. For richer L7/identity policy, Cilium
  `CiliumNetworkPolicy` extends it.

---

## 4. Storage

- **Volumes** are mounted into containers; lifetime depends on type. `emptyDir` lives with the pod;
  `configMap`/`secret`/`projected`/`downwardAPI` inject data; `persistentVolumeClaim` binds durable
  storage.
- **PV / PVC / StorageClass:**
  - **PersistentVolume (PV)** — a piece of storage in the cluster (cluster-scoped).
  - **PersistentVolumeClaim (PVC)** — a namespaced request for storage (size, access mode, class).
  - **StorageClass** — defines a *provisioner* and parameters for **dynamic provisioning**: creating a
    PVC auto-creates a PV. This is the normal path; static PVs are rare.
  - **`reclaimPolicy`**: `Delete` (default for dynamic — deletes backing volume with the PV) vs
    `Retain` (keeps data for manual recovery). Use `Retain` for anything precious.
  - **`volumeBindingMode: WaitForFirstConsumer`** delays binding until a pod is scheduled, so the volume
    lands in the same zone as the pod — **essential in multi-zone clusters** to avoid zonal pinning.
- **CSI** (Container Storage Interface) is how all modern storage drivers plug in; it also powers
  snapshots (`VolumeSnapshot`), cloning, and volume expansion (`allowVolumeExpansion: true`, then grow
  the PVC).
- **Access modes:** `ReadWriteOnce` (RWO — one node), `ReadWriteOncePod` (RWOP — exactly one pod, for
  strict single-writer), `ReadOnlyMany` (ROX), `ReadWriteMany` (RWX — needs a shared filesystem like
  NFS/CephFS; most block storage is RWO only). Don't assume RWX; most cloud block volumes are RWO.
- **StatefulSet `volumeClaimTemplates`** — each replica gets its own PVC (`data-web-0`, `data-web-1`)
  that follows the pod across reschedules. This is the canonical way to give stateful pods durable,
  per-instance storage.
- **Generic ephemeral volumes** give you a CSI-backed scratch volume that lives and dies with the pod
  (declared inline under `volumes`), when `emptyDir` isn't enough but you don't need persistence.

---

## 5. Config & secrets

- **ConfigMap** — non-sensitive key/value config. Inject as env vars or, **preferably, as files** via a
  volume mount (file mounts can update live; env vars are fixed at pod start). Keep config out of the
  image so the same image runs in every environment.
- **Secret** — same shape, for sensitive data, but **base64-encoded, not encrypted**. Caveats every
  practitioner must internalize:
  - **Not encrypted at rest by default** — anyone with etcd access or broad `get secrets` RBAC reads
    them. Enable **encryption at rest** (KMS provider) and/or use an external store (External Secrets
    Operator, Vault, cloud secret manager + CSI Secrets Store driver).
  - **Don't commit Secrets to Git** in plaintext. Use sealed-secrets / SOPS / external secret operators
    for GitOps.
  - **Prefer file mounts over env vars** for secrets: env vars leak into `/proc`, crash dumps, child
    processes, and logs more easily, and don't auto-update.
  - Lock down RBAC: who can `get`/`list` secrets in a namespace effectively owns them.
- **Projected volumes** combine multiple sources (configMap, secret, downwardAPI,
  serviceAccountToken) into one directory — and are how you get short-lived, audience-bound, auto-
  rotated **ServiceAccount tokens** (bound tokens) instead of legacy never-expiring token secrets.
- **Env vs file rule of thumb:** non-secret, small, fixed → env is fine; secret or reloadable → file.
  Mounted ConfigMap/Secret files update in place (with propagation delay); the app must re-read them.

---

## 6. Security

- **RBAC** — `Role`/`RoleBinding` (namespaced) and `ClusterRole`/`ClusterRoleBinding` (cluster-wide)
  grant verbs (`get`,`list`,`watch`,`create`,`update`,`patch`,`delete`) on resources to subjects
  (users, groups, ServiceAccounts). **Least privilege:** grant the narrowest verbs on the narrowest
  resources in the narrowest scope. **Never bind apps to `cluster-admin`.** Audit with
  `kubectl auth can-i --list --as=system:serviceaccount:ns:sa`.
- **ServiceAccounts** — the identity a pod uses to call the API. Each namespace has a `default` SA.
  **Set `automountServiceAccountToken: false`** on pods (or the SA) that don't call the API — most
  don't. Give API-using pods a dedicated SA with a tight Role. Tokens are now short-lived bound tokens.
- **Pod Security Admission (PSA)** — built-in, namespace-labeled enforcement replacing PodSecurityPolicy.
  Three levels, applied via namespace labels:
  - **`privileged`** — unrestricted (system/infra namespaces only).
  - **`baseline`** — blocks known privilege escalations; minimally restrictive.
  - **`restricted`** — hardened best practice: non-root, no privilege escalation, drop ALL capabilities,
    `seccompProfile: RuntimeDefault`, restricted volume types, etc.
  Label namespaces `pod-security.kubernetes.io/enforce: restricted` (plus `audit`/`warn` to ease
  rollout). **Default new app namespaces to `restricted`.** PSA is coarse; for fine-grained policy use a
  policy engine (Kyverno, OPA/Gatekeeper).
- **`securityContext`** (pod- and container-level) — the actual hardening knobs:
  ```yaml
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop: ["ALL"]
    seccompProfile:
      type: RuntimeDefault
  ```
  Drop ALL capabilities and add back only the few you truly need (almost never any). `privileged: true`
  and host namespaces (`hostNetwork`/`hostPID`/`hostIPC`) are red flags outside infra DaemonSets.
- **Images:** pin by digest, scan for CVEs, use minimal/distroless bases, run as non-root, sign and
  verify (cosign/Sigstore) with admission enforcement where it matters.

---

## 7. Resource management & multi-tenancy

- **Namespaces** are the primary tenancy/isolation boundary: scope for names, RBAC, quotas, network
  policy, and PSA. One team/app/environment per namespace is a sane default.
- **`ResourceQuota`** caps aggregate resource use per namespace (total CPU/memory requests+limits, pod
  count, PVC count, object counts). **A namespace with a CPU/memory quota *forces* every pod to set
  requests/limits** — a useful lever to enforce good behavior cluster-wide.
- **`LimitRange`** sets per-pod/container defaults and min/max for requests/limits in a namespace, so
  pods that omit them get sane values and can't request absurd amounts. Pair with ResourceQuota.
- **Fairness:** quotas prevent one tenant from starving others; PriorityClasses arbitrate under
  contention; node pools + taints physically isolate noisy/special workloads. For batch fairness and
  gang scheduling across tenants, use **Kueue** (`[[kueue-advanced]]`). Hard multi-tenancy (untrusted
  tenants) needs more than namespaces — separate clusters or virtual clusters / strong sandboxing.

---

## 8. Health, rollout & production ops (zero-downtime)

Zero-downtime rollout is a *system* of cooperating features — miss one and you drop requests:

1. **`readinessProbe`** so new pods receive traffic only when actually ready, and the rollout waits.
2. **RollingUpdate** with `maxUnavailable`/`maxSurge` so capacity never dips below what you need.
3. **`PodDisruptionBudget` (PDB)** — caps how many pods of a set can be **voluntarily** disrupted at
   once (`minAvailable`/`maxUnavailable`). Protects you during node drains/upgrades/autoscaler scale-
   down. **Every multi-replica service needs a PDB.** (PDBs don't stop involuntary disruptions like node
   crashes — that's what replicas + spread are for.)
4. **Graceful termination** — on pod delete, K8s: (a) removes the pod from Service endpoints, and (b)
   sends **SIGTERM** to PID 1, waits `terminationGracePeriodSeconds` (default 30), then **SIGKILL**.
   But endpoint removal and SIGTERM are **concurrent and racy** — in-flight requests may still arrive
   for a moment. The robust pattern:
   - Handle SIGTERM: stop accepting new work, drain in-flight, then exit.
   - Add a **`preStop` hook** sleep (e.g., `sleep 5–15`) so the pod keeps serving while endpoint
     de-registration propagates through kube-proxy/CNI/LB before the app stops.
   - Set `terminationGracePeriodSeconds` longer than your longest in-flight request + preStop sleep.
   ```yaml
   lifecycle:
     preStop:
       exec: { command: ["/bin/sh", "-c", "sleep 10"] }
   terminationGracePeriodSeconds: 45
   ```
5. **Rollback** — `kubectl rollout undo deploy/x [--to-revision=N]`. Keep `revisionHistoryLimit`
   reasonable. Practice rollbacks; a deploy you can't roll back isn't done.

Other ops staples: `kubectl drain --ignore-daemonsets --delete-emptydir-data` (respects PDBs) for node
maintenance; cordon before drain; respect PriorityClasses; use surge-friendly node upgrades.

---

## 9. Observability & the kubectl debugging workflow

Debugging is always **desired vs. actual**. Work top-down:

1. **`kubectl get pods -o wide`** — phase, restarts, node, IP. Restarts climbing = crash loop.
2. **`kubectl describe pod <p>`** — the **Events** at the bottom are the single highest-value signal:
   scheduling failures, image pulls, probe failures, OOM, mount errors.
3. **`kubectl logs <p> [-c container] [--previous]`** — `--previous` shows the *crashed* container's
   logs (the actual error). `-f` to follow.
4. **`kubectl get events --sort-by=.lastTimestamp -A`** (or `kubectl events`) — cluster/namespace-wide
   timeline.
5. **`kubectl debug`** — ephemeral debug container shares the target pod's namespaces without modifying
   it: `kubectl debug -it <pod> --image=busybox --target=<container>`. Or `--copy-to` to clone a pod
   with a debug image, or `node/<node>` for node debugging. Essential for distroless images with no shell.
6. `kubectl top pod/node` (needs metrics-server) for live resource use; `kubectl get --raw` /
   `port-forward` / `exec` as needed.

### Failure signatures (memorize these)

| Symptom | Likely cause | Where to look / fix |
|---|---|---|
| **Pending** | No node fits: insufficient requests, no matching affinity/taint toleration, no available PV in zone, quota exceeded | `describe pod` Events (FailedScheduling); check requests vs. allocatable, taints, `topologySpreadConstraints`, PVC binding |
| **ImagePullBackOff / ErrImagePull** | Bad image name/tag, private registry auth, rate limit | `describe` Events; fix tag/`imagePullSecrets`; verify registry access |
| **CrashLoopBackOff** | App exits/crashes on start, or **bad liveness probe** restarting a healthy app | `logs --previous`; check probe config, config/secret mounts, missing deps, exit code |
| **OOMKilled** (exit 137) | Container exceeded memory **limit** | `describe` shows `OOMKilled`; raise limit or fix leak; set `memory request==limit` |
| **CreateContainerConfigError** | Missing ConfigMap/Secret key referenced by env/volume | `describe` Events; create/fix the referenced object |
| **Evicted** | Node resource pressure (memory/disk); BestEffort/over-limit pods go first | `describe node`; add requests, fix QoS, free disk, scale nodes |
| **Service returns nothing / connection refused** | Selector ≠ pod labels, pods not Ready, wrong port | `kubectl get endpointslices`; check Service selector vs pod labels, readiness |
| **Init:Error / Init:CrashLoopBackOff** | An initContainer is failing | `logs <pod> -c <init-container>` |

`exit 137` = SIGKILL (often OOM or grace-period timeout); `exit 143` = SIGTERM. A pod stuck
**Terminating** usually means a finalizer is blocking, the grace period is long, or the node is gone.

---

## 10. kubectl mastery & manifest hygiene

- **Declarative over imperative.** `kubectl apply -f` / `kustomize` / Helm / Argo CD / Flux. Reserve
  `create`/`edit`/`patch`/`scale` on live objects for break-glass — they drift from Git. `apply` merges
  via the client-side three-way merge on the `last-applied-configuration` annotation; pass
  `--server-side` to opt into server-side apply (managedFields-based, no size limit, better for large
  objects and multi-writer ownership).
- **Dry-run before you ship:** `kubectl apply --dry-run=server -f` validates against admission/the API
  (catches PSA, webhook, schema, quota issues) — far stronger than `--dry-run=client`.
- **`kubectl diff -f`** shows exactly what an apply would change vs. live — run it before every apply.
- **`kubectl explain <kind>.spec.<field> --recursive`** is the authoritative, version-correct field
  reference for *your* cluster — use it instead of guessing field names.
- Handy: `-o yaml|json`, `-o jsonpath=...`, `-o custom-columns=...`, `--field-selector`, `-l <selector>`,
  `-w` (watch), `--context`/`--namespace`. Set a kubeconfig context per cluster; never operate prod with
  a default-namespace footgun.
- **Manifest hygiene / GitOps-friendly:**
  - Apply the **recommended labels** on every object: `app.kubernetes.io/name`, `instance`,
    `version`, `component`, `part-of`, `managed-by`. Selectors and tooling key off them.
  - **A Deployment's `.spec.selector` is immutable** — set it deliberately and keep it stable; changing
    it forces a delete/recreate.
  - Keep manifests **idempotent and namespaced**; avoid `generateName` in GitOps (you can't reconcile a
    random name). Avoid status fields and cluster-injected defaults in source.
  - One concern per file or use Kustomize bases/overlays; pin Helm chart versions; render and review.
  - Annotations carry non-identifying metadata (change-cause, checksum to force a roll on config change,
    tool config). A common trick: a `checksum/config` annotation that changes with the ConfigMap so the
    Deployment rolls when config changes.

---

## 11. Anti-patterns to reject in review

- **`image: foo:latest`** (or any mutable tag) — breaks rollbacks/reproducibility; pin tags or digests.
- **No resource requests/limits** — BestEffort pods, unschedulable surprises, OOM/eviction roulette.
- **Naked Pods / bare ReplicaSets** — no self-healing; use a controller.
- **Secrets in env vars or committed to Git in plaintext** — leak-prone; mount as files, use a secret
  store, enable encryption at rest.
- **Liveness probe that checks downstream dependencies** — turns a dependency blip into a restart storm.
- **`cluster-admin` (or wide ClusterRoles) for apps; using the `default` SA with token automount on.**
- **`privileged: true`, `hostNetwork`/`hostPID`/`hostIPC`, or running as root with all caps** on app
  pods — outside infra DaemonSets this is almost always wrong. Same for raw `hostPath` mounts.
- **Single replica "HA," or all replicas on one node/zone** — no PDB, no spread, no anti-affinity.
- **`Recreate` strategy or no readiness probe** on a service expected to be zero-downtime.
- **No PodDisruptionBudget** on a multi-replica service (node drains take it down).
- **No default-deny NetworkPolicy** in a multi-tenant namespace (everything can reach everything).
- **Imperative `kubectl edit`/`patch` in prod** as the normal workflow — config drifts from Git.
- **One giant namespace for everything** — no quota/RBAC/network isolation; blast radius is the cluster.
- **`emptyDir` for data you can't lose**, or assuming RWX from a block volume.

---

## 12. Production-readiness checklist & verification gate (definition of done)

One gate, two halves: check every box, then confirm the second half by actually running the commands
against the target cluster — don't assume. Paste this into PRs.

- [ ] **Controller, not a naked Pod** — Deployment/StatefulSet/DaemonSet/Job owns the pods.
- [ ] **Resources:** `requests` set on every container; **memory `limit == request`**; CPU limit chosen
      deliberately. QoS is Burstable or Guaranteed (never BestEffort).
- [ ] **Probes:** `readinessProbe` present; `liveness`/`startup` only where justified, cheap, and not
      checking downstream dependencies.
- [ ] **`securityContext`:** `runAsNonRoot`, `allowPrivilegeEscalation: false`,
      `readOnlyRootFilesystem`, `capabilities.drop: ["ALL"]`, `seccompProfile: RuntimeDefault`;
      namespace labeled PSA `restricted`.
- [ ] **RBAC/identity:** dedicated least-privilege ServiceAccount; `automountServiceAccountToken: false`
      unless the pod calls the API; no `cluster-admin`.
- [ ] **Secrets** mounted as files from a secret store / sealed, with encryption at rest — not env vars.
- [ ] **Availability:** ≥2 replicas, `topologySpreadConstraints` across zone **and** node, and a
      **PodDisruptionBudget**.
- [ ] **Zero-downtime:** RollingUpdate with sane `maxUnavailable`/`maxSurge`, `preStop` drain hook,
      SIGTERM handling, `terminationGracePeriodSeconds` ≥ longest request + preStop sleep.
- [ ] **Namespace guardrails:** `ResourceQuota` + `LimitRange`; default-deny `NetworkPolicy` + explicit
      allows.
- [ ] **Image pinned** to an immutable tag/digest, scanned, non-root.
- [ ] **Labels & GitOps:** recommended `app.kubernetes.io/*` labels on every object; immutable selector;
      GitOps-managed (no live drift).
- [ ] **Observability:** logs to stdout/stderr (structured); metrics + tracing exposed; alerts on the
      failure signatures.
- [ ] **Rollback path exists:** `kubectl rollout undo` tested; `revisionHistoryLimit` and Job
      `ttlSecondsAfterFinished` set.

**Commands to run before calling it done:**
```bash
kubectl apply --dry-run=server -f .          # validates against admission/PSA/quota/schema
kubectl diff -f .                            # exactly what will change vs. live
kubectl rollout status deploy/<x>            # rollout actually converged (probes gated it)
kubectl get pdb,hpa,networkpolicy -n <ns>    # PDB + autoscaling + default-deny exist
kubectl auth can-i --list \
  --as=system:serviceaccount:<ns>:<sa>       # SA grants are narrow as intended
kubectl get pod <p> -o jsonpath='{.status.qosClass}'   # Burstable/Guaranteed, not BestEffort
kubectl rollout undo deploy/<x> --dry-run=client       # a rollback path exists
```

---

## 13. Rationalizations & rebuttals

The excuses that precede a 2 a.m. page. When you catch yourself (or an agent) saying one of these, stop.

| Rationalization | Rebuttal |
|---|---|
| "No resource limits, it's fine — it barely uses anything." | No requests = BestEffort = first evicted/OOM-killed, and the scheduler can't reserve for it. Set requests; `memory limit == request`. |
| "`:latest` is convenient — always pulls the newest." | Mutable tags break rollbacks and reproducibility; two pods can run different code. Pin a digest/immutable tag. |
| "I'll add probes later." | Without a `readinessProbe` the rollout never gates and you ship broken pods into the LB. It is one of the cheapest correctness wins — add it now. |
| "It's a quick job, a naked Pod is enough." | A naked Pod is never rescheduled if its node dies. Wrap it in a Job/Deployment so the controller heals it. |
| "Single replica is fine, the node won't go down." | Nodes drain on every upgrade and autoscale event. One replica with no PDB means guaranteed downtime during routine ops. Run ≥2 + spread + PDB. |
| "cluster-admin so I stop fighting RBAC errors." | That SA owns the cluster the moment it leaks. Grant the narrow verbs the app actually uses; verify with `kubectl auth can-i --list`. |
| "Secrets in env vars are simpler than mounts." | Env leaks into `/proc`, crash dumps, child processes, and logs, and never rotates. Mount as files from a secret store with encryption at rest. |
| "Liveness probe hitting the DB proves it's healthy." | It turns a downstream blip into a cluster-wide restart storm and self-inflicted CrashLoopBackOff. Liveness checks the process only; depend on readiness. |

---

## 14. Version awareness

Kubernetes ships ~3 minor releases/year and supports ~the last 3; betas get enabled/disabled and
defaults flip between them. Before relying on a specific field, default, or API:
- Confirm with `kubectl version`, `kubectl api-resources`, and `kubectl explain` against the target
  cluster — not from memory.
- Don't assume a beta/alpha API or feature gate is enabled. Native sidecars, Gateway API, and
  in-place pod resize, for example, each graduated on their own timeline — check your version.
- When unsure of an exact flag/field/version, describe it generally and tell the reader to verify the
  current docs rather than inventing specifics.

---

## 15. Canonical references

- Kubernetes documentation — https://kubernetes.io/docs/
- Concepts (workloads, services, storage, config, security) — https://kubernetes.io/docs/concepts/
- kubectl reference & cheat sheet — https://kubernetes.io/docs/reference/kubectl/cheatsheet/
- Pod Security Standards — https://kubernetes.io/docs/concepts/security/pod-security-standards/
- Configure Liveness/Readiness/Startup Probes —
  https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
- Gateway API — https://gateway-api.sigs.k8s.io/
- Network Policies — https://kubernetes.io/docs/concepts/services-networking/network-policies/
- Recommended labels — https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/
- Production best practices / KEPs — https://github.com/kubernetes/enhancements

---

# Canonical Kubernetes manifests

Correct, production-grade reference manifests to imitate. Adjust names, images, and resource numbers to
your context, but keep the structural choices (probes, limits, securityContext, spread, PDB, default-deny).
Verify field availability against your cluster version with `kubectl explain` — see the guide's version note.

---

## 1. Production Deployment (probes + limits + securityContext + spread) and its PDB + Service

A stateless service done right: pinned image, requests with `memory limit == request`, readiness probe,
hardened securityContext, zonal+node spread, graceful drain, and a matching PodDisruptionBudget.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  namespace: shop
  labels:
    app.kubernetes.io/name: web
    app.kubernetes.io/part-of: shop
    app.kubernetes.io/version: "1.4.2"
spec:
  replicas: 3
  revisionHistoryLimit: 5
  selector:
    matchLabels: { app.kubernetes.io/name: web }   # immutable — set once, keep stable
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0       # never drop below desired capacity during a roll
      maxSurge: 1
  template:
    metadata:
      labels: { app.kubernetes.io/name: web, app.kubernetes.io/version: "1.4.2" }
    spec:
      serviceAccountName: web                 # dedicated, least-privilege SA
      automountServiceAccountToken: false     # this pod doesn't call the K8s API
      terminationGracePeriodSeconds: 45       # ≥ longest in-flight request + preStop sleep
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
        seccompProfile: { type: RuntimeDefault }
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector: { matchLabels: { app.kubernetes.io/name: web } }
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector: { matchLabels: { app.kubernetes.io/name: web } }
      containers:
        - name: web
          image: registry.example.com/web@sha256:<digest>   # pin by digest, never :latest
          ports: [{ name: http, containerPort: 8080 }]
          resources:
            requests: { cpu: "250m", memory: "256Mi" }
            limits:   { memory: "256Mi" }      # memory limit == request; CPU limit omitted deliberately
          readinessProbe:                      # gates traffic + makes the rollout wait
            httpGet: { path: /readyz, port: http }
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:                       # cheap, self-only check — does NOT touch dependencies
            httpGet: { path: /livez, port: http }
            periodSeconds: 10
            failureThreshold: 6
          startupProbe:                        # generous budget for slow boot
            httpGet: { path: /livez, port: http }
            periodSeconds: 5
            failureThreshold: 30
          lifecycle:
            preStop:                           # keep serving while endpoint removal propagates
              exec: { command: ["/bin/sh", "-c", "sleep 10"] }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
          volumeMounts:
            - { name: tmp, mountPath: /tmp }   # writable scratch since root FS is read-only
      volumes:
        - name: tmp
          emptyDir: {}
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: web, namespace: shop }
spec:
  minAvailable: 2                              # node drains/upgrades can't take us below 2
  selector:
    matchLabels: { app.kubernetes.io/name: web }
---
apiVersion: v1
kind: Service
metadata: { name: web, namespace: shop, labels: { app.kubernetes.io/name: web } }
spec:
  selector: { app.kubernetes.io/name: web }   # MUST match pod labels or endpoints are empty
  ports: [{ name: http, port: 80, targetPort: http }]
```

Why it's correct: readiness gates traffic and the rollout; `maxUnavailable: 0` + `maxSurge` keeps
capacity; PDB protects voluntary disruptions; spread gives zonal+node HA; `preStop` + grace + SIGTERM
handling (in the app) give zero-downtime termination; hardened securityContext satisfies PSA `restricted`.

---

## 2. StatefulSet (stable identity + per-pod storage via volumeClaimTemplates)

For quorum/stateful systems. Headless Service for stable DNS, ordered updates, a dedicated PVC per pod,
and a PDB. Each pod gets `db-0`, `db-1`, … resolvable at `db-0.db.data.svc.cluster.local`.

```yaml
apiVersion: v1
kind: Service
metadata: { name: db, namespace: data, labels: { app.kubernetes.io/name: db } }
spec:
  clusterIP: None                              # headless → stable per-pod DNS
  selector: { app.kubernetes.io/name: db }
  ports: [{ name: pg, port: 5432, targetPort: pg }]
---
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: db, namespace: data, labels: { app.kubernetes.io/name: db } }
spec:
  serviceName: db                              # the headless Service above
  replicas: 3
  podManagementPolicy: OrderedReady
  updateStrategy: { type: RollingUpdate }      # ordered, highest ordinal first
  selector: { matchLabels: { app.kubernetes.io/name: db } }
  template:
    metadata: { labels: { app.kubernetes.io/name: db } }
    spec:
      terminationGracePeriodSeconds: 60
      securityContext: { runAsNonRoot: true, runAsUser: 10001, fsGroup: 10001,
                         seccompProfile: { type: RuntimeDefault } }
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector: { matchLabels: { app.kubernetes.io/name: db } }
      containers:
        - name: db
          image: registry.example.com/postgres@sha256:<digest>
          ports: [{ name: pg, containerPort: 5432 }]
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits:   { memory: "1Gi" }
          readinessProbe:
            exec: { command: ["pg_isready", "-q"] }
            periodSeconds: 10
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: { drop: ["ALL"] }
          volumeMounts: [{ name: data, mountPath: /var/lib/postgresql/data }]
  volumeClaimTemplates:                         # one durable PVC per pod (data-db-0, …)
    - metadata: { name: data }
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd              # WaitForFirstConsumer class for zonal correctness
        resources: { requests: { storage: 50Gi } }
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: db, namespace: data }
spec:
  maxUnavailable: 1                             # keep quorum during drains
  selector: { matchLabels: { app.kubernetes.io/name: db } }
```

Notes: deleting the StatefulSet does **not** delete the PVCs (data safety). Use a StorageClass with
`volumeBindingMode: WaitForFirstConsumer` so volumes are created in the pod's zone. Most block storage is
`ReadWriteOnce` — design around per-pod volumes, not shared RWX.

---

## 3. NetworkPolicy — default-deny + explicit allows

Two policies per namespace: a baseline default-deny (ingress + egress), then a narrow allow. Requires a
CNI that enforces NetworkPolicy (Calico, Cilium, etc.). Note: DNS egress must be explicitly allowed once
you default-deny egress, or every in-cluster lookup breaks.

```yaml
# 3a. Default-deny ALL ingress and egress for the namespace.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny, namespace: shop }
spec:
  podSelector: {}                # selects every pod in the namespace
  policyTypes: ["Ingress", "Egress"]
  # no ingress/egress rules → nothing allowed
---
# 3b. Allow CoreDNS egress (UDP/TCP 53) — required once egress is denied.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-dns, namespace: shop }
spec:
  podSelector: {}
  policyTypes: ["Egress"]
  egress:
    - to:
        - namespaceSelector:
            matchLabels: { kubernetes.io/metadata.name: kube-system }
      ports:
        - { protocol: UDP, port: 53 }
        - { protocol: TCP, port: 53 }
---
# 3c. Allow ingress to `web` only from the ingress controller, and its egress only to `api`.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: web-allow, namespace: shop }
spec:
  podSelector: { matchLabels: { app.kubernetes.io/name: web } }
  policyTypes: ["Ingress", "Egress"]
  ingress:
    - from:
        - namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: ingress-nginx } }
      ports: [{ protocol: TCP, port: 8080 }]
  egress:
    - to:
        - podSelector: { matchLabels: { app.kubernetes.io/name: api } }
      ports: [{ protocol: TCP, port: 8080 }]
```

Why it's correct: NetworkPolicy is **allow-list and additive** — once any policy selects a pod, only the
union of matching allow rules is permitted. Start from default-deny, then open exactly what each tier
needs (don't forget DNS). This contains blast radius: a compromised pod can't freely traverse the
namespace.
