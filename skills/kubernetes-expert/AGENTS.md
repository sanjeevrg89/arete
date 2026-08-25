# Kubernetes Expert — always-on checklist

Authoritative source: **`kubernetes-expert-guide.md`** in this directory (read it for full detail and
rationale). Canonical manifests to imitate: **`examples.md`**. This file is the condensed, always-on
ruleset for using Kubernetes well in production. Scope: *using* K8s — not control-plane internals
(`[[kubernetes-internals-expert]]`) or writing controllers/operators
(`[[kubernetes-controller-expert]]`, `[[kubernetes-operator-expert]]`).

## Mental model
Kubernetes is **declarative and level-triggered**: you submit desired state (`spec`), controllers
reconcile actual state (`status`) toward it. Control plane (apiserver/etcd/controllers/scheduler) holds
state; nodes (kubelet/kube-proxy/CNI) run pods. **Debug = compare desired vs. actual.** Labels/selectors
are the glue. Apply manifests from Git; don't imperatively mutate live objects in prod.

## Non-negotiables
- **No naked Pods** — always a Deployment/StatefulSet/DaemonSet/Job. Pick the right one (stable
  identity/storage → StatefulSet; per-node → DaemonSet; run-to-completion → Job/CronJob; else Deployment).
- **Set resource `requests`; memory `limit == request`** (memory is incompressible → OOMKilled). Choose
  CPU limits deliberately (throttling vs. burst). Never run real workloads as BestEffort.
- **`readinessProbe` on every serving workload.** Add liveness/startup only where justified; a liveness
  probe that checks a dependency causes self-inflicted CrashLoopBackOff. Liveness restarts; readiness
  gates traffic; startup protects slow boots.
- **Zero-downtime = readiness + RollingUpdate (`maxUnavailable`/`maxSurge`) + PodDisruptionBudget +
  `preStop` drain sleep + SIGTERM handling + adequate `terminationGracePeriodSeconds`.**
- **Never `:latest`** — pin immutable tags/digests.
- **Secrets are base64, not encrypted by default.** Mount as files (not env), don't commit plaintext,
  enable encryption at rest / external secret store, tighten RBAC on `secrets`.
- **Least-privilege RBAC + dedicated ServiceAccount;** `automountServiceAccountToken: false` unless the
  pod calls the API. No `cluster-admin` for apps.
- **Harden every pod:** `runAsNonRoot`, drop ALL capabilities, `readOnlyRootFilesystem`,
  `allowPrivilegeEscalation: false`, `seccompProfile: RuntimeDefault`. Default namespaces to PSA
  `restricted`. No `privileged`/`hostNetwork`/host mounts on app pods.
- **HA = ≥2 replicas + `topologySpreadConstraints` (zone AND node) + PDB.** One node/zone is not HA.
- **Multi-tenancy per namespace:** `ResourceQuota` + `LimitRange` + **default-deny `NetworkPolicy`** +
  explicit allows. Without a policy, all pods can reach all pods.

## Scheduling / placement
Requests drive scheduling and QoS. Use `nodeSelector`/affinity to target, taints+tolerations for
dedicated pools (tolerations don't attract — pair with affinity), `topologySpreadConstraints` to spread
(prefer over `preferred` anti-affinity), PriorityClasses for contention.

## Networking / storage / config (quick rules)
- Service types: ClusterIP (default), NodePort, LoadBalancer (front many with Ingress/Gateway), headless
  (`clusterIP: None`, for StatefulSets). Check `endpointslices` first when a Service routes nothing
  (usually selector ≠ pod labels). **Prefer Gateway API over Ingress** for new L7 routing.
- DNS: `name.namespace.svc.cluster.local`; `ndots:5` taxes external lookups — use FQDNs with trailing dot.
- Storage: dynamic provisioning via StorageClass; `WaitForFirstConsumer` in multi-zone; `Retain` for
  precious data; most block volumes are RWO (don't assume RWX). StatefulSet → `volumeClaimTemplates`.
- Config out of the image: ConfigMap/Secret as **file mounts** (live-updatable) over env where possible.

## Debugging workflow
`get pods -o wide` → `describe pod` (**Events!**) → `logs --previous` → `get events --sort-by` /
`get endpointslices` → `kubectl debug` ephemeral container. Signatures: **Pending** (no node
fits/quota/PV), **ImagePullBackOff** (tag/auth), **CrashLoopBackOff** (app or bad liveness),
**OOMKilled/137** (memory limit), **CreateContainerConfigError** (missing ConfigMap/Secret key),
**Evicted** (node pressure; BestEffort first). Validate with `kubectl apply --dry-run=server` and
`kubectl diff`; use `kubectl explain` for version-correct fields.

## Reject in review
`:latest`; no requests/limits; naked Pods; secrets in env/Git; liveness probing dependencies;
`cluster-admin`/default-SA-token for apps; `privileged`/`hostNetwork`; single-replica "HA"/no PDB/no
spread; `Recreate` or no readiness on a zero-downtime service; no default-deny NetworkPolicy; imperative
prod edits that drift from Git.

**Version note:** K8s moves fast — verify fields/defaults/APIs with `kubectl explain` and the docs for
the cluster's exact version. Don't assume a beta API/feature gate is on; don't invent flags or versions.
