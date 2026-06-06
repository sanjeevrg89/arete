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
