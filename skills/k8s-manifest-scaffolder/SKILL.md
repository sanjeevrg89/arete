---
name: k8s-manifest-scaffolder
description: Use this to GENERATE production-grade Kubernetes manifests from a short spec — turn a one-line
  ask like "stateless API, 3 replicas, needs a config + secret, autoscale on CPU" or "Postgres
  StatefulSet with 50Gi per pod" or "nightly batch job" or "multi-host vLLM inference" into correct,
  ready-to-apply YAML. Picks the right workload kind (Deployment, StatefulSet + volumeClaimTemplates,
  Job/CronJob, DaemonSet, JobSet, LeaderWorkerSet) and bakes in safe defaults: resource requests+limits,
  liveness/readiness/startup probes, hardened securityContext (non-root, drop ALL caps,
  readOnlyRootFilesystem), PodDisruptionBudget, recommended labels, topologySpreadConstraints, a Service,
  an HPA when scaling is requested, and an optional default-deny NetworkPolicy. Emits manifests plus a
  `kubectl apply --dry-run=server` validation note. Use when scaffolding/bootstrapping new K8s YAML,
  Helm/Kustomize bases, or example manifests. For deep K8s judgment/debugging see [[kubernetes-expert]].
---

# K8s Manifest Scaffolder (spec → production manifests)

Generate manifests with the judgment of an engineer who has shipped these to production for years.
This is a **doer**: given a short spec, you EMIT correct, hardened, ready-to-apply Kubernetes YAML —
not advice about Kubernetes. Safe defaults are baked into every artifact so the generated output is
production-grade by construction, never a bare skeleton the user must harden afterward.

## How to use this skill

1. **Read `k8s-manifest-scaffolder-guide.md`** in this directory — it is the generation procedure:
   intake → choose workload kind → fill template → apply safe defaults → emit + validation note,
   plus reusable templates you fill in.
2. For full worked generations (input spec → emitted manifests), read **`examples.md`** and imitate
   the structure exactly (probes, limits, securityContext, spread, PDB, Service, HPA).
3. Match the target cluster's existing conventions (namespaces, labels, ingress class, storage classes,
   GitOps tooling) where known; apply the safe-default correctness rules regardless.

## The essentials (full detail in `k8s-manifest-scaffolder-guide.md`)

- **Intake first.** Resolve workload kind, replicas, image, ports, resources, config/secrets, storage,
  scaling, exposure, namespace. Anything unstated → apply the documented safe default and label it as an
  assumption in a comment, don't ask round-trips for trivia.
- **Pick the right kind:** stateless → Deployment; stable identity + per-pod storage → StatefulSet with
  `volumeClaimTemplates` + headless Service; run-to-completion → Job; scheduled → CronJob; one-per-node →
  DaemonSet; multi-host training → JobSet; multi-host inference → LeaderWorkerSet ([[jobset-leaderworkerset]]).
- **Every generated pod ships hardened:** `requests` + `limits` (memory `limit == request`),
  readiness + liveness + startup probes, `runAsNonRoot`, `drop: ["ALL"]`, `readOnlyRootFilesystem: true`,
  `allowPrivilegeEscalation: false`, `seccompProfile: RuntimeDefault`. This satisfies PSA `restricted`.
- **Every multi-replica workload ships HA:** `topologySpreadConstraints` (zone + node) + a
  PodDisruptionBudget + the recommended `app.kubernetes.io/*` labels.
- **Always emit a Service** matching the pod labels (headless for StatefulSets).
- **Emit an HPA when scaling is requested** (`autoscaling/v2`, CPU/memory/custom) — and then do NOT set
  `replicas` on the workload so the HPA owns it ([[autoscaling-kubernetes]]).
- **Offer a default-deny NetworkPolicy** (+ DNS allow) as an opt-in block; include it when the spec
  mentions isolation/security/multi-tenancy.
- **Never `:latest`** — pin a tag or digest placeholder and comment it.
- **GKE:** when the target is GKE, add the Workload Identity ServiceAccount annotation pattern and node
  selector hints ([[gke-master]]).
- **Close every generation with validation:** `kubectl apply --dry-run=server -f -` and
  `kubectl explain` for version-sensitive fields. Flag fields whose apiVersion may differ by cluster.

## Related skills

- `[[kubernetes-expert]]` — the deep "using Kubernetes well" reference (judgment, debugging, the why
  behind each safe default). This scaffolder GENERATES; that skill EXPLAINS and reviews.
- `[[autoscaling-kubernetes]]` — HPA/VPA/KEDA detail when the scaling block needs to go beyond CPU.
- `[[jobset-leaderworkerset]]` — multi-host training (JobSet) and multi-host inference (LeaderWorkerSet).
- `[[gke-master]]` — GKE specifics (Autopilot, Workload Identity, node pools) for GKE targets.
- `[[serving-frameworks]]` — vLLM/SGLang/Triton container args to drop into a serving manifest.
