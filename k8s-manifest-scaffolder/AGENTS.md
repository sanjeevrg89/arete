# K8s Manifest Scaffolder — always-on checklist

Authoritative source: **`k8s-manifest-scaffolder-guide.md`** in this directory (full procedure +
templates). Worked generations to imitate: **`examples.md`**. This is a **doer**: GENERATE
production-grade Kubernetes manifests from a short spec — emit hardened, ready-to-apply YAML, never a bare
skeleton. For K8s judgment/debugging see [[kubernetes-expert]].

## The generation procedure (always)
1. **Intake:** resolve workload kind, name/namespace, replicas, image, ports, resources, config/secrets,
   storage, scaling, exposure, isolation, platform. Unstated → apply the documented safe default and mark
   it `# assumption:` in the YAML. Ask the user only when an unstated field changes the output materially.
2. **Choose kind:** stateless → Deployment; stable identity / per-pod disk / ordering → StatefulSet (+
   headless Service + `volumeClaimTemplates`); run-to-completion → Job; scheduled → CronJob; per-node →
   DaemonSet; multi-host training → JobSet; multi-host inference → LeaderWorkerSet ([[jobset-leaderworkerset]]).
3. **Fill the template** from the guide (don't strip the safe defaults).
4. **Apply safe defaults** (checklist below).
5. **Emit + validation note** (`kubectl apply --dry-run=server`, `kubectl explain`).

## Baked-in safe defaults (every generated pod/workload)
- Resource `requests` (CPU+mem) on every container; memory `limit == request`. CPU limit only if isolation
  is required.
- `readinessProbe` always; `livenessProbe` self-only (never probes a dependency); `startupProbe` for slow boot.
- Hardened securityContext: `runAsNonRoot`, `runAsUser`, `fsGroup`, `seccompProfile: RuntimeDefault`,
  `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true` (+ emptyDir scratch), `drop: ["ALL"]`.
  No `privileged`/`hostNetwork`/host mounts.
- HA: `topologySpreadConstraints` (zone DoNotSchedule + node ScheduleAnyway) + PodDisruptionBudget for any
  multi-replica long-running workload.
- Recommended `app.kubernetes.io/*` labels on every object; selector uses ONLY the stable `name` label
  (selectors are immutable).
- A Service matching pod labels (headless for StatefulSet); skip only for listenerless Jobs.
- Image pinned to a tag/digest — **never `:latest`**.
- Dedicated ServiceAccount; `automountServiceAccountToken: false` unless the pod calls the API.
- Config/secrets out of the image; prefer file mounts; never emit plaintext secret values.
- `terminationGracePeriodSeconds` + `preStop` drain sleep for serving workloads.
- HPA (`autoscaling/v2`) ONLY when scaling is requested — and then REMOVE `replicas` from the workload.
- Offer default-deny + DNS-allow NetworkPolicy; include it when isolation/multi-tenancy is mentioned.
- GKE target: Workload Identity SA annotation + node selector hints ([[gke-master]]).

## Reject in your own output
`:latest`; missing requests/limits; naked Pods; secrets in env/Git; liveness probing a dependency;
`replicas` + HPA together; Service selector ≠ pod labels; default-deny egress with no DNS allow;
single-replica "HA" / no PDB / no spread; unverified apiVersion or invented field.

## Always close with
A `kubectl apply --dry-run=server` + `kubectl explain` validation note, plus the list of placeholders
(image digest, namespace, storage class, GKE GSA) and assumptions the user must confirm.

**Version note (2026):** stable apiVersions (`apps/v1`, `batch/v1`, `policy/v1`, `networking.k8s.io/v1`,
`autoscaling/v2`, `v1`) are long-stable but confirm against the target cluster. Flag HPA `behavior`,
topology `matchLabelKeys`, native sidecars, PDB `unhealthyPodEvictionPolicy`, and all CRDs (JobSet/LWS) for
`kubectl explain` verification. Never invent an apiVersion, field, or flag.
