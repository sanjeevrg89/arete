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

---

# Reference — k8s-manifest-scaffolder

# K8s Manifest Scaffolder — generation procedure + templates

This is a **generation procedure**, not a Kubernetes survey. For the *why* behind each decision and for
debugging, see [[kubernetes-expert]]. Here you turn a short spec into correct, production-grade manifests
by following the steps and filling the templates. Every template already has the safe defaults baked in —
that bakery is the value of this skill. Do not emit a bare skeleton and tell the user to harden it later;
emit the hardened artifact.

## Overview / mental model

A Kubernetes manifest is a desired-state document a controller reconciles toward. "Production-grade" is a
fixed set of properties the document must have *by construction*: it schedules deterministically (requests),
survives node loss (controller + spread + PDB), serves traffic only when ready (probes), terminates cleanly
(grace + preStop), and cannot be trivially abused (hardened securityContext, optional default-deny). Your
job is a deterministic transform: **short spec → those properties, materialized in the right workload kind.**

Generate the *minimum complete set* of objects for the workload, never just the workload object:
- a workload (Deployment / StatefulSet / Job / CronJob / DaemonSet / JobSet / LeaderWorkerSet),
- a Service (unless it's a pure batch Job with no listener),
- a PodDisruptionBudget (for any multi-replica long-running workload),
- an HPA (only when scaling is requested),
- optionally a default-deny + DNS-allow NetworkPolicy,
- optionally a ConfigMap / Secret stub and a dedicated ServiceAccount.

## Step 0 — Intake (what to resolve before generating)

Resolve each field below. If the spec states it, use it. If not, apply the **safe default** and record it
as an inline `# assumption:` comment in the emitted YAML. Only ask the user when a field is both unstated
*and* materially changes the output (e.g. stateful vs. stateless, exposed publicly vs. cluster-internal).

| Field | What to determine | Safe default if unstated |
|---|---|---|
| **Workload kind** | stateless / stateful / batch / scheduled / per-node / multi-host | Deployment |
| **Name / namespace** | object name, target namespace | name from spec; `namespace: default` (flag it) |
| **Replicas** | desired count (or HPA min/max) | 3 for serving; 1 for batch |
| **Image** | registry/repo + immutable tag or digest | `REPLACE_ME/<name>:vX.Y.Z` placeholder, never `:latest` |
| **Ports** | container port(s) + protocol + name | one named `http` on 8080 if a listener exists |
| **Resources** | CPU/mem requests; memory limit | `requests cpu 250m / mem 256Mi`, `limit mem 256Mi` |
| **Config** | env vars vs. ConfigMap file mount | ConfigMap mounted as a file when "config" is mentioned |
| **Secrets** | which secret keys, env vs. mount | Secret mounted as a file (never plaintext, never in Git) |
| **Storage** | persistent? size? access mode? | none (Deployment); `volumeClaimTemplates` 10Gi RWO (StatefulSet) |
| **Scaling** | autoscale? on what metric? min/max | none unless requested; CPU 70% util, min 3 / max 10 if requested |
| **Exposure** | cluster-internal / Ingress / LoadBalancer | ClusterIP Service |
| **Isolation** | NetworkPolicy / multi-tenant namespace | offer default-deny block; include if isolation mentioned |
| **Platform** | vanilla K8s / GKE / Autopilot | vanilla; add GKE notes only if target is GKE |
| **Probes** | health endpoints / commands | `/readyz`, `/livez` HTTP probes; or a TCP/exec fallback |

## Step 1 — Choose the workload kind

Decide once; it drives the whole template choice.

| Spec signal | Kind | apiVersion / kind | Key extras to generate |
|---|---|---|---|
| stateless service, request/response | **Deployment** | `apps/v1` / `Deployment` | Service, PDB, HPA?, spread |
| needs stable identity OR per-pod disk OR ordered start (DBs, queues, quorum) | **StatefulSet** | `apps/v1` / `StatefulSet` | **headless** Service, `volumeClaimTemplates`, PDB |
| run-to-completion task | **Job** | `batch/v1` / `Job` | `backoffLimit`, `completions`/`parallelism`; usually no Service |
| run on a schedule | **CronJob** | `batch/v1` / `CronJob` | `schedule`, `concurrencyPolicy`, `jobTemplate` |
| one pod per node (agents, log shippers) | **DaemonSet** | `apps/v1` / `DaemonSet` | tolerations, `updateStrategy: RollingUpdate` |
| multi-host distributed training | **JobSet** | `jobset.x-k8s.io/v1alpha2` / `JobSet` | replicated Jobs + headless svc — see [[jobset-leaderworkerset]] |
| multi-host inference (sharded model) | **LeaderWorkerSet** | `leaderworkerset.x-k8s.io/v1` / `LeaderWorkerSet` | leader+worker templates, size — see [[jobset-leaderworkerset]] |

Decision rule of thumb: if the workload has *no* durable per-pod state and *no* identity requirement →
Deployment. Reach for StatefulSet only when you can name a reason (storage, identity, ordering). Reach for
JobSet/LeaderWorkerSet only for genuinely multi-host single logical jobs.

## Step 2 — Fill the template

Pick the matching template below and substitute the intake values. Keep the structure intact; the safe
defaults are not optional. Use the recommended labels on every object:

```yaml
labels:
  app.kubernetes.io/name: <name>
  app.kubernetes.io/part-of: <system>
  app.kubernetes.io/version: "<image-tag>"
  app.kubernetes.io/managed-by: kustomize   # or helm / argocd, per the target
```

The `selector.matchLabels` MUST be a stable subset (use only `app.kubernetes.io/name`) and is **immutable**
after creation — never include the version label in the selector.

### Template A — Deployment (stateless) + Service + PDB [+ HPA] [+ NetworkPolicy]

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: <name>
  namespace: <namespace>
  labels: { app.kubernetes.io/name: <name>, app.kubernetes.io/part-of: <system>, app.kubernetes.io/version: "<tag>" }
spec:
  replicas: <N>                       # OMIT this line entirely if you emit an HPA below
  revisionHistoryLimit: 5
  selector:
    matchLabels: { app.kubernetes.io/name: <name> }   # immutable
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxUnavailable: 0, maxSurge: 1 }  # never drop below desired capacity
  template:
    metadata:
      labels: { app.kubernetes.io/name: <name>, app.kubernetes.io/version: "<tag>" }
    spec:
      serviceAccountName: <name>
      automountServiceAccountToken: false              # set true ONLY if the pod calls the K8s API
      terminationGracePeriodSeconds: 45                # >= longest in-flight request + preStop sleep
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
        seccompProfile: { type: RuntimeDefault }
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector: { matchLabels: { app.kubernetes.io/name: <name> } }
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector: { matchLabels: { app.kubernetes.io/name: <name> } }
      containers:
        - name: <name>
          image: <registry>/<repo>:<tag>               # pin a tag or @sha256 digest; NEVER :latest
          ports: [{ name: http, containerPort: <port> }]
          envFrom:
            - configMapRef: { name: <name>-config }     # include only if config requested
          env:
            - name: <SECRET_ENV>                        # prefer file mount below; env shown for completeness
              valueFrom: { secretKeyRef: { name: <name>-secret, key: <key> } }
          resources:
            requests: { cpu: "<cpu>", memory: "<mem>" }
            limits:   { memory: "<mem>" }               # memory limit == request; CPU limit omitted deliberately
          readinessProbe:
            httpGet: { path: /readyz, port: http }
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:                                # cheap self-check; must NOT touch dependencies
            httpGet: { path: /livez, port: http }
            periodSeconds: 10
            failureThreshold: 6
          startupProbe:                                 # generous budget for slow boot
            httpGet: { path: /livez, port: http }
            periodSeconds: 5
            failureThreshold: 30
          lifecycle:
            preStop: { exec: { command: ["/bin/sh", "-c", "sleep 10"] } }  # drain while endpoints propagate
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
          volumeMounts:
            - { name: tmp, mountPath: /tmp }            # writable scratch since root FS is read-only
            - { name: secret, mountPath: /etc/secret, readOnly: true }  # include if secrets requested
      volumes:
        - name: tmp
          emptyDir: {}
        - name: secret
          secret: { secretName: <name>-secret }
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: <name>, namespace: <namespace> }
spec:
  minAvailable: <N-1>                                   # or maxUnavailable: 1; keep capacity during drains
  selector: { matchLabels: { app.kubernetes.io/name: <name> } }
---
apiVersion: v1
kind: Service
metadata: { name: <name>, namespace: <namespace>, labels: { app.kubernetes.io/name: <name> } }
spec:
  selector: { app.kubernetes.io/name: <name> }         # MUST match pod labels or endpoints are empty
  ports: [{ name: http, port: 80, targetPort: http }]
```

### Template A-scale — HPA (emit only when scaling is requested)

When you emit this, **delete `spec.replicas` from the Deployment** so the HPA is the sole owner; otherwise
they fight. See [[autoscaling-kubernetes]] for VPA/KEDA/custom-metric variants.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: <name>, namespace: <namespace> }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: <name> }
  minReplicas: <min>
  maxReplicas: <max>
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }   # % of the CPU *request*
  behavior:                                                       # tame flapping (optional but recommended)
    scaleDown:
      stabilizationWindowSeconds: 300
```

HPA on CPU utilization REQUIRES a CPU `request` on the container and `metrics-server` installed — note both.

### Template A-netpol — default-deny + DNS allow (opt-in isolation block)

Requires a CNI that enforces NetworkPolicy (Calico, Cilium, GKE Dataplane V2). Once you deny egress you
MUST re-allow DNS or every in-cluster lookup breaks.

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny, namespace: <namespace> }
spec:
  podSelector: {}                       # every pod in the namespace
  policyTypes: ["Ingress", "Egress"]    # no rules -> nothing allowed
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-dns, namespace: <namespace> }
spec:
  podSelector: {}
  policyTypes: ["Egress"]
  egress:
    - to: [{ namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: kube-system } } }]
      ports: [{ protocol: UDP, port: 53 }, { protocol: TCP, port: 53 }]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: <name>-allow, namespace: <namespace> }
spec:
  podSelector: { matchLabels: { app.kubernetes.io/name: <name> } }
  policyTypes: ["Ingress"]
  ingress:
    - from: [{ namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: <ingress-ns> } } }]
      ports: [{ protocol: TCP, port: <port> }]
```

### Template B — StatefulSet + headless Service + volumeClaimTemplates + PDB

```yaml
apiVersion: v1
kind: Service
metadata: { name: <name>, namespace: <namespace>, labels: { app.kubernetes.io/name: <name> } }
spec:
  clusterIP: None                       # headless -> stable per-pod DNS: <name>-0.<name>.<ns>.svc.cluster.local
  selector: { app.kubernetes.io/name: <name> }
  ports: [{ name: <portname>, port: <port>, targetPort: <portname> }]
---
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: <name>, namespace: <namespace>, labels: { app.kubernetes.io/name: <name> } }
spec:
  serviceName: <name>                   # the headless Service above
  replicas: <N>
  podManagementPolicy: OrderedReady     # Parallel only if pods are independent
  updateStrategy: { type: RollingUpdate }
  selector: { matchLabels: { app.kubernetes.io/name: <name> } }
  template:
    metadata: { labels: { app.kubernetes.io/name: <name> } }
    spec:
      terminationGracePeriodSeconds: 60
      securityContext: { runAsNonRoot: true, runAsUser: 10001, fsGroup: 10001, seccompProfile: { type: RuntimeDefault } }
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector: { matchLabels: { app.kubernetes.io/name: <name> } }
      containers:
        - name: <name>
          image: <registry>/<repo>:<tag>
          ports: [{ name: <portname>, containerPort: <port> }]
          resources:
            requests: { cpu: "<cpu>", memory: "<mem>" }
            limits:   { memory: "<mem>" }
          readinessProbe:
            exec: { command: [<readiness-check>] }    # e.g. ["pg_isready","-q"]; or httpGet/tcpSocket
            periodSeconds: 10
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: { drop: ["ALL"] }
            # NOTE: many stateful engines write to their data dir -> readOnlyRootFilesystem often false here;
            #       set it true + add an emptyDir for any other writable path when the engine allows.
          volumeMounts: [{ name: data, mountPath: <data-path> }]
  volumeClaimTemplates:                 # one durable PVC per pod: data-<name>-0, data-<name>-1, ...
    - metadata: { name: data }
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: <sc>          # prefer a WaitForFirstConsumer class for zonal correctness
        resources: { requests: { storage: <size> } }
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: <name>, namespace: <namespace> }
spec:
  maxUnavailable: 1                     # keep quorum during drains
  selector: { matchLabels: { app.kubernetes.io/name: <name> } }
```

Deleting a StatefulSet does NOT delete its PVCs (data safety). Most block storage is `ReadWriteOnce` —
design per-pod volumes, not shared RWX.

### Template C — Job (run-to-completion)

```yaml
apiVersion: batch/v1
kind: Job
metadata: { name: <name>, namespace: <namespace>, labels: { app.kubernetes.io/name: <name> } }
spec:
  backoffLimit: 4                       # retries before the Job is marked Failed
  completions: 1                        # set >1 + parallelism for indexed/parallel work
  # completionMode: Indexed             # for parallel sharded work; pods get JOB_COMPLETION_INDEX
  activeDeadlineSeconds: 3600           # wall-clock cap so a hung job can't run forever
  ttlSecondsAfterFinished: 86400        # auto-clean finished Jobs
  template:
    metadata: { labels: { app.kubernetes.io/name: <name> } }
    spec:
      restartPolicy: Never              # Never or OnFailure (NOT Always) for Jobs
      securityContext: { runAsNonRoot: true, runAsUser: 10001, seccompProfile: { type: RuntimeDefault } }
      containers:
        - name: <name>
          image: <registry>/<repo>:<tag>
          command: [<cmd>]
          resources:
            requests: { cpu: "<cpu>", memory: "<mem>" }
            limits:   { memory: "<mem>" }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
```

For **CronJob**, wrap this template's pod spec in `spec.jobTemplate.spec` under
`apiVersion: batch/v1` / `kind: CronJob` with `schedule: "<cron>"`,
`concurrencyPolicy: Forbid`, `startingDeadlineSeconds`, and `successfulJobsHistoryLimit` /
`failedJobsHistoryLimit`. Batch Jobs usually need no Service or PDB.

### Template D — multi-host (pointer)

For JobSet (multi-host training) and LeaderWorkerSet (multi-host inference), do not hand-roll — generate
from the canonical templates in [[jobset-leaderworkerset]]. apiVersions: `jobset.x-k8s.io/v1alpha2` and
`leaderworkerset.x-k8s.io/v1` (both CRDs — confirm the installed version with `kubectl explain`). A minimal
LWS sketch is in `examples.md`. For the serving container args (vLLM/SGLang/Triton) pull from
[[serving-frameworks]].

## Step 3 — Apply safe defaults (the always-on baked-in set)

After filling a template, verify EVERY item below is present in the emitted output. These are not optional;
they are what makes the artifact production-grade. If you intentionally omit one, leave a comment saying why.

- **Resources:** `requests` for CPU+memory on every container; memory `limit == request`. CPU limit only
  if isolation is explicitly required (it causes throttling).
- **Probes:** `readinessProbe` always; `livenessProbe` that checks *only the process itself* (never a
  dependency); `startupProbe` for anything with a slow boot.
- **securityContext:** pod-level `runAsNonRoot`, `runAsUser`, `fsGroup`, `seccompProfile: RuntimeDefault`;
  container-level `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true` (+ emptyDir for scratch),
  `capabilities.drop: ["ALL"]`. No `privileged`, `hostNetwork`, `hostPID`, or host path mounts on app pods.
- **HA:** `topologySpreadConstraints` over zone (DoNotSchedule) and node (ScheduleAnyway); a
  PodDisruptionBudget for any multi-replica long-running workload.
- **Labels/annotations:** the `app.kubernetes.io/*` recommended set on every object; selector uses only the
  stable `name` label.
- **Service:** present and selector-matched (headless for StatefulSet). Skip only for listenerless Jobs.
- **Image:** pinned tag or digest; `:latest` is banned.
- **ServiceAccount:** dedicated SA; `automountServiceAccountToken: false` unless the pod calls the API.
- **Config/secrets:** out of the image; prefer file mounts over env; never emit plaintext secret values.
- **Graceful termination:** `terminationGracePeriodSeconds` + `preStop` drain sleep for serving workloads.
- **Scaling:** HPA only when requested, and then `replicas` removed from the workload.
- **GKE add-ons (only if target is GKE):** Workload Identity SA annotation
  `iam.gke.io/gcp-service-account: <gsa>@<project>.iam.gserviceaccount.com` on the ServiceAccount; node
  selection via `cloud.google.com/gke-nodepool` or accelerator labels; on Autopilot drop manual
  `topologySpreadConstraints`/DaemonSets that Autopilot manages. See [[gke-master]].

## Step 4 — Emit + validation note

Emit the manifests as a single YAML stream (objects separated by `---`), ordered: ServiceAccount →
ConfigMap/Secret → workload → Service → PDB → HPA → NetworkPolicy. Then ALWAYS append a validation note:

```
# Validate before applying (does NOT mutate the cluster):
kubectl apply --dry-run=server -f manifests.yaml      # server-side: schema + admission + defaulting
kubectl apply --dry-run=client -f manifests.yaml      # offline schema check if no cluster handy

# Confirm version-sensitive fields exist on the TARGET cluster:
kubectl explain deployment.spec.template.spec.topologySpreadConstraints
kubectl explain hpa.spec.behavior
kubectl explain leaderworkerset.spec    # CRD: only if installed

# Then apply for real and watch rollout:
kubectl apply -f manifests.yaml
kubectl rollout status deployment/<name> -n <namespace>
```

Tell the user which placeholders they must replace (image digest, namespace, storage class, ingress
namespace, GKE GSA) and which assumptions you baked in.

## Version awareness (flag, don't fabricate)

Kubernetes moves fast (it is 2026). The stable apiVersions used in the templates above
(`apps/v1`, `batch/v1`, `policy/v1`, `networking.k8s.io/v1`, `autoscaling/v2`, core `v1`) are long-stable,
but **always confirm against the target cluster** and flag these in emitted output:
- HPA `spec.behavior` and container-resource metric types — `autoscaling/v2` is GA but verify metric shapes.
- `NetworkPolicy` egress/`endPort`, `matchLabelKeys` on topology spread, native sidecars
  (`initContainers` with `restartPolicy: Always`), and PodDisruptionBudget `unhealthyPodEvictionPolicy` —
  availability varies by version/feature gate. Verify with `kubectl explain`.
- CRDs (JobSet, LeaderWorkerSet) — apiVersion tracks the installed controller; never assume it's present.
Never invent an apiVersion, field, or flag you are unsure of — emit it and tell the user to verify.

## Rationalizations & rebuttals

- *"It's just a scaffold, the user will harden it."* No — the value of this skill is that the emitted
  artifact is already hardened. A skeleton that gets applied unhardened is the failure mode you exist to
  prevent.
- *"Skip probes, the app starts fast."* Readiness gates traffic and the rollout; without it a roll sends
  traffic to a not-ready pod. Always emit readiness.
- *"One replica is fine for now."* Then there is no HA and a node drain is an outage. Default to 3 +
  spread + PDB; let the user explicitly downgrade.
- *"Set replicas AND an HPA."* They fight; the next reconcile flaps the count. HPA owns replicas — remove
  the field.
- *"`:latest` is convenient."* It breaks rollbacks and reproducibility and silently changes under you. Pin.
- *"Secrets in env are easier."* They leak to logs/child processes and `kubectl describe`. Mount as files.
- *"readOnlyRootFilesystem breaks the app."* Usually only `/tmp` or a cache dir needs writing — add an
  emptyDir mount, keep the root read-only.

## Red flags (stop and reconsider the generation)

- You generated a Deployment but the spec needs stable identity, ordered start, or per-pod disk → StatefulSet.
- The Service selector doesn't match the pod template labels → endpoints will be empty.
- You set both `replicas` and an HPA → remove `replicas`.
- A `livenessProbe` calls a database or downstream API → it will cause cascading restarts; make it self-only.
- No PDB on a multi-replica serving workload → node upgrades can take it to zero.
- You emitted a NetworkPolicy denying egress but no DNS allow → in-cluster name resolution breaks.
- You used `:latest`, or an apiVersion you didn't verify, or invented a field → stop and check `kubectl explain`.
- A "stateful" workload uses a shared RWX PVC across pods → most block storage is RWO; use volumeClaimTemplates.

## Verification gate (definition of done for a generation)

A generation is complete only when ALL hold:
1. The right workload kind was chosen and justified for the spec.
2. The minimum complete object set is emitted (workload + Service + PDB [+ HPA] [+ NetworkPolicy] [+ config]).
3. Every container has requests+limits, readiness (+liveness/startup where apt), and a hardened
   securityContext; the pod is non-root.
4. Multi-replica workloads have spread + PDB + recommended labels; selectors match pod labels.
5. Image is pinned (no `:latest`); secrets are mounted, not plaintext; SA token automount is off unless needed.
6. If scaling was requested: an `autoscaling/v2` HPA is emitted and `replicas` is removed from the workload.
7. The output ends with the `kubectl apply --dry-run=server` + `kubectl explain` validation note and a list
   of placeholders/assumptions for the user to confirm.

## Canonical references

- Kubernetes workloads & API reference: https://kubernetes.io/docs/concepts/workloads/
- Configure probes: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
- Resource management & QoS: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
- Pod Security Standards: https://kubernetes.io/docs/concepts/security/pod-security-standards/
- PodDisruptionBudget: https://kubernetes.io/docs/concepts/workloads/pods/disruptions/
- Topology spread constraints: https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/
- HorizontalPodAutoscaler: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- NetworkPolicy: https://kubernetes.io/docs/concepts/services-networking/network-policies/
- Recommended labels: https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/
- JobSet: https://jobset.sigs.k8s.io/  ·  LeaderWorkerSet: https://github.com/kubernetes-sigs/lws

---

# Worked generations: spec → emitted manifests

Each example shows the **input spec** then the **full emitted YAML** with the safe defaults baked in.
Imitate the structure exactly. Adjust names/images/numbers to the spec; keep probes, limits,
securityContext, spread, PDB, Service, and the closing validation note. Verify version-sensitive fields
against the target cluster with `kubectl explain` (see the guide's version note).

---

## Example A — Stateless API → Deployment + Service + PDB + HPA + NetworkPolicy

**Input spec:** "stateless `orders` API for the `shop` namespace, 3 replicas, listens on 8080, needs a
config and a secret, autoscale on CPU, lock it down with a default-deny network policy."

**Decisions:** stateless → Deployment. Config requested → ConfigMap file-not-needed here so use `envFrom`;
secret requested → mounted as a file. Scaling requested → emit HPA and **remove `replicas`** from the
Deployment (HPA min 3 / max 10, CPU 70%). Isolation requested → include the default-deny block.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: orders
  namespace: shop
  labels: { app.kubernetes.io/name: orders, app.kubernetes.io/part-of: shop }
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: orders-config
  namespace: shop
  labels: { app.kubernetes.io/name: orders }
data:
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"
---
# assumption: secret values are managed out-of-band (sealed-secrets / external-secrets); placeholder only.
apiVersion: v1
kind: Secret
metadata:
  name: orders-secret
  namespace: shop
  labels: { app.kubernetes.io/name: orders }
type: Opaque
stringData:
  DB_PASSWORD: REPLACE_ME      # do NOT commit a real value; inject via your secret manager
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders
  namespace: shop
  labels: { app.kubernetes.io/name: orders, app.kubernetes.io/part-of: shop, app.kubernetes.io/version: "1.0.0" }
spec:
  # replicas intentionally omitted — the HPA below owns the replica count
  revisionHistoryLimit: 5
  selector:
    matchLabels: { app.kubernetes.io/name: orders }   # immutable
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxUnavailable: 0, maxSurge: 1 }
  template:
    metadata:
      labels: { app.kubernetes.io/name: orders, app.kubernetes.io/version: "1.0.0" }
    spec:
      serviceAccountName: orders
      automountServiceAccountToken: false
      terminationGracePeriodSeconds: 45
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
        seccompProfile: { type: RuntimeDefault }
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector: { matchLabels: { app.kubernetes.io/name: orders } }
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector: { matchLabels: { app.kubernetes.io/name: orders } }
      containers:
        - name: orders
          image: REPLACE_ME/orders:1.0.0     # pin a tag or @sha256 digest; NEVER :latest
          ports: [{ name: http, containerPort: 8080 }]
          envFrom:
            - configMapRef: { name: orders-config }
          resources:
            requests: { cpu: "250m", memory: "256Mi" }
            limits:   { memory: "256Mi" }    # memory limit == request; CPU limit omitted (avoid throttling)
          readinessProbe:
            httpGet: { path: /readyz, port: http }
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet: { path: /livez, port: http }   # self-only; must not touch the DB
            periodSeconds: 10
            failureThreshold: 6
          startupProbe:
            httpGet: { path: /livez, port: http }
            periodSeconds: 5
            failureThreshold: 30
          lifecycle:
            preStop: { exec: { command: ["/bin/sh", "-c", "sleep 10"] } }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
          volumeMounts:
            - { name: tmp, mountPath: /tmp }
            - { name: secret, mountPath: /etc/orders, readOnly: true }
      volumes:
        - name: tmp
          emptyDir: {}
        - name: secret
          secret: { secretName: orders-secret }
---
apiVersion: v1
kind: Service
metadata: { name: orders, namespace: shop, labels: { app.kubernetes.io/name: orders } }
spec:
  selector: { app.kubernetes.io/name: orders }
  ports: [{ name: http, port: 80, targetPort: http }]
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: orders, namespace: shop }
spec:
  minAvailable: 2                  # with HPA min 3, keeps >=2 up during voluntary disruptions
  selector: { matchLabels: { app.kubernetes.io/name: orders } }
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: orders, namespace: shop }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: orders }
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }   # requires CPU request + metrics-server
  behavior:
    scaleDown: { stabilizationWindowSeconds: 300 }               # verify field with: kubectl explain hpa.spec.behavior
---
# default-deny + DNS allow + a narrow ingress allow (requires a policy-enforcing CNI)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny, namespace: shop }
spec:
  podSelector: {}
  policyTypes: ["Ingress", "Egress"]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-dns, namespace: shop }
spec:
  podSelector: {}
  policyTypes: ["Egress"]
  egress:
    - to: [{ namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: kube-system } } }]
      ports: [{ protocol: UDP, port: 53 }, { protocol: TCP, port: 53 }]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: orders-allow, namespace: shop }
spec:
  podSelector: { matchLabels: { app.kubernetes.io/name: orders } }
  policyTypes: ["Ingress"]
  ingress:
    - from: [{ namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: ingress-nginx } } }]
      ports: [{ protocol: TCP, port: 8080 }]
```

```
# Validate before applying:
kubectl apply --dry-run=server -f orders.yaml
kubectl explain hpa.spec.behavior        # version-sensitive
# Placeholders to replace: image digest, real secret injection, ingress namespace label.
# Note: HPA owns replicas (no replicas: on the Deployment); CPU HPA needs metrics-server.
```

---

## Example B — Stateful datastore → StatefulSet + headless Service + volumeClaimTemplates + PDB

**Input spec:** "3-node Postgres in the `data` namespace, 50Gi durable storage per pod, stable DNS."

**Decisions:** durable per-pod disk + stable identity → StatefulSet with `volumeClaimTemplates` and a
headless Service. `OrderedReady` for ordered bring-up; `maxUnavailable: 1` PDB to preserve quorum during
drains. `readOnlyRootFilesystem` is left at the default here because the engine writes to its data dir;
the data path is a PVC, and a `run`/socket dir is provided via emptyDir.

```yaml
apiVersion: v1
kind: Service
metadata: { name: pg, namespace: data, labels: { app.kubernetes.io/name: pg } }
spec:
  clusterIP: None                 # headless -> pg-0.pg.data.svc.cluster.local, pg-1.pg..., pg-2.pg...
  selector: { app.kubernetes.io/name: pg }
  ports: [{ name: pg, port: 5432, targetPort: pg }]
---
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: pg, namespace: data, labels: { app.kubernetes.io/name: pg } }
spec:
  serviceName: pg                 # the headless Service above
  replicas: 3
  podManagementPolicy: OrderedReady
  updateStrategy: { type: RollingUpdate }
  selector: { matchLabels: { app.kubernetes.io/name: pg } }
  template:
    metadata: { labels: { app.kubernetes.io/name: pg } }
    spec:
      terminationGracePeriodSeconds: 60
      securityContext:
        runAsNonRoot: true
        runAsUser: 999            # postgres uid in common images; adjust to your image
        fsGroup: 999
        seccompProfile: { type: RuntimeDefault }
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector: { matchLabels: { app.kubernetes.io/name: pg } }
      containers:
        - name: postgres
          image: REPLACE_ME/postgres:16.4    # pin a tag/digest; NEVER :latest
          ports: [{ name: pg, containerPort: 5432 }]
          envFrom:
            - secretRef: { name: pg-secret }  # POSTGRES_PASSWORD etc.; managed out-of-band
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits:   { memory: "1Gi" }       # memory limit == request
          readinessProbe:
            exec: { command: ["pg_isready", "-q", "-U", "postgres"] }
            periodSeconds: 10
            failureThreshold: 3
          livenessProbe:
            exec: { command: ["pg_isready", "-q", "-U", "postgres"] }
            periodSeconds: 15
            failureThreshold: 6
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: { drop: ["ALL"] }
            # readOnlyRootFilesystem omitted: engine writes outside the data PVC; tighten per your image.
          volumeMounts:
            - { name: data, mountPath: /var/lib/postgresql/data }
            - { name: run, mountPath: /var/run/postgresql }
      volumes:
        - name: run
          emptyDir: {}
  volumeClaimTemplates:           # one durable PVC per pod: data-pg-0, data-pg-1, data-pg-2
    - metadata: { name: data }
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd     # prefer a WaitForFirstConsumer class for zonal correctness
        resources: { requests: { storage: 50Gi } }
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: pg, namespace: data }
spec:
  maxUnavailable: 1               # keep quorum during node drains/upgrades
  selector: { matchLabels: { app.kubernetes.io/name: pg } }
```

```
# Validate before applying:
kubectl apply --dry-run=server -f pg.yaml
kubectl explain statefulset.spec.volumeClaimTemplates
# Notes: deleting the StatefulSet does NOT delete the PVCs (data safety).
#        Most block storage is ReadWriteOnce — design per-pod, not shared RWX.
#        Replication/HA between pg-0..2 is the DB engine's job, not Kubernetes'.
```

---

## Example C — Multi-host inference → LeaderWorkerSet (sketch) + headless Service

**Input spec:** "serve a large model that doesn't fit on one host — 2-host tensor-parallel vLLM, 2 such
groups." This is multi-host inference → **LeaderWorkerSet**. Generate from the canonical templates in
[[jobset-leaderworkerset]]; for the vLLM container args see [[serving-frameworks]]. Sketch below — confirm
the CRD apiVersion and the generated pod-label keys with `kubectl explain` before relying on them.

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1     # CRD — verify installed version: kubectl explain leaderworkerset
kind: LeaderWorkerSet
metadata: { name: vllm, namespace: serving, labels: { app.kubernetes.io/name: vllm } }
spec:
  replicas: 2                     # number of leader+worker GROUPS
  leaderWorkerTemplate:
    size: 2                       # pods per group: 1 leader + (size-1) workers = 2-host group
    restartPolicy: RecreateGroupOnPodRestart   # restart the whole group together (verify field name)
    leaderTemplate:
      metadata: { labels: { app.kubernetes.io/name: vllm, role: leader } }
      spec:
        securityContext: { runAsNonRoot: true, runAsUser: 10001, seccompProfile: { type: RuntimeDefault } }
        containers:
          - name: vllm-leader
            image: REPLACE_ME/vllm:0.x        # pin; NEVER :latest
            command: ["/bin/sh", "-c", "vllm serve $MODEL --tensor-parallel-size 2 --port 8000"]  # see [[serving-frameworks]]
            ports: [{ name: http, containerPort: 8000 }]
            resources:
              requests: { cpu: "8", memory: "64Gi", nvidia.com/gpu: "1" }
              limits:   { memory: "64Gi", nvidia.com/gpu: "1" }   # GPU request==limit (extended resource rule)
            readinessProbe:
              httpGet: { path: /health, port: http }
              periodSeconds: 10
              failureThreshold: 30                # generous: model load is slow
            securityContext:
              allowPrivilegeEscalation: false
              capabilities: { drop: ["ALL"] }
    workerTemplate:
      metadata: { labels: { app.kubernetes.io/name: vllm, role: worker } }
      spec:
        securityContext: { runAsNonRoot: true, runAsUser: 10001, seccompProfile: { type: RuntimeDefault } }
        containers:
          - name: vllm-worker
            image: REPLACE_ME/vllm:0.x
            resources:
              requests: { cpu: "8", memory: "64Gi", nvidia.com/gpu: "1" }
              limits:   { memory: "64Gi", nvidia.com/gpu: "1" }
            securityContext:
              allowPrivilegeEscalation: false
              capabilities: { drop: ["ALL"] }
---
apiVersion: v1
kind: Service
metadata: { name: vllm, namespace: serving, labels: { app.kubernetes.io/name: vllm } }
spec:
  selector:
    app.kubernetes.io/name: vllm
    role: leader                  # route client traffic to leaders only
  ports: [{ name: http, port: 8000, targetPort: http }]
```

```
# Validate before applying:
kubectl explain leaderworkerset.spec.leaderWorkerTemplate     # CONFIRM the CRD is installed + field names
kubectl apply --dry-run=server -f vllm-lws.yaml
# Notes: LeaderWorkerSet is a CRD (kubernetes-sigs/lws) — its apiVersion and the exact
#        restart/rollout field names are version-sensitive; verify before relying on them.
#        For multi-host TRAINING (run-to-completion) use JobSet instead — see [[jobset-leaderworkerset]].
#        GPU is an extended resource: request must equal limit, and nodes need the device plugin.
```

For a batch alternative (run-to-completion), use the **Job** template from the guide
(`backoffLimit`, `restartPolicy: Never`, `ttlSecondsAfterFinished`, hardened securityContext, no Service).
