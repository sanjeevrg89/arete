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
