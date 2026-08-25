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
