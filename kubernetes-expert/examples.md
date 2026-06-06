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
