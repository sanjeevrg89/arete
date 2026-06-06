# Autoscaling — Worked Examples

Canonical, imitate-ready manifests. Shapes are correct; **verify apiVersions and provider-specific
fields against your cluster version and provider docs** before applying (this ecosystem moves fast).

---

## 1. HPA with custom + external metrics and `behavior`

A web service scaled on **requests-per-second per pod** (custom metric via Prometheus Adapter) *and*
**SQS backlog** (external metric), with scale-up-fast / scale-down-slow tuning. HPA takes the **max**
desired across the two metrics — either hot signal scales you out.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web
  namespace: shop
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web
  minReplicas: 3
  maxReplicas: 60
  metrics:
    # (a) custom Pods metric — avg requests/sec per pod, served via custom.metrics.k8s.io
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "50"          # target 50 rps/pod
    # (b) external metric — total messages waiting in the order queue
    - type: External
      external:
        metric:
          name: sqs_approx_messages_visible
          selector:
            matchLabels: { queue: orders }
        target:
          type: AverageValue
          averageValue: "30"          # ~30 backlog msgs per pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0   # react immediately to spikes
      selectPolicy: Max
      policies:
        - { type: Percent, value: 100, periodSeconds: 60 }   # ≤ double / min
        - { type: Pods,    value: 4,   periodSeconds: 60 }    # ...or +4 / min
    scaleDown:
      stabilizationWindowSeconds: 300 # consider the max recommendation over 5 min
      selectPolicy: Max
      policies:
        - { type: Percent, value: 25, periodSeconds: 120 }    # shrink gently
```

Notes:
- The custom metric must be registered: `kubectl get --raw
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/shop/pods/*/http_requests_per_second"` should return
  data. Likewise the external one under `/apis/external.metrics.k8s.io/...`.
- `AverageValue` divides the metric by replica count under the hood — `desired = ceil(currentReplicas ×
  current/target)`. Use `Value` (not `Average`) only when the target is an absolute, not per-pod, figure.
- Watch `kubectl describe hpa web` → conditions. `ScalingActive: false`/`unknown` = broken pipeline.

---

## 2. KEDA `ScaledObject` — queue-driven, scale-to-zero

A worker Deployment that sits at **0 replicas** when the queue is empty and bursts on backlog. KEDA
generates the HPA; you still tune HPA `behavior` through `advanced`.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: image-worker
  namespace: media
spec:
  scaleTargetRef:
    name: image-worker            # the Deployment
  minReplicaCount: 0              # scale to zero when idle (KEDA-only capability)
  maxReplicaCount: 100
  idleReplicaCount: 0            # explicit idle floor (optional; must be < minReplicaCount range)
  cooldownPeriod: 300           # seconds idle before scaling back to 0
  pollingInterval: 15           # how often KEDA queries the scaler
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
            - { type: Percent, value: 50, periodSeconds: 60 }
  triggers:
    - type: aws-sqs-queue
      authenticationRef: { name: keda-sqs-auth }
      metadata:
        queueURL: https://sqs.us-east-1.amazonaws.com/123456789012/image-jobs
        awsRegion: us-east-1
        queueLength: "10"            # SCALING target: ~10 msgs per replica (drives 1→N)
        activationQueueLength: "5"   # ACTIVATION: need ≥5 msgs to wake 0→1
```

Notes:
- **Activation vs scaling:** `activationQueueLength` gates `0→1` (so a single stray message doesn't wake
  the fleet); `queueLength` governs `1→N`. They are independent — set activation deliberately.
- For **discrete batch items**, prefer `kind: ScaledJob` (KEDA spawns Jobs sized to the backlog) over a
  long-running Deployment.
- KEDA owns the HPA — don't also create an HPA targeting the same Deployment, and don't pin
  `.spec.replicas` from GitOps (it'll fight KEDA).

---

## 3. Karpenter `NodePool` (+ `EC2NodeClass`) sketch

Just-in-time, cost-aware nodes for a general workload with **spot-first, on-demand fallback** and
consolidation. Provider here is AWS; the `NodeClass` kind/fields are provider-specific.

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: general
spec:
  template:
    metadata:
      labels: { team: platform }
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: default
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]      # try spot, fall back to on-demand
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["m6i.large", "m6i.xlarge", "m6a.large", "m6a.xlarge", "c6i.large"]
      expireAfter: 720h                        # recycle nodes (drift/patching)
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 1m                       # debounce before merging/removing
    budgets:
      - nodes: "10%"                           # never disrupt > 10% of nodes at once
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: default
spec:
  amiFamily: AL2023
  amiSelectorTerms:
    - alias: al2023@latest                     # editing this triggers DRIFT → fleet replace
  subnetSelectorTerms:
    - tags: { karpenter.sh/discovery: my-cluster }
  securityGroupSelectorTerms:
    - tags: { karpenter.sh/discovery: my-cluster }
  role: KarpenterNodeRole-my-cluster
```

Protect a restart-hostile pod from consolidation/drift:

```yaml
metadata:
  annotations:
    karpenter.sh/do-not-disrupt: "true"        # on the Pod / pod template
```

Notes:
- Give Karpenter a **wide instance-type set** — that's the whole point; over-constraining it back to one
  shape recreates Cluster Autoscaler's limitations.
- A `NodeClass` edit (AMI alias, etc.) causes **drift** and can roll the entire fleet — gate it behind
  `budgets` and a maintenance window.
- For GPU/accelerator pools, add the GPU instance families and a corresponding NodePool `weight`/taint;
  pair with `do-not-disrupt` on long-running inference/training pods so consolidation can't evict them.

---

## 4. Anti-pattern: HPA and VPA fighting on CPU (do not do this)

```yaml
# HPA scaling on CPU utilization ...
kind: HorizontalPodAutoscaler
spec:
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
---
# ... AND VPA Auto adjusting CPU requests on the SAME Deployment  → oscillation
kind: VerticalPodAutoscaler
spec:
  updatePolicy: { updateMode: "Auto" }
  resourcePolicy:
    containerPolicies:
      - containerName: '*'
        controlledResources: ["cpu", "memory"]   # CPU here collides with the HPA above
```

**Fix:** let HPA own the horizontal signal (ideally a custom/external metric, not CPU) and restrict VPA
to the dimension HPA doesn't touch — or use GKE Multidimensional Pod Autoscaling (verify availability):

```yaml
# VPA limited to memory only; HPA scales horizontally on RPS / queue depth
spec:
  resourcePolicy:
    containerPolicies:
      - containerName: '*'
        controlledResources: ["memory"]
  updatePolicy:
    updateMode: "Off"        # start in Off, read recommendations, then graduate deliberately
```
