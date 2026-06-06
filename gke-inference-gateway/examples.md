# Examples — Gateway + HTTPRoute + InferencePool + InferenceModel → vLLM

> **READ THIS FIRST — VERIFY API VERSION & FIELDS.** The Gateway API Inference Extension is **young and
> fast-moving (2026)**. The `apiVersion`/group, kind names (`InferencePool`, `InferenceModel` vs
> `InferenceObjective`), and field layouts below are **shape-correct illustrations, not copy-paste truth.**
> Before applying, reconcile against the **target cluster** and the **release's install chart**:
>
> ```bash
> kubectl api-resources | grep -i infer        # discover the real group/version/kinds
> kubectl explain inferencepool                 # confirm the real spec fields
> kubectl explain inferencemodel                # (or inferenceobjective)
> # and read the install chart / project docs for the exact release you deploy
> ```
>
> The Gateway API base objects (`Gateway`, `HTTPRoute`, `GatewayClass`, `ReferenceGrant`) are stable
> upstream and reliable as shown. The **inference** objects are the ones to verify.

Apply order: model-server Deployment → EPP → InferencePool → InferenceModel → Gateway → HTTPRoute.

---

## 1. vLLM model-server Deployment (the backend pool members)

Plain OpenAI-compatible vLLM. Labeled so the `InferencePool` can select it; exposes the OpenAI port
(which also serves Prometheus `/metrics` that the EPP scrapes). Single-host (one GPU) here; for a model
sharded across nodes, replace this with a **LeaderWorkerSet** and select **leaders** (see
`[[jobset-leaderworkerset]]`).

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama3-8b-vllm
  labels: { app: llama3-8b }
spec:
  replicas: 4                          # the fleet the EPP picks among
  selector: { matchLabels: { app: llama3-8b } }
  template:
    metadata:
      labels:
        app: llama3-8b                 # <- InferencePool selects on this
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest        # pin a real digest in production
          args:
            - "--model=meta-llama/Meta-Llama-3-8B-Instruct"
            - "--served-model-name=llama3-8b"   # the OpenAI `model` clients send
            - "--port=8000"
            # LoRA-aware example: enable adapters so many can share these base weights
            - "--enable-lora"
            - "--max-loras=8"
          ports:
            - { name: http, containerPort: 8000 }   # OpenAI API + /metrics
          resources:
            limits: { nvidia.com/gpu: "1" }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
          # in-flight generations are long: give graceful drain room (see guide §8)
      terminationGracePeriodSeconds: 120
```

---

## 2. Endpoint Picker (EPP)

The EPP is best installed from the **project's release manifests / Helm chart** so its picker protocol
and metric expectations match the `InferencePool` CRD version you deploy. Conceptually it is a Deployment
+ Service that the pool references; run it **HA and close to the pods**.

```yaml
# Illustrative shape only — prefer the project's published install chart for the exact release.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama3-8b-epp
spec:
  replicas: 2                          # HA: it is on the request hot path (guide §3, §9)
  selector: { matchLabels: { app: llama3-8b-epp } }
  template:
    metadata: { labels: { app: llama3-8b-epp } }
    spec:
      affinity:                        # spread replicas; keep near the pool
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector: { matchLabels: { app: llama3-8b-epp } }
                topologyKey: kubernetes.io/hostname
      containers:
        - name: epp
          image: <endpoint-picker-image-from-the-release>   # VERIFY
          args:
            - "--pool-name=llama3-8b-pool"
            # scoring/feature flags (queue/KV/prefix/LoRA-aware) are version-specific — VERIFY
          ports:
            - { name: grpc, containerPort: 9002 }   # picker protocol port (VERIFY)
---
apiVersion: v1
kind: Service
metadata: { name: llama3-8b-epp }
spec:
  selector: { app: llama3-8b-epp }
  ports:
    - { name: grpc, port: 9002, targetPort: 9002 }   # VERIFY
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: llama3-8b-epp }
spec:
  minAvailable: 1
  selector: { matchLabels: { app: llama3-8b-epp } }
```

---

## 3. InferencePool (selects pods + references the EPP)

The inference-aware backend. **Verify `apiVersion`/group and every field** against your cluster.

```yaml
# apiVersion is illustrative — confirm with `kubectl api-resources | grep -i infer`.
apiVersion: inference.networking.x-k8s.io/v1alpha2   # <-- VERIFY group AND version
kind: InferencePool
metadata:
  name: llama3-8b-pool
spec:
  # select the model-server pods that form this pool (must match the leader for multi-host)
  selector:
    app: llama3-8b
  # the port on those pods serving the OpenAI API + /metrics
  targetPortNumber: 8000
  # reference to the Endpoint Picker that makes per-request decisions
  extensionRef:                         # field name VERIFY (extensionRef / endpointPickerRef / ...)
    name: llama3-8b-epp                  # the EPP Service above
```

---

## 4. InferenceModel / InferenceObjective (model name, criticality, adapters)

Declares per-model intent. **The kind name and fields vary by release — verify** (it may be
`InferenceModel` or an `InferenceObjective`).

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2   # <-- VERIFY group AND version
kind: InferenceModel                                  # <-- VERIFY kind (may be InferenceObjective)
metadata:
  name: llama3-8b
spec:
  modelName: llama3-8b                  # matches vLLM --served-model-name and clients' `model`
  criticality: Critical                 # enum VERIFY (e.g. Critical / Standard / Sheddable)
  poolRef:
    name: llama3-8b-pool
  # LoRA-adapter-aware routing example: a served model name backed by an adapter on the base pool
  # targetModels / adapter mapping fields are version-specific — VERIFY against the current API.
```

A second `InferenceModel` mapping a *different* served-model name to a LoRA adapter on the **same** pool
is how you serve many adapters on shared base weights (guide §4).

---

## 5. Gateway

Standard upstream Gateway API. The `gatewayClassName` selects the implementation (a managed GKE class, or
an OSS gateway class). This part is stable.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gw
spec:
  gatewayClassName: <your-gateway-class>    # managed GKE class or an OSS gateway class — VERIFY name
  listeners:
    - name: http
      protocol: HTTP
      port: 80
```

---

## 6. HTTPRoute → InferencePool (with a canary split)

The `backendRef` points at the **InferencePool** (not a `Service`). Two pools with weights show a 95/5
canary across model versions; collapse to a single 100% `backendRef` for steady state.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llama3-route
spec:
  parentRefs:
    - name: inference-gw
  rules:
    - matches:
        - path: { type: PathPrefix, value: /v1 }    # OpenAI-compatible surface
      backendRefs:
        # weighted canary across two pools (guide §4, §8). Group/kind VERIFY.
        - group: inference.networking.x-k8s.io       # <-- VERIFY
          kind: InferencePool                         # <-- VERIFY
          name: llama3-8b-pool                        # v1 (stable)
          weight: 95
        - group: inference.networking.x-k8s.io
          kind: InferencePool
          name: llama3-8b-pool-v2                      # v2 (canary)
          weight: 5
```

> Model selection happens on the request **body** `model` field via the `InferenceModel` mapping — clients
> just send the served-model name; they don't pick the pool. For cross-namespace `backendRefs`, add a
> `ReferenceGrant` in the pool's namespace.

---

## 7. Smoke test (OpenAI-compatible)

```bash
GW=$(kubectl get gateway inference-gw -o jsonpath='{.status.addresses[0].value}')
curl -s "http://${GW}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "llama3-8b",
        "messages": [{"role":"user","content":"Say hello in one word."}],
        "max_tokens": 8
      }'
```

If this 404s, the served-model name isn't mapped (check the `InferenceModel`). If it routes but TTFT is
terrible/uneven, the EPP likely isn't scraping pod metrics — check EPP health and the pods' `/metrics`
reachability (guide §10).

---

## Checklist for these manifests

- [ ] Group/version/kind of `InferencePool` and `InferenceModel`/`InferenceObjective` **verified on the
      target cluster** (`kubectl api-resources`, `kubectl explain`) and against the **release chart**.
- [ ] EPP installed from a **matching release**, run **HA** (≥2 + PDB), **near** the pods.
- [ ] Pool selector matches **serving** pods (the **leader** for multi-host LWS).
- [ ] `--served-model-name` == `InferenceModel.modelName` == client `model`.
- [ ] Autoscaling keyed to **KV-cache / queue** signals, not CPU/RPS (`[[autoscaling-kubernetes]]`).
- [ ] Per-model observability (TTFT, KV %, queue, EPP errors) and alerts wired up.
- [ ] Graceful drain (`terminationGracePeriodSeconds`, engine graceful shutdown) for long generations.
