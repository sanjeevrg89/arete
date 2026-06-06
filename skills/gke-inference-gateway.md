---
name: gke-inference-gateway
description: Expert LLM-aware inference routing on Kubernetes with the Gateway API Inference Extension
  (the OSS project) and the GKE Inference Gateway as a managed implementation. Use when fronting large
  fleets of model servers (vLLM, JetStream/MaxText, Triton, SGLang) and you need model/queue/KV-cache/
  prefix-cache/LoRA-aware request routing instead of round-robin L7 load balancing. Covers InferencePool,
  InferenceModel/InferenceObjective, the Endpoint Picker (EPP) extension, Gateway + HTTPRoute wiring,
  criticality/fairness, traffic splitting and canary across model versions, OpenAI-compatible routing,
  per-model observability (TTFT, queue depth, KV-cache utilization), and when to use it vs a plain
  Service / KServe / generic gateway. Triggers: InferencePool, InferenceModel, Endpoint Picker, EPP,
  inference.networking.k8s.io, body-based routing, smart/inference routing for LLMs on GKE.
---

# GKE Inference Gateway / Gateway API Inference Extension

Apply the judgment of an engineer who runs large model-serving fleets in production: someone who knows
that the bottleneck for LLM serving is the **endpoint picker's per-request decision**, not the L7 proxy,
and who treats KV cache, queue depth, and LoRA adapters as first-class routing inputs.

The standard here is the **Gateway API Inference Extension** (`gateway-api-inference-extension`, an
OSS Kubernetes SIG-Network project). The **GKE Inference Gateway** is one managed implementation of it.
This API is **evolving fast** — always verify the exact API group/version and field names against the
current project/docs before authoring manifests.

## How to use this skill

1. **Read `gke-inference-gateway-guide.md`** in this directory — the full reference (mental model,
   InferencePool/InferenceModel/EPP, smart-routing capabilities, deployment, ops, troubleshooting).
   Apply it to the task.
2. For a concrete, annotated manifest set to imitate (Gateway + HTTPRoute + InferencePool +
   InferenceModel → vLLM), read **`examples.md`**. It carries a "verify API version/fields" caveat.
3. Match the surrounding cluster's conventions (Gateway class, namespaces, naming). Apply the
   correctness/safety rules and the "verify against current docs" discipline regardless.

## The essentials (full detail in `gke-inference-gateway-guide.md`)

- **Round-robin is the wrong default for LLMs.** Requests have wildly variable cost and duration
  (prompt length, output length), each replica holds a **stateful per-replica KV cache**, prefill and
  decode have different profiles, and one long request causes **head-of-line blocking**. Routing must be
  queue-, cache-, and model-aware — a body-blind L7 hash cannot do this. See `[[serving-frameworks]]`.
- **The model is: Gateway API + an inference extension.** A standard `Gateway` and `HTTPRoute` send
  traffic to an **`InferencePool`** (a pool of model-server endpoints) instead of a `Service`. An
  **Endpoint Picker (EPP)** — an external extension the Gateway calls per request — chooses the *specific*
  pod using live metrics. `InferenceModel`/objective objects declare model-level intent (criticality,
  base-model name, adapter routing). **Verify exact kinds/fields against the current project.**
- **Smart routing inputs:** load/queue depth, **KV-cache utilization**, **prefix-cache locality**
  (route same-prefix requests to the replica that already cached it), and **LoRA-adapter awareness**
  (many adapters share base weights; route to a replica that has the adapter loaded).
- **Criticality / fairness:** mark traffic (e.g. Critical vs Sheddable) so the picker sheds or deprioritizes
  low-criticality requests under saturation instead of degrading everything.
- **Traffic management is Gateway-native:** canary/A-B/version splits via `HTTPRoute` weights and
  `backendRefs` across pools; model-name and request-**body-based** routing select the pool/adapter.
- **Backends are unchanged model servers** (vLLM/JetStream/Triton) exposing OpenAI-compatible endpoints
  and a metrics endpoint the EPP scrapes. Multi-host models pair with LeaderWorkerSet
  (`[[jobset-leaderworkerset]]`); pool metrics feed autoscaling (`[[autoscaling-kubernetes]]`).
- **Observe per model, not per pool:** TTFT, TPOT/ITL, queue depth, KV-cache utilization, tokens/sec,
  adapter hit rate. These are also your scaling and rollout signals.
- **When to use it:** many replicas of expensive LLM servers where placement matters. For a single
  small model or non-LLM service, a plain `Service`/HPA or KServe is simpler. See guide's comparison.
- **Reserve the LLM-aware path for LLM traffic.** Don't route generic REST through the EPP — you add a
  per-request hop for no benefit.

## Related skills

- `[[serving-frameworks]]` — the engine side: vLLM/SGLang/Triton, continuous batching, KV cache, LoRA.
- `[[jobset-leaderworkerset]]` — multi-host serving (a model sharded across pods) behind a pool.
- `[[autoscaling-kubernetes]]` — HPA/KEDA/custom metrics driven by pool/queue/KV signals.
- `[[gke-master]]` — GKE platform: GPU/TPU node pools, Gateway controller, networking.
- `[[aiml-on-kubernetes]]` — the umbrella for training+inference on K8s/GKE.
- `[[llm-app-agent-frameworks]]` — clients/agents calling the OpenAI-compatible endpoint in front.
- `[[ai-security-on-gke]]` — auth, model armor/safety, tenant isolation in front of the gateway.

---

# Reference — gke-inference-gateway

# GKE Inference Gateway / Gateway API Inference Extension — Deep Reference

> The standard is the **Gateway API Inference Extension** (`gateway-api-inference-extension`), an OSS
> Kubernetes SIG-Network project that extends the Kubernetes **Gateway API** with LLM-aware routing.
> The **GKE Inference Gateway** is a managed implementation of this standard on GKE. This API is young
> and **moving fast (it is 2026)** — treat every kind name, API group/version, and field below as
> *shape-correct but verify-against-current-docs*. Where this guide says "verify," it means the project
> may have renamed or restructured it since.

---

## 1. Why generic L7 load balancing is wrong for LLM inference

A normal HTTP load balancer (round-robin, least-connections, consistent-hash on a header) assumes
requests are roughly fungible: similar cost, short and bounded duration, stateless backends. **None of
those assumptions hold for LLM inference**, and getting them wrong wastes accelerators and blows up tail
latency.

- **Request cost varies by orders of magnitude.** Cost ≈ prefill (∝ input tokens) + decode (∝ output
  tokens). A 50-token prompt with a 20-token answer and a 100k-token prompt streaming a 4k-token answer
  hit the same endpoint very differently. Round-robin spreads *count*, not *work*, so some replicas
  saturate while others idle.
- **Duration is long and variable.** A single decode can run for many seconds. A connection-count or
  least-request balancer can't see that a replica is mid-way through ten expensive generations.
- **Backends are stateful: the per-replica KV cache.** Each model-server replica holds a **KV cache** in
  HBM keyed to the tokens it has processed. Two facts follow: (a) a replica's *spare capacity* is really
  "free KV-cache blocks," not CPU; (b) sending a request to a replica that already cached its **prefix**
  (system prompt, few-shot preamble, conversation history) skips recomputation. A body-blind balancer
  throws away both signals. See `[[serving-frameworks]]` for KV cache, paged attention, prefix caching.
- **Prefill vs decode are different workloads.** Prefill is compute-bound and bursty; decode is
  memory-bandwidth-bound and steady. Disaggregated setups run them on different replicas/pools, and the
  router must understand the split. Even in a fused server, a replica busy prefilling a huge prompt is a
  bad target for a latency-sensitive short request.
- **Head-of-line blocking.** Continuous batching helps, but a replica's queue and batch are finite.
  Routing a Critical short request behind a Sheddable 8k-token generation tanks its TTFT. The router must
  see **queue depth** and **criticality**, not just "is the pod Ready."

Net: **routing must be model-, queue-, and cache-aware**, made *per request* from *live backend
telemetry* — exactly what a stock L4/L7 balancer cannot do. That decision is the whole point of the
inference extension.

---

## 2. Mental model: Gateway API + an inference extension

The design deliberately **reuses the Kubernetes Gateway API** and adds an inference-aware layer rather
than inventing a new proxy.

```
                client (OpenAI-compatible request)
                          │
                          ▼
   ┌─────────────────────────────────────────────┐
   │  Gateway (GatewayClass = a Gateway impl)      │   L7 proxy / managed LB
   │   └─ HTTPRoute  (host/path/header/body match) │   data plane
   │         backendRef ─▶ InferencePool ──────────┼──┐
   └─────────────────────────────────────────────┘  │
                          │ per-request callout       │ selects pods (label selector)
                          ▼ (ext-proc / picker API)   ▼
   ┌──────────────────────────┐          ┌──────────────────────────────┐
   │  Endpoint Picker (EPP)    │ scrapes  │  Model-server pods (the pool) │
   │  - reads pool metrics     │◀────────▶│  vLLM / JetStream / Triton    │
   │  - reads request body     │  metrics │  OpenAI-compatible + /metrics │
   │  - picks ONE endpoint     │          │  each holds its own KV cache  │
   └──────────────────────────┘          └──────────────────────────────┘
```

- **Gateway + GatewayClass + HTTPRoute** — unchanged upstream Gateway API. The `GatewayClass` names the
  implementation (a managed GKE Gateway controller, or an OSS one such as an Envoy-based gateway). The
  `HTTPRoute` matches traffic (host, path, header, and — key for inference — **request body** fields like
  `model`) and forwards to a backend.
- **InferencePool** — the inference-aware *backend*. Instead of pointing the `HTTPRoute` at a `Service`,
  you point it at an `InferencePool`, which selects a set of model-server pods (by label) and references
  the **endpoint picker** to use. Think of it as "a Service that knows it fronts model servers and
  delegates endpoint choice to a smart picker."
- **Endpoint Picker (EPP)** — an external component the Gateway consults **per request** (over an
  ext-proc-style picker protocol). The Gateway hands the EPP the request (headers + body) and the set of
  candidate endpoints; the EPP returns the chosen endpoint. The EPP continuously **scrapes the pods'
  metrics** (queue depth, KV-cache utilization, loaded LoRA adapters) to make that choice. This is where
  all the "smart" lives; the proxy stays dumb and fast.
- **InferenceModel / InferenceObjective (model-level intent)** — declares per-model policy: the
  user-facing model name, which base model / pool it maps to, **criticality**, and adapter routing. The
  exact kind name and fields have changed across versions of the project — **verify the current kind**
  (it may be `InferenceModel`, an `InferenceObjective`, or similar) and its API group/version.

> **API group/version:** these objects live under an inference networking API group (commonly seen as
> `inference.networking.x-k8s.io` / `inference.networking.k8s.io` in various releases). **Do not hardcode
> from memory — check `kubectl api-resources | grep -i infer` on the target cluster.**

---

## 3. Core objects (precise, but verify fields)

### InferencePool
The set of endpoints + the picker. Conceptually:

- a **pod selector** (label selector) choosing the model-server replicas;
- a **target port** the model servers listen on;
- a reference to the **endpoint-picker** config/deployment (the `extensionRef` / EPP service);
- it is referenced as a `backendRef` from an `HTTPRoute` (the pool is a Gateway API backend, so RBAC and
  `ReferenceGrant` rules apply across namespaces).

One pool ≈ one "fleet of interchangeable replicas of the same served model (and its adapters)." Use
separate pools for genuinely different backends (different base model, different accelerator, prefill vs
decode in a disaggregated setup).

### InferenceModel / InferenceObjective
Per-model routing/serving intent. Typical concepts (names/fields **vary by version — verify**):

- **model name** — the OpenAI `model` value clients send (`"my-llama-3-8b"`), decoupled from the
  backend deployment name. Lets you rename, version, and A/B without changing clients.
- **target pool(s)** — which `InferencePool`(s) serve this model, optionally with **weights** for
  splitting/canary.
- **criticality / priority** — e.g. `Critical`, `Standard`, `Sheddable` (exact enum **varies — verify**).
  Drives shedding and fairness under load.
- **LoRA / adapter routing** — map a served model name to a base model + adapter, so requests for the
  adapter land on replicas that have it loaded.

### Endpoint Picker (EPP)
A deployment (usually one per pool, or shared) implementing the picker protocol. Responsibilities:

- **scrape** each candidate pod's `/metrics` (queue length, KV-cache utilization %, running/waiting
  requests, loaded adapters, etc. — exact metric names depend on the model server and the EPP version);
- **score** candidates with pluggable signals (queue-aware, KV-cache-aware, prefix-cache-aware,
  LoRA-aware) and **pick one**;
- enforce **criticality/fairness** (shed or deprioritize low-criticality traffic when the pool is hot);
- it is on the **request hot path** — keep it close (same cluster/zone), low-latency, and HA (≥2
  replicas). It must not become the bottleneck it exists to prevent.

---

## 4. Smart-routing capabilities

These are the reasons to adopt the extension. Which are GA vs experimental **moves quickly — verify the
current feature matrix and any tunables before relying on them.**

- **Load / queue-aware selection.** Pick the replica with the lowest pending work (waiting+running
  requests, queue depth) rather than the next in rotation. The direct fix for head-of-line blocking and
  uneven saturation.
- **KV-cache-utilization-aware selection.** Treat free KV-cache blocks as the real capacity signal.
  Avoid replicas near KV exhaustion (which would trigger preemption/recompute or rejects); prefer those
  with headroom. This is the inference-specific notion of "least loaded."
- **Prefix-cache-aware routing.** Route requests that **share a prefix** (system prompt, few-shot
  preamble, a chat session's history) to the replica that already has that prefix in its KV cache,
  skipping prefill recomputation. Large TTFT and throughput wins for chat/RAG/agent workloads with long
  shared preambles. (Engine-side prefix caching is in `[[serving-frameworks]]`; the gateway adds the
  *cross-replica* routing layer that makes locality pay off.)
- **LoRA-adapter-aware routing.** When many fine-tuned **LoRA adapters** share one base model on a pool,
  route a request for adapter X to a replica that already has X loaded (or can load it cheaply), rather
  than forcing a cold adapter swap. Lets you serve hundreds of adapters economically on shared weights.
- **Priority / criticality / fairness.** Tag traffic (per `InferenceModel`/objective). Under saturation
  the picker **sheds or deprioritizes** Sheddable traffic to protect Critical traffic's SLO, instead of
  letting everything degrade. Enables multi-tenant fairness and graceful overload.
- **Traffic splitting / canary / A-B across model versions.** Gateway-native: an `HTTPRoute` with
  weighted `backendRefs` (or weighted targets on the model object) sends e.g. 95% to `v1` pool, 5% to
  `v2` pool. Roll forward by shifting weights; roll back by reverting — no client change.
- **Model-name & body-based routing.** Match on the request **body** (`model` field, and other JSON
  fields where supported) to select pool/adapter. This is what makes a single endpoint serve many models
  through one OpenAI-compatible surface.
- **OpenAI-compatible endpoints.** Clients hit `/v1/chat/completions` etc.; the gateway reads `model`,
  routes, and the backend (vLLM/JetStream/etc.) speaks the OpenAI protocol. Drop-in for existing SDKs and
  agent frameworks (`[[llm-app-agent-frameworks]]`).

---

## 5. Architecture & integration

- **Backends:** any OpenAI-compatible model server that exposes a Prometheus `/metrics` endpoint the EPP
  understands — primarily **vLLM**, also **JetStream** (TPU/MaxText), **Triton/TensorRT-LLM**, **SGLang**.
  The richest signals (KV-cache utilization, per-adapter state, queue) come from servers that export them;
  the EPP's effectiveness is bounded by what the engine reports. See `[[serving-frameworks]]`.
- **Multi-host serving:** a model too big for one node is sharded across pods with **LeaderWorkerSet**
  (`[[jobset-leaderworkerset]]`). The *leader* exposes the OpenAI/metrics endpoint; the `InferencePool`
  selects **leaders**, and the EPP treats each leader as one logical endpoint. Don't select worker pods.
- **Autoscaling:** the pool's aggregate signals (queue depth, KV-cache utilization, tokens/sec) are the
  *right* scaling triggers — far better than CPU. Feed them to HPA/KEDA/custom metrics
  (`[[autoscaling-kubernetes]]`). Scale on **saturation of accelerator/KV**, not request count. Mind cold
  start: model load + weights pull is minutes; over-provision headroom and pre-warm.
- **Platform (GKE):** runs on GKE with GPU/TPU node pools and the managed Gateway controller
  (`[[gke-master]]`). The managed GKE Inference Gateway provisions the data plane (a managed L7 LB) and
  can integrate platform features (e.g. model-safety/Model Armor, authz) in front. **Which platform
  integrations exist and how they attach is fast-moving — verify against current GKE docs.**
- **Security front door** (`[[ai-security-on-gke]]`): authn/authz, rate limiting, and safety filtering
  attach at the Gateway/HTTPRoute layer (policies/filters), *before* the EPP picks an endpoint. Keep the
  pool/EPP on a private network; the public surface is the Gateway only.

---

## 6. Deployment & wiring

Order of objects (full annotated manifests in `examples.md`):

1. **Model-server Deployment(s)** — e.g. vLLM with `--served-model-name`, labeled so the pool can select
   them, exposing the OpenAI port and `/metrics`.
2. **InferencePool** — selects those pods, names the target port, references the **EPP**
   (extension/endpoint-picker).
3. **Endpoint Picker (EPP) Deployment + Service** — the picker the pool references (often installed via
   the project's Helm chart / config so its picker protocol and metrics scraping match the pool version).
4. **InferenceModel / InferenceObjective** — declares served model name(s), criticality, pool target(s),
   adapter routing.
5. **Gateway** — with the right `GatewayClass` (managed GKE class, or an OSS gateway class).
6. **HTTPRoute** — matches traffic and sets `backendRef` to the `InferencePool` (weighted for canary).

Wiring rules:

- **Pin to the project's install manifests/Helm chart for a given release.** The EPP, the pool CRD, and
  the picker protocol version must match. Mismatched versions are the #1 cause of "routes but never picks
  an endpoint."
- `HTTPRoute.backendRefs[].kind` is the **InferencePool** kind (not `Service`). Cross-namespace refs need
  a `ReferenceGrant`.
- Keep **one served-model name per logical model**; version via pools + weights, not by renaming the
  client-facing model.
- Run the **EPP HA** (≥2 replicas, PodDisruptionBudget, anti-affinity) and **close to the pods**.

---

## 7. Observability

Instrument and alert **per served model**, not just per pool. Core signals:

- **TTFT** (time to first token) — dominated by prefill + queueing; your primary latency SLO.
- **TPOT / ITL** (time per output token / inter-token latency) — decode smoothness.
- **Queue depth / waiting requests** per replica and per pool — the saturation leading indicator.
- **KV-cache utilization %** — proximity to preemption/recompute; the real "how full is it."
- **Tokens/sec (prompt and generation)** and **throughput** — capacity and cost tracking.
- **Adapter hit/miss** (LoRA) and **prefix-cache hit rate** — are the smart features actually paying off?
- **EPP health:** picker decision latency, pick errors, scrape failures. If the EPP is slow or failing
  its scrapes, routing silently degrades toward dumb.

Source: model-server `/metrics` (engine-specific), EPP metrics, and Gateway/proxy metrics. Wire into
Prometheus/Managed Prometheus + dashboards; alert on TTFT SLO breach, sustained high KV-cache
utilization, and EPP scrape/pick errors.

---

## 8. Rollout strategy

- **Canary a new model/weights/engine version** by standing up a second `InferencePool` (`v2`) and
  shifting `HTTPRoute` weight 1% → 5% → 25% → 100%, watching per-model TTFT/error rate/quality at each
  step. Roll back by reverting weights — instant, no redeploy.
- **Validate quality, not just liveness.** A new model can be healthy and Ready while producing worse
  answers. Gate promotion on eval/quality metrics, not only HTTP success.
- **Adapter rollouts** are cheaper than base-model rollouts: add a new served-model name mapping to a new
  adapter, canary by weight, retire the old name.
- **Drain gracefully.** In-flight generations are long; on scale-down/rollout, stop sending *new*
  requests to a replica and let it finish (respect the engine's graceful shutdown + a generous
  `terminationGracePeriodSeconds`). Killing a decoding pod drops user requests.

---

## 9. Anti-patterns / gotchas

- **Using it for non-LLM or single-small-model traffic.** The EPP hop earns nothing if requests are
  cheap and uniform. Use a plain `Service`/HPA or KServe. (See §11.)
- **Hardcoding API group/version/kind from memory.** This API is renaming things release to release.
  Always `kubectl api-resources | grep -i infer` and read the chart for the target version.
- **EPP as a single point of failure / single replica.** It's on every request's hot path. One replica,
  no PDB, or co-located with a noisy neighbor → cluster-wide latency. Make it HA and near the pods.
- **Mismatched EPP/pool/picker versions.** Routes resolve but no endpoint is ever picked, or picks are
  random. Install from one coherent release.
- **Selecting worker pods of a multi-host model.** Only the LWS **leader** serves; selecting workers
  routes traffic into the void. Label and select leaders only.
- **Routing on count instead of saturation.** HPA on CPU or RPS for LLM pods is wrong — scale on KV-cache
  utilization / queue depth (`[[autoscaling-kubernetes]]`).
- **Body-based routing without size limits.** Reading the request body on the hot path costs latency and
  is a DoS surface (huge prompts). Bound body size and timeouts at the Gateway.
- **Ignoring prefix-cache locality then blaming the gateway.** If the engine doesn't have prefix caching
  on, or clients randomize prompts, prefix-aware routing can't help. Tune both layers.
- **Treating Ready as routable.** A pod can be Ready while at KV-cache saturation. Trust the live metric,
  not the readiness gate, for placement.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 404 / "model not found" | served-model name not mapped to a pool; client `model` mismatch | Check `InferenceModel`/objective mapping and the client's `model` value |
| Traffic routes but TTFT terrible / uneven | EPP not scraping metrics; falling back to round-robin | Check EPP→pod metrics scrape (network policy, port, metric names); verify EPP healthy |
| No endpoint ever selected / 503 from pool | EPP/pool/picker version mismatch or EPP down | Reinstall from one release; check EPP logs and pick-error metrics |
| Cross-namespace route rejected | missing `ReferenceGrant` for the pool | Add `ReferenceGrant` allowing the route's namespace |
| Multi-host model gets no/half traffic | selecting worker pods, not leaders | Fix pool selector to match LWS leaders only |
| LoRA requests slow / thrash | adapter not loaded on picked replica; adapter-aware routing off or unsupported by engine | Verify engine reports loaded adapters; confirm adapter-aware scoring enabled |
| KV-cache exhaustion / preemptions | over-packing replicas; not scaling on KV signal | Scale pool on KV-cache utilization; lower per-replica concurrency |
| Canary shifts but quality drops | promoting on liveness, not eval | Gate on quality metrics; revert HTTPRoute weights |

General method: confirm **(1)** route resolves to the pool, **(2)** pool selects the right (leader) pods,
**(3)** EPP is up and successfully scraping pod metrics, **(4)** the served-model name maps correctly,
**(5)** the engine exports the signals the EPP needs. Most failures are one of these five.

---

## 11. When to use it vs alternatives

| Option | Use when | Trade-off |
|---|---|---|
| **Plain `Service` + kube-proxy / round-robin** | one small model, uniform cheap requests, single replica | no cache/queue awareness — fine until the fleet/cost grows |
| **Generic Gateway/HTTPRoute (no inference ext)** | you want L7 routing/TLS/header match but requests are uniform | body-blind, no KV/queue/adapter awareness |
| **KServe `InferenceService`** | you want a full serving abstraction: autoscale-to-zero, transformers, predictor/transformer graph, model registry | higher-level/opinionated; routing is not LLM-cost-aware the way the EPP is — can be **combined** behind the gateway |
| **Gateway API Inference Extension / GKE Inference Gateway** | **many replicas of expensive LLM servers** where *which* replica matters; need KV/prefix/LoRA/criticality-aware routing, model-name routing, canary across versions, OpenAI surface | adds the EPP component + a young, fast-moving API to operate |

Rule of thumb: **the more your accelerator cost and tail latency depend on *which* replica serves a
request, the more the inference extension pays for itself.** Below that threshold, simpler is better.

These are complementary, not exclusive — e.g. KServe or a plain Service can sit *behind* an
`InferencePool`, and the gateway fronts the whole `[[aiml-on-kubernetes]]` serving stack.

---

## 12. Version awareness

This project is **pre-1.0-era / fast-moving (2026)**. Specifically verify before relying on:

- **API group/version and kind names** (`InferencePool`, `InferenceModel` vs `InferenceObjective`, the
  `inference.networking.*` group) — these have changed across releases.
- **Which routing features are GA vs alpha** (prefix-cache-aware, LoRA-aware, disaggregated
  prefill/decode support) and their tunables.
- **The EPP picker protocol version** and its compatibility matrix with the Gateway implementation.
- **GKE-specific managed integrations** (data-plane provisioning, safety/Model Armor, authz) — managed
  surfaces move independently of the OSS project; check current GKE docs.

Always reconcile manifests with `kubectl api-resources`/`kubectl explain` on the target cluster and the
release's install chart. Never copy field names from memory.

---

## 13. Canonical references

- Gateway API Inference Extension project: https://github.com/kubernetes-sigs/gateway-api-inference-extension
- Project docs site: https://gateway-api-inference-extension.sigs.k8s.io/
- Kubernetes Gateway API (base): https://gateway-api.sigs.k8s.io/
- GKE Inference Gateway (managed) docs:
  https://cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway
- LeaderWorkerSet (multi-host serving): https://github.com/kubernetes-sigs/lws
- vLLM (backend + metrics): https://docs.vllm.ai/
- KServe (alternative/complementary serving): https://kserve.github.io/website/

Verify each against the current version; URLs and structure evolve.

---

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
