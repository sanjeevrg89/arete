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
