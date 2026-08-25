# AGENTS.md — GKE Inference Gateway / Gateway API Inference Extension

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`gke-inference-gateway-guide.md`** next to this file —
> read it before designing or debugging inference routing, and apply it. A concrete annotated manifest
> set to imitate is in **`examples.md`**. This file is the always-on summary.
>
> **Standard:** the OSS **Gateway API Inference Extension** (`gateway-api-inference-extension`), with the
> **GKE Inference Gateway** as one managed implementation. This API is **young and fast-moving (2026)** —
> verify every kind/group/version/field against the current project and the target cluster
> (`kubectl api-resources | grep -i infer`). Never copy API field names from memory.

## When fronting LLM model-server fleets on Kubernetes, apply these by default:

- **Round-robin/L7 is wrong for LLMs.** Requests vary by orders of magnitude in cost/duration; each
  replica holds a **stateful KV cache**; prefill≠decode; one long request causes head-of-line blocking.
  Routing must be **queue-, KV-cache-, prefix-, and model-aware**, decided per request from live
  backend metrics — not a body-blind hash.
- **The model is Gateway API + an inference extension.** A standard `Gateway` + `HTTPRoute` send traffic
  to an **`InferencePool`** (selects model-server pods) instead of a `Service`. An external **Endpoint
  Picker (EPP)** is consulted **per request** and chooses the specific pod using scraped pod metrics.
  **`InferenceModel`/`InferenceObjective`** declare model name, criticality, and adapter routing.
  **Verify exact kinds/fields.**
- **Smart routing inputs:** queue depth, **KV-cache utilization**, **prefix-cache locality** (route
  shared-prefix requests to the replica that cached them), **LoRA-adapter awareness** (route to a replica
  that has the adapter loaded). Which features are GA vs alpha **moves fast — verify.**
- **Criticality/fairness:** tag traffic (Critical/Sheddable-style) so the EPP sheds/deprioritizes
  low-criticality traffic under saturation instead of degrading everyone.
- **Traffic management is Gateway-native:** canary/A-B/version splits via weighted `HTTPRoute`
  `backendRefs`/pools; model-name and **request-body-based** routing pick pool/adapter; OpenAI-compatible
  surface for clients/SDKs.
- **Backends are unchanged OpenAI-compatible servers** (vLLM/JetStream/Triton/SGLang) exporting
  `/metrics` the EPP scrapes. Multi-host models: select the **LeaderWorkerSet leader** only. Pool signals
  feed autoscaling.
- **Scale on saturation, not count.** Drive HPA/KEDA on **KV-cache utilization / queue depth**, not CPU
  or RPS. Mind multi-minute cold start (weights load).
- **EPP is on the hot path:** run it **HA (≥2 replicas, PDB, anti-affinity), close to the pods**. A slow
  or failing EPP silently degrades routing toward dumb round-robin.
- **Pin one coherent release** for pool CRD + EPP + picker protocol. Version mismatch → routes resolve
  but no endpoint is ever picked. Cross-namespace pool refs need a `ReferenceGrant`.
- **Observe per served model:** TTFT, TPOT/ITL, queue depth, KV-cache %, tokens/sec, adapter/prefix hit
  rate, EPP pick latency & scrape errors. These are also the scaling and rollout signals.
- **Roll out by weight + gate on quality.** Canary a `v2` pool by shifting HTTPRoute weights; promote on
  eval/quality, not just liveness. Drain gracefully — in-flight generations are long.
- **When NOT to use it:** a single small model or uniform/non-LLM traffic — use a plain `Service`/HPA or
  KServe. Reserve the EPP path for fleets where *which* replica serves a request drives cost/tail latency.

## Definition of done for inference-routing changes
- Kinds/group/version/fields verified against the **target cluster** and the **release's install chart**
  (not from memory).
- Route resolves to the **InferencePool**; pool selects the correct (leader) pods; **EPP is HA and
  successfully scraping** pod metrics; served-model name maps correctly; engine exports the needed signals.
- Autoscaling keyed to **KV/queue** signals; per-model observability + alerts (TTFT SLO, KV saturation,
  EPP errors) in place; graceful drain on rollout/scale-down.

## Related skills
`[[serving-frameworks]]` · `[[jobset-leaderworkerset]]` · `[[autoscaling-kubernetes]]` · `[[gke-master]]`
· `[[aiml-on-kubernetes]]` · `[[llm-app-agent-frameworks]]` · `[[ai-security-on-gke]]`
