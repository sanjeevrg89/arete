---
name: serving-frameworks
description: Expert knowledge of modern LLM/ML inference serving — vLLM (PagedAttention, continuous
  batching), SGLang (RadixAttention), NVIDIA Dynamo (disaggregated prefill/decode, KV-aware routing),
  Triton Inference Server, TensorRT-LLM, Ray Serve, KServe, JetStream (JAX/XLA on TPU), NVIDIA NIM
  (prepackaged inference microservices), and TGI. Use when deploying, tuning, or
  choosing an inference engine; reasoning about prefill vs decode, KV cache / PagedAttention / prefix
  caching, continuous/in-flight batching, TTFT/ITL/TPOT/goodput, tensor/pipeline/expert parallelism,
  multi-host serving, disaggregated serving, speculative/chunked-prefill decoding, fp8/AWQ/GPTQ
  quantization, structured/guided output, or serving autoscaling and KV-cache-aware routing on
  Kubernetes/GKE. Covers engine differentiators, a decision matrix, and K8s deployment shapes.
---

# Serving Frameworks (LLM/ML Inference)

Apply the judgment of an engineer who has run large-scale LLM inference in production for years:
who knows that **throughput, latency, and cost are set far more by KV-cache and batching behavior
than by the model**, and who picks the engine to fit the workload rather than the other way round.

## How to use this skill

1. **Read `serving-frameworks-guide.md`** in this directory — the full reference. It covers the shared
   inference-systems concepts (the backbone) first, then each framework with differentiators, when to
   pick it, and its K8s deployment shape, plus a decision matrix.
2. For concrete manifests to imitate (single-host vLLM `Deployment`, multi-host vLLM on
   LeaderWorkerSet, Triton/TRT-LLM note), read **`examples.md`**.
3. Match the surrounding cluster's conventions (image registry, accelerator class, scheduler/gateway).
   Apply the correctness rules — KV-cache sizing, parallelism topology, health probes — regardless.
4. **These frameworks move very fast (it is 2026).** Name capabilities, but verify current flags,
   defaults, and feature status against the project's own docs/release notes before committing config.
   Never fabricate a config key or a benchmark number.

## Essentials (full detail in `serving-frameworks-guide.md`)

- **Two phases, two bottlenecks.** *Prefill* (process the prompt) is compute-bound and parallel; *decode*
  (generate tokens one at a time) is memory-bandwidth-bound and sequential. They scale differently — this
  asymmetry drives chunked prefill, disaggregation, and most tuning decisions.
- **KV cache dominates GPU memory** after weights. Its size = `2 · layers · kv_heads · head_dim ·
  seq_len · dtype · batch`. It caps how many concurrent sequences you fit, i.e. throughput.
- **PagedAttention (vLLM)** stores KV in fixed-size blocks like OS paging — near-zero fragmentation,
  enabling large effective batch sizes. **RadixAttention (SGLang)** auto-reuses shared prefixes via a
  radix tree. Prefix/prompt caching turns repeated system prompts into a near-free TTFT win.
- **Continuous (in-flight) batching** beats static batching massively for LLMs: sequences join and
  leave the running batch every step, so a finished short request doesn't block the GPU. This is table
  stakes — vLLM/SGLang/TRT-LLM/Triton all do it.
- **Measure goodput, not raw throughput.** Track **TTFT** (time to first token, prefill latency),
  **ITL/TPOT** (inter-token / per-output-token latency, decode speed), and tokens/s under an SLO.
  Throughput-vs-latency is a dial: bigger batches raise throughput and ITL together.
- **Parallelism:** *tensor parallel* (TP) splits each layer across GPUs (needs fast intra-node links,
  e.g. NVLink); *pipeline parallel* (PP) splits layers across stages (tolerates slower links, good
  cross-node); *expert parallel* (EP) shards MoE experts; *data parallel* runs independent replicas.
  Big models that span hosts need multi-host serving → see `[[jobset-leaderworkerset]]` (LWS).
- **Disaggregated serving** (Dynamo / DistServe idea): run separate **prefill** and **decode** pools,
  stream the KV cache between them. Lets each phase scale and batch independently; KV transfer
  bandwidth is the cost. Pick it at datacenter scale, not for a single replica.
- **Quantization** (fp8, int8, AWQ, GPTQ) shrinks weights and KV to fit more on each GPU and speed
  decode; verify accuracy on your eval set. **Speculative decoding** and **chunked prefill** cut
  latency. **Guided/structured decoding** constrains output to JSON/grammar/regex.
- **Engine pick (short form):** single-model max throughput → **vLLM** or **TRT-LLM**; heavy prefix
  reuse / agentic / structured → **SGLang**; multi-framework or non-LLM + LLM → **Triton**; absolute
  GPU peak on NVIDIA with a build step → **TensorRT-LLM**; disaggregated at scale → **Dynamo**;
  Python composition / multi-model pipelines → **Ray Serve**; K8s control plane over any engine →
  **KServe**; JAX/XLA models on **TPU** → **JetStream** (see `[[maxtext-jax-llm]]`); prepackaged
  "buy-not-build" optimized container with OpenAI API → **NVIDIA NIM**. Full rationale and matrix in
  the guide.
- **On Kubernetes:** request the right `nvidia.com/gpu` (or TPU) count, give the model a large
  `emptyDir`/`memory`-backed cache and shared memory (`/dev/shm`), set generous startup probes (model
  load is slow), and route with a KV-/prefix-aware Inference Gateway. Autoscale on queue depth /
  in-flight, not CPU → see `[[autoscaling-kubernetes]]`.

## Related skills

- `[[ml-frameworks]]` — PyTorch/JAX/XLA, GPU & TPU fundamentals underneath every engine.
- `[[jobset-leaderworkerset]]` — JobSet + LeaderWorkerSet for multi-host (multi-pod) serving of one model.
- `[[autoscaling-kubernetes]]` — HPA/KEDA/Karpenter, scale-to-zero and queue-based scaling for serving.
- `[[aiml-on-kubernetes]]` — umbrella for training/inference/RL on K8s & GKE.
- `[[gke-master]]` — GKE accelerator node pools, networking, Inference Gateway, security.
- `[[training-frameworks]]` — the other side: how the checkpoints you serve were produced.

---

# Reference — serving-frameworks

# Serving Frameworks — Deep Reference (LLM/ML Inference)

The full reference for running LLM/ML **inference serving** in production. Read the shared systems
concepts first — they are the backbone and explain *why* every engine makes the choices it does — then
the per-framework sections and the decision matrix. The ecosystem moves very fast (2026); treat
specific flags, defaults, and "supported?" claims as things to verify against current project docs.

---

## 1. Mental model: serving is a KV-cache and batching problem

An autoregressive LLM does two very different kinds of work for one request:

1. **Prefill** — ingest the whole prompt in one forward pass, producing the first token and the
   prompt's KV cache. Lots of parallel matmuls over many tokens at once → **compute-bound**
   (GPU FLOPs / tensor cores are the limit). Latency here = **TTFT** (time to first token).
2. **Decode** — generate output tokens one at a time, each step reading the entire KV cache and the
   weights to produce a single new token. Tiny matmuls, huge memory reads → **memory-bandwidth-bound**
   (HBM bandwidth is the limit), and inherently sequential. Latency here = **ITL / TPOT**
   (inter-token latency / time per output token).

Almost every advanced technique exists to exploit this asymmetry:

- Decode is bandwidth-bound and underutilizes compute, so you **batch many sequences together** to
  amortize the weight read across requests → continuous batching.
- Prefill can starve decode (a long prompt monopolizes the GPU and stalls everyone's token stream),
  so you **chunk prefill** and interleave it with decode steps.
- Prefill and decode want different hardware ratios and batch sizes, so at scale you **disaggregate**
  them into separate pools (Dynamo / DistServe).
- The KV cache is what makes decode cheap (no recompute) but expensive (it must be stored), so KV
  management — paging, prefix reuse, offload, quantization — is the central engineering problem.

**Internalize this:** for a fixed model and hardware, your throughput, latency, and \$/token are
determined mostly by how well the engine manages the KV cache and the running batch.

---

## 2. KV cache: the thing that actually limits you

After model weights, the **KV cache** is the dominant consumer of GPU memory. For every token in
every active sequence you store a key and value vector per layer:

```
kv_bytes ≈ 2 (K and V) · num_layers · num_kv_heads · head_dim · dtype_bytes · seq_len · batch
```

GQA/MQA (grouped/multi-query attention) cut `num_kv_heads` and so cut KV size — a big reason modern
models are servable at long context. The KV cache is what caps **how many concurrent sequences and
how long a context** you can hold, which *is* your throughput ceiling.

### PagedAttention (vLLM's core idea)
Naive engines pre-allocate a contiguous KV region per sequence sized to `max_len`, wasting memory to
internal/external fragmentation and reservation. **PagedAttention** stores KV in fixed-size **blocks**
(e.g. a handful of tokens each) tracked by a block table, exactly like OS virtual-memory paging.
Benefits: near-zero fragmentation, sequences grow block-by-block, and blocks can be **shared** across
sequences (parallel samples, beam search, common prefixes) via copy-on-write. The practical effect is
a much larger *effective* batch size on the same GPU, hence higher throughput.

### RadixAttention (SGLang's core idea)
SGLang keeps cached KV blocks in a **radix tree** keyed by token sequence, so any new request that
shares a prefix with a cached one (same system prompt, same few-shot examples, a tree of agent calls)
**automatically reuses** that KV — no manual cache key. Excellent for agents, multi-turn chat, and
templated prompts.

### Prefix / prompt caching
Repeated prefixes (system prompts, RAG boilerplate, long shared documents) are computed once and
reused, slashing TTFT and prefill cost for the shared part. vLLM exposes automatic prefix caching;
SGLang's RadixAttention does it intrinsically; TRT-LLM and Triton support KV reuse. Order your prompt
template so the stable, shared part is the **prefix**.

### KV offload & quantization
When KV won't fit in HBM you can **offload** colder blocks to CPU RAM (or NVMe) and page them back, or
**quantize** the KV cache (e.g. fp8) to roughly halve its footprint and speed the bandwidth-bound
decode reads. Both trade some accuracy/complexity for capacity. Verify accuracy impact on your evals.

---

## 3. Batching: continuous (in-flight) vs static

**Static batching** waits to assemble a fixed batch, runs all requests to completion together, then
returns — so the whole batch is held hostage by its longest sequence and the GPU idles between batches.
Disastrous for LLMs where output lengths vary wildly.

**Continuous batching** (a.k.a. in-flight / iteration-level batching) operates at the granularity of a
single decode **step**: each iteration, finished sequences leave the batch and waiting ones join.
The GPU stays saturated, short requests aren't blocked by long ones, and effective throughput rises
several-fold. This is **table stakes** — vLLM, SGLang, TensorRT-LLM, and Triton (with the TRT-LLM /
vLLM backends) all implement it. If an "engine" doesn't do continuous batching, it's not a serious LLM
server.

### The latency/throughput dial and goodput
Bigger running batches → higher tokens/s throughput but higher ITL (each step does more work) and
longer queueing → higher TTFT under load. There is no free lunch; you choose an operating point.
Define an **SLO** (e.g. p95 TTFT < X ms, p95 ITL < Y ms) and optimize **goodput** = useful
throughput that *meets the SLO*, not raw peak throughput. Key knobs: max batch size / max num
sequences, max batched tokens, chunked-prefill size, and how prefill is scheduled against decode.

**Metrics to instrument:** TTFT, ITL/TPOT, end-to-end latency, tokens/s (input and output separately),
queue time, KV-cache utilization, and preemption/recompute counts. Engines export most of these as
Prometheus metrics.

---

## 4. Parallelism for serving

Choose the topology to fit the model size and the interconnect:

| Strategy | What it splits | Needs | Use when |
|---|---|---|---|
| **Tensor parallel (TP)** | each layer's matmuls across GPUs | fast intra-node links (NVLink/NVSwitch); all-reduce every layer | model/activations don't fit on one GPU; stay within a host |
| **Pipeline parallel (PP)** | contiguous layer groups across stages | tolerant of slower/cross-node links; needs micro-batching to fill bubbles | model spans hosts; TP would saturate inter-node fabric |
| **Expert parallel (EP)** | MoE experts across GPUs | all-to-all for token routing | large Mixture-of-Experts models (route tokens to expert shards) |
| **Data parallel (DP)** | nothing — independent replicas | a load balancer in front | scale throughput horizontally; the unit of autoscaling |

Practical rules of thumb (verify against your hardware):

- Prefer **TP within a node** (e.g. TP=8 across 8 NVLinked GPUs); reach for **PP across nodes** only
  when the model exceeds one node. Combine them (TP within node, PP across nodes) for the largest
  models. Excess TP across a slow fabric wrecks decode latency due to per-layer collectives.
- **DP replicas** are how you scale serving throughput and how autoscalers add/remove capacity.
- MoE adds **EP**; modern large MoE serving often mixes TP+EP (and DP for the attention/router).
- A model that needs **more than one host** to hold its weights + KV requires **multi-host serving**:
  one logical replica spanning several pods, started together with stable identity. On Kubernetes use
  **LeaderWorkerSet** (LWS) on top of JobSet — see `[[jobset-leaderworkerset]]`. The leader runs the
  engine's head/scheduler; workers run shards; they form one Ray/NCCL/torch-distributed group.

---

## 5. Disaggregated serving (prefill/decode split)

Because prefill is compute-bound and decode is bandwidth-bound, co-locating them forces a single
batch size and hardware ratio onto two opposite workloads, and lets long prefills stall token streams.
**Disaggregated serving** runs **separate prefill and decode worker pools**: a request is prefilled on
a prefill worker, its KV cache is **transferred** (over NVLink/RDMA/IB, e.g. via NIXL/NCCL) to a decode
worker, which then streams tokens. This is the **DistServe** idea, productized by **NVIDIA Dynamo**.

Benefits: each phase scales and batches independently (e.g. fewer, fatter prefill workers; many decode
workers), better TTFT *and* ITL under mixed load, and a natural place for **KV-aware routing** (send a
request to the worker that already holds its prefix). Cost: KV-transfer bandwidth, more moving parts,
and an orchestration layer. **Choose it at datacenter scale**, not for a single replica or modest QPS —
it adds real complexity. Chunked prefill within a single pool is the lighter-weight alternative.

---

## 6. Throughput/latency techniques to know

- **Chunked prefill** — split a long prefill into chunks and interleave with ongoing decode steps so
  one big prompt doesn't freeze everyone's token stream. Smooths ITL; widely supported (often default).
- **Speculative decoding** — a cheap drafter (small model, n-gram, Medusa/EAGLE-style heads, or
  prompt-lookup) proposes several tokens; the target model verifies them in one pass. Fewer target
  steps → lower latency when acceptance is high. Gains depend heavily on draft quality and workload.
- **Quantization** — `fp8` (weights and/or KV; near-lossless on modern HW, big speed/memory win),
  `int8`, and weight-only **AWQ / GPTQ** (4-bit) shrink weights to fit bigger models / more KV and
  speed bandwidth-bound decode. Always re-evaluate accuracy on your own eval set; quantization is not
  free. fp8 KV cache is a common, high-leverage setting.
- **Structured / guided decoding** — constrain output to a JSON schema, regex, or context-free grammar
  by masking logits (backends like XGrammar / Outlines / llguidance, or Triton via logits processors).
  Essential for tool-calling and typed outputs; check the throughput cost of the chosen backend.
- **CUDA graphs / kernel fusion / FlashAttention** — engines capture the decode step as a CUDA graph
  and use fused attention kernels to cut per-step overhead; mostly automatic, but a reason TRT-LLM and
  vLLM are fast. Toggle off only when debugging.

---

## 7. The frameworks

> All of these implement continuous batching and some form of KV reuse; the differences are in design
> center, hardware reach, deployment shape, and operational maturity. Verify current feature status.

### vLLM — the de-facto OSS engine
- **Differentiator:** PagedAttention + continuous batching, broad **model and hardware** coverage
  (NVIDIA, AMD ROCm, and increasingly others), large community, OpenAI-compatible server. The default
  choice you should justify deviating from.
- **Capabilities:** automatic prefix caching, TP and PP, fp8 and AWQ/GPTQ/int8 quant, chunked prefill,
  speculative decoding, guided decoding, LoRA serving (multi-adapter), multi-host (via Ray), KV
  connectors for disaggregation. Ships an OpenAI-compatible REST/`/v1` server (`vllm serve`).
- **When to pick:** single-model **max throughput** with minimal friction; broad model support;
  standard OpenAI API; you want OSS with the biggest ecosystem. Also the engine Dynamo/KServe/Ray
  Serve most commonly orchestrate.
- **K8s shape:** a `Deployment` (one pod = one replica) for single-host; **LeaderWorkerSet** for
  multi-host TP+PP; front with a Service + Inference Gateway. See `examples.md`.

### SGLang — RadixAttention + programmable generation
- **Differentiator:** **RadixAttention** for automatic prefix/KV reuse across requests, plus a
  front-end language for structured/programmatic generation (control flow, parallel calls, constrained
  outputs). High throughput, strong on prefix-heavy and agentic workloads.
- **Capabilities:** continuous batching, TP/PP/EP, fp8, speculative decoding, structured output,
  disaggregation support; OpenAI-compatible server.
- **When to pick:** heavy **prefix reuse** (shared system prompts, RAG, few-shot), **agentic** trees,
  multi-turn chat, complex **structured generation**; when RadixAttention's automatic reuse beats
  manual prefix-cache tuning. Increasingly used for very large MoE models.

### NVIDIA Dynamo — datacenter-scale disaggregated inference
- **Differentiator:** a **distributed serving framework** (the spiritual successor to Triton's
  inference-server role for LLMs) built for **disaggregated** prefill/decode at **multi-node**
  datacenter scale. It is an *orchestrator*, not a kernel engine — it drives backend engines like
  **vLLM, SGLang, or TensorRT-LLM**.
- **Capabilities:** prefill/decode **disaggregation**, **KV-aware routing** (route to the worker holding
  the matching prefix), a global KV cache manager and KV transfer layer (NIXL), dynamic GPU planner,
  multi-node coordination.
- **When to pick:** you operate at **scale** (many nodes, high QPS, large models) and need
  disaggregation, KV-aware routing, and elastic prefill/decode pools. Overkill for a single replica or
  small deployment — start with plain vLLM/SGLang and graduate to Dynamo when scale demands it.
- **K8s shape:** multi-component (router/frontend, prefill workers, decode workers, KV manager),
  typically via its operator/CRDs; multi-host workers via LWS-style grouping.

### NVIDIA Triton Inference Server — multi-framework general server
- **Differentiator:** a **general-purpose, multi-backend** inference server — serve TensorRT, PyTorch,
  ONNX, Python, **and** LLMs (via the **TensorRT-LLM backend** or a vLLM backend) from one server.
  Supports **ensembles / business-logic scripting (BLS)** to chain models (e.g. tokenize → embed →
  rerank → LLM), **dynamic batching**, model versioning, and concurrent model execution.
- **When to pick:** a **mixed fleet** — classic ML (vision, embeddings, rerankers, recsys) alongside
  LLMs, multi-step pipelines as ensembles, standardized HTTP/gRPC + KServe v2 protocol, one serving
  layer across teams. For pure single-LLM max-throughput, a dedicated vLLM/SGLang is usually simpler.
- **K8s shape:** `Deployment` of Triton pods with a model repository (PVC / object store); often
  fronted by KServe. With the TRT-LLM backend you serve a **pre-built engine** (see below).

### TensorRT-LLM — compiled-engine peak performance (NVIDIA)
- **Differentiator:** NVIDIA's **ahead-of-time compiled** LLM runtime — you **build** an optimized
  engine for a specific model + GPU + precision + parallelism, then **serve** it. Squeezes maximum GPU
  performance via fused kernels, fp8/CUDA-graph optimizations, and **in-flight batching**.
- **Tradeoff:** a **build-then-serve** workflow (engine build per model/GPU/config) and tighter
  NVIDIA-only coupling, in exchange for top-tier latency/throughput on NVIDIA hardware. Less plug-and-
  play than vLLM; rebuild on model/precision/parallelism changes.
- **When to pick:** you need **absolute peak** performance/efficiency on NVIDIA GPUs, have a stable
  model set, and can afford the build step and ops. Usually served **via Triton's TRT-LLM backend** or
  driven by Dynamo.

### JetStream — JAX/XLA inference engine for TPUs
- **Differentiator:** a **JAX/XLA throughput-and-memory-optimized** LLM inference engine aimed at
  **TPUs** (it also runs on GPUs). It serves JAX-native models — most notably **MaxText** models (see
  `[[maxtext-jax-llm]]`) and the Flax/JAX model zoo — with **continuous batching** and KV-cache
  management, compiled through XLA. The natural decode path for a JAX/TPU stack, so you don't have to
  cross into the PyTorch/CUDA world to serve.
- **Capabilities:** continuous batching, paged/quantized KV cache, TP/sharding via JAX (`jax.sharding`
  / GSPMD), int8/fp8-style quantization, an OpenAI-style/HTTP serving front end, and interchangeable
  engine backends (a MaxText engine and a PyTorch-XLA engine). Strong **cost-performance on TPU** vs
  running a generic CUDA-first engine on accelerators it wasn't built for. Verify the current feature
  and quantization matrix against project docs — it moves fast.
- **When to pick:** your models and training pipeline already live in **JAX/XLA on TPUs** (e.g. you
  trained with MaxText) and you want to serve them on **TPU** with good \$/token without porting to a
  CUDA engine. **Choose vLLM/SGLang instead** when you are GPU/PyTorch-centric, need the broadest
  model and feature coverage, or want the largest OSS ecosystem — JetStream's reach is narrower and
  TPU-centered by design. (vLLM also has a TPU path; benchmark both for your model — verify current.)
- **K8s shape:** a `Deployment` (one pod = one replica) on a **TPU node pool**, requesting TPU chips
  (`google.com/tpu`) with the right topology; **multi-host** TPU slices (one replica spanning hosts)
  use **LeaderWorkerSet** over JobSet — see `[[jobset-leaderworkerset]]`. Front with a Service +
  Inference Gateway like any other engine.

### Ray Serve — Python-native scalable serving & composition
- **Differentiator:** build serving apps in **Python** as composable **deployments / DAGs**, each
  independently scaled and resourced, on a Ray cluster. Native autoscaling, fractional GPUs,
  multi-model, model multiplexing, and **first-class vLLM integration** (LLM serving APIs) — so you can
  wrap vLLM replicas in a Ray Serve graph.
- **When to pick:** **multi-model** systems, **pipelines** (retrieval → rerank → LLM → post-process),
  custom Python business logic, heterogeneous models, or you already run on Ray. It's the
  composition/orchestration layer; vLLM/TRT-LLM do the GPU work inside it.
- **K8s shape:** a RayService (via **KubeRay**) managing a RayCluster; Ray Serve handles intra-cluster
  autoscaling, KubeRay/Karpenter handle node scaling.

### KServe — Kubernetes model-serving control plane
- **Differentiator:** the **CRD-based control plane** (`InferenceService`) for serving on K8s — not an
  engine, but the layer that *runs* engines. Standardized **v2 inference protocol**, model storage
  abstraction, canary/traffic splitting, transformer/predictor/explainer components, and (via Knative)
  **scale-to-zero**. Its newer LLM-focused path adds KV-/prefix-aware routing and an Inference
  Gateway story.
- **When to pick:** you want a **declarative, multi-tenant** serving platform on K8s with autoscaling
  and traffic management, **fronting vLLM, Triton, or TRT-LLM** runtimes. The platform/control-plane
  choice, orthogonal to the engine choice.

### NVIDIA NIM — prepackaged optimized inference microservices
- **Differentiator:** not a new engine but a **buy-vs-build** packaging of existing ones. NIM ships
  **prebuilt, performance-tuned inference microservices** as **containers** with an **OpenAI-compatible
  API**, running **TensorRT-LLM** (or **vLLM**) under the hood with model-/GPU-specific optimized
  engine profiles baked in — so you skip the engine build, kernel tuning, and packaging work and get a
  supported, drop-in service. The tradeoff: less low-level control and an **enterprise licensing**
  model (NIM is distributed under NVIDIA AI Enterprise; check the current licensing/entitlement and
  pull-credential terms for your use — verify against current docs).
- **When to pick:** you value **time-to-production and vendor support** over squeezing the last bit of
  control out of the stack, you're standardized on **NVIDIA GPUs**, and an off-the-shelf optimized
  container with a stable OpenAI API is worth the license. **Build with vLLM/SGLang/TRT-LLM yourself**
  instead when you need full control over flags/quantization/parallelism, want pure OSS, run non-NVIDIA
  hardware, or are cost-sensitive on licensing.
- **K8s shape:** a `Deployment` of the NIM container requesting `nvidia.com/gpu`, with image-pull
  secrets for NVIDIA's registry (NGC) and a cache volume for the downloaded/optimized model profile
  (mount a fast PVC and use long **startup probes** — first-boot profile selection/download is slow).
  Front with a Service + Inference Gateway; multi-host follows the same LWS pattern as other engines.
  Because the API is OpenAI-compatible, NIM slots into the same routing/gateway layer as a self-built
  engine.

### TGI (Text Generation Inference) — brief
Hugging Face's Rust/Python LLM server: continuous batching, tensor parallel, quantization, tight HF
ecosystem integration. Solid and easy on the HF stack; in 2026 the OSS center of gravity for raw
throughput and features has largely consolidated around **vLLM** and **SGLang**, so reach for TGI
mainly when you're deep in the HF tooling. Verify current feature parity before standardizing on it.

---

## 8. Autoscaling, load balancing & routing

- **Scale on the right signal.** GPU/CPU utilization is a poor proxy — a saturated decode loop can sit
  at modest utilization. Scale on **queue depth / pending requests / in-flight requests / TTFT**.
  Use **KEDA** (Prometheus-driven) or custom metrics HPA; see `[[autoscaling-kubernetes]]`.
- **Scale-to-zero** for spiky or dev workloads (Knative/KServe), but mind **cold start**: model load +
  engine warm-up + (TRT-LLM) engine load can be minutes. Keep a warm pool for latency-critical paths;
  pre-pull images and cache weights on a fast volume.
- **Node provisioning:** Cluster Autoscaler / **Karpenter** / GKE NAP add GPU nodes when pods go
  pending. Make accelerator requests explicit and use the right node pool / accelerator class.
- **KV-cache-aware / prefix-aware routing.** A plain round-robin LB throws away prefix-cache hits.
  Route requests with a shared prefix to the **same replica** so its KV/prefix cache is reused, and
  balance on load. This is the job of an **Inference Gateway** — the **Gateway API Inference
  Extension** (GKE Inference Gateway and the K8s `inference.networking` project) adds model-/KV-/load-
  aware routing, per-model endpoints, and serving-aware traffic policy in front of
  vLLM/Triton/Dynamo/JetStream/NIM — see `[[gke-inference-gateway]]`.
- **Disaggregated + Dynamo** push KV-aware routing further (route to the worker that holds the prefix,
  separate prefill/decode pools). Verify the current state of the Inference Gateway APIs — they are new
  and evolving fast.

---

## 9. Decision matrix

| Use case | Pick | Why |
|---|---|---|
| Single model, **max throughput**, OSS, broad HW | **vLLM** | PagedAttention + continuous batching, biggest ecosystem, OpenAI API |
| **Absolute peak** perf on NVIDIA, stable model set | **TensorRT-LLM** (via Triton/Dynamo) | compiled engine, fused kernels — accept the build step |
| Heavy **prefix reuse / agentic / structured** output | **SGLang** | RadixAttention auto-reuse + programmable, constrained generation |
| **Multi-framework** fleet (LLM + vision/embeddings/recsys), ensembles | **Triton** | one server, many backends, BLS pipelines, dynamic batching |
| **Disaggregated** prefill/decode at **datacenter scale**, KV-aware routing | **NVIDIA Dynamo** | orchestrates vLLM/SGLang/TRT-LLM across nodes |
| Serving **JAX/XLA models on TPUs** (e.g. MaxText), good \$/token | **JetStream** | JAX-native continuous batching on TPU — see `[[maxtext-jax-llm]]` |
| **Buy not build**: supported, prepackaged optimized service, OpenAI API | **NVIDIA NIM** | TRT-LLM/vLLM in a container; trade control/licensing for time-to-prod |
| **Multi-model pipelines / DAGs**, Python composition | **Ray Serve** (+ vLLM) | independent-scaling deployments, model multiplexing |
| **K8s control plane** over any engine, scale-to-zero, traffic mgmt | **KServe** | `InferenceService` CRD fronting vLLM/Triton/TRT-LLM |
| Big model that **spans hosts** (one replica > 1 node) | engine + **LWS** | multi-host TP+PP via `[[jobset-leaderworkerset]]` |
| Deep in **Hugging Face** tooling, modest needs | **TGI** | tight HF integration; verify feature parity vs vLLM/SGLang |

> Engine vs platform are **orthogonal axes**. Common stacks: KServe → vLLM; Ray Serve → vLLM;
> Triton → TRT-LLM; Dynamo → (vLLM | SGLang | TRT-LLM). Pick a platform for ops, an engine for GPUs.

---

## 10. Anti-patterns & gotchas

- **Static batching / no continuous batching.** Rolling your own naive batch loop wastes most of the
  GPU. Use an engine that does iteration-level batching.
- **Sizing KV by feel.** Set `max_model_len` / batch / `gpu-memory-utilization` (vLLM) to the actual
  KV budget. Too high → OOM or constant preemption/recompute; too low → idle GPU. Watch KV-cache
  utilization and preemption counters.
- **Over-parallelizing.** TP=8 across a slow inter-node fabric adds a collective per layer and tanks
  decode latency. TP within a node, PP across nodes. Don't shard a model that fits on one GPU.
- **Ignoring `/dev/shm`.** Multi-GPU engines (NCCL, Ray, torch.distributed) need ample shared memory.
  The default container `/dev/shm` (often 64 MB) causes mysterious hangs/crashes — mount a large
  `Memory` `emptyDir` at `/dev/shm`.
- **Cold starts under an autoscaler.** Model load + warm-up (and TRT-LLM engine load) is slow; pods
  flapping to zero then cold-starting blows your SLO. Use long startup probes, keep a warm minimum,
  cache weights on a fast local/PVC volume, pre-pull images.
- **Round-robin in front of a prefix cache.** Destroys cache-hit rate. Use prefix-/KV-aware routing.
- **Quantizing without evaluating.** AWQ/GPTQ/fp8 can quietly degrade quality on *your* task.
  Re-run evals; don't ship a number you didn't measure.
- **Disaggregating too early.** Dynamo/disaggregation is for scale. At one replica it's pure overhead;
  start with chunked prefill in a single pool.
- **Mismatched parallelism across hosts.** Multi-host engines need every shard up with stable identity
  and matched config; one slow/missing worker stalls the whole replica. Gang-schedule the group (LWS).

---

## 11. Troubleshooting (symptom → likely cause → fix)

- **OOM at load or first long request** → KV budget too large for the GPU, or context too long →
  lower the memory-utilization target / `max_model_len` / max batch; enable fp8 KV; add a GPU (TP).
- **High TTFT under load, fine when idle** → prefill queueing / long prompts blocking → enable/tune
  chunked prefill; add replicas; consider disaggregation; check prefix-cache hit rate.
- **High/spiky ITL (slow, jittery token stream)** → batch too large or long prefills interleaving →
  cap batch size; tune chunked-prefill size; verify CUDA graphs enabled; check for preemption.
- **Throughput far below expectations** → continuous batching off / batch too small / cache thrash →
  raise max sequences & batched tokens; raise memory-utilization; check prefix-cache and preemption
  metrics.
- **Frequent preemption / recompute** → KV pressure forcing eviction → reduce concurrency or context;
  enable KV offload; add capacity.
- **Multi-host replica hangs at startup** → NCCL/`/dev/shm`/networking → enlarge `/dev/shm`, verify
  the LWS group size and headless Service, check NCCL env (interface, IB/RDMA), confirm all shards
  scheduled (gang scheduling).
- **Replicas thrash / cold-start storms** → autoscaling on a bad signal or scale-to-zero too eager →
  scale on queue/in-flight, set stabilization windows and a warm minimum.
- **Structured output slow or malformed** → guided-decoding backend cost or grammar mismatch → try a
  different backend, simplify the schema, measure the throughput hit.

---

## 12. Security & multi-tenancy

- **Don't expose the raw engine endpoint.** Put an auth/quota/rate-limit gateway in front; OpenAI-
  compatible servers usually have no real authn/z of their own.
- **Tenant isolation:** the KV/prefix cache can become a side channel — a shared prefix cache may leak
  *timing* signal about another tenant's prompts. For strict isolation, don't share caches across
  trust boundaries; isolate per tenant/namespace.
- **Untrusted input:** validate prompt size and output limits; enforce max tokens to bound cost and
  KV; treat prompts as untrusted (prompt injection) at the application layer.
- **Model provenance:** pin and verify model artifacts; serve from a trusted registry/PVC, not
  arbitrary download at boot. Pin engine image digests.
- **Resource limits:** set GPU/CPU/memory requests=limits for accelerators; protect the node from a
  runaway server; use namespaces/quotas for multi-tenant clusters.

---

## 13. Version awareness

This space changes monthly. Expect churn in: vLLM's V1 engine internals and scheduler, disaggregation
KV connectors, the Gateway API Inference Extension / GKE Inference Gateway, Dynamo's components, and
which quant/spec-decode methods are "supported." **Always confirm against current project docs and
release notes** before relying on a flag, default, or "is it supported?" claim. Never invent config
keys or quote benchmark numbers you haven't measured.

---

## 14. Canonical references

- vLLM — docs & PagedAttention paper: <https://docs.vllm.ai/> ·
  *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et al., SOSP 2023)
  <https://arxiv.org/abs/2309.06180>
- SGLang — *SGLang: Efficient Execution of Structured LM Programs* (RadixAttention)
  <https://arxiv.org/abs/2312.07104> · docs <https://docs.sglang.ai/>
- NVIDIA Dynamo — <https://github.com/ai-dynamo/dynamo> · docs <https://docs.nvidia.com/dynamo/>
- NVIDIA Triton Inference Server — <https://github.com/triton-inference-server/server> ·
  docs <https://docs.nvidia.com/deeplearning/triton-inference-server/>
- TensorRT-LLM — <https://github.com/NVIDIA/TensorRT-LLM> ·
  docs <https://nvidia.github.io/TensorRT-LLM/>
- JetStream — <https://github.com/AI-Hypercomputer/JetStream> ·
  MaxText engine <https://github.com/AI-Hypercomputer/maxtext> (see `[[maxtext-jax-llm]]`)
- NVIDIA NIM — <https://docs.nvidia.com/nim/> · catalog <https://build.nvidia.com/> (licensing under
  NVIDIA AI Enterprise — verify current terms)
- Ray Serve — <https://docs.ray.io/en/latest/serve/index.html> · KubeRay
  <https://github.com/ray-project/kuberay>
- KServe — <https://kserve.github.io/website/>
- Text Generation Inference (TGI) — <https://github.com/huggingface/text-generation-inference>
- DistServe (disaggregated prefill/decode) <https://arxiv.org/abs/2401.09670> ·
  Orca (continuous/iteration-level batching) (Yu et al., OSDI 2022)
- Gateway API Inference Extension — <https://github.com/kubernetes-sigs/gateway-api-inference-extension>
- LeaderWorkerSet — <https://github.com/kubernetes-sigs/lws>

---

# Examples — Serving Frameworks on Kubernetes

Canonical, correct-in-shape manifests to imitate. They are **sketches**: image tags, model names,
resource counts, and flags are illustrative — **verify current images/flags against the engine's docs**
(this space changes monthly, 2026). Apply your cluster's conventions (registry, accelerator class,
gateway, namespaces). See `serving-frameworks-guide.md` for the *why* behind each choice.

---

## 1. vLLM — single-host `Deployment` (one pod = one replica)

One model that fits on a single node (here 1 GPU; bump `tensor-parallel-size` and GPU count to shard
within the node). OpenAI-compatible server, prefix caching on, generous startup probe, big `/dev/shm`.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-vllm
  labels: { app: llama-vllm }
spec:
  replicas: 2                       # data-parallel replicas; the unit of (auto)scaling
  selector: { matchLabels: { app: llama-vllm } }
  template:
    metadata:
      labels: { app: llama-vllm }
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest          # PIN a digest in production
        args:
          - "--model=meta-llama/Llama-3.1-8B-Instruct"
          - "--tensor-parallel-size=1"          # = number of GPUs in this pod
          - "--gpu-memory-utilization=0.90"     # KV budget: leave headroom; watch OOM/preemption
          - "--max-model-len=8192"              # bound context to fit the KV cache
          - "--enable-prefix-caching"           # reuse shared system-prompt / RAG prefixes
          # - "--quantization=fp8"              # verify accuracy on YOUR evals before enabling
        ports:
          - { name: http, containerPort: 8000 }
        env:
          - name: HF_TOKEN
            valueFrom: { secretKeyRef: { name: hf-token, key: token } }
        resources:
          limits:
            nvidia.com/gpu: "1"                 # must match tensor-parallel-size
        volumeMounts:
          - { name: dshm,  mountPath: /dev/shm }        # NCCL/Ray need real shared memory
          - { name: cache, mountPath: /root/.cache/huggingface }  # cache weights; faster restarts
        readinessProbe:                          # gate traffic on /health
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
        startupProbe:                            # model load is SLOW — give it minutes
          httpGet: { path: /health, port: 8000 }
          failureThreshold: 60
          periodSeconds: 10
      volumes:
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 8Gi }
        - name: cache
          emptyDir: {}                           # or a PVC / hostPath for persistent weight cache
---
apiVersion: v1
kind: Service
metadata: { name: llama-vllm }
spec:
  selector: { app: llama-vllm }
  ports: [ { name: http, port: 80, targetPort: 8000 } ]
```

> Front the Service with an **Inference Gateway** (Gateway API Inference Extension / GKE Inference
> Gateway) for prefix-/KV-/load-aware routing instead of round-robin — see the guide §8.
> Autoscale on **queue depth / in-flight**, not CPU — see `[[autoscaling-kubernetes]]`.

---

## 2. vLLM multi-host on **LeaderWorkerSet** (one replica spans nodes)

When weights + KV exceed a single node, one logical replica spans multiple pods. **LeaderWorkerSet**
(LWS) creates a leader + N workers as one gang-scheduled group with stable identity and a headless
Service; vLLM runs over Ray across the group. Topology below: 2 pods × 8 GPUs = TP=8 (within each pod,
NVLink) × PP=2 (across the two pods). Details in `[[jobset-leaderworkerset]]`.

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: llama-405b-vllm
spec:
  replicas: 1                       # one multi-host replica (scale this for more replicas)
  leaderWorkerTemplate:
    size: 2                         # group = 1 leader + 1 worker = 2 pods (PP=2 here)
    restartPolicy: RecreateGroupOnPodRestart   # one dead shard → restart the whole group
    leaderTemplate:                 # leader: starts Ray head, runs `vllm serve`, exposes :8000
      spec:
        containers:
        - name: vllm-leader
          image: vllm/vllm-openai:latest        # PIN a digest
          command: ["/bin/sh","-c"]
          args:
            - |
              ray start --head --port=6379 &
              vllm serve meta-llama/Llama-3.1-405B-Instruct \
                --tensor-parallel-size 8 \
                --pipeline-parallel-size 2 \
                --gpu-memory-utilization 0.92 \
                --max-model-len 8192 \
                --enable-prefix-caching
          ports: [ { containerPort: 8000 } ]
          resources: { limits: { nvidia.com/gpu: "8" } }   # 8 GPUs per pod (TP=8)
          volumeMounts:
            - { name: dshm, mountPath: /dev/shm }
        volumes:
          - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
    workerTemplate:                 # workers: join the Ray cluster, run their shards
      spec:
        containers:
        - name: vllm-worker
          image: vllm/vllm-openai:latest
          command: ["/bin/sh","-c"]
          args:
            - |
              ray start --address=$(LWS_LEADER_ADDRESS):6379 --block
          resources: { limits: { nvidia.com/gpu: "8" } }
          volumeMounts:
            - { name: dshm, mountPath: /dev/shm }
        volumes:
          - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
```

Key correctness points:
- **GPUs per pod == TP**; **group size (`size`) == PP stages**. Total GPUs = TP × PP = 8 × 2 = 16.
- The whole group must come up together — **gang-schedule** it; a missing/slow shard stalls the replica.
- LWS injects `LWS_LEADER_ADDRESS` so workers find the Ray head; serve only from the **leader**.
- Big `/dev/shm` on **every** pod or NCCL/Ray hang. Ensure node networking (NVLink/IB/RDMA) and NCCL
  env are correct for cross-node collectives (see guide §11 troubleshooting).
- Verify the exact vLLM multi-host launch invocation against current vLLM docs — the Ray/launch wiring
  evolves.

---

## 3. Triton + TensorRT-LLM — build-then-serve note (no full manifest)

TensorRT-LLM is **ahead-of-time compiled**: you do not point it at a HF model and go. The workflow:

1. **Build** a TRT-LLM engine for a specific `model + GPU arch + precision (e.g. fp8) + parallelism
   (TP/PP)`. The engine is tied to that combination — change any of them and you **rebuild**. Run the
   build as a one-off `Job` (or in CI) on a GPU matching production, writing the engine to a PVC /
   object store.
2. Lay out a **Triton model repository** with the **TensorRT-LLM backend** (the engine, tokenizer,
   and an ensemble/BLS that chains pre-processing → TRT-LLM → post-processing). Triton provides
   in-flight batching, dynamic batching, and the KServe v2 HTTP/gRPC protocol.
3. **Serve** with a Triton `Deployment` mounting the model repository (same GPU shape, large
   `/dev/shm`, long startup probe — engine load is slow), fronted by a Service / KServe / Inference
   Gateway.

```text
# shape only — consult current TensorRT-LLM + Triton docs for exact commands/flags:
#  Job:        trtllm-build  --checkpoint_dir ...  --output_dir /models/engine  (fp8, TP=N, PP=M)
#  Repo:       /models/{preprocess, tensorrt_llm, postprocess, ensemble}/...
#  Deployment: tritonserver --model-repository=/models   (mount the PVC; GPUs == TP×PP)
```

When to prefer this over vLLM/SGLang: you need **peak** NVIDIA performance on a **stable** model set
and can absorb the build step and operational coupling. For a mixed fleet (LLM + vision/embeddings),
Triton serves them all from one server; for disaggregation at scale, drive TRT-LLM via **Dynamo**.

> Do not hand-copy `trtllm-build` flags or repository keys from memory — they change between releases.
> Generate them from the version of TensorRT-LLM / Triton you are actually deploying.
