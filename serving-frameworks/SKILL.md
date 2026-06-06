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
