# AGENTS.md — LLM/ML Inference Serving

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`serving-frameworks-guide.md`** next to this file — read it
> before deploying, tuning, or choosing an inference engine. Concrete manifests to imitate are in
> **`examples.md`**. This file is the always-on summary.
>
> **These frameworks move very fast (it is 2026).** Name capabilities, but verify current flags,
> defaults, and "supported?" claims against the project's own docs/release notes. Never fabricate a
> config key or a benchmark number.

## Apply by default when serving LLMs/ML

- **Two phases, two bottlenecks.** Prefill = compute-bound (sets **TTFT**); decode = memory-bandwidth-
  bound and sequential (sets **ITL/TPOT**). This asymmetry drives chunked prefill, disaggregation, and
  most tuning. Throughput and \$/token are set mainly by KV-cache and batching behavior.
- **KV cache is the limit** after weights: `≈ 2·layers·kv_heads·head_dim·dtype·seq_len·batch`. It caps
  concurrency and context. Size `max_model_len`/batch/memory-utilization to the real KV budget — too
  high → OOM/preemption, too low → idle GPU. GQA/MQA and fp8 KV shrink it.
- **PagedAttention** (vLLM, OS-paging-style KV blocks, near-zero fragmentation) and **RadixAttention**
  (SGLang, automatic prefix reuse via radix tree) are the two core KV ideas. Use **prefix/prompt
  caching**; put the stable shared text first in the template.
- **Continuous (in-flight) batching is table stakes** — never roll a static batch loop. Optimize
  **goodput** (throughput meeting an SLO), not raw peak. Instrument TTFT, ITL/TPOT, tokens/s, queue
  time, KV utilization, preemptions.
- **Parallelism:** **TP within a node** (fast NVLink, collective per layer), **PP across nodes**, **EP**
  for MoE, **DP** replicas to scale out. Don't over-shard a model that fits on one GPU. A model that
  spans hosts needs **multi-host serving** → LeaderWorkerSet, see `[[jobset-leaderworkerset]]`.
- **Disaggregated serving** (separate prefill/decode pools + KV transfer; Dynamo/DistServe) is for
  datacenter scale only. At small scale use chunked prefill in one pool; don't disaggregate early.
- **Quantization** (fp8/int8/AWQ/GPTQ) fits bigger models / more KV and speeds decode — but
  **re-evaluate accuracy on your own evals**. Speculative decoding cuts latency when acceptance is
  high. Guided/structured decoding for JSON/grammar/tool-calling.
- **On Kubernetes:** request explicit `nvidia.com/gpu`/TPU; mount a large `Memory` emptyDir at
  **`/dev/shm`** (NCCL/Ray hang otherwise); use long **startup probes** (model load is slow); cache
  weights on a fast volume; pre-pull images.
- **Autoscale on queue depth / in-flight / TTFT, not CPU/GPU%** (KEDA / custom HPA, see
  `[[autoscaling-kubernetes]]`). Mind cold starts with scale-to-zero; keep a warm minimum for
  latency-critical paths.
- **Route prefix-/KV-aware, not round-robin** — send shared-prefix requests to the same replica to
  reuse its cache. Use an **Inference Gateway** (Gateway API Inference Extension / GKE Inference
  Gateway) in front of vLLM/Triton/Dynamo. Verify these APIs' current state — they're new.

## Engine pick (short form — full matrix in the guide)
- Single-model **max throughput**, OSS, broad HW → **vLLM** (the default; justify deviating).
- **Absolute peak** on NVIDIA, stable models → **TensorRT-LLM** (build-then-serve, via Triton/Dynamo).
- Heavy **prefix reuse / agentic / structured** → **SGLang** (RadixAttention).
- **Multi-framework** fleet + ensembles (LLM + vision/embeddings) → **Triton**.
- **Disaggregated at scale**, KV-aware routing → **NVIDIA Dynamo** (orchestrates vLLM/SGLang/TRT-LLM).
- **Multi-model pipelines / Python composition** → **Ray Serve** (+ vLLM).
- **K8s control plane** over any engine, scale-to-zero, traffic mgmt → **KServe** (`InferenceService`).
- **JAX/XLA models on TPU** (e.g. MaxText), good \$/token → **JetStream** (see `[[maxtext-jax-llm]]`).
- **Buy-not-build**: prepackaged optimized container, OpenAI API, NVIDIA support/licensing → **NVIDIA NIM**.
- Deep in **Hugging Face** tooling, modest needs → **TGI** (verify parity vs vLLM/SGLang).
- **Engine and platform are orthogonal:** common stacks are KServe→vLLM, Ray Serve→vLLM,
  Triton→TRT-LLM, Dynamo→(vLLM|SGLang|TRT-LLM).

## Related skills
`[[ml-frameworks]]` · `[[jobset-leaderworkerset]]` · `[[autoscaling-kubernetes]]` ·
`[[aiml-on-kubernetes]]` · `[[gke-master]]` · `[[training-frameworks]]`
