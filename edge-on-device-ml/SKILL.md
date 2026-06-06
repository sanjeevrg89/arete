---
name: edge-on-device-ml
description: Deploying ML models on edge / mobile / embedded devices — the runtimes, hardware, and
  conversion craft for running models off the datacenter. Use when targeting phones, laptops, wearables,
  cameras, vehicles, or microcontrollers; when you see ExecuTorch, TensorFlow Lite / LiteRT, ONNX Runtime
  Mobile/Web, Core ML / coremltools, MediaPipe, llama.cpp / ggml / GGUF, MLC-LLM, MNN, NCNN, TFLite
  Micro; when targeting Apple Neural Engine, Qualcomm Hexagon, Google Tensor, Edge TPU / Coral, or
  GPU/NPU delegates; or when the task is exporting/converting a model (torch.export, .pte, .tflite,
  .mlpackage, .onnx), picking and tuning a delegate, debugging unsupported ops or numerical parity vs the
  source model, INT8/INT4 on-device quantization and the quality cliff, on-device LLM KV-cache/memory
  budgeting, OTA model updates and rollback, or on-device profiling. The general compression theory lives
  in `[[inference-optimization]]`; THIS skill is the edge *deployment target* — runtimes, hardware,
  conversion pipeline, and on-device validation.
---

# Edge / On-Device ML

Apply the judgment of an engineer who has shipped models onto phones, embedded boards, and
microcontrollers in production: who knows that **the model that passed eval in your notebook is not the
model that runs on the device** — conversion, quantization, and a fixed memory/thermal budget change its
behavior — and that the only honest ship gate is **parity validation on the target hardware against the
task that matters.**

## How to use this skill

1. **Read `edge-on-device-ml-guide.md`** in this directory — the full reference. It builds the edge
   constraint model first (why off-datacenter is *different*: memory, power/thermal, heterogeneous
   hardware, no GPU farm), then covers the runtimes (ExecuTorch, LiteRT/TFLite, ONNX Runtime Mobile/Web,
   Core ML, MediaPipe, the on-device-LLM stack), the hardware (mobile NPUs, Edge TPU/Coral, GPU
   delegates, TinyML/MCUs), on-device optimization specifics, and the conversion pipeline.
2. For concrete flows to imitate (PyTorch→ExecuTorch/TFLite export+quantize+parity, delegate/fallback
   selection, on-device-LLM memory budget), read **`examples.md`**.
3. Match the target platform's conventions and SDK (iOS/Core ML, Android/LiteRT-NNAPI, vendor NPU SDK).
   Apply the correctness gate — **validate numerical parity and task metrics on the device after every
   conversion/quant step** — regardless.
4. **This field moves fast (it is 2026).** Runtime names (TFLite→LiteRT), backend/delegate availability,
   supported ops, and especially on-device LLM sizes/speeds change every quarter. Name the tool, but
   **verify current APIs, supported ops, and any device benchmark number against the project's own docs
   before committing.** Never fabricate a benchmark.

## Essentials (full detail in `edge-on-device-ml-guide.md`)

- **Pick edge for a reason, and pay for it knowingly.** Edge buys **latency** (no round trip),
  **privacy** (data never leaves the device — see `[[privacy-preserving-ml]]`), **offline/availability**,
  **bandwidth**, and **per-inference $0 server cost**. It costs you a hard, fixed budget: device RAM
  (often single-digit GB shared with the OS and app), no datacenter GPU, a **power/thermal** ceiling that
  throttles sustained throughput, and **heterogeneous hardware** you don't control. Decide on the
  constraint, not the hype.
- **Choose the runtime by source framework + target OS.** **ExecuTorch** for PyTorch models, especially
  PyTorch-native LLMs, with a per-target *backend/delegate* model (XNNPACK CPU, Core ML/MPS, Qualcomm,
  Arm, Vulkan). **LiteRT** (the successor to TensorFlow Lite) for Android-first and cross-framework;
  rides **NNAPI/GPU/Hexagon** delegates. **Core ML** for Apple-only when you want the OS to schedule the
  **Neural Engine**. **ONNX Runtime Mobile/Web** for ONNX graphs and **in-browser (WASM/WebGPU)**.
  **MediaPipe** for ready-made vision/audio pipelines. For LLMs: **llama.cpp/GGUF**, **MLC-LLM**, **MNN/
  NCNN**.
- **The conversion pipeline is the real work, not an afterthought.** model → exported graph
  (`torch.export`, SavedModel/Keras, ONNX) → target format (`.pte`, `.tflite`, `.mlpackage`, `.ort`).
  Every hop can drop or rewrite ops, change numerics, and silently regress quality. Budget real
  engineering time for it.
- **Quantize *for the device*, and never assume server quant transfers.** On-device wants aggressive
  **INT8** (often the sweet spot) and increasingly **INT4** for LLM weights — but the **quality cliff**
  is real and task-dependent. Calibrate with representative data; prefer the runtime's own quant tooling
  so the chosen kernels actually exist on the target. Technique depth is in `[[inference-optimization]]`;
  here the rule is **eval the quantized model on the device, on your task.**
- **Delegates partition the graph; unsupported ops fall back to CPU.** A delegate (NPU/GPU/Hexagon)
  takes the subgraphs it supports; the rest runs on CPU. **Many fallbacks = many partition boundaries =
  expensive copies** and you can end up *slower* than plain CPU. Inspect the partitioning; aim for one
  big accelerated subgraph, not a dozen islands.
- **An unsupported op is a shipping blocker, find it at export time.** Before you commit to a target,
  confirm every op is supported (or has a fallback you accept). Discovering an unsupported op in the
  field is the classic edge failure.
- **On-device LLMs are bounded by memory, then bandwidth.** Weights (4-bit quantized) + **KV cache**
  (grows with context × layers) must fit in the app's RAM budget alongside everything else. Plan the
  KV-cache size and max context up front; quantize the KV cache; cap context. (As of 2026, 4-bit
  quantized ~7–8B-class models run interactively on high-end phones — **verify current sizes/speeds
  against device benchmarks; do not quote a number you haven't measured.**)
- **Mind thermal throttling.** Burst benchmarks lie. Sustained on-device inference heats the SoC and the
  scheduler downclocks; measure steady-state, not the first 10 inferences.
- **Ship updates like software: OTA, versioned, A/B-tested, rollback-able.** Models are artifacts that
  change. Version them, deliver over the air decoupled from app releases, **A/B / canary** on-device, and
  keep a **rollback** to the last-good model. Profile on real devices, not just the simulator.
- **Multimodal is the common edge workload.** Vision and speech (detection, segmentation, ASR, wake-word,
  on-device VLMs) dominate real deployments — see `[[multimodal-ml]]`. Edge also pairs with **federated /
  private learning** to train without centralizing data — see `[[privacy-preserving-ml]]`.
- **Anti-patterns:** assuming server quantization transfers to the device; no parity check after
  conversion; ignoring thermal throttling; shipping a model with an unsupported op; INT4 with no
  quality eval on the target task; no OTA/rollback plan; benchmarking a burst instead of steady state.

## Related skills

- `[[inference-optimization]]` — the general compression/decode-acceleration theory (PTQ/QAT, INT4/AWQ/
  GPTQ, GGUF, pruning, distillation, KV-cache quant). This skill is its **edge deployment target**: how
  those techniques land on actual device runtimes and hardware.
- `[[ml-frameworks]]` — PyTorch/JAX/XLA where the source model and `torch.export` live.
- `[[ml-compilers-codegen]]` — the compiler/codegen layer (graph lowering, op partitioning, kernel
  generation) underneath the on-device runtimes and delegates.
- `[[serving-frameworks]]` — the *server-side* counterpart (vLLM/Triton/etc.); contrast with running off
  the datacenter entirely.
- `[[privacy-preserving-ml]]` — on-device keeps data local; pairs with federated learning / DP for
  private edge training.
- `[[multimodal-ml]]` — vision/speech/VLM models that are the dominant edge inference workloads.
