# Edge / On-Device ML — Deep Reference

Deploying ML off the datacenter: onto phones, laptops, wearables, cameras, vehicles, set-top boxes, and
microcontrollers. This guide is the **deployment target** — the runtimes, the hardware, the conversion
craft, and the on-device validation. For the *general* compression theory (PTQ vs QAT, INT4/AWQ/GPTQ,
GGUF internals, pruning, distillation, KV-cache quant theory) read `[[inference-optimization]]`; this
guide tells you how those land on real device runtimes and silicon, and what breaks in between.

The field moves fast (it is 2026). Names change (TensorFlow Lite → **LiteRT**), backend/delegate support
shifts, supported-op lists grow, and on-device LLM sizes/speeds improve every quarter. Name the tool;
**verify current APIs, supported ops, and every device benchmark number against the project's own docs**
before committing. Never quote a benchmark you did not measure.

## 1. Mental model — why edge is a different game

A datacenter inference engine optimizes throughput and $/token across a fleet of identical GPUs with
near-unlimited power and cooling. **Edge inverts almost every assumption.** You optimize for a *single*
device's latency and energy, under a hard fixed budget, on hardware you do not control.

**Why ship to the edge:**

- **Latency** — no network round trip; the inference happens in milliseconds locally. Interactive UX
  (camera, keyboard, voice) needs this.
- **Privacy** — raw data (photos, audio, health, keystrokes) never leaves the device. This is the
  strongest argument and pairs directly with `[[privacy-preserving-ml]]` (federated learning, on-device
  DP).
- **Offline / availability** — works on a plane, in a tunnel, in a factory with no connectivity. No
  dependency on a backend being up.
- **Bandwidth** — sending a 4K video frame or continuous audio to a server is expensive and slow;
  inferring locally sends only the result.
- **Cost** — per-inference server cost is zero once the model is on-device; you trade capex/opex for the
  user's battery.

**What it costs you — the hard constraints (memory, compute, power/thermal, heterogeneity):**

- **Memory** — device RAM is shared between the OS, your app, and the model, often single-digit GB total
  on phones and *kilobytes* on microcontrollers. The model file, the runtime, intermediate activations,
  and (for LLMs) the KV cache all compete. OOM is a crash, not a slow path.
- **Compute** — no datacenter GPU. You have a mobile CPU, maybe a mobile GPU, maybe an NPU — all an
  order of magnitude weaker and far more memory-bandwidth-constrained than a server accelerator.
- **Power / thermal** — every joule is battery, and sustained compute heats the SoC until the scheduler
  **throttles** it. Steady-state throughput is well below burst. A model that benchmarks great for 10
  inferences may halve in speed after 60 seconds of continuous use.
- **Heterogeneous hardware** — you ship one app to thousands of device models with different SoCs, NPUs,
  driver versions, and OS levels. The accelerated path on a flagship may not exist on a mid-range phone;
  you need graceful fallback, not a hard requirement.

The job is to fit a useful model inside that budget *and prove it still works on the device*.

## 2. Runtimes & frameworks — what they are and when to use which

### ExecuTorch (PyTorch's on-device runtime)

PyTorch's official solution for on-device inference (mobile, embedded, edge), reaching its **1.x stable**
line. The pipeline is **`torch.export` → an edge/ExecuTorch program (`.pte`) → a lightweight C++ runtime**
on the device. The defining concept is the **backend / delegate** model: portions of the exported graph
are *delegated* to a hardware backend; whatever a backend doesn't support runs on the portable CPU
operators.

Backends/delegates (verify current list against the repo):

- **XNNPACK** — optimized CPU kernels for Arm and x86; the default, included in the published Android/
  iOS/pip packages. Your baseline that always works.
- **Core ML** and **MPS** — Apple Neural Engine / GPU on iOS/macOS.
- **Qualcomm** (HTP/QNN) — Hexagon NPU on Snapdragon.
- **Arm** (TOSA / Ethos-U), **Vulkan** (cross-vendor GPU), **MediaTek**, **Cadence** — others.

Reach for ExecuTorch when your model is **PyTorch-native** (especially LLMs and new research models) and
you want a single export path that can target CPU and multiple NPUs via delegates. It is the natural
landing zone for `[[inference-optimization]]` artifacts that started in PyTorch.

Repo/docs: `https://github.com/pytorch/executorch`, `https://docs.pytorch.org/executorch/`.

### LiteRT (formerly TensorFlow Lite) / Google AI Edge

**LiteRT is the successor to TensorFlow Lite** — same `.tflite` flatbuffer format and Interpreter API
(migrating from TFLite is largely a package rename), but rebranded to reflect that it now runs models
authored in **TensorFlow, Keras, JAX, and PyTorch**. It is the most mature **Android-first** on-device
runtime, with a rich set of **delegates**: **GPU** (OpenGL/OpenCL/Metal), **NNAPI** (Android's neural
network abstraction, routes to vendor NPUs), **Hexagon**, and **Core ML** on iOS. **MediaPipe** and the
**LiteRT-LM / MediaPipe LLM Inference** stack build on it for ready-made tasks and on-device LLMs.

Reach for LiteRT for Android-first apps, cross-framework conversion, and when you want the broad
delegate/NNAPI ecosystem. Docs: `https://ai.google.dev/edge/litert`.

### ONNX Runtime (Mobile / Web)

Runs **ONNX** graphs via **Execution Providers** (EPs): CPU, **NNAPI** (Android), **Core ML** (iOS),
**XNNPACK**, **QNN** (Qualcomm), and **WebGPU/WASM** in the browser. **ONNX Runtime Mobile** uses a
reduced build and the compact `.ort` format to shrink binary size; **ONNX Runtime Web** is the standard
way to run models **in-browser**. Reach for it when your models are already ONNX, you need one runtime
across mobile + web + desktop, or you want browser inference. Docs: `https://onnxruntime.ai`.

### Core ML (Apple)

Apple's first-party framework. You convert to a **`.mlpackage`** (via **`coremltools`**) and the OS
decides whether to run each layer on **CPU, GPU, or the Apple Neural Engine (ANE)** — you request a
compute unit preference but don't hand-schedule the ANE. Best Apple-only latency and energy, deep OS
integration (background, on-device personalization). The cost is Apple-only and a conversion step with
its own op-support quirks. `coremltools`: `https://github.com/apple/coremltools`.

### MediaPipe

Google's framework of prebuilt, tunable **perception pipelines** (face/hand/pose, segmentation, object
detection, audio, and LLM inference) on top of LiteRT. Reach for it when a standard vision/audio task
maps to an existing solution — you get a calibrated, GPU-delegated pipeline instead of wiring it
yourself. `https://ai.google.dev/edge/mediapipe`.

### The on-device LLM stack

- **llama.cpp / ggml / GGUF** — the de-facto CPU/edge LLM runtime. **GGUF** is its quantized weight
  format (k-quants, multiple bit-widths); runs on CPU with Arm NEON / x86 AVX and offloads to Metal/
  Vulkan/CUDA where present. The pragmatic choice for laptops, desktops, and Linux edge boxes.
  `https://github.com/ggml-org/llama.cpp`.
- **MLC-LLM** — compiles LLMs (via Apache TVM) to run across iOS, Android, WebGPU, and more.
  `https://github.com/mlc-ai/mlc-llm`.
- **MNN** (Alibaba) and **NCNN** (Tencent) — high-performance mobile inference engines, strong on Android
  and in vision; both have grown LLM support.

### Picking one

| Situation | Reach for |
|---|---|
| PyTorch model, want multi-NPU via one export | **ExecuTorch** |
| Android-first, cross-framework, NNAPI/GPU delegates | **LiteRT** |
| ONNX graphs across mobile + desktop, or **in-browser** | **ONNX Runtime (Mobile/Web)** |
| Apple-only, want the Neural Engine + OS integration | **Core ML** |
| Standard vision/audio task | **MediaPipe** |
| LLM on laptop/desktop/Linux edge, CPU-first | **llama.cpp / GGUF** |
| LLM across iOS/Android/web from one toolchain | **MLC-LLM** |
| Microcontroller / KB-scale | **TFLite Micro** (see §3) |

These overlap and shift; let the **source framework**, the **target OS**, and **which backend actually
supports your ops** decide — not brand loyalty.

## 3. Hardware — the silicon you actually run on

- **Mobile NPUs / accelerators.** Purpose-built matrix/convolution engines: **Apple Neural Engine
  (ANE)**, **Qualcomm Hexagon** (with the HTP tensor accelerator), **Google Tensor** (Edge TPU-derived).
  They are fast and power-efficient *for the ops and dtypes they support* — typically INT8 and some
  FP16/INT4 — and fall back to CPU/GPU for anything else. You rarely program them directly; you go
  through a runtime/delegate (Core ML→ANE, NNAPI/QNN→Hexagon, etc.).
- **GPUs via delegates.** Mobile/desktop GPUs accelerate well-batched FP16 work via the **GPU delegate**
  (LiteRT), **Vulkan/MPS** (ExecuTorch), or **WebGPU** (ORT Web, MLC). Good for vision; the
  CPU↔GPU transfer cost can dominate for small/sparse graphs.
- **Edge TPU / Coral.** Google's edge ASIC (USB/M.2/SoM). Requires a model **fully INT8-quantized and
  compiled by the Edge TPU compiler**; any unsupported op runs on the host CPU and breaks the pipeline at
  that boundary. Excellent perf/W for supported CNNs. `https://coral.ai`.
- **TinyML / microcontrollers (the extreme).** **LiteRT for Microcontrollers / TensorFlow Lite Micro**
  runs on Cortex-M-class MCUs with **no OS, no malloc** — you give it a fixed **tensor arena** (a
  preallocated byte buffer) and it runs interpreter-style with a hand-selected `OpResolver`. Memory is
  measured in **kilobytes**; models are tiny INT8 CNNs/keyword-spotters. Adjacent stacks: **CMSIS-NN**
  kernels, **microTVM**, vendor SDKs. Everything in this guide tightens by 1000× here: no dynamic
  shapes, no unsupported ops, no surprises. `https://github.com/tensorflow/tflite-micro`.

## 4. On-device optimization — the edge specifics

Defer technique *depth* to `[[inference-optimization]]`. What is edge-specific:

- **Aggressive INT8/INT4, and the quality cliff.** Server inference often runs FP16/BF16/FP8; edge wants
  **INT8** (the common sweet spot, broadly accelerated by mobile NPUs) and increasingly **INT4** for LLM
  weights to fit RAM. The **quality cliff is task-dependent and sharper at low bit-widths** — INT4 that
  is fine for a chat demo can wreck a structured-extraction or safety task. **Always eval the quantized
  model on the target task, on the device.**
- **Use the runtime's own quant tooling.** Quantize with LiteRT's post-training quantization / ExecuTorch's
  quantizer / `coremltools` quantization so the produced kernels match what the backend implements. A
  generically-quantized model can convert but then run *unaccelerated* (or not at all) on the target.
- **Calibration is on representative data.** PTQ needs a small calibration set drawn from the real edge
  distribution (the device's camera, the user's audio), not ImageNet by default. Wrong calibration data
  is a silent accuracy loss.
- **Weight clustering / pruning.** Clustering (palettization on Core ML) and structured pruning shrink
  the file and can help bandwidth-bound mobile inference; unstructured sparsity rarely speeds mobile
  kernels (no hardware support) and mostly just compresses the file. Recover accuracy with fine-tuning.
- **Hardware-aware NAS.** When you control the architecture, search for one that maps to the target's
  fast ops/dtypes (e.g. MobileNet/EfficientNet families, NPU-friendly blocks) rather than retrofitting a
  server model.
- **Delegate / operator selection & fallback** (the load-bearing edge skill). A delegate claims the
  subgraphs it supports; the rest stays on CPU. Each **partition boundary costs a tensor copy** (and
  sometimes a layout/dtype conversion). A model split into many islands can be **slower than pure CPU**.
  Inspect the partitioning, push toward **one large accelerated subgraph**, and remove/replace the ops
  that force fallbacks. Decide your fallback order explicitly (NPU → GPU → CPU) and confirm the CPU path
  is always viable for the long tail of devices.
- **Memory & KV-cache planning for on-device LLMs.** RAM budget = OS + app + **weights** (4-bit) +
  **runtime/activations** + **KV cache**. KV grows with `layers × context × 2 × kv_heads × head_dim ×
  bytes_per_elem`; on long context it can rival the weights. Levers: cap **max context**, **quantize the
  KV cache** (e.g. INT8/INT4 KV), use **GQA/MQA** models (fewer KV heads), and size the budget for the
  *lowest-RAM device* you support, not the flagship.

## 5. The conversion pipeline — the real pain

This is where edge projects actually slip. The pipeline is **source model → exported graph → target
format**, and **every hop can drop ops, rewrite numerics, or silently regress quality.**

**Typical paths:**

- PyTorch → `torch.export` → ExecuTorch `.pte` (with backend delegation + quantization).
- PyTorch/TF/JAX/Keras → LiteRT converter → `.tflite` (with PTQ).
- PyTorch → ONNX → ONNX Runtime `.ort` (or → LiteRT/Core ML via converters).
- PyTorch/TF → `coremltools` → `.mlpackage`.
- LLM weights → GGUF (llama.cpp) or MLC compile.

**The failure modes, in order of how often they bite:**

1. **Unsupported op.** The source graph contains an op the converter or the target backend doesn't
   implement. Find this at **export time** by inspecting the converted graph / partitioner output — never
   in the field. Fixes: rewrite the model to supported ops, register a custom op/kernel, or accept a CPU
   fallback for that subgraph (and measure the cost).
2. **Numerical mismatch.** The converted/quantized model produces different outputs than the source. This
   is normal in small amounts and dangerous in large amounts. **Debug it layer by layer**: dump
   intermediate activations from the source and the target on the *same* input, find the first layer
   whose output diverges beyond tolerance, and root-cause it (a fused op, a different rounding mode, a bad
   quant scale, an fp16 overflow).
3. **Quality regression that passes a single test.** Per-tensor agreement on one input is not enough.
   **Validate task metrics** (accuracy/WER/mAP/eval suite) on a representative set, comparing source vs
   on-device, before shipping.

**Parity validation — the non-negotiable ship gate:**

- Run the **same inputs** through the source model and the on-device model; compare outputs with a
  tolerance (and, for classifiers, top-k agreement) and compute the **task metric** on a held-out set.
- Run it **on the target device**, not just the host simulator — kernels, dtypes, and the scheduler
  differ.
- Gate releases on parity: if the on-device task metric drops beyond your bar, you do not ship.

**Operational lifecycle (treat the model like software):**

- **OTA model updates** — deliver models over the air, **decoupled from app-store releases**, so you can
  fix/improve without an app update. Sign and verify the payload; handle interrupted downloads.
- **Versioning** — every model is a versioned artifact tied to the runtime version and op set it needs;
  pin compatibility so an old app doesn't load a model it can't run.
- **On-device A/B / canary** — roll a new model to a fraction of devices, compare metrics, and **keep a
  rollback** to the last-good model on the device.
- **On-device profiling** — measure latency, memory, and **energy/thermal** on real representative
  hardware (low-end included), at **steady state**, not the first burst. Use the runtime's profiler
  (ExecuTorch ETDump/inspector, LiteRT benchmark tool, Core ML Instruments, Xcode/Android profilers).

## 6. Multimodal on edge & private/federated learning

- **Vision and speech are the dominant edge workloads** — object detection, segmentation, OCR, depth,
  ASR/wake-word, and increasingly small **on-device VLMs**. These map well to NPUs and to MediaPipe's
  prebuilt pipelines. See `[[multimodal-ml]]` for the model side.
- **Edge + private/federated learning.** Because data stays on the device, edge is the natural home for
  **federated learning** (train across devices, aggregate updates centrally) and **on-device differential
  privacy / personalization** (fine-tune locally to the user without uploading raw data). See
  `[[privacy-preserving-ml]]`.

## 7. Anti-patterns (the traps that ship broken models)

- **Assuming server quantization transfers to the device.** A model quantized for a GPU engine may use
  schemes/kernels the mobile NPU doesn't have — it converts but runs unaccelerated or wrong. Quantize
  with the *target runtime's* tooling and re-eval on-device.
- **No parity validation after conversion.** "It exported, ship it." The converted model is a *new*
  model; prove parity on the task, on the device.
- **Ignoring thermal throttling.** Reporting burst latency and being surprised when sustained use is half
  as fast and drains the battery. Measure steady state and energy.
- **Shipping a model with an unsupported op.** It loads, then crashes (or silently CPU-falls-back into a
  latency cliff) on a device you didn't test. Check op support at export time.
- **INT4 with no quality eval on the target task.** 4-bit looks fine on a chat vibe-check and fails the
  actual extraction/classification/safety task. Eval the bit-width you ship.
- **No OTA / rollback plan.** A bad model in an app-store binary is stuck for a release cycle. Build OTA
  delivery and rollback before you launch.
- **Over-fragmented delegation.** Tolerating a graph split into many accelerated/CPU islands; the copy
  overhead makes it slower than plain CPU. Consolidate the partition.
- **Designing for the flagship.** Sizing memory and choosing the accelerated path for the best device,
  then OOMing or failing on the mid-range majority. Budget for the floor of your supported devices.

## 8. Version awareness

It is 2026 and this ecosystem reorganizes constantly. **TensorFlow Lite is now LiteRT** (under Google AI
Edge). ExecuTorch has reached a stable 1.x line, with an evolving backend/delegate roster. Supported-op
lists, NNAPI behavior, Core ML compute-unit scheduling, GGUF quant types, and on-device LLM viable sizes
all change between releases. **As of 2026, 4-bit quantized ~7–8B-class LLMs can run interactively on
high-end phones** — treat that as time-scoped; **verify current sizes, speeds, and any benchmark against
device measurements you take yourself.** Confirm every API, flag, format, and op-support claim against the
project's own current docs before relying on it.

## 9. Canonical references (verify current)

- ExecuTorch — `https://github.com/pytorch/executorch`, docs `https://docs.pytorch.org/executorch/`
- LiteRT / Google AI Edge — `https://ai.google.dev/edge/litert`, repo `https://github.com/google-ai-edge/LiteRT`
- TFLite→LiteRT migration — `https://ai.google.dev/edge/litert/migration`
- LiteRT for Microcontrollers / TFLite Micro — `https://github.com/tensorflow/tflite-micro`
- ONNX Runtime (Mobile/Web) — `https://onnxruntime.ai`
- Core ML Tools (`coremltools`) — `https://github.com/apple/coremltools`, `https://apple.github.io/coremltools/`
- MediaPipe — `https://ai.google.dev/edge/mediapipe`
- llama.cpp / GGUF — `https://github.com/ggml-org/llama.cpp`
- MLC-LLM — `https://github.com/mlc-ai/mlc-llm`
- MNN — `https://github.com/alibaba/MNN`; NCNN — `https://github.com/Tencent/ncnn`
- Coral / Edge TPU — `https://coral.ai`
