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

---

# Reference — edge-on-device-ml

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

---

# Edge / On-Device ML — Worked Examples

Flows to imitate. APIs move fast (it is 2026) — these are **canonical-in-spirit**; verify exact names,
flags, and supported ops against the current ExecuTorch / LiteRT / coremltools docs before running. Never
quote a device benchmark you didn't measure.

---

## 1. PyTorch → ExecuTorch: export → quantize → delegate → validate parity

The whole point of this flow is the **last step**: the `.pte` is a *new* model and you must prove it
matches the source on your task, on the device.

```python
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
# Backend partitioner — XNNPACK is the always-available CPU baseline.
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# 1) Capture a clean exported graph. Inspect it for unsupported / unexpected ops NOW, not in the field.
exported = export(model, example_inputs)

# 2) (Optional) post-training quantization with ExecuTorch's quantizer so the produced INT8 kernels
#    match what the backend actually implements. Calibrate on REPRESENTATIVE edge data, not ImageNet.
#    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
#    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
#        XNNPACKQuantizer, get_symmetric_quantization_config)
#    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
#    prepared = prepare_pt2e(exported.module(), quantizer)
#    for x in calibration_loader:          # real device-distribution samples
#        prepared(x)
#    quantized = convert_pt2e(prepared)
#    exported = export(quantized, example_inputs)

# 3) Lower to the edge dialect and DELEGATE to a backend. Swap XnnpackPartitioner for
#    CoreMLPartitioner / QnnPartitioner / etc. for the target NPU.
edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
prog = edge.to_executorch()
with open("model.pte", "wb") as f:
    f.write(prog.buffer)

# 4) PARITY GATE — compare source vs lowered outputs on the SAME inputs, then the task metric.
from executorch.runtime import Runtime
runtime = Runtime.get()
program = runtime.load_program("model.pte")
method = program.load_method("forward")

import torch
max_abs = 0.0
for x, in eval_loader:                      # representative held-out set
    ref = model(x)
    out = method.execute([x])[0]
    max_abs = max(max_abs, (ref - out).abs().max().item())
print("max abs diff:", max_abs)             # within tolerance? else debug layer-by-layer
# Then compute the actual TASK metric (accuracy / mAP / WER) source-vs-pte and gate the release on it.
```

Key points:
- Inspect the exported/partitioned graph for **unsupported ops at export time**. An op that isn't
  delegated runs on CPU; a chain of them fragments the graph (see §2).
- **Run the parity check and task metric on the target device**, not just the host — kernels and the
  scheduler differ. A clean host parity does not guarantee on-device parity.
- For the **LiteRT** path instead: convert with `tf.lite.TFLiteConverter` (set
  `optimizations=[tf.lite.Optimize.DEFAULT]` and a `representative_dataset` for INT8 PTQ), then run the
  same parity + task-metric comparison via the LiteRT interpreter on-device.

---

## 2. Delegate / fallback selection note

A delegate claims the subgraphs it supports; the rest stays on CPU. **Each partition boundary costs a
tensor copy** (and sometimes a layout/dtype conversion). The failure you are guarding against:

```
# BAD: graph fragmented into many islands by a few unsupported ops.
#   [conv→relu] (NPU)  →copy→  [custom_op] (CPU)  →copy→  [conv] (NPU)  →copy→  [argmax] (CPU) ...
# The copies + repeated NPU setup can make this SLOWER than just running everything on CPU.

# GOOD: one large accelerated subgraph.
#   [conv→relu→...→conv→pool] (NPU, single partition)  →copy→  [argmax] (CPU tail)
```

Decision procedure:
1. **Inspect the partitioning** the partitioner/delegate produced (ExecuTorch lowering output; LiteRT
   delegate logs; ORT EP node placement). Count the boundaries.
2. If fragmented, find the ops forcing fallbacks and **rewrite the model to supported ops**, register a
   custom kernel, or move that op to a pre/post-processing step outside the graph.
3. Define an **explicit fallback order** and confirm each rung exists on your device floor:
   `NPU (Hexagon/ANE/EdgeTPU) → GPU (Vulkan/Metal/OpenCL/WebGPU) → CPU (XNNPACK)`. **The CPU path must
   always be viable** for the long tail of devices and driver versions.
4. **Benchmark each candidate path at steady state** on real hardware — the "accelerated" path is not
   always faster once copies and thermal throttling are counted. Pick by measurement.

Edge TPU / Coral special case: the model must be **fully INT8 and Edge-TPU-compiled**; any unsupported op
splits execution back to the host CPU at that point. Aim for a model that compiles end-to-end.

---

## 3. On-device LLM memory-budget note

On-device LLMs are bounded by **memory first, bandwidth second**. Plan the budget before you pick a model.

```
device RAM budget (for the app)
  = OS + other apps headroom            (you don't get all of physical RAM)
  − model weights                       (≈ params × bytes/param; 4-bit ≈ 0.5 byte/param + overhead)
  − runtime + activations               (runtime, scratch, embedding/lm_head buffers)
  − KV cache                            (the variable that bites on long context)

KV cache bytes ≈ layers × max_context × 2(K,V) × kv_heads × head_dim × bytes_per_elem
```

Worked sketch (illustrative — plug in your model's real config; do not treat as a benchmark):

- A 4-bit ~7–8B model's **weights** are on the order of a few GB. On a phone where the app may only get
  a few GB, that alone is most of the budget — leaving little for KV + activations.
- The **KV cache** scales linearly with `max_context`. Doubling context doubles KV; a long-context config
  can push KV toward the size of the weights.

Levers, in order of leverage:
1. **Cap `max_context`** to what the use case needs — the cheapest, biggest KV saving.
2. **Quantize the KV cache** (e.g. INT8/INT4 KV) — large KV reduction for modest quality cost; eval it.
3. **Prefer GQA/MQA models** (fewer `kv_heads`) — shrinks KV structurally (architecture-level).
4. **Quantize weights harder (INT4)** — but **eval the quality cliff on your task**, not a vibe check.
5. **Size for the lowest-RAM device** you support; if it doesn't fit there, pick a smaller model.

Validation gate: run your **task eval** (not just perplexity) on the on-device quantized model with the
chosen context/KV settings, on the **lowest-spec device**, at **steady state** (after the SoC has warmed
up and throttled). Ship only if it clears your bar there.

> As of 2026, 4-bit quantized ~7–8B-class models run interactively on high-end phones — **verify current
> viable sizes and speeds against measurements on your actual target devices; do not quote a number you
> have not measured.**

---

## Related

- `[[inference-optimization]]` — the quant/pruning/distillation/KV-quant theory these flows apply.
- `[[ml-frameworks]]` — PyTorch/`torch.export` source side.
- `[[ml-compilers-codegen]]` — the graph-lowering / op-partitioning / kernel layer underneath.
- `[[privacy-preserving-ml]]`, `[[multimodal-ml]]`, `[[serving-frameworks]]`.
