# AGENTS.md — Edge / On-Device ML

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`edge-on-device-ml-guide.md`** next to this file — read it
> before deploying, converting, or debugging a model for a device, and apply it. Concrete flows to
> imitate (PyTorch→ExecuTorch/TFLite export+quantize+parity, delegate/fallback selection, on-device-LLM
> memory budget) are in **`examples.md`**. This file is the always-on summary.
>
> Scope: running ML **off the datacenter** — phones, laptops, wearables, cameras, embedded, MCUs. The
> general compression theory is `[[inference-optimization]]`; this is the **deployment target**:
> runtimes, hardware, conversion pipeline, on-device validation. It is 2026 and this moves fast — name
> the tool, then **verify current APIs/op-support/benchmarks against the project's own docs.** Never
> fabricate a device benchmark number.

## When deploying ML to a device, apply these by default:

- **Pick edge for a constraint, not hype.** It buys latency, privacy (`[[privacy-preserving-ml]]`),
  offline, bandwidth, and $0 server cost — and costs you a fixed budget: limited RAM, no datacenter GPU,
  a power/thermal ceiling that throttles sustained throughput, and heterogeneous hardware you don't
  control. Budget for the **lowest-spec device** you support, not the flagship.
- **Choose the runtime by source framework + target OS.** **ExecuTorch** (PyTorch, multi-NPU via
  backend/delegate: XNNPACK CPU, Core ML/MPS, Qualcomm, Arm, Vulkan). **LiteRT** (successor to TF Lite;
  Android-first, cross-framework, NNAPI/GPU/Hexagon delegates). **Core ML** (Apple-only, Neural Engine
  via the OS). **ONNX Runtime Mobile/Web** (ONNX graphs; in-browser via WASM/WebGPU). **MediaPipe**
  (prebuilt vision/audio). LLMs: **llama.cpp/GGUF**, **MLC-LLM**, **MNN/NCNN**. MCUs: **TFLite Micro**.
- **The conversion pipeline is the real work:** source → exported graph (`torch.export`/SavedModel/ONNX)
  → target format (`.pte`/`.tflite`/`.mlpackage`/`.ort`). Every hop can drop ops or change numerics.
- **Quantize for the device with the runtime's own tooling, never assume server quant transfers.** Edge
  wants INT8 (sweet spot) / INT4 (LLM weights); the **quality cliff is task-dependent** — eval the
  quantized model **on the target task, on the device**. Technique depth: `[[inference-optimization]]`.
- **Delegates partition the graph; unsupported ops fall back to CPU.** Many fallbacks = many copy
  boundaries = possibly **slower than plain CPU**. Inspect partitioning; aim for one big accelerated
  subgraph; set an explicit fallback order (NPU→GPU→CPU) with a viable CPU path everywhere.
- **Find unsupported ops at export time**, never in the field. Confirm every op is supported or has an
  accepted fallback before committing to a target.
- **On-device LLMs are bounded by memory then bandwidth.** Budget = OS + app + 4-bit weights + runtime/
  activations + **KV cache**. Cap max context, quantize the KV cache, prefer GQA/MQA models, size for the
  lowest-RAM device.
- **Mind thermal throttling** — measure **steady-state** latency + energy on real hardware, not a burst.
- **Validate parity as the ship gate.** Same inputs through source vs on-device model; compare outputs
  (tolerance + top-k) **and** the task metric on a held-out set, **on the device**. Don't ship a
  regression. Debug numerical mismatch layer-by-layer to the first diverging layer.
- **Ship updates like software:** OTA-deliver models decoupled from app releases, version them against
  the runtime/op set, on-device A/B / canary, and always keep a **rollback** to last-good. Profile on
  real (including low-end) devices.
- **Multimodal is the common edge workload** (vision/speech/VLM — `[[multimodal-ml]]`); edge pairs with
  federated / private learning (`[[privacy-preserving-ml]]`).
- **Anti-patterns:** server-quant assumed to transfer; no post-conversion parity check; ignoring thermal
  throttling; shipping an unsupported-op model; INT4 with no target-task eval; no OTA/rollback;
  over-fragmented delegation; designing for the flagship only.

## Definition of done for an edge deployment
The model **converts** to the target format with **no unsupported ops** (or accepted fallbacks);
**delegation is consolidated** (not fragmented); **parity + task metrics validated on the target device**
after conversion/quant; **steady-state latency/memory/energy profiled** on representative (incl. low-end)
hardware; and an **OTA/versioning/rollback** path exists. Report honestly if any step is unverified.
