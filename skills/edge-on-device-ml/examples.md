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
