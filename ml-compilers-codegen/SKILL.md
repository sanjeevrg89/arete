---
name: ml-compilers-codegen
description: Deep expertise in ML compilers and code generation — how models are lowered to fast hardware
  kernels. Covers the compilation stack (graph capture → high-level IR → optimization passes → lowering →
  target codegen → runtime), MLIR (dialects, progressive lowering, reusable infrastructure) and
  StableHLO/HLO as the portable ML IR, XLA/OpenXLA (HLO pipeline, algebraic simplification, layout
  assignment, operator fusion, buffer assignment, XLA:GPU native-PTX-via-LLVM and Triton emitters, PJRT,
  AOT vs JIT, shape specialization/recompilation), Triton (tile model, TTIR→TTGIR→LLVM/PTX dialects),
  fusion in depth (vertical/horizontal/epilogue, FlashAttention, the memory wall), and torch.compile
  (Dynamo→AOTAutograd→Inductor→Triton), TensorRT, ONNX Runtime, IREE, TVM. Use when reading/debugging IR
  dumps (HLO, TTIR/TTGIR, FX/Inductor output), chasing recompilation or fusion failures, autotuning,
  writing a custom kernel/pass, or deciding when the compiler helps vs hurts. This is the compiler/IR/
  codegen deep dive beneath ml-frameworks and inference-optimization.
---

# ML Compilers & Code Generation

Apply the judgment of a compiler engineer who lowers frontier models to GPU/TPU kernels for a living —
who reads HLO and Triton IR before touching Python, knows exactly which optimization pass fused (or
failed to fuse) a chain, and can tell whether a compiler will help or hurt from the roofline alone. The
bar: **never fabricate a flag, an IR-op name, a pass name, or a benchmark.** When unsure whether
something is current, describe the concept and tell the reader to verify against current docs — flags,
dialects, and op names move every release (it is 2026).

## How to use this skill

1. **Read `ml-compilers-codegen-guide.md`** in this directory — the full reference (the stack, MLIR,
   XLA, Triton, fusion, the other compilers, debugging, anti-patterns). Apply it to the task.
2. For concrete artifacts to imitate — an HLO before/after fusion sketch, a minimal Triton tile kernel
   with its TTIR→TTGIR→PTX lowering note, and a torch.compile→Inductor→Triton trace note — read
   **`examples.md`**.
3. Match the surrounding stack's framework/compiler choice; apply the correctness, fusion, and
   shape-stability rules regardless. **Always dump and read the IR before theorizing, and measure
   (with warmup + device sync) before and after any change.**

## The essentials (full detail in `ml-compilers-codegen-guide.md`)

- **One pipeline, many names.** Every ML compiler is: framework program → **graph capture/tracing** →
  **high-level IR** (HLO/StableHLO, FX/ATen, Relay) → **target-independent passes** (algebraic
  simplification, CSE/DCE, **layout assignment**) → **fusion** → **progressive lowering** → **target
  codegen** (PTX/SASS, TPU ISA, Triton) → **runtime** (PJRT, Inductor, TRT engine). Debug at the IR.
- **Why ML needs its own compilers.** Instructions are whole tensor ops over arrays; targets have
  explicit memory hierarchies where **data movement, not arithmetic, dominates**. The compiler's job is
  to stop paying for HBM round-trips and to specialize on shapes/layouts a generic library can't.
- **Fusion is the headline optimization.** No intermediate HBM materialization, many ops → **one
  kernel**. **Vertical** (producer→consumer), **horizontal**, and **epilogue** (fold bias/activation/
  cast into a GEMM). Fusion attacks the **memory wall**: it raises arithmetic intensity by keeping
  intermediates in registers/SMEM. **FlashAttention** is the canonical fused, IO-aware, tiled kernel
  (online softmax, no N×N materialization).
- **MLIR is reusable compiler *infrastructure*, not a compiler.** **Dialects** are op sets at one
  abstraction level; you **progressively lower** dialect by dialect (e.g. `linalg` → `scf`/`vector` →
  `gpu`/`llvm`). The thesis: build a new backend by reusing the pass manager / lowering paths, not
  reinventing them. XLA:GPU, Triton, Torch-MLIR, IREE are all MLIR-based.
- **StableHLO/HLO is the portable ML IR.** **StableHLO** is the versioned interchange dialect both JAX
  and PyTorch/XLA emit; **HLO** is XLA's own working IR it lowers into.
- **XLA is the archetype.** HLO → algebraic simplification → layout assignment → **operator fusion** →
  buffer assignment (liveness-based reuse/donation) → backend. **XLA:GPU emits native PTX via the LLVM
  NVPTX backend and uses Triton-based emitters** for matmul/GEMM-fusion and some norm/attention shapes;
  heavy GEMMs may dispatch to cuBLAS/cuDNN. Frontends reach it through **PJRT**.
- **Shape specialization = recompilation.** XLA/`jit` compile per concrete signature; a new shape ⇒ a
  new compile. The #1 latency trap. **Bucket/pad** shapes, enable the **persistent compilation cache**,
  keep shapes static, and don't mark frequently-varying values `static`.
- **Triton democratizes custom kernels at the tile level.** You program blocks of the output; the
  compiler handles warps/coalescing/SMEM/pipelining. Lowering: **TTIR → TTGIR (adds GPU layout/encoding)
  → LLVM → PTX/SASS**; the hardware mapping is decided at TTIR→TTGIR. It sits between calling cuBLAS and
  writing CUTLASS/CUDA.
- **torch.compile = Dynamo (bytecode→FX) → AOTAutograd (joint fwd/bwd) → Inductor → Triton (GPU).**
  Inductor's fast path *is* Triton. **Graph breaks** (`.item()`, prints, data-dependent `if`, unsupported
  ops) shatter fusion; recompiles thrash on changing shapes. `mode="max-autotune"` trades compile time.
- **Both PyTorch and JAX target XLA.** JAX is XLA-native; PyTorch reaches it via PyTorch/XLA (`torch_xla`,
  lazy-tensor → StableHLO, the standard TPU path) and as a `torch.compile` backend.
- **Read the IR; don't fight the compiler blind.** Dump HLO (`.lower().compile().as_text()`,
  `XLA_FLAGS=--xla_dump_to=...`), the torch.compile pipeline (`TORCH_COMPILE_DEBUG`,
  `TORCH_LOGS`/`output_code`, `torch._dynamo.explain`), and Triton stages — verify exact flag names per
  version. Confirm a chain actually fused and that layout didn't insert `transpose`/`copy`.
- **Compiler helps vs hurts.** Helps: many small/memory-bound fusible ops, launch-overhead cutting,
  static steady state. Hurts: one-big-GEMM workloads (nothing to fuse), per-step shape changes,
  heavy data-dependent control flow. Custom kernel/pass **only after the IR proves** the compiler left
  value on the table — and prefer Triton/Pallas over raw CUDA.

## Related skills

- `[[ml-frameworks]]` — PyTorch/JAX/XLA/GPU/TPU primitives and the roofline this skill compiles for;
  that one *introduces* torch.compile/XLA/Triton, this one is the IR/codegen deep dive beneath it.
- `[[inference-optimization]]` — uses these compilers at the engine boundary (AOT engines, fixed shapes).
- `[[gpu-performance-engineering]]` — the hardware/memory-hierarchy/Nsight layer the codegen targets.
- `[[maxtext-jax-llm]]` — a JAX/XLA LLM codebase where shape-stability and HLO reading pay off directly.
- `[[training-frameworks]]` — DDP/FSDP/Megatron/MaxText recipes that lean on compiled steps.
- `[[aiml-on-kubernetes]]` — running compiled training/inference workloads on Kubernetes/GKE (umbrella).
