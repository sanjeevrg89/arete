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

---

# Reference — ml-compilers-codegen

# ML Compilers & Code Generation — Deep Reference

This is the craft of turning a model — a graph of huge tensor ops — into fast machine code for diverse
accelerators. It is the deep version of what [[ml-frameworks]] introduces and what
[[inference-optimization]] leans on at the engine boundary. The work is: read the IR, understand the
passes, know what fuses and why, and intervene only where you can beat or unblock the compiler.

> The ecosystem moves fast (it is 2026). XLA, MLIR, Triton, Inductor, and TensorRT change flags, op
> names, and dialect details across releases. Treat every specific flag / IR-op / pass name below as
> *verify against current docs for your version*. The mental models are stable; the surface is not.

---

## 1. Why ML needs domain-specific compilers

A general compiler (LLVM, GCC) optimizes scalar/vector code over a CPU/GPU ISA. An ML compiler operates
one level up: its "instructions" are whole tensor ops (matmul, conv, softmax, reduce, broadcast) over
multi-dimensional arrays, and its targets are massively parallel accelerators with explicit memory
hierarchies (registers → shared/SMEM → L2 → HBM) where **data movement, not arithmetic, is usually the
bottleneck**. Three forces make a dedicated compiler worth it:

- **Op granularity & fusion.** Naively, each framework op (every add, every layernorm) becomes a kernel
  launch that reads inputs from HBM and writes outputs back to HBM. For memory-bound ops that HBM
  round-trip *is* the cost. A compiler that sees the whole graph can **fuse** chains of ops into one
  kernel that keeps intermediates in registers/SMEM — the single most impactful ML optimization.
- **Hardware diversity.** The same model must run on NVIDIA GPUs, AMD GPUs, TPUs, and assorted NPUs,
  each with different ISAs, memory systems, and matmul units. A compiler with a portable IR amortizes
  the frontend across backends instead of rewriting kernels per device.
- **Specialization.** Knowing the static shapes, dtypes, and layouts lets the compiler pick tile sizes,
  layouts, and algorithms a generic library cannot. The cost is recompilation when those change.

The arithmetic-intensity / roofline model from [[ml-frameworks]] and [[gpu-performance-engineering]] is
the lens: an op is **compute-bound** (feed the tensor cores / MXU) or **memory-bound** (fuse it, raise
intensity). The compiler's main job for the long tail of memory-bound ops is to stop paying for HBM
traffic.

## 2. The compilation stack (the mental model)

Every ML compiler is some instance of this pipeline. Naming differs; the stages do not:

```
Framework program (PyTorch / JAX / TF)
        │  graph capture / tracing      (Dynamo, jax.jit tracing, tf.function)
        ▼
High-level IR  (graph of tensor ops)    (HLO/StableHLO, ATen/FX graph, Torch-MLIR, Relay/Relax)
        │  target-independent passes     (algebraic simplification, CSE, DCE, const-fold, layout)
        ▼
Optimized high-level IR
        │  fusion + scheduling           (decide which ops become one kernel)
        ▼
Lowering  (progressive: high-level → mid → low dialects)
        │
        ▼
Target codegen   (PTX/SASS, ROCm/LLVM, TPU ISA, C++/CUDA, Triton)
        │
        ▼
Runtime  (buffers, kernel dispatch, collectives, streams)   (PJRT, Inductor runtime, TRT engine, ORT)
```

Key distinctions to keep straight:

- **Tracing vs source-level capture.** JAX and `tf.function` *trace*: they run the Python with abstract
  values and record the ops touched — so Python control flow on traced values must be expressed with IR
  primitives (`lax.cond`/`scan`), not Python `if`. PyTorch Dynamo does **bytecode-level** capture and can
  *graph-break* back to eager on anything it cannot trace.
- **AOT vs JIT.** JIT (jax.jit, torch.compile, XLA-on-demand) compiles on first call and caches by a
  signature (shapes/dtypes/static args); great for iteration, pays a first-call latency and recompiles on
  signature change. AOT (XLA AOT, TensorRT engine build, IREE compile-to-vmfb) compiles ahead of time to
  a deployable artifact — predictable, no warmup, but you must know shapes (or build for shape ranges).
- **The IR is the contract.** Almost all real debugging happens by dumping and reading the IR at each
  stage, not by guessing from Python. If you take one habit from this skill: **dump the IR.**

## 3. MLIR — the multi-level IR infrastructure

MLIR (`mlir.llvm.org`, part of the LLVM project) is not a compiler; it is **reusable compiler
infrastructure** for building them. Its thesis: most ML/accelerator compilers re-implemented the same
machinery (an IR, a pass manager, pattern rewriting, a printer/parser) at every abstraction level. MLIR
provides that machinery once, parameterized by **dialects**.

- **Dialect** — a namespaced set of ops, types, and attributes representing one abstraction level. Ops
  carry semantics (and verifier rules) so passes can reason about them. Examples in the ecosystem:
  `func`, `arith`, `scf` (structured control flow), `linalg` (structured linear-algebra/named ops),
  `tensor`, `memref` (buffers), `vector`, `gpu`, `llvm` (maps to LLVM IR), plus domain dialects like
  `stablehlo`, `tosa`, and Triton's `tt`/`ttg`.
- **Progressive lowering.** Instead of one giant frontend→backend jump, you lower **gradually**, dialect
  by dialect: a high-level op (e.g. `linalg.matmul`) is rewritten into loops (`scf`/`affine`), then
  vector ops, then `gpu`/`llvm`, each step a small verifiable rewrite. Different parts of the program can
  sit at different levels simultaneously (the "multi-level" in the name).
- **Transformation as pattern rewriting.** Passes are built from rewrite patterns over the typed ops;
  the framework drives them. The payoff is the **reusable-infrastructure thesis**: a new accelerator
  backend reuses MLIR's pass manager, bufferization, the `linalg`/`vector` lowering paths, and LLVM
  codegen, instead of reinventing them.

You rarely write raw MLIR by hand for production models, but you *read* it constantly (XLA's GPU backend,
Triton, Torch-MLIR, and IREE are all MLIR-based) and you write MLIR when authoring a custom pass or
dialect.

### StableHLO / HLO — the portable ML IR

**HLO** (High-Level Optimizer IR) is XLA's classic op set — a smallish, well-specified algebra of tensor
ops (dot/convolution/reduce/broadcast/transpose/dynamic-slice/...). **StableHLO** is the MLIR dialect
that serves as the **portable, versioned serialization boundary** between ML frameworks and compilers in
the OpenXLA project: PyTorch (via Torch-XLA / `torch_xla`) and JAX both emit StableHLO, which compilers
(XLA, IREE, vendor backends) consume. Think of StableHLO as the stable interchange contract and HLO as
XLA's own working representation it lowers into. (StableHLO's compatibility/versioning guarantees
evolve — verify current status.)

## 4. XLA — the canonical ML compiler

XLA (Accelerated Linear Algebra; now developed in the OpenXLA project, `openxla.org`) is the reference
design and the one to understand deeply, because both JAX and PyTorch can target it and its pipeline is
the archetype.

### Pipeline (HLO in → optimized HLO → backend codegen)

1. **HLO graph** is produced from the frontend (StableHLO → HLO).
2. **Target-independent optimization** rewrites the HLO graph. The classic passes:
   - **Algebraic simplification** — constant folding, identity elimination (`x*1`, `x+0`, `transpose ∘
     transpose`), reassociation, simplifying `reshape`/`broadcast` chains.
   - **Common-subexpression elimination (CSE)** and **dead-code elimination (DCE)**.
   - **Layout assignment** — choose the physical memory layout (dimension order / minor-to-major) for
     each array so consumers and producers agree, minimizing inserted `transpose`/`copy` ops. Bad layout
     decisions are a real and under-appreciated source of slow kernels and extra HBM traffic.
   - **Operator fusion** — the headline pass (Section 6).
3. **Backend lowering & codegen.** For XLA:GPU this targets NVIDIA via LLVM: most fused elementwise/
   reduction regions are emitted as **native PTX through the LLVM NVPTX backend**, and XLA also has
   **Triton-based emitters** for important patterns (notably matmul/“GEMM-fusion” and some normalization/
   attention-shaped fusions), generating Triton IR that lowers to PTX. Heavy GEMMs/convs frequently
   dispatch to vendor libraries (cuBLAS / cuDNN) when that wins. (Which patterns go to which emitter is a
   version-specific detail — see `openxla.org/xla/gpu_architecture` and verify.)
4. **Buffer assignment** — XLA allocates and *reuses* buffers (liveness/interference analysis) so that
   the program runs in a bounded memory footprint, deciding what can be donated/aliased in place. This is
   why JAX's `donate_argnums` matters: it lets XLA reuse an input buffer for an output.
5. **Runtime: PJRT.** PJRT is the portable device/runtime API that frontends call to compile and execute
   — it abstracts "compile this StableHLO, give me an executable, run it on these buffers on this device"
   across GPU/TPU/CPU plugins. JAX and PyTorch/XLA both go through PJRT.

### Shape specialization & recompilation

XLA compiles for **concrete shapes**. Each distinct input signature (shapes × dtypes × static args) is a
distinct compiled executable, cached by that signature. New shape ⇒ new compile. This is the #1 latency
trap in JAX/XLA serving and training:

- Symptoms: periodic latency spikes, growing compile-cache, CPU-bound steps that should be GPU-bound.
- Fixes: **pad/bucket** dynamic dims to a small set of sizes; enable the **persistent compilation cache**
  so warm starts skip compile; keep batch/seq shapes static; use `jax.jit(static_argnums=...)` only for
  genuinely-constant args (each distinct value is a separate compile). Diagnose with the compilation/
  recompilation logs (e.g. `jax_log_compiles`) and XLA dump flags — verify exact flag names.

## 5. Triton — democratized tile-level kernels

Triton (`triton-lang.org`, OpenAI) lets you write custom GPU kernels in Python at the **tile** level: you
program in terms of blocks of the output and the data they read, and the compiler handles the
intra-tile detail — thread/warp assignment, memory coalescing, shared-memory staging, and (much of) the
software pipelining — that you'd hand-tune in raw CUDA/CUTLASS. It is the sweet spot between "call cuBLAS"
and "write CUDA": you get most of the performance with a fraction of the effort, which is why it has
become the default substrate for custom kernels (and the codegen target for PyTorch Inductor).

### Programming model

A `@triton.jit` kernel is launched over a **grid** of program instances (`tl.program_id`). Each instance
computes one tile: it builds index ranges (`tl.arange`), loads blocks with masks (`tl.load(ptr + idx,
mask=...)`), does math on register-resident tiles (`tl.dot` for matmul-on-tile, which uses tensor cores),
and `tl.store`s the result. `tl.constexpr` block sizes are compile-time constants the autotuner sweeps
(`@triton.autotune` over `triton.Config`s). See [[examples.md]] for a minimal kernel.

### The Triton lowering pipeline (MLIR dialects)

Triton is itself built on MLIR. The lowering chain is the thing to know when you read a Triton dump:

```
Python @triton.jit
   ▼  (frontend)
TTIR    — Triton IR        : tile-level, hardware-agnostic ops (tt dialect): tt.dot, tt.load, tt.store
   ▼  (analysis: pick layouts, warps, pipelining)
TTGIR   — Triton GPU IR    : adds GPU layout/encoding attributes (ttg dialect) — how tiles map to
                             warps / threads / shared memory, swizzling, pipeline stages
   ▼  (convert to LLVM dialect → LLVM IR)
LLVM IR  ──► PTX  ──► SASS  (via NVPTX / ptxas; ROCm path is analogous on AMD)
```

The interesting optimization (vectorization, shared-memory layout, pipelining, register allocation
pressure) happens at the **TTIR → TTGIR** transition, where the compiler decides how the logical tile
maps onto the hardware. (Exact dialect/pass names track the Triton release — verify.) Relation to
**CUTLASS/CUDA**: CUTLASS is NVIDIA's C++ template library of highly-tuned GEMM/conv building blocks
(closer to the metal, more control, steeper effort); Triton trades some peak performance for
productivity and portability of the kernel source.

## 6. Fusion in depth — attacking the memory wall

Fusion is the heart of ML compilation. The roofline says memory-bound ops are limited by HBM bandwidth;
fusion's entire purpose is **fewer HBM round-trips** by keeping intermediates in registers/SMEM and
emitting one kernel instead of many.

### Flavors

- **Vertical (producer→consumer) fusion.** Fuse a chain `a → b → c` (e.g. `matmul → bias → gelu`) so the
  intermediate `b` is never written to HBM. The dominant pattern.
- **Horizontal fusion.** Combine independent ops that share an input (or are tiny) into one kernel to cut
  launch overhead and reuse the loaded operand.
- **Epilogue fusion.** Fuse the elementwise tail (bias add, activation, scaling, residual, even a cast)
  *into* a preceding heavy op like a GEMM, so the GEMM's output is post-processed in-register before it
  ever hits HBM. This is where Triton/CUTLASS GEMM emitters and `torch.compile`'s Inductor shine.

### What blocks fusion

- A **materialization boundary** the compiler can't see through: a graph break (Dynamo falling back to
  eager), an op with no fusible lowering, a custom op the compiler treats as opaque.
- **Incompatible iteration spaces / layouts** between producer and consumer (a transpose or reduction in
  the middle that changes the access pattern) — the fused kernel would need an awkward or impossible
  schedule.
- **Reuse / multiple consumers** where materializing once is cheaper than recomputing in each fusion.
- **Dynamic shapes** that prevent the compiler from proving the tiling is valid.
- **Control flow / data-dependent shapes** inside the region.

### FlashAttention — the canonical fused kernel

Standard attention computes `S = QKᵀ`, `P = softmax(S)`, `O = PV`, materializing the `N×N` score matrix
`S`/`P` in HBM — O(N²) memory traffic that dominates for long sequences and makes attention brutally
memory-bound. **FlashAttention** (Dao et al.) is an **IO-aware**, **tiled**, **fused** kernel: it streams
Q/K/V in blocks through SRAM, computes attention with the **online-softmax** running-rescaling trick so
it never materializes the full `S`/`P` in HBM, and uses recomputation in the backward pass instead of
storing the big intermediate. The result is a single fused kernel that turns a memory-bound O(N²)-traffic
op into something far closer to compute-bound — the textbook demonstration of why fusion + IO-awareness
matters. It is the reason the rest of the stack works hard to either generate FlashAttention-shaped
kernels or call a hand-tuned one. (Cite: Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022; and the FlashAttention-2 / -3
follow-ups — verify the exact citation/arXiv IDs.)

## 7. The other compilers & paths

| Path | Frontend | IR / mechanism | Codegen target | Use it for |
|------|----------|----------------|----------------|-----------|
| **torch.compile** | PyTorch eager | Dynamo → FX → AOTAutograd → **Inductor** | Triton (GPU) / C++/OpenMP (CPU) | Speeding up PyTorch training/inference with minimal code change |
| **XLA / OpenXLA** | JAX, PyTorch/XLA, TF | StableHLO → HLO → passes | PTX (LLVM) + Triton emitters; TPU ISA | JAX everywhere; TPU; whole-graph compile |
| **TensorRT** | ONNX, framework parsers | TRT network → builder → engine | tuned CUDA engine | Max NVIDIA inference throughput/latency (AOT engine) |
| **ONNX Runtime** | ONNX | graph + execution providers | CPU/CUDA/TensorRT/etc. EPs | Portable cross-framework inference |
| **IREE** | StableHLO / TOSA / Torch-MLIR | MLIR end-to-end → VM | CUDA/ROCm/Vulkan/CPU/... (.vmfb) | AOT, embedded/edge, broad backends |
| **TVM** | Relay/Relax + framework importers | Relay/Relax + TensorIR; AutoTVM/auto-scheduler | LLVM/CUDA/etc. | Research/edge autotuning, exotic targets |

### Inside torch.compile (the path you'll touch most in PyTorch)

`torch.compile(fn)` chains four stages — the same chain [[ml-frameworks]] introduces:

1. **Dynamo** — symbolically evaluates Python **bytecode**, capturing an **FX graph** of the tensor ops.
   Anything it can't capture triggers a **graph break**: it compiles the captured region, runs the
   un-capturable bit in eager, and resumes. Each break is a fusion boundary and a recompile risk.
2. **AOTAutograd** — traces the **joint forward+backward** graph ahead of time (so the backward is also
   compiled and fusible), normalizing to ATen ops, handling functionalization.
3. **Inductor** — the backend that does the scheduling/fusion and **emits Triton kernels** on GPU (and
   C++/OpenMP on CPU). It does the vertical/epilogue fusion described above.
4. Result is cached; `mode="max-autotune"` lets Inductor autotune (and consider template GEMMs) at the
   cost of compile time.

This is the crucial connection: **torch.compile's fast path *is* Triton.** When you debug a slow
`torch.compile` region, you read the generated Triton.

### How PyTorch and JAX both reach XLA

JAX is XLA-native (jit → StableHLO/HLO → XLA → PJRT). PyTorch reaches XLA two ways: **PyTorch/XLA**
(`torch_xla`, lazy-tensor tracing → StableHLO → XLA → PJRT, the standard TPU path), and as a backend
option under `torch.compile`. Same compiler, two frontends — which is why the IR/fusion/recompilation
lessons transfer.

## 8. Practical: reading IR, debugging, autotuning

**Always dump the IR before theorizing.** Concrete habits:

- **XLA / JAX:** print the compiled HLO with `jax.jit(f).lower(*args).compile().as_text()` (or
  `.lower(...).as_text()` for pre-optimization StableHLO). Set the XLA dump flag
  (`XLA_FLAGS=--xla_dump_to=/tmp/xla_dump`) to get HLO before/after each pass plus the emitted PTX/LLVM
  IR — verify the exact flag for your version. Read the post-fusion HLO to confirm chains actually fused
  (look for `fusion` computations) and that layouts didn't insert copies/transposes.
- **torch.compile:** `TORCH_COMPILE_DEBUG=1` (and `TORCH_LOGS="graph_breaks,recompiles,inductor"` or
  `output_code`) dumps graph breaks, recompile reasons, the FX graph, and the **generated Triton** —
  verify current env-var/log names. `torch._dynamo.explain(fn)(*args)` enumerates graph breaks.
- **Triton:** dump the IR stages (TTIR/TTGIR/LLVM/PTX) via the Triton cache / `MLIR_ENABLE_DUMP`-style
  knobs (verify), or inspect `kernel.asm["ptx"]`/`["ttgir"]` on a compiled kernel.
- **Autotuning:** Triton `@triton.autotune` sweeps tile/warp/stage configs; Inductor `max-autotune`
  benchmarks template vs Triton GEMMs; XLA has autotuning for some emitters. Autotuning is per-shape —
  cache results and beware it inflating compile time. Always **measure with proper device sync and
  warmup**; a single un-synced timer lies.

**When the compiler helps vs hurts:**

- **Helps:** many small/memory-bound ops it can fuse (norms, activations, elementwise glue, attention
  epilogues); cutting Python/launch overhead; static-shaped steady-state training and serving.
- **Hurts / no-op:** workloads already dominated by one big cuBLAS GEMM (nothing to fuse); shapes that
  change every step (recompile thrash > any kernel win); heavy host-side/data-dependent control flow
  (graph breaks). Measure; don't assume `torch.compile`/`jit` is free speed.

**Writing a custom kernel or pass — when:** only after the IR proves the compiler left value on the
table (an unfused chain it *can't* fuse, a shape the autotuner handles poorly, a novel op). Prefer a
**Triton kernel** (or Pallas on TPU/GPU, see [[ml-frameworks]]) over raw CUDA; reach for a **custom MLIR
pass / XLA custom-call** only for a genuinely new pattern. Register custom ops so the compiler treats the
boundary correctly (it will *not* fuse across an opaque custom op).

## 9. Anti-patterns / gotchas

- **Dynamic-shape thrashing.** Variable batch/seq length recompiling every step. Bucket/pad shapes;
  enable persistent compile cache; mark dynamic dims (`dynamic=True`, `mark_dynamic`) or use symbolic
  shapes deliberately. This is the most common self-inflicted ML-compiler wound.
- **Fighting the compiler blind.** Rewriting Python to "force" fusion without ever reading the IR. The
  IR tells you exactly what fused and why something didn't — read it first.
- **Hand-writing kernels the compiler would have fused.** Maintaining a bespoke CUDA/Triton kernel for a
  norm/activation chain that Inductor/XLA fuses for free. Verify the compiler *isn't* already doing it.
- **Ignoring layout.** Letting transposes/copies pile up because producer/consumer layouts disagree;
  a "slow matmul" is often a layout problem (extra `transpose`/`copy` in the HLO), not a slow GEMM.
- **Graph breaks you didn't notice.** A stray `.item()`, `print`, data-dependent `if`, unsupported op,
  or numpy call shatters a `torch.compile` region into many small compiled+eager pieces, killing fusion.
  Count and eliminate breaks.
- **static_argnums abuse in JAX.** Marking a frequently-varying value static ⇒ a fresh compile per value.
- **Trusting wall-clock without sync/warmup.** GPU work is async; un-synced timing and cold-cache runs
  both lie. Warm up, synchronize, and measure the steady state.
- **Disabling the compiler at the first bug** instead of reading the dump — you lose all the free fusion.

## 10. Performance & scale notes

- The win is dominated by **HBM traffic reduction** (fusion) and **kernel-launch reduction** (fewer,
  bigger kernels), plus picking the right GEMM/conv algorithm and layout. Quantify with the roofline and
  with measured bytes-moved, not vibes.
- **Compile time is a real budget** at scale: thousands of distinct shapes × autotune = minutes-to-hours
  of compilation. Persistent caches, shape bucketing, and AOT artifacts (TRT engines, IREE vmfb, XLA AOT)
  are how production keeps warmup bounded.
- For serving, **AOT + fixed shape buckets** usually beats JIT-on-demand: predictable latency, no
  first-request cliff. See [[inference-optimization]] and [[serving-frameworks]] for the engine layer.
- Profiling is mandatory and tool-specific: device profilers + Nsight (GPU) / TPU profiler; correlate
  kernel time back to the fused HLO/Triton region. See [[gpu-performance-engineering]].

## 11. Version awareness

Everything flag-, dialect-, and op-name-specific here drifts across releases. Before relying on a
specific `XLA_FLAGS`/`TORCH_LOGS` name, a Triton dialect/pass name, an HLO op spelling, a StableHLO
compatibility guarantee, or which fusion an emitter produces — **check the current docs for your exact
versions**. The architectural mental model (capture → high-level IR → passes → fusion → lowering →
codegen → runtime; roofline-driven fusion; shape specialization) is stable and is what to reason from.

## 12. Canonical references (verify current)

- OpenXLA: project site `openxla.org`; **XLA:GPU architecture** `openxla.org/xla/gpu_architecture`;
  XLA aliasing/flags/HLO docs under `openxla.org/xla`.
- StableHLO: `openxla.org/stablehlo` (spec, versioning/compatibility).
- MLIR: `mlir.llvm.org` (language ref, dialects — `linalg`, `vector`, `gpu`, `scf`, `affine`; pass
  infrastructure; "MLIR: A Compiler Infrastructure for the End of Moore's Law", Lattner et al. — verify
  the citation).
- Triton: `triton-lang.org` (tutorials, language/`tl` reference, MLIR dialect notes); the Triton GitHub
  repo for the TTIR/TTGIR passes.
- PyTorch: `pytorch.org/docs` — `torch.compile` / TorchDynamo / TorchInductor docs; the
  `torch._dynamo`/`torch._inductor` and `torch.fx` references.
- JAX: `docs.jax.org` — JIT/AOT, "thinking in JAX", compilation cache, `lower`/`compile`/`as_text`.
- FlashAttention: Dao et al., NeurIPS 2022, and FlashAttention-2/-3 follow-ups — **verify exact arXiv
  IDs / citations** before quoting.
- TVM `tvm.apache.org`; IREE `iree.dev`; ONNX `onnx.ai` and ONNX Runtime `onnxruntime.ai`; NVIDIA
  TensorRT and CUTLASS docs on `docs.nvidia.com` / their GitHub repos.

---

# Examples — ML Compilers & Code Generation

Worked, imitable artifacts. These are *illustrative sketches* of the IR shapes and patterns — exact op
spellings, dialect attributes, and flag names drift across versions, so **always confirm against an
actual dump from your toolchain**. The point is to recognize the *shape* of what you'll see.

---

## 1. HLO operator fusion — before / after

A tiny graph: `out = gelu(x @ w + b)`. Without fusion, each op is its own kernel writing to HBM and the
next op reads it back. With fusion, the elementwise tail collapses into one fused computation (often the
GELU/bias **epilogue** is fused with — or right after — the dot).

### Before fusion (conceptual unoptimized HLO; one HBM round-trip per op)

```
HloModule matmul_gelu

ENTRY main {
  x = f32[1024,4096] parameter(0)
  w = f32[4096,4096] parameter(1)
  b = f32[4096]      parameter(2)

  dot   = f32[1024,4096] dot(x, w),
            lhs_contracting_dims={1}, rhs_contracting_dims={0}   // → HBM
  bcast = f32[1024,4096] broadcast(b), dimensions={1}            // → HBM
  add   = f32[1024,4096] add(dot, bcast)                         // reads dot,bcast; → HBM
  gelu  = f32[1024,4096] <gelu-expansion of add>                 // reads add;       → HBM
  ROOT  = gelu
}
```

`add` and the GELU expansion are pure elementwise: each reads ~16 MB from HBM and writes ~16 MB back,
purely to apply cheap arithmetic — textbook memory-bound waste.

### After fusion (conceptual; the elementwise tail becomes one fused kernel)

```
HloModule matmul_gelu

fused_epilogue {
  p_dot = f32[1024,4096] parameter(0)
  p_b   = f32[4096]      parameter(1)
  bcast = f32[1024,4096] broadcast(p_b), dimensions={1}
  add   = f32[1024,4096] add(p_dot, bcast)
  ROOT  = f32[1024,4096] <gelu-expansion of add>      // bias+gelu stay in registers; no HBM for add
}

ENTRY main {
  x = f32[1024,4096] parameter(0)
  w = f32[4096,4096] parameter(1)
  b = f32[4096]      parameter(2)
  dot = f32[1024,4096] dot(x, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT = f32[1024,4096] fusion(dot, b), kind=kLoop, calls=fused_epilogue
}
```

What to verify in a **real** dump (`XLA_FLAGS=--xla_dump_to=/tmp/xla`, or
`jax.jit(f).lower(x,w,b).compile().as_text()`):

- A `fusion` computation exists and the elementwise ops are *inside* it (the intermediate `add` no longer
  appears as a top-level array written to HBM).
- No stray `transpose`/`copy` inserted around the `dot` — that would signal a **layout** mismatch, often
  a bigger cost than the fusion saved. Check `layout`/minor-to-major if you see them.
- On XLA:GPU the `dot` itself may lower to a Triton GEMM emitter or dispatch to cuBLAS, sometimes with
  the epilogue fused into the GEMM rather than a separate `kLoop` fusion — which path you get is
  version- and shape-dependent; read the emitted PTX/Triton to confirm.

---

## 2. Minimal Triton tile kernel + the TTIR→TTGIR→PTX lowering note

A fused `y = x + bias` over a 1-D tensor (the "hello world" of Triton: one program instance per tile,
masked load/store). Real numerics aside, this shows the tile programming model.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_bias_kernel(x_ptr, bias_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid     = tl.program_id(axis=0)            # which tile this instance owns
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offsets < n_elements             # guard the ragged last tile
    x       = tl.load(x_ptr + offsets, mask=mask)
    bias    = tl.load(bias_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + bias, mask=mask)   # x+bias stays in registers

def add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)   # 1-D launch grid
    add_bias_kernel[grid](x, bias, y, n, BLOCK=1024)
    return y
```

A more realistic kernel adds `@triton.autotune` over several `triton.Config({"BLOCK": ...},
num_warps=..., num_stages=...)` and uses `tl.dot` for the matmul-on-tile (which targets tensor cores).

### What the compiler does with it (lowering note)

```
@triton.jit (Python)
   │  frontend / AST → IR
   ▼
TTIR   (tt dialect)   tile-level, hardware-agnostic: tt.get_program_id, tt.make_range,
                      tt.load/tt.store (with masks), tt.addptr, arith.addf, (tt.dot for matmul)
   │  analysis: choose layouts/encodings, warps, software-pipeline stages
   ▼
TTGIR  (ttg dialect)  same ops + GPU layout/encoding attributes describing how the logical tile
                      maps to warps/threads and shared memory (blocked/mma encodings, swizzling,
                      #stages). This is where coalescing, SMEM staging, and pipelining are decided.
   │  convert-triton-gpu-to-llvm
   ▼
LLVM IR  ──►  PTX  ──►  SASS      (NVPTX backend + ptxas; the AMD path is analogous via ROCm/LLVM)
```

Inspect the stages on a compiled kernel (names/knobs vary by version — verify): `kernel.asm["ttir"]`,
`kernel.asm["ttgir"]`, `kernel.asm["llir"]`, `kernel.asm["ptx"]`, or dump-to-disk env knobs. When a
Triton kernel is slow, the interesting decisions (register pressure, SMEM layout, pipeline depth,
coalescing) are visible at **TTGIR** — read it before tweaking `num_warps`/`num_stages`/`BLOCK`.

---

## 3. torch.compile → Inductor → Triton trace note

`torch.compile` turns the same `gelu(x @ w + b)` into a captured graph, fuses the epilogue, and **emits
a Triton kernel** for the fused elementwise part (the GEMM may go to a tuned template / cuBLAS).

```python
import torch

@torch.compile  # mode="max-autotune" to let Inductor benchmark GEMM templates vs Triton
def block(x, w, b):
    return torch.nn.functional.gelu(x @ w + b)

x = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096,       device="cuda", dtype=torch.bfloat16)
out = block(x, w, b)   # first call compiles; subsequent calls hit the cache
```

### Tracing the pipeline (env-var/log names vary by version — verify)

- **See the stages and the generated code:**
  ```bash
  TORCH_LOGS="dynamo,graph_breaks,recompiles,inductor,output_code" python prog.py
  # or
  TORCH_COMPILE_DEBUG=1 python prog.py   # dumps FX graph, Inductor IR, and generated Triton to a debug dir
  ```
- **Stage by stage:**
  1. **Dynamo** captures an FX graph from Python bytecode. Check for **graph breaks** —
     `torch._dynamo.explain(block)(x, w, b)` enumerates them with reasons. Zero breaks here; a stray
     `.item()`/`print`/data-dependent `if` would split the region and kill fusion.
  2. **AOTAutograd** traces the joint forward+backward (so the backward is compiled & fusible too) and
     normalizes to ATen ops.
  3. **Inductor** schedules and **fuses** the bias-add + GELU, then **emits a Triton kernel** for that
     fused elementwise region (you'll see a `@triton.jit` `triton_poi_fused_...` / `triton_red_...`
     function in `output_code`). The `mm`/`addmm` may be a Triton GEMM template or a cuBLAS call
     depending on shape and `max-autotune`.
- **Recompilation check:** call `block` again with a *different* shape and watch the `recompiles` log
  fire — the signal to bucket/pad shapes or pass `dynamic=True` / `torch._dynamo.mark_dynamic`.

The takeaway: **torch.compile's fast path is Triton**, so debugging a slow compiled region means reading
the generated Triton (and confirming the fusion actually happened) — exactly the IR-first habit from the
guide. See [[ml-frameworks]] for the framework-level view and [[gpu-performance-engineering]] for
correlating the kernel back to Nsight.
