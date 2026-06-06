# AGENTS.md — ML Compilers & Code Generation

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ml-compilers-codegen-guide.md`** next to this file —
> read it before working on compiler/IR/codegen tasks. Concrete artifacts to imitate (HLO before/after
> fusion, a Triton tile kernel with its lowering note, a torch.compile→Inductor→Triton trace) are in
> **`examples.md`**. This file is the always-on summary.
>
> **Hard rule: never fabricate a flag, an IR-op name, a pass name, a dialect detail, or a benchmark.**
> Flags/dialects/op names drift every release (it is 2026) — if unsure, describe the concept and say
> "verify against current docs for your version." For any arXiv ID you're not certain of, name the work
> and say "verify the citation."

## When working on ML compilation / codegen, apply these by default:

- **Dump and read the IR before theorizing.** Almost every real bug is visible in the IR, not the
  Python. XLA/JAX: `jax.jit(f).lower(*args).compile().as_text()`, `XLA_FLAGS=--xla_dump_to=...`.
  torch.compile: `TORCH_COMPILE_DEBUG`, `TORCH_LOGS` (graph_breaks/recompiles/output_code),
  `torch._dynamo.explain`. Triton: dump TTIR/TTGIR/LLVM/PTX. Verify exact flag names per version.
- **One pipeline, many names:** capture/trace → high-level IR (HLO/StableHLO, FX/ATen) →
  target-independent passes (algebraic simplification, CSE/DCE, **layout assignment**) → **fusion** →
  progressive lowering → target codegen (PTX/SASS, TPU ISA, Triton) → runtime (PJRT, Inductor, TRT).
- **Fusion is the point.** Many ops → one kernel, no intermediate HBM materialization. Vertical /
  horizontal / **epilogue** (fold bias/activation/cast into a GEMM). It raises arithmetic intensity by
  keeping intermediates in registers/SMEM — it attacks the memory wall. Confirm chains *actually* fused.
- **FlashAttention** is the canonical fused, IO-aware, tiled, online-softmax kernel (no N×N HBM
  materialization). It's why so much of the stack tries to generate or call attention-shaped fusions.
- **MLIR = reusable infrastructure, not a compiler.** Dialects = op sets per abstraction level;
  **progressive lowering** rewrites dialect by dialect. **StableHLO** = portable versioned interchange
  IR (JAX + PyTorch/XLA emit it); **HLO** = XLA's own working IR.
- **XLA archetype:** HLO → algebraic simplification → layout assignment → operator fusion → buffer
  assignment (liveness-based reuse/donation) → backend. **XLA:GPU emits native PTX via LLVM NVPTX and
  uses Triton-based emitters** (matmul/GEMM-fusion, some norm/attention); heavy GEMMs may go to cuBLAS/
  cuDNN. Reached through **PJRT**. Which emitter handles which pattern is version-specific — verify.
- **Shape specialization ⇒ recompilation.** New concrete shape = new compiled executable. The #1 latency
  trap. Bucket/pad shapes, enable the persistent compilation cache, keep shapes static, don't mark
  frequently-varying args `static`.
- **Triton:** tile-level kernels in Python; compiler handles warps/coalescing/SMEM/pipelining. Lowering
  **TTIR → TTGIR (adds GPU layout/encoding) → LLVM → PTX/SASS**; HW mapping decided at TTIR→TTGIR. Sits
  between cuBLAS and CUTLASS/CUDA. Autotune is per-shape — cache it.
- **torch.compile = Dynamo (bytecode→FX) → AOTAutograd (joint fwd/bwd) → Inductor → Triton (GPU) /
  C++ (CPU).** Inductor's fast path *is* Triton. Eliminate **graph breaks** (`.item()`, prints,
  data-dependent `if`, unsupported ops); they shatter fusion. Both PyTorch and JAX can target XLA.
- **When the compiler helps vs hurts.** Helps: many small/memory-bound fusible ops, launch-overhead
  cutting, static steady state. Hurts/no-op: single-big-GEMM workloads, per-step shape changes, heavy
  data-dependent control flow (graph breaks / recompiles).
- **Custom kernel/pass only after the IR proves it.** Don't hand-write a kernel the compiler would fuse;
  don't ignore layout (a "slow matmul" is often inserted `transpose`/`copy`). Prefer Triton/Pallas over
  raw CUDA; reach for a custom MLIR pass / XLA custom-call only for a genuinely new pattern.

## Definition of done for compiler/codegen changes
- IR dumped and read; the intended fusion/layout confirmed in the dump (not assumed).
- No new graph breaks / recompiles introduced; shapes stable or deliberately bucketed.
- Measured before/after with proper **warmup + device synchronization** (async work; un-synced timers lie).
- No fabricated flags/op names/benchmarks; fast-moving specifics flagged "verify against current docs."
