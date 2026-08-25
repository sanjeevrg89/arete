# AGENTS.md — Triton Kernel Authoring

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`triton-kernel-authoring-guide.md`** next to this file —
> read it before writing or optimizing a Triton kernel, and apply it. Full worked kernels to imitate (a
> masked fused vector op, an autotuned tiled matmul with `tl.dot`, a numerically-stable fused softmax —
> each with a `torch.allclose` check and a `do_bench` stub) are in **`examples.md`**. This file is the
> always-on summary.
>
> This is a **doer** skill: use it to **WRITE and optimize** a Triton GPU kernel. (Profiling is
> `gpu-performance-engineering`; how Triton lowers to IR is `ml-compilers-codegen`.)
>
> **Triton moves fast (it is 2026).** The tile model is stable; surface APIs are not. **Never fabricate a
> `tl.*` name or a decorator flag** — flag version-sensitive APIs "verify against current Triton docs"
> (block-pointer API `tl.make_block_ptr`/`tl.advance`/`boundary_check`/`padding_option`, the exact
> `triton.Config`/`@triton.autotune` keyword set, `do_bench` signature, non-NVIDIA backend semantics).

## When writing or optimizing a Triton kernel, apply these by default:

- **Tile + SPMD, not threads.** Write **one program that computes one output tile**; the runtime launches
  a **grid** of them and the compiler owns warps/coalescing/SMEM/pipelining/MMA. Tile size, `num_warps`,
  `num_stages` are your perf knobs.
- **Primitives:** `@triton.jit`; `tl.program_id(axis)` (tile index); `BLOCK_SIZE: tl.constexpr`
  (compile-time, usually powers of two); `tl.arange(0, BLOCK)`; pointer arithmetic with **strides** for
  multi-dim (`[:, None]`/`[None, :]` broadcast for 2-D tiles); `tl.load`/`tl.store`; `tl.dot`; `tl.sum`/
  `tl.max`; `tl.where` (never a Python `if` on block data).
- **Mask every boundary.** `mask = offs < n` on **both** load and store. Unmasked is correct only when the
  dim exactly divides the block — almost never. Masking bugs are the top silent-corruption source.
- **Launch with `triton.cdiv`** for the grid; the mask covers the partial last tile. Launch is async —
  `torch.cuda.synchronize()` before timing.
- **Follow the procedure, each step a gate:**
  1. Write a **PyTorch reference** (correctness oracle + speed baseline).
  2. Write the simplest correct **masked** kernel.
  3. **Validate:** `torch.testing.assert_close(out_tri, out_ref, rtol=1e-3, atol=1e-3)` on **multiple
     sizes incl. non-power-of-2, a size that doesn't divide the block, and a tiny size**, at the
     production dtype. Hard gate.
  4. **`@triton.autotune`** over block sizes × `num_warps` × `num_stages` with a sensible `key`; re-check
     correctness after.
  5. **Benchmark vs the reference** with `triton.testing.do_bench` (warmup/sync handled); report **GB/s**
     (memory-bound) or **TFLOP/s** (compute-bound: GEMM = `2*M*N*K`) vs the **roofline**, and vs
     `torch.compile` where relevant. State the win — or the honest no-win — with the metric.
- **`tl.dot` accumulates in fp32** (even fp16/bf16 inputs); cast down at the end.
- **Numerical stability:** softmax subtracts `tl.max` before `tl.exp`; norm/softmax reductions in fp32.
- **Fuse for the memory wall:** the reason to write a kernel is usually to fuse a chain PyTorch runs as
  separate HBM round-trips — keep intermediates in registers/SMEM, write only the result; epilogue fusion
  (bias+activation+cast) is highest leverage.
- **Coalesce** (innermost offset → contiguous memory; pass and use **strides**, don't assume row-major);
  **sweep** `num_warps`/`num_stages` (more isn't always better — spills/SMEM exhaustion tank occupancy).
- **Don't reinvent** a bare GEMM/elementwise that cuBLAS or Inductor already fuses well — write a kernel
  once you've shown bandwidth is left on the table.

## Definition of done
PyTorch reference exists · `torch.allclose`/`assert_close` passes at production dtype on multiple sizes
incl. non-power-of-2 + non-dividing + tiny · every boundary `tl.load`/`tl.store` masked · matmul/reduction
accumulators fp32, softmax/norm max-subtracted & reduced in fp32 · `@triton.autotune` applied and
correctness re-checked · `do_bench` benchmark vs reference reports GB/s or TFLOP/s vs roofline · any
version-sensitive API flagged "verify against current Triton docs."

## Anti-patterns to flag
Unmasked load/store on a non-dividing size · correctness checked only at a power-of-two size · `tl.dot`
accumulating in fp16/bf16 · softmax without max-subtraction · single cold timing (no warmup/sync/median) ·
Python `if` on block data · assuming row-major contiguity (no strides) · wrong `@triton.autotune` `key` ·
reinventing a library-tuned GEMM/elementwise · claiming a speedup vs an unfair baseline · inventing a
`tl.*` name or decorator flag instead of verifying it.
