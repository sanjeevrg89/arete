---
name: triton-kernel-authoring
description: Use this to WRITE and optimize a Triton GPU kernel — the doer's skill for producing working,
  correctness-checked, autotuned, benchmarked Triton code (not for profiling or reading IR — that's
  gpu-performance-engineering and ml-compilers-codegen). Covers the tile/SPMD programming model
  (@triton.jit, tl.program_id, BLOCK_SIZE as tl.constexpr, tl.arange, boundary masking on tl.load/tl.store,
  grid/launch with triton.cdiv, strides for multi-dim tiles, tl.dot for matmul/tensor-cores, tl.sum/tl.max
  reductions, epilogue fusion), the authoring procedure (PyTorch reference → write kernel → validate with
  torch.allclose on edge/non-power-of-2 sizes → @triton.autotune over block sizes/num_warps/num_stages →
  benchmark vs reference with do_bench and report GB/s or TFLOP/s vs roofline), performance patterns
  (coalescing, occupancy, fp32 accumulation, softmax max-subtraction, fusion and the memory wall), and the
  canonical kernels: fused elementwise/activation, fused softmax, layernorm/rmsnorm, tiled matmul, and the
  fused-attention/FlashAttention idea. Reach for it when you need to implement a custom or fused GPU op,
  speed up a memory-bound chain PyTorch runs as separate kernels, or write/debug a triton.jit kernel.
---

# Triton Kernel Authoring

Apply the judgment of an engineer who writes production Triton kernels for a living: who starts from a
correct PyTorch reference, diffs every kernel with `torch.allclose` (including on non-power-of-2 and masked
edge sizes), autotunes before believing a number, and reports GB/s or TFLOP/s against the roofline — never
a bare "it's faster." This skill **produces the kernel**; profiling it is [[gpu-performance-engineering]]
and how it lowers is [[ml-compilers-codegen]].

## How to use this skill

1. **Read `triton-kernel-authoring-guide.md`** in this directory — the full reference (tile/SPMD model,
   the authoring procedure with correctness + perf gates, performance patterns, the canonical kernel
   patterns, anti-patterns, and the definition of done). Apply it to the task.
2. For full worked kernels to imitate — a masked fused vector op, an autotuned tiled matmul with `tl.dot`,
   and a numerically-stable fused softmax, each with its `torch.allclose` check and a `do_bench` stub —
   read **`examples.md`**.
3. Match the surrounding codebase's conventions (dtypes, layouts, how kernels are wrapped/launched); apply
   the correctness-first and benchmark-honestly rules regardless. **Never fabricate a `tl.*` name or a
   decorator flag — flag version-sensitive APIs "verify against current Triton docs."**

## Essentials (full detail in `triton-kernel-authoring-guide.md`)

- **Tile + SPMD, not threads.** You write **one program that computes one tile** of the output; the
  runtime launches a **grid** of them and the compiler owns warps/coalescing/SMEM/pipelining/MMA. You
  reason about blocks and pointer arithmetic; tile size / `num_warps` / `num_stages` are your perf knobs.
- **The primitives:** `@triton.jit`; `tl.program_id(axis)` for the tile index; `BLOCK_SIZE: tl.constexpr`
  (compile-time, usually powers of two); `tl.arange(0, BLOCK)` for local indices; pointer arithmetic with
  **strides** for multi-dim (and `[:, None]`/`[None, :]` broadcasts for 2-D tiles); `tl.load`/`tl.store`.
- **Mask every boundary.** `mask=offs < n` on loads and stores. A kernel without masks is correct only
  when every dim exactly divides its block — almost never. **Masking bugs are the #1 silent corruption.**
- **Launch with `triton.cdiv`.** `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)`; the mask
  covers the partial last tile. Launch is async — **synchronize before timing.**
- **The procedure (in order, each a gate):** (1) write a **PyTorch reference** (oracle + baseline); (2)
  write the simplest correct masked kernel; (3) **validate with `torch.allclose`/`assert_close` on
  multiple sizes incl. non-power-of-2 and a non-dividing size**; (4) **`@triton.autotune`** over block
  sizes × `num_warps` × `num_stages` (re-check correctness after); (5) **benchmark vs the reference** with
  `triton.testing.do_bench` and report **GB/s** (memory-bound) or **TFLOP/s** (compute-bound) vs roofline.
- **`tl.dot` for matmul/tensor-cores; accumulate in fp32.** Even with fp16/bf16 inputs, `acc` is fp32;
  cast down at the end. fp16 accumulation loses precision over long K.
- **Numerical stability:** softmax subtracts `tl.max` before `tl.exp` (else `inf`/`NaN` on big logits);
  compute norm/softmax reductions in fp32.
- **Fusion wins on the memory wall.** The reason to write a kernel is usually to **fuse** a chain PyTorch
  runs as separate HBM round-trips — keep intermediates in registers/SMEM, write only the result.
  Epilogue fusion (bias+activation+cast into the matmul/reduction kernel) is the highest leverage.
- **Coalesce + tune.** Map the innermost offset to contiguous memory; sweep `num_warps`/`num_stages` via
  autotune (more isn't always better — large tiles × deep pipelines spill registers / exhaust SMEM).
- **Don't reinvent what's tuned.** A bare plain GEMM/elementwise that cuBLAS or `torch.compile`/Inductor
  already fuses well is not worth a hand kernel. Write one when you've shown bandwidth is left on the table.
- **Version-sensitive (verify current docs):** the block-pointer API (`tl.make_block_ptr`/`tl.advance`,
  `boundary_check`/`padding_option`), the exact `triton.Config`/`@triton.autotune` keyword set, `do_bench`
  signature, newer `tl.*` intrinsics, and non-NVIDIA backend warp/stage semantics. Never invent an API.

## Related skills

- `[[ml-compilers-codegen]]` — how your kernel **lowers**: TTIR→TTGIR (adds the layout/encoding deciding
  coalescing & MMA selection) →LLVM→PTX/SASS; read it to understand why tile shape/dtype are load-bearing.
- `[[gpu-performance-engineering]]` — how to **profile** what you wrote (roofline, Nsight Compute SOL,
  occupancy/coalescing/bank-conflict reads) and benchmark with statistical rigor. Pair with §5 here.
- `[[ml-frameworks]]` — PyTorch/JAX/XLA, the reference op you diff against, and how the kernel plugs into
  `torch.autograd`/`torch.library`/`torch.compile`.
- `[[inference-optimization]]` — where fused kernels (attention, norms, quantized matmul) pay off in
  decode/serving; the model-level memory-bound vs compute-bound reasoning.
- `[[ai-research-science]]` — when a novel op from a paper has no library kernel and you must author one.
