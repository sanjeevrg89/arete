# Triton Kernel Authoring — The Reference

How to **write** a correct, validated, autotuned, benchmarked Triton GPU kernel. This is the *doer's*
manual: a programming model, an authoring procedure with explicit correctness and performance gates, and
the canonical kernel patterns. It is the sibling of [[ml-compilers-codegen]] (how Triton *lowers*:
TTIR→TTGIR→PTX) and [[gpu-performance-engineering]] (how to *profile* what you wrote) — this one produces
the kernel.

**The bar:** never ship a kernel you have not diffed against a reference with `torch.allclose`, including
on non-power-of-2 and masked edge sizes; never report a speedup without warmup + device sync. Never
fabricate a Triton API: `tl.*` and decorator argument names are **version-sensitive** — when in doubt,
flag "verify against current Triton docs" rather than guess.

---

## 1. Mental model: tiles + SPMD, not threads

Triton is a Python-embedded DSL where you write **one program instance that computes one tile (block) of
the output**, and the runtime launches a **grid** of those instances (the SPMD / single-program-multiple-
data model — like CUDA blocks, but you never write thread-level code). The compiler owns everything below
the tile: thread/warp assignment, memory coalescing, shared-memory staging, software pipelining, register
allocation, and tensor-core (MMA) instruction selection.

What this buys you: you reason about **blocks of data and pointer arithmetic**, and the compiler reasons
about the hardware. What it costs you: the tile sizes, `num_warps`, and `num_stages` you choose are the
performance knobs — get them wrong and a correct kernel is slow. Hence autotuning is part of authoring,
not an afterthought.

Where Triton sits: **above** cuBLAS/cuDNN (you write the kernel) and **below** CUTLASS/raw CUDA (you don't
manage warps/MMA by hand). Reach for it when a fused or custom op doesn't exist in a library and you want
to beat the memory wall — not to reimplement a plain GEMM that cuBLAS already nails.

---

## 2. Core programming model (the primitives you actually use)

A kernel is a `@triton.jit`-decorated Python function. Inside it, values are either **scalars** or
**block tensors** (whose shapes are compile-time constants), and you operate on whole blocks.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):          # constexpr ⇒ specialized & shape-static
    pid = tl.program_id(axis=0)                    # which tile am I? (SPMD index)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # this tile's element indices
    mask = offs < n_elements                       # boundary guard — n_elements need not divide BLOCK
    x = tl.load(x_ptr + offs, mask=mask)           # masked load: out-of-range lanes don't touch memory
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)     # masked store: don't write past the end
```

Key objects and rules:

- **`@triton.jit`** — compiles the function to a kernel. It is traced symbolically; Python control flow
  on `constexpr`/scalar values is fine, but data-dependent Python `if` on a *block* is not — use `tl.where`.
- **`tl.program_id(axis=N)`** — this instance's coordinate in the launch grid (axes 0/1/2). Combined with
  block sizes, it tells the program *which* tile of the output it owns.
- **`BLOCK_SIZE: tl.constexpr`** — block/tile sizes are **compile-time constants**. Marking them
  `tl.constexpr` lets the compiler unroll, allocate registers, and pick vector widths. Tile sizes are
  normally powers of two (16/32/64/128/256...). The *problem* dimension need not be — that's what masks
  are for.
- **`tl.arange(0, BLOCK_SIZE)`** — a compile-time-shaped index vector `[0,1,...,BLOCK-1]`; the start/stop
  are constexpr. This is how you materialize a tile's local indices.
- **Pointer arithmetic** — you compute addresses yourself: `ptr + offs`. For multi-dim tensors you add
  contributions from each axis scaled by its **stride** (see §3). Triton is pointer-based, not
  shape-indexed (with the exception of the block-pointer API in §3).
- **Masking** is mandatory at boundaries. `mask=` on `tl.load` suppresses out-of-bounds reads (returns
  `other=`, default 0); `mask=` on `tl.store` suppresses out-of-bounds writes. **A kernel without masks
  is correct only when every dimension is an exact multiple of its block size** — almost never true in
  the wild. Masking bugs are the #1 source of silent corruption.
- **`tl.load(ptr, mask, other=0.0)` / `tl.store(ptr, val, mask)`** — the only memory ops. Loads/stores of
  contiguous, aligned `offs` are coalesced by the compiler; strided/gathered ones are not (see §6).
- **Reductions** — `tl.sum`, `tl.max`, `tl.min`, etc., reduce a block along an `axis`. Used for softmax,
  norms, dot-product reductions.
- **`tl.dot(a, b)`** — block-level matrix multiply that targets **tensor cores (MMA)** when dtypes/shapes
  allow. Accumulate in **fp32** (`acc += tl.dot(a, b)` with `acc` an fp32 block) even for fp16/bf16 inputs.
- **`tl.where(cond, a, b)`, `tl.maximum`, `tl.exp`, `tl.cast`/`.to(tl.float32)`** — elementwise/branch-free
  building blocks. Do conditional logic with `tl.where`, never a Python `if` on block data.

### Launch / grid

You launch a kernel with `kernel[grid](args..., BLOCK_SIZE=...)`. `grid` is a tuple or a callable of the
autotuned `meta` dict (so the grid can depend on the chosen block size):

```python
n = out.numel()
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)   # ceil-div → enough tiles to cover n
add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
```

`triton.cdiv(a, b)` is ceiling division — use it to size the grid so the last (partial) tile is covered;
the mask handles the overhang. Tensors are passed directly; Triton reads their data pointers. The kernel
launch is asynchronous on the CUDA stream — **synchronize before timing** (§5).

---

## 3. Multi-dimensional tensors: strides and 2-D tiles

For a 2-D tensor you compute row/col offsets and combine them with the tensor's **strides** (elements,
not bytes) so the kernel works for any (possibly non-contiguous / transposed) layout. Pass strides in as
kernel arguments — never assume row-major contiguity.

```python
@triton.jit
def row_kernel(x_ptr, out_ptr, M, N,
               stride_m, stride_n,                 # x.stride(0), x.stride(1)
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)                          # one program per row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    # address of x[row, cols]: base + row*stride_m + cols*stride_n
    x = tl.load(x_ptr + row * stride_m + cols * stride_n, mask=mask, other=0.0)
    ...
```

For a 2-D **output tile** (e.g. matmul), you build a 2-D offset grid via broadcasting:

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
# 2-D address block via outer broadcast: [BLOCK_M, BLOCK_N]
ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(ptrs, acc, mask=mask)
```

`[:, None]` / `[None, :]` are the broadcast idioms that turn two 1-D index vectors into a 2-D tile of
addresses — the workhorse of every tiled kernel.

> **Block-pointer API.** Recent Triton also offers `tl.make_block_ptr(...)` + `tl.advance(...)` and
> `boundary_check=`/`padding_option=` on load/store, which package the offset/stride/mask bookkeeping.
> Whether it (or manual pointer arithmetic) is preferred, and its exact signature, is
> **version-sensitive — verify against current Triton docs** before relying on it. Manual pointer
> arithmetic as shown above is the most portable and is what most reference kernels use.

---

## 4. The authoring procedure (numbered — follow it in order)

This is the loop. Each step has a gate; do not advance until it passes.

1. **Start from a correct PyTorch reference.** Write the op in plain PyTorch first
   (`def ref(x): return torch.softmax(x, dim=-1)`). This is your oracle for correctness *and* your
   baseline for speed. If you can't write the reference, you don't understand the op well enough to write
   the kernel.

2. **Write the kernel** for the *simplest* correct version: pick the parallelization (what does one
   program compute?), compute offsets with `tl.program_id` + `tl.arange`, **mask every boundary**, do the
   math, store with a mask. Don't autotune or micro-optimize yet. Get *a* correct kernel.

3. **Validate correctness against the reference — this is a hard gate.**
   ```python
   x = torch.randn(M, N, device='cuda', dtype=torch.float32)
   out_ref = ref(x)
   out_tri = triton_fn(x)
   torch.testing.assert_close(out_tri, out_ref, rtol=1e-3, atol=1e-3)  # or torch.allclose(...)
   ```
   Then **test edge sizes immediately**: non-power-of-2 (`N=1000`, `N=8191`), tiny (`N=1`), sizes that
   don't divide the block, and the dtype(s) you'll actually run (fp16/bf16 need looser tolerances). A
   kernel that passes at `N=1024` and fails at `N=1000` has a masking bug — find it now.

4. **Autotune** the kernel over its configuration space with `@triton.autotune` (block sizes,
   `num_warps`, `num_stages`, and any algorithmic constexpr like `GROUP_SIZE_M`). See §7. Re-run the
   correctness check **after** adding autotune — a bad config can change results if a reduction tile is
   wrong.

5. **Benchmark vs the reference** with proper methodology: warm up, `torch.cuda.synchronize()` (or
   `triton.testing.do_bench`, which handles warmup/sync and returns robust quantiles), and report a real
   metric. Compute **achieved bandwidth (GB/s)** for memory-bound kernels or **TFLOP/s** for compute-bound
   ones, and compare to the hardware roofline — see §5 and [[gpu-performance-engineering]]. If you're not
   faster than the reference (or than `torch.compile`), say so and find out why before claiming a win.

6. **Iterate on performance** using the patterns in §6, guided by the roofline classification — *not* by
   guessing. Re-validate correctness after every change.

**Do not skip 3 or 5.** A fast wrong kernel is worthless; a "fast" kernel you never measured is a guess.

---

## 5. Benchmarking & roofline (the perf gate)

Use `triton.testing.do_bench` (handles warmup, repeats, L2-cache flush, robust quantiles) rather than a
hand-rolled timer:

```python
ms = triton.testing.do_bench(lambda: triton_fn(x))          # median ms; supports quantiles=
ms_ref = triton.testing.do_bench(lambda: ref(x))
```

Then turn time into a hardware-relative number:

- **Memory-bound kernel** (elementwise, softmax, norms): report **GB/s** = `total_bytes / (ms * 1e-3) /
  1e9`, where `total_bytes` counts every byte read + written from HBM. Compare to peak HBM bandwidth.
  If you're near peak, you're done — there is nothing left to win.
- **Compute-bound kernel** (matmul, attention): report **TFLOP/s** = `total_flops / (ms * 1e-3) / 1e12`
  (a GEMM is `2*M*N*K` FLOPs). Compare to peak tensor-core throughput for the dtype.

Classifying memory- vs compute-bound (arithmetic intensity vs the ridge point) decides what to optimize;
the full roofline method, Nsight reads, and statistical rigor (percentiles, variance, clock pinning) live
in [[gpu-performance-engineering]] — defer to it. If you hand-roll a timer instead of `do_bench`: warm up
to skip JIT/autotune/cold clocks, **`torch.cuda.synchronize()` before stopping the clock**, and report a
median/percentile, never a single cold run.

---

## 6. Performance patterns (apply after correctness)

- **Memory coalescing.** Make the *innermost*, fastest-varying offset map to contiguous memory so adjacent
  program lanes touch adjacent addresses. For row-major data, iterate the last (contiguous) dim in
  `tl.arange`; access the contiguous axis with stride 1. Strided/transposed access kills bandwidth —
  re-tile so the contiguous dim is the inner one.
- **`num_warps` and `num_stages` = occupancy and pipelining.** `num_warps` sets threads per program
  (latency hiding / parallelism per tile); `num_stages` sets the depth of the **software pipeline** that
  overlaps global loads with compute (more stages = more overlap but more shared-memory/registers). These
  are not analytically derivable — **sweep them with `@triton.autotune`**. More is not always better:
  large tiles × high `num_stages` can exhaust shared memory or spill registers and tank occupancy.
- **Avoid shared-memory bank conflicts.** The compiler manages shared memory, but pathological tile
  shapes still conflict. If a kernel is slower than its roofline says it should be and Nsight shows bank
  conflicts, try different tile shapes / padding (the conflict-avoidance is mostly the compiler's job here
  — diagnose with Nsight per [[gpu-performance-engineering]] before hand-tuning).
- **Recompute vs reload.** Cheap arithmetic on data already in registers beats a round-trip to HBM to
  reload it. This is the core idea behind fused-attention's online softmax: recompute normalization
  incrementally rather than materialize the full `N×N` score matrix.
- **Fusion wins when you're memory-bound (the memory wall).** The whole point of writing a Triton kernel
  is usually to **fuse** a chain that PyTorch runs as separate kernels (each a full HBM round-trip).
  Compute intermediates in registers/SMEM and write only the final result. Epilogue fusion — folding
  bias + activation + cast into the same kernel as the matmul/reduction — is the highest-leverage version.
  Conversely, if a single op is already compute-bound and library-tuned (a big plain GEMM), fusing buys
  nothing — see [[ml-compilers-codegen]] on when the compiler/library already wins.
- **`tl.dot` accumulation in fp32.** Always accumulate matmuls into an fp32 block even with fp16/bf16
  inputs/outputs; cast the result down at the end. fp16 accumulation loses precision fast over a long K.
- **Numerical stability.** Subtract the row max before `exp` in softmax (`exp(x - max)`); without it,
  `exp` overflows to `inf` for large logits and the kernel returns NaNs while a small-input test passes.
  Compute reductions (sums for norms/softmax denominators) in fp32. RMSNorm/LayerNorm: compute mean/var in
  fp32, normalize, then cast.

For *how* these choices lower to the GPU (TTIR→TTGIR adds the layout/encoding that decides coalescing and
MMA selection; TTGIR→LLVM→PTX/SASS), see [[ml-compilers-codegen]]. You write the tile; the lowering turns
your tile decisions into the access pattern — which is why tile shape and dtype are load-bearing.

---

## 7. Autotuning

`@triton.autotune` benchmarks a list of `triton.Config`s and caches the best per **key** (the argument
values whose change should trigger a re-tune — typically the problem dimensions). The `key` matters: too
narrow and it re-tunes constantly; too broad and it reuses a config for shapes it wasn't tuned on.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        # ... sweep BLOCK sizes × num_warps × num_stages; powers of two for blocks
    ],
    key=['M', 'N', 'K'],          # re-tune when these change
)
@triton.jit
def matmul_kernel(...):
    ...
```

Notes: the constexpr names in each `Config` dict must be the kernel's `tl.constexpr` parameters.
Autotuning runs the kernel for real to time it, so the first call to a new shape pays the sweep cost —
**warm up before benchmarking** so you measure the chosen config, not the search. `triton.heuristics` can
derive a constexpr from arguments (e.g. set a flag when a dim is a multiple of the block). The exact
`Config`/`autotune` keyword set is **version-sensitive — verify against current Triton docs** (e.g.
pruning hooks and extra warp/stage-like knobs vary by release and backend).

---

## 8. Canonical kernel patterns

The recurring shapes. Full worked code with correctness + benchmark for three of these is in
`examples.md`.

- **Fused elementwise / activation.** One program per 1-D tile; masked load → arithmetic (e.g.
  `x * tl.sigmoid(x)` for SiLU, or fused `gelu(x*w + b)`) → masked store. The simplest pattern and the
  template for all masking. Wins by fusing a chain of pointwise ops into one HBM round-trip.
- **Fused softmax (row-wise).** One program per row; load the row (masked), subtract `tl.max` (stability),
  `tl.exp`, divide by `tl.sum`, store — all in registers, one read + one write instead of PyTorch's
  several. Assumes a row fits in one block; for very wide rows you tile/loop the row dimension.
- **LayerNorm / RMSNorm.** One program per row; compute mean & variance (LayerNorm) or mean-square
  (RMSNorm) in **fp32** via `tl.sum`, normalize, apply weight/bias, store. The backward pass needs a
  reduction across rows for the weight/bias gradients — often a second kernel.
- **Tiled matmul.** 2-D grid over output tiles `(BLOCK_M, BLOCK_N)`; loop over K in `BLOCK_K` chunks,
  `acc += tl.dot(a_tile, b_tile)` with `acc` in fp32; mask the K-edge and the M/N edges; store the tile.
  Add **L2-cache-friendly program ordering** (group program ids so co-scheduled tiles reuse the same rows
  of A / cols of B — the classic `GROUP_SIZE_M` super-grouping) and `@triton.autotune`. This is where
  Triton can match or beat cuBLAS on *fused* GEMMs (GEMM + epilogue) even if not on a bare GEMM.
- **Fused attention (the idea).** Tile Q×Kᵀ into blocks; for each Q-block, stream over K/V blocks keeping
  a running max and running sum (**online softmax**) and a running output accumulator — so you **never
  materialize the full `N×N` score matrix** (the memory wall) and never round-trip it through HBM. This is
  the FlashAttention pattern: recompute-in-registers beats reload. Writing a numerically-correct, masked
  (causal) fused-attention kernel is the graduation exercise; build it from a validated non-fused
  reference and check `allclose` against it before trusting it.

---

## 9. Anti-patterns & gotchas

- **No mask on a non-divisible size** → silent out-of-bounds reads/writes (garbage or corruption that a
  power-of-two test never catches). Always mask; always test a non-divisible size.
- **fp16/bf16 accumulation in `tl.dot`** → precision loss over long K. Accumulate in fp32.
- **softmax without max-subtraction** → `inf`/`NaN` on large logits; passes on toy inputs. Always subtract
  the max.
- **Trusting a single cold timing** → you measured JIT/autotune/cold clocks, not the kernel. Warm up,
  sync, take a median (`do_bench`).
- **Python `if` on block data** → won't trace / wrong result. Use `tl.where`.
- **Assuming row-major contiguity** → breaks on transposed/sliced tensors. Pass and use strides; or
  `.contiguous()` deliberately and document it.
- **Autotuning with a wrong `key`** → either constant re-tuning (key too fine) or a stale config used on
  an untuned shape (key too coarse). Key on the dims that change the optimal config.
- **Reinventing a plain GEMM/elementwise** that a library or `torch.compile`/Inductor already fuses well.
  Write a kernel when you've shown the library/compiler leaves bandwidth on the table — not reflexively.
- **Claiming a speedup vs the wrong baseline.** Compare against a *fair* reference (eager PyTorch *and*,
  where relevant, `torch.compile`) on the same shapes/dtype/device.

---

## 10. Version awareness

Triton (and the PyTorch/Inductor it ships under) moves fast — it is 2026. The **tile programming model is
stable**, but specific surface APIs are not. Treat as **verify-against-current-docs**: the block-pointer
API (`tl.make_block_ptr`/`tl.advance`, `boundary_check`/`padding_option`), the exact `triton.Config` /
`@triton.autotune` keyword set and pruning hooks, `do_bench` signature/return, any newer `tl.*` intrinsic,
and backend-specific knobs (AMD/Intel/other backends differ from NVIDIA on warps/stages semantics). Never
invent a `tl.*` name or a decorator flag — if you're unsure it exists in the target version, say so and
have the user check `import triton; triton.__version__` and the docs.

---

## 11. Rationalizations & rebuttals

- *"It passed at 1024, ship it."* → 1024 divides every block size you tried; run a non-power-of-2 and a
  tiny size before you trust the masks.
- *"`torch.allclose` is overkill for a quick kernel."* → it's the only thing standing between you and
  silent corruption that surfaces in someone's loss curve a week later.
- *"I'll autotune later."* → tile/warp/stage choice is *the* performance, not polish. An un-tuned kernel's
  benchmark number is meaningless.
- *"It's obviously faster than eager."* → then `do_bench` it and show the GB/s vs roofline; "obviously" is
  how regressions ship.
- *"Skip the fp32 accumulator, it's fine."* → fine on K=64, wrong on K=4096. Accumulate in fp32.
- *"The reference is slow, so any kernel is a win."* → compare against `torch.compile` too; Inductor may
  already fuse it and you've spent a day to lose.

---

## 12. Red flags (stop and reconsider)

- You're writing the kernel before you wrote (and ran) the PyTorch reference.
- There's a `tl.load`/`tl.store` with no `mask`.
- No correctness check, or only one at a power-of-two size.
- A reported speedup with no warmup, no `synchronize`, no roofline context.
- `tl.dot` accumulating in fp16/bf16; softmax without max-subtraction; reductions in low precision.
- Hardcoded contiguity assumptions; no strides passed in.
- Reaching for a hand-written kernel before checking whether a library / `torch.compile` already wins.

---

## 13. Verification gate (definition of done)

A Triton kernel is done only when **all** of these hold:

1. A **PyTorch reference** exists and is the correctness oracle.
2. `torch.testing.assert_close` / `torch.allclose` **passes** vs the reference, at the production dtype,
   on **multiple sizes including non-power-of-2, a size that doesn't divide the block, and a tiny size**.
3. **Every** `tl.load`/`tl.store` at a boundary is **masked**; matmul/reduction accumulators are **fp32**;
   softmax/norms subtract the max / reduce in fp32.
4. The kernel is **autotuned** (`@triton.autotune` over block sizes × `num_warps` × `num_stages`) with a
   sensible `key`, and correctness still passes after autotune.
5. A **benchmark vs the reference** (`do_bench`, warmed up, synced) reports **GB/s or TFLOP/s** and is
   compared to the roofline — and to `torch.compile` where relevant. The win (or the honest "no win") is
   stated with the metric, not asserted.
6. Any **version-sensitive API used is flagged** "verify against current Triton docs."

---

## 14. Canonical references (verify against current docs — it is 2026)

- Triton documentation & tutorials (vector-add, fused softmax, matmul, layernorm, fused attention):
  https://triton-lang.org/main/getting-started/tutorials/index.html
- Triton GitHub (source, `triton.language` API, `triton.testing`): https://github.com/triton-lang/triton
- `torch.compile` / TorchInductor (Triton as the GPU codegen backend) — PyTorch docs.
- FlashAttention (the fused, IO-aware attention pattern, online softmax):
  Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."
- Roofline model (memory- vs compute-bound, arithmetic intensity): Williams, Waterman, Patterson, 2009.
