# Examples — Triton Kernel Authoring

Three worked kernels, each through the **full loop**: a PyTorch reference, the kernel, a `torch.allclose`
correctness check on edge sizes, and a `triton.testing.do_bench` benchmark stub. Imitate the *shape* of
the loop, not the exact numbers.

These are **syntactically faithful** to the common Triton API. Surface APIs are **version-sensitive** —
flagged inline as "verify against current Triton docs." The tile programming model is stable; specific
`tl.*`/decorator details are not. Always run your own correctness check before trusting any kernel here.

```python
import torch
import triton
import triton.language as tl
```

---

## 1. Fused vector op (SiLU-ish) — mask + correctness check + benchmark

`out = x * sigmoid(x) + y` in one kernel: a fused pointwise chain that PyTorch would run as several HBM
round-trips. The template for masking.

### Reference

```python
def ref_silu_add(x, y):
    return x * torch.sigmoid(x) + y
```

### Kernel

```python
@triton.jit
def silu_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements                       # boundary guard: n need not divide BLOCK_SIZE
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    out = x * tl.sigmoid(x) + y                     # whole chain fused; intermediates stay in registers
    tl.store(out_ptr + offs, out, mask=mask)        # masked store: never write past the end

def silu_add(x, y, BLOCK_SIZE=1024):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)   # ceil-div → cover the last partial tile
    silu_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

### Correctness (test the edges — this is the gate)

```python
torch.manual_seed(0)
for n in [1, 7, 1000, 1024, 8191, 1 << 20]:        # incl. non-power-of-2 and non-dividing sizes
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    torch.testing.assert_close(silu_add(x, y), ref_silu_add(x, y), rtol=1e-4, atol=1e-4)
print("correctness OK")                            # n=1000/8191 exercise the mask; n=1 the tiny case
```

### Benchmark (memory-bound ⇒ report GB/s)

```python
n = 1 << 24
x = torch.randn(n, device='cuda'); y = torch.randn(n, device='cuda')
ms     = triton.testing.do_bench(lambda: silu_add(x, y))        # warmup + sync handled by do_bench
ms_ref = triton.testing.do_bench(lambda: ref_silu_add(x, y))
bytes_moved = 3 * n * x.element_size()             # read x, read y, write out
gbps = bytes_moved / (ms * 1e-3) / 1e9
print(f"triton {ms:.3f} ms ({gbps:.0f} GB/s) | ref {ms_ref:.3f} ms")
# Compare GB/s to peak HBM bandwidth; near-peak on a memory-bound op = done.
```

---

## 2. Tiled matmul — `@triton.autotune` + `tl.dot` (fp32 accumulation)

`C = A @ B`, A:[M,K], B:[K,N]. 2-D grid over output tiles; loop K in `BLOCK_K` chunks; accumulate in fp32.
Includes `GROUP_SIZE_M` super-grouping for L2 reuse and strides so it works on non-contiguous inputs.

### Reference

```python
def ref_matmul(a, b):
    return a @ b
```

### Kernel

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                      num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                      num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],            # re-tune when problem dims change
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    # --- L2-friendly program ordering: group tiles so co-scheduled programs reuse A rows / B cols ---
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- 2-D offset blocks for this output tile ---
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M     # %M keeps addresses in-bounds; mask on store
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)       # accumulate in fp32
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K            # mask the K-edge (last partial K tile)
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)                                    # tensor-core MMA when dtype/shape allow
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)                         # cast down at the end
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  # M/N boundary mask on store
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    M, K = a.shape; K2, N = b.shape
    assert K == K2, "inner dims must match"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

> **Verify against current Triton docs:** `triton.Config` / `@triton.autotune` keyword set (e.g.
> `num_warps`, `num_stages`, pruning hooks) and the block-pointer alternative (`tl.make_block_ptr` /
> `tl.advance` / `boundary_check`) vary by version. The `% M`/`% N` index-wrap trick keeps loads in-bounds
> while the store mask drops the overhang — confirm it matches the contiguity of your inputs.

### Correctness (include non-power-of-2 and the production dtype)

```python
for (M, N, K) in [(256, 256, 256), (512, 333, 129), (1000, 1000, 64), (1, 1, 1)]:
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    # fp16 inputs, fp32 accumulation: loosen tolerance accordingly
    torch.testing.assert_close(matmul(a, b), ref_matmul(a, b), rtol=1e-2, atol=1e-2)
print("matmul correctness OK")
```

### Benchmark (compute-bound ⇒ report TFLOP/s)

```python
M = N = K = 4096
a = torch.randn((M, K), device='cuda', dtype=torch.float16)
b = torch.randn((K, N), device='cuda', dtype=torch.float16)
ms     = triton.testing.do_bench(lambda: matmul(a, b))
ms_ref = triton.testing.do_bench(lambda: ref_matmul(a, b))   # cuBLAS via torch — a fair, strong baseline
tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
print(f"triton {ms:.3f} ms ({tflops:.0f} TFLOP/s) | cuBLAS {ms_ref:.3f} ms")
# Compare TFLOP/s to peak fp16 tensor-core throughput. A bare GEMM rarely beats cuBLAS —
# Triton's win is a FUSED GEMM (epilogue: + bias, activation, cast) that avoids the extra HBM passes.
```

---

## 3. Fused softmax (row-wise) — the max-subtraction trick

Row-wise softmax in one read + one write, with numerical stability. One program per row; assumes a row
fits in one block (the common case for attention logits / classifier heads).

### Reference

```python
def ref_softmax(x):
    return torch.softmax(x, dim=-1)
```

### Kernel

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):       # BLOCK_SIZE = next_power_of_2(n_cols)
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols                            # mask the padded lanes of the block
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float('inf'))  # -inf ⇒ exp→0
    x = x.to(tl.float32)                            # reduce in fp32 for stability
    x = x - tl.max(x, axis=0)                       # subtract row max BEFORE exp — prevents inf/NaN
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)                 # denominator excludes padded lanes (they're 0)
    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)     # one block must cover a row
    softmax_kernel[(n_rows,)](x, out, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

> **Verify against current Triton docs:** `triton.next_power_of_2` and whether very wide rows need a
> looped/online-softmax variant (a single block has a max size). For rows wider than one block, tile the
> row and carry a running max + running sum (the online-softmax / FlashAttention idea — §8 of the guide).

### Correctness (non-power-of-2 widths exercise the mask)

```python
for (R, C) in [(64, 1), (128, 1000), (32, 8191), (1, 5)]:
    x = torch.randn((R, C), device='cuda')
    torch.testing.assert_close(softmax(x), ref_softmax(x), rtol=1e-4, atol=1e-4)
# stability check: large logits must NOT produce NaN
x = (torch.randn((4, 1024), device='cuda') * 100.0)
assert not torch.isnan(softmax(x)).any(), "max-subtraction missing → NaN"
print("softmax correctness + stability OK")
```

### Benchmark (memory-bound ⇒ report GB/s)

```python
x = torch.randn((4096, 4096), device='cuda')
ms     = triton.testing.do_bench(lambda: softmax(x))
ms_ref = triton.testing.do_bench(lambda: ref_softmax(x))
bytes_moved = 2 * x.numel() * x.element_size()      # read x, write out
gbps = bytes_moved / (ms * 1e-3) / 1e9
print(f"triton {ms:.3f} ms ({gbps:.0f} GB/s) | torch {ms_ref:.3f} ms")
# torch.softmax may run 3+ kernels (max, exp/sum, div); the fused kernel's win is fewer HBM passes.
```

---

## Takeaways

- Every example: **reference → kernel → `assert_close` on edge sizes (incl. non-power-of-2 + tiny) →
  `do_bench` reporting GB/s or TFLOP/s vs a fair baseline.** That loop *is* the skill.
- **Mask every boundary**, **accumulate `tl.dot` in fp32**, **subtract the max in softmax** — the three
  bugs that pass a power-of-two toy test and corrupt real runs.
- Triton's edge is **fusion** (fewer HBM round-trips), not out-GEMM-ing cuBLAS. Pick kernels where the
  memory wall, not the math, is the bottleneck — see [[gpu-performance-engineering]] for the roofline and
  [[ml-compilers-codegen]] for how these tiles lower to PTX.
- Anything API-specific above is **version-sensitive — verify against current Triton docs** before relying
  on it; check `triton.__version__`.
