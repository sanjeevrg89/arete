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
