# ML Frameworks — Canonical Snippets

Small, correct, minimal patterns to imitate. Imports are shown; error handling and boilerplate are
trimmed. **APIs move fast (2026)** — if a name looks off for your version, verify against current docs
(see `ml-frameworks-guide.md` §7). Concepts here are stable; exact spellings may drift.

---

## 1. JAX — sharded `jit` matmul with a `Mesh` and `PartitionSpec`

A 2-D device mesh (`data` × `model`), sharded inputs, and `jit` letting GSPMD insert the collectives.
This is the *idiomatic* way to parallelize in modern JAX — annotate layout, let XLA write the comms.

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Build an N-d mesh over all available devices (e.g. 8 -> 2x4).
devices = jax.devices()                      # GPUs or TPU chips
mesh = jax.make_mesh((2, 4), ("data", "model"))   # convenience helper; or Mesh(np.array(...), axes)

# Shardings: A is sharded on rows over `data`; B is sharded on cols over `model`.
a_sharding = NamedSharding(mesh, P("data", None))     # (M, K): M over data, K replicated
b_sharding = NamedSharding(mesh, P(None, "model"))    # (K, N): N over model, K replicated
out_sharding = NamedSharding(mesh, P("data", "model"))  # (M, N): tiled over both axes

A = jax.device_put(jnp.ones((1024, 512)), a_sharding)
B = jax.device_put(jnp.ones((512, 2048)), b_sharding)

@jax.jit  # GSPMD partitions this; the all-gather/reduce-scatter is inferred from the shardings.
def matmul(a, b):
    return a @ b

C = matmul(A, B)                # distributed automatically; same code single- or multi-host
C = jax.block_until_ready(C)    # dispatch is async — force completion before timing
print(C.shape, C.sharding)      # (1024, 2048), sharded over ("data", "model")
```

Notes:
- `P(axis, None)` = shard this dim over `axis`, replicate that dim. `None` everywhere = fully replicated.
- You can also pass `in_shardings=`/`out_shardings=` to `jax.jit` instead of pre-`device_put`-ing.
- For **explicit** per-shard code with manual collectives, use `shard_map` (see §3).

---

## 2. JAX — `value_and_grad` + `jit` + buffer donation (one training step)

```python
import jax, optax
import jax.numpy as jnp

def loss_fn(params, batch):
    logits = model_apply(params, batch["x"])           # your forward
    return optax.softmax_cross_entropy(logits, batch["y"]).mean()

@jax.jit                                    # donate params+opt_state: their buffers are reused for the
def train_step(params, opt_state, batch):  # updated outputs, cutting peak memory.
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Mark which positional args may be donated (params=0, opt_state=1). Don't reuse the old refs after.
train_step = jax.jit(train_step, donate_argnums=(0, 1))
```

Recompilation guard: keep `batch` shapes fixed across steps (pad/bucket variable-length sequences) or
every new shape recompiles.

---

## 3. JAX — `shard_map` with an explicit collective

When GSPMD's automatic comms isn't what you want, run per-shard and call collectives yourself.

```python
from jax.experimental.shard_map import shard_map   # verify module path for your version
from jax.sharding import PartitionSpec as P
import jax, jax.numpy as jnp

# Each device gets its shard of x along `data`; we sum-reduce across that axis manually.
@jax.jit
def f(x):
    def body(x_shard):
        local = x_shard @ x_shard.T            # runs on each shard independently
        return jax.lax.psum(local, axis_name="data")  # explicit all-reduce over the mesh axis
    return shard_map(body, mesh=mesh,
                     in_specs=P("data", None),
                     out_specs=P(None, None))(x)
```

---

## 4. PyTorch — `torch.compile` + AMP (bf16) training step

The default modern training step: bf16 autocast (no scaler needed), `torch.compile` to fuse the
memory-bound tail, `set_to_none=True` to drop grad buffers.

```python
import torch

torch.set_float32_matmul_precision("high")   # enable tf32 for fp32 matmuls on Ampere+

model = MyModel().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
model = torch.compile(model)                 # Dynamo -> AOTAutograd -> Inductor (Triton on GPU)

for x, y in loader:                          # loader: num_workers>0, pin_memory=True, prefetch
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # bf16: no GradScaler
        loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
```

### 4a. fp16 variant — requires a `GradScaler`

fp16 has a narrow exponent and underflows without loss scaling. **bf16 above needs none; fp16 does.**

```python
scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()    # scale up to avoid grad underflow
scaler.step(opt)                 # unscales, skips step on inf/nan
scaler.update()                  # adapts the scale factor
```

Gotchas: don't time the loop without `torch.cuda.synchronize()` (dispatch is async); watch
`TORCH_LOGS=recompiles,graph_breaks` if shapes vary — use `torch.compile(model, dynamic=True)` for
variable batch/seq lengths.

---

## 5. PyTorch — memory snapshot for OOM / fragmentation

```python
import torch
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run the step(s) that OOM ...
torch.cuda.memory._dump_snapshot("snap.pickle")   # open in the PyTorch memory-viz tool
print(torch.cuda.memory_summary())
# reserved >> allocated and an OOM with free bytes available  ==>  fragmentation.
# Mitigate: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (env var, set before CUDA init)
```

---

## 6. Triton kernel — shape/tiling mental model (not a full kernel)

You reason in **tiles (BLOCK sizes)**, not individual threads; Triton schedules the block and
autotunes. The shape contract: pick block sizes, compute per-program offsets, mask the ragged edge.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                  # this program handles one BLOCK-sized tile
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                         # guard the last partial tile
    x = tl.load(x_ptr + offs, mask=mask)    # coalesced load into on-chip SRAM
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# launch: grid = (triton.cdiv(n, BLOCK),) ;  BLOCK a power of two (e.g. 1024), autotune in practice.
```

Mental model: `BLOCK` should keep the working set in registers/shared memory; the win is a **single
fused kernel** (one HBM read of x and y, one write of out) instead of separate launches. Matmul
kernels extend this to 2-D `BLOCK_M × BLOCK_N × BLOCK_K` tiles accumulating in registers, mapping tiles
onto tensor cores — but use cuBLAS/`torch.matmul` unless you have a real reason to hand-write.

---

## 7. Pallas — TPU/GPU kernel shape (block-over-memory model)

Pallas is JAX's kernel language (TPU via Mosaic, GPU via a Triton-style path). Like Triton, you write
in **blocks over the memory hierarchy**; `BlockSpec` describes how the global array is tiled into the
blocks each program instance sees.

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl   # verify import path for your version

def add_kernel(x_ref, y_ref, o_ref):        # refs are this block's view of the arrays
    o_ref[...] = x_ref[...] + y_ref[...]

def add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        # grid + BlockSpecs (omitted) tile the arrays; on TPU, prefer MXU-friendly tile dims
        # (commonly multiples of 128 — verify for the TPU generation).
    )(x, y)
```

Reach for Pallas only when profiling shows XLA's automatic fusion leaves real performance on the table
(e.g. a custom attention/block-sparse op). For ordinary work, let XLA fuse.

---

## 8. PyTorch FSDP2 — framework-level wrapping (strategy lives in [[training-frameworks]])

```python
from torch.distributed.fsdp import fully_shard   # FSDP2 API; verify for your version

# Shard each transformer block, then the whole model. Params/grads/optimizer state are sharded;
# full params are all-gathered just-in-time per layer and freed after use.
for block in model.transformer_blocks:
    fully_shard(block)
fully_shard(model)
```

This is the *primitive*. Wrapping policy, mixed precision config, TP/PP composition, and multi-node
launch are [[training-frameworks]].
