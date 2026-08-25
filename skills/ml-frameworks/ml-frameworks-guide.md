# ML Frameworks Guide — PyTorch · JAX · XLA · GPU · TPU

The authoritative reference for this skill. Written for a performance engineer optimizing frontier
models. Scope is the **compute frameworks and the accelerator substrate beneath them**; multi-node
*orchestration* of parallelism is [[training-frameworks]] and inference runtimes are
[[serving-frameworks]].

**Version reality (2026):** PyTorch (FSDP2, `torch.compile` maturity, fp8 paths), JAX/OpenXLA, CUDA
(Hopper/Blackwell, fp8/fp4), and TPU generations all move fast. Treat specific flag names, dtype
support, and shape rules below as *concepts to verify*, not gospel — confirm against the current
release notes for the exact version you target. Where a number would be a benchmark, this guide gives
a mental model instead.

---

## 0. The cross-cutting mental models (read this first)

Everything else is detail. These four models explain *why* an optimization works.

### 0.1 The roofline: compute-bound vs memory-bound

Every kernel is limited by either **how fast the chip can do FLOPs** or **how fast it can move bytes
to/from HBM**. The crossover is **arithmetic intensity** = FLOPs performed ÷ bytes moved (FLOP/byte).
Compare it to the hardware's peak-FLOP ÷ peak-bandwidth ratio:

- **Intensity below the ridge → memory-bound.** You are waiting on HBM. The tensor cores / MXU sit
  idle. Examples: elementwise ops, activations, LayerNorm/RMSNorm, softmax, bias-add, dropout, small
  GEMMs, embedding lookups, attention's softmax/glue. **Fix: fuse** so intermediates never hit HBM,
  and raise reuse.
- **Intensity above the ridge → compute-bound.** You are saturating the matmul units. Examples: large
  GEMMs, convolutions, the QK^T / PV matmuls in attention at large sizes. **Fix: better tiling, right
  dtype (tensor cores), better shapes** — there is no free lunch from fusion here.

A transformer training step is a mix: the big linear/attention matmuls are compute-bound; the long
tail of norms, activations, and elementwise ops is memory-bound. Most "compile" speedups come from
**fusing the memory-bound tail**, not from making the GEMMs faster.

### 0.2 MFU — the efficiency north star

**Model FLOP Utilization** = (useful model FLOPs/sec actually achieved) ÷ (hardware peak FLOPs/sec).
It folds together kernel efficiency, memory-boundedness, comms overhead, and pipeline bubbles into one
number. Low MFU points you at a bottleneck: kernel-bound (improve kernels/dtype), host-bound (fix the
input pipeline), or comms-bound (parallelism/topology). Do not chase MFU blindly — recompute
(activation checkpointing) raises FLOPs done per step but can *raise* MFU while *lowering* throughput.
Measure wall-clock tokens/sec too.

### 0.3 Numerical precision is a deliberate trade

| dtype | bits (E/M) | use | watch out for |
|---|---|---|---|
| fp32 | 8/23 | master weights, reductions, norms, softmax, loss | slow on tensor cores; rarely needed for the whole net |
| tf32 | 8/10 (matmul) | Ampere+ GEMMs, near-fp32 quality | it's a matmul mode, not a storage dtype; enable via matmul-precision setting |
| bf16 | 8/7 | **default training compute** | low mantissa precision — keep reductions/master weights fp32 |
| fp16 | 5/10 | training/inference | narrow exponent → overflow/underflow; needs **loss scaling** (`GradScaler`) |
| fp8 E4M3 | 4/3 | forward GEMMs (Hopper/Blackwell) | needs per-tensor/blockwise **scaling + amax tracking**; verify lib support |
| fp8 E5M2 | 5/2 | gradients (wider range) | even less mantissa; pair with E4M3 carefully |

Rules of thumb: **bf16 is the workhorse** — wide exponent means no loss scaling and stable training.
Use fp16 only when you must, always with a scaler. **fp8 is for the biggest GEMMs**, with the
surrounding accumulation, scaling, and sensitive ops (softmax, norms, the loss, optimizer state) kept
higher. Keep a fp32 master copy of weights in mixed-precision training.

### 0.4 Compiler × parallelism × hardware interact

A compiler can fuse and pick layouts; parallelism splits work across chips; hardware sets the
ceilings. They are not independent: an XLA layout choice changes whether a collective is cheap; a
sharding choice changes which fusions are legal; a dtype choice changes which kernel (tensor-core vs
not) is selected. Optimize them together, and always re-profile — a change that helps the kernel can
hurt comms.

---

## 1. PyTorch

### 1.1 Eager vs graph; autograd

PyTorch is **define-by-run**: each op executes immediately and records itself on a dynamic tape.
Autograd builds a backward graph of `grad_fn` nodes as the forward runs; `loss.backward()` traverses
it, accumulating into `.grad`. Eager is maximally flexible (plain Python control flow, debuggable) but
each op is a separate dispatch + kernel launch, so small ops are launch-overhead- and memory-bound.

Key autograd facts: leaf tensors with `requires_grad=True` accumulate grads (zero them each step or
use `set_to_none=True`); `torch.no_grad()` / `inference_mode()` disable tape recording for
eval/inference; `detach()` cuts the graph; in-place ops can corrupt saved tensors (autograd errors if
a needed value was overwritten). Gradient accumulation = run several micro-batches before `optimizer.step()`.

### 1.2 `torch.compile` — Dynamo → AOTAutograd → Inductor

`torch.compile(model)` is the JIT path. Pipeline:

1. **TorchDynamo** hooks CPython frame evaluation, traces Python into an FX graph, and falls back to
   eager at any construct it can't capture — a **graph break**. Many breaks ⇒ little benefit. Inspect
   with `torch._dynamo.explain` / logs (`TORCH_LOGS="graph_breaks,recompiles"`).
2. **AOTAutograd** traces a **joint forward+backward** graph ahead of time and partitions it, so the
   backward is compiled too (and activations can be recomputed/saved deliberately).
3. **Inductor** is the default backend: it lowers to optimized kernels, generating **Triton** on GPU
   (and C++/OpenMP on CPU), doing fusion, and calling cuBLAS/cuDNN/CUTLASS templates for big matmuls.

When it **helps**: lots of small/memory-bound ops to fuse; Python/launch overhead is significant;
static-ish shapes. `mode="max-autotune"` autotunes matmul/conv templates (long compile, faster run).
When it **hurts / no-ops**: shapes change every step (recompiles), heavy graph breaks, very short runs
where compile time dominates, or code already dominated by one big cuBLAS GEMM. Use `dynamic=True` or
mark dynamic dims (`torch._dynamo.mark_dynamic`) to compile once for variable sizes; `fullgraph=True`
forces a single graph (fails loudly on breaks) for debugging. `torch.compile` composes with AMP and
FSDP but verify the combination on your version.

### 1.3 CUDA streams, the caching allocator, and memory

- **Streams:** CUDA ops are async; the default stream serializes them. PyTorch enqueues kernels and
  returns; `.item()`/`.cpu()`/`torch.cuda.synchronize()` force a host sync. Naive timing without a
  sync measures launch time, not execution. Custom streams + events overlap copy and compute (the
  `DataLoader`'s pinned-memory H2D copy is the classic overlap).
- **Caching allocator:** PyTorch calls `cudaMalloc` rarely and **caches freed blocks** to avoid the
  cost and serialization of allocation. So **reserved memory ≥ allocated memory**, and freeing a
  tensor returns memory to the *cache*, not the driver. `torch.cuda.empty_cache()` returns cached
  blocks to the driver (rarely the right fix; it can fragment differently).
- **OOM is usually fragmentation,** not a hard wall: you have enough total free bytes but no single
  contiguous block big enough. Diagnose with `torch.cuda.memory_summary()` and the snapshot:
  `torch.cuda.memory._record_memory_history()` then `_dump_snapshot("snap.pickle")`, view in the
  memory-viz tool. Mitigate with **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (lets segments
  grow, reducing fragmentation from variable shapes) — verify it's appropriate for your version.
- **What actually fills memory:** params + grads + optimizer state (Adam ≈ 2 extra fp32 states per
  param) + **activations** (often the largest, scales with batch×seq×layers). Reduce via **activation
  checkpointing** (recompute in backward), smaller micro-batch + grad accumulation, FSDP sharding, or
  lower-precision optimizer states.

### 1.4 Mixed precision (AMP)

Use `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` to run the forward in low precision
while autograd keeps a fp32 master path for sensitive ops. **bf16: no scaler needed.** **fp16: wrap
the step with `torch.amp.GradScaler`** to avoid gradient underflow (`scaler.scale(loss).backward();
scaler.step(opt); scaler.update()`). Autocast picks per-op precision (matmuls low, reductions/softmax
fp32). Also set the matmul precision knob (`torch.set_float32_matmul_precision("high")` enables tf32
for fp32 matmuls on Ampere+). fp8 training goes through dedicated paths/libraries (e.g.
`torchao`/`transformer_engine`) — verify what your stack supports.

### 1.5 Distributed primitives

- **`torch.distributed`** is the collective layer; the **NCCL** backend is the GPU path (NVLink/network
  collectives), Gloo for CPU. One process per GPU; `init_process_group`, `all_reduce`, `all_gather`,
  `reduce_scatter`, `broadcast`, `barrier`. Launch with `torchrun` (sets `RANK`/`WORLD_SIZE`/`LOCAL_RANK`).
- **DDP** wraps the model, replicates it on every rank, and **all-reduces gradients** in the backward
  (bucketed and overlapped with compute). Simple and fast **when the full model+optimizer fits per
  device.** Use `no_sync()` to skip the all-reduce on gradient-accumulation micro-steps.
- **FSDP / FSDP2** shards **parameters, gradients, and optimizer state** across ranks; each layer's
  full params are **all-gathered just before use** and freed after, and grads are **reduce-scattered**.
  Trades extra comms for fitting far larger models. FSDP2 (the `fully_shard` API) is the current
  direction with cleaner per-parameter sharding and composition with `torch.compile`/TP — verify the
  exact API and status for your version. **Detailed parallelism strategy (sharding/wrapping policy, TP,
  PP, hybrid) is [[training-frameworks]].**

### 1.6 Data loading

The accelerator can starve waiting on the host. `DataLoader` knobs: enough `num_workers` (multiprocess
decode/augment), `pin_memory=True` (page-locked host buffers → faster async H2D), `prefetch_factor`,
`persistent_workers=True` (avoid per-epoch worker respawn). Watch for: slow Python decode/augmentation,
GIL contention, tiny batches, synchronous H2D copies, dataset `__getitem__` doing heavy I/O. Profile
the *step boundary* — if the GPU idles between steps, you are host-bound. For very large/streaming data
consider `webdataset`/`tensordict`/NVIDIA DALI (GPU-side decode).

### 1.7 Profiling

- **`torch.profiler`** with `record_shapes`, `profile_memory`, `with_stack`; export a Chrome/Perfetto
  trace and a TensorBoard view. Tells you op time, CUDA vs CPU time, and whether you're launch-bound.
- **Nsight Systems** (`nsys`) for the timeline: kernel/stream overlap, gaps (host-bound), NCCL behavior.
  **Nsight Compute** (`ncu`) for a single kernel's occupancy, memory throughput, and roofline position.
- First question to answer: **kernel-bound, host-bound, or comms-bound?** Optimize accordingly.

---

## 2. JAX

### 2.1 The functional / pure model

JAX programs are **pure functions of immutable arrays**. There is no in-place mutation
(`x = x.at[i].set(v)` returns a new array); randomness is explicit via splittable keys
(`jax.random.split`); side effects inside transformed functions are undefined. You write functions and
**apply transformations** to them: `jit` (compile), `grad` (differentiate), `vmap` (auto-batch),
`pmap`/sharding (parallelize). Transformations compose: `jit(grad(vmap(f)))`.

### 2.2 Tracing and abstract values

`jit` (and `grad`/`vmap`) work by **tracing**: JAX runs your Python once with **abstract tracers**
(carrying shape+dtype, not values) to record a graph (a jaxpr), which XLA compiles. Consequences that
bite everyone:

- **Python control flow that branches on a traced value fails** — the tracer has no concrete value.
  Use `jax.lax.cond`/`select`/`while_loop`/`scan`/`fori_loop` for value-dependent control flow, or mark
  the arg `static_argnums` (which **specializes** — a new value triggers a recompile).
- **Shapes must be static at compile time.** A function is compiled **per input shape/dtype signature**;
  a new signature = a new compilation. This is the #1 latency surprise (see 2.6 / §3).
- Printing/asserting on traced values needs `jax.debug.print` / `jax.debug.callback`.

### 2.3 `grad`, `vmap`

- **`grad(f)`** returns a function computing ∂f/∂x via reverse-mode autodiff (f must return a scalar);
  `value_and_grad` gives both; `has_aux=True` for extra outputs; `jax.jacfwd`/`jacrev` for Jacobians.
- **`vmap`** adds a batch axis without writing batched code: `vmap(f, in_axes=..., out_axes=...)`. It
  vectorizes (no Python loop), so it's efficient and composes with `grad`/`jit`. The mental shift:
  write the per-example function, let `vmap` batch it.

### 2.4 `jax.Array`, unified sharding, and SPMD

Modern JAX has **one array type, `jax.Array`, that carries a sharding** describing how it's laid out
across devices. The pieces:

- **`Mesh`** — a named N-d grid of devices, e.g. `Mesh(devices, axis_names=("data", "model"))`.
- **`PartitionSpec`** (`P`) — for each array dimension, which mesh axis (if any) it is sharded over;
  `None` means replicated. `P("data", "model")` shards dim 0 over `data` and dim 1 over `model`.
- **`NamedSharding(mesh, P(...))`** — binds a spec to a mesh; pass to `jax.device_put` or as `jit`'s
  `in_shardings`/`out_shardings`.

Under `jit`, **GSPMD** (the SPMD partitioner in XLA) takes the shardings on inputs/outputs and **infers
a fully distributed program**, inserting the needed collectives (all-gather/reduce-scatter/all-to-all)
automatically. This is the **pjit-style automatic parallelism** model — you annotate data layout, the
compiler writes the comms. The same code runs single-host or multi-host.

- **`shard_map`** — the explicit alternative: your function runs **per shard** with manual collectives
  (`jax.lax.psum`, `all_gather`, `ppermute`). Use it when you need precise control GSPMD won't give you
  (custom comms patterns, manual overlap). It's the lower-level escape hatch under the automatic model.
- **`pmap`** is the older single-program-per-device API — for new code prefer `jit` + shardings or
  `shard_map`.

### 2.5 Donation, compilation caching, and other knobs

- **Buffer donation** (`jax.jit(..., donate_argnums=...)`): lets XLA reuse an input's memory for an
  output (in-place at the buffer level), cutting peak memory for things like optimizer updates where
  the old param buffer is dead. The donated input must not be used afterward.
- **Persistent compilation cache:** XLA compilation is expensive; enable the on-disk cache
  (`jax.config.update("jax_compilation_cache_dir", ...)` / the corresponding env var — verify the
  current name) so re-runs and multi-host workers skip recompiling identical programs.
- `jax.block_until_ready(x)` forces completion (JAX dispatch is async, like CUDA). `jax.devices()` /
  `jax.local_devices()` enumerate the topology.

### 2.6 Recompilation traps

A new **shape, dtype, or `static_argnums` value** recompiles. Symptoms: periodic latency spikes, slow
"warmup," memory churn. Fixes: **pad/bucket** variable-length inputs to a fixed set of shapes; avoid
passing Python ints/flags that change as `static`; use `jax.lax` loops instead of unrolled Python
loops over data-dependent counts. For genuinely dynamic shapes, shape polymorphism / export exists
(esp. for AOT/serving) but adds complexity — verify the current API.

### 2.7 Why JAX shines on TPU

JAX was co-designed with XLA and the TPU. The functional/SPMD model maps cleanly onto the systolic
MXU and the pod's ICI mesh; `Mesh`/`PartitionSpec` expresses pod-scale sharding directly; XLA does the
heavy fusion/layout work the TPU needs. JAX runs well on GPU too (also via XLA), but TPU is where the
design pays off most.

---

## 3. XLA / OpenXLA

XLA is the **graph compiler** under both JAX and PyTorch/XLA. It does not interpret ops one at a time;
it compiles a whole computation into one (or few) optimized executables.

### 3.1 The lowering pipeline

1. **Frontend → HLO** (High-Level Operations): the framework emits an HLO graph (in OpenXLA, the
   portable form is **StableHLO**, the versioned input dialect).
2. **HLO optimization passes:** algebraic simplification, common-subexpression elimination, **operator
   fusion** (the big win — merges elementwise/reduction chains so intermediates stay in registers/SRAM,
   killing memory-bound HBM traffic), **layout assignment** (choosing physical memory layouts/tiling
   so consumers don't pay for transposes), constant folding, and the **SPMD partitioning** pass (GSPMD)
   that turns sharding annotations into a distributed program with collectives.
3. **Backend codegen:** lower to target — LLVM/PTX for GPU, the TPU backend for TPU — producing the
   executable.

### 3.2 Why shape specialization & compilation matter

XLA compiles for **specific static shapes**, which is what enables aggressive fusion, tiling, and
layout decisions. The cost: **a new shape recompiles.** This is the same trap as JAX §2.6 and
PyTorch/XLA — keep shapes static, bucket/pad, and **cache compilations**. Compilation can be slow
(seconds to minutes for big graphs); for latency-sensitive deploys consider **AOT** compilation
(`jax.export` / ahead-of-time lowering) instead of JIT.

### 3.3 PJRT — the device runtime

**PJRT** is OpenXLA's **device-plugin runtime interface**: a stable API that lets a backend (GPU, TPU,
and third-party accelerators) plug into the framework. JAX and PyTorch/XLA both dispatch through PJRT,
which is why new hardware can be supported without changing the frameworks. When you see a JAX/XLA
device, it's exposed via a PJRT plugin.

### 3.4 `xla_flags` and tuning

XLA behavior is tuned via the `XLA_FLAGS` env var (and `LIBTPU`/backend-specific flags). They control
things like fusion aggressiveness, autotuning of GEMMs, latency-hiding scheduling that overlaps
collectives with compute, and dumping HLO. **Do not memorize or fabricate flag names** — they change
between versions and are sparsely documented; look them up for your exact XLA build and validate the
effect with a profile. The universally useful one is dumping the optimized HLO
(`XLA_FLAGS=--xla_dump_to=...`) to *read what the compiler actually produced* — confirm the current
spelling.

### 3.5 PyTorch/XLA

PyTorch can target XLA via **PyTorch/XLA** (the historical TPU path for PyTorch, also usable on GPU).
It **lazily records** PyTorch ops into an XLA graph and compiles at a **mark step** (`xm.mark_step()` /
`torch_xla.sync()` — verify the current API). Because it's trace-and-compile, the **same shape-stability
and recompilation rules** apply, and graphs that change every step destroy performance. SPMD on
PyTorch/XLA uses an analogous sharding-annotation model. The newer **`torch_xla` + `torch.compile`**
integration and OpenXLA path are evolving — verify current status.

---

## 4. GPU substrate (CUDA)

### 4.1 Programming model

A CUDA kernel runs a grid of **threads** grouped into **warps** (32 threads executing in lockstep,
SIMT) grouped into **thread blocks**, scheduled onto **Streaming Multiprocessors (SMs)**. A block runs
entirely on one SM and its threads share that SM's **shared memory** and can synchronize
(`__syncthreads`). **Warp divergence** (threads in a warp taking different branches) serializes the
branches and wastes lanes.

### 4.2 Memory hierarchy

`registers` (per-thread, fastest) → `shared memory / L1` (per-block, software-managed scratchpad) →
`L2` (chip-wide) → `HBM` (global device memory, large but ~the slow tier). The whole game of a fast
kernel is **keeping data in registers/shared memory and minimizing HBM traffic**, plus **coalesced
access** (consecutive threads read consecutive addresses) so a memory transaction isn't wasted.

### 4.3 Occupancy, tensor cores, fusion

- **Occupancy** = active warps per SM ÷ max, bounded by registers/shared-mem/threads per block. Enough
  occupancy hides memory latency by switching warps; *more is not always better* (register spills).
- **Tensor cores** do a small matrix-multiply-accumulate per instruction and are the only way to reach
  peak FLOPs. To use them: right **dtype** (bf16/fp16/fp8/tf32), shapes that are **multiples of the
  tensor-core tile** (commonly 8/16; fp8 wants larger alignment — verify), contiguous/aligned data.
  cuBLAS/cuDNN pick tensor-core kernels automatically when conditions are met.
- **Kernel fusion** merges memory-bound ops into one launch so intermediates stay on-chip — exactly
  what Inductor/Triton and XLA do for you.

### 4.4 Libraries and custom kernels

- **cuBLAS** (GEMM), **cuDNN** (conv/attention/RNN primitives) — the tuned standard kernels; use them
  for the heavy compute-bound ops.
- **CUTLASS** — C++ templates for building custom high-performance GEMM/conv kernels with full control
  of tiling and tensor-core use.
- **Triton** — a Python-embedded language for writing fused GPU kernels (you reason in *blocks/tiles*,
  it handles intra-block scheduling and autotuning). Inductor generates Triton; you can also hand-write
  it for custom fused ops. Far more approachable than CUDA C++ for memory-bound fusions.
- **FlashAttention** — the canonical **IO-aware** attention kernel: tiles Q/K/V and computes softmax
  online so the full attention matrix never materializes in HBM, turning a memory-bound op compute-bound
  and cutting memory from O(n²) to O(n). The reference for "fuse to avoid HBM round-trips."

### 4.5 NCCL, collectives, topology

**NCCL** implements GPU collectives (all-reduce, all-gather, reduce-scatter, broadcast, all-to-all)
topology-aware: it uses **NVLink/NVSwitch** for fast intra-node bandwidth and the network
(InfiniBand/RoCE, ideally GPUDirect RDMA) across nodes. Your comms ceiling is set by this topology, and
collective cost depends on the ring/tree algorithm and message size. Overlapping collectives with
compute (DDP's bucketed all-reduce, FSDP's prefetch) is how you hide it. Tune via `NCCL_*` env vars
and verify the fabric is actually being used (e.g. with `NCCL_DEBUG=INFO`).

### 4.6 fp8 / quantization on Hopper/Blackwell

Recent GPUs add **fp8** (E4M3/E5M2) tensor-core support (Blackwell pushes to even lower precision —
verify what's available). fp8 training needs **scaling/amax management** (delayed or dynamic scaling)
to keep values in range; libraries like Transformer Engine handle this. Inference quantization (int8,
fp8, int4 weight-only, KV-cache quant) is largely a [[serving-frameworks]] concern, but the kernel-level
tensor-core dtype rules here still apply.

---

## 5. TPU substrate

### 5.1 Architecture

A TPU core is built around the **MXU (Matrix Multiply Unit)**, a **systolic array**: a fixed 2-D grid
of multiply-accumulate cells through which operands are *streamed*, computing a matmul tile with very
high efficiency and low control overhead. Around it: the **VPU (Vector Processing Unit)** for
elementwise/vector/reduction work, scalar units for control, and **HBM** for capacity. The design is
matmul-first — it is spectacular on big, well-tiled GEMMs and relatively weaker on irregular/scalar
work.

- **Shape friendliness:** because the MXU is a fixed systolic grid, performance loves dimensions that
  are **multiples of the MXU/tiling size** (commonly 128 — verify for the generation). Odd shapes waste
  the array (padding). XLA pads/tiles for you, but you pay for it.
- **megacore:** on some TPU generations two MXU/compute cores share HBM and are presented/used as one
  larger logical core — verify which generation and how it's exposed.

### 5.2 Interconnect, pods, hosts, SPMD

TPU chips connect directly via **ICI (Inter-Chip Interconnect)** into a high-bandwidth mesh/torus,
scaling to a **pod** of many chips. A **host** (CPU) drives some chips; large jobs are **multi-host**,
each host feeding its local chips while the chips talk to each other over ICI. **SPMD is the native
execution model** (the same program on every chip, collectives over ICI), which is exactly what
JAX `Mesh`/sharding + GSPMD targets. **XLA is mandatory** on TPU — there is no eager fallback; all TPU
work compiles through XLA. Provisioning TPU pods/slices and node pools is [[gke-master]] /
[[aiml-on-kubernetes]] territory; multi-host *training recipes* are [[training-frameworks]].

### 5.3 Pallas — custom kernels

**Pallas** is the JAX kernel language for writing custom kernels that target **TPU (via Mosaic) and GPU
(via a Triton-style path)** when XLA's automatic fusion isn't enough (e.g. a custom attention variant,
a fused block-sparse op). You write in terms of blocks/tiles over the memory hierarchy, much like
Triton. Reach for it only when profiling shows XLA leaves real performance on the table — verify
current capabilities and constraints per backend.

---

## 6. Practical guidance & gotchas (the field manual)

### 6.1 Recompilation / dynamic shapes

- **JAX/XLA:** new shape/dtype/static value ⇒ recompile. **Bucket and pad** variable-length sequences;
  keep batch/seq dims fixed; don't thread changing Python scalars as `static_argnums`. Watch for
  unexpected recompiles by logging compile events. Enable the persistent compilation cache.
- **PyTorch `torch.compile`:** recompiles on shape change and breaks on uncaptured Python. Use
  `dynamic=True` or `mark_dynamic`; reduce graph breaks (check `TORCH_LOGS=recompiles,graph_breaks`).
- **PyTorch/XLA:** same shape-stability rules as XLA; one changing dim each step destroys it.

### 6.2 OOM debugging

1. Is it **fragmentation or true capacity?** PyTorch: compare `memory_allocated` vs `memory_reserved`;
   capture a memory snapshot. Try `expandable_segments:True`.
2. What dominates — **activations** (most often), params/grads, or optimizer state? Apply activation
   checkpointing, smaller micro-batch + grad accumulation, FSDP/sharding, lower-precision optimizer.
3. JAX: peak is dominated by the largest live buffers in the compiled program; **donate** buffers,
   shard across more devices, and avoid materializing big intermediates (let fusion keep them off HBM).

### 6.3 Host-bound input pipelines

If the accelerator idles between steps, you are host-bound, not compute-bound. Add `DataLoader`
workers, `pin_memory`, `prefetch`, `persistent_workers`; move decode/augment off the critical path or
onto the GPU (DALI); for JAX/TPU use `tf.data`/Grain-style pipelines that prefetch to device. Confirm
with a timeline (Nsight / the device profiler) showing gaps between kernels.

### 6.4 Collective / comms bottlenecks

Symptoms: scaling efficiency drops as you add nodes; profile shows long NCCL/ICI time not overlapped
with compute. Check: is **NVLink/ICI** actually used (right placement/topology)? Are collectives
**overlapped** with compute (DDP buckets, FSDP prefetch, XLA latency-hiding scheduler)? Is the message
size healthy (not many tiny collectives)? Is one slow node/link straggling? Topology-aware placement
(node pools, gang scheduling) is [[gke-master]] / [[aiml-on-kubernetes]] / [[slurm-hpc-on-kubernetes]].

### 6.5 When compile (torch.compile / XLA) helps vs hurts

- **Helps:** many small/memory-bound ops to fuse; high Python/launch overhead; stable shapes; long
  runs that amortize compile time; `max-autotune` for a hot, fixed-shape matmul/conv.
- **Hurts / neutral:** shapes change every step; pervasive graph breaks / Python side effects;
  very short jobs; code already one big cuBLAS GEMM (nothing to fuse). **Always A/B with a profile.**

### 6.6 Correctness traps

- **JAX:** forgetting RNG key splitting (silent correlation); branching on a traced value; expecting
  in-place mutation; using stale donated buffers; assuming Python `print` runs at trace time.
- **PyTorch:** missing `optimizer.zero_grad()`; in-place ops breaking autograd; forgetting
  `model.eval()` / `no_grad`; non-deterministic kernels when you needed determinism
  (`torch.use_deterministic_algorithms`); timing without `cuda.synchronize()`.

---

## 7. Canonical references (verify versions against current docs)

- **PyTorch** — docs: https://pytorch.org/docs/ ; `torch.compile`:
  https://pytorch.org/docs/stable/torch.compiler.html ; FSDP: https://pytorch.org/docs/stable/fsdp.html ;
  CUDA caching allocator & memory: https://pytorch.org/docs/stable/notes/cuda.html ; profiler:
  https://pytorch.org/docs/stable/profiler.html
- **JAX** — docs: https://docs.jax.dev/ ; sharded computation / distributed arrays and `jit`,
  `Mesh`/`NamedSharding`/`PartitionSpec`, `shard_map` are in the JAX docs' parallelism guides.
- **OpenXLA / XLA / StableHLO** — https://openxla.org/ , https://openxla.org/xla , StableHLO:
  https://openxla.org/stablehlo ; PJRT plugin interface under the OpenXLA project.
- **CUDA / GPU** — CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/ ; cuDNN, cuBLAS, NCCL,
  CUTLASS (https://github.com/NVIDIA/cutlass), Transformer Engine
  (https://github.com/NVIDIA/TransformerEngine) docs on developer.nvidia.com.
- **Triton** — https://triton-lang.org/ . **FlashAttention** — https://github.com/Dao-AILab/flash-attention
  and the FlashAttention papers.
- **TPU** — Google Cloud TPU docs: https://cloud.google.com/tpu/docs (architecture, pods, ICI).
  **Pallas** — in the JAX docs (https://docs.jax.dev/en/latest/pallas/index.html).
- **Sibling skills:** [[training-frameworks]] (DDP/FSDP/Megatron/DeepSpeed/MaxText recipes),
  [[serving-frameworks]] (vLLM/SGLang/TensorRT-LLM), [[aiml-on-kubernetes]], [[gke-master]],
  [[slurm-hpc-on-kubernetes]].
