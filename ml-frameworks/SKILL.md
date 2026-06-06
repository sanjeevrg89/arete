---
name: ml-frameworks
description: Deep expertise in the core ML compute frameworks and the accelerator stack beneath them —
  PyTorch (eager/graph, autograd, torch.compile/Dynamo/Inductor, CUDA caching allocator & memory,
  AMP/bf16/fp8, torch.distributed/NCCL, DDP vs FSDP/FSDP2, profiling), JAX (jit/grad/vmap, tracing,
  jax.sharding/Mesh/NamedSharding/shard_map, SPMD/GSPMD, donation, compilation cache), XLA/OpenXLA
  (HLO/StableHLO, fusion, layout, PJRT, xla_flags, PyTorch/XLA), CUDA GPU substrate (warps/SMs, memory
  hierarchy, tensor cores, Triton, cuBLAS/cuDNN/CUTLASS, FlashAttention, NCCL/NVLink), and TPU
  substrate (MXU systolic array, VPU, ICI, pods, Pallas, megacore). Use when writing/optimizing/debugging
  PyTorch or JAX, tuning torch.compile or XLA, chasing CUDA/TPU OOM, recompilation, MFU/roofline,
  precision (tf32/bf16/fp16/fp8), or kernel-level performance. Sibling skills own distributed-training
  orchestration and serving.
---

# ML Frameworks (PyTorch · JAX · XLA · GPU · TPU)

Apply the judgment of a performance engineer who profiles and optimizes frontier-scale models on GPU
and TPU every day — someone who reads HLO and Nsight traces, knows why a kernel is memory-bound, and
can tell whether `torch.compile` or XLA will help or hurt before running it. The bar: **never fabricate
a flag, API name, or benchmark number.** When you are unsure whether something is current, describe the
concept and tell the reader to verify against current docs — APIs and hardware gens move fast (it is 2026).

## How to use this skill

1. **Read `ml-frameworks-guide.md`** in this directory — the full reference (PyTorch, JAX, XLA, GPU,
   TPU, and the cross-cutting mental models). Apply it to the task at hand.
2. For correct, minimal patterns to imitate — a sharded JAX `jit` matmul with `Mesh`/`PartitionSpec`,
   a `torch.compile` + AMP training step, and notes on Triton/Pallas kernel shapes — read **`examples.md`**.
3. Match the surrounding codebase's framework choice and conventions; apply the correctness, precision,
   and memory rules regardless. Always measure before and after a "performance" change.

## The essentials (full detail in `ml-frameworks-guide.md`)

- **Pick the right mental model.** PyTorch is **imperative/eager** (Python is the program; graphs are
  captured opt-in by `torch.compile`). JAX is **functional/pure**: you write traceable functions of
  arrays, transform them with `jit`/`grad`/`vmap`, and a *traced* (not executed) program is compiled by
  XLA. Side effects, Python control flow on traced values, and in-place mutation are where JAX bites.
- **The roofline decides everything.** Classify every hot op as **compute-bound** or **memory-bound**
  by its arithmetic intensity (FLOPs ÷ bytes moved) vs the hardware's FLOP:byte ratio. Big GEMMs and
  conv are compute-bound (feed the tensor cores / MXU); elementwise, norm, softmax, attention-glue,
  and small ops are memory-bound (fuse them, raise intensity). **Fusion's whole point is fewer HBM
  round-trips.** MFU (model FLOP utilization) is your north-star efficiency metric.
- **`torch.compile` = Dynamo (graph capture) → AOTAutograd (joint fwd/bwd) → Inductor (codegen,
  Triton on GPU).** It helps most on many small/memory-bound ops it can fuse, and on reducing Python
  overhead. It hurts when shapes change every step (recompiles / graph breaks) — use `dynamic=True` or
  mark dynamic dims, and watch `torch._dynamo` recompilation logs. `mode="max-autotune"` trades compile
  time for runtime.
- **CUDA memory is a caching allocator, not raw `cudaMalloc`.** OOM is usually **fragmentation**, not a
  true capacity wall. Reach for `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, dump
  `torch.cuda.memory._dump_snapshot()`, and remember reserved ≠ allocated. Activation memory, not
  params, usually dominates — use activation checkpointing.
- **Precision is a lever, not a default.** `tf32` for matmuls on Ampere+ (set the matmul precision
  flag), **bf16 as the workhorse** for training (wide exponent, no loss scaling needed), fp16 only with
  a `GradScaler`, **fp8 (E4M3/E5M2) on Hopper/Blackwell** for the biggest GEMMs with care around
  scaling/amax. Keep reductions, softmax, and norms in fp32. Use AMP `autocast` rather than hand-casting.
- **Distributed is a memory/comms trade.** DDP replicates the model and all-reduces gradients (simple,
  fast when the model fits per device). **FSDP/FSDP2 shards params+grads+optimizer state**, all-gathering
  shards just-in-time — it trades extra comms for fitting bigger models. NCCL is the GPU collective
  backend; topology (NVLink/NVSwitch intra-node, network inter-node) sets your comms ceiling. Detailed
  parallelism strategy lives in [[training-frameworks]].
- **JAX makes parallelism a property of arrays.** A `jax.Array` carries a sharding; place it on a
  `Mesh` with `NamedSharding(mesh, PartitionSpec(...))`; `jit` + GSPMD infers the distributed program
  (pjit-style automatic SPMD). Drop to **`shard_map`** for explicit per-shard code (manual collectives).
  **Donate** input buffers (`donate_argnums`) to reuse memory. This is why JAX is the native fit for TPU.
- **XLA compiles a whole graph and specializes on shapes.** HLO → fusion → layout assignment → codegen,
  reached through **PJRT** device plugins. **Every new input shape can trigger a recompile** — the #1
  JAX/XLA latency trap. Cache compilations (persistent compilation cache) and keep shapes static
  (bucket/pad). Both JAX and **PyTorch/XLA** lower to XLA via StableHLO/OpenXLA.
- **GPU substrate:** threads→warps(32)→blocks→SMs; registers→shared/L1→L2→HBM. Maximize occupancy and
  feed the **tensor cores** (right dtype, aligned shapes, contiguous layout). Custom fused kernels:
  **Triton** (Python-like, autotuned) or **CUTLASS**; **FlashAttention** is the canonical IO-aware,
  memory-bound-killing attention kernel. Use cuBLAS/cuDNN for the standard heavy lifting.
- **TPU substrate:** the **MXU** is a systolic array that streams a matmul through a fixed grid (loves
  big, well-tiled matmuls; pad to MXU-friendly shapes, typically 128-multiples — verify). VPU for
  vector/elementwise, HBM for capacity. Chips link by **ICI** into pods; **SPMD is the native model**
  and **XLA is mandatory**. Custom TPU/GPU kernels: **Pallas**.
- **Input pipelines are a real bottleneck.** A host-bound `DataLoader` (too few workers, slow decode,
  no pinned memory / prefetch) starves the accelerator — profile end-to-end, not just the GPU step.
- **Profile, don't guess.** PyTorch: `torch.profiler` + Nsight Systems/Compute. JAX/XLA: the device
  profiler and the compiled-HLO/trace viewer. Confirm whether you are kernel-bound, host-bound, or
  comms-bound *before* optimizing.

## Related skills

- `[[training-frameworks]]` — DDP/FSDP strategy, DeepSpeed, Megatron, NeMo, MaxText: the multi-node
  *orchestration* of parallelism. This skill owns the framework primitives; that one owns the recipes.
- `[[serving-frameworks]]` — vLLM, SGLang, TensorRT-LLM, Triton, Dynamo: inference/serving runtime.
- `[[aiml-on-kubernetes]]` — running this stack on Kubernetes/GKE (umbrella).
- `[[gke-master]]` — provisioning TPU/GPU node pools, drivers, topology-aware placement.
- `[[slurm-hpc-on-kubernetes]]` — Slurm/HPC scheduling, MPI, RDMA for these workloads.
