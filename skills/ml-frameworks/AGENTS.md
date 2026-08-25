# AGENTS.md — ML Frameworks (PyTorch · JAX · XLA · GPU · TPU)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ml-frameworks-guide.md`** next to this file — read it
> before writing or optimizing PyTorch/JAX/XLA or kernel-level GPU/TPU code, and apply it. Correct,
> minimal patterns to imitate (sharded JAX `jit` matmul, `torch.compile` + AMP step, Triton/Pallas
> kernel shapes) are in **`examples.md`**. This file is the always-on summary.
>
> **Hard rule: never fabricate a flag, API name, dtype-support claim, or benchmark number.** APIs and
> hardware gens move fast (2026). If unsure whether something is current, describe the concept and tell
> the reader to verify against current docs. Scope = frameworks + accelerator substrate; multi-node
> *orchestration* is [[training-frameworks]], serving is [[serving-frameworks]].

## When working in PyTorch / JAX / XLA / GPU / TPU code, apply these by default:

- **Use the right mental model.** PyTorch = imperative/eager (Python is the program; `torch.compile`
  captures graphs opt-in). JAX = pure/functional: traceable array functions transformed by
  `jit`/`grad`/`vmap`; you compile a *traced* program, not an executed one. No in-place mutation, no
  Python branching on traced values, explicit RNG keys.
- **Classify every hot op via the roofline.** Compute-bound (big GEMM/conv/attention matmul → feed
  tensor cores/MXU with the right dtype + shapes) vs memory-bound (elementwise, norm, softmax, small
  ops → **fuse** to cut HBM round-trips). Fusion's purpose is fewer HBM trips. Track **MFU**.
- **`torch.compile` = Dynamo → AOTAutograd → Inductor (Triton on GPU).** Helps on small/memory-bound
  ops and Python overhead; hurts on changing shapes (recompiles) and graph breaks. Use `dynamic=True` /
  `mark_dynamic`; watch `TORCH_LOGS=recompiles,graph_breaks`; `max-autotune` trades compile for run.
- **CUDA memory is a caching allocator** — reserved ≥ allocated; OOM is usually **fragmentation**, not
  capacity. Diagnose with `memory_summary()` + memory snapshot; try
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Activations usually dominate** — use activation
  checkpointing, smaller micro-batch + grad accumulation, or FSDP.
- **Precision is deliberate:** **bf16 is the training workhorse** (no scaler). fp16 only with a
  `GradScaler`. tf32 for fp32 matmuls (`set_float32_matmul_precision`). **fp8 (E4M3/E5M2) on
  Hopper/Blackwell** for the biggest GEMMs with scaling/amax care. Keep reductions/softmax/norms/master
  weights fp32. Prefer AMP `autocast` over hand-casting.
- **Distributed = memory/comms trade.** DDP replicates + all-reduces grads (model must fit per device).
  **FSDP/FSDP2 shards params+grads+optimizer state**, all-gather just-in-time — fits bigger models at
  more comms. NCCL is the GPU backend; NVLink/NVSwitch intra-node, network inter-node set the ceiling.
  Strategy detail → [[training-frameworks]].
- **JAX parallelism = a property of arrays.** Build a `Mesh`, annotate with
  `NamedSharding(mesh, PartitionSpec(...))`, `jit` lets **GSPMD** infer the SPMD program + collectives
  (pjit-style). Drop to **`shard_map`** for explicit per-shard code. **Donate** buffers
  (`donate_argnums`) to cut peak memory. Enable the persistent compilation cache.
- **XLA specializes on static shapes** → **new shape = recompile** (the #1 JAX/XLA/PyTorch-XLA latency
  trap). Bucket/pad shapes; cache compilations; consider AOT for serving. Pipeline: HLO/StableHLO →
  fusion → layout assignment → codegen, via **PJRT** device plugins. Don't invent `XLA_FLAGS` — look
  them up; the safe one is dumping optimized HLO.
- **GPU substrate:** threads→warps(32)→blocks→SMs; registers→shared/L1→L2→HBM. Maximize occupancy,
  coalesce memory, feed **tensor cores** (dtype + aligned/contiguous shapes). Custom fused kernels:
  **Triton** or **CUTLASS**; **FlashAttention** for IO-aware attention; cuBLAS/cuDNN for standard heavy
  ops; **NCCL** topology-aware for collectives.
- **TPU substrate:** **MXU systolic array** loves big, tile-friendly matmuls (pad to MXU multiples —
  verify, commonly 128); VPU for vector work; **ICI** links chips into pods; **SPMD is native** and
  **XLA is mandatory** (no eager). Custom TPU/GPU kernels: **Pallas**.
- **Don't guess — profile.** Answer **kernel-bound vs host-bound vs comms-bound** first. PyTorch:
  `torch.profiler` + Nsight Systems/Compute. JAX/XLA: device profiler + dumped HLO/trace. A starved
  accelerator (idle between steps) means a host-bound `DataLoader` (workers/`pin_memory`/prefetch).

## Correctness traps to never ship

- **JAX:** branching on a traced value (use `lax.cond`/`scan`); expecting in-place mutation; missing
  RNG key splits; using a donated buffer after donation; assuming `print` runs at trace time.
- **PyTorch:** missing `zero_grad`; in-place op breaking autograd; forgetting `eval()`/`no_grad`;
  timing without `cuda.synchronize()`; fp16 without a scaler.

## Definition of done
Measure before/after with a profiler and report honestly: is the workload now kernel-, host-, or
comms-bound? Confirm numerics didn't regress (precision change), peak memory fits, and no silent
recompiles. For any flag/API you weren't sure of, state that it needs verification against current docs.
