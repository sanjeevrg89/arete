# AGENTS.md — GPU Performance Engineering

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`gpu-performance-engineering-guide.md`** next to this
> file — read it before profiling or optimizing GPU code, and apply it. Concrete worked artifacts to
> imitate (roofline triage checklist, an Nsight Compute memory-vs-compute read, a cross-rank
> straggler note) are in **`examples.md`**. This file is the always-on summary.
>
> **The toolchain moves fast (it is 2026).** Verify flags, Nsight section/counter/metric names,
> `nvidia-smi`/DCGM field names, MLPerf rules, and any KPerfIR/Proton citation against current docs
> before relying on them. Never fabricate flags, counter names, or benchmark figures.

## When profiling or optimizing GPU workloads, apply these by default:

- **Measure before you optimize; classify against a roofline first.** Arithmetic intensity (FLOP/byte)
  vs the **ridge point** (`peak_FLOPs/peak_bandwidth`) decides the strategy: left of ridge =
  memory-bound (raise AI: fuse, reuse in shared/registers, coalesce, recompute, better dtype); right =
  compute-bound (tensor cores, tiling, ILP, lower-precision path); under both ceilings = latency-bound
  (more parallelism). Roofline against the peak for the **math path the kernel actually uses** (FP32 vs
  tensor-core FP16/BF16/FP8/INT8), and against each memory level (HBM/L2/shared).
- **Two tools, two jobs — `nsys` first, `ncu` second.** Nsight Systems (`nsys`) = the timeline: GPU
  gaps (host/comms-bound), overlap, which kernel dominates; annotate with NVTX. Nsight Compute (`ncu`)
  = one *named* kernel deep (it replays, so it's slow — never aim it at a whole step): read
  **Speed-of-Light (SOL)** to classify (high-compute/low-mem = compute-bound; reverse = memory-bound;
  *both low* = latency/occupancy-bound), then Memory & Compute Workload, Occupancy, Warp State.
- **`nvidia-smi` "GPU-Util" lies** — it means a kernel was resident, not that the GPU did useful work.
  Use SOL, achieved FLOP/s, and MFU as truth. Never optimize off the util number.
- **Occupancy is a means (latency hiding), not a goal.** High occupancy + low SOL = warps all stalled
  (fix the stall, named in Warp State — e.g. long-scoreboard = memory latency). Low occupancy can be
  fast with high ILP. Read the occupancy *limiter* (registers / shared mem / block size); watch
  register spills to local memory (HBM).
- **Memory mechanics:** coalesce global access (watch sectors-per-request); kill shared-memory **bank
  conflicts** (pad inner dim, e.g. `tile[32][33]`); confirm **tensor cores are actually engaged** via
  the MMA pipe utilization (wrong dtype/shape/alignment silently falls back to CUDA cores). Prefer
  tuned libraries (cuBLAS/cuDNN/CUTLASS) over hand-rolled tensor-core code.
- **IR-embedded profiling** (KPerfIR / Proton as MLIR/LLVM dialects in Triton) preserves op-level
  attribution through fusion — external profilers attribute fused-kernel time to the blob, not your op.
  Use NVTX and compiler profiling hooks for attribution. **Verify the citation** before quoting IDs.
- **At scale, the bug is the difference between ranks.** Profile *every* rank, not one. Hunt
  **stragglers** by the cross-rank **differential**: per-rank step-time *distribution*, SM clocks,
  temps, power, throttle-reason bitmask, NVLink/PCIe replays. The classic trap — one GPU
  thermal-throttles and clocks down, everyone blocks at the all-reduce, **every rank reads 100% util**;
  only the differential reveals it. Fix the root cause (cool/reseat/drain), then re-measure the spread.
- **Continuous, cross-layer, low-overhead telemetry:** CPU stack sampling (perf / **eBPF**) + GPU
  kernel tracing (CUPTI / NVTX phases) + **NCCL/collective** timing, always-on — so you have the
  profile from the run that was actually slow.
- **Benchmark like MLPerf, measure with statistics.** Tagged, unmodified **LoadGen**; documented
  preprocessing/weights/hyperparameters with **commit hashes/checksums**; respect **availability tiers**
  (Available / Preview / RDI — never compare across tiers). Always: **warm up** (skip JIT/autotune/cold
  clocks), **synchronize before stopping the clock** (or use CUDA events), report **percentiles
  (p50/p90/p99), not means**, quantify run-to-run variance, pin/log clocks and versions. (Do **not**
  assert a fixed "four scenarios" taxonomy — verify the current MLPerf scenario set.)

## Triage order (classify before optimizing)
1. **GPU idle / timeline gaps?** → host-bound (perf/eBPF → dataloader/GIL/sync/copies) or comms-bound
   (NCCL timing → overlap/buckets/topology). Don't profile kernels.
2. **Straggler?** (multi-rank) → per-rank differential (§ above).
3. **One kernel dominates?** → Nsight Compute SOL → memory- / compute- / latency-bound → fix per the
   roofline.
4. **Re-measure** with a reproducible benchmark. Amdahl: don't micro-optimize a non-bottleneck.

## Anti-patterns to flag
Optimizing without a roofline · trusting `nvidia-smi` 100% · averaging instead of percentiles ·
profiling one rank · micro-optimizing a non-bottleneck · no reproducible benchmark (no warmup/variance,
untagged loadgen) · `ncu` on a whole step · not synchronizing before timing · assuming tensor cores are
on · comparing across availability tiers.
