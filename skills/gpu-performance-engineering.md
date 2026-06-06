---
name: gpu-performance-engineering
description: GPU kernel performance engineering and cross-layer systems profiling — the discipline of
  actually finding and fixing where the FLOPs and bandwidth go, at frontier scale (distinct from
  framework-level "make torch.compile faster"). Use when profiling or optimizing GPU kernels, reading a
  roofline (arithmetic intensity, compute-bound vs memory-bound, ridge point), using NVIDIA Nsight
  Compute (ncu — Speed-of-Light/SOL, memory & compute workload analysis, occupancy, warp-stall reasons,
  tensor/MMA pipe utilization, coalescing, bank conflicts) or Nsight Systems (nsys timeline/overlap,
  NVTX, CUPTI), diagnosing host-bound vs kernel-bound vs memory-bound vs comms-bound vs straggler
  problems, hunting stragglers across ranks (the "nvidia-smi 100% but a GPU thermal-throttled" pattern),
  doing cross-layer continuous profiling (perf/eBPF + GPU kernel tracing + NCCL/collective
  instrumentation), IR-embedded profiling (KPerfIR/Proton MLIR/LLVM dialects in Triton), or rigorous
  benchmarking with MLPerf (LoadGen, reproducibility/source rules, availability tiers) and valid perf
  methodology (warmup, synchronize, percentiles not means, variance). Reach for it whenever the question
  is "why is this GPU/kernel/job slow and what do I optimize?"
---

# GPU Performance Engineering

Apply the judgment of an engineer who profiles and optimizes GPU workloads at frontier scale: who never
optimizes without a roofline, never trusts `nvidia-smi` "100%", and never reports a mean when the p99
and the straggler are what matter. The job is to **find where the FLOPs and bytes actually go**, classify
the bottleneck, and fix the one that counts — from a single kernel up to a multi-thousand-GPU run.

## How to use this skill

1. **Read `gpu-performance-engineering-guide.md`** in this directory — the full reference (roofline,
   the Nsight stack, occupancy/coalescing/bank-conflicts/tensor cores, IR-embedded profiling,
   cross-rank/continuous profiling, MLPerf methodology, the end-to-end triage workflow, anti-patterns).
   Apply it to the task.
2. For concrete worked artifacts to imitate — a roofline-driven kernel-triage checklist, an Nsight
   Compute "memory-bound vs compute-bound" read, and a cross-rank straggler-diagnosis note — read
   **`examples.md`**.
3. Match the surrounding tooling/cluster conventions; apply the measurement-before-optimization
   discipline regardless. The toolchain moves fast — **verify flags, section/counter names, MLPerf
   rules, and citations against current docs** before relying on them.

## The essentials (full detail in `gpu-performance-engineering-guide.md`)

- **Profile before you optimize; classify against a roofline first.** Arithmetic intensity (FLOP/byte)
  vs the ridge point decides everything: memory-bound → raise AI (fuse, reuse, coalesce, recompute);
  compute-bound → use tensor cores, better tiling/ILP; latency-bound → more parallelism. Roofline
  against the peak for the *math path the kernel actually uses*.
- **Two tools, two jobs.** `nsys` (Nsight Systems) **first** — the timeline: GPU gaps, overlap,
  host/comms stalls, which kernel dominates. `ncu` (Nsight Compute) **second** — one named kernel deep:
  Speed-of-Light, Memory & Compute Workload, occupancy, warp-stall reasons, tensor/MMA pipe.
- **`nvidia-smi` "GPU-Util" LIES** — it means "a kernel was resident," not "the GPU did useful work." A
  3%-of-SMs kernel reads 100%. Use SOL / achieved FLOP/s / MFU as truth.
- **Occupancy is a means (latency hiding), not a goal.** High occupancy + low SOL = warps all stalled;
  low occupancy can be fast with high ILP. Read the occupancy *limiter* (registers/shared/block).
- **Memory mechanics:** coalesce (watch sectors-per-request), kill shared-memory **bank conflicts**
  (pad the inner dim), confirm **tensor cores are actually on** (MMA pipe util — wrong dtype/shape
  silently falls back to CUDA cores), watch register spills to local memory.
- **At scale the bug is the difference between ranks.** Hunt stragglers by the **cross-rank
  differential** — per-rank step-time distribution, SM clocks, temps, throttle reasons. The classic:
  one GPU thermal-throttles, everyone waits at the all-reduce, *every rank reads 100%*.
- **Continuous, cross-layer, low-overhead observability:** CPU stack sampling (perf/eBPF) + GPU kernel
  tracing (CUPTI/NVTX) + NCCL/collective timing, always-on, so you have the profile from the slow run.
- **IR-embedded profiling** (KPerfIR/Proton as MLIR/LLVM dialects in Triton) keeps op-level attribution
  through fusion — **verify the citation** before quoting IDs/dialect names.
- **Benchmark like MLPerf:** tagged unmodified **LoadGen**, documented preprocessing/weights/hparams
  with commit hashes/checksums, **availability tiers** (Available / Preview / RDI — don't compare
  across). Always warm up, synchronize before stopping the clock, report **percentiles not means**,
  quantify variance, pin/log clocks.
- **Triage order:** GPU idle? → host- or comms-bound (don't profile kernels). Straggler? → per-rank
  differential. One kernel dominates? → Nsight Compute → memory/compute/latency-bound → fix per the
  roofline. Then re-measure (Amdahl: don't polish a non-bottleneck).

## Related skills

- `[[ml-compilers-codegen]]` — Triton/MLIR/LLVM codegen and IR-embedded profiling (KPerfIR/Proton).
- `[[ml-frameworks]]` — PyTorch/JAX/XLA; where the kernels and input pipeline come from.
- `[[inference-optimization]]` — model-level compression/decode acceleration; memory-bandwidth vs
  compute-bound reasoning at the model level.
- `[[ai-networking-collectives]]` — NCCL/collectives and the fabric, when the bottleneck is comms.
- `[[training-frameworks]]` — DDP/FSDP/Megatron input pipelines and overlap, for host-bound fixes.
- `[[aiml-on-kubernetes]]` — getting profilers/capabilities onto cluster nodes (GKE, K8s).

---

# Reference — gpu-performance-engineering

# GPU Performance Engineering — Guide

The full reference. This is the discipline of **finding and fixing where the FLOPs and bytes actually
go** on GPUs at frontier scale — not framework-level "make `torch.compile` go faster," but the layer
below: read a kernel profile, classify the bottleneck against a roofline, and decide what (if anything)
to optimize. It spans a single kernel up through a multi-thousand-GPU job, and it insists on
*measurement before optimization*. The ecosystem moves fast (it is 2026); treat every flag, counter
name, metric name, and section name below as **verify against current docs** unless you have a profile
open in front of you confirming it.

---

## 1. Mental model: the roofline, and "where do the bytes/FLOPs go?"

The roofline model is the single most important mental frame. Plot **attainable performance**
(FLOP/s, log scale) against **arithmetic intensity** (AI = FLOPs performed per byte moved from a given
memory level, FLOP/byte, log scale). Two ceilings bound any kernel:

- **Compute ceiling** — a horizontal line at the device's peak FLOP/s for the relevant datatype/path
  (FP32 vs FP16/BF16 tensor-core vs FP8 vs INT8). These differ by an order of magnitude; *which* peak
  you compare against matters enormously.
- **Memory ceiling** — a sloped line of slope = peak bandwidth of the memory level you're measuring
  against (HBM, L2, L1/shared). `attainable = min(peak_FLOPs, AI × bandwidth)`.

The **ridge point** is where the sloped memory line meets the flat compute line: `AI_ridge = peak_FLOPs
/ peak_bandwidth`. A kernel whose AI is left of the ridge is **memory-bound** (you are bandwidth
limited; more FLOPs are free); right of the ridge it is **compute-bound** (you are FLOP limited; moving
fewer bytes is free). This one decision — which side of the ridge am I on — dictates the entire
optimization strategy:

| Bound | What's saturated | What to do | What is *wasted effort* |
|---|---|---|---|
| Memory-bound | HBM/L2/shared bandwidth | Raise AI: fuse ops, reuse data in shared/registers, recompute instead of reload, better dtype, coalesce | Adding more math, more tensor cores |
| Compute-bound | ALU / tensor-core throughput | Use tensor cores, raise occupancy/ILP, better tiling, lower-precision math path | Bandwidth tricks, caching |
| Latency-bound | Neither — too little work in flight | More parallelism, hide latency, batch, bigger tiles | Roofline ceilings (you're under both) |

Two refinements that matter at frontier scale:

- **Hierarchical roofline.** Compute AI against *each* memory level (HBM, L2, shared/L1). A kernel can
  be HBM-bound but have huge L2 headroom — that tells you a caching/fusion fix exists. Nsight Compute's
  roofline chart can plot multiple ceilings; read all of them.
- **The "right peak" trap.** For a matmul on tensor cores, comparing to the FP32 CUDA-core peak makes
  it look gloriously compute-bound when it's actually leaving the tensor-core peak untouched. Always
  roofline against the peak for the *math path the kernel actually uses*.

**Arithmetic intensity drives algorithm design.** FlashAttention is the canonical example: standard
attention materializes the N×N scores matrix to HBM (low AI, memory-bound); the fused tiling keeps
tiles in SRAM and recomputes in the backward pass, raising AI and moving the kernel toward
compute-bound. You cannot reason about *why* it's faster without the roofline.

---

## 2. The CUDA profiling stack: Nsight Systems vs Nsight Compute

NVIDIA's two primary tools answer different questions. Use them in this order.

### Nsight Systems (`nsys`) — the timeline, first

System-wide, low-overhead **timeline** profiler. It answers *"where does wall-clock time go, and what
overlaps with what?"* across CPU threads, CUDA API calls, kernel execution, memcpy, and (with the
right trace options) NCCL/NVTX ranges. Use it **first**, always, before touching a single kernel.

```bash
# Capture a timeline; trace CUDA, NVTX ranges, OS runtime, and NCCL.
nsys profile -t cuda,nvtx,osrt,nccl \
  -o run_%p --force-overwrite true \
  python train.py
# Then open the .nsys-rep in the GUI, or summarize on the CLI:
nsys stats run_<pid>.nsys-rep
```

What you are reading for, in order:
- **Gaps on the GPU timeline** — the GPU is idle. The job is *host-bound* (CPU can't feed it) or
  *comms-bound* (waiting on a collective). No kernel optimization will help; fix the feeder.
- **Serialization that should overlap** — compute not overlapping with H2D/D2H copies, or compute not
  overlapping with NCCL. Often a stream/dependency issue.
- **Which kernels dominate** — the cumulative time-by-kernel table tells you *which* kernel is worth
  taking into Nsight Compute. Annotate your code with **NVTX** ranges (`torch.cuda.nvtx.range_push/pop`
  or `nvtx.annotate`) so the timeline maps to *your* phases (forward/backward/optimizer), not anonymous
  kernels.

### Nsight Compute (`ncu`) — one kernel, deeply

Interactive **kernel** profiler. It answers *"for this specific kernel, what is the bottleneck and how
close am I to the limit?"* It replays each kernel (it can run a kernel many times to gather all
counters), so it is **high-overhead** — never point it at a whole training step; target a handful of
kernels.

```bash
# Profile a few invocations of one kernel by name, full section set, with roofline.
ncu --launch-skip 20 --launch-count 3 \
    --kernel-name regex:".*attention.*" \
    --set full \
    -o attn_profile \
    python repro.py
```

The sections you live in:
- **GPU Speed Of Light (SOL)** — top-line: what % of peak compute and what % of peak memory throughput
  this kernel achieves. The single fastest classification: high compute SOL + low memory SOL =
  compute-bound; the reverse = memory-bound; *both low* = latency/occupancy-bound (the most common and
  most-missed case).
- **Memory Workload Analysis** — the memory hierarchy diagram: bytes and hit rates at L1/TEX, L2, HBM;
  whether loads are **coalesced** (sectors-per-request close to ideal) or scattered; shared-memory
  **bank conflicts**.
- **Compute Workload Analysis** — pipe utilization, including the **tensor (MMA) pipe**. If a matmul
  shows low tensor-pipe utilization, you're not actually using tensor cores (wrong dtype, misaligned
  shapes, or a non-MMA codegen path).
- **Occupancy** — achieved vs theoretical warps/SM, and the limiter (registers, shared memory, block
  size). Occupancy is a *means* to latency hiding, not a goal (see §3).
- **Warp State / Scheduler Statistics** — *why* warps stall: `Stall Long Scoreboard` (waiting on global
  memory loads → memory latency bound), `Stall MIO Throttle`, `Stall Barrier` (`__syncthreads`
  imbalance), `Stall Wait`, `Stall Not Selected` (enough work, just scheduling). The dominant stall
  reason is your single best clue.

The built-in **roofline chart** plots this kernel's achieved point against the ceilings — read it as in
§1. Confirm counter/section names against the current Nsight Compute docs; NVIDIA renames and
reorganizes sections across versions.

---

## 3. Occupancy, latency hiding, and the memory-access mechanics

### Occupancy is necessary, not sufficient

Occupancy = active warps per SM ÷ max warps per SM. Its *only* purpose is **latency hiding**: while
some warps stall on memory, the scheduler runs others. But:

- **High occupancy with low SOL** means you have warps but they're all stalled on the same thing
  (usually memory). More occupancy won't help — fix the stall.
- **Low occupancy can still be fast** if a kernel has high **ILP** (independent work per thread) — the
  classic Volkov result that you can hide latency with instruction-level parallelism instead of warp
  count. Register-heavy, low-occupancy kernels are sometimes optimal.
- Occupancy is *capped* by the scarcest resource per SM: registers/thread, shared memory/block, or the
  block-count limit. Nsight Compute's occupancy section names the limiter. Spilling registers to local
  memory (which lives in HBM) is a silent killer — watch for local-memory traffic.

### Memory coalescing and sectors

Global memory moves in **sectors** (commonly 32 bytes) grouped into 128-byte transactions. When the 32
threads of a warp access contiguous, aligned addresses, one transaction serves them all (coalesced).
Strided/scattered access inflates **sectors-per-request**, multiplying HBM traffic and tanking
effective bandwidth. In Nsight Compute, the Memory Workload section shows sectors/request and an
excessive-sector indicator. Fix with structure-of-arrays layout, transposes via shared memory, and
aligned, vectorized loads (`float4`).

### Shared-memory bank conflicts

Shared memory has banks (commonly 32); if threads in a warp hit different addresses in the *same* bank,
accesses serialize (an N-way conflict is N× slower). Classic in tiled GEMM/transpose with a stride that
is a multiple of the bank count. Fix by **padding** the inner dimension (e.g. `tile[32][33]`) or
swizzling. Nsight Compute reports shared-memory bank conflicts directly.

### Tensor-core utilization

Tensor cores (the MMA path) deliver the FP16/BF16/FP8/INT8 peaks. They are easy to *not* use:
- Wrong dtype (FP32 math won't touch them on most paths), or unsupported shapes/alignment.
- Tile shapes not matching the MMA fragment shape, so the compiler falls back to CUDA cores.
- Memory-bound feeding: the cores are idle waiting on operands (raise AI / improve the data path).

Check the tensor/MMA pipe utilization in the Compute Workload section. Library kernels (cuBLAS,
cuDNN, CUTLASS-based) are usually the right call; hand-rolled tensor-core code rarely beats them.

---

## 4. Compiler-level / IR-embedded profiling

A significant trend is **moving profiling into the compiler IR** rather than bolting it on as an
external sampler — so the profile is attributed to IR operations across lowering stages, surviving
fusion and rewriting. This matters because kernel fusion (Triton, Mosaic, Inductor, XLA) destroys the
source-line ↔ machine-instruction mapping that external profilers rely on; IR-level instrumentation
keeps attribution meaningful through the compilation pipeline. Link **[[ml-compilers-codegen]]** for
the codegen side.

- **KPerfIR** describes embedding performance instrumentation as **MLIR dialects inside the Triton
  compiler**, inserting/lowering profiling intrinsics at multiple IR levels (TTIR → TTGIR → LLVM) so
  measurement is co-designed with the kernel and survives optimization passes. Treat the specific paper
  ID as **verify the citation** — name the work ("KPerfIR, a compiler-centric / IR-embedded GPU
  profiling approach in Triton") and confirm the arXiv ID and exact dialect names against the source
  before quoting.
- The **Proton** profiling effort (a Triton-associated profiler with an LLVM/MLIR **Proton dialect**
  for intra-kernel instrumentation) is the same idea at the LLVM level. Again **verify the citation**
  and current dialect/op names.

Practical takeaway even if you never write a compiler pass: when you profile *fused* kernels, external
tools attribute time to the fused blob, not your op. Use NVTX ranges and, where available, the
compiler's own profiling hooks to get back op-level attribution. Expect this area to mature fast —
**verify current docs**.

---

## 5. Cross-layer, cross-rank, continuous profiling at scale

At a single kernel, Nsight Compute suffices. At a 1,000-GPU job, the bottleneck is rarely a kernel —
it's a host stall, a slow collective, or **one straggler rank**. You need *always-on, low-overhead,
cross-layer, cross-rank* observability, not a one-shot trace.

### The layers you must correlate

1. **CPU stack sampling** — `perf record`/`perf` and increasingly **eBPF**-based continuous profilers
   (e.g. Parca, Pyroscope, or bespoke eBPF tooling) to catch host-bound stalls: dataloader starvation,
   Python GIL contention, sync points (`.item()`, `torch.cuda.synchronize`), pinned-memory copies.
2. **GPU kernel tracing** — CUPTI-based traces (what `nsys` uses under the hood) or sampled kernel
   timing, ideally with NVTX phase annotations so GPU time maps to forward/backward/optimizer/comm.
3. **Collective / NCCL instrumentation** — NCCL timing, the `NCCL_DEBUG`/`NCCL_DEBUG_SUBSYS` logs,
   and per-collective timing to see whether time goes to all-reduce/all-gather vs compute. Link
   **[[ai-networking-collectives]]** for the collectives and fabric.

Continuous profilers (eBPF) are the right tool here precisely because they are low-overhead enough to
run *always*, so you have the profile from the run that was slow — not a re-run that doesn't reproduce.
Frameworks describing this whole-system, AI-cluster-oriented approach (combining eBPF host profiling +
GPU + collective views for diagnosis) exist; **verify the citation** for any specific named system
before quoting it.

### Why standard utilization metrics LIE

`nvidia-smi` "GPU-Util" (and the DCGM equivalent) reports the **fraction of time ≥1 kernel was
resident** — *not* how much of the GPU's compute the kernel used. A kernel that uses 3% of the SMs at
1% occupancy for 100% of the wall clock reads **100% utilization**. This is the most common and most
expensive trap in the field.

- **100% util, low throughput** → a memory-bound or low-occupancy kernel hogging the timeline. Only
  Nsight Compute SOL reveals it.
- **100% util everywhere but the job is slow** → a *straggler*. All ranks block at the next collective
  waiting for the slow one; from each rank's view it's "busy" (spinning in the all-reduce). See below.

Use SOL (Nsight Compute), tensor-active / SM-active counters (DCGM/CUPTI), and *achieved FLOP/s vs
peak* (Model FLOPs Utilization, MFU) as truth — not the binary "util" number.

### Straggler hunting: the cross-rank differential method

The defining failure mode at scale: a synchronous-SGD job runs at the speed of its slowest rank. One
GPU thermal-throttles (or has a bad NVLink, a noisy neighbor, a slow disk for that shard) and clocks
down; every other rank then *waits at the all-reduce*, so **every rank's `nvidia-smi` reads ~100%** and
every per-rank dashboard looks healthy. The signal is only visible in the **differential across ranks**.

The methodology:
1. **Collect a comparable metric per rank**, time-aligned: per-step time, per-collective wait time,
   SM clock (`nvidia-smi --query-gpu=clocks.sm,clocks_throttle_reasons.active`), GPU temp, power,
   HBM ECC errors, NVLink/PCIe replay counters.
2. **Look at the distribution, not the mean.** One rank with a fat right tail in step time, or
   conspicuously *lower SM clock*, or *less* collective-wait (because everyone waits on *it*), is the
   straggler. The throttle-reasons bitmask (`clocks_throttle_reasons.hw_thermal_slowdown`,
   `sw_thermal_slowdown`, `hw_power_brake`) confirms a thermal/power cause — verify exact field names
   against current `nvidia-smi`/DCGM docs.
3. **Confirm the topology of the wait.** In `nsys`/NCCL traces, the straggler enters the collective
   *late* and exits with everyone; the others sit in the collective long before it. That asymmetry is
   the fingerprint.
4. **Fix the root cause, not the symptom:** reseat/cool the throttling GPU, drain the bad node, fix the
   slow data shard, or pin clocks. Re-measure the distribution.

This is why per-rank, always-on, cross-layer telemetry is non-negotiable at scale — a single-rank
profile and a "100% util" dashboard will both tell you everything is fine while you burn a cluster.

---

## 6. Benchmarking & performance methodology (MLPerf)

Ad-hoc timing (`time.time()` around a loop, report the mean, eyeball it) is how teams ship regressions
and unreproducible numbers. Rigorous performance measurement is a discipline; **MLPerf** (MLCommons)
is the reference for how to do it right, and a good model even for your own private benchmarks.

### MLPerf Inference essentials

- **LoadGen** — MLCommons' standard load generator drives the system under test and *measures* it; the
  benchmark code (the SUT) does not get to implement its own timing. Submissions must use a **tagged,
  unmodified LoadGen** so latency/throughput accounting is identical and comparable across submitters.
  Verify the exact LoadGen version/tag and run rules against the current MLPerf Inference rules.
- **Reproducibility / source rules** — submissions are governed by strict rules: documented
  preprocessing, the exact weights/checkpoints, hyperparameters, and **commit hashes / checksums** so a
  result can be reproduced. The point is *auditable* numbers, not vibes.
- **Availability tiers** classify what you're allowed to claim based on what you can actually buy/use:
  **Available** (generally available hardware/software), **Preview** (will be available within a
  defined window), and **RDI** (Research, Development, or Internal — not for sale). A headline number
  in RDI is not comparable to one in Available; always check the tier before comparing.

Do **not** assert a fixed "four standardized scenarios" taxonomy — that specific claim does not hold
across MLPerf's evolution. Describe LoadGen, the reproducibility/source rules, and the availability
tiers; for anything about specific scenarios, query patterns, or metrics, **verify against the current
MLPerf Inference rules** (`github.com/mlcommons/policies` and the datacenter inference benchmark pages).

### Statistically valid performance testing (applies to any benchmark)

- **Warm up.** Discard the first iterations: lazy CUDA context init, JIT/autotuning (cuBLAS heuristics,
  `torch.compile`, Triton autotune), allocator caching, and clock spin-up all skew cold runs.
- **Synchronize correctly.** GPU work is async; `torch.cuda.synchronize()` (or CUDA events
  `cudaEventRecord`/`Elapsed`) before stopping the clock, or you're timing kernel *launch*, not
  *execution*.
- **Report percentiles, not means.** Latency distributions are right-skewed; the mean hides the tail.
  Report p50/p90/p99 (and max). For SLO-bound serving, **p99 is the number that matters**. Reporting a
  mean is the single most common benchmarking sin.
- **Quantify variance.** Multiple runs, report spread (stdev / IQR). A 3% "win" inside 5% run-to-run
  noise is not a win.
- **Control the environment.** Pin/record clocks (`nvidia-smi -lgc`), power limit, driver/CUDA/library
  versions, and shapes. Thermal drift across a long run is itself a confound — log clocks throughout.
- **Measure the right unit.** Throughput (tokens/s, samples/s) vs latency (TTFT, per-token) vs MFU
  answer different questions; pick the one tied to the decision and the SLO.

---

## 7. The end-to-end triage workflow

When something is "slow," classify *before* optimizing. The decision tree, with the tool per layer:

1. **Is the GPU even busy?** Look at the `nsys` timeline (and per-rank, at scale). GPU idle / gaps →
   it's **host-bound** or **comms-bound**, not kernel-bound. Stop; don't profile kernels.
   - Host-bound: CPU stack sampling (perf/eBPF) → dataloader, GIL, sync points, H2D copies. Fix the
     feeder (more workers, prefetch, pin memory, async copies, fewer syncs). Link
     **[[training-frameworks]]** / **[[ml-frameworks]]** for the input-pipeline fixes.
   - Comms-bound: NCCL/collective timing → bucket sizes, overlap comm with compute, topology, fabric.
     Link **[[ai-networking-collectives]]**.
2. **Is it a straggler?** (Multi-rank only.) Per-rank differential — clocks, temps, step-time
   distribution (§5). One slow rank masquerades as everyone at 100%.
3. **GPU busy, one kernel dominates?** Take *that* kernel into Nsight Compute. Read SOL → memory-bound
   vs compute-bound vs latency/occupancy-bound. Then §1's table tells you what to do (and what not to).
4. **Memory-bound kernel** → raise AI: fuse, reuse in shared/registers, coalesce, fix bank conflicts,
   better dtype, recompute-don't-reload. Often the fix is *algorithmic* (FlashAttention-style fusion),
   reachable via the compiler — link **[[ml-compilers-codegen]]**, **[[inference-optimization]]**.
5. **Compute-bound kernel** → ensure tensor cores are actually used (MMA pipe util), better tiling,
   lower-precision math path, raise ILP. Prefer tuned libraries (cuBLAS/cuDNN/CUTLASS).
6. **Latency/occupancy-bound kernel** (both SOLs low) → more parallelism, bigger launch, fix occupancy
   limiter (registers/shared/block size) or exploit ILP; consider CUDA graphs to kill launch overhead
   for many tiny kernels.
7. **Re-measure with a reproducible benchmark** (§6). Confirm the fix moved the *bottleneck* metric and
   didn't just shuffle time elsewhere (Amdahl: optimizing a 5%-of-time kernel caps you at a 5% win).

On Kubernetes/GKE clusters, getting profilers and the right capabilities onto nodes is its own task —
link **[[aiml-on-kubernetes]]**.

---

## 8. Anti-patterns (each one burns real GPU-hours)

- **Optimizing without a roofline.** You cannot know whether to add math or remove bytes until you know
  which side of the ridge you're on. Profile first; classify; then optimize.
- **Trusting `nvidia-smi` "100% util".** It means "a kernel was resident," not "the GPU is doing useful
  work." Use SOL / MFU / achieved FLOP/s as truth.
- **Averaging instead of percentiles.** The mean hides the p99 that your SLO actually cares about and
  hides the straggler. Always report and reason about the tail.
- **Profiling one rank.** At scale the bug is the *difference between ranks*. A single-rank profile is
  blind to stragglers, imbalance, and collective stalls.
- **Micro-optimizing a non-bottleneck.** Amdahl's law: a 2× speedup of a kernel that's 8% of the time
  is a 4% win. Find the dominant cost first; don't polish the 8%.
- **No reproducible benchmark.** Untagged loadgen, undocumented shapes/weights/versions, no warmup, no
  variance, cold clocks → numbers nobody (including future-you) can reproduce or trust.
- **Pointing Nsight Compute at a whole step.** Its replay makes it enormously slow; target a few named
  kernels. Use Nsight Systems for the timeline.
- **Forgetting to synchronize before stopping the clock.** Times kernel launch, not execution —
  garbage numbers, usually flattering ones.
- **Comparing across availability tiers.** An RDI number vs an Available number is not a fair fight.
- **Assuming tensor cores are on.** Wrong dtype/shape silently falls back to CUDA cores; check the MMA
  pipe utilization, don't assume.

---

## 9. Version awareness

It is 2026 and this stack moves fast. Specifically verify before relying on:

- **Nsight Compute / Nsight Systems** flags, **section names, and metric/counter names** — NVIDIA
  reorganizes sections (SOL, Memory/Compute Workload, Warp State) and renames counters across releases.
  Confirm at `docs.nvidia.com/nsight-compute` and the Nsight Systems docs.
- **`nvidia-smi` / DCGM field names** — throttle-reason bitmask fields, clock/util query fields, and
  DCGM field IDs change; verify exact names.
- **MLPerf** rules — LoadGen tags, reproducibility/source rules, availability-tier definitions, and the
  benchmark/scenario set evolve every round. Verify at the current MLCommons rules.
- **KPerfIR / Proton / IR-embedded profiling** — names, dialect/op names, and arXiv IDs: **verify the
  citation** before quoting.
- **Hardware peaks** — per-dtype FLOP/s and HBM bandwidth differ by GPU generation; use the spec for
  the *exact* part, and the peak for the *math path the kernel uses*.

---

## 10. Canonical references (verify current)

- **Nsight Compute** — `https://docs.nvidia.com/nsight-compute/` (Kernel Profiling Guide: SOL, Memory
  & Compute Workload Analysis, Occupancy, Warp State, roofline).
- **Nsight Systems** — `https://docs.nvidia.com/nsight-systems/` (timeline tracing, NVTX, NCCL trace).
- **CUDA C++ Programming Guide / Best Practices Guide** — `https://docs.nvidia.com/cuda/` (coalescing,
  shared-memory banks, occupancy, tensor cores).
- **CUPTI** — `https://docs.nvidia.com/cupti/` (the profiling API beneath the Nsight tools).
- **MLPerf Inference (datacenter)** — `https://mlcommons.org/benchmarks/inference-datacenter/`.
- **MLPerf rules / policies (LoadGen, reproducibility, availability tiers)** —
  `https://github.com/mlcommons/policies` and the MLPerf Inference repo `https://github.com/mlcommons/inference`.
- **Roofline model** — Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model
  for Multicore Architectures," CACM 2009 (verify the citation if quoting specifics).
- **FlashAttention** — Dao et al.; the canonical arithmetic-intensity / IO-aware kernel example
  (verify the citation / version).
- **KPerfIR / Proton (IR-embedded GPU profiling)** — name the work; **verify the citation** (arXiv ID,
  dialect/op names) before quoting.

---

# GPU Performance Engineering — Worked Examples

Concrete artifacts to imitate. As always: **verify flags, Nsight section/counter names,
`nvidia-smi`/DCGM field names, and any citation against current docs** — the toolchain changes between
releases.

---

## 1. Roofline-driven kernel-triage checklist

Run this top-to-bottom on any "this kernel/job is slow" report. Stop as soon as a step says stop.

### Step 0 — Is the GPU even the bottleneck? (Nsight Systems, all ranks at scale)
```bash
nsys profile -t cuda,nvtx,osrt,nccl -o triage_%p --force-overwrite true python repro.py
nsys stats triage_<pid>.nsys-rep   # cumulative time by kernel, gaps, NCCL ranges
```
- [ ] **GPU timeline has gaps / low GPU-active fraction?** → **host-bound or comms-bound. STOP** — do
      not profile kernels. Go to §3 (host) or the NCCL trace (comms; see `[[ai-networking-collectives]]`).
- [ ] **Multi-rank and all ranks "busy" but slow?** → suspect a **straggler**. Go to §3 below.
- [ ] **GPU busy, one kernel dominates the cumulative table?** → that's your target. Continue.

### Step 1 — Classify the target kernel (Nsight Compute, named, replayed a few times)
```bash
ncu --launch-skip 20 --launch-count 3 \
    --kernel-name regex:".*<kernel>.*" --set full -o k_profile python repro.py
```
Read **Speed-Of-Light (SOL)**:

| Compute SOL | Memory SOL | Verdict | Go to |
|---|---|---|---|
| high | low | **compute-bound** | Step 3 |
| low | high | **memory-bound** | Step 2 |
| low | low | **latency / occupancy-bound** | Step 4 |

Cross-check the built-in **roofline chart**: which ceiling does the achieved point sit under, against
the peak for the *dtype/math path this kernel uses*?

### Step 2 — Memory-bound → raise arithmetic intensity
- [ ] Memory Workload: **sectors-per-request** inflated? → uncoalesced access; fix layout (SoA),
      transpose via shared mem, vectorize (`float4`), align.
- [ ] **Shared-memory bank conflicts** reported? → pad inner dim (`tile[32][33]`) or swizzle.
- [ ] L2 hit rate high but HBM-bound? → there's a **fusion/caching** win (keep tiles in SRAM, recompute
      instead of reload — FlashAttention pattern). See `[[ml-compilers-codegen]]`, `[[inference-optimization]]`.
- [ ] Can a lower-precision dtype halve the bytes without hurting accuracy?

### Step 3 — Compute-bound → use the right pipe
- [ ] Compute Workload: is the **tensor/MMA pipe** actually utilized? If not, you're on CUDA cores —
      fix dtype/shape/alignment so the MMA path is taken, or call a tuned lib (cuBLAS/cuDNN/CUTLASS).
- [ ] Tiling matched to the MMA fragment shape? Raise ILP / better register blocking.
- [ ] Already near the *correct* peak? → you're done; look elsewhere (Amdahl).

### Step 4 — Latency / occupancy-bound → more work in flight
- [ ] Occupancy section: name the **limiter** (registers / shared mem / block size). Register spills to
      **local memory** (HBM traffic)? Reduce register pressure or accept low occupancy + high ILP.
- [ ] Dominant **warp stall** reason? `Long Scoreboard` = global-memory latency (treat as memory-bound);
      `Barrier` = `__syncthreads` imbalance; `Not Selected` = enough work, scheduling only.
- [ ] Many tiny kernels with launch overhead? → batch / **CUDA graphs**.

### Step 5 — Re-measure (reproducible)
- [ ] Warm up, **synchronize before stopping the clock**, report **p50/p90/p99 (not mean)**, quantify
      variance, pin/log clocks + driver/CUDA/lib versions + shapes.
- [ ] Confirm the fix moved the **bottleneck** metric, not just shuffled time. Amdahl: a 2× speedup of
      an 8%-of-time kernel is a 4% win.

---

## 2. Nsight Compute read: memory-bound vs compute-bound

Two kernels, same `--set full` profile, contrasting reads. Numbers are illustrative of the *pattern* —
they are not a benchmark of any specific device; verify section/counter names against your `ncu` version.

### Kernel A — an element-wise / normalization kernel
```
Section: GPU Speed Of Light
  Compute (SM) Throughput .................  9%
  Memory  Throughput ......................  88%   <-- near peak
Section: Memory Workload Analysis
  DRAM (HBM) throughput ...................  near peak
  Sectors/Req (global load) ...............  ~1.1   (coalesced, OK)
Section: Scheduler / Warp State
  Dominant stall ..........................  Stall Long Scoreboard   (waiting on global loads)
```
**Read:** memory-bound, and *already coalesced* — it is simply moving a lot of bytes for little math
(low AI, left of the ridge). Adding compute or occupancy does nothing. **Fix = raise AI:** fuse this op
into its producer/consumer so the data is touched once (eliminate the round-trip to HBM). This is why
fusion (compiler-level — `[[ml-compilers-codegen]]`) is the lever for the entire class of
element-wise/norm/activation kernels, not hand-tuning the kernel.

### Kernel B — a GEMM
```
Section: GPU Speed Of Light
  Compute (SM) Throughput .................  82%   <-- near peak
  Memory  Throughput ......................  31%
Section: Compute Workload Analysis
  Tensor (MMA) pipe utilization ...........  high   <-- tensor cores engaged, good
Section: Occupancy
  Achieved vs theoretical .................  high
```
**Read:** compute-bound and *correctly* on the tensor cores — near the right peak. Little to win here;
move on (Amdahl). **The failure mode to catch:** if instead the MMA pipe utilization were **low** while
compute SOL looked "okay," the kernel would be running matmul on **CUDA cores** (wrong dtype/shape) and
leaving the tensor-core peak — an order of magnitude — on the table. Fix dtype/shape/alignment or call a
tuned library; do **not** conclude "compute-bound, done" from SOL alone without checking the MMA pipe.

**The most-missed case:** *both* SOLs low (e.g. compute 12%, memory 18%). Not memory-bound, not
compute-bound — **latency/occupancy-bound**: too little work in flight. Fix occupancy limiter or launch
more parallelism; the roofline ceilings are irrelevant because you're far under both.

---

## 3. Cross-rank straggler diagnosis: "every rank reads 100%, the job is slow"

**Symptom.** A synchronous data-parallel training job's throughput drops ~30%. Every rank's dashboard
shows `nvidia-smi` GPU-Util ≈ 100%. Per-rank loss/grad-norm look normal. Single-rank Nsight profiles
look fine. Nothing is "obviously" wrong — which is exactly the straggler signature.

**Why the dashboards lie.** In synchronous SGD every rank blocks at the gradient all-reduce until the
slowest rank arrives. While blocked, ranks **spin** in the collective → a kernel is resident → GPU-Util
reads 100%. So a single slow rank makes *all* ranks look 100% busy. The signal is only in the
**differential across ranks**, never in any single rank's "util" number.

**Diagnosis — collect a comparable, time-aligned metric per rank, look at the distribution:**
```bash
# Per-rank, sampled over the slow window (run on each node; verify field names for your driver):
nvidia-smi --query-gpu=index,clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw \
           --format=csv -l 1
```
Plus, from the framework / `nsys` NCCL trace: **per-rank step time** and **per-collective wait time**.

| Signal across ranks | Straggler fingerprint |
|---|---|
| Per-step time distribution | one rank has a fat right tail; others tightly clustered |
| SM clock (`clocks.sm`) | the straggler is conspicuously **lower** (clocked down) |
| Throttle reasons bitmask | straggler shows `hw_thermal_slowdown` / `sw_thermal_slowdown` / `hw_power_brake` |
| Temperature / power | straggler hotter / power-capped |
| Collective wait time | straggler waits **least** (everyone waits on *it*); others wait a lot |
| NCCL trace timing | straggler enters the all-reduce **late**, exits with everyone |

**Confirm the wait topology:** in the `nsys`/NCCL trace, the other ranks sit *inside* the collective
long before the straggler arrives — that asymmetry (others wait long, straggler barely waits) is the
proof it's a straggler and not a global comms problem.

**Fix the root cause, not the symptom.** Here the throttle bitmask points at a **thermal** slowdown on
one GPU: cool/reseat it, check airflow, or drain the bad node and reschedule. Other common straggler
roots: a slow data shard for one rank (host-bound on one node — perf/eBPF on that rank), a degraded
NVLink/PCIe link (rising replay counters), or a noisy neighbor. Then **re-measure the distribution** —
success is the per-rank step-time spread collapsing, not the mean alone improving.

**Takeaway.** This case is the whole argument for **per-rank, always-on, cross-layer** telemetry: a
single-rank profile and a "100% util" dashboard will *both* report health while one throttled GPU drags
the entire cluster. See `[[ai-networking-collectives]]` for the collective/fabric side and
`[[training-frameworks]]` for the synchronous-SGD overlap mechanics.
