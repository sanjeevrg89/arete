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
