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
