# Verification & Debugging — Worked Examples

Concrete artifacts to imitate: (1) a systematic debugging walk-through of the classic AI-infra
"100% GPU util but slow" silent-straggler bug, (2) a verification checklist mapped back to acceptance
criteria, and (3) a reproduce-first template to fill in before you touch any code.

---

## 1. Worked example — "100% util but training is slow" (silent straggler)

**Symptom (reported):** A 64-GPU (8 nodes × 8) data-parallel training job is hitting ~60% of the
step/s it got last week. No errors. The cluster dashboard is green; `nvidia-smi` shows ~100% GPU
utilization on every device. Someone has already proposed "just restart the job" and "bump the NCCL
timeout." Resist both — neither is a diagnosis.

### Step 1 — Reproduce reliably, then minimize

- Confirm it's not transient: it reproduces on every restart of *this* job on *this* node set. Good —
  reliable repro.
- Minimize: drop from 1000 steps to 50; the slowdown shows in the per-step time within seconds. Cut
  scale to 2 nodes × 8 to see if it persists — it does *only when one specific node is in the set*.
  That's already a localization signal: a knob (node membership) changes the failure.

> If it had been intermittent: find the knob that moves the failure *rate* (a node, a data shard, a
> concurrency level). The knob points at the cause.

### Step 2 — Isolate / cross-rank differential diagnosis

"100% util everywhere" is the trap: utilization means a kernel is *resident*, not doing useful work —
a rank spinning inside an all-reduce barrier waiting for a slow peer reads as 100% utilized.
So stop trusting the green bar and **diff the same signal across ranks.**

- Collect **per-rank step time** and **per-rank time-in-collective**. Result: 56 ranks spend ~5 ms in
  all-reduce; **8 ranks on node-7 spend ~80 ms** — and every other rank is *blocked waiting on them*.
  The slow node is the straggler; everyone else is "100% utilized" doing nothing.
- This is a `git bisect` candidate too — but the per-node differential already localizes it to node-7,
  so chase the node, not the commit.

### Step 3 — Hypothesis (specific, falsifiable)

> "Node-7's GPUs are slow in the collective because its inter-node link has fallen back from the
> high-speed fabric (e.g. RDMA over the fast NIC) to a slow path (TCP / a degraded link), so every
> all-reduce stalls on node-7."

### Step 4 — Instrument, don't guess

- Enable collective debug logging (e.g. `NCCL_DEBUG=INFO`) and read which transport/path each rank
  negotiated — *verify the exact env var and output format against the current NCCL docs.*
- Pull NIC/link counters on node-7 (link speed, RDMA vs TCP fallback, retransmits/CRC errors).
- Capture a profiler timeline (Nsight Systems / PyTorch profiler) on a node-7 rank and a healthy rank
  and compare — look for the all-reduce *gap* (a bubble where the GPU waits), not a compute hotspot.

**Finding:** node-7 negotiated the slow transport / shows a degraded link with retransmits; healthy
nodes are on the fast fabric. The hypothesis holds. (Had the data contradicted it — say all nodes on
the same transport but node-7 doing extra recompute — discard the hypothesis and form a new one.)

### Step 5 — Test the hypothesis

Cordon node-7 / reschedule the job onto a healthy node set with the *same* config and code. Step/s
returns to last week's baseline. Hypothesis confirmed: the bug was a degraded link on one node, not
the model code.

### Step 6 — Fix the root cause, not the symptom

- **Symptom patches to reject:** restart the job (it'll land on node-7 again), bump the NCCL timeout
  (hides the stall, keeps you slow), pin the job off node-7 by hand (doesn't stop the next bad node).
- **Root cause:** a hardware/fabric fault on node-7 that the scheduler happily kept scheduling onto.
  Fix = get node-7 repaired/drained, *and* close the gap that let a degraded-link node serve traffic.

### Step 7 — Add a regression test / detection

You can't unit-test a flaky NIC, so the "regression test" is **automated detection**: a node
health/burn-in check (NCCL all-reduce bandwidth probe + link-speed assertion) that fails a node before
jobs land on it, plus a **per-rank step-time / straggler alert** so the next degraded node is caught
in minutes, not after a week of slow, expensive training ([[ml-observability-monitoring]],
[[ai-networking-collectives]], [[gpu-performance-engineering]]).

### Step 8 — Prevent recurrence

Make the straggler check part of pre-job validation and continuous monitoring; treat "100% util" as
*never* sufficient evidence of throughput — gate on **MFU / achieved throughput**, not the green bar.

**Lesson:** the same method applies to silent NaNs (diff gradient norms across ranks to find the
producer), a slow data loader (profile the input pipeline for the bubble), or a quality regression
(it's an eval delta, not an exception). *Reproduce → cross-rank/cross-layer diff → hypothesis →
instrument → confirm → root-cause fix → detection.*

---

## 2. Verification checklist mapped to acceptance criteria

Fill one of these per change before review/ship. Each acceptance criterion (from the spec —
[[spec-driven-development]]) gets a verification method and recorded evidence. Example for a new
inference endpoint:

| Acceptance criterion (from spec) | Verification method (realistic env) | Evidence recorded |
|---|---|---|
| Returns correct output for valid requests | e2e against staging with real model + real schema | 20/20 golden requests match expected; run log linked |
| p99 latency ≤ 200 ms at 500 QPS | load test at 500 QPS for 15 min on prod-like GPU | p50 / p95 / **p99** numbers (not the mean) |
| Quality ≥ baseline on the eval set | run eval suite, compare to baseline, **sliced** | eval delta table; no per-segment regression ([[ml-evaluation-evals]]) |
| No leaks / stable under sustained load | 6-hour soak at expected load | memory/FD/GPU-mem flat; throughput stable |
| Degrades gracefully when a replica dies | kill a replica during load | error rate + recovery time within SLO |
| Reproducible | re-run with same seed + pinned image | same result within stated tolerance; seed/commit/image-digest recorded |

Rules: **no row is "✅ should work"** — every row has a method run in a realistic environment and
concrete evidence. Percentiles, not means. Behavior rows are eval deltas, not single requests. If any
row fails → enter the debugging method; do not check the gate.

---

## 3. Reproduce-first template (fill in before touching code)

```
## Bug: <one-line symptom — what's actually observed, not the suspected cause>

### Reproduction
- Steps / command to trigger:
- Inputs (exact):                # data, request, batch size, prompt
- Environment:                   # commit SHA, image digest, framework+CUDA/driver versions
- Hardware / scale / topology:   # GPU/TPU SKU, #ranks, #nodes, interconnect
- Seed / config:
- Reliability:                   # 100%? intermittent at __%? which knob changes the rate?
- Minimized repro:               # smallest input / fewest ranks / shortest run that still fails

### Localization
- Bisect result (commit) / suspect component / layer:
- Cross-rank or cross-layer diff: <signal compared, which rank/layer is the outlier>

### Hypothesis (specific, falsifiable)
- "<X happens because Y, which would show up as Z in the data>"

### Instrumentation (data, not guesses)
- Logs / metrics / traces / profiler added at: <boundary>
- What the data showed:

### Verdict
- Hypothesis confirmed / rejected (if rejected, next hypothesis):

### Root cause
- The actual cause (not the symptom):

### Fix
- What changed and why it addresses the cause:

### Regression test
- Test/eval-case/benchmark that fails-without and passes-with the fix: <path/name>

### Prevent recurrence
- Assert / lint / CI gate / alert / better default added:
```

Anti-pattern this template kills: opening the editor and changing things before you can make the bug
fail on demand. If the "Reproduction → Reliability" line isn't filled in, you are guessing, and your
"fix" is unverifiable.
