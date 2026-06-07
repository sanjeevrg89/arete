# Verification & Debugging — The Reference

The **Verify** stage of the engineering lifecycle ([[engineering-lifecycle]]): after you have built
something and unit tests are green, *prove it actually works* — and when it doesn't, debug to the
**root cause** systematically rather than guessing. This is the stage where "it ran without erroring"
gets separated from "it is correct," which is uniquely hard in AI/ML infrastructure.

---

## Overview

Two disciplines, one stage:

1. **Verification** — establish, with evidence, that the change meets the spec's acceptance criteria
   in a *realistic* environment. Tests passing is necessary, not sufficient. You verify against
   acceptance criteria, in an environment that resembles production, with reproducible results.
2. **Debugging** — when verification fails (or production breaks), find the *root cause* through a
   repeatable scientific method: reproduce → isolate → hypothesize → instrument → test → fix the
   cause → add a regression test → prevent recurrence.

The bar: **you do not move to Review/Ship until the work is verified in a realistic environment and,
if you fixed a bug, a regression test guards it.** Green unit tests alone never clear this gate.

### Why this stage exists separately from "testing"

Unit tests check the code you wrote against the model in your head. Verification checks the system
against reality: the integration boundaries you stubbed, the data distribution you don't control, the
hardware that behaves differently under load, the concurrency you can't enumerate. In AI/ML infra the
gap between "passes tests" and "works" is enormous because failures are **partial, silent, and
statistical** — see below.

---

## When to use this skill

- After implementing a feature/fix and before opening it for review or shipping.
- Whenever something "should work" but you haven't *run it for real* against acceptance criteria.
- The moment a bug appears — in CI, staging, or production — and you're tempted to retry or patch the
  symptom.
- For any AI-infra change where correctness is statistical or distributed: training runs, inference
  serving, collective comms, GPU kernels, data pipelines, multi-host jobs.
- Use **[[test-driven-development]]** for the *write tests first* discipline; this skill is what you
  do *after* tests are green to prove the thing actually works and to chase down what's wrong.

---

## Part 1 — Verification: beyond unit tests

### Run it for real

A unit test exercises a function in isolation with mocked collaborators. Verification exercises the
**real system** along the path the spec describes:

- **Integration** — real adapters, real serialization, real schema. The bugs hide at the seams you
  stubbed out (timeouts, retries, encoding, version skew between services).
- **End-to-end / staging** — the full request or job path on infrastructure that resembles
  production: same image, same accelerator type, representative data volume, real network. A model
  server that returns 200 in a mock can deadlock on a real GPU under real batching.
- **Read the spec's acceptance criteria and check each one.** The acceptance criteria from the spec
  ([[spec-driven-development]]) are the *definition of done* — walk them one by one and record
  evidence for each. "I think it works" is not evidence; a logged run, a metric, a diff against the
  expected output is.

### Eval gates for ML behavior

For anything that changes model *behavior* (a new model, prompt, fine-tune, retrieval change, decoding
change), unit tests cannot tell you if quality regressed — the output is non-deterministic and
graded, not asserted. **Verification means running the eval suite and gating on it.** Offline evals
gate what is *allowed* to ship; online experiments decide what *does* ship. See
**[[ml-evaluation-evals]]** for how to build representative, contamination-free eval sets, control
LLM-as-a-judge bias, and wire evals into CI. The rule: *no behavior change ships without an eval
delta you've looked at*, sliced by the segments you care about (an aggregate win can hide a
per-segment regression).

### Reproducibility

A result you can't reproduce isn't a result. Verification requires that **the same inputs produce the
same result** (or the same distribution, within a stated tolerance):

- **Seed everything** that's seedable: Python `random`, NumPy, framework RNG (`torch.manual_seed`,
  JAX explicit PRNG keys), data shuffling, dropout. Record the seed in the run metadata.
- **Pin dependencies** — exact versions of frameworks, CUDA/cuDNN, drivers, container base image
  digest (not a floating tag). "Works on my machine" is almost always an unpinned-environment bug.
- **Record the full context** of a verified run: commit SHA, image digest, config, seed, hardware
  (GPU/TPU SKU, topology), dataset version. This is what makes a result auditable and a regression
  bisectable later.
- Accept that bitwise determinism is often impossible on GPU (non-deterministic reductions, atomics,
  autotuning). State the tolerance explicitly and verify the result is *stable within it* across
  repeated runs — don't pretend it's exact, and don't accept "it varies, who knows by how much."

### Load, soak, and SLO verification for infra

For infrastructure, "it works once" ≠ "it works in production":

- **Load test** at and beyond expected peak: throughput (QPS / tokens-per-second / samples-per-sec),
  latency percentiles (p50/p95/p99 — *not the mean*, which hides tail pain), and saturation behavior.
- **Soak test** — run for hours/days to surface leaks, fragmentation, slow drift, accumulating retry
  storms, checkpoint bloat, file-descriptor exhaustion. Many infra bugs are invisible in a 5-minute run.
- **Verify the SLOs the spec named** with numbers: the latency/throughput/error-rate/availability
  targets, measured under realistic load, not asserted from a single happy-path request.
- **Verify failure handling**, not just the happy path: kill a replica, drop a node, inject a slow
  disk, fail a collective — does the system degrade, recover, or corrupt? In distributed systems the
  failure modes *are* the product ([[distributed-systems-fundamentals]]).

---

## Part 2 — Why AI-infra verification is uniquely hard

Hold the conviction: **"it ran without erroring" is not "it's correct."** Five reasons the usual
green-check intuition fails here:

1. **Distributed systems fail partially and silently.** One rank out of 512 is slow, or quietly
   producing NaNs, or stuck — and the job *keeps running*, just wrong or 30% slower. There is no
   exception. A collective that hangs shows up as "training is slow," not an error
   ([[ai-networking-collectives]], [[distributed-systems-fundamentals]]).
2. **ML degrades without errors.** A model that's 8% less accurate, or has lost a capability, or is
   subtly miscalibrated, throws zero exceptions and passes every type check. The only detector is an
   **eval** ([[ml-evaluation-evals]]) and **monitoring/drift detection** in production
   ([[ml-observability-monitoring]]).
3. **GPU/perf problems hide behind green dashboards.** `nvidia-smi` showing **100% utilization** does
   *not* mean the GPU is doing useful work — utilization means "a kernel was resident," which is true
   even when the kernel is stalled on memory, spinning on a barrier, or running a tiny inefficient op.
   You can be "100% utilized" at 20% of achievable FLOP/s. Verify with the right metrics — achieved
   occupancy, memory-bandwidth/compute roofline position, tensor-core utilization, MFU — via a
   profiler, not the green bar ([[gpu-performance-engineering]]).
4. **Nondeterminism.** Concurrency, async collectives, autotuning, floating-point reduction order,
   and data-loader shuffling make bugs intermittent. "I ran it again and it passed" is not a fix —
   it's a Heisenbug you haven't cornered.
5. **Drift.** The system that verified clean last month silently degrades as data, traffic, or
   dependencies move. Verification is not one-time; it's backed by continuous monitoring
   ([[ml-observability-monitoring]]).

The consequence: in AI infra you verify with **statistical and systemic evidence** (evals, percentile
latencies, profiler counters, cross-rank comparisons), not just a passing assertion.

---

## Part 3 — The systematic debugging method

When verification fails, do **not** start changing things. Run the method. Each step has an exit
criterion; don't advance until you meet it.

### 1. Reproduce — make it reliably fail

You cannot fix what you cannot reproduce, and you cannot *prove* a fix without a repro. Get to a
**reliable** reproduction (ideally 100%; if intermittent, characterize the rate):

- Capture the exact conditions: inputs, seed, config, commit, image, hardware, scale, load.
- Shrink the loop: smallest input, fewest ranks, shortest run that still fails. A repro that takes
  4 hours on 256 GPUs is a repro you won't iterate on — get it to minutes on 2 ranks if you can.
- For intermittent bugs, find the knob that changes the failure rate (concurrency, batch size, a
  specific rank/node, a data shard). That knob points at the cause.

**Exit:** you can make it fail on demand (or at a known rate). If you can't, that's the bug to attack
first — *no reliable repro* is itself a red flag, not a license to guess.

### 2. Isolate / localize — bisect and minimize

Cut the search space in half repeatedly:

- **`git bisect`** across commits to find the change that introduced it (have a scripted pass/fail
  test so `git bisect run` does it automatically).
- **Binary-search the stack/layers**: app → framework → collective lib → driver → hardware; or
  request → load balancer → server → model → kernel. Disable/short-circuit halves to localize.
- **Minimize the input** that triggers it (delta-debugging): remove parts of the input until any
  further removal makes it pass. The minimal trigger usually names the cause.
- **Cross-rank / cross-layer differential diagnosis** for distributed/GPU bugs: collect the same
  signal from *every* rank and diff them — the outlier rank (the straggler, the NaN producer, the one
  on the bad NIC/NVLink) is your suspect. Compare a good run vs. bad run, good node vs. bad node, this
  layer vs. that layer. Differences localize; sameness exonerates.

**Exit:** you've narrowed the fault to a specific commit, component, rank, layer, or input region.

### 3. Form a hypothesis

State a *specific, falsifiable* claim about the cause: "Rank 3 is the straggler because its NIC is
falling back to TCP instead of RDMA, so every all-reduce waits on it." A hypothesis you can't test is
a guess. Write it down before you go looking — it stops you from rationalizing whatever you find.

### 4. Instrument — don't guess

This is the load-bearing rule. **Get data; do not speculate-and-patch.** Add observability targeted
at the hypothesis:

- **Logs** with the right context (rank, request ID, shapes, values) at the suspect boundary.
- **Metrics** — the specific counter that would confirm/deny (queue depth, retry count, per-rank step
  time, gradient norm, cache hit rate).
- **Traces** — distributed traces / timeline traces (PyTorch profiler / Nsight / XLA trace) to see
  *where time actually goes* and where a gap/stall is.
- **Profilers** for perf bugs — find the real hotspot or the bubble. Never optimize by intuition; the
  bottleneck is rarely where you think ([[gpu-performance-engineering]]).
- For NaNs/correctness: anomaly detection, checking intermediate tensors, comparing against a
  reference implementation.

Print-debugging is fine; *guess-and-change* is not. Every change you make to "see if it helps" without
a hypothesis and a measurement pollutes the experiment.

### 5. Test the hypothesis

Use the instrumentation to confirm or kill the hypothesis. If the data contradicts it, **discard the
hypothesis and form a new one** — don't bend the data to fit. Often the first hypothesis is wrong;
that's expected, and the measurement just saved you from a wrong fix. Loop back to step 3.

### 6. Fix the root cause — not the symptom

Once the cause is confirmed, fix *that*. Adding a retry around a flaky call, bumping a timeout,
catching-and-ignoring an exception, or restarting the bad rank are **symptom patches** — they hide the
bug and let it return, larger, later. Ask "why did this happen?" until you reach a cause whose fix
prevents the *class* of bug, not this instance. (If you genuinely must mitigate now to stop the
bleeding, do it explicitly, file the root-cause follow-up, and don't call it fixed.)

### 7. Add a regression test

**Before the fix is "done," encode the bug as a test that fails without your fix and passes with it.**
This is non-negotiable — it's how you *prove* the fix works and *guarantee* the bug can't silently
return. For ML, the regression test may be an **eval case** (the failing input added to the golden
set) rather than a unit assert. For perf, it may be a benchmark with a threshold. For distributed, a
fault-injection test. No regression test = the fix is unverified.

### 8. Prevent recurrence

Generalize the lesson: Was this a whole *class* of bug? Add a lint, an invariant check, a CI gate, an
assertion, an alert ([[ml-observability-monitoring]]), or better defaults so the next instance is
caught automatically — ideally at build time, then in CI, then in monitoring. A short, blameless
write-up of cause-and-fix for non-trivial bugs compounds across the team.

---

## The process (numbered, with the gate)

1. **Verify against acceptance criteria** — run the change for real (integration/e2e/staging), walk
   each acceptance criterion from the spec, and record evidence for each.
2. **Run the right realistic checks for the change type** — eval suite for behavior
   ([[ml-evaluation-evals]]); load/soak + SLO measurement for infra; reproducibility (seed, pinned
   deps, recorded context); failure-mode injection for distributed.
3. **If everything passes with evidence → proceed to the gate.** If something fails → enter debugging.
4. **Debug systematically:** reproduce reliably → isolate/localize (bisect, minimize, cross-rank
   diff) → hypothesize → **instrument** → test the hypothesis → fix the **root cause** → **add a
   regression test** → prevent recurrence.
5. **Re-verify** end-to-end after the fix; confirm the regression test fails-without / passes-with.

> **Checkpoint (must hold before Review/Ship):** the change is **verified in a realistic
> environment against the spec's acceptance criteria**, the result is **reproducible**, and **if you
> fixed a bug, a regression test guards it.** Only then move to [[code-review-discipline]] /
> [[shipping-and-release]].

---

## Rationalizations & rebuttals

| Rationalization | Rebuttal |
|---|---|
| "The unit tests pass, so it's done." | Tests check the code against your assumptions; verification checks the system against reality. Run it for real against the acceptance criteria. |
| "It works on my machine." | Then your machine is part of the spec. Pin deps, record the environment, and run it where it actually has to work. Unpinned environment is a bug, not an excuse. |
| "It errored, I'll just retry it / add a retry." | A retry that hides a deterministic fault ships the bug. Reproduce it first; retry only after you know it's a genuinely transient cause. |
| "It only fails sometimes, probably a fluke." | Intermittent = a real bug you haven't cornered. Find the knob that changes the failure rate; that's the cause. "Flaky" is a diagnosis you haven't done yet. |
| "nvidia-smi says 100%, the GPU is busy." | Utilization ≠ useful work. Profile it — you can be 100% "utilized" at a fraction of achievable FLOP/s ([[gpu-performance-engineering]]). |
| "The dashboard is green." | Green dashboards miss partial/silent failures and statistical degradation. Verify with evals, percentiles, and cross-rank diffs. |
| "I'll just bump the timeout / catch the exception." | That's a symptom patch. Ask why it timed out / threw, and fix the cause, or the bug returns bigger. |
| "I'll add the regression test later." | "Later" means the bug can silently return and you never proved the fix. The test *is* the proof — write it now. |
| "I changed a few things and now it works." | You don't know *what* fixed it or *why*, so you can't trust it or prevent recurrence. Isolate the actual cause. |

---

## Red flags (stop and reconsider)

- You're **fixing the symptom** (retry, timeout bump, swallow the error, restart the rank) without
  knowing the cause.
- You have **no reliable repro** but you're already changing code.
- You're **changing things randomly** ("try this, try that") instead of testing a written hypothesis.
- You're **trusting a single metric** (green dashboard, 100% util, mean latency) instead of the metric
  that actually answers the question.
- You **declared it fixed without a regression test**, or "it passed when I re-ran it" is your evidence.
- You verified only the **happy path** — no failure injection, no load/soak, no eval delta.
- You can't say **what environment / seed / commit** produced your verified result.
- Your "verification" was running it **once** and not erroring.

---

## Verification gate (definition of done)

Do not advance to Review/Ship until **all** hold, with evidence you can show:

- [ ] **Acceptance criteria met** — each criterion from the spec walked and evidenced in a *realistic*
      environment (integration/e2e/staging), not just unit tests.
- [ ] **Right checks for the change type run** — eval delta reviewed for behavior changes
      ([[ml-evaluation-evals]]); load/soak run and **SLOs measured** for infra; failure modes injected
      for distributed.
- [ ] **Reproducible** — same inputs → same result within a stated tolerance; seed, pinned deps, and
      run context (commit, image digest, hardware) recorded.
- [ ] **Root cause understood** — for any bug fixed, you can state the cause; the fix addresses the
      cause, not the symptom.
- [ ] **Regression test added** — fails without the fix, passes with it (unit test, eval case, or
      benchmark threshold as appropriate). Recurrence prevention considered (assert/lint/CI/alert).

---

## Version awareness

It is 2026 and the AI-infra toolchain moves fast. Profiler names/flags (Nsight Systems/Compute,
PyTorch profiler, XLA/JAX trace tooling), `nvidia-smi`/DCGM metric semantics, collective-debug env
vars (e.g. NCCL debug levels), and eval-harness APIs change between versions — **verify the exact tool
invocation and metric meaning against current docs** rather than from memory, and pin the versions you
verified against.

## Canonical references

- The Pragmatic Programmer (Hunt & Thomas) — "select isn't broken," reproduce-first, don't assume.
- Brian Kernighan & Rob Pike, *The Practice of Programming*, ch. 5 (Debugging).
- John Ousterhout, *A Philosophy of Software Design* — designing for verifiability.
- `git bisect` docs: https://git-scm.com/docs/git-bisect
- NVIDIA Nsight Systems / Nsight Compute docs: https://developer.nvidia.com/nsight-systems
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- NCCL troubleshooting & env vars: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html

## Related skills

- `[[engineering-lifecycle]]` — where Verify sits in the full Spec → Plan → Build → Verify → Review → Ship loop.
- `[[test-driven-development]]` — writing the tests (and regression tests) this stage relies on.
- `[[ml-evaluation-evals]]` — the eval gates that verify ML *behavior* (the only detector of silent quality loss).
- `[[ml-observability-monitoring]]` — drift, alerts, and the continuous verification that backs this stage in production.
- `[[gpu-performance-engineering]]` — profiling past "100% util" to verify real GPU efficiency.
- `[[distributed-systems-fundamentals]]` — partial/silent failure modes and how to reason about them.
- `[[ai-networking-collectives]]` — diagnosing stragglers, hangs, and slow collectives across ranks.
