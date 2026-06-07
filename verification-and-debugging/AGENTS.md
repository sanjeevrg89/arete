# AGENTS.md — Verification & Debugging

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`verification-and-debugging-guide.md`** next to this
> file — read it before verifying a change or debugging, and apply it. A worked debugging example,
> a verification checklist, and a reproduce-first template are in **`examples.md`**. This file is the
> always-on summary.
>
> **The one line:** "it ran without erroring" is **not** "it's correct." Tests passing is necessary,
> not sufficient. This is the **Verify** stage of [[engineering-lifecycle]].

## Verifying a change (after unit tests are green, before review/ship)

- **Run it for real** — integration / e2e / staging on infra resembling production. Walk **each
  acceptance criterion from the spec** and record evidence per criterion. Don't ship on unit tests alone.
- **Behavior changes → eval gate.** Any model/prompt/fine-tune/retrieval/decoding change can regress
  quality with zero errors. Run the eval suite, review the delta sliced by segment ([[ml-evaluation-evals]]).
- **Reproducibility.** Same inputs → same result (within a stated tolerance). Seed RNGs, pin
  deps/driver/image-digest, record commit + hardware + config. "Works on my machine" = unpinned-env bug.
- **Infra → at scale.** Load test throughput + **p95/p99 (not the mean)**, soak for leaks/drift,
  **measure the named SLOs**, inject failures (kill a replica/node/collective) — failure modes are the product.
- **Don't trust a single green signal.** Distributed systems fail **partially and silently**; ML
  **degrades without errors**; `nvidia-smi` 100% util ≠ useful work (profile it —
  [[gpu-performance-engineering]]). Verify with evals, percentiles, profiler counters, cross-rank diffs.

## When something is wrong — run the method, do not guess or randomly change things

1. **Reproduce** — make it reliably fail; minimize (smallest input / fewest ranks / shortest run).
   *No reliable repro is the first bug to fix.* Intermittent = uncornered, not "a fluke."
2. **Isolate** — `git bisect` commits; binary-search the stack/layers; minimize the input;
   **cross-rank / cross-layer differential diagnosis** — diff the same signal across ranks/layers, the
   outlier is the suspect ([[distributed-systems-fundamentals]], [[ai-networking-collectives]]).
3. **Hypothesize** — write a specific, falsifiable claim about the cause before you go looking.
4. **Instrument, don't guess** — logs/metrics/traces/profilers targeted at the hypothesis. Get data;
   never speculate-and-patch.
5. **Test the hypothesis** — confirm or kill it with the data; if killed, form a new one (don't bend
   data to fit).
6. **Fix the root cause, not the symptom** — retries / timeout bumps / swallowed exceptions / rank
   restarts hide bugs that return bigger. Ask "why?" until the fix kills the *class*.
7. **Add a regression test** — fails-without, passes-with (unit test, eval case, or benchmark
   threshold). The test is the proof. A fix without one isn't done.
8. **Prevent recurrence** — assert / lint / CI gate / alert ([[ml-observability-monitoring]]).

## Verification gate (definition of done — all must hold, with evidence)

- [ ] Acceptance criteria met in a **realistic environment** (not just unit tests).
- [ ] Right checks run for the change type: **eval delta** (behavior), **load/soak + SLOs** (infra),
      **failure injection** (distributed).
- [ ] **Reproducible** — seed, pinned deps, recorded run context (commit, image digest, hardware).
- [ ] **Root cause understood** for any bug fixed (fix addresses cause, not symptom).
- [ ] **Regression test added** (fails-without / passes-with); recurrence prevention considered.

Only then proceed to [[code-review-discipline]] / [[shipping-and-release]].

## Red flags
Fixing the symptom · no reliable repro but already changing code · changing things randomly · trusting
one metric (green dashboard / 100% util / mean latency) · "it passed when I re-ran it" as evidence · no
regression test after a fix · happy-path-only verification · can't say what env/seed/commit produced
the result.

It is 2026 — verify profiler flags, metric semantics (`nvidia-smi`/DCGM/NCCL), and eval-harness APIs
against current docs, not memory.
