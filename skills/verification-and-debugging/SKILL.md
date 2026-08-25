---
name: verification-and-debugging
description: The Verify stage of the engineering lifecycle — prove a change actually works beyond unit tests, and when it doesn't, debug to root cause systematically. Use after building a feature/fix and before review/ship, or whenever something "should work" but you haven't run it for real, or the moment a bug appears in CI/staging/production. Covers verification beyond tests (integration/e2e/staging, checking against the spec's acceptance criteria, eval gates for ML behavior, reproducibility via seeds and pinned deps, load/soak and SLO verification), why AI-infra verification is uniquely hard (partial/silent distributed failures, ML degrading without errors, GPU "100% util but slow", nondeterminism, drift — "it ran without erroring" ≠ "it's correct"), and the systematic debugging method (reproduce → isolate/bisect/minimize → hypothesize → instrument-don't-guess → test → fix root cause → add regression test → prevent recurrence), including cross-rank/cross-layer differential diagnosis for distributed/GPU bugs. Scope: distributed/ML/infra systems; for single-process application bugs a dedicated diagnosis skill (e.g. the vendored mattpocock `diagnosing-bugs`) may fit better.
---

# Verification & Debugging

Apply the judgment of an engineer who has shipped AI/ML infrastructure to production for years and
holds one line as sacred: **"it ran without erroring" is not "it's correct."** Unit tests check your
code against your assumptions; this stage proves the *system* works against reality — and when it
doesn't, finds the **root cause** by method, not by guessing.

## How to use this skill

1. **Read `verification-and-debugging-guide.md`** in this directory — the full reference (verification
   beyond tests, why AI-infra verification is hard, the systematic debugging method, the process and
   the gate). Apply it to the task at hand.
2. For a worked example — a "100% util but slow" silent-straggler debug walked through the method, a
   verification checklist mapped back to acceptance criteria, and a reproduce-first template — read
   **`examples.md`**.
3. Match the surrounding stack's tooling and conventions; apply the verification gate and the
   debugging method regardless. It is 2026 — verify profiler flags, metric semantics, and harness
   APIs against current docs, not memory.

## The essentials (full rationale in `verification-and-debugging-guide.md`)

- **Tests passing is necessary, not sufficient.** Verify the change *for real* — integration / e2e /
  staging on infra that resembles production — against each of the **spec's acceptance criteria**,
  with recorded evidence per criterion.
- **Behavior changes need eval gates.** Any model/prompt/fine-tune/retrieval/decoding change can
  regress quality with zero errors; run the eval suite and review the delta, sliced
  ([[ml-evaluation-evals]]). No behavior change ships without an eval delta you've looked at.
- **Reproducibility is part of correctness.** Same inputs → same result (within a stated tolerance):
  seed everything, pin deps/driver/image-digest, record commit + hardware + config. "Works on my
  machine" is an unpinned-environment bug.
- **For infra, verify at scale:** load test (throughput + p95/p99, not the mean), soak test for
  leaks/drift, **measure the named SLOs**, and inject failures — the failure modes are the product
  ([[distributed-systems-fundamentals]]).
- **AI-infra verification is uniquely hard:** distributed systems fail **partially and silently**; ML
  **degrades without errors**; GPU/perf problems **hide behind green dashboards** — `nvidia-smi` 100%
  util ≠ useful work; you can be "100% utilized" at a fraction of achievable FLOP/s, so profile it
  ([[gpu-performance-engineering]]). Plus nondeterminism and drift ([[ml-observability-monitoring]]).
- **Verify with statistical/systemic evidence** — evals, latency percentiles, profiler counters,
  cross-rank diffs — not a single green check or mean metric.
- **When it fails, run the method — don't change things randomly.** Reproduce reliably → isolate
  (bisect commits, minimize input, binary-search the stack) → form a falsifiable hypothesis →
  **instrument, don't guess** (logs/metrics/traces/profilers) → test it → fix the **root cause** →
  add a regression test → prevent recurrence.
- **No reliable repro = attack that first.** You can't fix, or prove a fix for, a bug you can't make
  fail on demand. Intermittent means uncornered, not "a fluke."
- **Cross-rank / cross-layer differential diagnosis** for distributed & GPU bugs: collect the same
  signal from every rank/layer and diff — the outlier (straggler, NaN producer, bad NIC/NVLink) is
  the suspect ([[ai-networking-collectives]]).
- **Fix the cause, not the symptom.** Retries, timeout bumps, swallowed exceptions, and rank restarts
  hide bugs that return larger. Ask "why?" until the fix kills the *class* of bug.
- **A fix isn't done without a regression test** that fails-without and passes-with it (unit test,
  eval case, or benchmark threshold). The test is the proof. Then prevent recurrence with an
  assert/lint/CI gate/alert.
- **The gate:** acceptance criteria met in a realistic env, result reproducible, root cause
  understood, regression test added — *then* go to [[code-review-discipline]] / [[shipping-and-release]].

## Related skills

- `[[engineering-lifecycle]]` — where Verify sits in Spec → Plan → Build → Verify → Review → Ship.
- `[[test-driven-development]]` — writing the tests and regression tests this stage relies on.
- `[[ml-evaluation-evals]]` — eval gates that verify ML *behavior*, the only detector of silent quality loss.
- `[[ml-observability-monitoring]]` — drift, alerts, and continuous verification that back this stage in prod.
- `[[gpu-performance-engineering]]` — profiling past "100% util" to verify real GPU efficiency.
- `[[distributed-systems-fundamentals]]` — partial/silent failure modes and how to reason about them.
- `[[ai-networking-collectives]]` — diagnosing stragglers, hangs, and slow collectives across ranks.
