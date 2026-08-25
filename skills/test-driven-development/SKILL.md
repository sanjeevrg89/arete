---
name: test-driven-development
description: >
  The Build stage of the engineering lifecycle done right — implement features and fixes test-first
  using red → green → refactor, keeping the build green with small commits. Use whenever you are about
  to write or change behavior: a new function, a bug fix, a controller/reconciler, a data transform,
  a training/serving config, or an ML behavior change. Tailored to AI infra / ML where people claim
  "you can't TDD this": shows what testing actually looks like — table-driven unit tests, controller
  tests with envtest, pipeline/integration tests, eval-as-test for stochastic ML behavior (invariants,
  contracts, properties, metric thresholds — not exact tokens), golden/snapshot tests, the race
  detector, determinism via injected seeds/clocks/fakes, and smoke tests for manifests/IaC. Covers the
  test pyramid, fakes-over-mocks, CI gates, and the checkpoint (tests written, passing, race-clean)
  before Verify/Review. Scope: AI-infra/ML work — for general application-level TDD loops prefer a
  generic TDD skill (e.g. the vendored mattpocock `tdd`) when installed.
---

# Test-Driven Development (the Build stage)

Implement the way an engineer who has been paged at 3am for an untested code path implements: **write
the executable check first, make it pass, then clean up — and never let the build go red.** A test is
not paperwork you add after the demo works; it is the specification of the behavior, written in code,
that lets you change the system later without fear. This is the **Build** stage of
`[[engineering-lifecycle]]` — it sits between planning the work and verifying/reviewing it.

The reflex this skill kills: "TDD doesn't apply to ML/infra because outputs are stochastic / it's a
controller / it's just YAML." It does apply. You test the **deterministic surface** (data transforms,
configs, control logic, contracts) directly, and you gate the **stochastic surface** (model behavior)
with **eval thresholds**. Reproducibility is a feature you build in, not an accident you hope for.

## How to use this skill

1. Read `test-driven-development-guide.md` in this directory — the full process and the AI-infra/ML
   testing playbook. Apply it to the task at hand.
2. For worked examples — a full red-green-refactor cycle, an eval-as-test with a seeded deterministic
   harness and a metric threshold, and a controller `envtest` sketch — read `examples.md`.
3. Match the surrounding repo's test framework and conventions (Go `testing` + table tests, `pytest`,
   Ginkgo/envtest, etc.); apply the red→green→refactor discipline and the determinism rules regardless.

## Essentials (full detail in `test-driven-development-guide.md`)

- **Red → green → refactor, in that order.** Write a failing test (or executable acceptance criterion)
  *first*, watch it fail for the right reason, write the least code to pass, then refactor on green.
- **Test behavior through the public surface, not implementation.** A test that breaks on every
  refactor is testing the wrong thing. Assert contracts and observable outputs.
- **The deterministic surface is always testable.** Data transforms, schema/config validation,
  reconcile logic, routing, parsing, serialization, contracts — unit-test these exhaustively.
- **Stochastic ML behavior is gated, not asserted exactly.** Test invariants, properties, contracts,
  and **metric thresholds** against a fixed eval set with a pinned seed — never exact tokens. See
  `[[ml-evaluation-evals]]`.
- **Determinism is engineered:** inject the seed, the clock, and external deps (fakes). No real
  wall-clock, no global `rand`, no live network in a unit test. Run with the **race detector**.
- **Test pyramid:** many fast unit tests, fewer integration tests (envtest, pipeline, real local
  deps), a thin layer of e2e/smoke. Prefer **fakes over mocks** — fakes catch contract drift.
- **Controllers/operators get `envtest`** (real apiserver + etcd, no kubelet): assert the reconcile
  *converges* and is idempotent, not that a function was called. See `[[kubernetes-controller-expert]]`.
- **Go tests are table-driven, race-clean, flake-free** — see `[[go-best-practices]]` for the idiom.
- **Small commits, build stays green.** Each commit compiles and its tests pass. Bisectable history.
- **Smoke-test manifests/IaC** (`kubeconform`, `helm template | kubeconform`, `terraform validate`,
  dry-run) — config is code and breaks prod just as hard.
- **Checkpoint before Verify/Review:** tests exist for the new behavior and pass, race/lint clean, CI
  green, stochastic parts gated by eval thresholds. Then hand off to `[[verification-and-debugging]]`.

## Related skills

- `[[engineering-lifecycle]]` — the umbrella; TDD is the Build stage within it.
- `[[task-planning-decomposition]]` — comes first; produces the slices/acceptance criteria you turn
  into red tests here.
- `[[verification-and-debugging]]` — comes after green; manual/system verification and debugging when
  a test (or prod) goes red.
- `[[go-best-practices]]` — table-driven tests, `go-cmp`, `-race`, fakes; the Go testing idiom.
- `[[kubernetes-controller-expert]]` — `envtest`/Ginkgo controller tests; testing reconcile loops.
- `[[ml-evaluation-evals]]` — evals as the test for stochastic model behavior; metrics, thresholds,
  golden sets, regression gates in CI.
