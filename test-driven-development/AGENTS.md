# AGENTS.md — Test-Driven Development (the Build stage)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative process lives in **`test-driven-development-guide.md`** next to this file —
> read it before implementing. Worked examples (red-green-refactor, eval-as-test, envtest sketch) are
> in **`examples.md`**. This file is the always-on summary.
>
> TDD is the **Build** stage of `[[engineering-lifecycle]]`: it comes after `[[task-planning-
> decomposition]]` and before `[[verification-and-debugging]]`.

## When about to add or change behavior, work test-first:

- **Red → green → refactor, in that order.** Write the failing test (or eval assertion) *first*, watch
  it fail for the right reason, write the least code to pass, then refactor on green. Never refactor and
  add behavior in the same step.
- **Bug fix = reproduce first.** Write the test that fails because of the bug, then fix it.
- **Test behavior through the public surface, not implementation.** No asserting mock call counts or
  private fields — those break on refactor and pass when the integration is broken.
- **Deterministic surface is always testable:** data transforms, config/schema validation, reconcile/
  routing/scheduling logic, parsing/serialization, contracts, output invariants. Table-driven tests
  (`[[go-best-practices]]`), `pytest.mark.parametrize`.
- **Stochastic ML behavior is gated, not asserted exactly.** Test invariants/contracts (valid JSON,
  grounded, no PII, length bounds), properties (metamorphic), and **metric thresholds** on a fixed,
  versioned eval set with a pinned seed and `temperature=0`. Never assert exact tokens. See
  `[[ml-evaluation-evals]]`.
- **Controllers/operators → `envtest`** (real apiserver + etcd, no kubelet): assert the reconcile
  *converges* and is **idempotent** (second reconcile is a no-op), use `Eventually(...)` not sleeps.
  See `[[kubernetes-controller-expert]]`.
- **Determinism is engineered:** inject the clock, the seed, and external deps as **fakes**. No
  `time.Now()`, no global `rand`, no live network in unit tests. Pin framework seeds + deterministic
  kernels where reproducibility matters.
- **Prefer fakes over mocks** — a fake exercises the real contract and catches drift; a mock couples
  the test to the implementation. Use real local deps (envtest, containerized store) at integration.
- **Test pyramid:** wide base of fast unit tests, narrower integration band, thin e2e/smoke cap.
- **Run the race detector** (`go test -race`, TSan). Concurrency tests must be deterministic.
- **Smoke-test manifests/IaC:** `kubeconform` / `helm template | kubeconform`, `kubectl --dry-run`,
  `terraform validate` + `plan`, Conftest. Config is code.
- **Small commits, build stays green.** Each commit compiles and its tests pass; bisectable history.
- **A flaky test is a bug** (real clock / real rand / ordering). Fix it; don't normalize retries.
- **Coverage is a signal, not a target.** Don't game it; meaningful assertions over executed lines.

## Checkpoint — definition of done for the Build stage
All must hold before handing to Verify/Review; report honestly if any fail:
- A test exists for every new/changed behavior, written red-first, now passing.
- Unit suite **race-clean** and **lint-clean**; integration tests pass (envtest / fixture run).
- Stochastic parts **gated by eval thresholds** on a pinned seed + versioned set, no regression.
- Manifests/IaC pass smoke checks.
- **CI green** at HEAD; commits small and bisectable; determinism controlled; no enabled flaky tests.

Then proceed to `[[verification-and-debugging]]`.
