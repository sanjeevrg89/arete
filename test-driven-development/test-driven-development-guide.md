# Test-Driven Development — the Build stage, tailored to AI infra / ML

This is the deep reference for the **Build** stage of `[[engineering-lifecycle]]`: turning a planned,
decomposed slice of work into correct, tested, committed code. It encodes *how to work*, not a catalog
of testing trivia. The center of gravity is the part most guides skip: **what testing actually looks
like across AI infrastructure and ML**, where the "you can't test this" objection is loudest and most
wrong.

## Overview

TDD is a *design and feedback* discipline, not a coverage ritual. You write a check that describes the
behavior you want, you watch it fail, you make it pass with the least code, then you improve the code
while the check holds the line. The payoff is not the tests themselves — it is that you can change the
system tomorrow and know in seconds whether you broke it. In infra and ML, where the blast radius of a
regression is a stuck cluster, a corrupted checkpoint, or a silent quality drop in production, that
feedback loop is the difference between a system you can evolve and one you are afraid to touch.

The hard part in this domain is that not everything is deterministic. The resolution is a clean split:

- **Deterministic surface** — data transforms, schema/config validation, control logic (reconcile,
  routing, scheduling), parsing/serialization, API contracts, tokenizer round-trips, feature
  computation. This is *most* of any infra/ML system, and it is testable exactly like any other code.
- **Stochastic surface** — the model's generated tokens, sampled outputs, learned behavior. You do not
  assert exact outputs here. You assert **invariants, contracts, properties, and metric thresholds**
  against a fixed eval set with a pinned seed. The eval *is* the test. See `[[ml-evaluation-evals]]`.

Reproducibility is the bridge between the two: by injecting seeds, clocks, and fakes you make as much
of the system deterministic as possible, shrinking the stochastic surface to only what is irreducibly
random — and then gating that with thresholds.

## When to use this

Use TDD whenever you are about to **add or change behavior**:

- A new function, data transform, or feature-engineering step.
- A bug fix — write the test that reproduces the bug *first* (it goes red), then fix it.
- A controller/reconciler, webhook, or scheduling/routing change.
- A training or serving **config** change (configs are code; a wrong dtype or shard count breaks runs).
- An ML behavior change (new prompt, new model, new retrieval) — gate it with an eval threshold.
- A manifest/IaC change — smoke-test it (`kubeconform`, `terraform validate`, dry-run).

You can relax to test-after only for genuine throwaway spikes you will delete. The moment a spike
becomes the implementation, it needs tests before it merges.

## The process (numbered, with the checkpoint)

1. **Start from an acceptance criterion.** Take one slice from `[[task-planning-decomposition]]`. State
   the observable behavior in one sentence: input → expected output / state transition / threshold.
   If you cannot state it, you do not understand the task yet — go back and decompose.
2. **Write the failing test first (RED).** Encode that criterion as an executable test against the
   *public surface*. For stochastic behavior, write it as an eval assertion (metric ≥ threshold on a
   seeded set). Run it. **Confirm it fails for the right reason** — a test that passes before you write
   the code, or fails with a compile error you didn't expect, is testing nothing.
3. **Make it pass with the least code (GREEN).** Write the minimum implementation that turns the test
   green. Resist building the abstraction you "know you'll need" — that is `[[task-planning-
   decomposition]]`'s job, not this loop's. Hard-coding to pass the first test is fine; the next test
   forces the generalization (triangulation).
4. **Run the full fast suite + the race detector.** `go test -race ./...` / `pytest -q` / the project's
   unit target. Green and race-clean, not just the one test you added.
5. **Refactor on green.** Now improve names, remove duplication, extract helpers, tighten the API — with
   the tests as a safety net. Refactor *only* while green; never refactor and add behavior in the same
   step.
6. **Commit small.** Each commit compiles and its tests pass — a bisectable, revertable unit. Commit
   message states the behavior, not the mechanics. Push frequently so CI exercises it.
7. **Repeat** for the next criterion / next case in the table until the slice is complete. Add cases
   for edge conditions, error paths, and the boundaries you found while implementing.
8. **CHECKPOINT (definition of done for Build).** Before handing to `[[verification-and-debugging]]`
   for Verify/Review, confirm: tests exist for *every* new behavior and pass; the suite is race-clean
   and lint-clean; CI is green; stochastic parts are gated by eval thresholds against a seeded set; the
   build is green at HEAD. Only then is the Build stage done.

> The loop is tight on purpose: red → green → refactor → commit, minutes not hours. If a single red
> phase is taking an hour, the slice is too big — decompose further.

## What testing looks like across AI infra / ML

This is the differentiator. Each surface has a canonical way to test it.

### Table-driven unit tests (the deterministic core)

Most infra/ML code is plain logic: a sharding calculator, a config validator, a retry policy, a token
budget, a label selector. Test it with **table-driven tests** — one test function, a slice of named
cases, asserted with a diff. This is the workhorse; see `[[go-best-practices]]` `examples.md` for the
Go idiom (`t.Run`, `t.Parallel`, `cmp.Diff`, helpers that return `error`). The Python equivalent is
`pytest.mark.parametrize`. Cover the happy path, the boundaries (0, 1, max), and every error path.

### Controller / operator tests with `envtest`

You do **not** unit-test a reconciler by mocking the client and asserting "Update was called." That
tests the implementation, not the behavior. Use **`envtest`** (from controller-runtime): it spins up a
real `kube-apiserver` and `etcd` (no kubelet, no scheduler), so you create the CR, run the reconciler,
and assert the cluster *converges* to the desired state — the child objects exist, the status condition
flips, and a second reconcile is a **no-op (idempotent)**. Assert eventual state with `Eventually(...)`,
never a fixed sleep. See `[[kubernetes-controller-expert]]` for the reconcile properties to assert
(level-triggered, idempotent, finalizer cleanup, owner refs / GC). `examples.md` has a sketch.

### Pipeline / integration tests

Data and training pipelines are tested in slices: each stage (ingest → clean → featurize → write) gets
a unit test on its transform, then an integration test runs the wired stages on a tiny fixture dataset
against **real local backends** (a local object store / Postgres in a container), asserting row counts,
schema, and a few known-value records. Run a 2-step "train" on 8 examples to prove the loop executes,
checkpoints, and resumes — not to prove it learns.

### Eval-as-test for ML behavior (the key move)

When the output is stochastic, the test is an **eval**, and you assert **invariants / contracts /
properties / metric thresholds**, never exact tokens:

- **Invariants/contracts:** output is valid JSON; matches the schema; cites only retrieved docs; never
  emits PII; length within bounds; tool calls are well-formed. These are often *deterministic* checks
  on a *stochastic* output and belong in the unit suite.
- **Properties:** metamorphic relations — paraphrasing the input shouldn't flip the label; adding an
  irrelevant sentence shouldn't change the answer; sorting is permutation-invariant.
- **Metric thresholds:** on a fixed, versioned eval set, `accuracy ≥ 0.82`, `faithfulness ≥ 0.9`,
  `p95_latency ≤ 800ms`, `regression vs baseline ≤ 1%`. The CI gate fails the build if the metric drops
  below threshold — this is a regression test for quality. See `[[ml-evaluation-evals]]` for choosing
  metrics, judge bias mitigation, golden sets, and contamination.
- **Determinism for the gate:** pin the seed, set `temperature=0` (or a fixed seed for sampling), pin
  the model/version, and pin the eval set. A flaky eval gate is worse than no gate — tighten the seed
  and widen the threshold band rather than letting it flap.

### Golden / snapshot tests

For complex deterministic outputs — a rendered manifest, a serialized plan, a tokenizer's output,
a generated config — store a reviewed **golden** file and diff against it. Regenerate intentionally
(`-update` flag / `pytest --snapshot-update`) and *review the diff in code review*. Never auto-accept
snapshots blindly; that turns a regression test into a rubber stamp.

### The race detector and determinism

Concurrency bugs are the worst infra bugs — nondeterministic, load-dependent, invisible in light
testing. **Always run the race detector** (`go test -race`, ThreadSanitizer for C++/CUDA host code).
Make tests deterministic by construction:

- **Inject the clock** — pass a `Clock` interface; use a fake clock in tests. Never call `time.Now()`
  or `time.Sleep` directly in code under test.
- **Inject the seed** — thread an explicit RNG/seed through training, sampling, shuffling, data
  augmentation. No global `rand`. Pin `PYTHONHASHSEED`, framework seeds (`torch.manual_seed`,
  `jax.random.PRNGKey`), and `cudnn.deterministic` where reproducibility is required.
- **Inject external deps as fakes** — fake the object store, the queue, the model endpoint.

### Smoke tests for manifests / IaC

Config is code. Smoke-test it in CI: `kubeconform` (or `helm template | kubeconform`) for schema
validity, `kubectl apply --dry-run=server` for admission, `terraform validate` + `plan`, OPA/Conftest
policy checks. A typo in a resource request or a missing `nodeSelector` for a GPU pool is a production
incident; catch it before merge.

## Handling "you can't TDD ML/infra" head-on

The objection comes in flavors; the rebuttal is always "test the deterministic surface, gate the
stochastic one."

- **"Outputs are stochastic, so I can't write an assertion."** You can't assert exact tokens. You can
  assert the output is valid, grounded, PII-free, and that the metric on a seeded eval set clears a
  threshold. Most of the failures you actually ship are deterministic anyway (bad parsing, wrong dtype,
  off-by-one shard, schema drift) — and those are trivially testable.
- **"It's a controller, it needs a real cluster."** `envtest` is a real apiserver + etcd in-process.
  You can drive a full reconcile in a unit-test-speed loop.
- **"Training takes hours; I can't run it in CI."** You don't test that it learns; you test that the
  loop *runs*: tiny model, 8 examples, 2 steps, on CPU — proves wiring, checkpointing, resume,
  determinism. The full training run is validated by the eval gate, not by CI.
- **"It's nondeterministic hardware (GPU/TPU)."** Inject seeds and set deterministic kernels for the
  test path; for the irreducibly nondeterministic parts, assert tolerances (`allclose`) and metric
  thresholds, not bitwise equality.

## Test pyramid for infra/ML — and what to mock vs use real

Shape: a wide base of fast unit tests, a narrower band of integration tests, a thin cap of e2e/smoke.

| Layer | What it covers | Speed | Use real / fake |
|-------|----------------|-------|-----------------|
| Unit | transforms, validators, reconcile logic, contracts, output invariants | ms | pure / injected fakes |
| Integration | controller via envtest, pipeline on fixture data, eval on small set | s–min | real local deps (envtest, containerized store) |
| E2E / smoke | manifest validity, dry-run apply, end-to-end on a tiny job | min | real, in a sandbox |

**Prefer fakes over mocks.** A *fake* is a working in-memory implementation (an in-memory object store,
a fake clock, a fake clientset); it exercises the real contract and catches drift when the interface
changes. A *mock* asserts "method X was called with Y" — it couples the test to the implementation and
passes even when the integration is broken. Use mocks only for true side-effect boundaries you cannot
fake (e.g., "did we call PagerDuty exactly once"). Use **real** dependencies at the integration layer
when a local instance is cheap (envtest, a container). Never put live network calls or the real
production model endpoint in the unit suite.

## CI gates

CI is where the checkpoint is enforced for the whole team. Gates, in order, fail-fast:

1. Format + lint (`gofmt`/`goimports`/`golangci-lint`, `ruff`/`black`).
2. Build / type-check (`go build ./...`, `mypy`/`pyright`).
3. Unit tests **with the race detector** (`go test -race ./...`, `pytest -q`).
4. Integration tests (envtest, pipeline-on-fixture) — may run on a separate, slower lane.
5. Manifest/IaC smoke (`kubeconform`, `terraform validate`).
6. **Eval gate** for ML changes — metrics on the versioned eval set must clear thresholds and not
   regress vs baseline.
7. Coverage as a *signal*, not a target — a number you watch, never a quota you game.

A red CI is a stop-the-line event. Do not merge through a red or flaky gate; fix or quarantine (with a
ticket) the flake.

## Rationalizations & rebuttals

- *"ML/infra can't be tested."* → The deterministic surface (most of it) tests like any code; the
  stochastic surface is gated with eval thresholds on a seeded set.
- *"I'll add tests after it works."* → "After" never comes, and a test written after the code tests
  what the code does, not what it should do. Red-first or it's not a spec.
- *"The happy path works, ship it."* → The incidents live on the error paths and boundaries. An
  untested error path is an outage waiting for the right input.
- *"It's just config."* → A wrong shard count or missing GPU selector takes down a training run.
  Config is code; smoke-test it.
- *"The test is flaky, I'll just retry it."* → A flaky test is a bug (usually a real-clock, real-rand,
  or ordering bug). Fix the determinism; don't normalize retries.
- *"100% coverage means it's tested."* → Coverage measures lines executed, not behaviors verified. You
  can have 100% coverage and zero meaningful assertions.
- *"Tests slow me down."* → They slow the first hour and save the next hundred. The slow path is
  debugging an untested regression in production.

## Red flags (stop and reconsider)

- New or changed behavior merged with **no test** for it.
- A test that passed **before** you wrote the implementation (it asserts nothing).
- Tests asserting **implementation details** (mock call counts, private fields) — they break on every
  refactor and pass when the integration is broken.
- **Flaky tests** retried or `skip`-ped instead of fixed; a quarantine list that only grows.
- **No determinism control:** `time.Now()`, global `rand`, live network, or unpinned model in a test.
- Asserting **exact tokens** from a stochastic model.
- Snapshots/goldens updated without anyone reviewing the diff.
- A **red or flaky CI** that the team merges through.
- Refactoring and adding behavior in the same commit (you can't tell which broke the test).

## Verification gate (definition of done for Build)

Before the work leaves the Build stage for Verify/Review:

- [ ] A test exists for **every** new/changed behavior, written red-first, and now passes.
- [ ] The full unit suite is **race-clean** (`go test -race ./...` / TSan) and **lint-clean**.
- [ ] Integration tests pass where applicable (envtest for controllers, fixture run for pipelines).
- [ ] Stochastic ML behavior is **gated by eval thresholds** on a pinned seed + versioned eval set,
      with no regression vs baseline.
- [ ] Manifests/IaC pass smoke checks (`kubeconform` / `terraform validate` / dry-run).
- [ ] **CI is green** at HEAD; commits are small and bisectable.
- [ ] Determinism is controlled (injected clock/seed/fakes); no flaky tests left enabled.

Meet this and hand off to `[[verification-and-debugging]]`. Fail any line and the Build stage is not
done — fix it before review.

## Version awareness

The ecosystem moves fast (it is 2026). `envtest`'s setup (`setup-envtest`, the apiserver/etcd binary
provisioning) and controller-runtime's testing helpers change across releases — verify the current
controller-runtime docs for the binaries and `Eventually` semantics. Eval tooling (lm-eval-harness,
Inspect, Promptfoo, DeepEval and their CI integrations) and framework determinism knobs
(`torch.use_deterministic_algorithms`, JAX/XLA determinism flags) evolve — confirm against current
docs rather than trusting a remembered flag name. Pin versions in the eval gate so the threshold is
meaningful across runs.

## Canonical references

- Kent Beck, *Test-Driven Development: By Example* — the original red→green→refactor source.
- Go testing: https://pkg.go.dev/testing and https://go.dev/doc/tutorial/add-a-test ; `go-cmp`:
  https://pkg.go.dev/github.com/google/go-cmp/cmp
- controller-runtime `envtest`: https://book.kubebuilder.io/reference/envtest and
  https://pkg.go.dev/sigs.k8s.io/controller-runtime/pkg/envtest
- Ginkgo/Gomega: https://onsi.github.io/ginkgo/ and https://onsi.github.io/gomega/
- pytest: https://docs.pytest.org/ ; Hypothesis (property-based): https://hypothesis.readthedocs.io/
- kubeconform: https://github.com/yannh/kubeconform ; Conftest/OPA: https://www.conftest.dev/
- Eval harnesses: https://github.com/EleutherAI/lm-evaluation-harness ,
  https://inspect.aisi.org.uk/ , https://www.promptfoo.dev/
- See also `[[ml-evaluation-evals]]`, `[[kubernetes-controller-expert]]`, `[[go-best-practices]]`.
