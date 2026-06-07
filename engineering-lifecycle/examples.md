# Engineering Lifecycle — Worked Examples

## End-to-end: adding a new TPU training pipeline for a fine-tuned model

A platform team needs a new pipeline that fine-tunes a base model on a fresh dataset, on a TPU pod
slice, and produces a checkpoint that serving can pick up. This is a **large** task (new pipeline,
expensive compute, shared accelerators), so the full lifecycle applies. Below is each stage, what it
produced, and the gate that let it advance.

### Define → `[[spec-driven-development]]`

What was written down:
- **Problem & scope:** add a reproducible fine-tuning pipeline for model X on dataset Y; out of scope:
  changing the base model, changing the serving stack.
- **Requirements/SLOs:** training throughput ≥ a target step time on the chosen TPU slice; full run
  wall-clock under a stated budget; checkpoints written every N steps to durable storage.
- **Eval / acceptance criteria:** the fine-tuned checkpoint must beat the current production model by
  ≥ X on the agreed held-out eval set, with **no regression** on a guardrail/safety eval. *(This exact
  criterion is reused at the Verify gate — see `[[ml-evaluation-evals]]`.)*
- **Cost & capacity estimate:** ~Z accelerator-hours per full run × expected iterations; peak = one pod
  slice of size S; confirmed quota and region availability; storage and egress for checkpoints + data.
- **Failure modes / blast radius:** divergence late in training (wasted compute), preempting other jobs
  by over-requesting quota, a bad checkpoint silently picked up by serving.

> **GATE passed:** a spec with requirements/SLOs, a measurable eval criterion, a cost estimate, and named
> failure modes — reviewed and agreed by the requesting stakeholder. *Without this, there is no
> definition of "the run worked."*

### Plan → `[[task-planning-decomposition]]`

Decomposed into small, sequenced steps, **risky/expensive first**:
1. **De-risk cheap:** a single-step smoke run on a *minimal* TPU slice that exercises data loading,
   sharding, the checkpoint write, and one eval call — to catch config/topology bugs for cents, not
   thousands of accelerator-hours.
2. A short partial run (a few hundred steps) to confirm loss decreases and throughput hits target.
3. The pipeline scaffolding: data snapshot pinning, deterministic seeding, checkpoint cadence.
4. The full run, behind a flag, writing to a staging checkpoint location.
5. Wiring the eval gate to run automatically on each produced checkpoint.

Each step has an abort-cheaply answer; the most expensive step (the full run) is last and gated by all
the cheap de-risking steps.

> **GATE passed:** ordered step list with the riskiest/most expensive work sequenced first and a cheap
> way to fail early.

### Build → `[[test-driven-development]]`

- Wrote unit tests first for the data-prep transform and the config/sharding logic, then implemented.
- Added an assertion-style smoke check for the checkpoint round-trip (write → read → shape/values).
- Small commits, tree green locally; the smoke run from step 1 was run before anything bigger.

> **GATE passed:** the smoke and partial-run slices do what the plan intended; unit tests pass locally;
> commits are small and reviewable. *(Local confidence only — proving the full run is the next stage.)*

### Verify → `[[verification-and-debugging]]`

- **Integration/e2e:** ran the full pipeline end-to-end on the real TPU slice to a real checkpoint.
- **Eval gate (mandatory):** ran the Define eval — the checkpoint beat production by ≥ X **and** held the
  guardrail eval. (First attempt regressed the guardrail; debugged to a data-filtering bug — **fixed at
  root cause**, not by loosening the threshold — then re-ran.)
- **Reproducibility (mandatory):** pinned the data snapshot, recorded library/runtime/XLA and accelerator
  versions, recorded the seed and full config; reran and got matching metrics within tolerance.

> **GATE passed:** e2e green, **eval gate passed vs the stated criterion and the production baseline**,
> result **reproducible** with a recorded recipe, the one failure fixed at root cause. Evidence (eval
> numbers vs baseline, the repro recipe, logs) attached. *Unit-green alone would NOT have passed this.*

### Review → `[[code-review-discipline]]`

- Independent reviewer checked correctness (sharding, checkpoint cadence, eval wiring), **security/
  multi-tenancy** (least-privilege access to data and storage, no secrets in config/logs), **simplicity**
  (no speculative abstraction), and **blast radius** (quota request sized so it cannot starve other
  tenants; the new checkpoint path is staging, not the live serving path yet).

> **GATE passed:** approving review from a non-author covering correctness, security, simplicity, and
> blast radius; blocking comments resolved.

### Ship → `[[shipping-and-release]]`

- **Progressive & reversible:** the new checkpoint is promoted to serving via a **canary** — small
  traffic percentage / shadow first — with the **quality/eval metric monitored live**, not just latency
  and error rate. Previous model version retained.
- **Tested rollback:** confirmed the prior checkpoint is retained and that reverting serving to it is
  fast; rehearsed the revert before ramping.
- Ramped only after the canary held both infra **and** quality signals; updated the runbook and closed
  the loop with the Define stakeholder.

> **GATE passed:** canary healthy on infra **and** eval/quality signals, rollback proven (not assumed),
> monitoring/alerts live, rollout ramped, runbook updated.

### The loop in action

The guardrail-eval failure in Verify sent the work back to **Build** (fix the data filter), then
re-Verify — *before* the full expensive run was trusted or shipped. That cycle is the lifecycle paying
for itself: a silent quality regression caught at the eval gate instead of in production.

---

## Right-sizing the same discipline to other tasks

The depth scales with blast radius and cost; the **Verify eval gate and Ship rollback never drop**.

| Task | Size | What the lifecycle looks like |
|------|------|-------------------------------|
| Bump a non-breaking library version in a serving image | **Small** | One-line "why + risk" in the PR; smoke test; **eval gate because it can change model behavior**; one reviewer; canary the image, rollback ready. |
| Tune an autoscaling/quota policy on an inference deployment | **Small–Medium** | Short spec w/ the SLO it targets + the risk; smoke in staging; verify it doesn't starve other workloads; one reviewer focused on blast radius; progressive rollout, watch saturation, tested rollback. |
| Add a new data-prep / tokenization stage to an existing pipeline | **Medium** | Written spec w/ eval criteria + cost; risky-first plan (tiny sample first); TDD on the transform; **e2e + eval gate + repro** (does downstream quality hold?); correctness/security review; progressive rollout behind a flag, rollback ready. |
| Roll out a new model-serving version (new weights) | **Medium–Large** | Full spec w/ eval criteria, latency/throughput SLOs, cost; plan w/ shadow eval first; build/verify the serving path; **eval gate vs baseline + repro**; review incl. blast radius; **canary/shadow rollout with live quality monitoring + proven rollback** — the silent-regression case, so the eval gate is the whole point. |
| Upgrade a shared device-plugin / CNI DaemonSet across the cluster | **Large** | Full spec w/ failure modes & blast radius; plan that rolls one node pool first; dry-run/diff in Build; verify on a canary pool incl. a real workload; multi-reviewer incl. security; **node-pool-by-node-pool progressive rollout, bake times, proven rollback, game-day for the eviction risk.** |
| New platform component (controller/operator + CRD) | **Large** | Full ceremony: design review in Define, decomposition w/ de-risking spikes, TDD, e2e + scale test + (eval gate if it touches model behavior) + repro, multi-reviewer + security review, staged rollout w/ runbook and proven rollback. |
| One-line, low-blast-radius config (e.g. a log level, a non-behavioral comment) | **Small** | Paragraph of "why + risk"; smoke test; standard deploy with rollback ready. No eval gate needed if it genuinely cannot affect model behavior — but if in doubt, run it. |

Rule of thumb: the cost of the process should be **smaller than the cost of the failure it prevents** —
right-size **down** for trivial reversible changes and **up** for anything that touches shared infra,
burns serious compute, or can degrade quality silently.
