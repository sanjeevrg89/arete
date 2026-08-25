# AGENTS.md — Spec-Driven Development (the Define stage)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative process lives in **`spec-driven-development-guide.md`** next to this file —
> read it before defining work, and apply it. A fill-in spec template and a worked example are in
> **`examples.md`**. This file is the always-on summary.
>
> **Principle: the cheapest place to fix a mistake is the spec; the most expensive is a running job or a
> live cluster.** Building the wrong thing is the costliest failure — doubly so when "building" is a
> multi-thousand-GPU-hour run or a shared-cluster change. Define before you build.

## When an ask is vague or the work is expensive/irreversible, apply by default:

- **Interview before you spec.** Extract and write down: **what** (restate and confirm), **why** (the
  goal, not a mechanism — five-whys until you hit a goal), **for whom** (the consumer), **constraints**
  (budget/quota, deadline, latency, data residency), and the **success metric** (a number/condition, not
  an adjective). If the requester can't answer, that *is* the finding — record it as an open question.
- **Don't silently default load-bearing decisions** (budget, latency target, quality bar). Ask, or state
  the assumption explicitly in the spec's Open Questions where a reviewer can cheaply correct it.
- **Write a lightweight spec/RFC** covering: problem & goal · metric & **testable acceptance criteria** ·
  scope & **non-goals** · **SLOs/SLAs** (p50/p95/p99 latency, QPS/tokens-per-sec throughput,
  availability) · **capacity & cost** (GPU/TPU-hours + $) · data & dependencies (incl. licensing/PII) ·
  **failure modes & blast radius** · security/quota/multi-tenancy · **rollback/exit plan & kill
  criteria** · open questions.
- **Acceptance criteria must be testable.** Each is a yes/no a script or eval can produce, with metric +
  threshold + dataset + operating point. "p95 < 200 ms at 500 QPS; quality ≥ baseline − 0.5% on
  `eval-v3`; ≤ \$0.30 / 1k requests" — not "fast and accurate".
- **Cost is a required section for expensive work.** Estimate GPU/TPU-hours and dollars before launching
  any large run, sweep, or new always-on serving footprint. No estimate = stop.
- **Name the blast radius and rollback** for any shared-cluster / production change. On multi-tenant
  clusters your failure is someone else's outage.
- **Right-size to blast radius.** A config tweak gets a sentence; a serving stack or large run gets a
  full spec. The gate never disappears; the artifact scales.
- **Treat the spec as living.** When reality contradicts it (3x cost, wrong metric), update and
  re-confirm — don't drift silently.

## Hard checkpoint (do not cross without it)
**The spec is reviewed and agreed by a stakeholder before any Plan or Build.** For expensive runs, the
budget/quota owner signs off on cost; for serving/infra changes, the cluster/endpoint owner signs off on
blast radius. A spec nobody read bought you almost nothing.

## Definition of done for the Define stage
All must be true; report honestly if any are not:
written spec exists · acceptance criteria are testable · cost & risk assessed (GPU/TPU-hours + $, blast
radius, rollback, kill criteria) · reviewed/agreed by the right stakeholder · open questions listed, not
hidden. Only then proceed to `[[task-planning-decomposition]]` and `[[ml-system-design]]`.

## Related skills
`[[engineering-lifecycle]]` (umbrella) · `[[task-planning-decomposition]]` (next stage) ·
`[[ml-system-design]]` (architecture) · `[[ml-evaluation-evals]]` (the metric/eval harness) ·
`[[staff-plus-engineering]]` (doc craft & alignment).
