# Spec-Driven Development — Template & Worked Example

A fill-in **AI-infra / ML spec (lightweight RFC / design doc)** template, then one fully worked example.
Imitate the structure. Right-size it: drop sections that carry no weight for your change, but never drop
acceptance criteria, cost (for expensive work), or blast radius/rollback (for cluster/prod changes).

Keep it prose and tables, not ceremony. Defer doc-writing craft and alignment to
[[staff-plus-engineering]], the system architecture to [[ml-system-design]], and the metric/eval-harness
design to [[ml-evaluation-evals]].

---

## Template — copy and fill in

```markdown
# Spec: <short title>

- Author: <name>   Date: <date>   Status: Draft | In review | Agreed
- Reviewers: <budget/quota owner for expensive work; cluster/endpoint owner for serving/infra>

## 1. Problem & goal
<One paragraph in user/business terms. What problem, for whom, and what does success enable?
 The "why" from the interview — a goal, not a mechanism.>

## 2. Metric & acceptance criteria  (the heart of the spec)
Primary metric: <ML/business metric — e.g. recall@k, win-rate, conversion, cost/request>
Acceptance criteria (each a pass/fail a script or eval can produce):
- [ ] <metric ≥/≤ threshold> on <dataset> at <operating point>
- [ ] <latency p95 < N ms at Q QPS over a T-minute load test>
- [ ] <cost ≤ $X per 1k requests / per run>

## 3. Scope & non-goals
In scope: <...>
Non-goals (deliberately excluded): <...>

## 4. SLOs / SLAs
- Latency: p50 <...> / p95 <...> / p99 <...>
- Throughput: <QPS / tokens-per-sec / samples-per-sec>
- Availability / error budget: <...>

## 5. Capacity & cost estimate
- Accelerators: <type × count>, <GPU/TPU-hours>, <wall-clock>
- Cost: <$ estimate with stated assumptions and formula>

## 6. Data & dependencies
- Data: <source, size, freshness, licensing / PII / governance>
- Depends on: <services, teams, models/checkpoints, infra, quota pool>

## 7. Failure modes & blast radius
- <failure> → <how it manifests> → <who it affects> (name shared-cluster impact)

## 8. Security, quota & multi-tenancy
- <data access/secrets, tenant isolation, quota consumed & from whose pool, authn/authz, rate limits>

## 9. Rollback / exit plan & kill criteria
- Rollback: <revert / drain / scale-to-zero / restore checkpoint>
- Kill criteria: <stop early if loss diverges / cost > X / eval regresses>

## 10. Open questions
- <unknowns and deferred decisions; route reviewers; mark which block the build>
```

---

## Worked example — serve model X under a latency / QPS / budget target

> **The vague ask that started it:** *"Can you serve model X for the new suggestions feature?"*
>
> **Interview (step 1) surfaced:** *what* = a low-latency suggestion endpoint, not just "run model X";
> *why* = the new compose-box feature needs inline suggestions that feel instant; *for whom* = an
> interactive, latency-sensitive product surface; *constraints* = launch in 3 weeks, must fit existing
> GPU quota, unit economics must work; *success metric* = p95 latency and cost per 1k requests, with
> quality no worse than the current heuristic. Those answers became the spec below.

```markdown
# Spec: Serve model X for compose-box suggestions

- Author: A. Engineer   Date: 2026-06-06   Status: In review
- Reviewers: quota owner (accelerator pool), endpoint owner (suggestions cluster), product lead

## 1. Problem & goal
The new compose-box feature needs inline text suggestions that feel instant. Today a heuristic returns
weak suggestions; model X is materially better in offline tests. Goal: serve model X behind a
low-latency endpoint so the feature can ship, without blowing the unit economics or destabilizing the
shared suggestions cluster.

## 2. Metric & acceptance criteria
Primary metric: suggestion acceptance rate (online); offline proxy = win-rate vs. heuristic on `sugg-eval-v2`.
Acceptance criteria:
- [ ] Quality: win-rate vs. heuristic ≥ 55% on `sugg-eval-v2` (10k prompts); never below parity.
- [ ] Latency: p95 < 200 ms and p99 < 350 ms at 500 QPS over a 10-minute load test, single region.
- [ ] Throughput: sustains 500 QPS steady-state with < 1% timeouts.
- [ ] Cost: ≤ $0.30 per 1k requests at the target QPS (see §5).
- [ ] Availability: meets the endpoint SLO in §4 in a 24-hour soak.

## 3. Scope & non-goals
In scope: a single-region inference endpoint for model X behind the existing suggestions gateway;
autoscaling; load + soak test; cost validation.
Non-goals: multi-region/HA failover (follow-up); streaming token output; fine-tuning model X;
serving any model other than X; changing the gateway API contract.

## 4. SLOs / SLAs
- Latency: p50 < 90 ms / p95 < 200 ms / p99 < 350 ms at 500 QPS.
- Throughput: 500 QPS steady-state, headroom to 650 QPS peak.
- Availability: 99.9% monthly; error budget shared with the suggestions gateway.

## 5. Capacity & cost estimate
- Model X ≈ 8B params; one accelerator per replica at the target batch/latency.
- Measured ~0.18 s compute per request batched; ~120 req/s per replica at p95 budget →
  500 QPS needs ~5 replicas + 1 for headroom/rollout = 6.
- 6 replicas × $2.50/accelerator-hr (assumption — confirm with quota owner) ≈ $360/day ≈ $11k/month.
- At 500 QPS sustained that is ~1.3B requests/month → ≈ $0.0085 per 1k requests at full load; the
  $0.30/1k criterion holds with large margin even at low utilization. Quota: 6 accelerators from the
  shared inference pool (confirm availability — see Open Questions).

## 6. Data & dependencies
- No new training data; model X checkpoint `model-x@2026-05` from the model registry (license cleared
  for this use; no PII in prompts per product review).
- Depends on: suggestions gateway (routing/authn), the shared inference accelerator pool, the serving
  stack (continuous batching), and metrics/observability ([[ml-observability-monitoring]]).

## 7. Failure modes & blast radius
- Replica OOM / crash-loop on long prompts → elevated p99 and timeouts on the compose feature only,
  but consumes pool quota others share → mitigate with input length cap + per-replica concurrency limit.
- Autoscaler over-provisions under a traffic spike → starves co-tenant workloads in the shared pool →
  cap max replicas at 8; alert the quota owner.
- Bad checkpoint promoted → quality drops below parity → caught by the §2 gate in canary before full rollout.
Blast radius: bounded to the suggestions cluster and the shared accelerator pool's headroom.

## 8. Security, quota & multi-tenancy
- Endpoint behind the gateway's existing authn; per-tenant rate limits inherited from the gateway.
- Quota: 6 accelerators (cap 8) from the shared pool; isolation via the pool's existing scheduling/quota
  mechanism. No new secrets; checkpoint pulled with the existing service identity.

## 9. Rollback / exit plan & kill criteria
- Rollback: route 100% back to the heuristic via the gateway flag (instant); scale model X to zero.
- Canary: 5% traffic for 24h before full rollout.
- Kill criteria: roll back if p95 > 250 ms sustained 10 min, error rate > 2%, win-rate < parity, or
  the pool can't supply the replicas without starving co-tenants.

## 10. Open questions
- Is 6 accelerators available in the shared pool for the launch window? (Blocks build — owner: quota owner.)
- Quality bar 55% vs. 60%? Product to confirm before canary. (Does not block initial build.)
- Confirm the $2.50/accelerator-hr rate used in §5.
```

**Why this gates well:** every line of §2 is something a load test or eval prints a yes/no for; §5 turns
"serve model X" into a defended dollar number *before* the deployment; §7–§9 name who an incident hurts
on the shared cluster and exactly how to back out; §10 puts the one build-blocking unknown (quota) in
front of the reviewer who can resolve it — at review-time cost, not build-time cost. Compare to the
original one-line ask: there was no metric, no cost, no blast radius, no exit. That is the difference
between a spec and a guess.
