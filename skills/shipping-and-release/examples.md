# Shipping & Release — worked examples

Concrete artifacts to imitate: a progressive model-rollout plan, a pre-ship readiness checklist, and a
rollback-runbook template. Adapt the names, numbers, and thresholds to your system — the *structure* and
the *gates* are the reusable part.

---

## 1. Progressive model-rollout plan (shadow → canary → ramp → 100%)

Scenario: promoting a new LLM serving the production `assistant` endpoint. The current production model
is the **champion**; the new one is the **challenger**. Traffic split via the inference gateway
([[gke-inference-gateway]]); signals via [[ml-observability-monitoring]].

### Baseline & criteria (agreed BEFORE any traffic)

| Metric | Source | Champion baseline | Challenger must... |
|---|---|---|---|
| Error / 5xx rate | gateway + serving | 0.2% | ≤ 0.3% |
| TTFT p95 | serving | 850 ms | ≤ 900 ms (no >10% regression) |
| Inter-token latency p95 | serving | 35 ms/token | ≤ 40 ms/token |
| Online quality (LLM-as-judge win rate) | eval pipeline | n/a (50/50 ref) | ≥ 50% vs champion |
| Task success / accept rate | product telemetry | 71% | ≥ 71% (no regression) |
| Cost / 1k requests | billing/token meter | $X | ≤ 1.15 × $X |

**Global abort criteria (roll back immediately at any step):** error rate > 1% for 5 min · TTFT p95 >
1.5 × baseline for 10 min · quality win rate < 45% over the bake window · any pod crashloop/OOM · cost
> 1.5 × baseline.

### Stages

| Stage | Traffic to challenger | Bake | Watch | Gate to proceed |
|---|---|---|---|---|
| **0. Shadow** | 0% served; 100% *mirrored* (responses discarded) | 24 h | TTFT/ITL, errors, cost, offline quality on logged outputs | No latency/cost regression; no gross quality failures; no crashes |
| **1. Canary** | 1% | 2–4 h | all metrics on *served* traffic + global abort criteria | All within thresholds; quality ≥ 50% |
| **2. Ramp** | 5% | 4 h | same | within thresholds |
| **3. Ramp** | 25% | 8 h (overnight ok with on-call) | same + slice/segment quality | within thresholds; no bad segment |
| **4. Ramp** | 50% | 8–24 h | same | within thresholds |
| **5. Full** | 100% | — | continue monitoring 48 h | post-deploy verification passes |

### Rules

- **Champion stays warm** through stage 5; rollback = route 100% back to champion (seconds), not a
  redeploy.
- **Bake long enough for the quality signal**, which is noisier and slower than latency. Don't ramp
  faster than your quality read arrives.
- **Don't skip stages** — a regression can be sublinear in traffic; segment-level quality issues only
  surface with volume (stage 3+).
- **Decommission the champion only after 100% has baked** (e.g., 48 h clean), and remove the split
  config last. Keep the rollback path until then.

---

## 2. Pre-ship readiness checklist (the go/no-go gate)

Fill this in before ramping. **Any unchecked box = NO-GO.** Don't ship around a gap — surface it.

```
RELEASE: <what is shipping, version/commit/model-id>
OWNER:   <name>            DATE/WINDOW: <when>            ON-CALL: <name, acknowledged? Y/N>

BLAST RADIUS
[ ] Who/what breaks if this is wrong, and how bad:  ____________________________
[ ] First step exposes a small, recoverable slice:  ____________________________

DELIVERY PLAN
[ ] Pattern chosen (canary / blue-green / shadow / flag): __________
[ ] Ramp schedule (% steps + bake time):  ____________________________
[ ] Acceptance criteria (numbers, what "success in prod" means): ______________
[ ] Abort criteria (numbers, what triggers rollback):  ________________________

REVERSIBILITY  (the heart of the gate)
[ ] Rollback procedure written (exact steps/commands):  see runbook
[ ] Rollback TESTED — executed in staging/pre-prod, not just written.  RTO observed: ____ min
[ ] Kill switch / flag-off path exists and was exercised
[ ] Data migration is reversible (expand/contract; no destructive step in this release)

OBSERVABILITY  (must be LIVE before the ramp)
[ ] Dashboards for errors, latency/TTFT, eval/quality, cost — live now
[ ] Alerts wired to the abort criteria — firing now (tested)
[ ] SLOs / error budget known                          (see [[ml-observability-monitoring]])

CAPACITY
[ ] Headroom for the new version confirmed (incl. double footprint if blue-green)
[ ] GPU/TPU/accelerator quota confirmed                (see [[autoscaling-kubernetes]], [[gke-master]])
[ ] Autoscaling limits won't cap the ramp

PROCESS
[ ] Runbook written and linked
[ ] On-call aware and acknowledged
[ ] Not a Friday-evening / unattended risky ship (or on-call is staffed for it)
[ ] One change only — not bundled with unrelated config/infra changes

GO / NO-GO:  ______   (all boxes checked → GO)
```

---

## 3. Rollback runbook template

One page per release. Written in step 1, *tested* in step 2, used in step 6.

```
# Runbook: <release name / model-id / version>

## What this change is
<1–3 sentences: what shipped, why, what behavior changed.>

## How to watch it (during ramp)
- Dashboard:   <link>
- Key signals & thresholds:
    errors    > 1% / 5min           -> ABORT
    TTFT p95  > 1.5x baseline /10min -> ABORT
    quality   win-rate < 45%         -> ABORT
    cost      > 1.5x baseline        -> ABORT
- Alerts: <links to the alerts wired to the above>

## HOW TO ROLL BACK  (do this FIRST, debug later)
Target RTO: <N> minutes.  (Verified on: <date> in <env>.)

Option A — flag/route (fastest, preferred):
  1. <command to flip flag off / route traffic 100% back to champion-or-blue>
     e.g.  kubectl ... set the traffic split to champion=100, challenger=0
  2. Confirm: <command/dashboard showing traffic on the good version>

Option B — redeploy last-known-good (if A unavailable):
  1. <git revert <sha> && sync   |   deploy <previous-version>>
  2. Confirm healthy: <command/check>

After rollback:
  [ ] Confirm error/latency/quality back to baseline on the dashboard
  [ ] Post in <incident channel>; page <owner> if not already
  [ ] DO NOT re-ramp until root cause is found and fixed

## Data / migration notes
- Reversible? <yes/no + how>.  If a backfill ran: <how to reconcile on rollback>.

## Escalation
- Primary on-call: <name/rotation>   Secondary: <name>
- Owning team / channel: <link>

## After the incident
- [ ] Blameless review scheduled: timeline, what monitoring missed, what slowed rollback
- [ ] Action items filed; readiness-gate updated so this class of failure can't recur silently
```

---

### Anti-pattern (do NOT imitate)

```
# "Ship it" — the big-bang
git push prod main        # 0 -> 100% of traffic, all at once
# no canary, no shadow, no flag
# monitoring: "we'll add it next sprint"
# rollback: "just git revert if something breaks" (never tested)
# migration: ALTER TABLE ... DROP COLUMN  (in the same deploy as the new code)
# 5:40pm Friday, on-call not told
```

Every line above is a red flag: big-bang, no progressive delivery, no monitoring before ramp, untested
rollback, irreversible migration coupled to the deploy, and an unattended Friday ship. This is how a
"small change" becomes a weekend outage.
