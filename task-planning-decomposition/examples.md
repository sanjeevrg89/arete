# Examples — Task Planning & Decomposition

Worked, imitatable artifacts for the **Plan** stage. The first is a full decomposition of a real
AI-infra task into riskiest-first, verifiable steps. The second is a reusable planning-checklist
template you can copy into a design doc or tracking issue.

---

## Example 1 — Decomposing "stand up multi-node training"

**Reviewed spec (input):** Train model *M* (≈7B params) to convergence on dataset *D* across **4 nodes ×
8 GPUs (32 GPUs)** with FSDP-style sharding, restartable across preemption, checkpoints to object
storage, target ≥ X tokens/s/GPU. (The *what/why* came from `[[spec-driven-development]]`; here we plan
the *how*.)

### Anti-pattern: the order most people reach for (horizontal, scary-thing-last)

> 1. Write the full data pipeline.  2. Write the full model + sharding.  3. Write the training loop.
> 4. Add checkpointing.  5. **Launch the 32-GPU run.**  6. (Only now) discover NCCL won't use the RDMA
> NICs / the run OOMs / checkpoints don't restore — after burning hours of 32-GPU time.

Everything risky (multi-node collectives, memory fit at scale, checkpoint *restore*, throughput) is
concentrated in the final, most expensive step. A blocker there invalidates all the work beneath it.

### The plan: vertical slice first, then riskiest/most-expensive next

Each step lists **deps**, a **done-signal** (observable + specific), and **effort / cost / risk**. Note
the order: a thin end-to-end slice proves the path exists, then we de-risk the things that would kill a
32-GPU run *at small scale* before committing the budget.

| # | Step | Deps | Done-signal (how you know it's done & correct) | Eff / Cost / Risk |
|---|------|------|-----------------------------------------------|-------------------|
| 0 | **Spike: cluster + fabric sanity** | — | `nccl-tests` `all_reduce` of 1 GB across 2 nodes completes; measured busbw ≥ 80% of link line rate. Throwaway. | 0.5d / ~2 GPU-hr / **LOW conf — do first** |
| 1 | **Vertical slice: 1-GPU train, 50 steps** | — | `train.py --steps 50` on a tiny config exits 0; loss strictly decreases over 50 steps; 1 checkpoint written to the bucket. | 1d / ~1 GPU-hr / med |
| 2 | **Spike: memory fit at target sharding** | 1 | Single-node 8-GPU FSDP run holds model+optimizer+activations within HBM at the planned shard/precision; no OOM over 100 steps; record peak mem. | 0.5d / ~8 GPU-hr / **LOW conf** |
| 3 | **Multi-node 2-node (16-GPU) run** | 0,1,2 | 16-GPU run completes 200 steps; loss curve matches single-node within tolerance; tokens/s/GPU measured (early read on the throughput target). | 1d / ~30 GPU-hr / **HIGH risk** (real multi-node) |
| 4 | **Checkpoint write + RESTORE verified** | 1,3 | Kill the run mid-step; resume from last checkpoint; loss/step continues from the saved point (not from zero); weights bit-for-bit (or within tolerance) match. Uses `[[ml-checkpointing-orbax]]`. | 1d / ~10 GPU-hr / **HIGH risk** |
| 5 | **Idempotent / preemption-safe launch** | 4 | Job re-submitted after a simulated preemption converges to same state; outputs written to temp path + atomic rename; no duplicate/corrupt checkpoints. See `[[distributed-systems-fundamentals]]`. | 0.5d / ~5 GPU-hr / med |
| 6 | **Scale to 4 nodes (32 GPUs), short run** | 3,4,5 | 32-GPU run holds 500 steps; tokens/s/GPU ≥ X (the spec target); scaling efficiency vs 16-GPU within tolerance; one preempt/resume cycle survives. | 1d / ~60 GPU-hr / med |
| 7 | **Full convergence run** | 6 | Train to target metric on *D*; periodic checkpoints; survives ≥1 real preemption; final eval meets spec. **Most expensive — last on purpose, because everything that could kill it is already proven.** | days / **largest cost** / low (de-risked) |

### Why this order

- **Spikes (0, 2) come first** because they're the lowest-confidence, and their failure would replan the
  whole approach. They're cheap (a few GPU-hours) and throwaway — they buy *information* before the
  budget is committed.
- **The vertical slice (1)** proves an end-to-end path before any component is "finished" — integration
  risk attacked immediately.
- **Multi-node (3) and checkpoint-restore (4) are pulled forward** because they are the classic
  multi-node-job killers. We prove them at 16 GPUs, not by discovering them at 32.
- **The big expensive run (7) is last** not because it's "the end" but because by then every failure mode
  that could waste its budget is already verified at small scale. That's the entire point of
  riskiest/most-expensive-first: **the cheap steps de-risk the expensive one.**

### Partial-failure design baked in

Steps 4 and 5 exist *specifically* so that the multi-day step 7 can survive preemption. "Restore" is its
own verified step (4) — not an assumption — and idempotency (5) means a re-submit after a node failure is
safe. Before step 7 launches, the answer to "what if it dies at 60%?" is written down and tested.

### Replan triggers (decided up front)

- Spike 0 shows busbw far below line rate → fabric/driver problem; replan networking before anything else.
- Spike 2 OOMs at target sharding → revisit sharding/precision/offload in the spec before continuing.
- Step 3 throughput far below X → renegotiate the target or the parallelism strategy *now*, not after the
  full run.

---

## Example 2 — Planning-checklist template

Copy this into the design doc / tracking issue. Fill it in, get the **approach** reviewed, then Build.

```markdown
# Plan: <task name>

Spec: <link to reviewed spec — the what/why>            Owner: <name>    Reviewer: <name>
Status: DRAFT | IN REVIEW | APPROVED | IN PROGRESS | DONE

## Unknowns (flag everything you're not sure of)
- [ ] <unknown 1> → spike? Y/N  (timebox: __)
- [ ] <unknown 2> → spike? Y/N  (timebox: __)

## Steps  (ordered: dependencies, then riskiest/most-expensive first)
| # | Step (vertical slice where possible) | Deps | Done-signal (command+output / metric+threshold / artifact) | Effort | Cost (GPU-hr × $) | Risk (H/M/L) |
|---|--------------------------------------|------|------------------------------------------------------------|--------|-------------------|--------------|
| 0 | spike: <riskiest unknown>            | —    | <what answer ends the spike>                               |        |                   | L conf       |
| 1 | thin end-to-end slice                | —    |                                                            |        |                   |              |
| 2 | de-risk: <expensive thing, small scale> |   |                                                            |        |                   |              |
| … |                                      |      |                                                            |        |                   |              |
| N | full / expensive run (last)          |      |                                                            |        | largest           | L (de-risked)|

## Partial-failure design
- Long/expensive steps: ___ are checkpointed (how/where: ___) and idempotent (how: ___).
- Resume point for the longest step: "if it dies, run ___; re-running is safe because ___."

## Replan triggers (results that would force a revision)
- If <step> shows <result>, then <replan how>.

## Review gate — ALL must be checked before Build
- [ ] Plan is written here (reviewers can see it).
- [ ] Every step is independently verifiable and has a concrete done-signal.
- [ ] Decomposition is vertical; a thin end-to-end path is an early step.
- [ ] Riskiest / most uncertain / most expensive steps are sequenced first.
- [ ] Unknowns flagged; spikes added where uncertainty × cost is high.
- [ ] Effort / cost / risk estimated per step; low-confidence + high-cost de-risked first.
- [ ] Long/expensive steps are checkpointed + idempotent with a defined resume point.
- [ ] Approach reviewed by someone other than the author; concerns resolved or recorded.
```

---

## Smaller example — when NOT to over-plan

**Task:** bump a serving image's base CUDA version and confirm latency unchanged.

This is small and low-uncertainty, so the "plan" is a 30-second mental sequence with done-signals — the
same discipline, compressed:

1. Build image on new base → `docker build` succeeds; image runs `nvidia-smi` showing the new version.
2. Smoke test → existing serving integration test passes against the new image.
3. Latency check → p50/p99 on the canary within ±5% of baseline over a fixed request set.

No spike (you've done CUDA bumps before), no checkpointing (no long-running state), but each step still
has a concrete done-signal and the riskiest check (latency regression) is explicit. Decompose to the
*level the risk warrants* — not more, not less.
