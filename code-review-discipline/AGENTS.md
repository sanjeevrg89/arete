# AGENTS.md — Code Review Discipline

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`code-review-discipline-guide.md`** next to this file —
> read it before reviewing a diff, requesting review, or approving a merge, and apply it. A concrete
> reviewer checklist (general + AI-infra/IaC/manifest), an author pre-review self-check, and
> good-vs-bad comment examples are in **`examples.md`**. This file is the always-on summary.
>
> **Review is a quality gate, not a formality.** Its job is to catch what tests cannot — design, blast
> radius, security, simplicity — and to spread knowledge. Green CI is necessary, not sufficient.
> Mechanics (PR/MR/CL, approver count, merge buttons) vary by team; the discipline is universal.

## When reviewing a diff (or getting one reviewed), apply these by default:

- **Review catches what tests miss.** Tests check known behavior; review catches design flaws, blast
  radius, security holes, and needless complexity. A change with green CI is not yet a safe change.
- **Use the reviewer's lens in priority order** — don't bikeshed the easy stuff while the risk slips
  by: **correctness & edge cases → blast radius / safety → security → tests present & meaningful →
  simplicity / reuse / readability → observability & rollback.** If you only have time for three, do
  the top three.
- **Blast radius first for infra/config.** A bad manifest, RBAC binding, quota, network policy, or
  resource limit can take down a cluster; a bad training/serving/rollout config can waste huge compute.
  **Diff the rendered manifests, read the `terraform plan`** — not just the template/HCL. For every
  change ask: worst case if wrong, how many workloads/tenants it touches, how fast we can undo it.
  Watch for: `*`/`cluster-admin` RBAC, destroy/replace on stateful resources, removed default-deny,
  missing resource limits, big-bang (uncanaried) rollouts, wrong accelerator/replica counts.
- **Security every time** (not just security PRs): no hardcoded/logged secrets; least privilege; no
  privileged/hostPath/root without justification; nothing wrongly exposed; untrusted input validated;
  new deps/images vetted and pinned. AI-infra threat model: `[[ai-security-on-gke]]`.
- **Tests must exist and be meaningful.** No tests for new logic is **blocking**, not a nit. Verify the
  test would fail if the code were wrong; assertion-free / mock-the-thing-under-test / always-green
  tests are theater. For infra, "tests" = rendered diff, plan, dry-run, policy lint, or staging deploy.
- **Simplicity & reuse.** The simplest change that works, reusing what exists. Push back on speculative
  abstraction (rule of three) and copy-paste. Code-level taste: `[[go-best-practices]]`.
- **Prod changes need observability + rollback.** Can we see it working/failing? Can we undo it fast
  and safely? No rollback story for a prod/infra change is a blocking concern. See
  `[[shipping-and-release]]`.
- **Author self-reviews first:** small PR, description that says **why** (problem, approach, blast
  radius, rollback, how validated), every line self-read, CI green, right reviewers tagged. Don't make
  a human be your linter.
- **Feedback: specific, kind, labeled.** Mark **blocking** vs. **nit:/suggestion:**; explain the
  *why*; suggest the fix; review the code not the person; note what's good. Receiving: assume good
  intent, respond to every comment, push back with technical reasons. Resolve on merit, escalate to a
  tie-breaker if stuck — never on rank or volume.
- **Don't be the danger patterns:** rubber-stamp "LGTM" on a non-trivial diff; nitpicking naming while
  missing the unbounded retry loop; approving a huge unreviewable PR; merging with CI red or tests
  disabled.

## Rationalizations to reject
"LGTM, it's small" (small diffs cause big outages) · "tests pass so it's fine" (review is for what
tests can't see) · "I'll fix it in a follow-up" (blocking is blocking; follow-ups for blockers don't
happen) · "it's just infra/config" (config *is* the production system — review it harder) · "the author
is senior, I trust them" (review is a system property, not a trust score) · "it's urgent, no time"
(the urgent change is the most likely to be wrong).

## Definition of done for a review (the merge gate)
All must hold before approve/merge/Ship — confirm each:
**reviewed against the checklist in priority order** · **CI green** (incl. rendered diff / `terraform
plan` for infra) · **blast radius, security, and rollback considered and acceptable** · **tests present
and meaningful** · **all blocking comments resolved**. An approval that skips any of these is a
liability with your name on it, not an approval.

## When asked to review or to prepare a change for review
Use the reviewer checklist (general + AI-infra/IaC/manifest), the author pre-review self-check, and the
good-vs-bad comment examples in `examples.md`.
