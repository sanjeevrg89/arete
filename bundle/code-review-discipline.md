---
name: code-review-discipline
description: How to review code and get code reviewed effectively — the Review stage of the engineering
  lifecycle, tailored to AI infra / ML platforms where the blast radius is large. Use when reviewing a
  PR/MR/diff/changelist, requesting review, responding to review comments, or deciding whether a change
  is safe to approve and merge — especially for Kubernetes manifests, Helm charts, Terraform/IaC, RBAC,
  network policy, resource limits/quotas, and training/serving/rollout configs where a bad change can
  take down a cluster or waste large amounts of compute. Covers the reviewer's lens (correctness, blast
  radius/safety, security, tests, simplicity, observability/rollback), the author's pre-review
  self-check, giving and receiving feedback well, blocking-vs-nit, and the approval + CI-green gate
  before merge.
---

# Code Review Discipline

Apply the judgment of an engineer who has reviewed and shipped infrastructure where a single bad diff
could take down a cluster or burn a six-figure compute bill. **Review is a quality gate, not a
formality.** Its job is to catch what tests cannot — design, blast radius, security, simplicity — and
to spread knowledge so no change has exactly one person who understands it.

## How to use this skill

1. **Read `code-review-discipline-guide.md`** in this directory — the full reference: why review is a
   gate, the reviewer's priority-ordered lens, the author's self-check, how to give/receive feedback,
   the numbered process with its merge gate, plus rationalizations, red flags, and the verification
   gate. Apply it to the review at hand.
2. For a concrete reviewer checklist (general + AI-infra/IaC/manifest-specific), an author pre-review
   self-check, and good-vs-bad review-comment examples to imitate, read **`examples.md`**.
3. Match the team's existing review norms (PR template, approval count, merge mechanics); apply the
   correctness, blast-radius, and security rules regardless of local culture.

## The essentials (full rationale in `code-review-discipline-guide.md`)

- **Review catches what tests miss.** Tests check known behavior; review catches design flaws, blast
  radius, security holes, and needless complexity. A green CI run is necessary, not sufficient.
- **Review in priority order — don't bikeshed the easy stuff.** Correctness & edge cases → **blast
  radius / safety** → security → tests present & meaningful → simplicity/reuse/readability →
  observability & rollback. Spend your attention where the risk is.
- **Blast radius first for infra/config.** A bad manifest, RBAC binding, quota, network policy, or
  resource limit can take down a whole cluster; a bad training/serving/rollout config can waste huge
  compute. Review IaC, Kubernetes manifests, Helm, and Terraform with extra care — diff the rendered
  output, ask "what is the worst case if this is wrong, and how fast can we undo it?"
- **Security is in scope every time.** Secrets, privilege, exposure, supply chain, untrusted input.
  See `[[ai-security-on-gke]]` for the AI-infra threat model.
- **Tests must exist and mean something.** No tests for new logic is a blocking issue; assertion-free
  or always-green tests are theater. Verify the test would fail if the code were wrong.
- **Demand simplicity and reuse.** The simplest change that works, reusing what exists. Push back on
  speculative abstraction and copy-paste. Code-level taste lives in `[[go-best-practices]]`.
- **Prod changes need observability and a rollback path.** Can we see it working? Can we undo it fast
  if it isn't? No rollback story is a blocking concern for anything that touches production.
- **Authors do the first review.** Small PRs, a description that says *why* (not just *what*), a
  self-reviewed diff, and green CI before requesting a human. Respect the reviewer's time.
- **Feedback: specific, kind, and labeled.** Mark blocking vs. `nit:`; explain the *why*; assume good
  intent. Disagreements resolve on technical merit or escalate — never on rank or volume.
- **The gate:** approved against the checklist **and** CI green **and** blast-radius/security/rollback
  considered **and** all blocking comments resolved — before merge/Ship. "LGTM, it's small" is not a
  review.

## Related skills

- `[[engineering-lifecycle]]` — the full lifecycle; Review is the stage between build and Ship.
- `[[verification-and-debugging]]` — proving a change works before and during review; debugging what
  review surfaces.
- `[[shipping-and-release]]` — what happens after approval: rollout, monitoring, rollback.
- `[[go-best-practices]]` — the code-level standard a reviewer holds the diff to (and its own review
  checklist).
- `[[ai-security-on-gke]]` — the security lens for AI-infra changes (RBAC, secrets, isolation, supply chain).
- `[[staff-plus-engineering]]` — review as a leverage and knowledge-spreading tool, and how to raise
  the bar across an org.

---

# Reference — code-review-discipline

# Code Review Discipline Guide

The reference for the **Review** stage of the engineering lifecycle: how to review code and how to get
your code reviewed so that review actually catches problems instead of rubber-stamping them. The bar
is an engineer who has reviewed AI-infrastructure and ML-platform changes where the blast radius is
large — where one wrong manifest, RBAC binding, or training config can take down a cluster or waste
enormous amounts of compute. Mechanics (PR vs. MR vs. changelist, required approver count, merge
buttons) vary by team and tool; the discipline is universal. Match local norms, apply the safety rules
regardless.

## Overview: why review is a quality gate, not a formality

Code review exists to catch the classes of problem that *nothing else in the pipeline catches*, and to
spread knowledge so that no change is understood by exactly one person.

- **Tests verify behavior you already thought to check.** Review catches the rest: a design that will
  not extend, an edge case no test covers, an unbounded blast radius, a security hole, needless
  complexity, a config that quietly costs ten times what it should. Green CI means "the known checks
  pass," not "this change is safe."
- **Review is the cheapest place to catch a defect.** A problem caught in review costs minutes; the
  same problem caught in production — a cluster outage, a corrupted training run, a leaked credential —
  costs hours to days and real money. For AI infra the production cost can be a wasted GPU/TPU run worth
  far more than any human-review time you "saved."
- **Review spreads knowledge.** A second engineer who has read the change can maintain it, debug it at
  3am, and extend it later. Single-author code is an operational risk. Review is how the team's
  understanding stays larger than any one person's.
- **Review raises the bar.** It is where standards get taught and enforced in context — far more
  effective than a style doc nobody reads. Done well, it makes the whole team better; done as a
  rubber stamp, it teaches that the standard is "ship anything that compiles."

The mental shift: review is **a gate the change must pass**, with explicit criteria, not a social
courtesy you extend by typing "LGTM." If you would not bet your on-call shift on the change, it is not
approved.

## When to use this discipline

- Reviewing any PR / MR / changelist before it merges.
- Requesting review on your own change.
- Responding to review comments (yours or others').
- Deciding whether a change is safe to approve and merge — especially anything touching Kubernetes
  manifests, Helm, Terraform/IaC, RBAC, network policy, quotas, resource limits, or model/rollout
  configs.
- Setting or improving a team's review norms.

## The reviewer's lens (priority-ordered)

Review in this order. The point of ordering is **triage**: spend your scarce attention on what can
actually hurt you, and don't let a tidy bikeshed about naming crowd out the design flaw that takes the
cluster down. If you only have time for the top three, do the top three.

### 1. Correctness & edge cases

- Does the change do what the description says, and does it actually solve the stated problem?
- Walk the edge cases the author may have skipped: empty input, nil/None, zero and negative numbers,
  concurrent access, partial failure, retries and idempotency, timeouts, the very large and the very
  small. Off-by-one and boundary conditions.
- Error handling: are errors handled, propagated with context, and recoverable? Or swallowed?
- Concurrency: races, deadlocks, unbounded goroutines/threads, shared mutable state. (Language-level
  taste in `[[go-best-practices]]`.)
- Don't trust the diff in isolation — read enough surrounding code to know the change is correct *in
  context*, not just locally plausible.

### 2. Blast radius / safety — the one that bites hardest in AI infra

This is where AI-infra and ML-platform review differs from ordinary application review: the worst case
is not a broken feature, it is a **down cluster or a burned budget**. Ask, for every change: *what is
the worst thing that happens if this is wrong, how many workloads/tenants does it touch, and how fast
can we undo it?*

Review these artifact classes with extra care:

- **Kubernetes manifests / Helm / Kustomize.** Diff the *rendered* output, not just the template or
  `values.yaml` — a one-line values change can rewrite many resources. Watch namespace/selector scope,
  `replicas`, rollout strategy, PodDisruptionBudgets, probes (a wrong readiness/liveness probe can
  crash-loop or mask failures), and anything cluster-scoped.
- **Terraform / IaC.** Read the **plan**, not just the HCL. Look for resources being *destroyed* or
  *replaced* (especially stateful: disks, databases, node pools), changes to shared/foundational
  infra, and provider/version bumps. `terraform plan` showing a delete on a stateful resource is a
  stop-the-line moment.
- **RBAC.** Over-broad verbs (`*`), `cluster-admin` or cluster-scoped bindings, wildcards on resources,
  bindings to default or shared service accounts, anything granting more privilege than the workload
  needs. Least privilege is the default; escalation must be justified in the description.
- **Network policy.** Default-deny preserved? Does this open ingress/egress wider than necessary, or
  accidentally cut off a dependency? A wrong policy can silently break traffic across a namespace.
- **Resource limits / requests / quotas.** Missing limits (a workload that can starve neighbors),
  oversized requests (waste / unschedulable), quota changes that let one team consume a shared
  accelerator pool. On GPU/TPU clusters, a wrong request count or a missing limit is directly money.
- **Model & rollout configs.** Batch size, replica/parallelism counts, accelerator type and count,
  autoscaling bounds, traffic-split / canary percentages, checkpoint and serving paths. A wrong value
  here doesn't crash — it quietly runs at huge cost, serves the wrong model, or rolls 100% of traffic
  to an untested version. Confirm the change is gated/canaried, not a big-bang flip.

For high-blast-radius changes, require a **rollback plan in the description** and confirm the change is
staged (canary/progressive rollout) wherever the platform supports it. "We can roll it back" is only
true if someone has said *how*, fast.

### 3. Security

Security is in scope on every review, not just security-labeled ones. The AI-infra threat model and the
concrete controls live in `[[ai-security-on-gke]]`; at minimum check:

- **Secrets:** none hardcoded, logged, or printed; pulled from a secret manager, not committed.
- **Privilege:** least privilege (ties into RBAC above); no `privileged` containers, `hostPath`,
  host networking, or running as root without a justified, reviewed reason.
- **Exposure:** nothing made public that shouldn't be (Services, buckets, endpoints, dashboards).
- **Input:** untrusted input validated and bounded — including model inputs, prompts, and data
  pipelines, not just HTTP handlers.
- **Supply chain:** new dependencies and base/container images vetted; pinned, not `latest`; image
  provenance considered.

### 4. Tests present and meaningful

- New or changed logic has tests. Absence of tests for new behavior is a **blocking** issue, not a nit.
- The tests are *meaningful*: they assert real behavior, they cover the edge cases from lens #1, and
  **they would fail if the code were wrong.** Assertion-free tests, tests that mock the thing under
  test, and always-green tests are theater — call them out.
- For infra/config, "tests" may mean a rendered-manifest diff, a `terraform plan`, a dry-run, a
  policy lint (e.g. conftest/OPA), or a staging deploy — there should be *some* evidence beyond "it
  parses." See `[[verification-and-debugging]]` for proving a change works.

### 5. Simplicity, reuse, readability

- Is this the **simplest change that solves the problem**? Push back on speculative generality and
  abstraction added "for later." (Rule of three: don't abstract until you have three real cases.)
- Does it **reuse** existing helpers/modules instead of re-implementing or copy-pasting? Duplication
  is a maintenance liability.
- Is it **readable** by the next person — clear names, small functions, no dead code, comments that
  explain *why* not *what*? Code-level standards: `[[go-best-practices]]`.
- Readability matters, but **do not let a readability nit block a change while a design flaw goes
  unmentioned.** Fix the order of importance, not just the comment.

### 6. Observability & rollback (for production changes)

- **Observability:** will we be able to *see* this working (and failing) in prod — logs, metrics,
  traces, alerts? A change that adds behavior with no signal is a change you can't operate.
- **Rollback:** is there a fast, known way to undo it? Feature flag, image tag, Helm revision,
  Terraform revert, traffic shift back. For stateful or migration changes, is the rollback *actually*
  safe (no data loss)?
- These are blocking for anything that touches production. Detail in `[[shipping-and-release]]`.

## The author's pre-review self-check

The fastest way to a good review is to make the change easy to review. Before you request a human,
**review your own diff as if it were someone else's** and clear this bar:

- **Small.** Scope the change to one logical thing. A 200-line PR gets a real review; a 2,000-line PR
  gets an "LGTM." Split refactors out from behavior changes. If it must be large, structure it into
  reviewable commits and say so.
- **Described.** The description says **why**, not just **what** — the problem, the approach, the
  alternatives you rejected, the blast radius, and (for prod/infra) the rollback plan and how it was
  validated. Link the issue/design doc. The diff shows *what*; only you can supply *why*.
- **Self-reviewed.** Read every line of your own diff. Remove debug prints, dead code, TODOs you can
  finish now, and anything unrelated that snuck in. Most review comments are things the author would
  have caught by reading their own diff first.
- **CI green.** Don't make a human be your linter or test runner. Tests, lint, build, and (for infra)
  the rendered diff / `terraform plan` are attached or passing before you request review.
- **Right reviewers.** Tag someone who knows the code or the risk area — and for high-blast-radius
  infra/security changes, someone who owns that domain.

A PR that arrives small, described, self-reviewed, and green gets reviewed faster and better. Respect
the reviewer's time and they will respect your change.

## Giving and receiving feedback well

Review is a human interaction with technical content. Bad review interactions make people batch fewer,
bigger changes and avoid review — the opposite of the goal.

### Giving feedback

- **Be specific.** Point at the line, explain the problem, and where possible suggest the fix. "This
  could be cleaner" is useless; "this re-reads the file on every iteration — hoist it out of the loop"
  is actionable.
- **Be kind, and review the code, not the person.** "This function does X" not "you always do X."
  Praise good work too — it calibrates the rest of your feedback and makes blocking comments land.
- **Label severity.** Distinguish **blocking** (must fix before merge: correctness, blast radius,
  security, missing tests) from **nit:** / **suggestion:** / **optional:** (preferences and polish the
  author can take or leave). Don't let nits read as blockers — and don't bury a blocker among nits.
- **Explain the why.** "Use a context with a timeout *because this call can hang and block the
  rollout*" teaches; a bare "add a timeout" just demands. The why is what makes the author better and
  what lets them push back intelligently if you're wrong.
- **Assume good intent.** The author was solving a real problem under real constraints. Ask before you
  accuse — "is there a reason this isn't behind a flag?" beats "this must be behind a flag."
- **Bound the round-trips.** If a thread has gone three rounds, switch to a call or a pairing session.
  Comment threads are a poor medium for design disagreement.

### Receiving feedback

- **Assume good intent and don't take it personally.** The reviewer is improving the change, not
  attacking you. Every comment caught is a bug that didn't reach production.
- **Respond to every comment** — fix it, or explain why not. Don't silently resolve threads.
- **Push back when you disagree** — with a technical reason. The author often knows context the
  reviewer doesn't; surfacing it is the point. Disagreement is healthy; "because I said so" (from
  either side) is not.

### Disagreeing and resolving

- Resolve on **technical merit**, not seniority or volume. Bring data, a benchmark, a failing test, or
  the relevant doc/standard.
- If you're stuck, **escalate to a tie-breaker** (a domain owner, a third reviewer, a quick group
  decision) rather than letting the change rot or one side steamrolling the other.
- Capture the decision in the thread so the next person doesn't relitigate it.
- The reviewer's job is to ensure the change is *good enough to ship*, not *exactly how I would have
  written it*. Approve changes you'd write differently but that are correct, safe, and clear.

## The process

A repeatable Review stage with explicit gates:

1. **Author self-reviews and prepares.** Diff is small, described (why + blast radius + rollback +
   how validated), self-reviewed, CI green. Then request review from the right people.
2. **Reviewer reads for context first.** Read the description and the linked issue/design before the
   code. Understand *what problem* this solves and *what it touches* before judging the lines.
3. **Reviewer applies the lens in priority order.** Correctness → blast radius/safety → security →
   tests → simplicity/reuse/readability → observability/rollback. For infra/config, diff the rendered
   manifests / read the plan, not just the source.
4. **Reviewer leaves specific, labeled feedback.** Blocking vs. nit, with the *why* and a suggested
   fix where possible. Note what's good, not only what's wrong.
5. **Author responds to every comment.** Fix or justify; push back with reasons where warranted; keep
   the diff small as it evolves.
6. **Iterate** until all **blocking** comments are resolved. Nits are the author's discretion (or a
   fast follow they explicitly commit to in the thread, not a vague "later").
7. **Disagreements escalate, not stall.** If a blocking disagreement can't be resolved on merit,
   bring in a tie-breaker.
8. **Checkpoint — the merge gate.** Before merge / Ship, ALL must hold:
   **approved against the checklist** · **CI green** · **blast radius, security, and rollback
   considered** · **all blocking comments resolved**. Only then merge. Approval without these is not
   approval.

## Rationalizations & rebuttals

The excuses — from authors and reviewers alike — that turn review into theater, each with its rebuttal:

- *"LGTM, it's small."* — Small diffs cause big outages: a one-character RBAC verb, a wrong replica
  count, a flipped boolean. Size is not safety. Read it.
- *"Tests pass, so it's fine."* — Tests check what you thought to check. Review exists for design,
  blast radius, security, and simplicity — the things tests *can't* see. Green CI is a precondition,
  not the gate.
- *"I'll fix it in a follow-up."* — Follow-ups for blocking issues mostly don't happen, and the risk
  ships now. Blocking is blocking; follow-up is for nits, and only when explicitly committed in the
  thread.
- *"It's just infra/config, not real code."* — Config *is* the production system. A YAML typo, a quota
  bump, or a values-file change has a larger blast radius than most application code. Review it harder,
  not softer — diff the rendered output.
- *"The author is senior, I trust them."* — Review is a system property, not a trust score. Seniors
  make blast-radius and security mistakes too, and rubber-stamping their work means no second pair of
  eyes on the riskiest changes. Trust *and* verify.
- *"It's urgent, no time to review."* — The urgent change is exactly the one most likely to be wrong,
  and the most expensive to get wrong. Do a focused, top-of-the-lens review fast; never skip the gate.
- *"I already reviewed v1."* — The risk is in the delta since you approved. Re-read the new commits;
  force-pushed changes especially.
- *"Nobody else can review this; only I understand it."* — That is the bug, not the excuse. Single-
  author code is an operational liability — get a second person up to speed *via* the review.

## Red flags — stop and reconsider

- **A huge, unreviewable PR.** Hundreds/thousands of lines, refactor mixed with behavior change. Ask
  for a split before you start; an "approval" you can't actually give is worthless.
- **Rubber-stamp approval.** "LGTM" seconds after the diff posts, on a non-trivial change. No one read
  it. Approval without engagement is negligence, not courtesy.
- **No description, or a what-not-why description.** "Update config." If the author can't say why and
  what it touches, the reviewer can't assess risk — send it back.
- **Blast radius / cost / security unconsidered.** An IaC, RBAC, quota, network, or model-config change
  with no mention of worst case, scope, cost, or rollback. The most dangerous changes hiding as
  routine ones.
- **Nitpicking while missing the design flaw.** Twelve comments about naming and zero about the
  unbounded retry loop that will hammer the API server. Wrong altitude — review the risk first.
- **CI red, or skipped/disabled tests, merged anyway.** "Flaky, ignore it." A disabled test is an
  unreviewed assumption.
- **`terraform plan` showing destroy/replace on stateful resources**, or RBAC granting `*`/`cluster-
  admin`, or a manifest removing default-deny — treated as routine. Stop the line.
- **A change with no rollback story touching production.** "It'll be fine" is not a rollback plan.
- **Author and reviewer are arguing on rank/volume**, not technical merit. Escalate to a tie-breaker.

## Verification gate (definition of done for a review)

A change is approved-to-merge only when **all** of the following are true — confirm each explicitly:

- [ ] **Reviewed against the checklist** in priority order (correctness → blast radius → security →
      tests → simplicity/reuse → observability/rollback), with the diff actually read in context.
- [ ] **CI green** — tests, lint, build, and (for infra) rendered-manifest diff / `terraform plan`
      reviewed, not red and not skipped.
- [ ] **Blast radius, security, and rollback considered and acceptable** — worst case understood,
      least privilege held, and a fast, safe undo exists for prod/infra changes.
- [ ] **Tests present and meaningful** for new/changed logic (or equivalent infra validation), and
      they would fail if the code were wrong.
- [ ] **All blocking comments resolved**; nits dispositioned (fixed or explicitly deferred in-thread).

Only then: approve and merge / proceed to Ship (`[[shipping-and-release]]`). An approval that skips any
of these is not an approval — it's a liability with your name on it.

## Canonical references

- Google Engineering Practices — How to do a code review (reviewer's guide):
  https://google.github.io/eng-practices/review/reviewer/
- Google Engineering Practices — The CL author's guide:
  https://google.github.io/eng-practices/review/developer/
- Google Engineering Practices — The standard of code review:
  https://google.github.io/eng-practices/review/reviewer/standard.html
- Conventional Comments (a shared vocabulary for severity: blocking / nit / suggestion):
  https://conventionalcomments.org/
- Go Code Review Comments — https://go.dev/wiki/CodeReviewComments
- SRE / blast-radius and progressive-rollout thinking — https://sre.google/books/
  *(Mechanics — required approvers, merge gates, tooling — vary by team and platform; it is 2026 and
  the AI-infra ecosystem moves fast. Verify your platform's specifics against current docs.)*

---

# Code Review Discipline — Examples & Checklists

Concrete, copy-pasteable checklists and worked comment examples to imitate. Pair with
`code-review-discipline-guide.md` for the rationale.

---

## Reviewer checklist — general (paste into your review)

Review top-to-bottom; spend attention in this order.

**Correctness & edge cases**
- [ ] Does the change actually solve the stated problem (read the description first)?
- [ ] Edge cases handled: empty/nil, zero/negative, very large/small, concurrency, partial failure,
      retries/idempotency, timeouts, off-by-one/boundaries.
- [ ] Errors handled and propagated with context — not swallowed.
- [ ] No races / deadlocks / unbounded concurrency / unsynchronized shared state.
- [ ] Read enough surrounding code to confirm correctness *in context*, not just in the diff.

**Blast radius / safety**
- [ ] Worst case if this is wrong is understood; scope (workloads/tenants/clusters) is bounded.
- [ ] A fast, safe rollback exists and is named in the description (for prod/infra changes).
- [ ] High-blast-radius changes are staged (canary / progressive rollout), not big-bang.

**Security**
- [ ] No secrets hardcoded, logged, or committed; pulled from a secret manager.
- [ ] Least privilege; no privileged/hostPath/host-network/root without a justified reason.
- [ ] Nothing wrongly exposed (Services, buckets, endpoints, dashboards).
- [ ] Untrusted input (incl. model inputs / prompts / data pipelines) validated and bounded.
- [ ] New dependencies and images vetted, pinned (not `latest`), provenance considered.

**Tests**
- [ ] New/changed logic has tests; absence is *blocking*.
- [ ] Tests assert real behavior, cover the edge cases above, and would fail if the code were wrong.
- [ ] No assertion-free, mock-the-thing-under-test, or disabled/skipped tests slipping through.

**Simplicity / reuse / readability**
- [ ] Simplest change that works; no speculative abstraction (rule of three).
- [ ] Reuses existing helpers/modules; no copy-paste duplication.
- [ ] Readable: clear names, small functions, no dead code/debug prints, comments say *why*.

**Observability / rollback (production changes)**
- [ ] We can see it working and failing (logs/metrics/traces/alerts).
- [ ] Rollback is fast and, for stateful/migration changes, safe (no data loss).

**Gate**
- [ ] Approved against this checklist · CI green · blast-radius/security/rollback considered · all
      blocking comments resolved.

---

## Reviewer checklist — AI-infra / IaC / Kubernetes manifests (extra care)

The high-blast-radius section. A wrong value here can take down a cluster or burn a large compute bill.
**Review the rendered/planned output, not just the source.**

**Kubernetes manifests / Helm / Kustomize**
- [ ] Reviewed the *rendered* output (`helm template`, `kustomize build`) — a one-line values change
      can rewrite many resources.
- [ ] Namespace and label/selector scope correct; nothing accidentally cluster-scoped.
- [ ] `replicas`, rollout `strategy` (maxSurge/maxUnavailable), and PodDisruptionBudget sane.
- [ ] Readiness/liveness/startup probes correct — won't crash-loop or mask failures.
- [ ] Resource `requests`/`limits` present and right-sized; not starving neighbors, not unschedulable.
- [ ] GPU/TPU `requests` count and accelerator type correct (this is directly money).

**Terraform / IaC**
- [ ] Reviewed the **plan**, not just the HCL.
- [ ] No unexpected *destroy*/*replace* — especially on stateful resources (disks, DBs, node pools).
      A destroy on a stateful resource is a stop-the-line moment.
- [ ] Provider/module/version bumps understood; changes to shared/foundational infra flagged.

**RBAC**
- [ ] No `*` verbs/resources, no `cluster-admin` or cluster-scoped binding unless justified.
- [ ] Bound to a dedicated, least-privilege service account — not `default` or a shared SA.
- [ ] Privilege granted matches what the workload actually needs.

**Network policy**
- [ ] Default-deny preserved; ingress/egress not opened wider than necessary.
- [ ] No dependency accidentally cut off.

**Quotas / limits**
- [ ] Quota changes don't let one team consume a shared accelerator pool or starve others.

**Model / rollout configs**
- [ ] Batch size, parallelism/replica counts, accelerator type+count, autoscaling bounds sane.
- [ ] Traffic-split / canary percentage is staged, not 100% to an untested version.
- [ ] Checkpoint and serving paths point at the intended model/artifact.
- [ ] Rollback (previous image tag / Helm revision / traffic shift back) named and fast.

---

## Author pre-review self-check (run before requesting review)

- [ ] **Small** — one logical change; refactor split out from behavior change. If large, structured
      into reviewable commits and explained.
- [ ] **Described** — the description says **why** (problem, approach, alternatives rejected), the
      **blast radius**, the **rollback plan**, and **how it was validated**. Issue/design linked.
- [ ] **Self-reviewed** — read every line; removed debug prints, dead code, unrelated changes,
      finishable TODOs.
- [ ] **CI green** — tests, lint, build, and (infra) rendered diff / `terraform plan` attached or
      passing. Don't make a human your linter.
- [ ] **Right reviewers** — someone who knows the code/risk; a domain owner for high-blast-radius
      infra/security changes.

---

## Good vs. bad review comments

The same finding, badly and well. Imitate the right column's specificity, severity label, and *why*.

### Vague vs. specific + actionable

- BAD: `this could be cleaner`
- GOOD: `nit: this re-reads the config file on every loop iteration — hoist the read above the loop so
  it happens once.`

### Demand without the why vs. teaching

- BAD: `add a timeout here`
- GOOD: `blocking: this gRPC call has no deadline, so if the backend hangs it blocks the rollout
  goroutine indefinitely. Pass a context with a timeout (e.g. 30s) and handle DeadlineExceeded.`

### Accusatory vs. assume-good-intent question

- BAD: `you always forget to gate risky changes`
- GOOD: `Is there a reason this traffic shift goes straight to 100%? Given the blast radius I'd expect
  a canary (e.g. 5% → 50% → 100%) — happy to be wrong if there's a constraint I'm missing.`

### Nit dressed as a blocker vs. labeled severity

- BAD: `Rename this variable.` *(reads as required; it's a preference)*
- GOOD: `nit (optional): tmp → renderedManifest would read better, but not blocking.`

### Blocker buried in nits vs. surfaced

- BAD: twelve `nit:` comments on naming, and somewhere in the middle, unlabeled: `also this retries
  forever`
- GOOD: `blocking: the retry loop in reconcile() has no backoff or max attempts — on a transient API
  error this will hammer the apiserver. Add capped exponential backoff and a max retry count.`
  *(Lead with the blocker; nits go after, clearly labeled.)*

### Infra blast-radius catch (the kind tests won't make)

- GOOD: `blocking: this Role binds verbs: ["*"] on resources: ["*"] cluster-wide. The controller only
  needs get/list/watch on pods and create on events. Scope it down — a wildcard cluster binding is a
  privilege-escalation path if the controller is compromised. (see [[ai-security-on-gke]])`
- GOOD: `blocking: the terraform plan shows the GKE node pool being *replaced*, not updated — that
  drains and recreates every node and will interrupt running training jobs. Confirm this is intended;
  if it's just a label change, we may be able to avoid the replace.`
- GOOD: `blocking: this Deployment dropped the resources.limits block. Without a memory limit one pod
  can OOM the node and take its neighbors with it. Restore limits sized from the observed usage.`

### Receiving feedback — good responses

- GOOD (accept): `Good catch — fixed, added a 30s deadline and a test that asserts DeadlineExceeded is
  surfaced.`
- GOOD (push back with reason): `I left this synchronous on purpose: the caller already runs it in a
  bounded worker pool, and making it async here would reorder the writes. Documented that in a comment
  — does that address the concern?`
- BAD: silently resolving the thread with no reply, or `done` with no change actually pushed.
