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
