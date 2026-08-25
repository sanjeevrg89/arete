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
