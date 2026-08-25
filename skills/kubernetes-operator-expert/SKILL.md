---
name: kubernetes-operator-expert
description: Designing, building, and shipping production Kubernetes Operators — the API/packaging/lifecycle dimension. Use when defining CustomResourceDefinitions (CRDs), designing api groups/versions/kinds and the spec/status split, writing OpenAPI v3 / structural schemas with CEL validation rules, adding /status or /scale subresources, printer columns, defaulting, and immutability; versioning APIs (alpha/beta/GA, served vs stored versions, conversion webhooks); writing validating/mutating/defaulting admission webhooks or CEL ValidatingAdmissionPolicy; scaffolding with kubebuilder or Operator SDK (controller-gen markers that generate CRDs/RBAC/webhooks); packaging with OLM (ClusterServiceVersion, bundles, catalogs, OperatorHub) or kustomize/Helm; status conditions, events, metrics, leader election, RBAC, and capability levels. For reconcile-loop mechanics (controller-runtime, client-go, watches, workqueues) see kubernetes-controller-expert.
---

# Kubernetes Operator Expert

Apply the judgment of an engineer who has designed, published, and maintained several production
Operators (kubebuilder/Operator SDK + OLM) across major and breaking API versions. The Operator is
an **API product**: the CRD is your public contract, and a bad CRD outlives a bad reconciler.

## How to use this skill

1. **Read `kubernetes-operator-expert-guide.md`** in this directory — the full reference. Apply it to
   the API design, CRD, webhook, packaging, and lifecycle decisions at hand.
2. For concrete artifacts to imitate — an annotated CRD with `/status`, CEL validation and printer
   columns; a kubebuilder Go type with markers; a validating-webhook skeleton — read **`examples.md`**.
3. Reconcile-loop mechanics (controller-runtime, watches, workqueues, status patching) live in
   `[[kubernetes-controller-expert]]`. This skill is the operator/API/packaging/lifecycle layer; defer
   there for the loop itself.
4. Match the surrounding repo/cluster conventions (existing api group, scaffolding tool, distribution
   channel); apply the correctness/compatibility/security rules regardless.

## Essentials (full detail in `kubernetes-operator-expert-guide.md`)

- **Use an Operator only when you need ongoing operational logic** — failover, backup/restore,
  in-place upgrades, reconciling external state. If a Helm chart or a few manifests fully express the
  app, you don't need an Operator. Measure ambition with the **capability levels** (Basic Install →
  Seamless Upgrades → Full Lifecycle → Deep Insights → Auto Pilot).
- **The CRD is a public API. Design it like one:** declarative desired state, minimal, forward-
  compatible. `spec` = user intent; `status` = observed state owned by the controller. Never put
  status-like fields in spec or accept fields the user shouldn't set.
- **Always enable a structural schema** (full OpenAPI v3, no bare `x-kubernetes-preserve-unknown-fields`
  at the root) — it's required for CEL, conversion, defaulting, and pruning. Validate with **CEL
  `x-kubernetes-validations`** in the schema before reaching for a webhook.
- **Enable the `/status` subresource** so spec and status update independently and `metadata.generation`
  tracks spec changes; add `/scale` only if the resource is genuinely scalable. Add printer columns,
  `categories`, and short names for UX.
- **Status conditions are the user/automation contract.** Use `metav1.Condition` (type/status/reason/
  message/observedGeneration/lastTransitionTime); set `Ready`/`Available`/`Progressing`/`Degraded`.
  Emit Events for transitions. Without good status, your Operator is unobservable and un-GitOps-able.
- **Never break a GA API.** alpha (may break) → beta (deprecation policy) → GA (stable forever).
  Add fields, don't remove or repurpose. New incompatible shape ⇒ a new version + **conversion webhook**
  (hub-and-spoke: one storage/hub version, spokes convert to/from it). Exactly one `storage: true`.
- **Mark immutable fields with CEL** (`self == oldSelf`) instead of a mutating webhook where possible.
  Default values in the schema (`default:`); only use a defaulting webhook for logic the schema can't express.
- **Webhooks can brick a cluster.** Set `failurePolicy` deliberately (Fail blocks writes if the webhook
  is down), scope with `namespaceSelector`/`objectSelector` to **exclude kube-system and the operator's
  own namespace**, declare `sideEffects: None`, keep latency low, and run ≥2 replicas. Manage certs with
  cert-manager (or the controller-runtime cert rotator). Prefer **CEL `ValidatingAdmissionPolicy`**
  (in-process, no webhook server, no certs) when the check is expressible in CEL.
- **controller-gen markers are the source of truth.** `// +kubebuilder:...` markers generate the CRD,
  RBAC, and webhook manifests; never hand-edit generated YAML — edit the marker and regenerate.
- **Least-privilege RBAC, leader election on, run multiple replicas.** Generate RBAC from markers,
  scope to the resources you actually touch, prefer Role over ClusterRole where namespace-scoped.
  Leader election means only one active reconciler across replicas; the standbys give fast failover.
- **Pick scope deliberately:** namespace-scoped CRD for tenant-owned resources, cluster-scoped for
  infrastructure. Decide multi-tenancy (one operator watching all namespaces vs per-namespace) up front.
- **Package for the distribution channel:** OLM bundle + ClusterServiceVersion for OperatorHub/OpenShift
  and managed upgrades; plain kustomize/Helm for self-managed installs. The Operator must own **its own
  upgrade**, including CRD schema migration — that's the hard part of Full Lifecycle.
- **Anti-patterns:** god-CRDs, leaking implementation/runtime detail into the API, ignoring status,
  breaking changes on GA, webhooks with `failurePolicy: Fail` and no namespace exclusions, an Operator
  that can't upgrade itself.

## Related skills

- `[[kubernetes-controller-expert]]` — the reconcile loop: controller-runtime, client-go, watches,
  workqueues, owner refs, status patching, rate limiting. **Read it for the controller half.**
- `[[kubernetes-internals-expert]]` — how the apiserver serves CRDs, admission chain ordering, etcd
  storage, conversion plumbing.
- `[[go-best-practices]]` — the Operator is Go; apply these standards to the code.
- `[[kueue-advanced]]` — a real, large operator-style project (CRDs, webhooks, conversion) to study.
