# Kubernetes Operator Engineering Guide

Production guidance for designing, building, and shipping Kubernetes Operators, from the perspective
of someone who has published and maintained several across alpha→beta→GA transitions and breaking
schema migrations. This is the single source of truth; `SKILL.md` and `AGENTS.md` defer to it.

**Scope boundary.** This guide is the *operator / API / packaging / lifecycle* layer. The reconcile
loop itself — controller-runtime managers, watches, workqueues, predicates, owner references, status
patch vs update, rate limiting — lives in `[[kubernetes-controller-expert]]`. Read that for the loop;
read this for the CRD, webhooks, versioning, packaging, and operability around it.

> It is 2026 and this ecosystem moves fast. Kubernetes ships ~3 minor releases a year; kubebuilder,
> Operator SDK, OLM, and cert-manager release independently. Treat specific feature-gate and API
> maturity claims as "verify against the docs for your cluster version" — the *principles* here are
> stable; the exact GA version of a given feature may not be. Never invent a marker, field, or version.

---

## 1. The Operator pattern — and when NOT to use it

An **Operator** = a custom controller + one or more **CustomResourceDefinitions (CRDs)** that encode
operational knowledge for a specific application. It extends the Kubernetes API with new objects, then
runs a control loop that drives the real world toward the declared `spec` — automating what a skilled
human operator would do by hand: provision, configure, back up, restore, fail over, scale, and
**upgrade** a stateful or complex application.

You probably want an Operator when:
- The app has **non-trivial day-2 operations** — backup/restore, point-in-time recovery, leader
  election/failover, online schema or version migration, topology-aware reconfiguration.
- You must **continuously reconcile external state** (a cloud resource, a database cluster, a mesh).
- Users should declare *intent* (`size: 3`, `version: 8.0`) and have the system figure out the steps.

You probably **don't** need one when:
- A **Helm chart or a handful of manifests** fully expresses the app and its config. Templating is not
  operational logic. Don't build a CRD whose only job is to render a Deployment — that's a worse Helm.
- The behavior is one-shot install/config with no ongoing reconciliation beyond what built-in
  controllers (Deployment, StatefulSet, HPA) already provide.

| Need | Reach for |
|------|-----------|
| Parameterized install/config, no day-2 logic | Helm chart / kustomize |
| Built-in reconciliation suffices (rollout, scale) | Deployment/StatefulSet + HPA |
| Custom reconciliation but no new API surface needed | plain controller on built-in types |
| New declarative API + ongoing operational automation | **Operator (CRD + controller)** |

### Capability levels (the Operator maturity model)

A useful yardstick for *how much* operator to build (from the Operator Framework / OperatorHub):

1. **Basic Install** — provision the app and its config from the CR.
2. **Seamless Upgrades** — upgrade the managed app (and the operator) without downtime/data loss.
3. **Full Lifecycle** — backup/restore, failover, scale-out/in, reconfigure.
4. **Deep Insights** — metrics, alerts, log processing, workload health surfaced in status/Prometheus.
5. **Auto Pilot** — auto-scaling, auto-tuning, auto-healing, anomaly response without human input.

Each level subsumes the ones below it. Most internal operators are right to stop at 2–3; don't build
Auto Pilot you can't operate. The level is a *design target*, not a checklist to game.

---

## 2. CRD & API design — the contract

The CRD is your **public API**. Reconcilers can be rewritten; an API shape, once users depend on it,
is forever (or until a painful migration). Spend the most design effort here.

### Group / Version / Kind

- **Group**: a DNS-style namespace you own, e.g. `cache.example.com`, `acme.io`. Group your related
  kinds under one group. Don't squat on `*.k8s.io` / `*.kubernetes.io` (reserved).
- **Version**: `v1alpha1`, `v1beta1`, `v1`, `v2` — maturity + evolution axis (Section 4).
- **Kind**: singular PascalCase noun (`MemcachedCluster`), with a lowercase plural resource name
  (`memcachedclusters`). One kind = one cohesive concept the user reasons about.

A `GroupVersionKind` (GVK) like `cache.example.com/v1, Kind=MemcachedCluster` is the addressable type.

### The spec / status split (non-negotiable)

- **`spec`** = *desired state*, written by the user/GitOps. Declarative intent only.
- **`status`** = *observed state*, written **only** by the controller. Never require the user to set
  status; never read status as input to your own desired-state decisions other than for idempotency.

This split is what makes the resource declarative and GitOps-friendly: a user `kubectl apply`s spec;
the controller reports status; the two never fight. Enabling the `/status` subresource (below) enforces
it at the apiserver level.

### Structural schemas & OpenAPI v3 (required)

`apiextensions.k8s.io/v1` CRDs **must** carry an OpenAPI v3 `validation.openAPIV3Schema` that is
**structural**: every field typed, no ambiguous `x-kubernetes-preserve-unknown-fields` at the root,
no top-level `oneOf`/`anyOf` that hides the type. Structural schemas are the prerequisite for:

- **Pruning** of unknown fields (on by default for v1 CRDs) — clients can't smuggle junk into etcd.
- **Defaulting** via `default:` in the schema.
- **CEL validation** (`x-kubernetes-validations`).
- **Conversion** between versions.

Use `x-kubernetes-preserve-unknown-fields: true` **only** on a deliberately open sub-object (e.g. an
embedded third-party config blob), never as a way to skip schema work.

### CEL validation (`x-kubernetes-validations`) — prefer over webhooks

Common Expression Language rules live **in the schema** and run in the apiserver — no webhook server,
no certs, no availability risk. Reach for these *first*; only fall back to a validating webhook when
the rule genuinely can't be expressed in CEL (cross-resource lookups, external calls).

```yaml
x-kubernetes-validations:
  - rule: "self.replicas <= self.maxReplicas"
    message: "replicas must not exceed maxReplicas"
  - rule: "self.minReplicas <= self.replicas"
    message: "replicas must be >= minReplicas"
```

- **Immutability**: with `oldSelf` (transition rules), enforce that a field can't change after creation:
  ```yaml
  x-kubernetes-validations:
    - rule: "self == oldSelf"
      message: "storageClass is immutable"
  ```
  Apply at the field level for the immutable field. This replaces a mutating/validating webhook for the
  classic "you can't change the storage class after provisioning" case.
- CEL has cost limits; very large lists / deeply nested rules can be rejected as too expensive. Keep
  rules focused. `x-kubernetes-validations` can sit at any level of the schema (root for cross-field).

### Subresources

- **`/status`** — enable it. Spec and status get separate update endpoints; status updates don't bump
  `metadata.generation`, so `status.observedGeneration == metadata.generation` cleanly tells you
  "controller has caught up to the latest spec." Without it, a status write can clobber a concurrent
  spec write.
- **`/scale`** — enable **only** if the resource is meaningfully scalable by replica count. It lets
  `kubectl scale` and the **HPA** target your CR. Map `specReplicasPath`, `statusReplicasPath`, and
  (for HPA) `labelSelectorPath`. Don't add `/scale` to things that aren't replica-scalable.

### Quality-of-life: printer columns, categories, short names

- **`additionalPrinterColumns`** — what shows in `kubectl get`. Surface the fields an operator checks
  first (phase, ready replicas, a key status field, `Age`). JSONPath into spec/status.
- **`categories`** (e.g. `all`) — include your kind in `kubectl get all`-style group queries (use
  sparingly; `all` is noisy).
- **`shortNames`** — `kubectl get mc` for `MemcachedCluster`. Pick something that won't collide.

### Good API design principles

- **Declarative, not imperative.** Express the *what*, never an RPC verb (`action: restart` is a smell;
  prefer a declarative field or an annotation that the controller observes and clears).
- **Minimal.** Every field is forever. Start small; add fields later (backward-compatible). You can't
  remove a GA field. When in doubt, leave it out.
- **Forward-compatible.** Optional new fields with sensible defaults; old clients omitting them must
  keep working. Use pointers / `+optional` so "unset" is distinguishable from "zero".
- **Required vs optional.** Mark required fields in the schema (`required:`) and unmarked-but-needed
  fields with defaults. In Go types, optional = pointer or has `omitempty` + `// +optional`.
- **Don't leak implementation.** The API describes user intent, not your internal data structures,
  pod names, or reconcile phases. If you rename your internal controller, the CRD must not change.
- **Units and enums are explicit.** Use `resource.Quantity` for sizes, typed enums (`// +kubebuilder:
  validation:Enum=...`) for closed sets, durations as strings parsed to `metav1.Duration`.

---

## 3. controller-gen markers — generated CRDs, RBAC, webhooks

In Go operators (kubebuilder / Operator SDK), the **Go type + `// +kubebuilder:` markers are the source
of truth**. `controller-gen` reads them and emits the CRD YAML, RBAC, and webhook configs. **Never
hand-edit generated manifests** — change the marker and run `make manifests` / `make generate`.

High-value markers (verify exact spelling against the controller-gen version you use):

- Type/printer/scope:
  - `// +kubebuilder:object:root=true` — mark the Go struct as a root API type.
  - `// +kubebuilder:subresource:status` — enable `/status`.
  - `// +kubebuilder:subresource:scale:specpath=...,statuspath=...,selectorpath=...` — enable `/scale`.
  - `// +kubebuilder:resource:scope=Cluster,shortName=mc,categories=all,path=memcachedclusters`
  - `// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status``
  - `// +kubebuilder:storageversion` — which version is stored in etcd (Section 4).
- Validation/defaulting (field-level):
  - `// +kubebuilder:validation:Required` / `// +optional`
  - `// +kubebuilder:validation:Minimum=1`, `Maximum=`, `MinLength=`, `Pattern=`, `Enum=A;B;C`
  - `// +kubebuilder:default=3`
  - `// +kubebuilder:validation:XValidation:rule="self >= oldSelf",message="size cannot shrink"` — CEL.
- RBAC (on the reconciler):
  - `// +kubebuilder:rbac:groups=cache.example.com,resources=memcachedclusters,verbs=get;list;watch;create;update;patch;delete`
  - `// +kubebuilder:rbac:groups=cache.example.com,resources=memcachedclusters/status,verbs=get;update;patch`
  - `// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete`
- Webhooks:
  - `// +kubebuilder:webhook:path=...,mutating=false,failurePolicy=fail,sideEffects=None,groups=...,resources=...,verbs=create;update,versions=...,name=...,admissionReviewVersions=v1`

`make manifests` regenerates CRDs+RBAC+webhook configs; `make generate` regenerates `zz_generated.
deepcopy.go` (the `DeepCopyObject` methods every API type needs). CI must fail if generated files are
stale (`git diff --exit-code` after `make generate manifests`).

---

## 4. API versioning & evolution

The single most consequential discipline in operator work. Get this wrong and you brick users' clusters
on upgrade or trap yourself in a shape you can't change.

### Maturity conventions

- **`v1alpha1`** — experimental. May change or be removed in any release, no migration promised.
  Disabled-by-default in some distributions. Use freely while exploring; tell users it's unstable.
- **`v1beta1`** — reasonably stable. Follow a deprecation policy: keep a beta API and its successor
  served for a window before removal. Schema should be close to final.
- **`v1` / GA** — stable **forever**. **Never break it.** No removed fields, no changed semantics,
  no tightened validation that rejects previously-valid objects. New behavior is additive only.

### Served vs stored versions

A CRD can **serve** multiple versions simultaneously (`served: true`) but **stores exactly one**
(`storage: true`) in etcd. Clients read/write any served version; the apiserver converts to the storage
version on write and back on read.

- Promote a new version by adding it as `served`, then flipping `storage` to it, then migrating
  existing stored objects (re-write them so etcd holds the new storage version), then eventually
  dropping the old served version after the deprecation window.
- Migrating stored objects: touch every object (e.g. with the **storage-version-migrator**, or a
  controller that re-applies) so nothing in etcd remains at a soon-to-be-removed storage version.
  You cannot drop a version from the CRD while objects are still stored at it.

### Conversion webhooks — hub-and-spoke

When two served versions have **different shapes**, the apiserver needs to convert between them. With
`spec.conversion.strategy: Webhook`, it calls your conversion webhook for every cross-version
read/write.

The maintainable pattern is **hub-and-spoke**: pick **one version as the hub** (usually the storage
version) and implement conversion only *to/from the hub*. In controller-runtime:
- The hub implements `conversion.Hub` (marker method, no-op).
- Each spoke implements `conversion.Convertible` with `ConvertTo(hub)` and `ConvertFrom(hub)`.
- N versions ⇒ N-1 spoke conversions, not N², because everything routes through the hub.

Rules:
- Conversion must be **lossless and round-trippable** where possible. If a new field has no equivalent
  in the old version, stash it via annotations on down-conversion so up-conversion can restore it — or
  accept documented lossiness.
- The conversion webhook is on the critical read/write path for those types; it must be fast and
  highly available, with proper certs (same cert concerns as admission webhooks, Section 5).
- `strategy: None` (no conversion) is only valid when all versions are structurally identical
  (same schema, just a version bump).

### Backward-compatibility rules

- **Add fields, never remove or repurpose.** A removed field breaks clients; a repurposed field is worse.
- **Don't tighten validation on an existing version** in a way that rejects objects that used to be
  valid — that breaks `apply` for existing users mid-upgrade.
- **Defaults must not change** observable behavior for objects that omit the field.
- Deprecate visibly: mark the version/field deprecated (`deprecated: true`, `deprecationWarning` on the
  CRD version) so `kubectl` warns users; give a real migration window before removal.

---

## 5. Admission webhooks

Webhooks intercept API requests after authn/authz and schema validation, before persistence. Three
roles (a single webhook server commonly hosts all three for different objects):

- **Mutating** (`MutatingWebhookConfiguration`) — can patch the object (set defaults, inject sidecars,
  add labels). Runs **before** validating webhooks.
- **Validating** (`ValidatingWebhookConfiguration`) — accept/reject only; cannot mutate. Runs last.
- **Defaulting** — in kubebuilder this is just a mutating webhook implementing `Defaulter`; prefer
  schema `default:` for static defaults and reserve the webhook for logic-dependent defaults.

### Admission ordering

1. apiserver authn → authz
2. **mutating** admission (webhooks + mutating admission policies), possibly re-run
3. object schema validation + structural/CEL schema checks
4. **validating** admission (webhooks + `ValidatingAdmissionPolicy`)
5. persist to etcd

Within a phase, webhook ordering is **not guaranteed** — never have one webhook depend on another
webhook's mutation having already happened. Mutating webhooks may be **re-invoked** if a later mutating
webhook changes the object (`reinvocationPolicy: IfNeeded`), so they must be **idempotent**.

### Configuration knobs that matter

- **`failurePolicy`**: `Fail` (default — if the webhook is unreachable, the API request is **rejected**)
  vs `Ignore` (request proceeds). `Fail` is safer for correctness but means **a down webhook blocks
  writes to matching objects** — including, if you're not careful, the writes needed to recover the
  webhook itself.
- **`sideEffects`**: declare `None` (or `NoneOnDryRun`). Anything else breaks `kubectl --dry-run` and
  server-side apply diffing. A webhook should not mutate external state during admission.
- **`namespaceSelector` / `objectSelector`**: scope tightly. **Exclude `kube-system` and the operator's
  own namespace** (e.g. match-expression excluding a control-plane label) so you can't deadlock the
  cluster or yourself. Only intercept the objects you actually need to.
- **`matchPolicy: Equivalent`** so the webhook still fires when the object is created via a different
  served API version.
- **`timeoutSeconds`**: keep it short (e.g. 5–10s). Every matching API write waits on your webhook;
  slow webhooks add latency to the whole cluster.
- **`admissionReviewVersions: ["v1"]`** and serve the `v1` AdmissionReview.

### The ways a webhook bricks a cluster (and how to avoid them)

- **Self-deadlock**: `failurePolicy: Fail` + a selector that matches the namespace/objects the webhook
  itself needs to come up ⇒ the webhook can never start because its own pods can't be admitted.
  **Fix**: exclude control-plane and operator namespaces; never intercept the resources required to
  run the webhook.
- **Single point of failure**: one webhook replica + `Fail` ⇒ one crash blocks all matching writes.
  **Fix**: ≥2 replicas, PodDisruptionBudget, fast readiness, anti-affinity.
- **Latency tax**: a slow or far-away webhook slows every matching write cluster-wide. Keep it local,
  cached, and cheap.
- **Cert expiry**: an expired serving cert ⇒ TLS failures ⇒ with `Fail`, blocked writes. Automate
  rotation (below).

### Cert management

Webhooks are TLS servers; the apiserver pins the CA via `caBundle` in the webhook config. Options:
- **cert-manager** — issue a `Certificate`, and use the cert-manager **CA-injector** to populate
  `caBundle` into the webhook/CRD-conversion configs automatically. The standard production choice.
- **controller-runtime's built-in cert rotator / a sidecar** — self-signs and rotates, patches the
  `caBundle`. Fine for self-contained operators without cert-manager.
Never ship hardcoded/long-lived certs that you have to rotate by hand.

### CEL `ValidatingAdmissionPolicy` — the webhook-less alternative

For *validation* expressible in CEL, prefer a **`ValidatingAdmissionPolicy`** (+ its binding) over a
validating webhook. It runs **in-process in the apiserver** — no webhook server, no certs, no
availability/latency/deadlock risk, no extra hop. (A mutating analogue, `MutatingAdmissionPolicy`, has
been progressing through the maturity stages — **verify its status for your cluster version** before
relying on it.) Use webhooks when you need cross-resource lookups, external data, or mutation the schema
and policies can't express; use CEL policies and schema CEL for everything else.

---

## 6. Tooling: kubebuilder, Operator SDK, and the rest

- **kubebuilder** (the upstream SIG project) and **Operator SDK** (Red Hat, builds on kubebuilder for
  Go) are the two mainstream **Go** scaffolders. Both give you: `init` a project, `create api` to
  scaffold a GVK + controller, `create webhook`, controller-runtime wiring, a `Makefile` with
  `make manifests/generate/docker-build/deploy`, and Kustomize `config/` overlays.
- They share the same engine: **controller-gen** (deepcopy + CRD/RBAC/webhook generation from markers)
  and **controller-runtime** (the manager/reconciler library). Marker syntax is largely common.
- **Operator SDK** adds OLM-oriented tooling (`operator-sdk` for bundles, scorecard, OLM integration,
  and **Ansible** / **Helm** operator types) on top.
- **Ansible operators** wrap an Ansible role: each reconcile runs a playbook. **Helm operators** turn a
  Helm chart into a CR — install/upgrade/uninstall map to a CRD. Both are quick to stand up and good
  for level-1/2 operators with **no custom Go logic**, but they cap your ceiling: no webhooks-as-code,
  limited status/condition control, harder Full-Lifecycle automation. Reach for them when the app is
  essentially "a chart with a CR face"; reach for Go when you need real operational logic.

Workflow: edit Go types + markers → `make generate manifests` → `make install` (apply CRDs) →
`make run` (out-of-cluster) or `make deploy` (in-cluster) → iterate. Keep generated files in CI lockstep.

---

## 7. Packaging & distribution

How users *install and upgrade* your operator. Pick by audience.

### Plain manifests / Kustomize / Helm

- **Kustomize** (`config/` from kubebuilder) — the default. Bases + overlays for CRDs, RBAC, manager
  Deployment, webhook config, cert-manager. `kubectl apply -k`. Simple, GitOps-friendly, no extra
  runtime. **Good default for self-managed installs.**
- **Helm chart** — convenient distribution + values-based config; you own upgrade ordering (CRDs are
  not templated/upgraded by Helm the way other resources are — keep CRDs in `crds/` or a separate
  release and manage their migration explicitly).

### OLM (Operator Lifecycle Manager)

OLM is the lifecycle manager for operators on OpenShift and any cluster that installs it. It handles
install, dependency resolution, RBAC, and **automated upgrades along an update graph**. Core objects:

- **ClusterServiceVersion (CSV)** — the operator's "package manifest": metadata, install strategy
  (Deployment + RBAC + service accounts), owned/required CRDs, install modes (OwnNamespace,
  SingleNamespace, AllNamespaces), and the **upgrade graph** (`replaces` / `skips` / channels).
- **Bundle** — an immutable container image holding one CSV + its CRDs + metadata for **one version**.
- **Catalog (CatalogSource)** — an index image aggregating many bundles; OLM resolves installs/upgrades
  from it. **OperatorHub.io** is the public catalog.
- **Subscription** — user's declaration "install operator X from channel Y, auto/manual upgrades."

When to use OLM: you publish to **OperatorHub / OpenShift**, or you want managed, graph-driven upgrades
and dependency resolution across many operators. It's heavier than Kustomize; for a single internal
operator, Kustomize/Helm is often enough. (Note OLM has a v1 line modernizing this model — **verify the
current OLM version and its API** for your target.)

### The Operator must own its own upgrade

This is the hard, distinguishing part of **Full Lifecycle**. Upgrading the operator may require:
- **CRD schema migration** — new served/storage version + conversion webhook + storage migration of
  existing objects (Section 4). Never ship a new schema without a migration path for in-flight objects.
- **Managed-app migration** — e.g. a database engine version bump done online, in the right order,
  with rollback. The CR's `spec.version` change should drive a safe, observable upgrade with status.
- **Backward compat across the rolling operator upgrade** — old and new operator pods may briefly
  coexist; the new one must tolerate objects written by the old one and vice-versa.

An operator that can't upgrade itself or migrate its CRs is a level-1 toy regardless of feature count.

---

## 8. Operability

### Status conditions — the user/automation contract

`status.conditions` is how users, `kubectl wait`, GitOps tools, and other controllers learn whether the
resource is healthy. Use the standard **`metav1.Condition`** shape:
`type`, `status` (`True`/`False`/`Unknown`), `reason` (CamelCase machine token), `message` (human),
`observedGeneration`, `lastTransitionTime`. Use `meta.SetStatusCondition` to upsert.

- Adopt conventional condition types: **`Ready`** (overall), and where useful `Available`,
  `Progressing`, `Degraded`, `Reconciling`. Keep them orthogonal and documented.
- Set `observedGeneration` so consumers can tell whether the condition reflects the **current** spec.
- Make conditions **abnormal-true or abnormal-false consistently** and document the polarity. Prefer
  positive-polarity (`Ready=True` is good).
- Pair status with a coarse `phase`/`state` only if users want a one-word summary in printer columns —
  conditions remain the source of truth; don't build a brittle phase state-machine as the API.

### Events

Emit Kubernetes **Events** (`record.EventRecorder`) on meaningful transitions (Created, Scaled,
UpgradeStarted, Failed). Events are the timeline users see in `kubectl describe`. Don't spam — event
per reconcile is noise; event per state change is signal.

### Metrics

controller-runtime exposes a metrics endpoint with reconcile counts, latency, queue depth, and work
errors out of the box. Add **domain metrics** (managed clusters by phase, last-backup age, drift count)
as Prometheus metrics — this is most of capability level 4 (Deep Insights). Wire alerts off them.

### Scope, multi-tenancy, RBAC

- **CRD scope**: namespace-scoped for tenant-owned resources (the common case); cluster-scoped for
  infrastructure that genuinely spans namespaces. This is fixed at CRD creation — changing it later is
  a breaking migration.
- **Multi-tenancy**: decide between **one operator watching all namespaces** (simple, but a noisy/
  malicious tenant can affect the shared controller) vs **per-namespace operators** (isolation, more
  moving parts). Scope the manager's cache/namespaces accordingly.
- **RBAC least privilege**: generate from `+kubebuilder:rbac` markers; grant only the groups/resources/
  verbs you use. Prefer **Role** (namespaced) over **ClusterRole** where the operator is namespace-
  scoped. Audit the generated `role.yaml` — markers tend to over-grant if copy-pasted.

### Leader election & multiple replicas

- Run the controller **`Deployment` with ≥2 replicas** and **leader election enabled** (controller-
  runtime `Manager{ LeaderElection: true }` via a `Lease`). Exactly one replica is the active leader
  reconciling; the rest stand by for **fast failover**. This is HA, not horizontal scale — work isn't
  sharded across replicas by default.
- Webhook serving, by contrast, is served by **all** replicas (it's stateless), giving the webhook HA
  independent of leader election.
- Set resource requests/limits, a PDB, and graceful shutdown so the leader hands off cleanly on rollout.

---

## 9. Anti-patterns (reject these)

- **God-CRD**: one giant kind modeling the whole platform with dozens of loosely-related fields. Split
  into focused kinds; compose via references/owner refs.
- **Leaking implementation into the API**: exposing internal phases, pod names, reconcile bookkeeping,
  or your data structures as spec/status fields. The API is user intent, not your internals.
- **Status in spec / user-writable status**: breaks the declarative model and confuses GitOps.
- **Imperative fields**: `spec.action: restart`/RPC-style verbs instead of declarative state. If you
  must trigger one-shot ops, model them as their own short-lived CRs or observed-and-cleared annotations.
- **Breaking a GA API**: removing/repurposing a field, tightening validation, changing defaults. Never.
- **No structural schema / preserve-unknown-fields at root**: loses pruning, CEL, defaulting, conversion.
- **Webhook that bricks the cluster**: `failurePolicy: Fail`, single replica, no kube-system/operator-
  namespace exclusion, slow, or with self-signed certs nobody rotates.
- **Validating *everything* in a webhook**: when CEL schema rules or a `ValidatingAdmissionPolicy` would
  do it in-apiserver with zero availability risk.
- **Ignoring status/conditions**: an operator that reconciles silently is unobservable and un-GitOps-able.
- **Multiple `storage: true` versions / no conversion plan**: data-loss / corruption on version skew.
- **Hand-editing generated CRD/RBAC/deepcopy**: it'll be clobbered; edit markers and regenerate.
- **An operator that can't upgrade itself or migrate its CRs**: caps you at capability level 1.
- **God-mode RBAC**: `cluster-admin`-equivalent ClusterRoles "to be safe." Scope to what you touch.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `kubectl apply` rejected with schema error after upgrade | tightened validation on existing version | loosen, or move the constraint to a new version; never break served versions |
| CR created but unknown fields silently dropped | pruning + missing field in structural schema | add the field to the schema; check for `preserve-unknown-fields` misuse |
| All writes to a kind hang/503 after deploying a webhook | `failurePolicy: Fail` + webhook down/unreachable/cert bad | check webhook pod, certs/`caBundle`, selectors; temporarily set `Ignore` / delete the webhook config to recover |
| Cluster control-plane components fail to schedule | webhook intercepting kube-system | add namespaceSelector excluding control-plane/kube-system |
| Can't drop an old CRD version | objects still stored at it | migrate stored objects (re-write / storage-version-migrator), then remove |
| `status.observedGeneration` never catches up | reconciler not setting it, or erroring before status write | set it after successful reconcile; ensure status subresource enabled |
| Conversion errors on cross-version read | conversion webhook down / lossy / cert bad | verify hub-and-spoke impl, round-trip tests, webhook health and `caBundle` |
| HPA can't scale the CR | `/scale` not enabled or selector path wrong | enable subresource with correct spec/status/selector paths |
| Two operator pods both reconciling (double writes) | leader election off/misconfigured | enable leader election; verify the `Lease` exists and is held by one pod |
| Generated CRD differs from committed YAML in CI | stale generation | run `make generate manifests`, commit; fail CI on `git diff` |

Diagnostic commands: `kubectl get crd <plural>.<group> -o yaml` (versions, storage flag, conversion),
`kubectl get validatingwebhookconfiguration/mutatingwebhookconfiguration -o yaml` (selectors,
failurePolicy, caBundle), `kubectl explain <kind>.spec` (served schema), controller logs + the
controller-runtime metrics endpoint, and Events via `kubectl describe <cr>`.

---

## 11. Version awareness

- Kubernetes minor releases (~3/yr) change CRD/admission feature maturity. **CEL in CRDs**,
  **`ValidatingAdmissionPolicy`**, and structural-schema requirements have matured over recent
  releases; the **mutating** CEL policy line is newer — **verify the exact maturity/gate for your
  cluster version** rather than assuming GA.
- kubebuilder, Operator SDK, controller-gen, controller-runtime, cert-manager, and OLM all version
  independently. Marker syntax and `config/` layout shift across kubebuilder majors; pin the
  controller-gen version and regenerate consistently. **OLM has a v1 line** reworking the CSV/catalog
  model — confirm which OLM you target.
- When unsure whether a marker, field, subresource path, or feature is available, **say "verify against
  current docs"** and keep the guidance general. Do not invent markers, fields, or version numbers.

---

## Rationalizations & rebuttals

The excuses that precede a bricked cluster or an un-migratable API. Each is a stop sign.

- **"I'll add the validating webhook later — CEL is too limited for this."** Most of what you'd webhook
  is expressible in schema CEL (`x-kubernetes-validations`, including immutability via `oldSelf`) or a
  `ValidatingAdmissionPolicy`, with zero certs/availability risk. Reach for those first; only the
  genuinely cross-resource/external rule needs a webhook — and that webhook is its own operational
  burden, not a freebie you defer.
- **"It's still v1beta1-ish, breaking the GA shape is fine."** If you served it as `v1`, it's GA and
  forever — no removed/repurposed fields, no tightened validation, no changed defaults. "Beta-ish" is
  not a version; the apiserver and your users go by the served version string, not your intentions.
- **"One god-CRD is simpler than five kinds."** It's simpler to *scaffold* and worse to *own*: coupled
  reconcile paths, unsplittable RBAC, validation that can't express "these fields only apply when…".
  Split into focused kinds composed by references/owner refs.
- **"I'll just leak the pod names / reconcile phase into the API — it's convenient for debugging."**
  Convenient now, a frozen contract forever. Internal detail in spec/status means renaming your
  controller or changing your data model becomes a breaking API change. Put debug detail in
  logs/events/metrics, not the CRD.
- **"Two versions, both `storage: true` — saves me writing conversion."** Exactly one version stores in
  etcd; a second storage flag (or no conversion plan for differently-shaped versions) is data loss on
  version skew. Pick one storage version, implement hub-and-spoke conversion, migrate stored objects.
- **"`failurePolicy: Ignore` so the webhook can't ever block writes."** Then your "validating" webhook
  silently no-ops whenever it's down — invalid objects sail into etcd. The real fix isn't `Ignore`;
  it's a webhook that's HA, fast, cert-rotated, and scoped to exclude the namespaces it needs to boot.
- **"cluster-admin RBAC for now, I'll tighten it before release."** It never gets tightened, and the
  generated `role.yaml` from copy-pasted `+kubebuilder:rbac` markers already over-grants. Scope to the
  groups/resources/verbs you actually touch from day one; prefer `Role` over `ClusterRole`.
- **"I'll cut a new schema this release and worry about in-flight objects later."** Existing CRs are
  stored at the old version; without a served new version + conversion + storage migration, you can't
  drop the old one and old/new operator pods clash during the rolling upgrade. No schema change ships
  without a migration path.

## Red flags — stop and reconsider

- A webhook with `failurePolicy: Fail` whose `namespaceSelector` does **not** exclude `kube-system` and
  the operator's own namespace — it can deadlock the control plane or itself on cold start.
- A single webhook replica (no ≥2 replicas / PDB / anti-affinity) gating writes cluster-wide.
- Serving-cert lifetime managed by hand (no cert-manager CA-injector or built-in rotator) — expiry
  becomes a cluster outage with `Fail`.
- Two CRD versions with different shapes and `conversion.strategy: None`, or more than one version
  marked `storage: true`.
- A new served/storage version shipped with no storage-version migration of existing objects.
- Internal state in the API — pod names, reconcile phase bookkeeping, your data structures — in spec or
  status; or user-writable `status` / status-in-spec.
- No `status.conditions` (or conditions without `observedGeneration`): the resource is unobservable,
  `kubectl wait`-incompatible, and not GitOps-able.
- `spec.action: restart` / RPC-style verbs, an `x-kubernetes-preserve-unknown-fields` at the schema
  root, or hand-edited generated CRD/RBAC/deepcopy.
- `cluster-admin`-equivalent ClusterRole "to be safe," and `/scale` enabled on a kind that isn't
  replica-scalable.

## Verification gate (definition of done)

Before the operator/CRD change counts as done, confirm:

- [ ] **CRD schema** — structural OpenAPI v3 schema, every field typed; no root
      `x-kubernetes-preserve-unknown-fields`; required/optional and defaults set; cross-field/immutability
      rules in CEL (`x-kubernetes-validations`) where possible rather than a webhook.
- [ ] **Subresources** — `/status` enabled; `/scale` enabled **only** if replica-scalable, with correct
      spec/status/selector paths.
- [ ] **API versioning** — exactly one `storage: true` version; for differently-shaped served versions a
      hub-and-spoke conversion webhook with round-trip tests; existing stored objects migrated before any
      old version is dropped; no GA-breaking change (no removed/repurposed fields, no tightened validation,
      no changed defaults).
- [ ] **Webhooks** — `sideEffects: None`, `admissionReviewVersions: ["v1"]`, `matchPolicy: Equivalent`,
      short `timeoutSeconds`; `failurePolicy` chosen deliberately with selectors excluding `kube-system`
      and the operator namespace; ≥2 replicas + PDB; certs issued and `caBundle` injected/rotated
      automatically (cert-manager or built-in rotator) — no hand-managed certs.
- [ ] **RBAC** — least privilege; generated from `+kubebuilder:rbac` markers and the `role.yaml` audited;
      `Role` over `ClusterRole` where the operator is namespace-scoped; no `cluster-admin`.
- [ ] **Operability** — standard `metav1.Condition` status conditions (incl. `Ready`) with
      `observedGeneration`; Events on meaningful transitions; domain metrics exposed.
- [ ] **HA** — controller `Deployment` ≥2 replicas with leader election (`Lease`); requests/limits, PDB,
      graceful shutdown.
- [ ] **Upgrade path** — operator can upgrade itself and migrate its CRs; old/new operator pods tolerate
      each other's objects during the rolling upgrade.
- [ ] **Generation in CI** — `make generate manifests` run and committed; CI fails on
      `git diff --exit-code` for generated CRD/RBAC/deepcopy.
- [ ] **Version claims verified** — any feature-gate / API-maturity assumption (CEL policies, OLM v1,
      controller-gen markers) checked against the docs for the target cluster version, not assumed GA.

---

## 12. Canonical references

- Operator pattern (concept) — https://kubernetes.io/docs/concepts/extend-kubernetes/operator/
- Custom Resources / CRDs — https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/
- CRD versioning & conversion — https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definition-versioning/
- CRD validation (incl. CEL `x-kubernetes-validations`) — https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#validation
- Validating Admission Policy (CEL) — https://kubernetes.io/docs/reference/access-authn-authz/validating-admission-policy/
- Dynamic admission control (webhooks) — https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/
- Kubernetes API conventions (spec/status, conditions) — https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md
- kubebuilder book — https://book.kubebuilder.io/
- Operator SDK — https://sdk.operatorframework.io/
- Operator Lifecycle Manager — https://olm.operatorframework.io/
- OperatorHub — https://operatorhub.io/
- controller-gen markers reference — https://book.kubebuilder.io/reference/markers
- cert-manager (webhook/CA injection) — https://cert-manager.io/docs/
- Operator capability levels — https://sdk.operatorframework.io/docs/overview/operator-capabilities/
- Reconcile-loop mechanics — see `[[kubernetes-controller-expert]]`.
