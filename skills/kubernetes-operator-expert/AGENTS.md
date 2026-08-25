# AGENTS.md — Kubernetes Operator Engineering

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference is **`kubernetes-operator-expert-guide.md`** next to this file —
> read it before designing a CRD, webhook, version migration, or packaging change, and apply it.
> Concrete artifacts to imitate (annotated CRD, kubebuilder type with markers, validating-webhook
> skeleton) are in **`examples.md`**. This file is the always-on summary.
>
> **Scope:** the operator / API / packaging / lifecycle layer. The reconcile loop itself (controller-
> runtime, watches, workqueues, status patching) lives in `[[kubernetes-controller-expert]]` — defer
> there for the loop. It is 2026; verify feature maturity/version claims against current docs and
> **never invent a marker, field, subresource path, or version number**.

## When designing/building/shipping an Operator, apply by default:

- **Operator only for real day-2 operational logic** (failover, backup/restore, in-place upgrade,
  reconciling external state). If a Helm chart or a few manifests express it, don't build an Operator.
  Target a deliberate **capability level** (Basic Install → Seamless Upgrades → Full Lifecycle → Deep
  Insights → Auto Pilot); don't build Auto Pilot you can't operate.
- **The CRD is a public API.** Declarative, minimal, forward-compatible. `spec` = user intent;
  `status` = controller-owned observed state. Never put status in spec, accept user-writable status, or
  leak implementation/internal phases/pod names into the API. Add fields later; you can't remove them.
- **Structural OpenAPI v3 schema, always.** No root `x-kubernetes-preserve-unknown-fields`. It's the
  prerequisite for pruning, defaulting, CEL, and conversion. Validate with **CEL
  `x-kubernetes-validations`** in the schema first; immutability via `self == oldSelf`.
- **Enable `/status`** (separate spec/status updates; `observedGeneration` tracks catch-up). Add
  `/scale` only if genuinely replica-scalable. Add printer columns, short names, categories for UX.
- **Status conditions are the contract.** Use `metav1.Condition` (type/status/reason/message/
  observedGeneration/lastTransitionTime); set `Ready` (+ `Available`/`Progressing`/`Degraded`).
  Emit Events on transitions. Surface domain metrics (Prometheus) for Deep Insights.
- **Never break a GA API.** alpha (may break) → beta (deprecation policy) → GA (forever). Add, don't
  remove/repurpose; don't tighten validation or change defaults on a served version.
- **Versioning:** serve many versions, **store exactly one** (`storage: true`). New incompatible shape
  ⇒ new version + **conversion webhook**, hub-and-spoke (one hub = storage version; spokes
  `ConvertTo`/`ConvertFrom` the hub; round-trip lossless). Migrate stored objects before dropping a version.
- **Webhooks can brick a cluster.** Set `failurePolicy` deliberately (`Fail` blocks writes when the
  webhook is down); scope with selectors to **exclude kube-system and the operator's own namespace**;
  `sideEffects: None`; short timeout; idempotent mutating webhooks (re-invocation); ≥2 replicas + PDB.
  Manage certs with **cert-manager** (CA injection) or controller-runtime's rotator — never hand-rolled.
- **Prefer CEL `ValidatingAdmissionPolicy`** (in-apiserver, no webhook server/certs) over a validating
  webhook whenever the check is expressible in CEL. Webhooks only for cross-resource/external/mutation
  logic the schema and policies can't express.
- **controller-gen markers are the source of truth.** `// +kubebuilder:...` generate CRD+RBAC+webhook
  manifests and deepcopy; **never hand-edit generated YAML/code** — edit the marker, `make generate
  manifests`, and fail CI on `git diff` if stale. kubebuilder/Operator SDK for Go; Ansible/Helm
  operators only for chart-with-a-CR-face, no custom logic.
- **Least-privilege RBAC** from markers (Role over ClusterRole when namespace-scoped; grant only
  groups/resources/verbs used). **Leader election on, ≥2 replicas** (HA + fast failover, not sharding;
  webhook serving is on all replicas). Choose CRD **scope** (namespace vs cluster) and **multi-tenancy**
  (one operator-all-namespaces vs per-namespace) deliberately — scope is fixed at CRD creation.
- **Package for the audience:** Kustomize (`config/`) or Helm for self-managed; **OLM** (CSV + bundle +
  catalog) for OperatorHub/OpenShift and graph-driven managed upgrades. The Operator must **own its own
  upgrade** including CRD schema migration + managed-app migration — that's the hard part of Full Lifecycle.

## Reject these anti-patterns
God-CRDs; implementation/internals leaked into the API; imperative `action:` fields; user-writable
status; breaking a GA API; missing structural schema; webhooks with `Fail` + single replica + no
kube-system/self exclusion; validating in a webhook what CEL could do in-apiserver; ignoring status;
multiple `storage: true` / no conversion plan; hand-edited generated files; an operator that can't
upgrade/migrate itself; `cluster-admin`-grade RBAC "to be safe."

## Definition of done for Operator changes
Generated CRD/RBAC/webhook/deepcopy regenerated and committed (CI green on `git diff` after
`make generate manifests`); `/status` enabled and conditions set with `observedGeneration`; new/changed
API version has a tested conversion path and migration plan; webhooks scoped to exclude control-plane,
`sideEffects: None`, certs automated; RBAC least-privilege; leader election + ≥2 replicas. Apply
`[[go-best-practices]]` to the Go. For the reconcile loop, see `[[kubernetes-controller-expert]]`.
