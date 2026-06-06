# Operator Examples (imitate these)

Canonical, correct-in-spirit artifacts for the API/webhook/packaging layer. Verify exact marker spelling
and feature maturity against the kubebuilder/controller-gen and Kubernetes versions you target — never
copy a marker or field you haven't confirmed exists. Reconcile-loop code lives in
`[[kubernetes-controller-expert]]`.

---

## 1. Annotated CRD — `/status` subresource, CEL validation, printer columns

A hand-written CRD (in practice you'd *generate* this from Go markers, Section 2). Shown to illustrate
every load-bearing field: structural schema, `default`, CEL `x-kubernetes-validations` (cross-field +
immutability), the `status` subresource, conditions, and printer columns.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: memcachedclusters.cache.example.com   # <plural>.<group>
spec:
  group: cache.example.com
  names:
    kind: MemcachedCluster
    listKind: MemcachedClusterList
    singular: memcachedcluster
    plural: memcachedclusters
    shortNames: [mc]
    categories: [all]
  scope: Namespaced            # tenant-owned resource → namespaced. Fixed at creation; changing = breaking.
  versions:
    - name: v1
      served: true
      storage: true            # exactly ONE version may be the storage version
      subresources:
        status: {}             # /status: spec & status update independently; status writes don't bump generation
        # scale:               # add ONLY if genuinely replica-scalable (lets kubectl scale + HPA target it)
        #   specReplicasPath: .spec.size
        #   statusReplicasPath: .status.replicas
        #   labelSelectorPath: .status.selector
      additionalPrinterColumns:
        - name: Size
          type: integer
          jsonPath: .spec.size
        - name: Ready
          type: string
          jsonPath: .status.conditions[?(@.type=="Ready")].status
        - name: Version
          type: string
          jsonPath: .spec.version
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:        # structural schema: every field typed, no preserve-unknown at root
          type: object
          required: [spec]
          properties:
            spec:
              type: object
              required: [size]
              # cross-field CEL: applies to the whole spec object
              x-kubernetes-validations:
                - rule: "self.size <= self.maxSize"
                  message: "size must not exceed maxSize"
              properties:
                size:
                  type: integer
                  minimum: 1
                  maximum: 9
                  default: 3
                maxSize:
                  type: integer
                  minimum: 1
                  default: 9
                version:
                  type: string
                  pattern: '^[0-9]+\.[0-9]+$'
                  default: "1.6"
                storageClass:
                  type: string
                  # field-level immutability: reject any change after creation (replaces a webhook)
                  x-kubernetes-validations:
                    - rule: "self == oldSelf"
                      message: "storageClass is immutable"
            status:
              type: object
              properties:
                replicas:
                  type: integer
                observedGeneration:        # lets consumers tell if status reflects the CURRENT spec
                  type: integer
                  format: int64
                conditions:                # the user/automation contract — standard metav1.Condition shape
                  type: array
                  items:
                    type: object
                    required: [type, status, reason, lastTransitionTime]
                    properties:
                      type:    { type: string }
                      status:  { type: string, enum: ["True", "False", "Unknown"] }
                      reason:  { type: string }
                      message: { type: string }
                      observedGeneration: { type: integer, format: int64 }
                      lastTransitionTime: { type: string, format: date-time }
```

```console
$ kubectl get mc
NAME      SIZE   READY   VERSION   AGE
cache-a   3      True    1.6       5m
```

---

## 2. kubebuilder Go type with markers (the source of truth)

This is what you actually write; `controller-gen` (`make generate manifests`) emits the CRD above, the
deepcopy methods, the RBAC, and the webhook config from these markers. **Never hand-edit the generated
files** — change the marker and regenerate. Apply `[[go-best-practices]]` to the surrounding code.

```go
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=mc,categories=all,path=memcachedclusters
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="Size",type=integer,JSONPath=`.spec.size`
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// MemcachedCluster is the Schema for the memcachedclusters API.
type MemcachedCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   MemcachedClusterSpec   `json:"spec,omitempty"`
	Status MemcachedClusterStatus `json:"status,omitempty"`
}

// MemcachedClusterSpec is the desired state — user intent only.
// +kubebuilder:validation:XValidation:rule="self.size <= self.maxSize",message="size must not exceed maxSize"
type MemcachedClusterSpec struct {
	// Size is the number of Memcached replicas.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=9
	// +kubebuilder:default=3
	Size int32 `json:"size"`

	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=9
	// +optional
	MaxSize int32 `json:"maxSize,omitempty"`

	// +kubebuilder:validation:Pattern=`^[0-9]+\.[0-9]+$`
	// +kubebuilder:default="1.6"
	// +optional
	Version string `json:"version,omitempty"`

	// StorageClass is immutable after creation.
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="storageClass is immutable"
	// +optional
	StorageClass string `json:"storageClass,omitempty"`
}

// MemcachedClusterStatus is observed state — written ONLY by the controller.
type MemcachedClusterStatus struct {
	// +optional
	Replicas int32 `json:"replicas,omitempty"`
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// Conditions follow the standard metav1.Condition contract.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}
```

Setting a condition in the reconciler (loop details in `[[kubernetes-controller-expert]]`):

```go
meta.SetStatusCondition(&mc.Status.Conditions, metav1.Condition{
	Type:               "Ready",
	Status:             metav1.ConditionTrue,
	Reason:             "AllReplicasReady",
	Message:            "all memcached replicas are available",
	ObservedGeneration: mc.Generation, // so consumers know status reflects THIS spec
})
mc.Status.ObservedGeneration = mc.Generation
if err := r.Status().Update(ctx, &mc); err != nil { /* requeue */ }
```

RBAC + webhook markers (placed on the reconciler / webhook types). Generated, least-privilege:

```go
// +kubebuilder:rbac:groups=cache.example.com,resources=memcachedclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=cache.example.com,resources=memcachedclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete

// +kubebuilder:webhook:path=/validate-cache-example-com-v1-memcachedcluster,mutating=false,failurePolicy=fail,sideEffects=None,groups=cache.example.com,resources=memcachedclusters,verbs=create;update,versions=v1,name=vmemcachedcluster.kb.io,admissionReviewVersions=v1
```

---

## 3. Validating webhook skeleton (controller-runtime)

Use a webhook only when CEL schema rules / `ValidatingAdmissionPolicy` can't express the check (e.g. a
cross-resource lookup). Most field/cross-field validation belongs in the schema (examples 1–2).

```go
// +kubebuilder:webhook:path=/validate-cache-example-com-v1-memcachedcluster,mutating=false,failurePolicy=fail,sideEffects=None,groups=cache.example.com,resources=memcachedclusters,verbs=create;update,versions=v1,name=vmemcachedcluster.kb.io,admissionReviewVersions=v1

type MemcachedClusterValidator struct {
	Client client.Client // for cross-resource checks the schema can't do
}

var _ webhook.CustomValidator = &MemcachedClusterValidator{}

func (v *MemcachedClusterValidator) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	mc, ok := obj.(*cachev1.MemcachedCluster)
	if !ok {
		return nil, fmt.Errorf("expected a MemcachedCluster, got %T", obj)
	}
	return v.validate(ctx, mc)
}

func (v *MemcachedClusterValidator) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	mc, ok := newObj.(*cachev1.MemcachedCluster)
	if !ok {
		return nil, fmt.Errorf("expected a MemcachedCluster, got %T", newObj)
	}
	// NB: field-level immutability is already enforced by CEL (self == oldSelf) in the schema —
	// keep webhook checks to things CEL can't do.
	return v.validate(ctx, mc)
}

func (v *MemcachedClusterValidator) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return nil, nil // no delete-time validation
}

// validate does a cross-resource check that CEL cannot express in-schema.
func (v *MemcachedClusterValidator) validate(ctx context.Context, mc *cachev1.MemcachedCluster) (admission.Warnings, error) {
	var sc storagev1.StorageClass
	if err := v.Client.Get(ctx, client.ObjectKey{Name: mc.Spec.StorageClass}, &sc); err != nil {
		return nil, apierrors.NewForbidden(
			schema.GroupResource{Group: "cache.example.com", Resource: "memcachedclusters"},
			mc.Name,
			fmt.Errorf("storageClass %q must exist: %w", mc.Spec.StorageClass, err),
		)
	}
	return nil, nil
}
```

Register it in `main`:

```go
if err := ctrl.NewWebhookManagedBy(mgr).
	For(&cachev1.MemcachedCluster{}).
	WithValidator(&MemcachedClusterValidator{Client: mgr.GetClient()}).
	Complete(); err != nil {
	setupLog.Error(err, "unable to create webhook", "webhook", "MemcachedCluster")
	os.Exit(1)
}
```

### Webhook config to NOT brick the cluster

The generated `ValidatingWebhookConfiguration` — note `failurePolicy`, `sideEffects`, the
namespace exclusion, short timeout, and cert-manager CA injection:

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: memcached-validating-webhook
  annotations:
    cert-manager.io/inject-ca-from: operator-system/serving-cert   # cert-manager populates caBundle
webhooks:
  - name: vmemcachedcluster.kb.io
    admissionReviewVersions: ["v1"]
    sideEffects: None                 # required for dry-run / server-side apply
    failurePolicy: Fail               # blocks writes if the webhook is down — see exclusions below
    matchPolicy: Equivalent           # still fires for requests via other served versions
    timeoutSeconds: 10                # every matching write waits on this — keep it short
    namespaceSelector:                # NEVER intercept control-plane / the operator's own namespace
      matchExpressions:
        - key: kubernetes.io/metadata.name
          operator: NotIn
          values: ["kube-system", "operator-system"]
    clientConfig:
      service:
        name: operator-webhook-service
        namespace: operator-system
        path: /validate-cache-example-com-v1-memcachedcluster
    rules:
      - apiGroups: ["cache.example.com"]
        apiVersions: ["v1"]
        operations: ["CREATE", "UPDATE"]
        resources: ["memcachedclusters"]
        scope: Namespaced
```

---

## 4. CEL `ValidatingAdmissionPolicy` — the webhook-less alternative

When the rule is pure CEL, prefer this over a validating webhook: it runs **in the apiserver** — no
server, no certs, no `failurePolicy`/deadlock/latency risk. (Use it for the cross-field checks that
don't fit the per-object schema, or to enforce policy across a kind cluster-wide.)

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: memcached-size-policy
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
      - apiGroups:   ["cache.example.com"]
        apiVersions: ["v1"]
        operations:  ["CREATE", "UPDATE"]
        resources:   ["memcachedclusters"]
  validations:
    - expression: "object.spec.size <= object.spec.maxSize"
      message: "size must not exceed maxSize"
---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: memcached-size-policy-binding
spec:
  policyName: memcached-size-policy
  validationActions: ["Deny"]
  # matchResources: scope to namespaces/objects as needed
```

---

## 5. Conversion: hub-and-spoke (sketch)

When `v1` and `v2` differ in shape, one version is the **hub** (the storage version); each spoke
converts to/from the hub, so you write N-1 conversions, not N². Wire `strategy: Webhook` on the CRD and
implement the controller-runtime interfaces. (Implement real, round-trippable conversions; round-trip
tests are mandatory.)

```go
// v2 is the hub (storage version): a no-op marker method.
func (*MemcachedCluster /* v2 */) Hub() {}

// v1 is a spoke: convert to/from the v2 hub.
func (src *MemcachedCluster /* v1 */) ConvertTo(dstRaw conversion.Hub) error {
	dst := dstRaw.(*v2.MemcachedCluster)
	dst.ObjectMeta = src.ObjectMeta
	dst.Spec.Size = src.Spec.Size
	// map renamed/restructured fields; stash v2-only fields in annotations on down-convert so
	// ConvertFrom can restore them losslessly.
	return nil
}

func (dst *MemcachedCluster /* v1 */) ConvertFrom(srcRaw conversion.Hub) error {
	src := srcRaw.(*v2.MemcachedCluster)
	dst.ObjectMeta = src.ObjectMeta
	dst.Spec.Size = src.Spec.Size
	return nil
}
```
