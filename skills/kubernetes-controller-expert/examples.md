# Kubernetes Controller Examples

Canonical, imitate-this artifacts for a controller-runtime controller. Go idiom follows
**[[go-best-practices]]**; controller semantics follow `kubernetes-controller-expert-guide.md`.

These compile in spirit against a recent controller-runtime; **verify exact imports/signatures against
the version in your `go.mod`** (the framework moves with each Kubernetes minor). Import paths and the
`cache.Options`/metrics defaults are the most version-sensitive parts.

Assume an API type `Foo` (group `apps.example.com/v1`) with a `.spec.replicas` and `.spec.image`, a
status subresource, and these kubebuilder markers on the type:

```go
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`
```

---

## 1. A canonical Reconciler

Demonstrates: fetch-or-return-on-NotFound, finalizer add/remove with idempotent external cleanup,
desired-state construction, owner references, **server-side apply**, observation, and a status update
with `observedGeneration` + `metav1.Condition` written only when changed.

```go
package controller

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"github.com/go-logr/logr"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	appv1 "example.com/foo/api/v1"
)

const (
	finalizer  = "apps.example.com/cleanup"
	fieldOwner = client.FieldOwner("foo-controller")
)

// FooReconciler reconciles a Foo object by managing a backing Deployment.
type FooReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// RBAC — keep these in sync with the verbs the code actually uses.
// +kubebuilder:rbac:groups=apps.example.com,resources=foos,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=apps.example.com,resources=foos/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=apps.example.com,resources=foos/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

func (r *FooReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// 1. Fetch the primary object from the cache. Gone? Nothing to do — GC handles children.
	var foo appv1.Foo
	if err := r.Get(ctx, req.NamespacedName, &foo); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// 2. Deletion path: run idempotent external cleanup, then drop the finalizer.
	if !foo.DeletionTimestamp.IsZero() {
		if controllerutil.ContainsFinalizer(&foo, finalizer) {
			if err := r.deleteExternalResources(ctx, &foo); err != nil {
				return ctrl.Result{}, fmt.Errorf("external cleanup: %w", err) // retry w/ backoff
			}
			controllerutil.RemoveFinalizer(&foo, finalizer)
			if err := r.Update(ctx, &foo); err != nil {
				return ctrl.Result{}, fmt.Errorf("remove finalizer: %w", err)
			}
		}
		return ctrl.Result{}, nil // object is now deletable
	}

	// 3. Ensure our finalizer is present BEFORE creating external resources.
	if !controllerutil.ContainsFinalizer(&foo, finalizer) {
		controllerutil.AddFinalizer(&foo, finalizer)
		if err := r.Update(ctx, &foo); err != nil {
			return ctrl.Result{}, fmt.Errorf("add finalizer: %w", err)
		}
		return ctrl.Result{}, nil // requeue with the finalizer persisted
	}

	// 4. Build DESIRED state purely from spec, stamp the owner reference.
	desired := desiredDeployment(&foo)
	if err := controllerutil.SetControllerReference(&foo, desired, r.Scheme); err != nil {
		return ctrl.Result{}, fmt.Errorf("set owner ref: %w", err)
	}

	// 5. Apply desired → observed via Server-Side Apply: idempotent, no read, conflict-free.
	if err := r.Patch(ctx, desired, client.Apply, fieldOwner, client.ForceOwnership); err != nil {
		return ctrl.Result{}, fmt.Errorf("apply deployment: %w", err)
	}

	// 6. Observe the child to compute status. Not in cache yet → requeue and re-observe.
	var dep appsv1.Deployment
	if err := r.Get(ctx, client.ObjectKeyFromObject(desired), &dep); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// 7. Status: observedGeneration + Ready condition, written ONLY if it changed.
	return r.reconcileStatus(ctx, &foo, &dep, logger)
}

func (r *FooReconciler) reconcileStatus(
	ctx context.Context, foo *appv1.Foo, dep *appsv1.Deployment, logger logr.Logger,
) (ctrl.Result, error) {
	base := foo.DeepCopy() // compare against this to avoid no-op status writes

	ready := dep.Status.ReadyReplicas == *foo.Spec.Replicas
	cond := metav1.Condition{
		Type:               "Ready",
		ObservedGeneration: foo.Generation,
		Status:             metav1.ConditionFalse,
		Reason:             "DeploymentProgressing",
		Message:            fmt.Sprintf("%d/%d replicas ready", dep.Status.ReadyReplicas, *foo.Spec.Replicas),
	}
	if ready {
		cond.Status = metav1.ConditionTrue
		cond.Reason = "DeploymentAvailable"
		cond.Message = "all replicas ready"
	}
	foo.Status.ObservedGeneration = foo.Generation
	foo.Status.ReadyReplicas = dep.Status.ReadyReplicas
	meta.SetStatusCondition(&foo.Status.Conditions, cond)

	if apiequality.Semantic.DeepEqual(base.Status, foo.Status) {
		return ctrl.Result{}, nil // nothing changed — DON'T write (avoids a self-triggered watch)
	}
	if err := r.Status().Update(ctx, foo); err != nil {
		if apierrors.IsConflict(err) {
			return ctrl.Result{}, nil // someone else won; we'll be re-enqueued
		}
		return ctrl.Result{}, fmt.Errorf("update status: %w", err)
	}
	return ctrl.Result{}, nil
}

// desiredDeployment is a PURE function of spec — easy to unit-test, deterministic name for idempotency.
func desiredDeployment(foo *appv1.Foo) *appsv1.Deployment {
	labels := map[string]string{"app.kubernetes.io/managed-by": "foo-controller", "foo": foo.Name}
	return &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"}, // required for SSA
		ObjectMeta: metav1.ObjectMeta{
			Name:      foo.Name,
			Namespace: foo.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: foo.Spec.Replicas,
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "app",
						Image: foo.Spec.Image,
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("100m"),
								corev1.ResourceMemory: resource.MustParse("128Mi"),
							},
						},
					}},
				},
			},
		},
	}
}

// deleteExternalResources MUST be idempotent and tolerate "already gone".
func (r *FooReconciler) deleteExternalResources(ctx context.Context, foo *appv1.Foo) error {
	// e.g. delete a cloud LB / DNS record keyed deterministically off foo. Deleting twice, or when it
	// was never created, must succeed. The Deployment itself is GC'd via the owner ref — not here.
	_ = ctx
	_ = foo
	return nil
}

// SetupWithManager wires the controller: reconcile Foo, own its Deployment, skip status-only updates.
func (r *FooReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&appv1.Foo{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.Deployment{}). // Deployment events → owner Foo via the controller ownerRef
		WithOptions(controller.Options{MaxConcurrentReconciles: 4}).
		Named("foo").
		Complete(r)
}
```

Notes on what makes this correct:

- **`For(...).WithPredicates(GenerationChangedPredicate{})`** stops the controller's *own* status writes
  (which don't bump `.metadata.generation`) from re-enqueuing it — the canonical hot-loop fix.
- **SSA (`client.Apply`)** needs `TypeMeta` populated and a stable `FieldOwner`; it requires no prior
  read and is conflict-free for the fields this controller owns.
- **Status written only when changed**, via the **subresource**, with `ObservedGeneration` set on both
  the status and the condition.
- **Finalizer added before** any external resource exists; cleanup idempotent; finalizer **always**
  removed on success → no stuck `Terminating`.
- `desiredDeployment` is a **pure function** → trivially table-testable without a cluster.

---

## 2. A minimal `main.go` wiring a Manager

Demonstrates: scheme registration, leader election (HA), graceful-shutdown signal context, health/ready
probes, metrics, structured logging.

```go
package main

import (
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	appv1 "example.com/foo/api/v1"
	"example.com/foo/internal/controller"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme)) // built-in types (Deployment, etc.)
	utilruntime.Must(appv1.AddToScheme(scheme))          // our CRD types
}

func main() {
	var metricsAddr, probeAddr string
	var enableLeaderElection bool
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8443", "metrics endpoint")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "health probe endpoint")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false, "enable leader election for HA")
	opts := zap.Options{Development: false}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts))) // structured logr backend, set ONCE

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsserver.Options{BindAddress: metricsAddr},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "foo-controller.apps.example.com", // stable, unique per controller
		// SyncPeriod / Cache scoping go here. Default cache is cluster-wide; scope it for scale:
		//   Cache: cache.Options{DefaultNamespaces: map[string]cache.Config{"my-ns": {}}},
		// (verify cache.Options shape for your controller-runtime version)
	})
	if err != nil {
		setupLog.Error(err, "unable to create manager")
		os.Exit(1)
	}

	if err := (&controller.FooReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to set up controller", "controller", "Foo")
		os.Exit(1)
	}

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	// SetupSignalHandler returns a ctx cancelled on SIGTERM/SIGINT → graceful drain + lease release.
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
```

Notes:

- **`ctrl.GetConfigOrDie()`** loads in-cluster config (or `$KUBECONFIG` locally).
- **Leader election** ON in production: only the leader reconciles; replicas stand by for fast
  failover. Without it, every replica reconciles and they **fight**.
- **`mgr.Start(ctrl.SetupSignalHandler())`** blocks until SIGTERM; the Manager then stops sources, lets
  in-flight reconciles finish, and releases the leader `Lease`. Never `os.Exit` out from under it.
- The metrics bind address default has shifted across controller-runtime versions (and recent versions
  protect it with auth) — **confirm the default and protection for your version** before exposing it.

---

## 3. Watching a non-owned related object (mapping function)

When `Foo` references a `ConfigMap` it does **not** own, watch the ConfigMap and map its events back to
every `Foo` that references it — using a **field index** so the lookup is cheap, not a full LIST+filter.

```go
// In SetupWithManager: register an index on the referenced ConfigMap name, once.
const cmRefField = ".spec.configMapRef"

if err := mgr.GetFieldIndexer().IndexField(ctx, &appv1.Foo{}, cmRefField,
	func(o client.Object) []string {
		ref := o.(*appv1.Foo).Spec.ConfigMapRef
		if ref == "" {
			return nil
		}
		return []string{ref}
	}); err != nil {
	return err
}

// Then on the builder:
//   .Watches(&corev1.ConfigMap{}, handler.EnqueueRequestsFromMapFunc(r.foosForConfigMap))

// foosForConfigMap maps one ConfigMap event → the Foos that reference it. Cache-only, bounded.
func (r *FooReconciler) foosForConfigMap(ctx context.Context, cm client.Object) []ctrl.Request {
	var foos appv1.FooList
	if err := r.List(ctx, &foos,
		client.InNamespace(cm.GetNamespace()),
		client.MatchingFields{cmRefField: cm.GetName()},
	); err != nil {
		return nil
	}
	reqs := make([]ctrl.Request, 0, len(foos.Items))
	for i := range foos.Items {
		reqs = append(reqs, ctrl.Request{
			NamespacedName: client.ObjectKeyFromObject(&foos.Items[i]),
		})
	}
	return reqs
}
```

This is the pattern Kueue, JobSet, and LeaderWorkerSet use heavily — see [[kueue-advanced]] and
[[jobset-leaderworkerset]] for production-scale examples of indexed watches and mapping functions.
