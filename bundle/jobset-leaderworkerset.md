---
name: jobset-leaderworkerset
description: Production mastery of JobSet (jobset.sigs.k8s.io/v1alpha2) and LeaderWorkerSet / LWS
  (leaderworkerset.x-k8s.io/v1) — the SIG-driven Kubernetes workload APIs for multi-host ML. Use when
  authoring or debugging multi-host distributed training or multi-host LLM inference manifests:
  replicatedJobs / replicas / startupPolicy / successPolicy / failurePolicy / coordinator / exclusive
  placement for JobSet; leaderWorkerTemplate / size / replicas / subGroupPolicy / rolloutStrategy /
  restartPolicy for LWS. Triggers on JobSet or LeaderWorkerSet CRDs, gang restart of a Job group,
  multi-node vLLM/SGLang serving (leader+workers, headless Service, predictable hostnames), TPU/GPU
  multi-host pods, "stuck group", "startup ordering", "pods can't resolve each other", and choosing
  JobSet vs LWS vs raw StatefulSet/indexed Job. Pairs with Kueue, GKE node pools, vLLM/SGLang.
---

# JobSet & LeaderWorkerSet (multi-host ML on Kubernetes)

Apply the judgment of an engineer who has run multi-host training and multi-host LLM inference in
production on GPU/TPU clusters for years. JobSet and LWS exist because raw Jobs/StatefulSets do not
model a **gang** — a set of pods that must start together, share a stable network identity, and
restart as a unit. Pick the right API, get the gang semantics right, and the rest is plumbing.

## How to use this skill

1. **Read `jobset-leaderworkerset-guide.md`** in this directory — the full reference (mental model,
   both APIs' spec fields, gang/restart semantics, networking, Kueue/GKE integration, troubleshooting).
   Apply it to the task.
2. For correct, annotated manifests to imitate — a multi-host training JobSet and a multi-host vLLM
   LWS — read **`examples.md`**.
3. Match the surrounding cluster's conventions (node-pool labels, scheduler, quota). Apply the gang
   correctness rules (start ordering, all-or-nothing restart, stable hostnames) regardless.
4. These APIs evolve fast (it is 2026). Treat exact field/enum names here as a strong prior, but
   verify against `kubectl explain jobset.spec` / `kubectl explain leaderworkerset.spec` and the
   project docs for the version installed in the cluster.

## Essentials (full detail in `jobset-leaderworkerset-guide.md`)

- **JobSet = a group of Jobs managed as one unit.** `spec.replicatedJobs[]` each have a `name`,
  `replicas`, and a `template` (a full `JobTemplateSpec`). Child Jobs are
  `<jobset>-<replicatedJob>-<jobIndex>`; pods get stable DNS hostnames via a headless Service when
  `spec.network.enableDNSHostnames: true`. Use it for distributed training and multi-host batch.
- **LWS = groups of (1 leader + N workers) as the unit of replication.** `spec.replicas` = number of
  groups; `spec.leaderWorkerTemplate.size` = pods per group (leader counts as 1).
  `leaderTemplate` + `workerTemplate` are separate pod templates. Built for multi-host inference where
  one model is sharded across hosts (vLLM/SGLang TP+PP across nodes).
- **Gang restart is the whole point.** JobSet `failurePolicy.restartStrategy: BlockingRecreate`
  tears down *all* child Jobs before recreating — true gang restart. LWS
  `restartPolicy: RecreateGroupOnPodRestart` recreates the entire group when any pod fails. Do not
  hand-roll this with bare Jobs.
- **Startup ordering.** JobSet `startupPolicy.startupPolicyOrder: InOrder` starts replicatedJobs
  sequentially in list order (e.g. driver/launcher before workers). LWS `startupPolicy: LeaderReady`
  makes the leader become Ready before workers start; `LeaderCreated` starts them together.
- **Success targeting.** JobSet `successPolicy.operator: All|Any` over
  `targetReplicatedJobs` — e.g. JobSet succeeds when the single `driver` replicatedJob completes,
  ignoring long-lived workers. Default (empty target) means all.
- **Scale LWS by groups, not pods.** It exposes a `scale` subresource on `spec.replicas`, so an HPA
  targets *number of model replicas (groups)*. Never autoscale `size` — that's the model's shard count.
- **Stable identity is load-bearing for ML.** Torch `MASTER_ADDR`, NCCL/`torchrun` rendezvous, and
  vLLM Ray head all need a fixed, resolvable hostname. Both APIs give you predictable hostnames + a
  headless Service. Validate DNS resolves before blaming the framework.
- **Exclusive placement** pins one replicatedJob (JobSet) or one group (LWS) per topology domain
  (node pool / rack / TPU slice) via an annotation — critical for TPU slices and tight-coupling.
- **Put a real Service in front of LWS** that selects only leader pods (`role: leader`) for inference
  traffic; the headless Service is for intra-group pod-to-pod, not for clients.
- **Queue both with [[kueue-advanced]].** JobSet and LWS both have native Kueue integration for gang
  admission and quota. JobSet itself uses owner refs + foreground deletion (no finalizer on the
  JobSet object); finalizers you see usually come from Kueue.
- **JobSet vs LWS vs StatefulSet/Job:** training/batch that runs to completion → JobSet; long-running
  multi-host *serving* → LWS; single-host serving → Deployment; single-host stateful → StatefulSet;
  single multi-pod batch → indexed Job. Don't force a StatefulSet to be a training gang.

## Related skills

- `[[kueue-advanced]]` — gang admission, quotas, MultiKueue, topology-aware scheduling for both APIs.
- `[[serving-frameworks]]` — vLLM/SGLang/Dynamo multi-node, what runs in the LWS leader vs workers.
- `[[training-frameworks]]` — FSDP/DeepSpeed/Megatron/torchrun/MaxText that run inside JobSet.
- `[[aiml-on-kubernetes]]` — umbrella: where these fit in the training/inference stack.
- `[[gke-master]]` — GPU/TPU node pools, multi-host TPU slices, `localQueue`/placement on GKE.
- `[[autoscaling-kubernetes]]` — HPA on the LWS `scale` subresource; cluster autoscaler/NAP for gangs.

---

# Reference — jobset-leaderworkerset

# JobSet & LeaderWorkerSet — Deep Reference

The two SIG-driven workload APIs for **multi-host** ML on Kubernetes. Both model a *gang*: a set of
pods that must be scheduled together, addressed by stable identity, and restarted as a unit. JobSet
targets run-to-completion training/batch; LWS targets long-running multi-host inference.

> Versions move fast (2026). Field/enum names below match JobSet `v1alpha2` and LWS `v1` at time of
> writing — strong priors, but confirm with `kubectl explain` and the project docs for the installed
> version. Never assume a field exists because it would be convenient.

---

## 1. Mental model: why these exist

A single distributed-ML workload is **N pods that are one logical thing**. They must:

1. **Start together (or in a defined order).** A 64-pod FSDP run where 1 pod is `Pending` for quota
   is 63 pods burning GPU on a NCCL barrier. You want all-or-nothing admission (gang scheduling) and,
   sometimes, ordered start (launcher before workers).
2. **Find each other by name.** `torchrun`/NCCL rendezvous, Torch `MASTER_ADDR`, Ray head, vLLM
   pipeline ranks — all need a *stable, resolvable* hostname that survives a pod restart. A bare Job's
   pod gets a random name; a StatefulSet gives stable names but not a Job's completion semantics.
3. **Restart as a unit.** If rank 7 of a 64-way collective dies, ranks 0–63 are wedged. You must kill
   and recreate **the whole group**, not just the dead pod, or the survivors deadlock. This is *gang
   restart* — the single feature that justifies these APIs over raw primitives.

Raw Kubernetes gives you Job (completion + indexed pods, but no multi-template, no gang restart, no
ordering), StatefulSet (stable identity + ordering, but no completion/batch semantics), and
Deployment (none of it). JobSet and LWS compose these into the gang abstraction.

### JobSet vs LWS in one line

- **JobSet** — a group of *Jobs* (`replicatedJobs`) managed as a unit. Run-to-completion. Multiple
  heterogeneous templates (driver + workers + a separate eval job). For **training and batch**.
- **LWS** — a group of *(1 leader + N workers)* pods as the unit of replication; you scale the number
  of groups. Long-running. Two templates (leader, worker). For **multi-host serving / inference**.

---

## 2. JobSet — core concepts

`apiVersion: jobset.sigs.k8s.io/v1alpha2`, `kind: JobSet`. Install the JobSet controller in the
cluster (`kubectl apply --server-side -f <release manifest>`); it is not built into Kubernetes.

### 2.1 ReplicatedJobs and the generated objects

`spec.replicatedJobs[]` is the core. Each entry:

- `name` — short name, becomes a suffix in child Job/Pod names. Map key; must be unique.
- `replicas` — how many identical child Jobs to stamp out from this template (default 1).
- `template` — a full `JobTemplateSpec` (i.e. an embedded `batchv1.JobSpec` under
  `template.spec`, with its own `parallelism`, `completions`, `completionMode: Indexed`, `template`
  pod spec, `backoffLimit`, etc.).

For a replicatedJob `workers` with `replicas: 4`, JobSet creates child Jobs:
`<jobset>-workers-0` … `<jobset>-workers-3`. Each Job (with `completionMode: Indexed`) produces pods
`<jobset>-workers-<jobIndex>-<podIndex>.<subdomain>` with a stable, resolvable hostname.

Why multi-template matters: a real training run is often a **launcher/driver** replicatedJob
(`replicas: 1`, the `torchrun` rendezvous host / Ray head) plus a **workers** replicatedJob. Two
templates, one gang, one success/failure policy.

### 2.2 Network identity

`spec.network`:

- `enableDNSHostnames: true` — creates a headless Service so every pod gets a resolvable DNS A record.
  **Turn this on for any multi-pod collective.** Without it, pods can't address each other by name.
- `subdomain` — explicit Service/subdomain name; otherwise defaults from the JobSet name.
- `publishNotReadyAddresses` — publish DNS before pods are Ready (commonly needed so the rendezvous
  host is resolvable before everyone is up). Verify the exact field name/behavior against the docs.

Hostnames follow the indexed-Job convention:
`<jobset>-<replicatedJob>-<jobIndex>-<podIndex>.<subdomain>.<namespace>.svc.cluster.local`. This
predictability is what lets you compute `MASTER_ADDR` from the pod index without service discovery.

### 2.3 startupPolicy — ordered start

`spec.startupPolicy.startupPolicyOrder`:

- `AnyOrder` (default) — all replicatedJobs start concurrently.
- `InOrder` — replicatedJobs start sequentially **in the order listed**; the next one starts only
  after the previous one's pods are ready. Put the driver/launcher first so the rendezvous endpoint
  exists before workers try to connect.

### 2.4 successPolicy — when the JobSet is "done"

`spec.successPolicy`:

- `operator: All | Any` — All (default) requires every targeted replicatedJob to succeed; Any
  requires at least one.
- `targetReplicatedJobs: [names]` — which replicatedJobs the policy applies to (empty = all).

Canonical use: a long-lived `workers` replicatedJob that never exits cleanly + a `driver` that exits
0 when training finishes → `operator: All`, `targetReplicatedJobs: ["driver"]`. The JobSet completes
when the driver does, and the workers are torn down.

### 2.5 failurePolicy — gang restart

`spec.failurePolicy`:

- `maxRestarts` — how many times to restart the whole JobSet before marking it Failed.
- `restartStrategy`:
  - `Recreate` (default) — recreate child Jobs on restart.
  - `BlockingRecreate` — **delete all child Jobs/Pods of the previous iteration first**, then recreate.
    This is true gang restart: no survivor pods from the old generation linger to corrupt rendezvous.
    Prefer this for tightly-coupled collectives.
- `rules[]` — fine-grained `FailurePolicyRule`s evaluated in order. Each has:
  - `action` — `FailJobSet`, `RestartJobSet`, `RestartJobSetAndIgnoreMaxRestarts`, and (newer)
    `RestartJob` / `RestartJobAndIgnoreMaxRestarts` for single-Job recreation. Verify which actions
    exist in your installed version.
  - `onJobFailureReasons` — match specific Job failure reasons (e.g. `PodFailurePolicy`); empty
    matches any.
  - `targetReplicatedJobs` — scope the rule to specific replicatedJobs.

Pattern: ignore-max-restarts for *infrastructure* failures (host maintenance, preemption) so you
don't burn your restart budget, while a genuine application crash counts against `maxRestarts`.
Combine with the **pod-level** `Job.spec.podFailurePolicy` inside each template to distinguish
retryable (SIGTERM/preemption) from terminal (assert/OOM) exits.

### 2.6 coordinator

`spec.coordinator` designates a specific pod (`replicatedJob` + `jobIndex` + `podIndex`) as the
coordinator. JobSet then stamps a `jobset.sigs.k8s.io/coordinator` annotation/label carrying that
pod's stable endpoint onto **every** pod, so any pod can read its master address from the downward
API without computing it. Useful for Torch `MASTER_ADDR` / Ray head address.

### 2.7 Exclusive placement

To pin **one replicatedJob per topology domain** (one Job per rack / node pool / TPU slice), set the
exclusive-placement annotation on the JobSet (commonly
`alpha.jobset.sigs.k8s.io/exclusive-topology: <topologyKey>`, e.g. `cloud.google.com/gke-nodepool`).
The controller enforces that pods of different replicatedJobs don't co-locate in the same domain.
Essential for **multi-host TPU slices**, where each slice must host exactly one worker group, and for
avoiding two gangs sharing a NUMA/NVLink domain. Confirm the exact annotation key for your version.

### 2.8 Suspend & Kueue

`spec.suspend: true` suspends all child Jobs (sets their `suspend`), releasing pods without deleting
the JobSet — this is the hook [[kueue-advanced]] uses to gate admission. **JobSet itself manages
children via owner references + foreground deletion; it does not put a finalizer on the JobSet
object.** Finalizers you observe on a JobSet typically belong to Kueue (for quota accounting). If a
JobSet is stuck Terminating, check whose finalizer is on it before suspecting the JobSet controller.

---

## 3. LeaderWorkerSet (LWS) — core concepts

`apiVersion: leaderworkerset.x-k8s.io/v1`, `kind: LeaderWorkerSet`. Install the LWS controller
separately (it is not built in). Designed for **multi-host inference**: a model too large for one
host, sharded with tensor + pipeline parallelism across nodes (vLLM, SGLang).

### 3.1 The group: leader + workers

- `spec.replicas` — number of **groups** (= number of model replicas). This is what you scale.
- `spec.leaderWorkerTemplate`:
  - `leaderTemplate` — a `PodTemplateSpec` for the single leader of each group.
  - `workerTemplate` — a `PodTemplateSpec` for the workers.
  - `size` — pods **per group**, *including the leader*. `size: 4` = 1 leader + 3 workers. This is
    your shard count (TP×PP across hosts).
  - `restartPolicy` — group restart behavior (see 3.4).
  - `subGroupPolicy` — subgrouping within a group (see 3.5).
  - `volumeClaimTemplates` / `persistentVolumeClaimRetentionPolicy` — per-pod PVCs (e.g. model cache).

If you omit `leaderTemplate`, the leader uses the worker template — common when leader and workers run
the same image and differ only by an index-derived role at runtime.

### 3.2 Naming & network identity

- Leader (and group) name: `<lws>-<groupIndex>` (e.g. `vllm-0`, `vllm-1`).
- Worker name: `<lws>-<groupIndex>-<workerIndex>` (workerIndex starts at 1; leader is index 0).
- Each group gets a **headless Service** so pods within a group resolve each other by hostname —
  this is intra-group plumbing (the leader addressing its workers for Ray/NCCL).
- `spec.networkConfig.subdomainPolicy` controls the subdomain wiring (default `Shared`). The
  controller injects env like the group index and group size into pods so the entrypoint can compute
  ranks. Verify exact env var names against the installed version's docs.

**Client traffic does not go to the headless Service.** Put a *normal* `Service` (or Gateway/Ingress)
in front that selects **only leader pods** (label `role: leader`, injected by the controller). Clients
hit the leader; the leader fans the request across its workers behind the leader.

### 3.3 startupPolicy — leader-first

`spec.startupPolicy`:

- `LeaderCreated` (default) — leader and workers are created at the same time (scheduled together;
  with gang scheduling, `minMember = size`).
- `LeaderReady` — the leader must be scheduled and **Running/Ready first**, then workers start. Use
  when the leader is a Ray head / rendezvous host that workers must connect to on boot. Note the gang
  still reserves the full group's resources either way.

### 3.4 restartPolicy — all-or-nothing group restart

`spec.leaderWorkerTemplate.restartPolicy`:

- `RecreateGroupOnPodRestart` — if **any** pod in a group restarts/fails, **the entire group is
  recreated**. This is the gang-restart behavior you want for a sharded model: a dead worker means the
  whole group's KV/parallel state is gone, so restart all of it.
- `None` — the group is recreated only when no pods are currently pending (a softer policy).

For multi-host vLLM/SGLang, `RecreateGroupOnPodRestart` is almost always correct — a single missing
shard makes the replica unable to serve, and you don't want a half-alive group taking traffic.

### 3.5 subGroupPolicy — hierarchical topology

`spec.leaderWorkerTemplate.subGroupPolicy`:

- `subGroupPolicyType` (default `LeaderWorker`) and `subGroupSize`.

Splits a large group into fixed-size subgroups so the scheduler/topology can co-locate each subgroup
(e.g. pin each subgroup of 8 to one NVLink domain / one host with 8 GPUs, while the group spans
multiple hosts). Use when a single group is large enough that you need *intra-group* topology hints —
e.g. pipeline stages within a host vs across hosts. Confirm field shape against the docs.

### 3.6 rolloutStrategy — rolling update at group granularity

`spec.rolloutStrategy`:

- `type: RollingUpdate`
- `rollingUpdateConfiguration`:
  - `maxUnavailable` (IntOrString, default 1) — groups that may be unavailable during a rollout.
  - `maxSurge` (IntOrString, default 0) — extra groups spun up during a rollout.
  - `partition` (default 0) — staged/canary rollouts: only groups with index ≥ partition are updated.

Updates happen **per group**, not per pod — a whole model replica is drained and replaced atomically,
because you cannot mix old/new shards within one sharded replica. Set `maxSurge ≥ 1` when you can't
afford to drop capacity during a rollout and have spare GPUs; otherwise `maxUnavailable: 1` rolls one
replica at a time.

### 3.7 Scaling: HPA on groups

LWS exposes a **`scale` subresource** on `spec.replicas`. An HPA (or KEDA) therefore scales the
**number of groups** — i.e. number of model replicas — driven by a metric like queue depth or
GPU/token throughput. **Never** autoscale `size`; that is the model's fixed shard count, and changing
it reshards the model. See [[autoscaling-kubernetes]] for the HPA/KEDA wiring and
[[serving-frameworks]] for good serving metrics.

### 3.8 Exclusive placement

Like JobSet, LWS supports an exclusive-topology annotation to pin **one group per topology domain**
(per node pool / TPU slice / rack), so a sharded replica owns its hosts and isn't co-scheduled with
another group's shards. Confirm the exact annotation key for your version.

---

## 4. Multi-host serving with LWS (the canonical pattern)

For a model that doesn't fit one host (e.g. a large MoE across 2 nodes of 8 GPUs):

1. **`size`** = total hosts/pods in the parallel group (e.g. `size: 2` for 2 nodes; TP=8 within a
   node, PP/extra TP across the 2). **`replicas`** = how many such replicas you serve.
2. **Leader** runs the serving entrypoint / router and the head of the distributed runtime — e.g.
   the vLLM API server + Ray head (`ray start --head`), or the SGLang router. It joins workers via the
   group headless Service and exposes the OpenAI-compatible HTTP port.
3. **Workers** run `ray start --address=<leader-hostname>` (or the framework's worker entrypoint) and
   host the remaining shards. They have no client-facing port.
4. **`startupPolicy: LeaderReady`** so the Ray head is up before workers connect.
5. **`restartPolicy: RecreateGroupOnPodRestart`** so a lost worker recreates the whole replica.
6. A front **Service** selects leader pods only; that's your stable inference endpoint.

This is exactly how the upstream LWS + vLLM and LWS + SGLang multi-node examples are structured. See
`examples.md` for an annotated manifest. Framework specifics (vLLM `--tensor-parallel-size` /
`--pipeline-parallel-size`, Ray bootstrapping, SGLang `--dist-init-addr`) are in
[[serving-frameworks]].

---

## 5. Integration: Kueue, GKE, frameworks

### 5.1 Kueue (queueing, quota, gang admission)

Both JobSet and LWS have **native Kueue integration**. Label the workload with
`kueue.x-k8s.io/queue-name: <localQueue>` and Kueue gates admission: the workload stays suspended
(JobSet `spec.suspend`; LWS via its suspend mechanism) until the *whole gang* fits within a
ClusterQueue's quota — all-or-nothing admission, so you never half-schedule a collective. Kueue also
drives topology-aware scheduling (TAS) to place gangs within a topology domain, and MultiKueue to
dispatch across clusters. Kueue adds the finalizers/labels you'll see on these objects. Full quota,
cohort, preemption, and TAS detail is in [[kueue-advanced]].

### 5.2 GKE / TPU & GPU node pools

- Run on dedicated GPU/TPU node pools; use exclusive placement keyed on
  `cloud.google.com/gke-nodepool` (or your topology key) so one gang owns a slice.
- **Multi-host TPU slices**: each slice is one worker group. With JobSet, one replicatedJob per slice
  via exclusive placement; with LWS, one group per slice. The number of pods must match the slice
  topology (e.g. a v5e-16 multi-host slice → matching `replicas`/`size`). GKE injects the TPU
  topology env; your code reads it. See [[gke-master]] for slice topologies, node-pool setup, and
  Autopilot vs Standard considerations.
- Pair with the cluster autoscaler / NAP so a pending gang triggers node provisioning of the right
  accelerator shape ([[autoscaling-kubernetes]]).

### 5.3 Training & serving frameworks

- **JobSet ↔ [[training-frameworks]]**: `torchrun`/`torch.distributed` (FSDP, DDP), DeepSpeed,
  Megatron-LM, NeMo, MaxText/JAX, Ray Train, Kubeflow Trainer. The launcher replicatedJob is the
  rendezvous host; workers join it. Use `coordinator` or the predictable hostname for `MASTER_ADDR`.
- **LWS ↔ [[serving-frameworks]]**: vLLM, SGLang, Dynamo for multi-node serving; the leader holds the
  router/head, workers hold shards.

---

## 6. When to use which (decision table)

| Need | Use | Why |
|------|-----|-----|
| Multi-host training / batch, runs to completion | **JobSet** | Job semantics + gang restart + multi-template + ordering |
| Multi-host *serving* of one sharded model, long-running | **LWS** | leader+worker group as the scaling unit, group restart, HPA on groups |
| Single-pod or single-host serving | Deployment | No gang needed |
| Single multi-pod batch, one template, no gang restart | indexed **Job** | Don't add a controller you don't need |
| Stable identity + ordered start, but it's a *service* not a gang-restart workload | StatefulSet | But it won't gang-restart or do completion |
| Many heterogeneous Jobs as one unit (driver+workers+eval) | **JobSet** | Multiple replicatedJobs |
| Tight HPC/MPI gang, queueing-heavy | JobSet **+ [[kueue-advanced]]**, or Volcano/Slinky | See [[slurm-hpc-on-kubernetes]] |

**Anti-pattern:** forcing a StatefulSet to act as a training gang (no gang restart, no completion, no
success policy) or running multi-host inference as a Deployment of independent pods (no shared
identity, no group restart — a dead shard silently breaks a replica).

---

## 7. Anti-patterns & gotchas

- **Forgetting `enableDNSHostnames` (JobSet).** Pods can't resolve each other; `torchrun` rendezvous
  hangs. Symptom looks like a framework bug; it's DNS.
- **Per-pod restart instead of gang restart.** Letting a Job's `backoffLimit` restart one pod inside a
  collective leaves survivors deadlocked on a barrier. Use `BlockingRecreate` (JobSet) /
  `RecreateGroupOnPodRestart` (LWS) so the *group* restarts.
- **Wrong success target.** With long-lived workers, default `operator: All` means the JobSet never
  completes (workers never exit). Target the driver replicatedJob.
- **Autoscaling LWS `size`.** That reshards the model. Scale `replicas` (groups) only.
- **Sending client traffic to the LWS headless Service.** It targets all group pods including workers
  with no API server. Front it with a Service selecting `role: leader`.
- **Mismatched `size` vs accelerator topology.** On multi-host TPU, `size`/replica pod count must
  match the slice topology or pods won't schedule / the framework miscomputes ranks.
- **Burning the restart budget on preemptions.** Without a failurePolicy rule that ignores max
  restarts for infra/preemption reasons (and a pod-level `podFailurePolicy`), spot/maintenance churn
  exhausts `maxRestarts` and fails a healthy run.
- **No gang admission.** Without Kueue (or a gang scheduler), the scheduler can place 63 of 64 pods
  and block on the 64th — GPUs idle on a barrier. Gate admission with [[kueue-advanced]].
- **Assuming the JobSet has a finalizer.** It doesn't; foreground deletion + owner refs handle
  cleanup. A stuck-Terminating JobSet is almost always a *Kueue* finalizer.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Pods stuck `Pending`, never all schedule | No gang admission; partial placement | Add Kueue / gang scheduler; check ClusterQueue quota; check accelerator availability/NAP |
| `torchrun`/NCCL rendezvous hangs | DNS hostnames off, or master not resolvable yet | JobSet `network.enableDNSHostnames: true` (+ `publishNotReadyAddresses`); use `InOrder` so driver starts first |
| Workers can't reach leader (LWS) | Leader not Ready before workers; wrong subdomain | `startupPolicy: LeaderReady`; verify group headless Service and injected group env |
| One pod dies, run wedges, survivors idle | Per-pod restart instead of gang restart | `BlockingRecreate` (JobSet) / `RecreateGroupOnPodRestart` (LWS) |
| JobSet never completes | Success policy targets long-lived workers | `successPolicy.operator: All`, `targetReplicatedJobs: [driver]` |
| Run fails after a few spot preemptions | maxRestarts exhausted by infra failures | failurePolicy rule with `RestartJobSetAndIgnoreMaxRestarts` on infra reasons + pod `podFailurePolicy` |
| LWS HPA does nothing / scales pods weirdly | HPA pointed at wrong field | Target the `scale` subresource (replicas/groups), not pods; never `size` |
| Inference 5xx after a worker restart | Group serving with a missing shard | `RecreateGroupOnPodRestart`; readiness gate the leader so it leaves the front Service while regrouping |
| JobSet stuck `Terminating` | Kueue finalizer (not JobSet's) | Inspect `metadata.finalizers`; resolve the Kueue workload |
| Two gangs sharing a node pool / NVLink domain | No exclusive placement | Set the exclusive-topology annotation keyed on your topology label |

Diagnostic commands:

```bash
kubectl get jobset <name> -o yaml                 # status conditions, restarts, child refs
kubectl get jobs -l jobset.sigs.k8s.io/jobset-name=<name>
kubectl get leaderworkerset <name> -o yaml        # group status, replicas
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=<name> -o wide
kubectl explain jobset.spec.failurePolicy         # confirm fields/enums for installed version
kubectl explain leaderworkerset.spec.leaderWorkerTemplate
# DNS check from inside a pod:
kubectl exec <pod> -- nslookup <jobset>-<rjob>-0-0.<subdomain>
```

(Label keys above are the common ones; confirm exact keys with `kubectl get pod -o yaml` for your
installed controller version.)

---

## 9. Version awareness

- **JobSet** is `v1alpha2` and actively evolving — failurePolicy `rules`, single-Job restart actions,
  and coordinator have all changed/landed across recent releases. Pin the controller version and read
  that release's docs. Don't assume an action/field exists; check `kubectl explain`.
- **LWS** is `v1` (`leaderworkerset.x-k8s.io/v1`) but features like `subGroupPolicy`, rolling-update
  `partition`, and network config have been added incrementally. Same rule: verify against the
  installed version.
- Both are pre-1.0 ecosystem projects; CRD field shapes can change minor-to-minor. Treat this guide as
  a strong prior, the cluster's CRD as ground truth.

---

## 10. Canonical references

- JobSet docs: https://jobset.sigs.k8s.io/ — concepts, tasks (failure policy, startup, success).
- JobSet API ref (`v1alpha2`): https://jobset.sigs.k8s.io/docs/reference/jobset.v1alpha2/
- JobSet source: https://github.com/kubernetes-sigs/jobset (`api/jobset/v1alpha2/jobset_types.go`).
- LWS docs: https://lws.sigs.k8s.io/ — overview, failure handling, examples (vLLM, SGLang, llama.cpp).
- LWS API ref (`v1`): https://lws.sigs.k8s.io/docs/reference/leaderworkerset.v1/
- LWS source: https://github.com/kubernetes-sigs/lws
- Kueue + JobSet/LWS: https://kueue.sigs.k8s.io/docs/tasks/run/ (jobset, leaderworkerset).
- vLLM multi-node on LWS: https://docs.vllm.ai/en/latest/deployment/frameworks/lws.html
- Verify everything against the version of the CRD installed in your cluster.

---

# JobSet & LWS — Annotated Worked Examples

Two correct, imitable manifests: a **multi-host training JobSet** and a **multi-host vLLM inference
LWS**. Field/enum names match JobSet `v1alpha2` and LWS `v1` at time of writing — verify against the
CRD installed in your cluster (`kubectl explain ...`) before relying on any specific field.

---

## 1. Multi-host distributed training — JobSet

A `torchrun`/FSDP run: one **driver/launcher** replicatedJob (the rendezvous host, exits 0 when
training finishes) plus a **workers** replicatedJob spanning 4 nodes of 8 GPUs each. Demonstrates
`startupPolicy` (driver first), `successPolicy` (succeed when the driver completes), `failurePolicy`
(gang restart, ignore preemptions), `coordinator`, DNS hostnames, and Kueue admission.

```yaml
apiVersion: jobset.sigs.k8s.io/v1alpha2
kind: JobSet
metadata:
  name: llama-fsdp
  labels:
    kueue.x-k8s.io/queue-name: training-lq        # Kueue gates gang admission against quota
  annotations:
    # One replicatedJob per node pool (topology domain). Confirm the exact key for your version.
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  # --- Whole-group network identity: pods get resolvable, predictable hostnames ---
  network:
    enableDNSHostnames: true            # REQUIRED for the collective to find each other by DNS
    publishNotReadyAddresses: true      # master resolvable before all pods are Ready

  # --- Start the driver before the workers so the rendezvous endpoint exists first ---
  startupPolicy:
    startupPolicyOrder: InOrder         # replicatedJobs start in list order

  # --- JobSet succeeds when the driver exits 0; long-lived workers are then torn down ---
  successPolicy:
    operator: All
    targetReplicatedJobs:
      - driver                          # ignore workers (they never exit cleanly)

  # --- Gang restart: recreate the whole group on failure; don't burn budget on preemptions ---
  failurePolicy:
    maxRestarts: 6
    restartStrategy: BlockingRecreate   # delete ALL old child Jobs/Pods before recreating
    rules:
      - action: RestartJobSetAndIgnoreMaxRestarts   # infra/preemption: free retry
        onJobFailureReasons:
          - PodFailurePolicy            # paired with the pod-level podFailurePolicy below
        # targetReplicatedJobs omitted => applies to all

  # --- Designate the driver pod as coordinator; its endpoint is stamped onto every pod ---
  coordinator:
    replicatedJob: driver
    jobIndex: 0
    podIndex: 0

  replicatedJobs:
    # ---------- DRIVER / LAUNCHER (rendezvous host) ----------
    - name: driver
      replicas: 1
      template:
        spec:
          parallelism: 1
          completions: 1
          backoffLimit: 0               # let the JobSet's failurePolicy own restarts, not the Job
          template:
            spec:
              restartPolicy: Never
              # Retryable vs terminal exits: preemption (SIGTERM/137) -> ignore via the rule above.
              # (Field shape under Job.spec.template — shown on the Job spec below for workers too.)
              containers:
                - name: trainer
                  image: ghcr.io/example/llama-trainer:2026.05
                  command: ["torchrun"]
                  args:
                    - "--nnodes=4"
                    - "--nproc_per_node=8"
                    - "--rdzv_backend=c10d"
                    # Master address comes from the coordinator annotation, downward-API'd in:
                    - "--rdzv_endpoint=$(COORDINATOR):29500"
                    - "train.py"
                  env:
                    - name: COORDINATOR
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.annotations['jobset.sigs.k8s.io/coordinator']
                  ports:
                    - containerPort: 29500
                  resources:
                    limits:
                      nvidia.com/gpu: 8

    # ---------- WORKERS (4 nodes) ----------
    - name: workers
      replicas: 4                       # 4 child Jobs => 4 nodes; pods get stable hostnames
      template:
        spec:
          parallelism: 1
          completions: 1
          backoffLimit: 0
          # Distinguish retryable (preemption) from terminal (crash) at the pod level:
          podFailurePolicy:
            rules:
              - action: FailJob          # genuine crash -> count against the JobSet restart budget
                onExitCodes:
                  operator: NotIn
                  values: [143]          # 143 = SIGTERM (preemption); treat as retryable
          template:
            spec:
              restartPolicy: Never
              containers:
                - name: trainer
                  image: ghcr.io/example/llama-trainer:2026.05
                  command: ["torchrun"]
                  args:
                    - "--nnodes=4"
                    - "--nproc_per_node=8"
                    - "--rdzv_backend=c10d"
                    - "--rdzv_endpoint=$(COORDINATOR):29500"
                    - "train.py"
                  env:
                    - name: COORDINATOR
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.annotations['jobset.sigs.k8s.io/coordinator']
                  resources:
                    limits:
                      nvidia.com/gpu: 8
```

**Why it's correct:** `enableDNSHostnames` + `InOrder` make rendezvous deterministic; `coordinator`
removes the need to compute `MASTER_ADDR`; `BlockingRecreate` gives true gang restart; the
failurePolicy rule + pod-level `podFailurePolicy` keep preemptions from exhausting `maxRestarts`;
`successPolicy` targets the driver so the JobSet actually completes; the Kueue label gates atomic gang
admission. `backoffLimit: 0` cedes restart control to the JobSet.

---

## 2. Multi-host vLLM inference — LeaderWorkerSet

A model sharded across **2 nodes** of 8 GPUs (TP within a node, PP across the 2). Each **group** is
one servable replica; `replicas: 3` runs three replicas. Leader runs the vLLM API server + Ray head;
workers run `ray start --address=<leader>`. A separate Service fronts only the leaders.

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-moe
  labels:
    kueue.x-k8s.io/queue-name: serving-lq         # Kueue admits the whole group atomically
spec:
  replicas: 3                                      # 3 servable model replicas (groups). HPA scales THIS.
  startupPolicy: LeaderReady                        # Ray head (leader) Ready before workers connect
  leaderWorkerTemplate:
    size: 2                                         # pods per group, leader included => 2 nodes
    restartPolicy: RecreateGroupOnPodRestart        # a lost shard recreates the whole replica

    # ---------- LEADER: vLLM API server + Ray head ----------
    leaderTemplate:
      metadata:
        labels:
          role: leader                              # the front Service selects this
      spec:
        containers:
          - name: vllm-leader
            image: vllm/vllm-openai:v0.9.2
            command: ["/bin/sh", "-c"]
            args:
              - |
                ray start --head --port=6379 --disable-usage-stats;
                python3 -m vllm.entrypoints.openai.api_server \
                  --model meta-llama/Llama-3.1-70B \
                  --tensor-parallel-size 8 \
                  --pipeline-parallel-size 2 \
                  --port 8000
            ports:
              - containerPort: 8000                 # OpenAI-compatible HTTP (client-facing)
            readinessProbe:                          # leave the front Service while regrouping
              httpGet: { path: /health, port: 8000 }
              initialDelaySeconds: 60
              periodSeconds: 10
            resources:
              limits:
                nvidia.com/gpu: 8

    # ---------- WORKERS: Ray workers hosting the remaining shards ----------
    workerTemplate:
      spec:
        containers:
          - name: vllm-worker
            image: vllm/vllm-openai:v0.9.2
            command: ["/bin/sh", "-c"]
            args:
              # LWS injects the leader's address; the framework's env var name may differ by version.
              - |
                ray start --address=$(LWS_LEADER_ADDRESS):6379 --block
            resources:
              limits:
                nvidia.com/gpu: 8
            # No client-facing port: workers are reached only via the group headless Service.
---
# Front Service: client-facing endpoint -> LEADERS ONLY (never the headless intra-group Service).
apiVersion: v1
kind: Service
metadata:
  name: vllm-moe-endpoint
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: vllm-moe      # confirm exact label key for your version
    role: leader
  ports:
    - name: http
      port: 80
      targetPort: 8000
---
# HPA scales the NUMBER OF GROUPS (replicas) via the LWS scale subresource — never `size`.
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-moe-hpa
spec:
  scaleTargetRef:
    apiVersion: leaderworkerset.x-k8s.io/v1
    kind: LeaderWorkerSet
    name: vllm-moe
  minReplicas: 3
  maxReplicas: 12
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting           # a real serving signal (see serving-frameworks)
        target:
          type: AverageValue
          averageValue: "5"
```

**Why it's correct:** `size: 2` = the parallel group across 2 hosts; `replicas` = number of replicas
and the only thing the HPA scales (scaling `size` would reshard the model). `LeaderReady` ensures the
Ray head is up before workers `ray start --address`. `RecreateGroupOnPodRestart` means a dead worker
recreates the whole replica rather than leaving a half-alive group serving 5xx. The front Service
selects `role: leader` only; the per-group headless Service (managed by LWS) handles intra-group Ray
traffic. The leader's readiness probe pulls it out of the front Service during regrouping.

> Framework specifics — exact vLLM flags, the injected leader-address env var name, SGLang's
> `--dist-init-addr` — vary by version. Verify against the upstream LWS + vLLM/SGLang examples and
> [[serving-frameworks]]. Confirm label keys with `kubectl get pod -o yaml`.

---

## 3. subGroupPolicy sketch (intra-group topology)

For a large group where you want each subgroup co-located on one host / NVLink domain:

```yaml
spec:
  leaderWorkerTemplate:
    size: 16                              # 16-pod group...
    subGroupPolicy:
      subGroupPolicyType: LeaderWorker    # default
      subGroupSize: 8                     # ...split into subgroups of 8 (e.g. one 8-GPU host each)
```

Use this when a single sharded replica spans multiple hosts and you need the scheduler to respect
intra-group topology (pipeline stage within a host vs across hosts). Confirm the field shape against
your installed LWS version.
