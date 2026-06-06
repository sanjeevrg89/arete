---
name: ray-on-kubernetes
description: Expert guidance for running Ray on Kubernetes via the KubeRay operator — RayCluster, RayJob,
  and RayService CRDs for distributed training (Ray Train), HPO (Ray Tune), model serving (Ray Serve),
  streaming batch inference and data (Ray Data), and RL/RLHF (RLlib). Use when authoring or debugging
  KubeRay manifests, sizing head nodes, configuring the Ray autoscaler, placement groups for gang
  scheduling, GCS fault tolerance with external Redis, object-store/plasma spilling, GPU/TPU pools on
  GKE, or queueing RayJobs with Kueue. Covers the Ray core mental model (tasks, actors, object store,
  GCS, raylets, ownership), zero-downtime RayService upgrades, and KubeRay troubleshooting (pending
  actors/tasks, autoscaler not scaling, GCS restart, OOM).
---

# Ray on Kubernetes (KubeRay)

Apply the judgment of an engineer who has run multi-hundred-node Ray clusters on Kubernetes in
production for years — for large-scale training, RLHF, batch inference, and HA serving. Ray's
distributed runtime and Kubernetes' scheduling model are two different control planes; the whole job
is making them cooperate cleanly. Get the head node, the object store, placement groups, and the
autoscaler right and almost everything else follows.

## How to use this skill

1. **Read `ray-on-kubernetes-guide.md`** in this directory — the full reference (Ray core model,
   KubeRay CRDs, the libraries, operations, anti-patterns, troubleshooting). Apply it to the task.
2. For concrete, annotated manifests to imitate — RayCluster, RayJob, RayService, and a gang-scheduling
   placement-group snippet — read **`examples.md`**.
3. Match the surrounding cluster's conventions (namespaces, node pools, image registry, GPU/TPU
   labels). Apply the correctness/safety rules (GCS HA, resource requests==limits for accelerators,
   placement groups for gang work) regardless.

## Essentials (full detail in `ray-on-kubernetes-guide.md`)

- **One Ray cluster = one head + N worker groups.** The head runs the GCS (Global Control Store), the
  dashboard, and (by default) the autoscaler. Workers run a raylet + object store + your tasks/actors.
  Never schedule heavy compute on the head; size it for control-plane load, not for work.
- **The object store is shared-memory (plasma) per node.** Large objects live there zero-copy;
  `ray.put`/task returns produce `ObjectRef`s. When it fills, Ray **spills to disk** — provision fast
  local SSD and watch spill, or you'll silently tank throughput.
- **Ownership matters.** The worker that creates an `ObjectRef` owns it; if that worker (or the actor)
  dies, the object is lost and dependents fail. Don't pass refs through long, fragile lineage chains.
- **Use placement groups for any gang/colocated work.** Multi-host training and tensor-parallel serving
  need all workers at once: `STRICT_PACK`/`PACK` to colocate, `STRICT_SPREAD`/`SPREAD` to distribute.
  Bundles reserve resources atomically — no placement group means partial allocation and deadlock.
- **Pick the right CRD.** `RayCluster` = long-lived cluster you submit to. `RayJob` = run-to-completion
  (creates a cluster, runs an entrypoint, tears down). `RayService` = HA serving with zero-downtime
  rolling upgrades. Don't hand-roll Jobs around a RayCluster when RayJob exists.
- **GCS is a single point of failure unless you enable HA.** For RayService and any long-lived cluster,
  back the GCS with **external Redis** so the head can restart without killing the cluster. Plain
  clusters lose all state on head restart.
- **The Ray autoscaler ≠ Cluster Autoscaler.** Ray scales the *number of Ray worker pods* by pending
  resource demands; Cluster Autoscaler/Karpenter then provides nodes for those pending pods. Set
  `enableInTreeAutoscaler`/`autoscalerOptions` and size min/max replicas per worker group. See
  `[[autoscaling-kubernetes]]`.
- **Requests must equal limits for GPUs/TPUs**, and Ray's logical resources (`num_gpus`, custom
  resources) must match what the pod actually requests. Mismatch → tasks pending forever or oversubscribed
  hardware.
- **Pick the library deliberately:** Ray Train (distributed Torch/JAX), Ray Tune (HPO), Ray Serve
  (serving + autoscaling + composition/multiplexing), Ray Data (streaming batch inference), RLlib (RL).
  See `[[training-frameworks]]`, `[[serving-frameworks]]`, `[[rl-rlhf-frameworks]]`.
- **Avoid the classic traps:** oversized head node hiding the real problem, no placement groups for gang
  work, autoscaler min replicas at 0 with slow startup, object-store spilling to a slow disk, and pinning
  refs that prevent GC.
- **On GKE, run head and workers on the right pools** (CPU pool for head, GPU/TPU pool for workers with
  the matching node selectors/tolerations). See `[[gke-master]]`. **Queue RayJobs with Kueue** for quota
  and gang admission — see `[[kueue-advanced]]`.
- **The ecosystem moves fast (it is 2026).** KubeRay CRD fields, autoscaler v2, and Ray library APIs
  change between releases — pin versions and **verify against current docs** before relying on a field.

## Related skills

- `[[aiml-on-kubernetes]]` — umbrella for training/inference/RL/RLHF on Kubernetes; start here for the big picture.
- `[[training-frameworks]]` — Ray Train internals vs DDP/FSDP/DeepSpeed/Megatron; when Ray Train is the right wrapper.
- `[[serving-frameworks]]` — Ray Serve vs vLLM/SGLang/KServe/Triton; composing Serve in front of an engine.
- `[[rl-rlhf-frameworks]]` — RLlib and RLHF pipelines (Ray is a common substrate for RLHF rollouts).
- `[[autoscaling-kubernetes]]` — how the Ray autoscaler layers on Cluster Autoscaler / Karpenter / NAP.
- `[[kueue-advanced]]` — quota, gang admission, and queueing for RayJob/RayCluster.
- `[[gke-master]]` — GKE GPU/TPU node pools, networking, and node selectors for Ray workers.

---

# Reference — ray-on-kubernetes

# Ray on Kubernetes — Deep Reference (KubeRay)

The authoritative reference for running Ray at scale on Kubernetes via KubeRay. Written for engineers
operating large clusters for training, RLHF, batch inference, and serving. The ecosystem moves quickly
(it is 2026); CRD fields, the autoscaler, and library APIs change between releases — pin versions and
**verify fast-moving details against current docs**.

---

## 1. Ray core mental model

Ray is a distributed runtime: you write Python (or Java/C++) that calls remote **tasks** and **actors**,
and Ray schedules them across a cluster. Two control planes are in play — Ray's own scheduler and
Kubernetes' — and the operational work is making them agree.

### Cluster anatomy

- **Head node.** Runs the **GCS (Global Control Store)** — the cluster's metadata/control store (actor
  registry, node membership, placement groups, resource accounting). Also hosts the dashboard, the
  Ray client/job server, and by default the **autoscaler**. The head also runs a raylet and object
  store like any node, but you should keep real compute off it.
- **Worker nodes.** Each runs a **raylet** (the per-node scheduler + resource manager + object manager)
  and an **object store** (plasma, shared memory). Your tasks and actors run here as **worker
  processes** that the raylet starts and reuses.
- **Driver.** The process running your top-level script (`ray.init()` / the RayJob entrypoint). It owns
  the objects it creates.

### Tasks vs actors

- **Task** (`@ray.remote` function): stateless, runs once, returns an `ObjectRef`. Ray schedules it on
  any node with the required resources. Use for embarrassingly parallel and functional work.
- **Actor** (`@ray.remote` class): a stateful worker process pinned to one node for its lifetime.
  Method calls are serialized per actor. Use for model replicas, parameter servers, RL rollout workers,
  anything holding state (a loaded model, a GPU context).

### The object store, ObjectRefs, and ownership

- Calling a remote task or `ray.put(x)` stores the result in the **object store** and returns an
  **`ObjectRef`** (a future + a distributed pointer). `ray.get(ref)` blocks and fetches the value.
- Objects live in **shared memory (plasma)** per node, so multiple workers on the same node read them
  **zero-copy**. Cross-node fetches transfer over the network and are cached locally.
- **Ownership:** the worker that creates a ref *owns* it and its metadata. If the owner dies, the object
  is lost; tasks depending on it fail (Ray may reconstruct via **lineage** for task outputs, but not for
  `ray.put` objects). Long fragile lineage chains and passing refs through many hops are an anti-pattern.
- **Spilling:** when the object store fills, Ray spills objects to disk (`/tmp/ray` by default). Spilling
  to a slow disk is a top throughput killer — see §8.

### Distributed scheduling

A task/actor declares **resource requirements** (`num_cpus`, `num_gpus`, `memory`, custom resources).
The raylet schedules locally if it can; otherwise it forwards to a remote raylet, consulting the GCS for
cluster-wide resource availability. Resources in Ray are **logical accounting**, not enforcement — Ray
trusts your declarations. `num_gpus=1` reserves one logical GPU slot; it does not by itself sandbox the
process to one physical GPU (Ray sets `CUDA_VISIBLE_DEVICES` for the worker based on assignment).

### Placement groups (gang scheduling & colocation)

A **placement group (PG)** reserves a set of resource **bundles** atomically across the cluster, so all
participants get capacity together (gang scheduling) or are laid out a particular way (colocation).
Strategies:

| Strategy        | Meaning                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `PACK`          | Pack bundles onto as few nodes as possible (best-effort).               |
| `STRICT_PACK`   | All bundles on **one** node; fail if impossible.                        |
| `SPREAD`        | Spread bundles across distinct nodes (best-effort).                     |
| `STRICT_SPREAD` | Each bundle on a **distinct** node; fail if impossible.                 |

Use `STRICT_SPREAD` for multi-host training (one worker per host) and `STRICT_PACK`/`PACK` to colocate a
model's shards or to keep an actor with its data. **Any gang workload without a PG can deadlock**: half
the workers schedule, the rest sit pending, and the started workers hold resources waiting on peers.
Ray Train/Tune create PGs for you; raw task/actor code must create them explicitly. See `examples.md`.

### Accelerators

- `num_gpus=N` (fractional allowed, e.g. `0.5` to pack two actors on one GPU). For TPUs, Ray exposes a
  custom resource (commonly `TPU`); on GKE TPU pods, Ray detects chips and you request them as a custom
  resource — **verify the exact resource name and per-pod chip count against current Ray/GKE docs**.
- Always make the pod's accelerator request match Ray's logical resource. Mismatch → pending tasks or
  oversubscription.

---

## 2. KubeRay: operator and CRDs

**KubeRay** is the Kubernetes operator for Ray. Install it (Helm chart `kuberay-operator`, or manifests)
into its own namespace; it watches three CRDs cluster-wide (or per-namespace if scoped). Pin the operator
version to a Ray version range you've tested.

### `RayCluster`

The base CRD: a long-lived cluster you submit work to (Ray Jobs API, Ray client, or `kubectl exec`).

- **`headGroupSpec`** — one head pod. `rayStartParams` tune the head (`num-cpus`, `dashboard-host`,
  `block`). The pod template is a full Kubernetes pod spec (resources, volumes, node selectors).
- **`workerGroupSpecs[]`** — one or more worker groups, each with `groupName`, `replicas`, `minReplicas`,
  `maxReplicas`, a pod template, and `rayStartParams`. Multiple groups let you mix node shapes (CPU
  group, A100 group, TPU group) in one logical cluster.
- **`enableInTreeAutoscaler`** + **`autoscalerOptions`** — turn on the Ray autoscaler sidecar on the head.
- Set `rayStartParams` `num-gpus`/custom `resources` so Ray's logical view matches the pod's requests.

### `RayJob`

Run-to-completion. KubeRay creates a RayCluster, runs the **entrypoint**, captures the exit status, and
(optionally) tears the cluster down.

- **`entrypoint`** — the command (e.g. `python train.py`).
- **`rayClusterSpec`** — embeds a RayCluster spec, **or** `clusterSelector` to target an existing cluster.
- **`shutdownAfterJobFinishes: true`** + **`ttlSecondsAfterFinished`** — ephemeral cluster lifecycle.
- **`submissionMode`** — `K8sJobMode` (default; a K8s Job submits the entrypoint) vs `HTTPMode`
  (submit via the Ray Job Submission API). **Verify available modes against your KubeRay version.**
- **`suspend: true`** — gate admission; this is the hook **Kueue** uses for queued/gang admission of
  RayJobs (see `[[kueue-advanced]]`).
- Use RayJob — not a hand-rolled Job calling `ray job submit` — so you get status, retries, and cleanup.

### `RayService`

HA online serving for **Ray Serve** with zero-downtime upgrades.

- **`serveConfigV2`** — the Serve application(s)/deployment config (import path, route prefix, per-
  deployment replicas/resources/autoscaling).
- **`rayClusterConfig`** — the underlying RayCluster spec.
- **Zero-downtime upgrade:** on a spec change KubeRay brings up a **new RayCluster** ("pending"),
  waits until its Serve apps are healthy, then switches the Kubernetes Service traffic over and tears
  down the old one (blue/green). Config-only changes to `serveConfigV2` can be applied in place. **Verify
  the exact upgrade semantics/fields for your version.**
- **Health:** RayService monitors Serve app status and restarts/recreates on unhealthy deployments.

### Pod templates, services, and networking

- Worker/head pod templates are ordinary pod specs: set `nodeSelector`/`tolerations` to land on the right
  pools, `resources.requests==limits` for accelerators, `securityContext`, volumes for spill/checkpoints.
- KubeRay creates a **head service** exposing GCS (6379), the client server (10001), the dashboard
  (8265), and the Serve HTTP proxy (8000). Workers reach the head via this service DNS name. Don't
  hardcode pod IPs — the head can be recreated.

### Autoscaling interplay

- The **Ray autoscaler** watches pending resource demands (tasks/actors/PGs that can't be placed) and
  adjusts `replicas` of worker groups between min/max. It also **scales down idle workers**
  (`idleTimeoutSeconds`).
- That creates pending **pods**; the **Cluster Autoscaler / Karpenter / GKE NAP** then provisions
  **nodes** for them. Two-layer scaling — tune both. Common failure: Ray wants workers but no node pool
  can satisfy the pod (wrong selectors/taints/quota), so pods stay Pending forever.
- Prefer **autoscaler v2** where available for faster, more accurate scaling — **verify availability and
  config in your KubeRay/Ray version.** Full treatment in `[[autoscaling-kubernetes]]`.

### GCS fault tolerance / HA (external Redis)

- By default the GCS keeps cluster state **in the head's memory**. If the head pod dies, the cluster's
  control state is gone and the cluster is effectively dead.
- **GCS FT:** point the head at an **external Redis** (annotation `ray.io/ft-enabled: "true"` plus a
  `RAY_REDIS_ADDRESS` and password via env/secret — **verify exact keys for your version**). The GCS
  persists to Redis, so a restarted head **recovers** membership and actor state, and workers reconnect.
- **Mandatory for RayService and any long-lived cluster.** Use a managed/HA Redis (Sentinel or a managed
  offering), not a single ephemeral pod, or you've just moved the SPOF.

---

## 3. Ray libraries — what to reach for

| Library      | Use it for                                          | Notes / cross-link                          |
|--------------|-----------------------------------------------------|---------------------------------------------|
| **Ray Train**| Distributed training (Torch DDP/FSDP, JAX, etc.)    | Wraps your loop, manages workers + PGs. `[[training-frameworks]]` |
| **Ray Tune** | Hyperparameter search / scheduling (ASHA, PBT)      | Runs many trials, each possibly a Train run.|
| **Ray Serve**| Online model serving, composition, autoscaling      | Deployments, DAGs, multiplexing. `[[serving-frameworks]]` |
| **Ray Data** | Streaming batch inference + last-mile data          | Map/transform over datasets, GPU batch infer.|
| **RLlib**    | Reinforcement learning / RLHF rollouts              | Actors for rollout workers. `[[rl-rlhf-frameworks]]` |

### Ray Train

- A `TorchTrainer`/`Trainer` runs a `train_loop_per_worker` across `num_workers`, each a Ray actor with
  `num_gpus`/`num_cpus` via a `ScalingConfig`. Train sets up the process group (rank, world size,
  master addr) and **creates a placement group** for the workers — you don't hand-wire PyTorch DDP.
- Use Train's checkpoint API + a shared/object-store-backed path (GCS bucket, PVC) so checkpoints survive
  worker loss and the autoscaler. For framework internals (FSDP/DeepSpeed/Megatron), see
  `[[training-frameworks]]`.

### Ray Tune

- Each trial is scheduled with its own resources; with many GPU trials, ensure the autoscaler max and
  the cluster quota can hold the concurrency you ask for, or trials queue. Compose Tune **around** Train
  for distributed-per-trial.

### Ray Serve

- **Deployment** = a scalable group of replicas (actors) of a class/function. **Application** = a graph of
  deployments. Compose with **deployment handles** (call one deployment from another) to build DAGs
  (e.g. preprocessor → model → postprocessor).
- **Autoscaling** per deployment: `min_replicas`/`max_replicas`/`target_ongoing_requests` (name varies by
  version — **verify**). Serve autoscaling drives the Ray autoscaler, which drives node autoscaling.
- **Multiplexing:** serve many models (e.g. LoRA adapters) from a shared replica pool, routing by model
  id so you don't pay a replica per model.
- Serve often fronts a dedicated inference engine (vLLM/SGLang) — Serve does routing/composition/
  autoscaling, the engine does the fast token generation. See `[[serving-frameworks]]`.

### Ray Data

- Streaming execution: reads → transforms → writes flow through the cluster without materializing
  everything. Use `map_batches` with an **actor** (a loaded-model class) and `num_gpus` for GPU batch
  inference; Ray Data pipelines GPU compute with CPU read/decode for high utilization. Watch object-store
  pressure (§8) on wide pipelines.

---

## 4. Sizing and topology

- **Head node sizing.** Size for control-plane load: GCS, dashboard, autoscaler, and the **owner
  metadata** of all the objects/actors created by the driver. CPU and especially **memory** scale with
  the number of actors, tasks-in-flight, and objects. A common mistake is making the head huge to mask an
  OOM that's really caused by the driver creating millions of refs — fix the workload, don't just inflate
  the head. Keep real compute off the head (`num-cpus: 0` on the head's `rayStartParams` is a common
  pattern to stop tasks landing there).
- **Worker groups.** One group per distinct node shape. Set `resources.requests==limits` for GPUs/TPUs.
  Give each group the right `nodeSelector`/`tolerations`. Keep `rayStartParams` resource declarations in
  sync with pod requests.
- **Object store size.** Defaults to a fraction of node memory; size it for your working set. Too small →
  spilling; too large → less room for heaps and OOM. Mount **fast local SSD** for the spill directory.

---

## 5. Operations & observability

- **Dashboard** (port 8265): cluster/node/actor/task views, object-store usage, logs, and profiling.
  Expose it carefully (it has no auth by default — front it with auth/ingress, never raw on the internet).
- **Metrics:** Ray exports **Prometheus** metrics (per-node, GCS, autoscaler, Serve). Scrape them and
  alert on object-store spill, OOM kills, actor restarts, pending tasks, and autoscaler scale-up latency.
  Wire Grafana with the Ray dashboards. **Verify current metric names against docs.**
- **Logs:** task/actor logs land under `/tmp/ray/session_*/logs` on each node and in the dashboard.
  For durability, run a log shipper (Fluent Bit) as a sidecar/DaemonSet — pod logs vanish when workers
  scale down. The KubeRay docs describe a sidecar logging pattern.
- **Networking:** workers find the head through the KubeRay head **Service**; ensure NetworkPolicies allow
  GCS (6379), client (10001), dashboard (8265), Serve (8000), and the object manager/raylet ports.
- **GKE GPU/TPU pools:** run the head on a CPU pool and workers on GPU/TPU pools with matching node
  selectors and tolerations; install the device plugins/drivers per `[[gke-master]]`. For TPU, use the
  TPU-enabled node pool and the matching Ray TPU image; **verify TPU resource naming and topology.**
- **Queueing with Kueue:** integrate RayJob (and RayCluster) with **Kueue** for quota, fair sharing, and
  gang admission. Kueue uses `suspend` on the RayJob and admits when quota/topology allows. See
  `[[kueue-advanced]]`.

---

## 6. Best practices

- **Always pick the most specific CRD:** RayService for serving, RayJob for batch, RayCluster only for a
  shared interactive/long-lived cluster.
- **Enable GCS FT (external Redis) for anything long-lived or serving.** Treat head restart as a
  when-not-if event.
- **Use placement groups for all gang/colocated work** and let Ray Train/Tune/Serve create them where
  they do. Match PG strategy to topology (STRICT_SPREAD for multi-host, PACK for colocation).
- **Keep Ray logical resources == pod requests**, and **requests==limits** for accelerators.
- **Right-size the object store and put the spill dir on fast SSD.** Monitor spill volume.
- **Pin versions** of the operator, Ray, and your images together; upgrade deliberately. KubeRay and Ray
  versions are coupled — a Ray image too new/old for the operator breaks subtly.
- **Make checkpoints durable** (object storage / PVC) so the autoscaler and worker loss don't cost you
  progress.
- **Set autoscaler min replicas > 0 for latency-sensitive serving** so you don't cold-start on every
  spike; set sane `idleTimeoutSeconds`.
- **Scope the operator and use namespaces/quotas** for multi-tenancy; one operator can manage many teams'
  clusters.

---

## 7. Anti-patterns

- **Oversized head node to "fix" OOM.** Almost always masks a driver creating too many refs/actors, or
  pulling large objects to the driver with `ray.get`. Fix the access pattern (stream, fan out, avoid
  global `ray.get`).
- **Gang work with no placement group.** Half the ranks start, the rest pend, the cluster deadlocks while
  holding resources. Classic with hand-rolled multi-host training.
- **Autoscaler misconfig.** `minReplicas: 0` with a multi-minute image pull and a latency SLO; or Ray
  wanting workers that no node pool can satisfy (wrong selector/taint/quota) so pods Pend forever; or
  conflicting Ray scale-down vs Cluster Autoscaler. Tune both layers together.
- **Object-store spilling to the root/EBS-backed disk.** Tanks throughput invisibly; mount local SSD.
- **Pinning ObjectRefs forever** (holding them in a long-lived list/actor) so Ray can't GC them — leads to
  object-store pressure and spill. Drop refs when done.
- **Heavy compute on the head.** Starves the GCS/dashboard/autoscaler and causes head OOM. Set
  `num-cpus: 0` on the head.
- **Pod IP assumptions / no GCS FT.** Treating the head as immortal; one restart wipes the cluster.
- **Mismatched Ray vs pod resources** → silent pending tasks or GPU oversubscription.

---

## 8. Performance & scale

- **Object store / plasma.** Keep large objects in the store and pass refs, not values. Same-node reads
  are zero-copy; design data locality so consumers run where the data is. Watch **spill rate** — sustained
  spilling means the working set exceeds the store; shrink batch/partition size or add nodes.
- **Avoid `ray.get` on the critical path** for large fan-out; use `ray.wait` to process as results land,
  and avoid pulling everything to the driver.
- **Resource fragmentation.** Many small actors/tasks can leave nodes with stranded fractional resources
  so a large request can't be placed. Use placement groups to reserve coherent bundles, and keep task
  granularity reasonable.
- **Worker startup.** Cold start = image pull + Ray init + (often) model load. Big images and per-task
  imports dominate. Bake deps into the image, use a runtime env sparingly, pre-pull images, and keep
  warm minReplicas for serving.
- **Large clusters.** GCS and the head are the scaling bottleneck; millions of tiny tasks stress GCS
  metadata. Batch work, use actors to amortize, and size/protect the head accordingly.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tasks/actors stuck **PENDING** | No node satisfies the resource request; PG can't be placed; autoscaler at max | Check `ray status`/dashboard for demands; confirm a worker group + node pool can satisfy it; raise max replicas/quota |
| **Autoscaler not scaling up** | Ray scaling pods but Cluster Autoscaler/Karpenter can't place them (selector/taint/quota); autoscaler not enabled | Check pending **pods** and CA logs; fix nodeSelector/tolerations/quota; confirm `enableInTreeAutoscaler` |
| **GCS / head restart** kills cluster | No GCS fault tolerance | Enable GCS FT with external Redis; treat head as restartable |
| Throughput collapses under load | **Object-store spilling** to slow disk | Mount fast local SSD for spill; shrink working set; add nodes |
| Head **OOM** | Driver creating too many refs/actors; pulling big objects to driver; compute on head | Stream/fan-out; `num-cpus: 0` on head; right-size head memory |
| Multi-host training **deadlocks** | No placement group / partial gang | Use STRICT_SPREAD PG (or let Ray Train create it) |
| RayService upgrade drops traffic | Misunderstood upgrade semantics; new cluster never goes healthy | Check pending-cluster Serve status; ensure new app passes health before cutover |
| Workers can't reach head | NetworkPolicy / wrong service DNS / IP hardcoded | Allow GCS/client/object-manager ports; use head Service DNS |
| GPU tasks oversubscribe | Ray `num_gpus` ≠ pod GPU request | Make logical resources match pod requests; requests==limits |

Diagnostic toolkit: `ray status` (autoscaler view of demands/nodes), the **dashboard** (actors, object
store, logs), Prometheus metrics, `kubectl describe pod` (Pending reasons), Cluster Autoscaler/Karpenter
logs, and the **KubeRay operator logs** (`kubectl logs -n <op-ns> deploy/kuberay-operator`).

---

## 10. Version awareness

KubeRay CRD fields (`submissionMode`, `serveConfigV2` shape, GCS-FT keys, autoscaler v1 vs v2), Ray
library APIs (Serve autoscaling param names, Train `ScalingConfig`), and TPU resource naming **change
between releases**. Pin the operator + Ray image + library versions together, test upgrades in staging,
and **verify any field referenced here against the docs for your exact version** before relying on it.

---

## 11. Canonical references

- Ray docs — https://docs.ray.io/
- Ray on Kubernetes (KubeRay) docs — https://docs.ray.io/en/latest/cluster/kubernetes/index.html
- KubeRay project & docs — https://github.com/ray-project/kuberay and https://ray-project.github.io/kuberay/
- Ray architecture whitepaper — https://docs.ray.io/en/latest/ray-contribute/whitepaper.html
- Placement groups — https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html
- GCS fault tolerance — https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/kuberay-gcs-ft.html
- Ray Serve — https://docs.ray.io/en/latest/serve/index.html
- Ray Train — https://docs.ray.io/en/latest/train/train.html
- Ray Data — https://docs.ray.io/en/latest/data/data.html
- Kueue + KubeRay — https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kueue.html
- GKE + Ray (TPU/GPU) — https://cloud.google.com/kubernetes-engine/docs (search "Ray")

---

# Ray on Kubernetes — Worked Examples

Annotated, runnable-in-spirit manifests and snippets to imitate. KubeRay CRD fields move between
releases — **verify `apiVersion`, field names, and GCS-FT/autoscaler keys against the docs for your exact
KubeRay/Ray version.** Replace image tags, namespaces, node selectors, and resource sizes for your env.

---

## 1. RayCluster (long-lived, GPU workers, autoscaling, GCS FT)

A shared cluster: CPU head (no compute), an autoscaling GPU worker group, GCS fault tolerance via
external Redis.

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-shared
  namespace: ray
  annotations:
    ray.io/ft-enabled: "true"            # enable GCS fault tolerance (verify key for your version)
spec:
  rayVersion: "2.x"                       # MUST match the ray image tag below
  enableInTreeAutoscaler: true
  autoscalerOptions:
    idleTimeoutSeconds: 300               # scale a worker down after 5m idle
    upscalingMode: Default

  headGroupSpec:
    rayStartParams:
      num-cpus: "0"                       # keep tasks/actors OFF the head
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray:2.x      # pin; must match rayVersion
            resources:                     # size for GCS + dashboard + object-owner metadata
              requests: { cpu: "4", memory: "16Gi" }
              limits:   { cpu: "4", memory: "16Gi" }
            env:
              # External Redis for GCS FT (use a Secret; verify exact env keys for your version)
              - name: RAY_REDIS_ADDRESS
                value: "redis-master.ray.svc.cluster.local:6379"
              - name: REDIS_PASSWORD
                valueFrom: { secretKeyRef: { name: ray-redis, key: password } }
            ports:
              - { containerPort: 6379, name: gcs }
              - { containerPort: 10001, name: client }
              - { containerPort: 8265, name: dashboard }
            volumeMounts:
              - { name: ray-logs, mountPath: /tmp/ray }
        volumes:
          - { name: ray-logs, emptyDir: {} }
        nodeSelector:
          cloud.google.com/compute-class: "cpu-pool"   # head on a CPU pool

  workerGroupSpecs:
    - groupName: gpu-a100
      replicas: 1
      minReplicas: 1                       # >0 if you want warm GPUs; 0 to scale to zero
      maxReplicas: 16
      rayStartParams:
        num-gpus: "1"                      # logical resource MUST match the pod request below
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray:2.x-gpu
              resources:
                requests: { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }
                limits:   { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }  # requests==limits for GPU
              volumeMounts:
                - { name: spill, mountPath: /tmp/ray }   # spill to FAST LOCAL SSD
          volumes:
            - name: spill
              ephemeral:                   # local SSD-backed; do NOT spill to slow network disk
                volumeClaimTemplate:
                  spec: { accessModes: ["ReadWriteOnce"], resources: { requests: { storage: 200Gi } } }
          nodeSelector:
            cloud.google.com/gke-accelerator: "nvidia-tesla-a100"
          tolerations:
            - { key: "nvidia.com/gpu", operator: "Exists", effect: "NoSchedule" }
```

---

## 2. RayJob (run-to-completion, ephemeral cluster, Kueue-ready)

Creates a cluster, runs the entrypoint, tears down. `suspend` + a queue label let **Kueue** gate
admission for quota/gang scheduling (`[[kueue-advanced]]`).

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: train-llm
  namespace: ray
  labels:
    kueue.x-k8s.io/queue-name: gpu-queue   # Kueue local queue (verify label for your Kueue version)
spec:
  entrypoint: "python /home/ray/train.py --epochs 3"
  shutdownAfterJobFinishes: true           # delete the cluster when the job ends
  ttlSecondsAfterFinished: 600
  suspend: true                            # Kueue flips this to false on admission
  # submissionMode: K8sJobMode             # verify available modes for your version
  rayClusterSpec:
    rayVersion: "2.x"
    headGroupSpec:
      rayStartParams: { num-cpus: "0" }
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.x
              resources:
                requests: { cpu: "4", memory: "16Gi" }
                limits:   { cpu: "4", memory: "16Gi" }
    workerGroupSpecs:
      - groupName: gpu
        replicas: 4                         # gang: Ray Train will form a PG across these 4 workers
        minReplicas: 4
        maxReplicas: 4
        rayStartParams: { num-gpus: "1" }
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray:2.x-gpu
                resources:
                  requests: { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }
                  limits:   { cpu: "12", memory: "96Gi", nvidia.com/gpu: "1" }
            nodeSelector: { cloud.google.com/gke-accelerator: "nvidia-tesla-a100" }
            tolerations:
              - { key: "nvidia.com/gpu", operator: "Exists", effect: "NoSchedule" }
```

---

## 3. RayService (HA serving, zero-downtime upgrades, autoscaling)

Ray Serve behind a RayService. Spec changes trigger a blue/green upgrade (new cluster comes up healthy,
then traffic cuts over). Enable GCS FT in `rayClusterConfig` for production.

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: text-model
  namespace: ray
spec:
  serveConfigV2: |                          # verify schema for your Serve version
    applications:
      - name: text
        import_path: app:deployment_graph   # module:variable for the Serve app
        route_prefix: /
        deployments:
          - name: Model
            num_replicas: auto              # Serve autoscaling (param names vary by version — verify)
            autoscaling_config:
              min_replicas: 2               # >0: keep warm replicas, no cold start on spikes
              max_replicas: 20
              target_ongoing_requests: 8
            ray_actor_options:
              num_gpus: 1                    # one GPU per replica; matches pod GPU request
  rayClusterConfig:
    rayVersion: "2.x"
    headGroupSpec:
      rayStartParams: { num-cpus: "0", dashboard-host: "0.0.0.0" }
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.x
              ports:
                - { containerPort: 8000, name: serve }    # Serve HTTP proxy
              resources:
                requests: { cpu: "4", memory: "16Gi" }
                limits:   { cpu: "4", memory: "16Gi" }
    workerGroupSpecs:
      - groupName: gpu
        replicas: 2
        minReplicas: 2
        maxReplicas: 20
        rayStartParams: { num-gpus: "1" }
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray:2.x-gpu
                resources:
                  requests: { cpu: "8", memory: "48Gi", nvidia.com/gpu: "1" }
                  limits:   { cpu: "8", memory: "48Gi", nvidia.com/gpu: "1" }
            nodeSelector: { cloud.google.com/gke-accelerator: "nvidia-l4" }
            tolerations:
              - { key: "nvidia.com/gpu", operator: "Exists", effect: "NoSchedule" }
```

---

## 4. Placement group — gang scheduling for multi-host work

Reserve resources atomically so all ranks start together. `STRICT_SPREAD` = one bundle per node (typical
multi-host training); `PACK`/`STRICT_PACK` = colocate.

```python
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

ray.init()  # connects to the cluster (RAY_ADDRESS / ray client)

NUM_WORKERS = 4
# One bundle per worker; STRICT_SPREAD forces each onto a distinct node (one GPU host each).
pg = placement_group(
    bundles=[{"GPU": 1, "CPU": 8} for _ in range(NUM_WORKERS)],
    strategy="STRICT_SPREAD",
)
ray.get(pg.ready())  # BLOCKS until the whole gang is reserved — no partial allocation / deadlock

@ray.remote(num_gpus=1, num_cpus=8)
class Worker:
    def rank_info(self):
        import os
        return os.environ.get("CUDA_VISIBLE_DEVICES")

# Pin each actor into its own bundle of the placement group.
workers = [
    Worker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i
        )
    ).remote()
    for i in range(NUM_WORKERS)
]

print(ray.get([w.rank_info.remote() for w in workers]))
remove_placement_group(pg)  # always release the reservation when done
```

> In practice, **Ray Train / Tune / Serve create and manage placement groups for you** — only hand-write
> a PG for raw task/actor gang workloads. Always `pg.ready()` (or let the library gang-admit) before
> launching work, and always remove the PG to free the reservation.
