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
