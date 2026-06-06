# AGENTS.md — Ray on Kubernetes (KubeRay)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ray-on-kubernetes-guide.md`** next to this file — read it
> before authoring or debugging KubeRay manifests or Ray apps. Annotated RayCluster/RayJob/RayService
> manifests and a placement-group snippet are in **`examples.md`**. This file is the always-on summary.
>
> The ecosystem moves fast (it is 2026): KubeRay CRD fields, autoscaler v2, and Ray library APIs change
> between releases. Pin versions and **verify fast-moving fields against current docs**.

## Mental model (apply by default)

- **Cluster = one head + N worker groups.** Head runs the **GCS** (control store), dashboard, and
  autoscaler; workers run a **raylet + object store** and your tasks/actors. Keep real compute **off the
  head** (`num-cpus: 0`).
- **Object store is shared-memory (plasma) per node.** Pass `ObjectRef`s, not values; same-node reads are
  zero-copy. When it fills, Ray **spills to disk** — mount **fast local SSD** and monitor spill.
- **Ownership:** the worker that creates a ref owns it; if it dies the object is lost. Avoid long lineage
  chains and pinning refs forever (blocks GC → spill).
- **Tasks** = stateless functions; **actors** = stateful pinned processes (model replicas, rollout
  workers). Resources (`num_cpus`/`num_gpus`/custom) are **logical accounting** — keep them == pod requests.

## Rules

- **Use placement groups for all gang/colocated work.** STRICT_SPREAD for multi-host training,
  PACK/STRICT_PACK for colocation. No PG on gang work → partial allocation + deadlock. Ray Train/Tune/Serve
  create PGs for you; raw task/actor code must create them.
- **Pick the right CRD:** `RayService` for HA serving (zero-downtime blue/green upgrades), `RayJob` for
  run-to-completion (don't hand-roll a Job), `RayCluster` only for a shared long-lived cluster.
- **Enable GCS fault tolerance (external Redis)** for RayService and any long-lived cluster — head restart
  is when-not-if; without FT a head restart wipes the cluster. Use HA/managed Redis, not one pod.
- **Two-layer autoscaling:** the **Ray autoscaler** scales worker **pods** by pending resource demand;
  **Cluster Autoscaler/Karpenter/NAP** then provisions **nodes**. Tune both. Common bug: Ray wants workers
  no node pool can satisfy (selector/taint/quota) → pods Pend forever. See `[[autoscaling-kubernetes]]`.
- **Accelerators:** `resources.requests == limits` for GPU/TPU pods, and Ray `num_gpus`/custom resources
  must match the pod request. Mismatch → pending tasks or oversubscription. Verify TPU resource naming.
- **Sizing:** size the head for control-plane + object-owner metadata, not compute; don't inflate it to
  mask a driver creating millions of refs or `ray.get`-ing huge objects to the driver.
- **Libraries:** Ray Train (distributed Torch/JAX → `[[training-frameworks]]`), Ray Tune (HPO), Ray Serve
  (serving/composition/multiplexing/autoscaling → `[[serving-frameworks]]`), Ray Data (streaming batch
  inference), RLlib (RL/RLHF → `[[rl-rlhf-frameworks]]`).
- **GKE:** head on a CPU pool, workers on GPU/TPU pools with matching selectors/tolerations; install
  device plugins/drivers. See `[[gke-master]]`. **Queue RayJobs with Kueue** (uses `suspend`) for quota +
  gang admission → `[[kueue-advanced]]`.
- **Make checkpoints durable** (object storage/PVC) so worker loss and autoscaler scale-down don't lose
  progress.
- **Observability:** scrape **Prometheus** metrics; alert on spill, OOM kills, actor restarts, pending
  tasks, autoscaler latency. Ship logs (Fluent Bit) — pod logs vanish on scale-down. Don't expose the
  dashboard unauthenticated.

## Anti-patterns to flag in review

Oversized head node masking OOM · gang work without a placement group · `minReplicas: 0` for
latency-sensitive serving · spilling to a slow disk · pinned refs blocking GC · compute on the head · no
GCS FT / pod-IP assumptions · Ray logical resources ≠ pod requests · hand-rolled Job instead of RayJob.

## Definition of done

Right CRD chosen · GCS FT enabled where long-lived/serving · placement groups for gang work · accelerator
requests==limits and == Ray logical resources · autoscaler min/max + node pool selectors verified · spill
on fast SSD · durable checkpoints · metrics/logs wired · versions pinned and **fast-moving fields verified
against current docs**.
