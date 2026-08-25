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
