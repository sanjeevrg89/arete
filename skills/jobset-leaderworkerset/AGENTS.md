# AGENTS.md — JobSet & LeaderWorkerSet (multi-host ML on Kubernetes)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`jobset-leaderworkerset-guide.md`** next to this file —
> read it before authoring or debugging multi-host training/inference manifests. Correct annotated
> manifests to imitate are in **`examples.md`**. This file is the always-on summary.
>
> APIs: **JobSet** `jobset.sigs.k8s.io/v1alpha2` (training/batch gang of Jobs) and **LeaderWorkerSet
> / LWS** `leaderworkerset.x-k8s.io/v1` (multi-host inference gang of leader+workers). Both evolve
> fast (2026) — treat field/enum names as strong priors and verify with `kubectl explain` and the
> installed version's docs. Never fabricate spec fields.

## Pick the right API

- Multi-host **training/batch**, runs to completion, possibly multi-template (driver+workers) →
  **JobSet**.
- Multi-host **serving** of one sharded model, long-running, scale by replicas → **LWS**.
- Single-host serving → Deployment. Single-host stateful → StatefulSet. Single multi-pod batch →
  indexed Job. Don't force a StatefulSet to be a training gang or a Deployment to be a sharded replica.

## JobSet rules

- `spec.replicatedJobs[]`: each has `name`, `replicas`, `template` (a full `JobTemplateSpec`). Child
  Jobs are `<jobset>-<rjob>-<jobIndex>`; pods get stable hostnames.
- **Always set `network.enableDNSHostnames: true`** for multi-pod collectives, or pods can't resolve
  each other (rendezvous hangs). Consider `publishNotReadyAddresses` so the master resolves early.
- `startupPolicy.startupPolicyOrder: InOrder` starts replicatedJobs in list order — put the
  driver/launcher first.
- `successPolicy.operator: All|Any` + `targetReplicatedJobs` — target the **driver** when workers are
  long-lived, or the JobSet never completes.
- `failurePolicy`: `maxRestarts`, `restartStrategy: BlockingRecreate` for true gang restart (delete
  all old child Jobs first), and `rules[]` with `action` (`FailJobSet` / `RestartJobSet` /
  `RestartJobSetAndIgnoreMaxRestarts` / `RestartJob*`) + `onJobFailureReasons` + `targetReplicatedJobs`.
  Use `*IgnoreMaxRestarts` for infra/preemption so churn doesn't fail a healthy run. Verify actions
  exist in your version.
- `coordinator` (`replicatedJob`/`jobIndex`/`podIndex`) stamps the master endpoint onto all pods.
- Exclusive placement: annotation keyed on a topology label (e.g. `cloud.google.com/gke-nodepool`) →
  one replicatedJob per domain. Required for multi-host TPU slices.
- `spec.suspend` is the Kueue gate. **JobSet has no finalizer of its own** (owner refs + foreground
  deletion); a stuck-Terminating JobSet is almost always a Kueue finalizer.

## LWS rules

- `spec.replicas` = number of **groups** (model replicas). `leaderWorkerTemplate.size` = pods **per
  group, leader included** (the shard count). Two templates: `leaderTemplate`, `workerTemplate`.
- Leader name `<lws>-<g>`; worker `<lws>-<g>-<w>`. Per-group **headless Service** for intra-group
  pod-to-pod. Client traffic goes to a **separate Service selecting `role: leader`**, never the
  headless one.
- `startupPolicy: LeaderReady` (leader Ready before workers) vs `LeaderCreated` (together). Use
  `LeaderReady` when the leader is a Ray head / rendezvous host.
- `restartPolicy: RecreateGroupOnPodRestart` = all-or-nothing group restart (correct for sharded
  serving); `None` = recreate only when no pods pending.
- `rolloutStrategy.type: RollingUpdate` with `maxUnavailable`/`maxSurge`/`partition` — updates happen
  **per group** (whole replica drained atomically). `maxSurge ≥ 1` to hold capacity during rollout.
- **Scale via the `scale` subresource on `replicas` (groups). Never autoscale `size`** — that reshards
  the model. HPA/KEDA target groups.
- `subGroupPolicy` (`subGroupPolicyType`, `subGroupSize`) for intra-group topology (subgroups per host
  / NVLink domain). Exclusive placement → one group per topology domain.

## Multi-host serving pattern (LWS)

Leader = serving entrypoint + distributed-runtime head (vLLM API server + Ray head, or SGLang
router). Workers = `ray start --address=<leader>` hosting remaining shards. `size` = hosts in the
parallel group; `replicas` = number of replicas. `LeaderReady` + `RecreateGroupOnPodRestart` + front
Service on `role: leader`.

## Cross-cutting

- Gate admission with **[[kueue-advanced]]** (`kueue.x-k8s.io/queue-name`) so the whole gang is
  admitted atomically — never half-schedule a collective.
- Run on GPU/TPU node pools per **[[gke-master]]**; match pod count to multi-host TPU slice topology.
- JobSet runs **[[training-frameworks]]** (torchrun/FSDP/DeepSpeed/Megatron/MaxText); LWS runs
  **[[serving-frameworks]]** (vLLM/SGLang/Dynamo). HPA wiring in **[[autoscaling-kubernetes]]**.

## Definition of done

- Right API chosen (training→JobSet, serving→LWS); not a StatefulSet/Deployment forced into a gang.
- Stable network identity verified (DNS resolves; LWS leader Service separate from headless).
- Gang restart configured (`BlockingRecreate` / `RecreateGroupOnPodRestart`), not per-pod restart.
- Start ordering and success/failure targeting correct; restart budget protected from preemptions.
- Gang admission via Kueue; placement matches accelerator topology.
- Fields/enums verified against the installed CRD version (`kubectl explain`) — nothing fabricated.
