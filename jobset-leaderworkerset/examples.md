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
