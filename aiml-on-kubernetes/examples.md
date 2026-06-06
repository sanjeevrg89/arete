# Examples — AI/ML on Kubernetes & GKE

Annotated, imitate-able manifest **sketches** for the two canonical frontier patterns: a multi-host
training Job (JobSet + Kueue) and a multi-host inference deployment (LeaderWorkerSet + vLLM). These are
structurally correct and idiomatic, but **field values (image tags, machine families, topologies, NCCL
env, API versions) move fast — verify against current GKE / project docs before applying.** See
[[jobset-leaderworkerset]], [[kueue-advanced]], [[serving-frameworks]], and [[gke-master]] for depth.

---

## 1. Multi-host training: JobSet + Kueue on GKE

End-to-end: a Kueue queue gates admission (gang/all-or-nothing + quota), and a JobSet models the
multi-host run with stable identity and a headless service for rendezvous. Two interchangeable accelerator
variants are shown (TPU multi-host slice, and GPU with GPUDirect).

### 1a. Kueue: queue + quota (admit the whole gang or nothing)

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: tpu-v5p-flavor
spec:
  nodeLabels:
    cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: training-cq
spec:
  namespaceSelector: {}            # all namespaces
  resourceGroups:
  - coveredResources: ["google.com/tpu"]   # use nvidia.com/gpu for the GPU variant
    flavors:
    - name: tpu-v5p-flavor
      resources:
      - name: "google.com/tpu"
        nominalQuota: 256          # total chips this queue may admit
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: training-lq
  namespace: ml-team
spec:
  clusterQueue: training-cq
```

Kueue admits the JobSet only when the **entire** gang fits in `nominalQuota`; otherwise it waits in the
queue instead of half-scheduling and deadlocking. For contiguous placement within a network domain, enable
**Topology-Aware Scheduling** (see [[kueue-advanced]]).

### 1b. JobSet: the multi-host training group (TPU v5p, e.g. 4x4x4 = 64-chip slice)

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: llm-pretrain
  namespace: ml-team
  labels:
    kueue.x-k8s.io/queue-name: training-lq      # <-- routes admission through Kueue
spec:
  failurePolicy:
    maxRestarts: 10                               # whole-gang restart -> resume from checkpoint
  replicatedJobs:
  - name: workers
    replicas: 1                                   # one slice; >1 for data-parallel replicas of slices
    template:
      spec:
        parallelism: 16                           # = #hosts in the slice (64 chips / 4 chips-per-host)
        completions: 16
        backoffLimit: 0                           # fail fast -> let JobSet/Kueue handle restart
        template:
          metadata:
            annotations:
              kueue.x-k8s.io/podset-preferred-topology: "cloud.google.com/gke-nodepool"
          spec:
            # TPU multi-host slice selection: type + topology must match the per-pod chip request.
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
              cloud.google.com/gke-tpu-topology: "4x4x4"
            # JobSet injects a headless Service: pods are addressable as
            #   <jobset>-<rjob>-<jobIndex>-<podIndex>.<subdomain>, giving a stable rendezvous.
            restartPolicy: Never
            containers:
            - name: trainer
              image: REGION-docker.pkg.dev/PROJECT/repo/maxtext:TAG   # verify tag
              ports:
              - containerPort: 8471                # TPU/JAX coordination (verify)
              securityContext:
                privileged: true                   # TPU access (per current GKE recipe)
              resources:
                limits:
                  google.com/tpu: "4"              # chips per host for this topology
              env:
              - name: JAX_COORDINATOR_ADDRESS       # JAX/XLA reads rendezvous from JobSet DNS + env
                value: "llm-pretrain-workers-0-0.llm-pretrain"
              command: ["python", "train.py"]
              args:
              - "--dataset_path=gs://my-bucket/data"
              - "--checkpoint_dir=/ckpt"            # local SSD first tier; async copy to GCS
              volumeMounts:
              - { name: gcs, mountPath: /gcs }      # GCS FUSE CSI: dataset + durable checkpoints
              - { name: scratch, mountPath: /ckpt } # Local SSD: fast first-tier checkpoint
            volumes:
            - name: gcs
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes: { bucketName: my-bucket }
            - name: scratch
              emptyDir: { medium: Memory }          # or a Local SSD-backed volume
```

**GPU variant of the same JobSet** — swap the slice selection and resources:

```yaml
            nodeSelector:
              cloud.google.com/gke-accelerator: nvidia-h100-mega-80gb   # verify family/SKU
            hostNetwork: true                       # max network perf for GPUDirect
            dnsPolicy: ClusterFirstWithHostNet
            tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
            containers:
            - name: trainer
              image: REGION-docker.pkg.dev/PROJECT/repo/torch-fsdp:TAG
              resources:
                limits:
                  nvidia.com/gpu: "8"               # whole node
              env:
              # torchrun/c10d rendezvous off JobSet's stable hostname (rank-0 = jobIndex 0, podIndex 0):
              - { name: MASTER_ADDR, value: "llm-pretrain-workers-0-0.llm-pretrain" }
              - { name: MASTER_PORT, value: "29500" }
              # GPUDirect-TCPXO/NCCL tuning is supplied per the current GKE recipe (plugin + NCCL_* env);
              # values are machine-family-specific -> copy from live GKE GPUDirect docs, don't guess.
              - { name: NCCL_DEBUG, value: "INFO" }
              command: ["torchrun"]
              args: ["--nnodes=16", "--nproc-per-node=8", "--rdzv-backend=c10d",
                     "--rdzv-endpoint=$(MASTER_ADDR):$(MASTER_PORT)", "train.py"]
```

Key points: `replicas/parallelism/completions` encode the gang shape; `backoffLimit: 0` +
`failurePolicy.maxRestarts` make the whole gang restart and resume from checkpoint on any host failure;
the Kueue label gates admission; the JobSet headless Service provides the rendezvous addresses.

---

## 2. Multi-host inference: LeaderWorkerSet + vLLM

When a model + KV cache don't fit one node, one logical replica = **leader + workers** holding a shard-set
and talking over the fabric per token. LWS gives the group stable identity and scales by *group*. Engine
config depth lives in [[serving-frameworks]].

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-llm
  namespace: serving
spec:
  replicas: 2                       # 2 independent serving groups (scale unit = a whole group)
  leaderWorkerTemplate:
    size: 2                         # group size = 1 leader + 1 worker (2 nodes per replica)
    restartPolicy: RecreateGroupOnPodRestart   # a dead worker recreates the whole group
    leaderTemplate:
      metadata:
        labels: { role: leader }
      spec:
        nodeSelector:
          cloud.google.com/gke-accelerator: nvidia-h100-80gb     # verify SKU
        containers:
        - name: vllm-leader
          image: vllm/vllm-openai:TAG                            # verify tag
          # Tensor-parallel within a node (8), pipeline-parallel across the 2 group nodes (2).
          # LWS exposes group membership; vLLM/Ray uses it to form the multi-host engine.
          command: ["sh","-c"]
          args:
          - >
            bash /vllm-workspace/ray_init.sh leader
            --ray_cluster_size=$(LWS_GROUP_SIZE);
            python -m vllm.entrypoints.openai.api_server
            --model /models/big-model
            --tensor-parallel-size 8
            --pipeline-parallel-size 2
            --gpu-memory-utilization 0.92       # leave HBM headroom for KV cache
          env:
          - { name: LWS_GROUP_SIZE, value: "2" }     # injected by LWS; shown for clarity
          ports: [{ containerPort: 8000 }]
          resources:
            limits: { nvidia.com/gpu: "8" }
          volumeMounts:
          - { name: models, mountPath: /models }     # weights staged on a fast volume
          - { name: shm, mountPath: /dev/shm }        # NCCL/Ray need ample shared memory
          readinessProbe:                             # don't route until the engine has loaded weights
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 120                   # cold start = weight load time; size accordingly
            periodSeconds: 10
        volumes:
        - name: models
          csi:
            driver: gcsfuse.csi.storage.gke.io        # or Hyperdisk-ML / Parallelstore for faster load
            volumeAttributes: { bucketName: model-bucket }
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: "16Gi" }
    workerTemplate:
      spec:
        nodeSelector:
          cloud.google.com/gke-accelerator: nvidia-h100-80gb
        containers:
        - name: vllm-worker
          image: vllm/vllm-openai:TAG
          command: ["sh","-c"]
          args: ["bash /vllm-workspace/ray_init.sh worker --ray_address=$(LWS_LEADER_ADDRESS)"]
          # LWS_LEADER_ADDRESS is injected -> worker joins the leader's Ray/engine cluster.
          resources:
            limits: { nvidia.com/gpu: "8" }
          volumeMounts:
          - { name: models, mountPath: /models }
          - { name: shm, mountPath: /dev/shm }
        volumes:
        - name: models
          csi:
            driver: gcsfuse.csi.storage.gke.io
            volumeAttributes: { bucketName: model-bucket }
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: "16Gi" }
---
# Route only to leaders; the leader fronts the multi-host engine for its group.
apiVersion: v1
kind: Service
metadata:
  name: vllm-llm
  namespace: serving
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: vllm-llm
    role: leader
  ports: [{ port: 8000, targetPort: 8000 }]
```

**Autoscaling** ([[autoscaling-kubernetes]]): drive HPA/KEDA off **model-aware** metrics the engine
exports (KV-cache utilization, pending queue, TTFT) — never CPU. LWS scales by **group**, so the HPA
target is the LeaderWorkerSet, adding/removing whole leader+worker groups.

**Routing** ([[serving-frameworks]], guide §4): in front of multiple models/replicas, the **GKE Inference
Gateway** / Gateway API Inference Extension does model-aware, load-aware, cache/prefix-aware routing and
LoRA-adapter multiplexing — superior to round-robin for LLM serving.

---

## Notes on imitation

- These are **sketches**: TPU coordination ports, GPUDirect/NCCL env, the `ray_init.sh` bootstrap, and
  API versions (`v1alpha2`/`v1beta1`/`v1`) vary by version — pull current values from live docs.
- Always pair multi-host training with Kueue gang admission; never let a partial gang start.
- Always set a readiness probe that reflects real weight-load time, and stage weights on a fast volume to
  cut cold start.
- Keep multi-host slices contiguous (Topology-Aware Scheduling) so collectives hit line rate.
