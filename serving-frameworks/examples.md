# Examples — Serving Frameworks on Kubernetes

Canonical, correct-in-shape manifests to imitate. They are **sketches**: image tags, model names,
resource counts, and flags are illustrative — **verify current images/flags against the engine's docs**
(this space changes monthly, 2026). Apply your cluster's conventions (registry, accelerator class,
gateway, namespaces). See `serving-frameworks-guide.md` for the *why* behind each choice.

---

## 1. vLLM — single-host `Deployment` (one pod = one replica)

One model that fits on a single node (here 1 GPU; bump `tensor-parallel-size` and GPU count to shard
within the node). OpenAI-compatible server, prefix caching on, generous startup probe, big `/dev/shm`.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-vllm
  labels: { app: llama-vllm }
spec:
  replicas: 2                       # data-parallel replicas; the unit of (auto)scaling
  selector: { matchLabels: { app: llama-vllm } }
  template:
    metadata:
      labels: { app: llama-vllm }
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest          # PIN a digest in production
        args:
          - "--model=meta-llama/Llama-3.1-8B-Instruct"
          - "--tensor-parallel-size=1"          # = number of GPUs in this pod
          - "--gpu-memory-utilization=0.90"     # KV budget: leave headroom; watch OOM/preemption
          - "--max-model-len=8192"              # bound context to fit the KV cache
          - "--enable-prefix-caching"           # reuse shared system-prompt / RAG prefixes
          # - "--quantization=fp8"              # verify accuracy on YOUR evals before enabling
        ports:
          - { name: http, containerPort: 8000 }
        env:
          - name: HF_TOKEN
            valueFrom: { secretKeyRef: { name: hf-token, key: token } }
        resources:
          limits:
            nvidia.com/gpu: "1"                 # must match tensor-parallel-size
        volumeMounts:
          - { name: dshm,  mountPath: /dev/shm }        # NCCL/Ray need real shared memory
          - { name: cache, mountPath: /root/.cache/huggingface }  # cache weights; faster restarts
        readinessProbe:                          # gate traffic on /health
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
        startupProbe:                            # model load is SLOW — give it minutes
          httpGet: { path: /health, port: 8000 }
          failureThreshold: 60
          periodSeconds: 10
      volumes:
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 8Gi }
        - name: cache
          emptyDir: {}                           # or a PVC / hostPath for persistent weight cache
---
apiVersion: v1
kind: Service
metadata: { name: llama-vllm }
spec:
  selector: { app: llama-vllm }
  ports: [ { name: http, port: 80, targetPort: 8000 } ]
```

> Front the Service with an **Inference Gateway** (Gateway API Inference Extension / GKE Inference
> Gateway) for prefix-/KV-/load-aware routing instead of round-robin — see the guide §8.
> Autoscale on **queue depth / in-flight**, not CPU — see `[[autoscaling-kubernetes]]`.

---

## 2. vLLM multi-host on **LeaderWorkerSet** (one replica spans nodes)

When weights + KV exceed a single node, one logical replica spans multiple pods. **LeaderWorkerSet**
(LWS) creates a leader + N workers as one gang-scheduled group with stable identity and a headless
Service; vLLM runs over Ray across the group. Topology below: 2 pods × 8 GPUs = TP=8 (within each pod,
NVLink) × PP=2 (across the two pods). Details in `[[jobset-leaderworkerset]]`.

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: llama-405b-vllm
spec:
  replicas: 1                       # one multi-host replica (scale this for more replicas)
  leaderWorkerTemplate:
    size: 2                         # group = 1 leader + 1 worker = 2 pods (PP=2 here)
    restartPolicy: RecreateGroupOnPodRestart   # one dead shard → restart the whole group
    leaderTemplate:                 # leader: starts Ray head, runs `vllm serve`, exposes :8000
      spec:
        containers:
        - name: vllm-leader
          image: vllm/vllm-openai:latest        # PIN a digest
          command: ["/bin/sh","-c"]
          args:
            - |
              ray start --head --port=6379 &
              vllm serve meta-llama/Llama-3.1-405B-Instruct \
                --tensor-parallel-size 8 \
                --pipeline-parallel-size 2 \
                --gpu-memory-utilization 0.92 \
                --max-model-len 8192 \
                --enable-prefix-caching
          ports: [ { containerPort: 8000 } ]
          resources: { limits: { nvidia.com/gpu: "8" } }   # 8 GPUs per pod (TP=8)
          volumeMounts:
            - { name: dshm, mountPath: /dev/shm }
        volumes:
          - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
    workerTemplate:                 # workers: join the Ray cluster, run their shards
      spec:
        containers:
        - name: vllm-worker
          image: vllm/vllm-openai:latest
          command: ["/bin/sh","-c"]
          args:
            - |
              ray start --address=$(LWS_LEADER_ADDRESS):6379 --block
          resources: { limits: { nvidia.com/gpu: "8" } }
          volumeMounts:
            - { name: dshm, mountPath: /dev/shm }
        volumes:
          - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
```

Key correctness points:
- **GPUs per pod == TP**; **group size (`size`) == PP stages**. Total GPUs = TP × PP = 8 × 2 = 16.
- The whole group must come up together — **gang-schedule** it; a missing/slow shard stalls the replica.
- LWS injects `LWS_LEADER_ADDRESS` so workers find the Ray head; serve only from the **leader**.
- Big `/dev/shm` on **every** pod or NCCL/Ray hang. Ensure node networking (NVLink/IB/RDMA) and NCCL
  env are correct for cross-node collectives (see guide §11 troubleshooting).
- Verify the exact vLLM multi-host launch invocation against current vLLM docs — the Ray/launch wiring
  evolves.

---

## 3. Triton + TensorRT-LLM — build-then-serve note (no full manifest)

TensorRT-LLM is **ahead-of-time compiled**: you do not point it at a HF model and go. The workflow:

1. **Build** a TRT-LLM engine for a specific `model + GPU arch + precision (e.g. fp8) + parallelism
   (TP/PP)`. The engine is tied to that combination — change any of them and you **rebuild**. Run the
   build as a one-off `Job` (or in CI) on a GPU matching production, writing the engine to a PVC /
   object store.
2. Lay out a **Triton model repository** with the **TensorRT-LLM backend** (the engine, tokenizer,
   and an ensemble/BLS that chains pre-processing → TRT-LLM → post-processing). Triton provides
   in-flight batching, dynamic batching, and the KServe v2 HTTP/gRPC protocol.
3. **Serve** with a Triton `Deployment` mounting the model repository (same GPU shape, large
   `/dev/shm`, long startup probe — engine load is slow), fronted by a Service / KServe / Inference
   Gateway.

```text
# shape only — consult current TensorRT-LLM + Triton docs for exact commands/flags:
#  Job:        trtllm-build  --checkpoint_dir ...  --output_dir /models/engine  (fp8, TP=N, PP=M)
#  Repo:       /models/{preprocess, tensorrt_llm, postprocess, ensemble}/...
#  Deployment: tritonserver --model-repository=/models   (mount the PVC; GPUs == TP×PP)
```

When to prefer this over vLLM/SGLang: you need **peak** NVIDIA performance on a **stable** model set
and can absorb the build step and operational coupling. For a mixed fleet (LLM + vision/embeddings),
Triton serves them all from one server; for disaggregation at scale, drive TRT-LLM via **Dynamo**.

> Do not hand-copy `trtllm-build` flags or repository keys from memory — they change between releases.
> Generate them from the version of TensorRT-LLM / Triton you are actually deploying.
