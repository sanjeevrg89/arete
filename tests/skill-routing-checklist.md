# Skill Routing Test Checklist

One discriminating prompt per skill (46). Use it to confirm each skill is installed and **routes
correctly** — i.e. the agent picks the expected skill for a task in that domain.

## How to run
In Claude Code (with the skills installed via `./install.sh claude`), paste each prompt **prefixed
with a routing instruction** so it tells you what it chose:

> **Pick the right skill(s) for this task, name them and why, then answer/build:** `<prompt>`

Check the box if the agent names the **expected** skill (neighbors in _italics_ are acceptable extra
pulls — many real tasks span several skills). To list everything at once: `/skills` should show 46.

> Note: this validates **Claude Code** model-routing. A flat-markdown loader (Gemini-style) routes
> differently (retrieval or always-on) — there, confirm the relevant `<name>.md` is in scope instead.

---

## 1. Language / platform
- [ ] **`go-best-practices`** — "Review this Go service for idiomatic style, error wrapping, and goroutine lifecycle, and fix the issues."
- [ ] **`kubernetes-expert`** — "A pod is in CrashLoopBackOff and another is Pending — diagnose it, then write a production Deployment with probes, resource limits, and a PodDisruptionBudget." _(kubernetes-internals-expert)_
- [ ] **`kubernetes-controller-expert`** — "Write a controller-runtime reconciler with a finalizer, owner references, and status conditions, and explain how to avoid hot-loop requeues." _(go-best-practices)_
- [ ] **`kubernetes-operator-expert`** — "Design a CRD with a status subresource and CEL validation and scaffold a kubebuilder operator with a conversion webhook." _(kubernetes-controller-expert)_
- [ ] **`kubernetes-internals-expert`** — "Explain the API server request flow (auth→admission→etcd) and how the watch cache and resourceVersion work; my watches go stale."
- [ ] **`gke-master`** — "Create a GKE cluster with TPU v5p and H100 node pools, GPUDirect networking, Workload Identity, and GCS FUSE for model loading." _(aiml-on-kubernetes)_
- [ ] **`autoscaling-kubernetes`** — "Configure autoscaling for an LLM inference service: HPA on custom metrics, KEDA scale-to-zero, and Karpenter vs Cluster Autoscaler for GPU nodes." _(gke-master)_

## 2. Batch / scheduling
- [ ] **`kueue-advanced`** — "Set up Kueue with a ClusterQueue, ResourceFlavors, cohort borrowing, and gang admission for TPU training; some Workloads are stuck Inadmissible." _(jobset-leaderworkerset)_
- [ ] **`jobset-leaderworkerset`** — "Build a multi-host LLM training job with JobSet (startup + failure policy) and a multi-host vLLM inference deployment with LeaderWorkerSet." _(kueue-advanced, serving-frameworks)_
- [ ] **`slurm-hpc-on-kubernetes`** — "An HPC team uses sbatch and an ML team uses Kubernetes — should we run Slurm on K8s (Slinky), use Volcano/Kueue, and how do MPI jobs fit?"
- [ ] **`ray-on-kubernetes`** — "Deploy a RayCluster with KubeRay for distributed training and a RayService for serving, with the Ray autoscaler and placement groups for gang scheduling." _(rl-rlhf-frameworks)_

## 3. Model compute & perf
- [ ] **`ml-frameworks`** — "Explain JAX sharding with Mesh/PartitionSpec and how XLA fuses ops; my torch.compile keeps recompiling — why?" _(ml-compilers-codegen)_
- [ ] **`training-frameworks`** — "Choose a parallelism strategy (FSDP vs DeepSpeed ZeRO vs Megatron TP/PP) to train a 70B model on 256 GPUs and outline the config." _(ml-frameworks)_
- [ ] **`maxtext-jax-llm`** — "Train a Llama-class model with MaxText on a multi-host TPU v5p slice — sharding, multislice — and serve it with JetStream." _(training-frameworks, ml-checkpointing-orbax)_
- [ ] **`ml-checkpointing-orbax`** — "Set up async, sharded Orbax checkpointing for a large JAX run, with multi-tier (local SSD + GCS) for fast restart."
- [ ] **`ai-networking-collectives`** — "My 512-GPU run is at 40% MFU — diagnose the NCCL/collective and InfiniBand/topology bottleneck and tune it." _(gpu-performance-engineering, training-frameworks)_
- [ ] **`fine-tuning-peft`** — "Fine-tune Llama-8B with QLoRA on a single GPU — target modules, memory budget, chat templating — and serve multiple LoRA adapters."
- [ ] **`inference-optimization`** — "Make a 70B model cheaper to serve: AWQ/FP8 quantization, speculative decoding, TensorRT-LLM — what changes for latency- vs throughput-bound?" _(serving-frameworks)_
- [ ] **`ml-compilers-codegen`** — "Explain how XLA fuses ops and lowers HLO to PTX/Triton, and how to read the IR to debug a fusion that isn't happening." _(ml-frameworks)_
- [ ] **`gpu-performance-engineering`** — "Profile my GPU kernel with Nsight Compute using the roofline model — memory- or compute-bound? nvidia-smi shows 100% but throughput is low across ranks." _(ai-networking-collectives)_
- [ ] **`ai-research-science`** — "Explain the theory behind DPO vs PPO vs GRPO, reward overoptimization/Goodhart, and Chinchilla scaling laws — with the open problems." _(rl-rlhf-frameworks)_

## 4. Serving / apps
- [ ] **`serving-frameworks`** — "Compare vLLM, SGLang, TensorRT-LLM, and Triton for high-throughput serving and recommend one for disaggregated prefill/decode." _(inference-optimization)_
- [ ] **`gke-inference-gateway`** — "Set up LLM-aware routing with the Gateway API Inference Extension — InferencePool, KV-cache/prefix-aware load balancing, and canary model rollout." _(serving-frameworks)_
- [ ] **`llm-app-agent-frameworks`** — "Build a tool-using agent with LangGraph/ADK, wire MCP tools, make it durable/resumable, and sandbox untrusted tool execution." _(rag-vector-databases, ai-security-on-gke)_
- [ ] **`rag-vector-databases`** — "Design a production RAG pipeline — chunking, hybrid retrieval + reranking, Qdrant/Milvus on Kubernetes; my recall is poor." _(embedding-model-training)_
- [ ] **`multimodal-ml`** — "Train and serve a vision-language model — encoder→projector→LLM architecture, data, and the serving token-budget implications." _(ml-frameworks)_
- [ ] **`recsys-ranking`** — "Design a large-scale ranking system — two-tower retrieval → DLRM/DCN ranking → re-ranking, multi-task objectives, online eval." _(ml-system-design)_
- [ ] **`embedding-model-training`** — "Train a retrieval embedding model — contrastive InfoNCE with hard-negative mining (avoiding false negatives) — and evaluate on MTEB/BEIR." _(rag-vector-databases)_
- [ ] **`edge-on-device-ml`** — "Deploy a quantized LLM on a phone with ExecuTorch/Core ML — conversion pipeline, INT4, NPU delegates, and validate parity with the server model." _(inference-optimization)_

## 5. ML lifecycle & craft
- [ ] **`aiml-on-kubernetes`** — "Lay out an end-to-end ML platform on GKE for training and serving LLMs — accelerators, orchestration, storage, observability, cost." _(routes broadly)_
- [ ] **`ml-system-design`** — "Design an end-to-end ML system for a video recommendation feed — problem framing, data, model, evaluation, serving, monitoring." _(recsys-ranking)_
- [ ] **`mlops-lifecycle`** — "Build a CI/CD/CT pipeline with a model registry and experiment tracking, with canary and champion/challenger deployment."
- [ ] **`data-engineering-feature-stores`** — "Build an ML feature pipeline with a feature store, point-in-time joins to avoid leakage, and streaming features with Kafka/Flink; kill training-serving skew."
- [ ] **`ml-evaluation-evals`** — "Set up an eval harness for my LLM app — LLM-as-judge with bias controls, RAG faithfulness metrics, and eval gates in CI."
- [ ] **`ml-observability-monitoring`** — "My model's accuracy is silently degrading in prod — set up drift, training-serving skew, and LLM tracing/online-eval with retrain triggers."
- [ ] **`rl-rlhf-frameworks`** — "Stand up an RLHF pipeline: reward model + PPO/GRPO with vLLM rollouts; which framework (TRL/veRL/OpenRLHF) and how to place actors vs learners?" _(ai-research-science, ray-on-kubernetes)_
- [ ] **`experimentation-causal-inference`** — "Design a trustworthy A/B test — OEC, guardrails, power, SRM, CUPED — and estimate impact when I can't randomize (diff-in-differences)." _(ml-evaluation-evals)_
- [ ] **`pretraining-data-tokenizers`** — "Build a web-scale pretraining-data pipeline — extraction, quality filtering, MinHash-LSH dedup, decontamination — and choose a tokenizer (vocab/fertility)."

## 6. CS foundations & breadth
- [ ] **`distributed-systems-fundamentals`** — "Explain CAP/PACELC and why I shouldn't roll my own leader election with heartbeats; when do I need consensus (Raft) and fencing tokens?"
- [ ] **`graph-ml-gnns`** — "Build a GNN for fraud on a transaction graph — GraphSAGE with neighbor sampling — and scale to a billion-edge graph without leakage." _(recsys-ranking)_
- [ ] **`time-series-forecasting`** — "Forecast demand across thousands of series — LightGBM on lag features vs deep vs foundation models — with rolling-origin backtesting and prediction intervals."

## 7. Trust & leadership
- [ ] **`ai-security-on-gke`** — "Harden an LLM serving workload on GKE against prompt injection and data exfiltration — guardrails, gVisor sandboxing, Workload Identity, egress NetworkPolicy." _(adversarial-ml-robustness)_
- [ ] **`responsible-ai-governance`** — "Set up responsible-AI governance — NIST AI RMF mapping, model cards, fairness evaluation, and red-teaming." _(ml-evaluation-evals)_
- [ ] **`staff-plus-engineering`** — "I'm a senior engineer aiming for Staff — what changes (scope, influence, design docs, strategy), and help me write an RFC."
- [ ] **`adversarial-ml-robustness`** — "Evaluate and harden my model against adversarial examples and data poisoning — adaptive attacks, AutoAttack, and detecting backdoors in third-party weights." _(ai-security-on-gke)_
- [ ] **`privacy-preserving-ml`** — "Train on sensitive data with differential privacy (DP-SGD + accounting) and federated learning with secure aggregation; handle a right-to-be-forgotten deletion." _(responsible-ai-governance)_

---

**Pass criteria:** the agent names the bolded skill for its prompt (extra _italic_ neighbors are fine).
If a prompt routes to the *wrong* skill or none, sharpen that skill's `SKILL.md` `description` (the
router) — add the missing trigger terms — and re-test.
