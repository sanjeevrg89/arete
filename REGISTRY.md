# Skill Registry

Index of every skill in this library. Each skill is a self-contained directory consumable by
Claude Code (`SKILL.md`), AGENTS.md-compatible agents / agentic IDEs (`AGENTS.md`), and Gemini CLI
(`GEMINI.md`). The `description` in each `SKILL.md` is the router that decides when it loads.

| Skill | Domain |
|-------|--------|
| [`go-best-practices`](go-best-practices/) | World-class Go aligned to the Google Go Style Guide — naming, errors, concurrency, testing, review. |
| [`kubernetes-expert`](kubernetes-expert/) | End-to-end Kubernetes practitioner mastery: workloads, scheduling, networking, storage, security, prod ops, debugging. |
| [`kubernetes-controller-expert`](kubernetes-controller-expert/) | Writing controllers: controller-runtime, client-go informers/workqueues, reconcile, finalizers, status, testing. |
| [`kubernetes-operator-expert`](kubernetes-operator-expert/) | Operator pattern: CRD/API design, CEL, conversion & admission webhooks, kubebuilder/Operator SDK, OLM. |
| [`kubernetes-internals-expert`](kubernetes-internals-expert/) | Source-level internals: apiserver/etcd/scheduler/kubelet/kube-proxy, DRA, dataplane, control-plane debugging. |
| [`aiml-on-kubernetes`](aiml-on-kubernetes/) | Umbrella: training/inference/fine-tuning/RL/RLHF/agentic on K8s & GKE; accelerators, orchestration, observability. |
| [`kueue-advanced`](kueue-advanced/) | Kueue: Workloads, ClusterQueue/Cohort quota, gang admission, preemption, TAS, ProvisioningRequest, MultiKueue. |
| [`jobset-leaderworkerset`](jobset-leaderworkerset/) | JobSet (multi-host training) + LeaderWorkerSet (multi-host inference): gang restart, topology, networking. |
| [`ml-frameworks`](ml-frameworks/) | PyTorch, JAX, XLA, and the GPU/TPU substrate: compilers, parallelism primitives, performance mental models. |
| [`serving-frameworks`](serving-frameworks/) | LLM inference: vLLM, SGLang, Dynamo, Triton, TensorRT-LLM, Ray Serve, KServe; KV cache, batching, disaggregation. |
| [`training-frameworks`](training-frameworks/) | Distributed training: DDP/FSDP, DeepSpeed, Megatron/NeMo, Ray Train, MaxText; ND parallelism, checkpointing. |
| [`slurm-hpc-on-kubernetes`](slurm-hpc-on-kubernetes/) | Slurm/HPC on K8s: Slinky/SUNK, Volcano vs Kueue, MPI Operator, RDMA/InfiniBand, Slurm-vs-K8s decisions. |
| [`gke-master`](gke-master/) | GKE specifics: Standard/Autopilot, TPU/GPU node pools, NAP, Dataplane V2, GPUDirect, Workload Identity, GCS FUSE. |
| [`autoscaling-kubernetes`](autoscaling-kubernetes/) | HPA/VPA/Cluster Autoscaler/Karpenter/KEDA/NAP + ProvisioningRequest; ML/GPU/inference scaling and tuning. |
| [`rl-rlhf-frameworks`](rl-rlhf-frameworks/) | RL/RLHF/RLAIF post-training: PPO/DPO/GRPO, reward models, actor/rollout/learner loop; TRL, veRL, OpenRLHF, NeMo-RL, RLlib. |
| [`ray-on-kubernetes`](ray-on-kubernetes/) | Ray + KubeRay: RayCluster/RayJob/RayService, Train/Tune/Serve/Data/RLlib, placement groups, autoscaling, GCS fault tolerance. |
| [`maxtext-jax-llm`](maxtext-jax-llm/) | MaxText + JAX LLM stack on TPU/GPU: Flax/Optax/Grain/Pathways, named-axis sharding, multislice, JetStream serving. |
| [`ml-checkpointing-orbax`](ml-checkpointing-orbax/) | Checkpointing at scale: Orbax (async/sharded/emergency), torch DCP, Multi-Tier Checkpointing, resiliency & goodput. |
| [`llm-app-agent-frameworks`](llm-app-agent-frameworks/) | LLM apps & agents: ADK, LangChain/LangGraph, LlamaIndex, MCP, tool use, multi-agent, eval, deployment on K8s. |
| [`rag-vector-databases`](rag-vector-databases/) | RAG pipelines + vector DBs: chunking, hybrid retrieval, reranking, HNSW/IVF-PQ, Milvus/Qdrant/Weaviate/pgvector, eval. |
| [`ai-security-on-gke`](ai-security-on-gke/) | Defensive AI security: OWASP LLM Top 10, guardrails (Model Armor), gVisor sandboxing, supply chain, Workload Identity, egress control. |
| [`gke-inference-gateway`](gke-inference-gateway/) | LLM-aware routing: Gateway API Inference Extension, InferencePool/EPP, KV-cache/prefix/LoRA-aware load balancing, canary. |

## Cross-link graph (high-traffic edges)
- The four `kubernetes-*` skills form the platform core; `aiml-on-kubernetes` is the umbrella that routes
  into `kueue-advanced`, `jobset-leaderworkerset`, `ml-frameworks`, `training-frameworks`,
  `serving-frameworks`, `autoscaling-kubernetes`, `slurm-hpc-on-kubernetes`, and `gke-master`.
- `gke-master` underpins the ML skills (TPU/GPU node pools, networking, storage, observability).
- The LLM-app stack chains `llm-app-agent-frameworks` → `rag-vector-databases` / `gke-inference-gateway`
  / `serving-frameworks`, guarded by `ai-security-on-gke`.
- Post-training chains `rl-rlhf-frameworks` → `ray-on-kubernetes` + `serving-frameworks` (rollouts) +
  `training-frameworks` (updates); `maxtext-jax-llm` + `ml-checkpointing-orbax` cover the JAX/TPU path.

> Maintenance: add a new skill as a directory following `SKILL-AUTHORING-SPEC.md`, then add a row here.
