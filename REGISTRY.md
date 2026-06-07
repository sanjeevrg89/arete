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
| [`mlops-lifecycle`](mlops-lifecycle/) | MLOps maturity 0→1→2, CI/CD/CT, pipelines (Kubeflow/Vertex/Argo), model registry, experiment tracking, deployment patterns. |
| [`ml-observability-monitoring`](ml-observability-monitoring/) | Production ML/LLM monitoring: data/concept/prediction drift, training-serving skew, LLM tracing & online eval, alerting/retrain triggers. |
| [`inference-optimization`](inference-optimization/) | Model-level efficiency: quantization, pruning, distillation, speculative decoding, GQA/MQA/MLA, compilation (TensorRT-LLM/torch.compile). |
| [`data-engineering-feature-stores`](data-engineering-feature-stores/) | ML data pipelines, feature stores (Feast/Tecton/Vertex), point-in-time joins, training-serving skew, data quality/validation. |
| [`ai-networking-collectives`](ai-networking-collectives/) | Collectives (all-reduce/all-gather/all-to-all), NCCL/RCCL, InfiniBand/RoCE/GPUDirect-RDMA, topology-aware placement, comms debugging. |
| [`fine-tuning-peft`](fine-tuning-peft/) | Fine-tuning LLMs: full vs PEFT, LoRA/QLoRA/DoRA, SFT/instruction tuning, adapters & multi-LoRA serving, memory math, eval. |
| [`ml-system-design`](ml-system-design/) | End-to-end ML system design (architect + interview): problem→metric→data→model→eval→serving→monitoring framework, archetypes, tradeoffs. |
| [`ml-evaluation-evals`](ml-evaluation-evals/) | Evaluating ML/LLM systems: classical metrics, benchmarks, LLM-as-judge (+ bias controls), RAG/agent eval, A/B, eval-in-CI. |
| [`staff-plus-engineering`](staff-plus-engineering/) | Staff/Principal/Distinguished competencies: archetypes, scope/leverage, influence, design docs/RFCs, technical strategy. |
| [`responsible-ai-governance`](responsible-ai-governance/) | Responsible AI: NIST AI RMF, EU AI Act, model cards, fairness/bias, LLM safety & red-teaming, privacy, auditability. |
| [`recsys-ranking`](recsys-ranking/) | Recommender & ranking systems at scale: retrieval→ranking→re-ranking funnel, two-tower, DLRM/DCN, multi-task, online eval, position bias. |
| [`multimodal-ml`](multimodal-ml/) | Multimodal AI: CLIP/SigLIP, VLMs (encoder→projector→LLM), speech (Whisper/TTS), video, diffusion generation, multimodal RAG & serving. |
| [`time-series-forecasting`](time-series-forecasting/) | Time-series & classical/tabular ML: forecasting/classification/anomaly/imputation, ARIMA/ETS/Theta/Prophet, GBDT (LightGBM/XGBoost) on lag features, deep (DeepAR/N-HiTS/TFT/PatchTST) & foundation models (TimesFM/Chronos), rolling-origin backtesting, leakage, intervals. |
| [`distributed-systems-fundamentals`](distributed-systems-fundamentals/) | CAP/PACELC/FLP, consensus (Paxos/Raft), replication & consistency models, quorums, sharding, clocks, 2PC/sagas, idempotency, fencing tokens. |
| [`ml-compilers-codegen`](ml-compilers-codegen/) | ML compilers: MLIR, XLA (HLO/StableHLO, fusion, PJRT), Triton (TTIR→TTGIR→PTX), torch.compile/Inductor, TensorRT, kernel codegen. |
| [`gpu-performance-engineering`](gpu-performance-engineering/) | Roofline, Nsight Compute/Systems, occupancy/memory analysis, eBPF + NCCL cross-rank profiling, straggler diagnosis, MLPerf benchmarking methodology. |
| [`experimentation-causal-inference`](experimentation-causal-inference/) | Online controlled experiments / A/B testing: OEC, guardrails, SRM, CUPED, sequential testing, interference; causal inference (DiD/RD/IV/PSM, uplift/CATE). |
| [`pretraining-data-tokenizers`](pretraining-data-tokenizers/) | Web-scale pretraining-data curation (extraction, filtering, MinHash-LSH dedup, decontamination, mixtures) + tokenizer engineering (BPE/Unigram, vocab, fertility). |
| [`graph-ml-gnns`](graph-ml-gnns/) | Graph ML & GNNs: message passing, GCN/GraphSAGE/GAT/GIN, scalability (neighbor/Cluster-GCN sampling), PyG/DGL, recsys/fraud/molecules, GNN4TS. |
| [`embedding-model-training`](embedding-model-training/) | Training embedding/retrieval models: contrastive/InfoNCE, hard-negative mining (false negatives, positive-aware), cross-encoder distillation, Matryoshka, MTEB/BEIR. |
| [`adversarial-ml-robustness`](adversarial-ml-robustness/) | Model-level adversarial ML (defense): NIST AML taxonomy & MITRE ATLAS, evasion/poisoning/backdoors/extraction/membership-inference, adversarial training, robustness eval (AutoAttack/RobustBench). |
| [`privacy-preserving-ml`](privacy-preserving-ml/) | PETs for ML: differential privacy (DP-SGD/accounting), federated learning (FedAvg/secure aggregation), HE/MPC/TEEs, machine unlearning, privacy budgets. |
| [`edge-on-device-ml`](edge-on-device-ml/) | Edge/mobile/embedded deployment: ExecuTorch, LiteRT/TFLite, ONNX Runtime, Core ML, llama.cpp; NPUs/TinyML, on-device quantization, conversion & parity validation. |
| [`ai-research-science`](ai-research-science/) | Research-scientist depth across training/inference/fine-tuning/RL/RLHF: architectures, objectives, optimizers, scaling laws, reward modeling, PPO/DPO/GRPO/RLVR theory, test-time compute, open problems. |
| [`engineering-lifecycle`](engineering-lifecycle/) | **Process meta:** Define→Plan→Build→Verify→Review→Ship for AI infra, with the gate between each stage; routes to the stage skills. |
| [`spec-driven-development`](spec-driven-development/) | **Define:** clarify before building — interview, write an AI-infra spec with testable acceptance criteria, SLOs, cost & failure modes; review gate. |
| [`task-planning-decomposition`](task-planning-decomposition/) | **Plan:** decompose into small verifiable steps, sequence riskiest/most-expensive first, plan for partial failure; approach-review gate. |
| [`test-driven-development`](test-driven-development/) | **Build:** red-green-refactor for infra/ML — table tests, envtest, eval-as-test (invariants not exact tokens), determinism, race; CI gate. |
| [`verification-and-debugging`](verification-and-debugging/) | **Verify:** prove it works (e2e, eval gates, reproducibility) + systematic root-cause debugging for distributed/GPU/ML; regression-test gate. |
| [`code-review-discipline`](code-review-discipline/) | **Review:** correctness/blast-radius/security/simplicity lens, IaC & manifest care, feedback etiquette; approved-against-checklist gate. |
| [`shipping-and-release`](shipping-and-release/) | **Ship:** small, reversible, watched — progressive delivery, model canary/champion-challenger, tested rollback, monitoring-before-ramp gate. |
| [`accelerator-memory-estimator`](accelerator-memory-estimator/) | **Doer:** estimate GPU/TPU memory for training/inference (weights+grads+optimizer+activations, KV cache) and recommend a fitting strategy (FSDP/TP/quant/QLoRA). |
| [`k8s-manifest-scaffolder`](k8s-manifest-scaffolder/) | **Doer:** generate production-grade Kubernetes manifests from a short spec — right kind + probes/limits/securityContext/PDB/HPA/NetworkPolicy baked in. |
| [`triton-kernel-authoring`](triton-kernel-authoring/) | **Doer:** write & optimize Triton GPU kernels — tile model, masking, `tl.dot`, autotune, correctness vs PyTorch, benchmark vs roofline. |

## Cross-link graph (high-traffic edges)
- The four `kubernetes-*` skills form the platform core; `aiml-on-kubernetes` is the umbrella that routes
  into `kueue-advanced`, `jobset-leaderworkerset`, `ml-frameworks`, `training-frameworks`,
  `serving-frameworks`, `autoscaling-kubernetes`, `slurm-hpc-on-kubernetes`, and `gke-master`.
- `gke-master` underpins the ML skills (TPU/GPU node pools, networking, storage, observability).
- The LLM-app stack chains `llm-app-agent-frameworks` → `rag-vector-databases` / `gke-inference-gateway`
  / `serving-frameworks`, guarded by `ai-security-on-gke`.
- Post-training chains `rl-rlhf-frameworks` → `ray-on-kubernetes` + `serving-frameworks` (rollouts) +
  `training-frameworks` (updates); `maxtext-jax-llm` + `ml-checkpointing-orbax` cover the JAX/TPU path.
- The **ML lifecycle / craft** layer ties it together: `ml-system-design` is the architect hub →
  `data-engineering-feature-stores` → `mlops-lifecycle` → `ml-evaluation-evals` →
  `ml-observability-monitoring`, with `inference-optimization` + `fine-tuning-peft` on the model axis,
  `recsys-ranking` / `multimodal-ml` as domains, `ai-networking-collectives` under training at scale,
  and `responsible-ai-governance` as the cross-cutting guardrail.
- `staff-plus-engineering` is the non-technical multiplier that turns the rest into org-level impact.

## Library map (7 clusters)
1. **Language/platform:** `go-best-practices`, the four `kubernetes-*`, `gke-master`, `autoscaling-kubernetes`.
2. **Batch/scheduling:** `kueue-advanced`, `jobset-leaderworkerset`, `slurm-hpc-on-kubernetes`, `ray-on-kubernetes`.
3. **Model compute & perf:** `ml-frameworks`, `training-frameworks`, `maxtext-jax-llm`, `ml-checkpointing-orbax`,
   `ai-networking-collectives`, `fine-tuning-peft`, `inference-optimization`, `ml-compilers-codegen`,
   `gpu-performance-engineering`, `ai-research-science` (the research/science layer over all of these).
4. **Serving/apps:** `serving-frameworks`, `gke-inference-gateway`, `llm-app-agent-frameworks`,
   `rag-vector-databases`, `multimodal-ml`, `recsys-ranking`, `embedding-model-training`, `edge-on-device-ml`.
5. **ML lifecycle & craft:** `aiml-on-kubernetes`, `ml-system-design`, `mlops-lifecycle`,
   `data-engineering-feature-stores`, `ml-evaluation-evals`, `ml-observability-monitoring`, `rl-rlhf-frameworks`,
   `experimentation-causal-inference`, `pretraining-data-tokenizers`.
6. **CS foundations & breadth:** `distributed-systems-fundamentals`, `graph-ml-gnns`, `time-series-forecasting`.
7. **Trust & leadership:** `ai-security-on-gke`, `responsible-ai-governance`, `staff-plus-engineering`,
   `adversarial-ml-robustness`, `privacy-preserving-ml`.
8. **Engineering process (Define→Plan→Build→Verify→Review→Ship):** `engineering-lifecycle` (meta),
   `spec-driven-development`, `task-planning-decomposition`, `test-driven-development`,
   `verification-and-debugging`, `code-review-discipline`, `shipping-and-release`.
9. **Doer / tools (produce an artifact on invocation):** `accelerator-memory-estimator`,
   `k8s-manifest-scaffolder`, `triton-kernel-authoring`.

> Maintenance: add a new skill as a directory following `SKILL-AUTHORING-SPEC.md`, then add a row here.
