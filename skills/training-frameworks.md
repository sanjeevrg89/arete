---
name: training-frameworks
description: Expert distributed-training knowledge for training large models on hundreds-to-thousands of
  accelerators — the parallelism strategies and the frameworks that implement them. Use when choosing,
  configuring, or debugging multi-GPU/TPU training: data parallel (DDP), ZeRO/FSDP/FSDP2 sharding,
  tensor parallel (Megatron), pipeline parallel (GPipe/1F1B/interleaved, bubble), sequence/context
  parallel, expert parallel (MoE all-to-all), 3D/ND parallelism, activation checkpointing, gradient
  accumulation, bf16/fp8 mixed precision, NCCL/RCCL collectives & comm overlap, distributed/async
  checkpointing, elastic/fault-tolerant training, MFU/goodput. Covers PyTorch DDP/FSDP/TorchTitan/
  torchrun, DeepSpeed (ZeRO/Offload/Infinity/MoE), Megatron-LM/Megatron-Core, NVIDIA NeMo, HF
  Accelerate/Trainer/TRL, PyTorch Lightning, Ray Train, JAX MaxText/Levanter/Paxml/Pathways, and how
  these map onto Kubeflow Trainer/PyTorchJob/MPIJob/JobSet. Triggers: torchrun/deepspeed launch args,
  FSDP/ZeRO config, "model won't fit", low MFU, OOM at scale, checkpoint sharding, MoE training, 3D
  parallelism, picking tp/pp/dp degrees.
---

# Distributed Training Frameworks & Parallelism

Apply the judgment of an engineer who has trained frontier-scale dense and MoE models on thousands of
accelerators for years: pick the **fewest parallelism axes that make the model fit**, map the chattiest
comms onto the fastest interconnect, keep every device doing math (high MFU), and survive failures
without losing the run.

## How to use this skill

1. **Read `training-frameworks-guide.md`** in this directory — the full reference (parallelism
   strategies, framework selection, comms, checkpointing/fault tolerance, decision guide,
   anti-patterns, troubleshooting). Apply it to the task.
2. For concrete artifacts to imitate — an FSDP2 `torchrun` launch, a Megatron/DeepSpeed config sketch,
   and a `torchrun`→JobSet/Kueue mapping — read **`examples.md`**.
3. Match the surrounding repo's framework and conventions; apply the correctness rules
   (topology-aware mesh shape, comm overlap, sharded checkpoints, resumable data loader) regardless.
   Never fabricate a flag/config key — when unsure, verify against the pinned version's docs.

## Essentials (full detail in `training-frameworks-guide.md`)

- **Static state for Adam mixed precision ≈ 16–18 bytes/param** (bf16 params+grads + fp32 master +
  fp32 m,v). That plus activations is your memory budget. If it fits on one GPU → **DDP**; else shard.
- **Parallelism is memory-vs-comms.** DDP replicates everything (limit: model must fit one device).
  **ZeRO/FSDP** shard optimizer(1)→grads(2)→params(3=FSDP `FULL_SHARD`). **TP** splits inside a layer
  (Megatron). **PP** splits across layers (bubble ∝ `(p-1)/m`). **CP/SP** split the sequence. **EP**
  splits MoE experts (all-to-all). Compose into a **3D/ND device mesh**.
- **Topology is the design constraint:** TP/EP must stay inside NVLink/one node (chattiest collectives);
  PP/DP/FSDP can cross nodes; gang-schedule the whole job into one fabric domain.
- **FSDP2** (`fully_shard`, DTensor-based) is the current PyTorch path; use a **HYBRID/2-D mesh**
  multi-node so the param all-gather stays on NVLink. Wrap per transformer block, not whole-model.
- **Always-on memory tools:** selective activation checkpointing, gradient accumulation (skip the
  all-reduce on non-final micro-batches), bf16 (fp8 on the big GEMMs with careful scaling), fused/
  distributed optimizer (ZeRO-1).
- **Comm overlap is MFU.** Overlap grad all-reduce / FSDP all-gather+reduce-scatter / TP comms with
  compute; lost overlap, a CPU-bound data loader, or `m≈p` pipelines are the top MFU killers.
- **Collectives:** all-reduce (DDP/TP), reduce-scatter (FSDP grads), all-gather (FSDP params),
  all-to-all (MoE), P2P (PP). NCCL/RCCL on GPU; XLA collectives on TPU.
- **Checkpoint sharded + async** (PyTorch DCP / Megatron dist / DeepSpeed universal) — never gather to
  rank 0. Use elastic `torchrun`/torchft/Pathways + a **resumable data loader** to recover from node
  loss. Optimize **MFU**, then **goodput**.
- **Framework fit:** TorchTitan/FSDP2 for PyTorch-from-scratch; Megatron-Core/NeMo for frontier
  dense/MoE on NVIDIA; DeepSpeed for ZeRO-3 + offload; HF Accelerate/Trainer/TRL for fine-tune;
  JAX MaxText/Levanter/Paxml+Pathways on TPU; Ray Train to orchestrate. Don't add an axis you don't need.

## Related skills

- `[[ml-frameworks]]` — PyTorch/JAX/XLA compute internals, DTensor, `torch.compile`, GPU/TPU (sibling).
- `[[aiml-on-kubernetes]]` — umbrella for training/inference/RL on K8s & GKE.
- `[[jobset-leaderworkerset]]` — the multi-host gang primitive a `torchrun` rendezvous targets.
- `[[kueue-advanced]]` — gang scheduling, quota, and Topology-Aware Scheduling for training jobs.
- `[[slurm-hpc-on-kubernetes]]` — Slurm/MPI/RDMA, MPIJob, Volcano for HPC-style launches.
- `[[gke-master]]` — TPU/GPU node pools, networking, and placement on GKE.
- `[[serving-frameworks]]` — inference/serving (vLLM, SGLang, TensorRT-LLM), the other side of the stack.

---

# Reference — training-frameworks

# Distributed Training Frameworks & Parallelism — Deep Reference

The single source of truth for **how large models are trained across many accelerators**: the
parallelism strategies, the memory/comms/compute trade-offs, and the frameworks that implement them
(PyTorch FSDP, DeepSpeed, Megatron-Core/NeMo, JAX/MaxText, Ray Train, Kubeflow Trainer). Compute
framework internals (PyTorch eager/compile, JAX/XLA, GPU/TPU) are `[[ml-frameworks]]`; running these
as cluster jobs is `[[aiml-on-kubernetes]]`, `[[kueue-advanced]]`, `[[jobset-leaderworkerset]]`,
`[[slurm-hpc-on-kubernetes]]`, `[[gke-master]]`.

The ecosystem moves fast — this is 2026-current. Flags, config keys, and defaults drift between
releases; treat exact names as "verify against the version you pin." Never invent a flag.

---

## 1. Mental model: why you parallelize at all

A training step has a fixed memory budget per device and a fixed amount of compute. For a model with
`P` parameters in bf16 with an Adam-class optimizer, the *static* state alone is roughly:

- params: 2 bytes/param (bf16) — often a 4-byte fp32 master copy too in mixed precision
- gradients: 2 bytes/param (bf16) or 4 (fp32)
- optimizer state (Adam m, v): 8 bytes/param (fp32) — often 12 with the fp32 master copy

So Adam on a model in mixed precision costs **~16–18 bytes/param** before a single activation. A 70B
model is ~1.1–1.3 TB of static state — it does not fit on one 80GB device. On top of that sit
**activations**, which scale with `batch × seq_len × hidden × layers` and are what actually dominates
memory at long context.

Parallelism is the art of **splitting one of {the batch, the parameters, a tensor dimension, the
layers, the sequence, the experts} across devices** so each holds a fraction, then paying
communication to stitch the math back together. Every strategy trades **memory saved** against
**communication added** (and sometimes recompute). The whole game at scale is: keep every device
busy with math (high **MFU**), keep comms overlapped behind compute, and map the heaviest comms onto
the fastest interconnect.

**The interconnect hierarchy is the thing you design around:**

| Tier | Examples (2026) | Bandwidth | Put here |
|---|---|---|---|
| Intra-node | NVLink/NVSwitch (GH200/GB200 NVL72), TPU ICI | very high (TB/s class) | TP, EP, the chattiest collectives |
| Inter-node fabric | InfiniBand/RoCE (NDR/XDR), TPU ICI across slices | high but lower | PP, DP/FSDP all-gather/reduce-scatter |
| Cross-pod / DC | DCN, multi-slice | lowest | only DP-style, sync-tolerant comms |

The cardinal rule: **tensor/expert parallel must live inside the high-bandwidth domain (one node /
NVLink island); pipeline and data parallel can cross nodes.** Violate this and you stall on comms.

---

## 2. Parallelism strategies (the core toolkit)

### 2.1 Data parallel (DDP)

Replicate the full model on every device; each processes a different micro-batch shard; **all-reduce**
the gradients before the optimizer step so every replica stays in sync. Throughput scales near-linearly
until comms or the global batch size (convergence) becomes the limit.

- PyTorch `DistributedDataParallel` buckets gradients and overlaps the all-reduce with the backward
  pass — the canonical, fastest dense DP. One process per GPU, `ddp` over NCCL.
- **Limit:** every replica holds the *entire* model + optimizer state. DDP works only while the model
  fits on one device. Beyond that you must shard — that is what ZeRO/FSDP do.
- Gradient all-reduce volume is ~`2P` bytes/step regardless of batch; with large clusters this is the
  bottleneck, mitigated by bucketing, `bf16` gradients, and hierarchical/topology-aware all-reduce.

### 2.2 ZeRO and FSDP — sharded data parallel

Same *data-parallel* compute pattern, but **shard the redundant state** across the DP group instead of
replicating it. DeepSpeed calls the stages ZeRO; PyTorch calls the implementation FSDP. They are the
same idea (and FSDP's design is explicitly ZeRO-3-shaped).

**ZeRO stages (memory saved, comms added):**

| Stage | Shards | Extra comms vs DDP | Memory |
|---|---|---|---|
| ZeRO-1 | optimizer state | none (reduce-scatter replaces all-reduce, same volume) | big win, cheap |
| ZeRO-2 | + gradients | none material | bigger win |
| ZeRO-3 / FSDP | + parameters | **all-gather params per layer fwd & bwd** (~+50% comms) | model no longer replicated |

**FSDP mechanics (PyTorch):** parameters are flattened and sharded across the DP group into "units"
(per layer / transformer block). At the start of a unit's forward, an **all-gather** reconstructs the
full params; after use they're freed; backward all-gathers again; gradients are **reduce-scattered**
back to shards. The peak resident params is one unit, not the whole model.

- **FSDP1 vs FSDP2:** FSDP2 (`torch.distributed.fsdp.fully_shard`, the per-parameter-sharding
  rewrite) is the current path in recent PyTorch — it shards on a per-parameter `DTensor` basis
  (cleaner state dicts, better composability with TP/PP, simpler mixed precision, no flat-parameter
  foot-guns). FSDP1 (`FullyShardedDataParallel` wrapper) still exists but new code should target FSDP2.
  Verify the exact API surface against your PyTorch version.
- **Sharding strategy / mesh:** classic FSDP1 had `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2),
  `NO_SHARD` (DDP), and **`HYBRID_SHARD`** — shard within a node, replicate across nodes, so the
  expensive all-gather stays on NVLink and only a smaller all-reduce crosses the slow fabric. FSDP2
  expresses the same via a 2-D `DeviceMesh` (shard dim × replicate dim). HYBRID/2-D is usually the
  right default at multi-node scale.
- **Offload:** CPU offload (params/grads/optimizer to host RAM) trades PCIe/NVLink-C2C bandwidth for
  capacity; DeepSpeed **ZeRO-Offload** (optimizer+grads to CPU) and **ZeRO-Infinity** (params+state to
  NVMe) let you train models far larger than aggregate HBM at a throughput cost. Use when you are
  memory-bound and can tolerate lower MFU; not on the critical path for well-provisioned clusters.
- **ZeRO++**: communication-reduction extensions (quantized weights for the all-gather `qwZ`,
  hierarchical weight partitioning `hpZ` to keep a secondary replica on-node, quantized gradients
  `qgZ`) — cuts ZeRO-3 comms substantially on bandwidth-constrained clusters.

**When FSDP/ZeRO is enough:** dense models up to the tens of billions where the model fits across one
DP group's aggregate memory and you don't need to split a single matmul. Past that — or when a single
layer's activations/params blow the budget — add TP/PP.

### 2.3 Tensor parallel (TP) — intra-layer, Megatron-style

Split *within* a layer: shard the weight matrices of attention and MLP across devices and split the
matmul. Megatron's scheme partitions the first MLP GEMM column-wise and the second row-wise so only
**two all-reduces per transformer block** (one in fwd, one in bwd of each of attention and MLP) are
needed; attention is sharded by heads.

- **Properties:** shards both params *and* activations along the hidden dim → big activation-memory
  relief, but adds an all-reduce on the critical path of *every* layer. That comm is large and
  latency-sensitive → **TP must stay inside NVLink/one node.** Typical TP degree = GPUs per node (4/8),
  rarely more.
- Combines with **sequence parallel** to also shard the parts Megatron-TP leaves replicated
  (LayerNorm/dropout regions), all-gather/reduce-scatter around them — cuts activation memory further
  at no extra comms volume. Almost always enable SP with TP in Megatron-Core.

### 2.4 Pipeline parallel (PP) — inter-layer

Partition the model's *layers* into stages placed on different devices; micro-batches flow through the
pipeline. The cost is the **bubble** — idle time while the pipeline fills and drains.

- **GPipe:** all-forward-then-all-backward; bubble fraction ≈ `(p-1)/m` for `p` stages, `m`
  micro-batches → need `m ≫ p` to amortize.
- **1F1B (PipeDream-Flush):** interleave one-forward-one-backward in steady state → far lower
  activation memory (bounded in-flight micro-batches) at the same bubble.
- **Interleaved 1F1B (Megatron virtual pipeline):** each device owns several non-contiguous layer
  chunks → shrinks the bubble by the number of virtual stages, at the cost of more pipeline
  communication. The standard choice for very deep models.
- **Newer schedules:** zero-bubble / `1F1B-V` style schedules split the backward into input-grad and
  weight-grad to nearly eliminate the bubble. Megatron-Core has been adding these; verify support and
  flag names for your version.
- **Properties:** point-to-point sends of activations between stages — modest, *can* cross nodes.
  Pairs naturally with TP (TP within node, PP across nodes).

### 2.5 Sequence / context parallel

Shard along the **sequence dimension** so each device holds part of the tokens — the only way to fit
very long context (100k–1M+) where activations dominate.

- **Megatron "sequence parallel"** specifically means the SP-with-TP optimization above (sharding the
  TP-replicated regions). Don't confuse it with general long-context **context parallel**.
- **Context parallel** shards the full sequence across a CP group and uses **Ring Attention** (or
  all-to-all/Ulysses-style) to exchange K/V blocks so attention is computed correctly across shards.
  Megatron-Core (`context_parallel_size`), and DeepSpeed-Ulysses implement variants. Essential for
  long-context pretraining and long-RL rollouts.

### 2.6 Expert parallel (EP) — Mixture of Experts

In an MoE layer only a few experts fire per token. **Expert parallel** places different experts on
different devices; a router sends each token to its experts via **all-to-all** (dispatch) and back
(combine).

- The defining comm is the **all-to-all**, which is bandwidth- and latency-heavy and load-imbalance
  sensitive (a hot expert hotspots a device). Keep EP inside the high-bandwidth domain where possible;
  use capacity factors / token dropping or dropless (e.g. grouped-GEMM/MegaBlocks-style) kernels.
- EP composes with TP/DP/PP: e.g. EP across the experts, TP within each expert's matmuls, DP/FSDP for
  the dense (non-expert) parameters. Megatron-Core and DeepSpeed-MoE both support this; this is how
  frontier sparse models (hundreds of B–T params, tens of B active) are trained.

### 2.7 3D / ND parallelism — composition

Real large-scale training composes several axes into an **N-D device mesh**. The canonical "3D" is
**DP × PP × TP**; modern stacks add **SP, CP, EP** for 5–6 axes. You assign each axis a slice of the
mesh and map it to the topology:

- **TP / EP** → innermost, on NVLink / one node (chattiest).
- **CP / SP** → typically with/near TP, on fast links.
- **PP** → across nodes (point-to-point, bubble-tolerant of fabric latency).
- **DP / FSDP** → outermost, can span the whole cluster (sync once per step).

`world_size = dp × pp × tp × cp × ep` (with the MoE caveat that EP overlaps the DP group). Getting the
*shape* right for your model and cluster is the single highest-leverage decision — see §10.

---

## 3. Memory & precision techniques (orthogonal to parallelism, always on the table)

- **Activation checkpointing / recomputation:** don't store activations for every layer; recompute them
  in the backward pass from saved checkpoints. Trades ~30% extra compute for a large activation-memory
  cut. **Selective/granular checkpointing** (recompute only cheap-to-redo ops, e.g. not attention
  softmax/flash) gets most of the memory for far less recompute — prefer it. PyTorch
  `torch.utils.checkpoint` / `apply_activation_checkpointing`; Megatron `--recompute-granularity`.
- **Gradient accumulation & micro-batching:** accumulate grads over `k` micro-batches before the
  optimizer step → larger *effective* global batch without more memory. In FSDP/DDP, use
  `no_sync()`/equivalent on the non-final micro-batches to skip the gradient all-reduce until the last
  one. Micro-batch count also feeds the pipeline (need `m ≫ p`).
- **Mixed precision:** bf16 is the default compute dtype (no loss scaling needed, unlike fp16). Keep an
  **fp32 master copy** of weights + fp32 optimizer state for stability. **fp8** training (E4M3/E5M2 via
  Transformer Engine on Hopper/Blackwell, or scaled-fp8 paths) gives a real throughput win on the big
  GEMMs but needs careful scaling/amax tracking and per-tensor or block scaling — verify recipe and
  numerics for your stack; keep sensitive ops (softmax, norms, master weights) in higher precision.
- **Optimizer at scale:** Adam/AdamW is the workhorse (but 8 bytes/param of state — the thing ZeRO
  shards). **Distributed/fused optimizers** (Megatron distributed optimizer = ZeRO-1 built in; Apex
  fused Adam) cut memory and kernel launches. 8-bit optimizers (bitsandbytes) and newer optimizers
  (Lion, and Muon-style/second-order-ish methods gaining traction in 2026) reduce state or improve
  token-efficiency — evaluate per-workload; the well-trodden path is fused AdamW + ZeRO-1 sharding.

---

## 4. Communication: collectives, overlap, topology

The collectives you must know (NCCL on NVIDIA, RCCL on AMD, XLA collectives on TPU):

| Collective | Used by | Note |
|---|---|---|
| **all-reduce** | DDP grads, Megatron-TP | sum across ranks; = reduce-scatter + all-gather |
| **reduce-scatter** | FSDP/ZeRO grad sharding | each rank gets its shard of the sum |
| **all-gather** | FSDP/ZeRO param reconstruction | reassemble full params from shards |
| **all-to-all** | MoE dispatch/combine, Ulysses CP | every rank exchanges with every rank — pricey |
| **broadcast / P2P send-recv** | PP stage handoff, init | point-to-point |

- **Overlap is everything.** DDP overlaps grad all-reduce with backward; FSDP prefetches the next
  unit's all-gather during current compute (`backward_prefetch`/`forward_prefetch`) and overlaps
  reduce-scatter; Megatron overlaps TP comms and gradient reduction with compute. Lost overlap = the
  most common cause of low MFU.
- **Topology awareness:** NCCL auto-detects NVLink/PCIe/NIC topology and builds rings/trees; set
  `NCCL_*` env (e.g. interface/HCA selection, algorithm) and use the GPUDirect-RDMA stack (right
  NIC↔GPU affinity) on multi-node. SHARP / in-network reduction (InfiniBand) and NVLink-domain
  collectives accelerate large all-reduces. On TPU, XLA's GSPMD/Shardy picks collectives from your
  sharding annotations over ICI/DCN.
- **Diagnose comms** with NCCL tests (`all_reduce_perf`), `NCCL_DEBUG=INFO`, and profiler timelines
  (Nsight Systems / PyTorch profiler / XLA trace) — look for collectives *not* overlapped with compute.

---

## 5. Checkpointing & fault tolerance at scale

At thousands of accelerators something fails constantly; **goodput** (useful training time ÷ wall time)
is dominated by checkpoint cost and recovery speed.

- **Sharded / distributed checkpoints:** never gather the whole model to rank 0. Each rank writes its
  shard in parallel. PyTorch **Distributed Checkpoint (DCP)** (`torch.distributed.checkpoint`) writes
  a resharding-tolerant checkpoint — you can resume on a *different* parallelism layout. This is the
  modern default for FSDP/TP. DeepSpeed and Megatron have their own sharded/"universal" checkpoint
  formats (Megatron-Core "distributed"/torch-dist; DeepSpeed Universal Checkpoint to convert between
  parallelism configs).
- **Asynchronous checkpointing:** copy state to host/staging fast, then flush to storage in the
  background so the training step barely stalls (DCP async save; frameworks add their own). Combine
  with a fast local tier and async upload to object storage.
- **Save cadence:** balance checkpoint cost vs expected work lost on failure (the classic Young/Daly
  optimal-interval logic) — frequent enough that a crash loses minutes, not hours.
- **Elastic / restartable training:** `torchrun`/`torch.distributed.elastic` (TorchElastic) with a
  rendezvous backend (c10d/etcd) lets the job survive node loss and rescale; **`torchft`** and
  in-job hot-spare / failure-domain restart (re-form the process group, reload the last shard, fast-
  forward the data loader) cut recovery from a full job restart to seconds–minutes. On TPU,
  **Pathways** provides resilient multislice with similar goals. The data loader must be
  **checkpointable/deterministic** so you resume mid-epoch without re-seeing or skipping data.
- **MFU / goodput as the north star:** MFU = achieved FLOP/s ÷ peak FLOP/s — your efficiency of
  compute; goodput folds in failures, restarts, and stragglers. Optimize the parallelism shape and
  overlap for MFU, then attack goodput with async checkpoints + fast elastic recovery + straggler
  detection. Quote MFU only when you've actually measured it — don't cite numbers you haven't seen.

---

## 6. PyTorch-native stack

- **DDP** — dense models that fit on one GPU; simplest, fastest DP. `torchrun` + `DistributedDataParallel`.
- **FSDP / FSDP2** — sharded DP for models that don't fit; FSDP2 (`fully_shard`, DTensor-based) is the
  forward path. Compose with TP via `DeviceMesh` + `parallelize_module` and `[[ml-frameworks]]`'s
  DTensor/`torch.compile`.
- **`torch.distributed`** — the primitives: process groups, `DeviceMesh`, collectives, DTensor, the
  pipelining and tensor-parallel APIs.
- **`torchrun` / TorchElastic** — the launcher: sets `RANK`/`WORLD_SIZE`/`LOCAL_RANK`, handles
  rendezvous, restarts on failure (`--nnodes=min:max`, `--max-restarts`). The standard entrypoint.
- **TorchTitan** — PyTorch's *reference* large-scale pretraining codebase: composes FSDP2 + TP + PP +
  CP + `torch.compile` + DCP + Float8 cleanly on a `DeviceMesh`. The best place to see the native
  stack assembled correctly; use it as a template/starting point for from-scratch PyTorch training.
- **Pipeline (`torch.distributed.pipelining`, ex-PiPPy)** — native pipeline schedules (GPipe, 1F1B,
  interleaved) integrated with the rest.

Use the PyTorch-native path when you want full control, `torch.compile`, and to avoid a heavyweight
framework; it now covers the full 3D+ space that previously required Megatron/DeepSpeed.

## 7. DeepSpeed

A training-acceleration library bolted onto PyTorch via a config JSON + `deepspeed`/`torchrun` launch.

- **ZeRO-1/2/3** and **ZeRO-Offload / ZeRO-Infinity** (CPU/NVMe) — its headline feature; train huge
  models on modest HBM. **ZeRO++** for comm reduction.
- **Pipeline parallel** (`PipelineModule`) and **3D parallelism** (with Megatron tensor parallel).
- **DeepSpeed-MoE** — expert parallel for sparse models.
- **MII / inference** is separate (serving lives in `[[serving-frameworks]]`).
- Reach for DeepSpeed when you want ZeRO-3 + aggressive offload with minimal model-code changes, or
  HF-Trainer integration. For dense frontier pretraining, Megatron-Core or TorchTitan often hit higher
  MFU; for fitting an oversized model on limited memory, ZeRO-Infinity is unmatched.

## 8. Megatron-LM / Megatron-Core, and NeMo

- **Megatron-Core** — NVIDIA's library of the canonical **TP + SP + PP (interleaved) + CP + EP + DP**
  building blocks plus a distributed optimizer, fused kernels, Transformer Engine (fp8) integration,
  and distributed checkpointing. The reference 3D/ND-parallel LLM stack; what most frontier *dense and
  MoE* pretraining uses or descends from. Tune degrees via `--tensor-model-parallel-size`,
  `--pipeline-model-parallel-size`, `--context-parallel-size`, `--expert-model-parallel-size`,
  `--sequence-parallel`, `--num-layers-per-virtual-pipeline-stage` (verify exact flags per release).
- **NVIDIA NeMo** — a framework *built on Megatron-Core* that adds data/config/recipe management,
  PEFT/SFT/alignment, multimodal, and launchers (NeMo-Run). Use NeMo when you want batteries-included
  recipes on the Megatron engine; use Megatron-Core directly for maximum control / custom architectures.
- Choose Megatron when you need the highest dense/MoE pretraining MFU on NVIDIA at scale and are
  willing to work within its (powerful, somewhat rigid) structure.

## 9. Higher-level wrappers & orchestration

- **Hugging Face Accelerate** — thin abstraction over DDP/FSDP/DeepSpeed/TP via an `accelerate config`
  + `accelerate launch`; you keep your training loop. Great for portability and fine-tuning.
- **HF Trainer / TRL** — full training loop with FSDP/DeepSpeed plugins; SFT/DPO/PPO/GRPO via TRL. The
  fast path for fine-tuning and post-training of HF models.
- **PyTorch Lightning** — structures the loop and abstracts strategies (`FSDPStrategy`,
  `DeepSpeedStrategy`, `ModelParallelStrategy`). Good for research velocity; less common at frontier
  pretraining scale.
- **Ray Train** — orchestrates distributed training as Ray tasks/actors: handles worker placement,
  fault tolerance, elastic scaling, and integrates with the backends above (`TorchTrainer` runs your
  FSDP/DeepSpeed/Megatron/Lightning code). Use it to run training inside a Ray cluster, especially
  alongside Ray Data preprocessing and Ray-based RL/RLHF; see `[[aiml-on-kubernetes]]` for RayJob.

## 9b. JAX side — SPMD on TPU/GPU via XLA

JAX expresses parallelism as **sharding annotations** over a logical `Mesh`; XLA's GSPMD/**Shardy**
partitioner inserts the collectives. You describe *what* is sharded (named axes like `data`/`fsdp`/
`tensor`/`expert` via `NamedSharding`/`PartitionSpec`), not *how* to communicate. See `[[ml-frameworks]]`
for JAX/XLA internals.

- **MaxText** — Google's reference high-MFU LLM (dense + MoE) training codebase in JAX/Flax for
  TPU and GPU; config-driven mesh axes (`ici_*_parallelism` / `dcn_*_parallelism` for intra- vs
  inter-slice). The JAX analog of TorchTitan/Megatron.
- **Levanter** (with the Haliax named-tensor lib) — readable, reproducible, scalable JAX training
  (Stanford CRFM).
- **Paxml / Praxis** and **Pathways** — Google's large-scale JAX training framework and the
  single-controller runtime for **multislice / multipod** resilient training. Pathways is how you go
  past one TPU slice with fault tolerance.
- Choose JAX/MaxText when on TPU (or want SPMD's clean scaling and don't need PyTorch's ecosystem);
  PyTorch otherwise. The two ecosystems implement the *same* parallelism math — the difference is the
  programming model and the hardware affinity.

## 9c. Running training as Kubernetes jobs

How the launch maps onto a cluster (orchestration depth lives in the sibling skills):

- **Kubeflow Trainer (v2)** — the current API: a `TrainJob` references a reusable `TrainingRuntime`/
  `ClusterTrainingRuntime`; it builds on **JobSet** under the hood for the multi-node gang. The v1
  **Training Operator** (`PyTorchJob`, `TFJob`, etc.) is the predecessor — a `PyTorchJob` runs a
  master + workers and wires up the rendezvous env for `torchrun`. See `[[aiml-on-kubernetes]]`.
- **MPI Operator** — `MPIJob` for MPI-launcher-style runs (`mpirun`/`hostfile`), common for
  Horovod/NCCL-allreduce and Megatron-via-MPI; integrates with RDMA. See `[[slurm-hpc-on-kubernetes]]`.
- **JobSet / LeaderWorkerSet** — the multi-host primitive most modern training (and Kubeflow Trainer
  v2) sits on: a leader + worker pods forming one gang with stable network identity. The natural target
  for a `torchrun` rendezvous across nodes. See `[[jobset-leaderworkerset]]`.
- **Kueue** — gang-schedules and quota-manages these jobs (all-or-nothing admission so a multi-node
  training job doesn't half-start and deadlock); **Topology-Aware Scheduling** packs a job into one
  fabric domain so TP/PP land on fast links. See `[[kueue-advanced]]`.
- **Mapping rule:** one process per accelerator; one pod per node holding `LOCAL_WORLD_SIZE` procs;
  `torchrun --nnodes=N --nproc-per-node=G` with the rendezvous endpoint = the leader pod's stable DNS.
  Put TP/EP within a pod/node, PP/DP across pods. Gang-schedule the whole set. See `examples.md`.

---

## 10. Decision guide — which parallelism, which framework

**Step 1 — does it fit?** Estimate static state (≈16–18 B/param for Adam mixed precision) + activations
(scale with batch×seq×hidden×layers, cut by checkpointing). If it fits on one GPU → **DDP**. If not →
shard.

**Step 2 — pick the parallelism shape (rule of thumb):**

| Situation | Start with |
|---|---|
| Fits on 1 GPU | DDP |
| Doesn't fit, dense, ≤ ~30–70B | **FSDP2 / ZeRO-3** (HYBRID/2-D mesh multi-node), activation checkpointing |
| Large dense (70B–500B+) | **TP (≤ node) + PP (across nodes) + DP/FSDP outer + SP**, i.e. 3D — Megatron-Core or TorchTitan |
| Very long context | add **CP / sequence parallel** |
| MoE / sparse | **EP** (+ TP within expert, DP/FSDP for dense params) — Megatron-Core MoE or DeepSpeed-MoE |
| Memory-bound, limited HBM, can trade speed | **ZeRO-Infinity** (NVMe/CPU offload) |
| Fine-tune / PEFT (LoRA) | FSDP or DeepSpeed via **HF Accelerate/Trainer/TRL** — usually 1 node |

**Step 3 — map to topology:** TP/EP inside NVLink/one node; PP/DP across nodes; gang-schedule into one
fabric domain (`[[kueue-advanced]]` TAS). `tp × pp × cp` should divide neatly into your per-node and
per-island GPU counts.

**Step 4 — pick the framework/ecosystem:**

| You are… | Use |
|---|---|
| Training from scratch in PyTorch, want control + `torch.compile` | **TorchTitan / FSDP2 + native TP/PP** |
| Frontier dense/MoE pretraining on NVIDIA, max MFU | **Megatron-Core** (or **NeMo** for recipes) |
| Need ZeRO-3 + heavy offload, minimal code change | **DeepSpeed** |
| Fine-tuning / post-training HF models | **HF Accelerate / Trainer / TRL** |
| On TPU, or want SPMD | **JAX: MaxText / Levanter / Paxml+Pathways** |
| Orchestrating in a Ray cluster | **Ray Train** wrapping the above |
| Running on Kubernetes | **Kubeflow Trainer v2 / JobSet + Kueue** around your launcher |

Bias: **don't add a parallelism axis you don't need.** Each one adds comms, complexity, and a way to
mis-shape the mesh. Start with the fewest axes that make it fit, then add only to relieve the proven
bottleneck (memory → shard; single-layer too big → TP; too deep → PP; context too long → CP; sparse →
EP). Measure MFU after every change.

---

## 11. Anti-patterns / gotchas

- **TP across the slow fabric.** Putting tensor (or expert) parallel across nodes stalls on the
  per-layer all-reduce/all-to-all. Keep TP/EP ≤ one NVLink domain.
- **`m ≈ p` pipelines.** Too few micro-batches → giant bubble. Need `m ≫ p` (and consider interleaved/
  zero-bubble schedules).
- **Lost comm/compute overlap.** Wrong FSDP prefetch settings, syncing every micro-batch under grad
  accumulation, or CPU-bound data loading silently halves MFU. Profile the timeline.
- **Wrong effective batch size.** Forgetting that global batch = micro-batch × grad-accum × DP degree;
  scaling DP changes your LR/schedule and convergence.
- **Gather-to-rank-0 checkpoints** at scale — OOMs the coordinator and serializes I/O. Use sharded/DCP.
- **Mismatched precision state.** Sharding the bf16 weights but not keeping the fp32 master copy →
  silent divergence. fp8 without proper scaling/amax → NaNs.
- **Non-resumable data loader.** Resuming from a checkpoint but re-seeding the loader from scratch →
  re-seeing or skipping data; subtle quality regressions. Make the loader checkpointable.
- **Non-deterministic mesh ↔ topology placement.** Letting ranks land arbitrarily so the TP group spans
  nodes. Pin placement (TAS / `LOCAL_RANK` ordering / `NCCL` topology).
- **Mixing FSDP1 and FSDP2 idioms**, or wrapping at the wrong granularity (whole model as one unit → no
  memory savings; every tiny module → comm overhead). Wrap per transformer block.

## 12. Troubleshooting (symptom → likely cause → fix)

| Symptom | Likely cause | Fix |
|---|---|---|
| Low MFU, GPUs idle in profiler gaps | comms not overlapped / pipeline bubble / slow data loader | check FSDP prefetch, raise micro-batches, async/prefetch data, profile |
| OOM only at certain layers | activation peak / one FSDP unit too big | activation (selective) checkpointing, finer FSDP wrap, add TP/CP |
| OOM during checkpoint save | gather-to-rank-0 | switch to DCP/sharded async checkpoint |
| Loss diverges/NaN | fp8/fp16 scaling, missing fp32 master, bad LR after DP scaling | bf16, keep master weights, re-tune LR for global batch |
| Hang at step start | collective mismatch / one rank diverged in control flow / NCCL timeout | match collectives across ranks, raise `NCCL`/`TORCH_NCCL` timeout, check logs per rank |
| Throughput drops at multi-node | all-gather crossing fabric | HYBRID/2-D FSDP, ZeRO++, topology-aware placement |
| MoE device hotspot / imbalance | expert load skew | capacity factor / aux loss / dropless kernels, EP placement |
| Job dies on any node loss | no elasticity | `torchrun` elastic + sharded checkpoint + checkpointable loader / torchft / Pathways |

---

## Rationalizations & rebuttals

- *"DDP is enough for a 70B."* No — Adam mixed-precision static state is ~16–18 B/param ≈ 1.1–1.3 TB,
  and DDP replicates all of it per device. It doesn't fit on one 80GB GPU. Shard with FSDP2/ZeRO-3
  (add TP/PP past the tens of billions).
- *"Skip activation checkpointing, we have the memory."* Activations scale with
  `batch×seq×hidden×layers` and dominate at long context — they're what OOMs you mid-run. Use
  selective/granular checkpointing: most of the memory back for far less than the naive ~30% recompute.
- *"Gather-to-rank-0 checkpoint sync is fine."* It OOMs the coordinator and serializes I/O at scale,
  and stalls every rank while one writes. Use sharded Distributed Checkpoint (DCP), async, resharding-
  tolerant.
- *"Don't bother measuring MFU."* MFU is the one number that tells you whether your parallelism shape
  and overlap are working; without it you tune blind and ship 30%-MFU runs. Measure it after every
  shape change — and only quote numbers you've actually seen.
- *"Comms aren't the bottleneck."* At multi-node scale they usually are — lost comm/compute overlap is
  the most common cause of low MFU, and TP/EP on the slow fabric stalls on per-layer
  all-reduce/all-to-all. Profile the timeline before assuming compute-bound.
- *"Add TP everywhere, more parallelism is better."* Every axis adds comms, complexity, and a way to
  mis-shape the mesh. Use the fewest axes that make it fit, then add only to relieve a *proven*
  bottleneck.
- *"We'll add fault tolerance later."* At thousands of accelerators something fails constantly; without
  elastic restart + sharded async checkpoints + a checkpointable loader, one node loss kills the job
  and goodput collapses. Build it in from the start.

## Red flags

- **Low MFU accepted without investigation** — GPUs idle in profiler gaps and nobody checks overlap,
  bubble, or data-loader stalls.
- **OOM "fixed" by blindly lowering batch size** — without checking the activation peak, FSDP wrap
  granularity, or adding selective checkpointing / TP / CP. Often silently wrecks effective global
  batch (→ LR/convergence) too.
- **No async / no sharded checkpoint** — still gathering to rank 0, or a synchronous save that stalls
  every step.
- **No elastic restart** — the job dies on any single node loss; recovery is a full manual job restart.
- **Comms not overlapped** — wrong FSDP prefetch settings, syncing every micro-batch under grad
  accumulation, or TP/EP placed across the slow fabric.
- **MFU never measured** (or numbers quoted that nobody has profiled) — no baseline to tune against.
- **Mesh ↔ topology placement left to chance** — TP/EP groups span nodes because ranks landed
  arbitrarily; no TAS / `LOCAL_RANK` ordering / NCCL topology pinning.
- **Data loader not checkpointable** — resume re-seeds from scratch, re-seeing or skipping data, with
  subtle quality regressions.

## Verification gate (definition of done)

- [ ] **Parallelism shape justified** for *this* model and cluster: it actually fits (static state +
      activation estimate), uses the fewest axes that work, and TP/EP land inside the NVLink/one-node
      high-bandwidth domain while PP/DP cross nodes.
- [ ] **`tp × pp × cp × ep`** divides cleanly into per-node and per-island accelerator counts;
      mesh→topology placement is pinned (TAS / `LOCAL_RANK` / NCCL topology), not arbitrary.
- [ ] **MFU measured** (achieved ÷ peak FLOP/s) and re-checked after the final shape change; profiler
      timeline confirms comms overlapped with compute and no oversized pipeline bubble.
- [ ] **Effective global batch** = micro-batch × grad-accum × DP degree is the intended value, and the
      LR/schedule was (re)tuned for it.
- [ ] **Sharded + async checkpointing** in place (DCP or framework equivalent) — no gather-to-rank-0;
      save cadence set so a crash loses minutes, not hours; the data loader is checkpointable/resumable.
- [ ] **Fault tolerance tested** — kill a node and confirm elastic restart (torchrun elastic / torchft
      / Pathways) re-forms the group, reloads the last shard, and fast-forwards the loader to the right
      step.
- [ ] **Scaling efficiency verified** — throughput scales near-expected when you add nodes/devices (no
      cliff from all-gather crossing the fabric); goodput acceptable under the observed failure rate.

---

## 13. Canonical references (verify against current versions)

- PyTorch FSDP / FSDP2: pytorch.org/docs (`torch.distributed.fsdp`, `fully_shard`), FSDP paper
  (arXiv:2304.11277).
- TorchTitan: github.com/pytorch/torchtitan (and its paper).
- `torch.distributed` / DeviceMesh / DTensor / elastic / pipelining / DCP: pytorch.org/docs.
- DeepSpeed: deepspeed.ai, github.com/deepspeedai/DeepSpeed; ZeRO (arXiv:1910.02054), ZeRO-Offload
  (2101.06840), ZeRO-Infinity (2104.07857), ZeRO++ (2306.10209).
- Megatron-LM / Megatron-Core: github.com/NVIDIA/Megatron-LM, docs.nvidia.com; Megatron papers
  (1909.08053 TP, 2104.04473 PP/3D, 2205.05198 sequence-parallel + selective recompute).
- NVIDIA NeMo: github.com/NVIDIA/NeMo, docs.nvidia.com/nemo.
- HF Accelerate / Trainer / TRL: huggingface.co/docs.
- PyTorch Lightning: lightning.ai/docs.
- Ray Train: docs.ray.io/en/latest/train.
- JAX SPMD / sharding / Shardy: jax.readthedocs.io; MaxText github.com/AI-Hypercomputer/maxtext;
  Levanter github.com/stanford-crfm/levanter; Paxml github.com/google/paxml; Pathways
  (arXiv:2203.12533).
- Kubeflow Trainer: kubeflow.org/docs/components/trainer; MPI Operator github.com/kubeflow/mpi-operator.
- Background: Megatron-style 3D-parallelism survey & GPipe (1811.06965), PipeDream/1F1B (1806.03377),
  Ring Attention (2310.01889).

---

# Distributed Training — Worked Examples

Correct-*in-shape* artifacts to imitate. Exact flag/config-key names drift between releases — verify
against the version you pin (see `training-frameworks-guide.md` §13). The structure, not any specific
benchmark number, is the thing to copy.

---

## 1. FSDP2 PyTorch training launch (single-node, 8 GPUs)

### 1a. `torchrun` invocation

```bash
# One process per GPU on one node. torchrun sets RANK / LOCAL_RANK / WORLD_SIZE for you.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  train.py --model-config configs/llama_8b.yaml
```

Multi-node (2 nodes × 8 GPUs = 16 ranks), elastic so the job survives a node loss:

```bash
# Run on every node; rdzv endpoint = a stable address all nodes can reach (e.g. the leader pod's DNS).
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --max-restarts=3 \
  --rdzv-backend=c10d \
  --rdzv-id=llama8b-run \
  --rdzv-endpoint="$LEADER_ADDR:29500" \
  train.py --model-config configs/llama_8b.yaml
# Elastic: --nnodes=min:max (e.g. 1:2) lets the job continue/rescale instead of dying on node loss.
```

### 1b. FSDP2 sharding setup in `train.py` (shape-correct skeleton)

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 2-D mesh = HYBRID shard: shard within a node, replicate across nodes -> param all-gather stays on
# NVLink, only a smaller reduce crosses the slow fabric. For a pure 1-D full-shard use one axis.
mesh = init_device_mesh("cuda", (NUM_NODES, GPUS_PER_NODE), mesh_dim_names=("replicate", "shard"))

model = build_model(cfg).to("cuda")

# Keep compute in bf16; keep an fp32 master copy + fp32 reduction for stability.
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

# Wrap PER TRANSFORMER BLOCK first (each becomes an all-gather/reduce-scatter unit), then the root.
# Whole-model-as-one-unit = no memory savings; every tiny module = comm overhead.
for block in model.transformer_blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)

# Selective activation checkpointing on the blocks (recompute cheap ops, keep flash-attn output).
# from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

for step, batch in enumerate(loader):           # loader must be checkpointable/deterministic
    for i, micro in enumerate(split_microbatches(batch, cfg.grad_accum)):
        is_last = (i == cfg.grad_accum - 1)
        model.set_requires_gradient_sync(is_last)   # skip grad reduce-scatter until the last micro-batch
        loss = model(micro).loss / cfg.grad_accum
        loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)

    if step % cfg.ckpt_every == 0:               # SHARDED checkpoint via DCP — never gather to rank 0
        sd_model, sd_opt = get_state_dict(model, opt)
        dcp.save({"model": sd_model, "optim": sd_opt, "step": step}, checkpoint_id=ckpt_path(step))
        # dcp.async_save(...) to overlap the flush with the next steps.
```

Notes: `fully_shard`/`MixedPrecisionPolicy`/`set_requires_gradient_sync` are FSDP2 API — confirm exact
signatures against your PyTorch version. Effective global batch = micro-batch × `grad_accum` ×
DP-degree; re-tune LR when you change DP. To add **tensor parallel**, build a `(dp, tp)` mesh and
`parallelize_module(block, mesh["tp"], plan)` *before* `fully_shard` on the `dp` axis.

---

## 2. Megatron-Core / Megatron-LM 3D-parallel launch (sketch)

Megatron is launched via `torchrun` with parallelism degrees passed as flags. Shape only — **verify
every flag against your Megatron-LM/Megatron-Core release**, they change.

```bash
# Example: 16 nodes × 8 = 128 GPUs.  TP=8 (inside one node, on NVLink) · PP=4 (across nodes) ·
# DP = 128 / (TP*PP) = 4.  SP on.  Optional CP for long context, EP for MoE.
torchrun --nnodes=16 --nproc-per-node=8 \
         --rdzv-backend=c10d --rdzv-endpoint="$LEADER_ADDR:29500" \
  pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --sequence-parallel \
  --num-layers-per-virtual-pipeline-stage 2 \   # interleaved 1F1B -> smaller bubble
  --context-parallel-size 1 \                   # raise for very long context
  --use-distributed-optimizer \                 # ZeRO-1 optimizer-state sharding over the DP group
  --recompute-granularity selective \           # selective activation recomputation
  --bf16 \
  --micro-batch-size 1 \
  --global-batch-size 512 \                     # = micro-batch × grad-accum × DP, set by Megatron
  --num-layers 80 --hidden-size 8192 --num-attention-heads 64 --seq-length 8192 \
  --lr 1.5e-4 --min-lr 1.5e-5 --lr-warmup-iters 2000 --clip-grad 1.0 \
  --save /ckpts/run1 --load /ckpts/run1 --ckpt-format torch_dist   # sharded/distributed checkpoint
# MoE adds (verify): --num-experts N --expert-model-parallel-size E --moe-router-topk K
# fp8 via Transformer Engine adds (verify): --fp8-format hybrid (with the right recipe flags)
```

Topology mapping that matters: **TP=8 lands within a node** (NVLink); **PP=4 crosses nodes** (P2P,
bubble-tolerant); **DP=4 is outermost** (one all-reduce/step). `world_size = TP·PP·DP·CP·EP`.

---

## 3. DeepSpeed ZeRO-3 config sketch (`ds_config.json`)

Launched with `deepspeed train.py --deepspeed --deepspeed_config ds_config.json` (or via HF Trainer /
Accelerate which generate/consume this). Shape only — verify keys against your DeepSpeed version.

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param":     { "device": "none" }
  },
  "gradient_clipping": 1.0,
  "zero_allow_untested_optimizer": false
}
```

- `stage: 3` = shard params + grads + optimizer (FSDP-equivalent). `stage: 2` = grads + optimizer;
  `stage: 1` = optimizer only.
- `offload_optimizer.device: "cpu"` = **ZeRO-Offload**; set `offload_param.device: "nvme"` (+ an
  `nvme_path`) for **ZeRO-Infinity** when you're memory-bound and can trade throughput.
- For comm reduction on bandwidth-constrained clusters, ZeRO++ adds quantized-weight / hierarchical-
  partition / quantized-gradient options — verify the exact keys for your version.

---

## 4. `torchrun` → JobSet / Kueue mapping (K8s)

How the launcher above lands on a cluster. Orchestration depth is in `[[jobset-leaderworkerset]]`,
`[[kueue-advanced]]`, `[[aiml-on-kubernetes]]`; this is just the mapping.

```
Process model                         Kubernetes object
------------------------------------  ----------------------------------------------------
1 process per accelerator             1 container process; nproc-per-node = GPUs per pod
1 node = 1 pod (LOCAL_WORLD_SIZE)     1 Pod per node, requesting all node GPUs + RDMA NIC
N nodes = 1 gang                       1 JobSet (or Kubeflow Trainer v2 TrainJob, built on JobSet)
rendezvous endpoint                    leader Pod's stable headless-Service DNS : 29500
gang scheduling (all-or-nothing)       Kueue admission (whole gang or none — no half-start deadlock)
TP/EP on fast links                    Kueue Topology-Aware Scheduling -> one fabric/NVLink domain
survive node loss                      JobSet restart policy + torchrun --max-restarts + DCP + Kueue
```

Each pod runs the same `torchrun --nnodes=$N --nproc-per-node=$G --rdzv-endpoint=$LEADER:29500 ...`;
`$N`, `$LEADER`, and rank/replica identity come from the JobSet/LeaderWorkerSet env. Put TP/EP within
a pod (one node), PP/DP across pods. Kubeflow Trainer v2's `TrainJob`/`TrainingRuntime` builds this
JobSet for you; the v1 `PyTorchJob` (Training Operator) wires the master+worker rendezvous env directly.
Always gang-schedule and place into one topology domain — a multi-node job that half-starts deadlocks,
and a TP group split across nodes stalls on comms.
