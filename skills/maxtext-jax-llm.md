---
name: maxtext-jax-llm
description: Expert guidance for MaxText and the JAX LLM stack — training and serving frontier-scale LLMs
  (Llama, Gemma, Mistral, DeepSeek, Qwen, Mixtral/MoE) on TPU pods and GPU. Use when working with MaxText
  configs (base.yml/model YAML, ici_*_parallelism / dcn_*_parallelism, per_device_batch_size, remat_policy,
  attention=flash, weight_dtype/quantization=int8), JAX sharding (Mesh, PartitionSpec, NamedSharding,
  logical_axis_rules, GSPMD), Flax (linen/NNX), Optax, Grain, Orbax, Pathways, or JetStream inference;
  scaling to multi-host / multislice TPU (ICI vs DCN); chasing MFU, OOM, or recompilation; or launching on
  GKE with XPK/JobSet. Covers the FSDP/TP/sequence/expert sharding mental model and the TPU-JAX-vs-GPU-PyTorch
  decision.
---

# MaxText & the JAX LLM Stack

Apply the judgment of an engineer who has trained frontier-scale LLMs on multi-thousand-chip TPU pods and
served them in production — someone who reads HLO to chase a 3% MFU regression and knows why a stray
`reshape` triggered a recompile. The bar: high MFU, no surprise OOM, correct sharding by construction, and
checkpoints you can actually restore. The ecosystem moves fast (it is 2026) — **verify fast-moving flags,
config keys, and version-specific behavior against current MaxText/JAX docs** before relying on them.

## How to use this skill

1. **Read `maxtext-jax-llm-guide.md`** in this directory — the full reference (mental model, sharding,
   configs, scaling, quantization, inference, troubleshooting). Apply it to the task.
2. For concrete artifacts to imitate — a multi-host TPU training launch (config + JobSet/XPK), a JetStream
   serving sketch, and a raw JAX `Mesh`/`PartitionSpec` snippet — read **`examples.md`**.
3. Match the surrounding repo/cluster conventions (config layout, mesh axis names, launcher). Apply the
   correctness rules — sharding constraints, dtype/quant choices, checkpoint compatibility — regardless.

## Essentials (full detail in `maxtext-jax-llm-guide.md`)

- **MaxText is config-driven.** Behavior is one merged config (`base.yml` + a model YAML + CLI
  `key=value` overrides). Parallelism, remat, dtype, attention kernel, dataset, and checkpoint paths are
  all config keys — change YAML/CLI, not Python, for 95% of work.
- **Sharding = a `Mesh` of named axes + logical-axis rules.** You declare *logical* axes on arrays
  (`embed`, `mlp`, `heads`, ...); `logical_axis_rules` map them to *physical* mesh axes
  (`data`, `fsdp`, `tensor`, `sequence`, `expert`). GSPMD/XLA infers the rest. Get the mapping right and
  the parallelism strategy is correct by construction.
- **ICI vs DCN is the core scaling distinction.** `ici_*_parallelism` shards *within* a slice over the
  fast inter-chip interconnect; `dcn_*_parallelism` shards *across* slices (multislice) over slower data-
  center network. Product of all ICI×DCN factors must equal total chips. Keep TP and heavy collectives on
  ICI; put DP/FSDP across DCN.
- **MFU is the scoreboard.** Track `per_device_tflops/s` (logged) vs the chip's peak. Below ~50–60% on
  a known-good config, suspect remat policy, small per-device batch, host input bottleneck, or a recompile.
- **Rematerialization trades compute for memory.** `remat_policy` (`full`, `save_dot_except_mlp`,
  `qkv_proj_offloaded`, `minimal`, `none`, ...) controls what activations are kept. Use the lightest remat
  that fits HBM; `full` only when memory-bound.
- **Quantization via AQT.** `quantization=int8` (and FP8 on supported hardware) for training/inference
  speedups; verify supported modes and accuracy impact against current docs before shipping.
- **Recompilation and OOM are the two recurring fires.** Fixed shapes + stable mesh = compile once.
  Variable sequence length, dynamic batch, or changing the mesh recompiles. Read the compiled-memory
  breakdown; reduce per-device batch or add remat for OOM.
- **JetStream serves MaxText on TPU** with continuous batching + paged attention. Reach for it for
  TPU-native serving of MaxText/JAX models; use vLLM/SGLang for the GPU-PyTorch path ([[serving-frameworks]]).
- **Checkpoint with Orbax.** Async, sharded checkpoints; conversion to/from HF format is a distinct step.
  Depth in [[ml-checkpointing-orbax]].
- **Goodput matters at scale.** Measure effective vs wall-clock training time; the failure rate of a
  10k-chip job makes fast checkpoint/restore and elastic restart essential, not optional.
- **Decision:** TPU + JAX + MaxText when you want top MFU on TPU pods, GSPMD sharding, and a clean config-
  driven reference. GPU + PyTorch (FSDP/Megatron, [[training-frameworks]]) when the ecosystem, custom CUDA
  kernels, or existing PyTorch code dominate.

## Related skills

- `[[ml-frameworks]]` — JAX/XLA/PyTorch and TPU/GPU compute-framework internals (the layer below this).
- `[[training-frameworks]]` — DDP/FSDP, DeepSpeed, Megatron, NeMo; MaxText's PyTorch-world counterparts.
- `[[serving-frameworks]]` — vLLM, SGLang, TensorRT-LLM; choose JetStream vs these for inference.
- `[[ml-checkpointing-orbax]]` — Orbax checkpoint depth (async, sharded, format conversion).
- `[[jobset-leaderworkerset]]` — multi-host gang scheduling for TPU training/inference on K8s.
- `[[gke-master]]` · `[[aiml-on-kubernetes]]` — TPU node pools, topology, and running these jobs on GKE.

---

# Reference — maxtext-jax-llm

# MaxText & the JAX LLM Stack — Deep Reference

Training and serving frontier-scale LLMs on TPU pods (and GPU) with MaxText and the surrounding JAX stack.
This guide assumes you know JAX/XLA basics (`jit`, `vmap`, tracing, `pjit`/`shard_map`, the SPMD model) —
that compute-framework layer is [[ml-frameworks]]. Here we focus on the **LLM stack**: how MaxText is
structured, how sharding is expressed, how to scale across a pod and across slices, and how to keep MFU high.

> **Version note (2026):** MaxText, JAX, Flax (linen→NNX), Grain, Orbax, JetStream, and Pathways all move
> quickly. Config-key names, `remat_policy` options, quantization modes, supported models, and API surfaces
> change between releases. Treat every concrete flag/key below as *illustrative of the idiom* and **verify
> against the version you are running**. Never assume a benchmark number — measure on your hardware.

---

## 1. Mental model

**MaxText is an open, high-performance reference implementation of LLM training and decoding in pure
JAX/Flax.** It is deliberately small and hackable: a handful of model definitions (Llama, Gemma, Mistral,
Mixtral and other MoE, DeepSeek, Qwen, GPT-style, etc.), a single training loop, and a large config surface.
It is *not* a framework with a plugin system — it is reference code you fork and adapt. Its reason to exist is
**high Model FLOPs Utilization (MFU) at scale via GSPMD sharding**, with everything expressed declaratively so
you change *config*, not *code*, to go from one chip to a multislice pod.

The layers, bottom to top:

| Layer | Component | Role |
|---|---|---|
| Compute | **JAX + XLA** | Tracing, SPMD partitioning (GSPMD), TPU/GPU codegen ([[ml-frameworks]]) |
| Modeling | **Flax** (linen, newer **NNX**) | Module/parameter definitions, logical sharding annotations |
| Optimization | **Optax** | Optimizers (AdamW, Adafactor), LR schedules, gradient clipping |
| Data | **Grain** (also tf.data / HF) | Deterministic, checkpointable input pipeline |
| Checkpoint | **Orbax** | Async, sharded checkpoints; format conversion ([[ml-checkpointing-orbax]]) |
| Reference impl | **MaxText** | The training/decoding loop + model defs + config glue |
| Orchestration | **Pathways** / single-controller | Multi-controller runtime for very large TPU jobs |
| Inference | **JetStream** | TPU continuous-batching serving engine for MaxText/JAX models |

Sibling JAX LLM efforts you should know exist: **Paxml/Praxis** (older, more configurable, heavier),
**Levanter** (Haliax named-tensor training), and **MaxDiffusion** (the diffusion analogue of MaxText). When
someone wants "the simple, fast TPU LLM baseline," MaxText is the default answer.

---

## 2. The config-driven workflow

The entire run is one **merged config**. Precedence (later wins):

1. `configs/base.yml` — the master list of every key with defaults.
2. A model config (e.g. `configs/models/*.yml` or a `model_name=...` preset) — architecture + recommended
   parallelism/remat.
3. CLI overrides — `key=value` pairs appended to the launch command.

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
    model_name=llama3.1-8b \
    run_name=$RUN \
    base_output_directory=gs://my-bucket/maxtext \
    dataset_path=gs://my-bucket/data \
    per_device_batch_size=4 \
    ici_fsdp_parallelism=8 \
    steps=50000 \
    remat_policy=save_dot_except_mlp \
    attention=flash \
    weight_dtype=bfloat16
```

Key categories you will touch constantly (names are illustrative — **verify**):

- **Run/IO:** `run_name`, `base_output_directory` (GCS), `dataset_path`, `dataset_type`
  (`grain`/`tfds`/`hf`), `tokenizer_path`, `load_parameters_path` / `load_full_state_path`.
- **Model:** `model_name` (or explicit `base_emb_dim`, `base_num_query_heads`, `base_num_kv_heads`,
  `base_mlp_dim`, `base_num_decoder_layers`, `head_dim`, `vocab_size`, `mlp_activations`).
- **Parallelism:** `ici_*_parallelism` and `dcn_*_parallelism` for `data` / `fsdp` / `tensor` /
  `sequence` / `expert` (see §4).
- **Memory/perf:** `remat_policy`, `per_device_batch_size`, `attention` (`flash`/`dot_product`/...),
  `quantization`, `weight_dtype`, `dtype`, `gradient_accumulation_steps`.
- **Optim/schedule:** `opt_type` (adamw/adafactor), `learning_rate`, `warmup_steps_fraction` (or warmup
  steps), `cosine_learning_rate_final_fraction`, `adam_b1/b2/eps`, `gradient_clipping_threshold`.
- **Checkpoint:** `enable_checkpointing`, `checkpoint_period`, `async_checkpointing`.

Because everything is config, the **same code** runs on 1 chip, a v5e-256 slice, or 4 slices — you change
parallelism factors, not Python. This is the whole point.

---

## 3. The sharding mental model (named axes / GSPMD)

This is the part people get wrong. JAX uses **GSPMD**: you annotate *how arrays are partitioned*, and XLA
inserts the collectives. MaxText layers a **logical-axis** abstraction on top so you reason in model terms.

### 3.1 Physical mesh

A `jax.sharding.Mesh` is an n-dimensional grid of devices with **named physical axes**. MaxText names them
roughly `data`, `fsdp`, `tensor`, `sequence`, `expert` (and may split ICI vs DCN variants). Example:

```python
import jax, numpy as np
from jax.sharding import Mesh

devices = np.array(jax.devices()).reshape(data, fsdp, tensor)  # product == device count
mesh = Mesh(devices, axis_names=("data", "fsdp", "tensor"))
```

### 3.2 Logical axes → physical axes

Model arrays are tagged with **logical** axis names (`embed`, `mlp`, `heads`, `kv`, `vocab`,
`batch`, `length`, ...). A `logical_axis_rules` table maps each logical name to zero or more **physical**
mesh axes:

```python
logical_axis_rules = (
    ("batch", ("data", "fsdp")),   # batch dim sharded over data+fsdp
    ("embed", "fsdp"),             # model/hidden dim sharded for FSDP
    ("mlp",   "tensor"),           # MLP hidden sharded for tensor parallel
    ("heads", "tensor"),           # attention heads sharded for TP
    ("length","sequence"),         # sequence/context parallel
    ("vocab", "tensor"),
)
```

Map a logical axis to `None` (or omit it) to **replicate** that dimension. This indirection is why the
*same model code* expresses DP, FSDP, TP, sequence, and expert parallelism just by editing the rules +
the `ici_*/dcn_*` factors. Under the hood MaxText converts logical specs to `PartitionSpec`/`NamedSharding`
and feeds them to `jit`'s `in_shardings`/`out_shardings`. Use `jax.lax.with_sharding_constraint`
(MaxText wraps this) to pin intermediate activations and stop XLA from choosing a bad layout.

### 3.3 The parallelism strategies, in these terms

| Strategy | What is sharded | Logical→physical idea |
|---|---|---|
| **Data parallel (DP)** | batch only; params replicated | `batch → data` |
| **FSDP / ZeRO-3** | params + optimizer state + grads sharded; gathered per-layer | `embed/mlp → fsdp`, `batch → fsdp` |
| **Tensor parallel (TP)** | weight matrices within a layer | `mlp/heads/vocab → tensor` |
| **Sequence/context parallel** | the activation sequence dim | `length → sequence` |
| **Expert parallel (MoE)** | experts spread across devices | `expert → expert` |
| **Pipeline** | layers across stages (MaxText support is more limited; verify) | layer-stage assignment |

FSDP is the workhorse for dense models: low comm volume, scales well over ICI and even DCN. Add **TP** when
a single layer's weights or activations don't fit, or to cut latency. **Sequence parallel** for very long
context. **Expert parallel** for MoE (Mixtral/DeepSeek-style). You almost always *combine* them
(e.g. FSDP × TP) — the mesh just has more axes.

---

## 4. Scaling: single host → pod → multislice

### 4.1 ICI vs DCN — the central distinction

- **ICI (Inter-Chip Interconnect):** the fast, high-bandwidth, low-latency mesh *within a single TPU slice*
  (e.g. a v5e/v5p/v6e slice). All `ici_*_parallelism` factors shard over ICI.
- **DCN (Data-Center Network):** the (much slower, higher-latency) network *between slices*. All
  `dcn_*_parallelism` factors shard across slices — this is **multislice**.

**Rule:** `∏ ici_*_parallelism × ∏ dcn_*_parallelism == total_chips`. MaxText validates this; a mismatch is
a config error, not a silent fallback.

**Placement strategy:**
- Keep **bandwidth-heavy collectives on ICI**: tensor parallel (per-layer all-reduce/all-gather) and
  FSDP weight gathers belong inside a slice.
- Put **DP / FSDP-with-tolerable-comm across DCN**: gradient all-reduce across slices tolerates the slower
  link because it overlaps with compute and happens once per step.
- Common multislice recipe: **FSDP within slice (ICI), data parallel across slices (DCN)** —
  `ici_fsdp_parallelism = chips_per_slice`, `dcn_data_parallelism = num_slices`.

### 4.2 Topology and multi-host mechanics

A multi-host TPU slice is many VM hosts each attached to some chips; **every host runs the same program**
(SPMD) and JAX forms a global device mesh across all of them via the TPU runtime. You do **not** write
per-host logic. On GKE this means a multi-host TPU node pool with a specific `topology` (e.g. `4x4x4`), and
the pods must be **gang-scheduled** so all hosts of a slice start together (see §10, [[jobset-leaderworkerset]]).

### 4.3 Multislice

Multislice connects N slices into one logical job over DCN. The `dcn_*_parallelism` axes describe how work
is split across slices. Pathways (§9) or the multislice runner sets up cross-slice connectivity. Watch for:
DCN collectives stalling on stragglers; uneven slice topologies; and host-side input pipelines becoming the
bottleneck once compute is spread thin.

---

## 5. MFU, performance, and the input pipeline

**MFU (Model FLOPs Utilization)** = achieved model FLOPs/s ÷ hardware peak FLOPs/s. MaxText logs a
per-device TFLOP/s; divide by the chip's bf16 peak to get MFU. Well-tuned dense training on modern TPU
typically lands in a healthy band (don't quote a fixed number — **measure**). When MFU is low, work this
checklist in order:

1. **Recompilation** — if step time spikes periodically or the first steps are slow, you're recompiling
   (§8). Compile must happen *once*.
2. **Per-device batch too small** — TPUs need work to hide latency. Raise `per_device_batch_size` or use
   `gradient_accumulation_steps`; watch HBM.
3. **Remat too aggressive** — `full` remat recomputes everything; if you have HBM headroom, use a lighter
   policy (§6) and reclaim compute.
4. **Input bottleneck** — host can't feed the chips. Check Grain worker count, prefetch, and that data is
   sharded per host. Profile: if XLA is idle waiting on infeed, fix the pipeline, not the model.
5. **Suboptimal sharding** — an `all-to-all` or oversized `all-gather` from bad `logical_axis_rules`;
   inspect the HLO / profile for unexpected collectives.
6. **Attention kernel** — use `attention=flash` (fused/flash-style kernel) rather than the naive
   `dot_product` path for long sequences.

Always run the **JAX/XLA profiler** (TensorBoard profile / `jax.profiler`) and read the trace: the gaps
between fusions tell you whether you're compute-bound, comm-bound, or input-bound.

---

## 6. Rematerialization (activation checkpointing)

Remat trades **compute for HBM**: instead of storing all activations for the backward pass, recompute some.
`remat_policy` selects what is *saved* vs *recomputed*. Typical options (names **vary by version — verify**):

| Policy | Behavior | Use when |
|---|---|---|
| `none` / `minimal` | save (almost) nothing recomputed / keep little | you have HBM headroom, want max compute reuse |
| `save_dot_except_mlp` | keep matmul outputs except MLP | common balanced default |
| `save_qkv_proj` / `qkv_proj_offloaded` | keep/offload attention projections | attention-heavy memory pressure |
| `full` | recompute everything possible | strongly memory-bound (large model/seq/batch) |

**Use the lightest remat that still fits HBM.** `full` everywhere is a common reflex that silently caps MFU
— reach for it only when the compiled-memory report says you're out of room. Offloading variants move
activations to host memory; they relieve HBM but add transfer time — measure the net.

---

## 7. Quantization (AQT) and numerics

- **Training/inference quantization uses AQT** (Accurate Quantized Training). `quantization=int8` is the
  common int8 path; FP8 is available on hardware that supports it. **Verify the exact modes and any
  accuracy caveats against current docs** before training a real run.
- **Default compute dtype is bf16.** `weight_dtype`/`dtype` control storage/compute precision. Keep
  optimizer state and reductions in higher precision where it matters (e.g. fp32 master/accumulators) to
  avoid divergence — Adafactor and AdamW configs expose this.
- For **inference**, int8 (and FP8 where supported) cut HBM and boost throughput; validate quality on your
  eval set, not just perplexity, before serving.

---

## 8. Recompilation & OOM — the two recurring fires

### Recompilation
XLA compiles per **(shape, dtype, sharding, mesh) signature**. Anything that changes the signature triggers
a fresh, expensive compile. Causes and fixes:

- **Variable sequence length / dynamic batch** → pad/bucket to fixed shapes. MaxText pads to `max_target_length`.
- **Changing the mesh or device count mid-run** → keep the mesh constant.
- **Python-level branching on traced values** → restructure so shapes are static.
- Symptom: step time jumps, or logs show repeated "Compiling ..."; the *first* step is always slow (that's
  the expected one compile).

### OOM (HBM exhaustion)
- Read the **compiled-memory breakdown** XLA emits (params, optimizer state, activations, scratch). It tells
  you which bucket is too big.
- Levers, cheapest first: lower `per_device_batch_size`; heavier `remat_policy`; more FSDP/TP sharding (more
  chips holding the params/optimizer state); enable activation/optimizer offload; reduce
  `max_target_length`; quantize.
- MoE: experts blow up memory fast — use expert parallelism and watch capacity factor.

---

## 9. Pathways and orchestration

- **Single-controller (default JAX):** one client process drives the whole SPMD program; fine up to large
  jobs but the controller and host coordination can become a bottleneck and a failure point at extreme scale.
- **Pathways:** a multi-controller runtime designed for **very large, multislice TPU jobs** — it decouples
  the client from the workers, enables more flexible/elastic device usage, and improves resilience and
  Goodput at pod scale. Reach for it when single-controller coordination or fault tolerance becomes the
  limiter. (Capabilities evolve — **verify current Pathways support and integration** for your setup.)
- **Goodput:** effective training time ÷ wall-clock. At thousands of chips, hardware/host failures are
  frequent, so fast checkpoint + restart, elastic rescheduling, and good monitoring directly raise Goodput.
  Google publishes Goodput/monitoring libraries for exactly this — instrument from day one.

---

## 10. Running on Kubernetes / GKE

- **TPU node pools** are created with a machine type and a **`topology`** (e.g. `2x2x2`, `4x4x4`); a
  multi-host slice spans several nodes that must be treated as one unit. See [[gke-master]] for node-pool/
  topology details and [[aiml-on-kubernetes]] for the broader ML-on-K8s picture.
- **Gang scheduling is mandatory** for multi-host: every host of a slice must start together or the job
  deadlocks. Use **JobSet** (and/or LeaderWorkerSet) to model the slice as a coordinated set of pods —
  [[jobset-leaderworkerset]]. Multislice = multiple coordinated JobSet replicated groups.
- **XPK (Accelerated Processing Kit) / the runner pattern** is the common launcher: it provisions the TPU
  capacity, builds the workload, and submits it (typically materializing JobSet/Kubernetes objects under
  the hood) so you don't hand-write the multi-host plumbing. Treat XPK as the convenience layer over the
  JobSet/topology machinery.
- **Queueing/quota:** schedule TPU workloads through **Kueue** for batch admission, quota, and gang/TAS
  (topology-aware scheduling) — [[kueue-advanced]].
- Checkpoints go to **GCS** (`base_output_directory=gs://...`); ensure Workload Identity / bucket perms.

---

## 11. Inference: JetStream

**JetStream is the JAX/XLA (TPU) inference engine** for serving MaxText/JAX models. Key properties:

- **Continuous (in-flight) batching** + **paged-attention-style KV cache management** for high throughput
  on long, ragged request mixes.
- **Prefill/generate decomposition** (and disaggregation patterns) to use the accelerator efficiently
  across the compute-bound prefill and memory-bound decode phases.
- Loads MaxText checkpoints (often after conversion / an inference-optimized export); int8/FP8 quantization
  for throughput. **Verify the exact conversion + serving flow for your model and version.**

**When to use JetStream vs vLLM/SGLang ([[serving-frameworks]]):**

| Use **JetStream** | Use **vLLM / SGLang / TRT-LLM** |
|---|---|
| Serving a **MaxText/JAX model on TPU** | Serving on **GPU**, or PyTorch/HF-native checkpoints |
| You want TPU-native continuous batching + JAX stack consistency | You want the widest model/feature ecosystem, fast-moving OSS serving features |
| Keeping training and serving in one (JAX) toolchain | Heavy reliance on GPU kernels, speculative decoding variety, broad quant zoo |

There is also growing **vLLM-on-TPU** support — if you want vLLM's interface on TPU, evaluate that path too;
**check current maturity**. The honest decision is: JAX/TPU shop serving MaxText → JetStream; everything
else → the GPU/PyTorch serving stack.

---

## 12. Decision guide — MaxText/JAX/TPU vs PyTorch/GPU

| Choose **MaxText + JAX + TPU** | Choose **PyTorch + GPU** ([[training-frameworks]]) |
|---|---|
| TPU pod capacity; want top MFU via GSPMD | GPU fleet; CUDA ecosystem |
| Want a clean, config-driven, hackable reference | Need broad ecosystem, custom CUDA kernels, HF-native code |
| Like declarative named-axis sharding | Prefer explicit FSDP/Megatron tensor/pipeline parallel control |
| Multislice scaling, Pathways, JetStream serving | DeepSpeed/Megatron-LM, vLLM/TRT-LLM serving |
| Reproducible large-scale pretraining baseline | Existing PyTorch training/research stack |

JAX's strengths here: functional/SPMD model maps cleanly onto GSPMD, the compiler does global optimization,
and the named-axis sharding makes complex parallelism declarative. PyTorch's strengths: ecosystem breadth,
imperative debugging, and the maturity of GPU kernels/serving. Neither is universally "better" — pick by
**hardware you have, ecosystem you depend on, and team expertise.**

---

## 13. Best practices (opinionated)

- **Pin your stack.** JAX, jaxlib (matched to the TPU/CUDA runtime), MaxText commit, Flax, Optax, Orbax,
  Grain — record exact versions. JAX/jaxlib mismatch is a top cause of cryptic failures.
- **Start small, scale the config.** Validate the model + data on 1 host, then scale parallelism factors.
  The code shouldn't change; only `ici_*/dcn_*` and batch.
- **Get sharding right before chasing MFU.** A wrong `logical_axis_rules` is a correctness/throughput bug
  that no remat tuning will fix. Inspect the actual `NamedSharding` of your params.
- **Compile once.** Fixed shapes, stable mesh. Treat every recompile as a bug to hunt.
- **Lightest remat that fits.** Don't reflexively set `full`.
- **Checkpoint early and async.** `async_checkpointing` so saves don't stall the step loop; test **restore**
  before you need it. Depth in [[ml-checkpointing-orbax]].
- **Instrument Goodput and the profiler from step 0**, not after a 10k-chip run falls over.
- **Prefer config + CLI overrides** over forking model code; keep your diffs against upstream MaxText small.
- **Determinism:** fix seeds and use Grain's checkpointable, deterministic pipeline so a restart resumes the
  exact data stream.

## 14. Anti-patterns / gotchas

- **`full` remat everywhere "to be safe"** → silently caps MFU. Right-size it.
- **Putting tensor parallel on DCN** → collectives over the slow link tank throughput. TP belongs on ICI.
- **ICI×DCN factors not multiplying to chip count** → config error; don't fudge it.
- **Variable sequence length / dynamic shapes** → endless recompiles. Pad/bucket.
- **Ignoring the input pipeline** → you tuned the model to 60% MFU but the host can't feed it.
- **Quantizing without an eval gate** → shipped quality regressions. Measure on real evals.
- **Assuming HF and MaxText checkpoints interchange freely** → conversion is a real, lossy-if-careless step.
- **Flax linen vs NNX confusion** → NNX is the newer, more Pythonic/explicit-state API; linen is the
  established one. Know which your MaxText version/model uses; don't mix mental models. **Verify** which the
  current code path expects.
- **Letting XLA pick activation layouts** → use sharding constraints to pin the important ones.
- **Single-controller at extreme scale** → coordination/fault-tolerance ceiling; consider Pathways.

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Periodic step-time spikes | Recompilation | Fix dynamic shapes; keep mesh constant; pad to `max_target_length` |
| First step very slow, rest fast | Normal one-time compile | Expected — warm up before timing |
| OOM at start | Params+optimizer+activations exceed HBM | More sharding (FSDP/TP), lighter batch, heavier remat, offload |
| OOM mid-run | Activation growth (long seq / large batch) | Heavier remat, smaller `per_device_batch_size`, reduce seq len |
| MFU low, chips idle in profile | Input bottleneck | More Grain workers/prefetch; per-host data sharding |
| MFU low, lots of collectives | Bad `logical_axis_rules` / sharding | Re-map logical axes; pin with sharding constraints; inspect HLO |
| Throughput drops across slices | DCN straggler / TP on DCN | Move heavy collectives to ICI; check slice health |
| `jaxlib`/runtime errors on TPU | Version mismatch | Align JAX/jaxlib with the TPU runtime; reinstall matched wheels |
| Loss diverges with quantization | Quant numerics | Re-check AQT mode; keep reductions/master in higher precision |
| Restart resumes wrong data | Non-deterministic pipeline | Use Grain checkpointable iterator; checkpoint data state |

---

## 16. Canonical references (verify against current versions)

- **MaxText** — https://github.com/AI-Hypercomputer/maxtext
- **JAX** — https://github.com/jax-ml/jax · docs: https://jax.readthedocs.io
- **JAX sharding / distributed** — https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- **Flax (linen + NNX)** — https://github.com/google/flax · https://flax.readthedocs.io
- **Optax** — https://github.com/google-deepmind/optax
- **Grain** — https://github.com/google/grain
- **Orbax** — https://github.com/google/orbax (see [[ml-checkpointing-orbax]])
- **AQT (quantization)** — https://github.com/google/aqt
- **JetStream** — https://github.com/AI-Hypercomputer/JetStream
- **MaxDiffusion** — https://github.com/AI-Hypercomputer/maxdiffusion
- **XPK** — https://github.com/AI-Hypercomputer/xpk
- **Pathways / multislice & Goodput docs** — Google Cloud TPU multislice and ML Goodput documentation
- **Cloud TPU multislice overview** — https://cloud.google.com/tpu/docs/multislice-introduction
- **Levanter** — https://github.com/stanford-crfm/levanter · **Paxml** — https://github.com/google/paxml

---

# MaxText / JAX LLM — Worked Examples

Canonical, shape-correct sketches to imitate. **Config keys, flags, and APIs move fast (2026) — verify
every concrete key against the MaxText/JAX/JetStream version you are running.** These show the *idiom*,
not a guaranteed-runnable copy-paste.

---

## 1. Multi-host TPU training launch (config + JobSet/XPK)

### 1a. The MaxText invocation

Train Llama-3.1-8B on a multi-host TPU slice with **FSDP within the slice** (ICI). Everything is config —
the same command scales by changing the parallelism factors and adding a `dcn_*` factor for multislice.

```bash
# Run INSIDE each TPU worker host (SPMD — the same program runs on every host).
python3 -m MaxText.train MaxText/configs/base.yml \
    model_name=llama3.1-8b \
    run_name="${RUN_NAME}" \
    base_output_directory="gs://my-bucket/maxtext/${RUN_NAME}" \
    dataset_type=grain \
    dataset_path="gs://my-bucket/data" \
    tokenizer_path="gs://my-bucket/tokenizers/llama3.1" \
    steps=50000 \
    per_device_batch_size=4 \
    max_target_length=8192 \
    attention=flash \
    remat_policy=save_dot_except_mlp \
    weight_dtype=bfloat16 \
    ici_fsdp_parallelism=-1 \
    ici_tensor_parallelism=1 \
    enable_checkpointing=true \
    async_checkpointing=true \
    checkpoint_period=2000 \
    opt_type=adamw \
    learning_rate=3e-4 \
    gradient_clipping_threshold=1.0
# ici_fsdp_parallelism=-1 means "fill the remaining mesh axis" — product of all
# ici_* (and dcn_*) factors must equal the total chip count. VERIFY the -1 convention.
```

**To go multislice** (e.g. 4 slices, FSDP in-slice + data parallel across slices), add:

```bash
    ici_fsdp_parallelism=<chips_per_slice> \
    dcn_data_parallelism=4               # one factor per slice; ICI×DCN == total chips
```

### 1b. Launching it on GKE

**Option A — XPK (the convenience launcher).** XPK provisions the TPU capacity and submits the workload,
materializing the multi-host Kubernetes objects (typically JobSet) for you:

```bash
# Illustrative — verify current XPK subcommands/flags.
xpk workload create \
    --cluster my-tpu-cluster \
    --workload "${RUN_NAME}" \
    --tpu-type=v5litepod-256 \
    --num-slices=1 \
    --command "python3 -m MaxText.train MaxText/configs/base.yml model_name=llama3.1-8b run_name=${RUN_NAME} ..."
```

**Option B — JobSet directly** (when you want explicit control; see [[jobset-leaderworkerset]]). A multi-host
TPU slice is one `replicatedJob` whose `parallelism`/`completions` equal the number of hosts, with the TPU
topology requested via nodeSelector and the chip count via resource limits. All hosts must be
**gang-scheduled** (via Kueue, [[kueue-advanced]]) so the SPMD program starts together.

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: maxtext-llama3-8b
  labels:
    kueue.x-k8s.io/queue-name: tpu-queue          # gang admission via Kueue
spec:
  replicatedJobs:
    - name: workers
      replicas: 1                                  # 1 slice; bump for multislice
      template:
        spec:
          parallelism: 64                          # = number of HOSTS in the slice
          completions: 64
          backoffLimit: 0
          template:
            spec:
              nodeSelector:
                cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
                cloud.google.com/gke-tpu-topology: 16x16   # slice topology
              containers:
                - name: maxtext
                  image: gcr.io/my-project/maxtext:latest
                  command: ["bash", "-c"]
                  args: ["python3 -m MaxText.train MaxText/configs/base.yml model_name=llama3.1-8b ..."]
                  resources:
                    limits:
                      google.com/tpu: 4            # chips attached to THIS host VM
# Verify apiVersion, accelerator/topology label values, and chips-per-host against current
# GKE TPU docs ([[gke-master]]). Multislice = replicas>1 plus dcn_* parallelism in the config.
```

---

## 2. JAX `Mesh` / `PartitionSpec` sharding snippet

Raw-JAX illustration of what MaxText's logical-axis machinery does under the hood: build a device mesh with
named axes, then shard a parameter and an activation with `PartitionSpec` / `NamedSharding`.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# 1. Physical mesh: 2D grid of named axes. Product of dims == device count.
#    Here: data-parallel x model(tensor)-parallel.
devices = mesh_utils.create_device_mesh((4, 2))          # e.g. 8 chips -> 4 x 2
mesh = Mesh(devices, axis_names=("data", "model"))

# 2. Shard a weight matrix [d_model, d_ff] over the "model" axis (tensor parallel),
#    replicated over "data".
def shard(spec):
    return NamedSharding(mesh, spec)

w = jnp.ones((8192, 28672))                               # an MLP up-projection
w = jax.device_put(w, shard(P(None, "model")))           # column-sharded across TP axis

# 3. Shard a batch of activations [batch, seq, d_model]: batch over "data",
#    hidden over "model" (a common FSDP/TP-style activation layout).
x = jnp.ones((256, 8192, 8192))
x = jax.device_put(x, shard(P("data", None, "model")))

# 4. jit with explicit in/out shardings; XLA (GSPMD) inserts the collectives.
@jax.jit
def mlp(x, w):
    y = jnp.einsum("bsd,df->bsf", x, w)                  # all-gather/all-reduce inferred by XLA
    return jax.lax.with_sharding_constraint(             # pin the intermediate layout
        y, shard(P("data", None, "model")))

y = mlp(x, w)
print(y.sharding)                                        # inspect the actual NamedSharding

# MaxText layer: instead of writing P(...) by hand, you tag arrays with LOGICAL axes
# ("embed", "mlp", "batch", ...) and a logical_axis_rules table maps them to these
# physical axes ("data", "model"/"tensor", "fsdp", ...). Same result, expressed in model terms.
```

**Mental check:** `P("data", None, "model")` means dim-0 sharded over the `data` mesh axis, dim-1
replicated, dim-2 sharded over the `model` axis. `None` = replicated. The mesh axis names are the *physical*
parallelism dimensions; MaxText's `logical_axis_rules` is just a named indirection on top of this.

---

## 3. JetStream serving note (TPU inference)

Serve a MaxText checkpoint on TPU with **JetStream** — continuous batching + paged KV cache. The flow is:
(1) get an inference-ready checkpoint, (2) start the JetStream server bound to the MaxText engine,
(3) send requests.

```bash
# (1) Convert / export the trained MaxText checkpoint to an inference-optimized form.
#     The exact tool/flow varies by version — VERIFY against current MaxText/JetStream docs.

# (2) Launch the JetStream server backed by the MaxText engine on TPU.
#     Flags below are ILLUSTRATIVE of the idiom — verify names/values.
python3 -m maxengine_server \
    MaxText/configs/base.yml \
    model_name=llama3.1-8b \
    load_parameters_path="gs://my-bucket/maxtext/inference_ckpt" \
    per_device_batch_size=32 \
    max_target_length=8192 \
    ici_tensor_parallelism=8 \          # tensor-parallel decode within the slice (ICI)
    ici_fsdp_parallelism=1 \
    quantization=int8 \                  # int8/FP8 for throughput — gate quality on a real eval
    attention=flash

# (3) Query it (JetStream exposes a gRPC/HTTP serving interface; verify the current client).
```

**Choosing JetStream vs vLLM/SGLang ([[serving-frameworks]]):** serving a MaxText/JAX model on **TPU** →
JetStream (TPU-native continuous batching, one JAX toolchain for train+serve). GPU fleet, HF/PyTorch-native
checkpoints, or you want the broadest serving-feature ecosystem → vLLM / SGLang / TensorRT-LLM. If you want
vLLM's interface specifically on TPU, evaluate **vLLM-on-TPU** maturity for your model.

---

## Notes on correctness of these examples

- The `∏ ici_* × ∏ dcn_* == total_chips` invariant is real and enforced — keep it satisfied.
- TP and other bandwidth-heavy collectives belong on **ICI** (`ici_tensor_parallelism`), DP/FSDP can cross
  **DCN** (`dcn_data_parallelism`) for multislice.
- `google.com/tpu` resource limit is the **chips on one host VM**, not the whole slice; `parallelism` is the
  **host count**. Together they cover the slice.
- All specific config keys, XPK/JobSet fields, GKE TPU labels, and JetStream flags here are **idiom
  illustrations** — confirm exact names against the versions you run before depending on them.
