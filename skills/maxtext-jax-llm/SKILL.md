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
