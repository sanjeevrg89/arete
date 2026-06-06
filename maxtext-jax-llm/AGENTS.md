# AGENTS.md — MaxText & JAX LLM Stack

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`maxtext-jax-llm-guide.md`** next to this file — read it
> before doing MaxText/JAX-LLM work. Concrete artifacts to imitate (training launch, JetStream serving,
> `Mesh`/`PartitionSpec` snippet) are in **`examples.md`**. This is the always-on summary.
>
> **The ecosystem moves fast (2026).** Config keys, `remat_policy` options, quantization modes, supported
> models, and APIs change between releases. Treat concrete flags as idiomatic illustrations and **verify
> against the version in use**. Never fabricate flags, config keys, or benchmark numbers.

## When working with MaxText / JAX LLMs, apply these by default:

- **Config-driven, not code-driven.** Behavior = merged config (`base.yml` + model YAML + CLI `key=value`).
  Change YAML/CLI for parallelism, remat, dtype, attention kernel, dataset, checkpoints — not Python.
- **Sharding = `Mesh` of named physical axes (`data`/`fsdp`/`tensor`/`sequence`/`expert`) + `logical_axis_rules`
  mapping logical axes (`embed`/`mlp`/`heads`/`length`/`vocab`/`batch`) onto them.** GSPMD/XLA inserts the
  collectives. Get the mapping right → parallelism is correct by construction. Pin key activations with a
  sharding constraint.
- **ICI vs DCN is the scaling axis.** `ici_*_parallelism` shards within a slice (fast interconnect);
  `dcn_*_parallelism` shards across slices (multislice, slower DCN). `∏ ici × ∏ dcn == total_chips`.
  Keep TP / heavy collectives on **ICI**; put DP/FSDP across **DCN**. Common: FSDP in-slice, DP across slices.
- **FSDP is the dense-model workhorse.** Add TP when a layer doesn't fit or to cut latency; sequence parallel
  for long context; expert parallel for MoE. Combine them (mesh gains axes).
- **MFU is the scoreboard.** Track logged per-device TFLOP/s vs chip peak (measure — don't assume a number).
  Low MFU → check: recompilation, per-device batch too small, remat too heavy, input bottleneck, bad sharding,
  wrong attention kernel. Profile with the JAX/XLA profiler.
- **Remat trades compute for HBM.** Use the **lightest `remat_policy` that fits HBM**; reserve `full` for
  memory-bound runs. Don't reflexively set `full` — it caps MFU.
- **Compile once.** XLA compiles per (shape, dtype, sharding, mesh) signature. Fixed shapes (pad/bucket
  sequence length) + stable mesh. Treat every recompile as a bug.
- **OOM levers (cheapest first):** lower `per_device_batch_size` → heavier remat → more FSDP/TP sharding →
  activation/optimizer offload → reduce seq len → quantize. Read XLA's compiled-memory breakdown first.
- **Quantization via AQT** (`int8`, FP8 where supported). Always gate on a real eval, keep
  reductions/master state in higher precision. **Verify modes.**
- **Checkpoint with Orbax**, async + sharded, to GCS; test **restore** before relying on it. Conversion
  to/from HF is a distinct, careful step. Depth: [[ml-checkpointing-orbax]].
- **On GKE:** TPU node pools have a `topology`; multi-host slices need **gang scheduling** via
  **JobSet/LWS** ([[jobset-leaderworkerset]]); **XPK** is the common launcher; queue/quota via Kueue
  ([[kueue-advanced]]); see [[gke-master]] / [[aiml-on-kubernetes]].
- **Serving:** **JetStream** for TPU-native serving of MaxText/JAX models (continuous batching, paged KV).
  GPU/PyTorch-native → vLLM/SGLang/TRT-LLM ([[serving-frameworks]]). Evaluate vLLM-on-TPU maturity if needed.
- **Instrument Goodput + profiler from step 0.** At pod scale, fast checkpoint/restart and elastic restart
  are essential. Pin exact JAX/jaxlib/MaxText/Flax/Optax/Orbax/Grain versions (jaxlib↔runtime match).

## Decision (one line)
TPU pods + want top MFU, GSPMD sharding, clean config-driven reference → **MaxText/JAX**. GPU fleet, CUDA
ecosystem, PyTorch code, broad kernels → **PyTorch/FSDP/Megatron** ([[training-frameworks]]).

## Definition of done
Sharding verified (inspect actual `NamedSharding` of params); compiles once (no recompile loop); fits HBM
with right-sized remat; MFU measured and explained; checkpoints save **and restore**; versions pinned;
any quantization passed an eval gate. Flag every fast-moving flag/key you relied on as "verify current docs".
