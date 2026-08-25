---
name: ml-checkpointing-orbax
description: Expert ML checkpointing at scale for resilient large-model training, centered on Orbax (the
  JAX checkpointing library) with the full landscape. Use when saving/restoring model state in JAX/Flax
  or PyTorch training, when training stalls or wastes work on failures, or when designing checkpoint
  resilience for thousand-accelerator jobs. Covers Orbax CheckpointManager (async, sharded jax.Array,
  composite/PyTree, retention policies, transformations on restore, emergency/in-memory peer-replica
  checkpointing), PyTorch torch.distributed.checkpoint (DCP) with FSDP full/sharded state dicts and async
  staging, Multi-Tier Checkpointing (MTC) to node-local SSD + Cloud Storage on GKE, the GCS/Hyperdisk
  ML/Parallelstore IO story, goodput/MFU and save-stall math, deterministic data-iterator resume,
  resharding on a different topology at restore, and elastic/restartable training. Triggers on orbax,
  CheckpointManager, AsyncCheckpointer, DCP, FSDP state_dict, MTC, checkpoint frequency, save stall,
  resharding, goodput.
---

# ML Checkpointing at Scale (Orbax-centered)

Apply the judgment of an engineer who keeps thousand-accelerator training jobs resilient: checkpointing
is not "save a file" — it is the dominant lever on **goodput** (useful compute / wall-clock) for long
runs. The bar: saves overlap compute and never stall the step, restores are correct across topology
changes, and a node loss costs minutes, not hours.

## How to use this skill

1. **Read `ml-checkpointing-orbax-guide.md`** in this directory — the full reference (mental model,
   Orbax API idioms, PyTorch DCP, Multi-Tier Checkpointing, resiliency engineering, anti-patterns,
   troubleshooting). Apply it to the task.
2. For concrete code to imitate — Orbax `CheckpointManager` async sharded save/restore, torch DCP
   save/load, and an MTC tiered config — read **`examples.md`**.
3. Match the surrounding stack's conventions (framework, storage backend, orchestrator). Apply the
   correctness rules — async non-blocking save, what-to-checkpoint completeness, deterministic resume —
   regardless. This ecosystem moves fast; **verify exact API field names and flags against current docs.**

## Essentials (full detail in `ml-checkpointing-orbax-guide.md`)

- **Goodput is the metric.** Wasted work per failure ≈ time since last checkpoint + restart cost.
  Optimal checkpoint interval ≈ `sqrt(2 · checkpoint_cost · MTBF)` (Young/Daly). At thousand-chip scale
  MTBF is hours, so checkpoint **often** — but only if the save is asynchronous and nearly free.
- **The save-stall problem is the whole game.** A synchronous save freezes every accelerator while bytes
  flush to storage. Use **async checkpointing**: copy device arrays to host, return control, flush in a
  background thread. The step blocks only for the device→host copy, not the network write.
- **Checkpoint everything needed for bit-exact resume:** params, optimizer state (momentum/Adam moments),
  PRNG keys, `step`, the **data-iterator position**, and EMA/extra metrics. Missing the data position
  silently changes your training distribution on resume.
- **Orbax `CheckpointManager` owns the lifecycle:** numbered step directories, `save(step, args=...)` /
  `restore(...)`, retention (`max_to_keep`, `keep_period`), and `should_save(step)`. Wrap async with
  `wait_until_finished()` before exit. It is the canonical JAX/Flax checkpointer (replaces old
  `flax.training.checkpoints`).
- **Sharded `jax.Array` checkpoints write per host, in parallel.** Each process writes only its local
  shards; no single-writer bottleneck. On restore, pass the **target `sharding`/`abstract` PyTree** so
  Orbax reshards from disk to the live mesh — this is how you restore on a **different topology**.
- **Resharding on restore is a first-class feature, not a hack.** Save on 512 chips, restore on 256:
  give the restore the new shardings and Orbax handles the redistribution. Don't gather to one host.
- **Emergency / in-memory checkpointing** keeps a recent checkpoint in host RAM and replicates across
  peer slices; on a single-node failure you restart from a peer in seconds instead of re-reading GCS.
  Pair frequent in-memory snapshots with less-frequent persistent ones.
- **PyTorch: use `torch.distributed.checkpoint` (DCP), not `torch.save(model.state_dict())`** for
  sharded models. DCP does sharded, resharding-capable, parallel save/load; use FSDP `SHARDED_STATE_DICT`
  for scale and async DCP (`async_save`) to overlap. `FULL_STATE_DICT` only for small models/export.
- **Multi-Tier Checkpointing (MTC):** write to **node-local SSD first** (fast) and replicate to durable
  **Cloud Storage** in the background; restart reads from local SSD or a peer replica, falling back to
  GCS only when the slice is truly gone. This is the GKE pattern for fast restart at scale — see
  `[[gke-master]]` for the node-pool/storage wiring.
- **Storage IO must match scale:** GCS for durability and throughput at fleet scale, **Hyperdisk ML**
  for fast read-mostly loads, **Parallelstore** for high-throughput parallel POSIX. Single-bucket,
  single-prefix patterns hot-spot; shard the layout.
- **Deterministic resume or it didn't happen.** Restore PRNG and data position; reseed dataloaders by
  `step`; verify loss continuity across the restart. Non-deterministic data resume is a silent
  correctness bug, not a performance one.

## Anti-patterns (never do these)

- Synchronous save that blocks the training step every N steps.
- Checkpointing on every step (IO-bound) or once a day (lose hours per failure) — compute the interval.
- A single host gathering the full state and writing alone (memory blow-up + serial bottleneck).
- Checkpointing params but not optimizer state / PRNG / data position → non-resumable or distribution drift.
- No retention policy → unbounded storage growth; or `max_to_keep=1` with no durable copy → one bad write loses everything.
- Assuming restore only works on the identical topology; not testing restore on a different mesh.

## Related skills

- `[[maxtext-jax-llm]]` — production JAX/Flax LLM training that uses Orbax for checkpointing end-to-end.
- `[[ml-frameworks]]` — JAX/`jax.Array`/sharding, XLA, PyTorch/FSDP fundamentals underneath this.
- `[[training-frameworks]]` — FSDP/DeepSpeed/Megatron/MaxText training loops that produce the state to checkpoint.
- `[[gke-master]]` — node-local SSD, Hyperdisk ML, Parallelstore, GCS wiring for Multi-Tier Checkpointing.
- `[[aiml-on-kubernetes]]` — orchestrating elastic/restartable training jobs that recover from failures.
