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

---

# Reference — ml-checkpointing-orbax

# ML Checkpointing at Scale — Orbax-centered Guide

The full reference. Mental model first, then Orbax, then the PyTorch side, then Multi-Tier
Checkpointing and the storage IO story, then resiliency engineering, anti-patterns, and
troubleshooting. This ecosystem moves fast (it is 2026): **verify exact API field/argument names,
class names, and flags against the current Orbax and PyTorch docs** — they are flagged inline.

## 1. Why checkpointing dominates large-training resilience

A pretraining run on hundreds-to-thousands of accelerators is a distributed system that *will* fail:
a host OOMs, an optical link flaps, a NIC resets, an XLA compilation hangs, preemption reclaims a
node. The only thing standing between a transient fault and re-running days of compute is the last
good checkpoint. So checkpointing is a **resilience** subsystem, judged by **goodput**, not a
serialization convenience.

### Goodput and MFU

- **Goodput** = (time spent on useful forward/backward compute) / (wall-clock time). Everything
  else — save stalls, restart, recompute of lost steps, idle-while-rescheduling — is lost goodput.
- **MFU** (Model FLOPs Utilization) measures how well a single healthy step uses the hardware. You
  can have great MFU and terrible goodput if failures and save stalls eat the wall clock.
- Two checkpointing levers move goodput: **save stall** (does saving freeze the step?) and **wasted
  work per failure** (how much progress is lost + how long to get computing again).

### The wasted-work / MTTF math (intuition)

Let `T_ckpt` be the cost to write a checkpoint and `MTBF` the mean time between failures for the whole
job (which *shrinks* as you add chips — more components, more failures). On a failure you lose, in
expectation, half a checkpoint interval of compute plus the restart cost.

The classic **Young/Daly** optimum for checkpoint interval is:

```
interval* ≈ sqrt(2 · T_ckpt · MTBF)
```

Intuition: checkpoint more often when checkpoints are cheap or failures are frequent; less often when
checkpoints are expensive or the job is stable. At thousand-chip scale `MTBF` is **hours**, which would
push the interval low — but a *synchronous* `T_ckpt` of minutes makes frequent saving ruinous. The
resolution is to drive `T_ckpt` (the part that blocks the step) toward zero with **async** and
**multi-tier** checkpointing, which then *lets* you checkpoint frequently and keep wasted work small.

### What to checkpoint (completeness)

A checkpoint must let you resume **bit-for-bit equivalently**. Save:

- **Model params** — the weights (sharded across the mesh).
- **Optimizer state** — Adam/AdamW first and second moments, Adafactor factors, momentum, loss
  scaler state. This is often *as large as* the params; forgetting it makes resume meaningless.
- **PRNG keys** — dropout, augmentation, sampling. Restore so randomness continues, not restarts.
- **`step` / global step** — the scalar that schedules LR, data, and stopping.
- **Data-iterator position** — where the input pipeline is. Omitting this re-reads from epoch start
  or a random spot, **silently changing the training distribution**. This is the most-skipped item.
- **Schedules / metrics / EMA** — LR schedule state if stateful, EMA weights, best-metric trackers.

If any of these is missing, you don't have a resumable checkpoint — you have a warm-start initializer.

## 2. Orbax — the JAX checkpointing library

Orbax is the canonical checkpointing stack for JAX/Flax. It replaced the older
`flax.training.checkpoints` API; new code should use Orbax directly. Two layers matter:

- **`orbax.checkpoint`** — the low-level building blocks: `Checkpointer`, `AsyncCheckpointer`,
  `PyTreeCheckpointHandler`, `StandardCheckpointHandler`, type handlers, and `args`-based save/restore.
- **`CheckpointManager`** — the lifecycle manager over a directory of numbered steps: when to save,
  what to keep, async coordination, metrics. This is what your training loop talks to.

> Orbax has migrated APIs across versions (e.g. the move to the `args=` interface and `ocp.args.*`,
> handler registration, and an evolving `emergency` / replicator-based checkpointing surface). Treat
> exact class and argument names below as **shape, not gospel — verify against the installed version.**

### 2.1 CheckpointManager: lifecycle, retention, versioning

`CheckpointManager` owns a root directory. Each save creates a `…/<step>/` subdirectory. It exposes:

- `save(step, args=...)` — returns immediately for async; the actual write may complete later.
- `restore(step, args=...)` — reconstruct the PyTree (optionally onto target shardings).
- `should_save(step)` — honor the configured save interval / predicate.
- `wait_until_finished()` — block until all in-flight async saves have flushed. **Call before exit.**
- `latest_step()` / `all_steps()` — discover what's on disk for resume.

Retention and versioning live in `CheckpointManagerOptions`:

- `save_interval_steps` — periodic cadence.
- `max_to_keep` — keep the N most recent step dirs; older ones are garbage-collected.
- `keep_period` — additionally retain every k-th step permanently (e.g. milestone checkpoints).
- `enable_async_checkpointing` — overlap save with compute (default on in recent versions).
- `cleanup_tmp_directories` — remove partial/incomplete step dirs left by a crash mid-write.

Orbax marks a step directory **complete** with a commit/finalize step (a sentinel file written last),
so a crash mid-write leaves an *incomplete* directory that restore skips and cleanup removes. This is
what makes restore safe: you never half-restore a torn write.

### 2.2 Async checkpointing — beating the save stall

The save stall problem: writing tens to hundreds of GB to network storage takes seconds-to-minutes;
doing it synchronously freezes every accelerator for that whole time.

Async checkpointing splits the save into two phases:

1. **Blocking, short:** copy device arrays (`jax.Array` shards) from accelerator HBM to host memory.
   This is the only part that gates the next step — typically a fraction of a second.
2. **Non-blocking, long:** a background thread serializes the host copy to storage. Training continues.

Use `AsyncCheckpointer` (or `CheckpointManager` with async enabled). The contract:

- After `save(...)` returns, the **device** is free but the **bytes may not be durable yet**.
- `wait_until_finished()` forces completion (before shutdown, or before relying on durability).
- Orbax coordinates across hosts so a checkpoint is only "complete" when *every* host finished —
  there is a distributed barrier on finalize.

Pitfall: don't mutate or donate the buffers you just handed to an async save before the host copy is
taken — and remember the next step's compute can race a still-flushing previous save's *network* I/O
(that's fine and intended), but not its *host copy* (Orbax handles the ordering).

### 2.3 Sharded / distributed checkpoints over `jax.Array`

Large model state is a PyTree of `jax.Array`s, each sharded across a device `Mesh` via a
`NamedSharding`. Orbax checkpoints these **natively and in parallel**:

- **Per-host writes:** each JAX process writes **only the shards it owns** to storage. No gather to a
  single host, no single-writer bottleneck. Throughput scales with the number of hosts.
- **Storage format:** arrays are stored in a chunked, sharded on-disk layout (Orbax uses a
  TensorStore-backed format under the hood), so individual shards can be read back independently.
- **Type handlers:** Orbax dispatches per leaf type (`jax.Array`, numpy, scalars, strings, custom
  objects) to a registered handler. You can register custom type handlers for non-standard leaves.
- **Transformations on restore:** you can rename keys, reshape, slice, or fill-in-new params when the
  model definition changed between save and restore — via the restore-time transform/`restore_args`
  mechanism. This is how you load an old checkpoint into an evolved model (added layers, renamed
  modules) without a manual surgery script.

#### Restoring onto a target sharding (and resharding)

On restore you pass a **target PyTree of shardings** (or an abstract/`ShapeDtypeStruct` tree with
shardings). Orbax reads each on-disk shard and **places it according to the target sharding** for the
*current* mesh. Because the on-disk layout is chunked independently of any particular mesh, you can:

- Restore on a **different number of chips / different mesh shape** than you saved on (e.g. 512 → 256,
  or change the data/tensor-parallel split). Orbax reshards from disk to the live topology.
- Restore params and optimizer state onto different shardings than they were saved with.

Do **not** restore by gathering the full unsharded array to one host and re-scattering — that defeats
the parallelism and can OOM the host. Always restore directly into the sharded target.

### 2.4 Composite / PyTree checkpoints

Real training state is heterogeneous: a params PyTree, an optimizer-state PyTree, a JSON-ish metadata
blob (step, config), a dataset-iterator state. Orbax composes these into one logical checkpoint via a
**composite/args** interface — multiple named items saved under one step directory, each with the
right handler:

- Arrays/PyTrees → `StandardSave`/`PyTreeSave` (the array handler).
- Plain metadata → a JSON handler.
- Dataset/iterator state → a handler that serializes the input pipeline's checkpoint (for `tf.data`,
  Grain, or a custom iterator). **Verify the exact handler/args names for your iterator library.**

This keeps one atomic, versioned step directory containing everything needed to resume.

### 2.5 Emergency / in-memory checkpointing (peer-replica recovery)

Reading a multi-hundred-GB checkpoint back from GCS on every restart is slow. Orbax's
**emergency / in-memory checkpointing** addresses fast restart at scale:

- Keep a recent checkpoint **in host RAM** (and/or node-local storage), replicated across **peer
  slices/replicas** so the state survives the loss of any one slice.
- On a single-node/slice failure, the replacement reads the latest state from a **healthy peer's
  in-memory copy** instead of from durable storage — restart in seconds, not minutes.
- Combine **frequent in-memory snapshots** (cheap, for the common single-failure case) with
  **less-frequent persistent GCS checkpoints** (for catastrophic / whole-job loss). This is the same
  tiering idea as MTC (§4), expressed at the Orbax layer.

> The emergency/replicator checkpointing surface is comparatively new and has evolved across Orbax
> releases (class names, replica/mesh-axis configuration). **Verify the current API and its
> assumptions (e.g. how replicas are defined) against the docs for your version.**

## 3. The PyTorch side — `torch.distributed.checkpoint` (DCP)

For PyTorch (FSDP / FSDP2 / tensor-parallel), the scalable equivalent of Orbax is
**`torch.distributed.checkpoint` (DCP)**. The single most important rule:

> **Do not `torch.save(model.state_dict())` for a sharded model.** That gathers the full state to one
> rank (memory blow-up, serial write) and produces a checkpoint tied to that exact world size. Use DCP.

### DCP model

- DCP saves a **sharded** checkpoint: each rank writes its own shards in parallel to a directory
  (`dcp.save(state_dict, checkpoint_id=path)`), and loads with `dcp.load(state_dict, checkpoint_id=path)`.
- DCP load is **resharding-aware**: it can load a checkpoint saved on a different world size / sharding
  by reading the needed pieces per rank — the analogue of Orbax restore-onto-target-sharding.
- DCP load is **load-in-place**: you pass an already-allocated `state_dict` (with the right sharded
  tensors) and DCP fills it, rather than returning fresh tensors.

### FSDP state-dict types

FSDP exposes the model/optimizer state in different shapes; pick deliberately:

| State dict type      | Shape on each rank          | Use for                                            |
|----------------------|-----------------------------|----------------------------------------------------|
| `FULL_STATE_DICT`    | full, unsharded (one rank)  | small models, final export, HF-compatible artifacts |
| `SHARDED_STATE_DICT` | this rank's shards          | scale training checkpoints with DCP (the default at scale) |
| `LOCAL_STATE_DICT`   | rank-local, flat            | legacy / niche; avoid for portable checkpoints      |

Use the current `torch.distributed.checkpoint.state_dict` helpers (`get_state_dict` /
`set_state_dict`, and the model/optimizer state-dict options) to extract and load sharded model +
optimizer state correctly — they handle the FSDP wrapping. **Verify exact helper names against your
torch version**; the state-dict API was reworked across releases and the old
`FSDP.state_dict_type(...)` context-manager pattern is being superseded.

### Async DCP

DCP supports **asynchronous save** (`dcp.async_save`) with the same stall-avoidance idea as Orbax:
stage tensors to CPU/pinned memory quickly, then write in the background while training proceeds.
Wait on the returned future before assuming durability or before process exit.

### Crossing frameworks

DCP and Orbax use different on-disk formats; there is no free interop. For portability prefer a
**neutral export** (e.g. a consolidated full state dict for PyTorch, or a defined export format). Keep
*training* checkpoints in the native sharded format (DCP / Orbax) for speed, and produce occasional
consolidated exports for serving/sharing.

## 4. Multi-Tier Checkpointing (MTC) on GKE

MTC is the pattern that makes frequent checkpointing essentially free at fleet scale by writing to a
fast local tier and a durable remote tier simultaneously. See `[[gke-master]]` for the node-pool and
storage wiring, and `[[aiml-on-kubernetes]]` for the job orchestration.

### Tiers and dataflow

1. **Tier 0 — node-local SSD (Local SSD / NVMe on the node):** the in-flight save lands here first.
   Fast, local, no network. Frequent checkpoints go here cheaply.
2. **Tier 1 — peer replica:** the local checkpoint is replicated to a peer node/slice so it survives a
   single node loss without touching durable storage.
3. **Tier 2 — Cloud Storage (GCS):** the durable copy, written in the background, for whole-job loss,
   preemption of the entire slice, or cross-cluster restore.

### Restart decision tree

On a failure, restart reads from the **fastest tier that still has a valid recent checkpoint**:

- Node restarted but local SSD intact → read **node-local** (seconds).
- Node gone, peer healthy → read from the **peer replica** (fast).
- Slice gone / cold start → fall back to **GCS** (slower but durable).

This collapses the common single-node-failure case from a multi-minute GCS reload to a near-instant
local/peer restore, which is what lets you checkpoint often without paying for it.

### Storage IO story

Match the storage backend to the access pattern (see `[[gke-master]]`):

| Backend           | Best for                                              | Notes                                              |
|-------------------|-------------------------------------------------------|----------------------------------------------------|
| **GCS**           | Durable checkpoints at fleet scale, high aggregate BW | Shard the key layout; avoid a single hot prefix.   |
| **Hyperdisk ML**  | Fast read-mostly loads (weights, restore)             | Read-optimized; great for fan-out reads on restart.|
| **Parallelstore** | High-throughput parallel POSIX checkpoint IO          | Parallel filesystem for many concurrent writers.   |
| **Local SSD/NVMe**| Tier-0 fast local checkpoints                         | Ephemeral — must be backed by a durable tier.       |

Per-host parallel writes (Orbax §2.3, DCP §3) are what saturate these backends; a single-writer design
cannot. With GCS, **shard the object layout** (per-host prefixes / step directories) so writes spread
across the backend instead of hammering one prefix. **Verify current product names, throughput
characteristics, and integration paths (CSI drivers) against current docs — these evolve.**

## 5. Resiliency engineering around checkpointing

Checkpointing is necessary but not sufficient; the surrounding loop must detect failures and recover.

- **Elastic / restartable training:** the job must be able to lose and replace workers and resume from
  the latest checkpoint without manual intervention. The orchestrator (JobSet/LeaderWorkerSet,
  elastic launchers) reschedules; the loop reads `latest_step()` and restores. See
  `[[aiml-on-kubernetes]]` and `[[training-frameworks]]`.
- **In-flight failure detection:** detect hangs (no step progress within a timeout), collective
  failures (NCCL/collective timeouts), and unhealthy hosts; fail fast rather than wedging the whole
  job. A hung step that never trips a watchdog wastes more than a clean crash.
- **Resharding at restore (different topology):** treat "restore on a different mesh/world size" as a
  *supported, tested* path — capacity changes, preemption gives you fewer nodes, you scale up after a
  warm-start. Orbax target-shardings (§2.3) and DCP resharding (§3) make this work; **test it in CI**,
  don't discover it during an incident.
- **Deterministic resume:** restore PRNG keys and **reseed the data pipeline by `step`** so the
  sequence of batches continues. Validate by checking **loss continuity** across the restart (no spike,
  no suspicious drop). Deterministic resume is a *correctness* property — a non-deterministic resume
  silently retrains on a different data distribution.
- **Validate the latest checkpoint before trusting it:** keep `max_to_keep > 1` and a `keep_period`
  milestone so one torn/corrupt write can't strand you. Orbax's finalize sentinel + cleanup handles
  *torn* writes; you still want depth for *bad* writes (NaNs saved, bug at save time).

## 6. Anti-patterns (call these out in review)

- **Synchronous save blocking the step.** Any save that freezes accelerators while bytes go to the
  network is wrong at scale. Make it async; the step blocks only for the device→host copy.
- **Checkpointing too often.** Every-step saving turns the job IO-bound for no goodput gain. Compute
  the interval (§1).
- **Checkpointing too rarely.** Once-a-day means a failure costs hours of compute. With async +
  multi-tier you can afford frequent saves — use them.
- **Single-writer bottleneck.** One host gathers the full state and writes alone: host OOM, serial
  throughput, and a checkpoint locked to one world size. Always per-host parallel writes.
- **Incomplete checkpoint.** Params only — no optimizer state, PRNG, or data position. Non-resumable,
  or silently changes the data distribution on resume.
- **Non-deterministic data resume.** Restarting the input pipeline from scratch or a random offset.
  Silent correctness bug.
- **No retention policy.** Unbounded step directories fill the bucket; or `max_to_keep=1` with no
  durable milestone, so one bad write loses the run.
- **Restore tied to fixed topology.** Code that only restores on the exact save-time mesh. Breaks the
  moment capacity changes. Use target-shardings / DCP resharding and test it.
- **Trusting async durability too early.** Assuming bytes are on storage right after `save()` returns,
  before `wait_until_finished()` / the future resolves. They aren't.

## 7. Troubleshooting (symptom → diagnosis → fix)

- **Throughput drops periodically, synced to save cadence.** → Save is (partly) synchronous, or the
  device→host copy is large/serial. → Enable async; confirm only the host copy blocks; reduce save
  frequency if still costly; consider multi-tier so the durable write is fully backgrounded.
- **Restore fails with sharding/shape mismatch.** → Target shardings don't match the live mesh, or the
  model definition changed. → Pass the correct target `restore_args`/shardings for the current mesh;
  use restore-time transformations for renamed/added params.
- **Restore OOMs the host.** → You're gathering the full array to one host. → Restore directly into the
  sharded target; never `device_get` the whole tree.
- **Loss spikes/dips right after a restart.** → PRNG not restored, optimizer state missing, or data
  position not resumed (distribution changed). → Checkpoint and restore the full state set (§1);
  verify loss continuity as a smoke test.
- **Partial/garbled checkpoint after a crash.** → A torn write. → Orbax skips incomplete step dirs
  (finalize sentinel) and `cleanup_tmp_directories` removes them; ensure you restore from
  `latest_step()` among *complete* steps, and keep depth (`max_to_keep > 1`).
- **GCS write throughput plateaus / 429s under many writers.** → Hot single prefix. → Shard the object
  layout (per-host prefixes, step subdirectories); consider Parallelstore/Hyperdisk ML for the hot path.
- **Restarts take many minutes reading weights from GCS.** → No fast tier. → Add node-local SSD +
  peer-replica (MTC / Orbax emergency checkpointing) so the common failure restores locally.
- **Process exits and the last checkpoint is missing.** → Async save didn't finish before shutdown. →
  `wait_until_finished()` (Orbax) / await the future (DCP) before exit, and on signal handlers.

## 8. Version awareness

This is fast-moving (2026). Specifically verify against current docs before relying on them:

- Orbax `args=`/`ocp.args.*` interface, handler names, `CheckpointManagerOptions` field names, and the
  **emergency/replicator** in-memory checkpointing API (newest and most-changed).
- PyTorch DCP state-dict helpers (`get_state_dict`/`set_state_dict`, options objects) and the
  deprecation status of the old `FSDP.state_dict_type` context manager; FSDP vs FSDP2 differences.
- GKE storage product names, throughput numbers, and CSI integration for GCS / Hyperdisk ML /
  Parallelstore, and the packaging of Multi-Tier Checkpointing.

Do not hardcode version numbers or benchmark figures from memory — measure on your stack and read the
current release notes.

## 9. Canonical references

- Orbax documentation: https://orbax.readthedocs.io/
- Orbax source: https://github.com/google/orbax
- JAX distributed arrays / sharding: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- Flax (uses Orbax for checkpointing): https://flax.readthedocs.io/
- PyTorch Distributed Checkpoint (DCP): https://pytorch.org/docs/stable/distributed.checkpoint.html
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- TensorStore (Orbax's array storage backend): https://google.github.io/tensorstore/
- Young/Daly optimal checkpoint interval (J. T. Daly, 2006): the basis for the `sqrt(2·T_ckpt·MTBF)` rule.
- ML goodput / large-scale training resilience: see vendor docs (Google Cloud "ML Goodput", GKE
  Multi-Tier Checkpointing) — **verify current product docs.**

---

# Examples — ML Checkpointing at Scale

Canonical, imitate-able snippets: Orbax `CheckpointManager` async sharded save + restore, PyTorch DCP
save/load, and a Multi-Tier Checkpointing config note. These are correct **in shape and idiom**. This
ecosystem moves fast (2026): **exact argument/field/class names are flagged where they have changed
across versions — verify against the docs for your installed version before relying on them.**

---

## 1. Orbax `CheckpointManager` — async, sharded save + restore

State is a PyTree of `jax.Array`s sharded over a device mesh, plus scalar metadata. We save params +
optimizer state + step + PRNG, asynchronously, with retention, and restore onto the *current* mesh's
shardings (so this also works after a topology change).

```python
import jax
import numpy as np
import orbax.checkpoint as ocp
from etils import epath  # epath.Path works for both local and gs:// paths

# --- training state (illustrative): a PyTree of sharded jax.Arrays + scalars ---
# train_state has .params (PyTree of jax.Array), .opt_state (PyTree), .step (int),
# and .rng (a jax PRNGKey). Each array carries a NamedSharding for the live mesh.

ckpt_dir = epath.Path("gs://my-bucket/runs/exp42/checkpoints")

options = ocp.CheckpointManagerOptions(
    save_interval_steps=500,          # cadence; also gate with mgr.should_save(step)
    max_to_keep=3,                    # keep the 3 most recent step dirs
    keep_period=10_000,               # also retain every 10k-th step permanently (milestones)
    enable_async_checkpointing=True,  # overlap the durable write with compute
    cleanup_tmp_directories=True,     # drop partial dirs from a crash mid-write
)

# CheckpointManager owns the directory of numbered step subdirs.
mgr = ocp.CheckpointManager(ckpt_dir, options=options)

# --- SAVE (non-blocking): returns after the device->host copy; write flushes in background ---
def save(train_state):
    if mgr.should_save(int(train_state.step)):
        # ocp.args.* is the current args-based interface; StandardSave handles the
        # PyTree of jax.Arrays (params + opt_state). VERIFY arg/handler names per version.
        mgr.save(
            int(train_state.step),
            args=ocp.args.StandardSave(train_state),
        )
    # NOTE: after save() returns, bytes may NOT be durable yet.

# --- RESTORE onto the current mesh's shardings (handles resharding) ---
def restore(abstract_state):
    # abstract_state: a PyTree matching train_state but with jax.ShapeDtypeStruct
    # leaves carrying the TARGET NamedSharding for the *current* mesh. Orbax reads each
    # on-disk shard and places it per this sharding -> restore on a different topology works.
    step = mgr.latest_step()            # None on a fresh run
    if step is None:
        return None
    return mgr.restore(
        step,
        args=ocp.args.StandardRestore(abstract_state),
    )

# --- on shutdown / before relying on durability ---
mgr.wait_until_finished()   # block until all in-flight async saves have flushed
mgr.close()
```

Key points (see the guide for detail):

- `save()` is async: only the device→host copy gates the next step; the GCS write is backgrounded.
- Each JAX process writes **only its local shards** — parallel, no single-writer bottleneck.
- On restore, the **target shardings come from `abstract_state`**, not from the on-disk layout, which
  is what makes resharding onto a different mesh work. Never `device_get` the whole tree to one host.
- For a **composite** checkpoint (params + opt_state + a JSON metadata blob + dataset-iterator state in
  one step dir), use the composite/args interface with one named item per component and the matching
  handler (array handler for PyTrees, JSON handler for metadata, an iterator handler for the input
  pipeline). **Verify the composite args class and the iterator handler names for your Orbax version
  and your data library (tf.data / Grain / custom).**

---

## 2. PyTorch DCP — sharded save + load (FSDP)

Use `torch.distributed.checkpoint` for sharded, resharding-capable, parallel checkpoints. Do **not**
`torch.save(model.state_dict())` for a sharded model.

```python
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict,   # VERIFY: current helpers; old FSDP.state_dict_type ctx is superseded
)

CKPT_DIR = "/mnt/ckpt/exp42/step_0001000"   # a shared/durable path all ranks can reach

# model is FSDP/FSDP2-wrapped; optimizer is its optimizer.
# get_state_dict returns SHARDED model + optimizer state by default (each rank: its shards).
def make_state_dict(model, optimizer, step, rng_state):
    model_sd, optim_sd = get_state_dict(model, optimizer)
    return {
        "model": model_sd,
        "optim": optim_sd,
        "step": step,            # scalar metadata travels in the same checkpoint
        "rng": rng_state,        # restore RNG for deterministic resume
    }

# --- SAVE: each rank writes its own shards in parallel ---
def save(model, optimizer, step, rng_state, ckpt_dir=CKPT_DIR):
    sd = make_state_dict(model, optimizer, step, rng_state)
    dcp.save(sd, checkpoint_id=ckpt_dir)
    # For overlap with compute, use dcp.async_save(...) and await the returned future before exit.

# --- LOAD: in-place into an allocated state_dict; resharding-aware ---
def load(model, optimizer, ckpt_dir=CKPT_DIR):
    # Allocate the target sharded state_dict first; DCP fills it in place.
    sd = make_state_dict(model, optimizer, step=0, rng_state=None)
    dcp.load(sd, checkpoint_id=ckpt_dir)         # reads only what each rank needs
    set_state_dict(                              # push restored shards back into model/optimizer
        model, optimizer,
        model_state_dict=sd["model"],
        optim_state_dict=sd["optim"],
    )
    return sd["step"], sd["rng"]
```

Key points:

- `SHARDED_STATE_DICT` semantics (via `get_state_dict`) at scale; `FULL_STATE_DICT` only for small
  models or export. DCP load is **load-in-place** and **resharding-aware** (different world size OK).
- `dcp.async_save` mirrors Orbax async: stage to CPU/pinned memory fast, write in the background, await
  the future before assuming durability or before process exit.
- The exact state-dict helper names and options objects have changed across torch versions and FSDP vs
  FSDP2. **Verify against your installed torch.**

---

## 3. Multi-Tier Checkpointing (MTC) — tiered config note

MTC writes a fast local tier and a durable remote tier so frequent checkpointing is essentially free at
scale, and restart reads from the fastest valid tier. Wiring (node-local SSD, Parallelstore/Hyperdisk
ML, GCS, CSI drivers) is in `[[gke-master]]`; orchestration in `[[aiml-on-kubernetes]]`.

Conceptual tier policy (names illustrative — **verify against the current MTC product/library config**):

```yaml
# ILLUSTRATIVE shape, not a verified schema — confirm exact keys against current MTC docs.
checkpointing:
  tiers:
    - name: local            # Tier 0: node-local SSD/NVMe — fastest, ephemeral
      backend: local-ssd
      path: /mnt/local-ssd/ckpt
      interval_steps: 100    # frequent, cheap
      retain: 2
    - name: peer             # Tier 1: replicate local ckpt to a peer node/slice
      backend: peer-replica  # survives single-node loss without touching durable storage
      replicas: 1
    - name: durable          # Tier 2: durable Cloud Storage — background, for whole-job loss
      backend: gcs
      path: gs://my-bucket/runs/exp42/checkpoints
      interval_steps: 1000   # less frequent; the catastrophic-recovery copy
      retain: 5
  restore_order: [local, peer, durable]   # read from the fastest tier that has a valid recent ckpt
```

Notes:

- Restart decision tree: node back + local SSD intact → **local** (seconds); node gone, peer healthy →
  **peer**; slice gone / cold start → **GCS**. This collapses the common single-node failure from a
  multi-minute GCS reload to a near-instant restore.
- Orbax **emergency / in-memory checkpointing** expresses the same tiering at the library layer (recent
  checkpoint in host RAM, replicated across peer slices). **Verify the current emergency-checkpointing
  API and how replicas/mesh axes are configured** — it is the newest, most-changed surface.
- Local SSD is **ephemeral** — it must always be backed by the durable GCS tier; never run with the
  local tier alone.
