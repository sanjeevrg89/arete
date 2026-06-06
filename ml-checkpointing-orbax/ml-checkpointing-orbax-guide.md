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
