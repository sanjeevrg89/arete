# AGENTS.md — ML Checkpointing at Scale (Orbax-centered)

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ml-checkpointing-orbax-guide.md`** next to this file —
> read it before designing or reviewing checkpointing. Concrete code to imitate (Orbax async sharded
> save/restore, torch DCP, MTC config) is in **`examples.md`**. This file is the always-on summary.
>
> **The metric is goodput** (useful compute / wall-clock), not "a file got saved." This ecosystem moves
> fast (2026): **verify exact Orbax/PyTorch API field names, class names, and flags against current docs.**

## When checkpointing ML training (JAX/Flax/Orbax or PyTorch/FSDP), apply by default:

- **Saves must be asynchronous.** The step blocks only for the device→host copy; the network write
  happens in the background. A synchronous save that freezes accelerators is wrong at scale. After
  `save()` returns the bytes are **not yet durable** — call `wait_until_finished()` (Orbax) / await the
  future (DCP) before exit and in signal handlers.
- **Checkpoint the *complete* state:** params, optimizer state (Adam moments / Adafactor / momentum),
  PRNG keys, `step`, and the **data-iterator position**. Missing the data position silently changes the
  training distribution on resume — a correctness bug, not a perf one.
- **Pick the interval, don't guess.** Wasted work per failure ≈ half an interval + restart cost.
  Young/Daly optimum `interval ≈ sqrt(2 · T_ckpt · MTBF)`. At thousand-chip scale MTBF is hours, so
  checkpoint **often** — which only works if saves are async/multi-tier and nearly free.
- **Write sharded, per host, in parallel.** Each process writes only its local `jax.Array`/FSDP shards.
  Never gather the full state to one host (host OOM + serial bottleneck + checkpoint locked to one
  world size).
- **Orbax is the JAX/Flax checkpointer.** `CheckpointManager` owns numbered step dirs, `save(step,
  args=...)`/`restore(...)`, `should_save`, retention (`max_to_keep`, `keep_period`,
  `save_interval_steps`). It finalizes with a sentinel so torn writes are skipped. Don't use the
  deprecated `flax.training.checkpoints`.
- **Restore onto target shardings; resharding is supported.** Pass the current mesh's shardings/abstract
  tree on restore so Orbax reshards from disk to the live topology (e.g. 512→256 chips). Use restore-time
  transformations for renamed/added params. **Test restore-on-different-topology in CI.**
- **Orbax emergency / in-memory checkpointing** keeps a recent checkpoint in host RAM replicated across
  peer slices for seconds-fast restart on single-node loss. Pair frequent in-memory snapshots with
  less-frequent durable GCS ones. (Newest, most-changed API — verify.)
- **PyTorch: use `torch.distributed.checkpoint` (DCP), never `torch.save(model.state_dict())` for
  sharded models.** DCP does parallel, resharding-aware, load-in-place save/load. Use FSDP
  `SHARDED_STATE_DICT` at scale (`FULL_STATE_DICT` only for small models/export), and `dcp.async_save`
  to overlap. Use the current `get_state_dict`/`set_state_dict` helpers (verify names per torch version).
- **Multi-Tier Checkpointing (MTC) on GKE:** write node-local SSD first → replicate to a peer →
  background to GCS. Restart from the fastest valid tier (local → peer → GCS). Match storage to access:
  GCS (durable, shard the prefix), Hyperdisk ML (fast read-mostly loads), Parallelstore (parallel POSIX
  IO), Local SSD (tier-0, must be backed durably). See `[[gke-master]]`.
- **Deterministic resume or it didn't happen.** Restore PRNG + reseed data by `step`; verify loss
  continuity across the restart (no spike/dip).
- **Retention with depth.** `max_to_keep > 1` plus a `keep_period` milestone so one bad/NaN write can't
  strand the run; never unbounded growth, never `max_to_keep=1` with no durable copy.

## Anti-patterns (reject in review)
Synchronous save blocking the step · single-host gather-and-write · params-only (no optimizer/PRNG/data
position) · non-deterministic data resume · checkpoint every step (IO-bound) or once a day (lose hours) ·
no retention policy · restore tied to a fixed topology · trusting async durability before the wait/future.

## Reviewing / designing a checkpointing change
Walk the troubleshooting and anti-pattern sections of `ml-checkpointing-orbax-guide.md`.
