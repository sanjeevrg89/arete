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
