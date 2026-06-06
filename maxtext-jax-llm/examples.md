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
