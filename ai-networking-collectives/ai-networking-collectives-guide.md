# AI Networking & Collective Communication — Full Reference

The single source of truth for this skill. The competency here is the one that decides whether a
10,000-GPU job runs at 50% or 90% MFU. Parallelism *strategy* (which axes, what degrees) lives in
[[training-frameworks]]; this guide is the **communication layer underneath it** — the collectives,
the libraries that implement them, the interconnects they ride, and how to place jobs and diagnose
problems so the network stops being the bottleneck.

> **Fast-moving caveat.** NCCL/RCCL versions, NCCL_* env knobs, cloud transports (TCPX/TCPXO), driver
> stacks, and hardware bandwidths change frequently. This guide names mechanisms and the *shape* of
> tunables conceptually. **Never copy a flag or a bandwidth number from memory into production — verify
> against the current NCCL docs and your platform's docs.** Flags below marked *(verify)* especially.

---

## 1. Why the network decides your MFU

A distributed training step is `compute → communicate → compute`. The accelerators do matmuls; between
and within steps they must exchange gradients, parameters, activations, and tokens. **Communication
overhead is a dominant bottleneck in large distributed training**: as you scale out, the compute per
device stays roughly fixed but the volume and fan-out of communication grows, and any time a GPU spends
*waiting on the network instead of doing math* is lost MFU (Model FLOPs Utilization) / goodput.

At frontier scale — tens of thousands of GPUs — two things become non-negotiable:

1. **Topology-aware placement**: keeping the chattiest collectives inside high-bandwidth domains and
   minimizing expensive cross-fabric hops.
2. **A tuned collective stack**: the right algorithm, enough channels, the right transport
   (RDMA/SHARP), and comm overlapped with compute.

Get these wrong and a cluster that should run at ~90% MFU runs at ~50% — the GPUs are fine, they're
just idle waiting. The survey *"Understanding the network needs of LLM training"* (arXiv 2407.20018)
and Meta's *"How we built large-scale infrastructure to train LLMs"* (engineering.fb.com, 2024) both
make the same point: the network and collective stack are first-class design surfaces, not plumbing.

**Mental model:** treat the cluster as a hierarchy of bandwidth domains —
`intra-GPU (NVLink/NVSwitch) ≫ intra-node PCIe ≫ intra-rail/leaf (IB/RoCE) > cross-spine`. Your job is
to map each collective onto the smallest, fastest domain that can contain it, and to overlap whatever's
left with compute.

---

## 2. Collective operations — the vocabulary

A *collective* is a communication primitive over a group of ranks (one rank ≈ one GPU process). The
canonical set (MPI lineage, exposed by NCCL/RCCL and `torch.distributed`):

| Collective | What it does | Data moved (N ranks) |
|---|---|---|
| **AllReduce** | Reduce (sum/max/…) a tensor across all ranks; every rank gets the full result | ~2(N−1)/N · S |
| **ReduceScatter** | Reduce across ranks, each rank keeps a **shard** of the result | ~(N−1)/N · S |
| **AllGather** | Concatenate each rank's shard so every rank has the full tensor | ~(N−1)/N · S |
| **Broadcast** | One rank's tensor copied to all ranks | ~S |
| **Reduce** | Reduce to a single root rank | ~S |
| **AllToAll** | Each rank sends a distinct chunk to every other rank (transpose-like) | up to ~(N−1)/N · S |
| **Send/Recv (P2P)** | Point-to-point between two ranks | S |

Key identity: **AllReduce = ReduceScatter + AllGather.** Bandwidth-optimal AllReduce is implemented
exactly this way (ring), which is why ReduceScatter+AllGather (as in FSDP/ZeRO) costs the same total
bytes as one AllReduce but lets you shard memory in between.

### Where each collective appears (map to parallelism — see [[training-frameworks]])

- **Data parallel (DDP)** — gradient sync = **AllReduce** (or ReduceScatter then AllGather). Runs once
  per step over the whole DP group; the biggest single collective in classic DDP.
- **FSDP / ZeRO-3** — **AllGather** parameters before each layer's forward and backward, then
  **ReduceScatter** gradients. Many smaller collectives, latency- and overlap-sensitive; prefetch is
  what hides them.
- **Tensor parallel (TP / Megatron)** — **AllReduce** after each parallel GEMM block (two per
  transformer layer in the classic scheme); with **sequence parallelism** it becomes ReduceScatter +
  AllGather. Extremely frequent and latency-critical → must live on NVLink.
- **Mixture-of-Experts (MoE)** — **AllToAll** to dispatch tokens to expert ranks and combine results.
  Irregular, bursty, sensitive to load imbalance; often the dominant comm in MoE training.
- **Pipeline parallel (PP)** — **Send/Recv (P2P)** of activations forward and gradients backward
  between adjacent stages. Cheap per message but exposes the pipeline *bubble* if not overlapped.
- **Context/sequence parallel** — AllGather / ReduceScatter / AllToAll variants depending on scheme
  (e.g. ring-attention P2P, Ulysses all-to-all).

### Algorithms: ring vs tree vs hierarchical

- **Ring** — ranks form a logical ring; data flows around it in N−1 steps of S/N each. **Bandwidth-
  optimal** (each link saturated, total ≈ 2(N−1)/N·S for AllReduce) but **latency grows with N** (N−1
  hops). Best for **large messages**.
- **Tree** — reduce up a tree then broadcast down, depth ~log(N). **Latency-optimal** (log-depth) but
  links aren't all saturated. Best for **small messages / many ranks** (e.g. small DP all-reduces at
  high node count). NCCL ships a double-binary-tree all-reduce for this regime.
- **Hierarchical / multi-level** — do the collective *within* the NVLink domain first (cheap, fast),
  then one representative per node crosses the slower inter-node fabric, then redistribute. Maps the
  algorithm onto the bandwidth hierarchy; this is how large multi-node all-reduces stay efficient.
- **In-network reduction (SHARP)** — the switch ASIC performs the reduction, so GPUs send once and
  receive the result; removes a full traversal from the critical path for AllReduce/Reduce.

**Bandwidth- vs latency-optimal** is the core trade: small messages are latency-bound (favor tree /
low hop count), large messages are bandwidth-bound (favor ring / saturate links). NCCL picks per
message size and topology — your job is usually to *verify it chose well*, and to make messages big
enough (bucketing, fusion) that you're in the bandwidth-efficient regime.

---

## 3. NCCL / RCCL — the collective engine

**NCCL** (NVIDIA Collective Communications Library) on NVIDIA GPUs and **RCCL** (its ROCm counterpart)
on AMD GPUs are the de-facto implementations under PyTorch (`nccl` backend), JAX/XLA, DeepSpeed,
Megatron, etc. They provide optimized inter-GPU and multi-node collectives over NVLink, PCIe, IB, and
RoCE, and hide the topology behind a uniform API. NVIDIA's *"Scaling deep learning training with NCCL"*
(developer.nvidia.com/blog/scaling-deep-learning-training-nccl) is the canonical orientation.

### What NCCL does for you

1. **Topology discovery** — at init, builds a graph of GPUs, NVLink/NVSwitch, PCIe switches, NUMA
   nodes, and NICs (reads system topology; can ingest an XML topology file).
2. **Path/algorithm search** — constructs rings and trees across that graph and chooses algorithm
   (Ring / Tree / and collective-specific paths) and **protocol** (LL, LL128, Simple — latency vs
   bandwidth trade) per collective and message size.
3. **Channels** — splits each collective across multiple parallel **channels** (each ≈ a ring/tree
   instance) to use multiple NVLink/NIC links concurrently. More channels = more parallel bandwidth,
   up to a point.
4. **Transport** — picks the datapath: NVLink (P2P), PCIe/SHM intra-node, and for inter-node either
   **IB/RoCE verbs with GPUDirect RDMA**, or a TCP/sockets path (incl. cloud GPUDirect-TCPX/TCPXO).

### Reading what NCCL actually did — start here, always

Before touching any knob, make NCCL tell you its plan:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV   # rings/trees, chosen transports, env overrides
# run one short step / an nccl-test and read the log
```

The log prints the rings/trees, the transport per connection (`NVL`, `PCI`, `IB`, `NET/IB`,
`NET/Socket`, etc.), channel count, and the chosen algo/protocol. **90% of "comms are slow" bugs are
visible here**: a TP group on `NET/Socket` instead of `NVL`, GPUDirect RDMA not engaged, too few
channels, or a NIC missing from the graph.

### Tunables — conceptually (names *(verify)* against current NCCL docs)

NCCL is mostly self-tuning; reach for knobs only after the `NCCL_DEBUG` log shows a wrong choice.
Categories you will encounter (exact names/semantics change across versions — **verify**):

- **Interface/transport selection** — restrict or pin which NICs/HCAs NCCL uses
  (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` *(verify)*); enable/confirm GPUDirect RDMA
  (`NCCL_NET_GDR_LEVEL` / `NCCL_IB_GDR_LEVEL` *(verify)*).
- **Algorithm/protocol override** — force Ring vs Tree, or a protocol (`NCCL_ALGO`, `NCCL_PROTO`
  *(verify)*). Almost always a diagnostic, not a permanent setting.
- **Parallelism** — channel count, buffer sizes, chunk sizes (`NCCL_MIN/MAX_NCHANNELS`,
  `NCCL_BUFFSIZE` *(verify)*).
- **RDMA/IB specifics** — GID index, traffic class, service level, timeout/retry, QPs per connection,
  adaptive routing (`NCCL_IB_GID_INDEX`, `NCCL_IB_TC`, `NCCL_IB_SL`, `NCCL_IB_QPS_PER_CONNECTION`,
  `NCCL_IB_TIMEOUT` *(verify)*) — these are what you tune for RoCE/IB fabrics.
- **SHARP / collnet** — enable in-network reduction where the fabric supports it (`NCCL_COLLNET_ENABLE`
  / SHARP plugin env *(verify)*).
- **Cloud transports** — GPUDirect-TCPX/TCPXO use platform-specific env and a `libnccl-net` plugin;
  **follow the cloud provider's exact, current instructions** rather than guessing.

**Rule:** change one knob at a time, re-run the nccl-test, keep only what measurably helps, and
document why. A pile of copy-pasted NCCL_* exports is a classic source of silent regressions.

### Common NCCL failure modes

- **Hang at init or first collective** — usually a connectivity/topology mismatch: a NIC NCCL can't
  reach, a firewall/IFNAME picking the wrong interface, mismatched `world_size`/ranks, or one rank
  that never joined. `NCCL_DEBUG=INFO` + per-rank logs localize it.
- **Hang mid-run** — a straggler rank stuck in compute (so others block in the collective barrier), a
  link flap, or one process that died and left the others waiting. Look for the rank *not* in the
  collective.
- **Slow but correct** — wrong transport (Socket where RDMA was expected), too few channels, tree
  where ring was wanted (or vice versa), NUMA/PCIe misaffinity, or a single slow NIC/link dragging the
  ring.
- **Mismatched topology** — ranks placed such that a high-frequency group (TP) crosses the slow fabric.
  Fix with placement (§5), not flags.

---

## 4. Interconnects & transports

The physical and transport layers the collectives ride. Bandwidth/latency numbers are
generation-specific — **always verify against the spec for your exact hardware**; this section gives the
*relationships and roles*, not figures.

### Intra-node: NVLink / NVSwitch (and PCIe)

- **NVLink** — direct high-bandwidth GPU↔GPU links, far faster than PCIe. **NVSwitch** is the on-board
  crossbar that gives all-to-all NVLink bandwidth among the GPUs in a node (and, with NVLink switch
  systems, across a small multi-node domain). This is the fastest domain — **put TP and MoE all-to-all
  here.**
- **PCIe** — fallback GPU↔GPU and the GPU↔NIC path. Much lower bandwidth; crossing the PCIe root
  complex or the wrong NUMA node costs you. Check with `nvidia-smi topo -m`.
- **NUMA** — pin each process to the CPU/memory node nearest its GPU and NIC. Misaffinity shows up as
  reduced effective bandwidth and jitter.

### Inter-node: InfiniBand vs RoCEv2

Both carry RDMA between nodes; the choice shapes how much fabric tuning you do.

| | **InfiniBand (IB)** | **RoCEv2 (RDMA over Converged Ethernet)** |
|---|---|---|
| Fabric | Dedicated IB switches, credit-based lossless by design | Ethernet; lossless must be *engineered* |
| Congestion/flow ctrl | Built-in | **PFC + ECN/DCQCN must be configured end-to-end** |
| Routing | Subnet manager; adaptive routing options | Ethernet ECMP; needs careful hashing for rails |
| Tuning burden | Lower | **Higher — the #1 RoCE foot-gun** |
| Ecosystem | Mature for HPC/AI | Common on cloud/commodity Ethernet |

**RoCEv2 reality:** RDMA assumes a near-lossless fabric. On Ethernet that means **Priority Flow Control
(PFC)** to stop drops and **ECN (with DCQCN or equivalent)** for congestion control, tuned end-to-end
across NICs and switches. Misconfigured, you get PFC pause storms, head-of-line blocking, drops, and
all-reduces that are intermittently slow. IB gives you this lossless behavior with much less hand-
tuning, which is why many large AI fabrics use IB — but RoCE at scale is well-proven *when properly
configured* (Meta's LLM-training infra writeup discusses RoCE design choices at scale).

### GPUDirect RDMA, TCPX/TCPXO

- **GPUDirect RDMA** — the NIC DMAs directly to/from GPU memory, bypassing a bounce through host
  memory. Essential for high inter-node collective bandwidth and low latency. Confirm it's actually
  engaged in the `NCCL_DEBUG` log (GDR level / `[GDRDMA]`), not silently disabled by topology/affinity.
- **GPUDirect-TCPX / TCPXO** — cloud GPU-direct datapaths **over TCP/Ethernet** for environments
  without RDMA NICs; a `libnccl-net` plugin gives NCCL a GPU-direct path on standard Ethernet VMs.
  Setup (plugin, NIC/queue config, env) is platform-specific — **follow the provider's current docs
  exactly; do not guess the env.**

### SHARP — in-network reduction

**SHARP** offloads the reduction (sum) into the switch ASIC: GPUs send their data once, the switch
tree reduces it, and the result is sent back — removing a full data traversal from AllReduce/Reduce on
the GPUs. Big win for DP gradient all-reduce on supported IB fabrics. Enabled via the collnet/SHARP
path (*verify* the current env and that your fabric/switches support it).

### TPU: ICI vs DCN

On TPU the picture is analogous but distinct (see [[ml-frameworks]] / [[maxtext-jax-llm]]):

- **ICI (Inter-Chip Interconnect)** — dedicated high-bandwidth links wiring chips into a Pod in a
  torus/mesh; the fast domain, akin to NVLink's role. XLA SPMD lowers collectives onto ICI.
- **DCN (Data Center Network)** — standard datacenter networking between Pods/slices; the slow domain,
  akin to inter-node IB/RoCE. Multi-slice/multi-Pod training crosses DCN.
- Same principle: keep the heaviest collectives on ICI; let the slowest, least-frequent axis cross DCN.

---

## 5. Topology awareness & placement

The single highest-leverage thing after picking the right collective: **place ranks so the chattiest
collectives stay in the fastest domain and cross the fewest expensive hops.**

### Fabric shapes

- **Fat-tree (Clos)** — leaf (ToR) switches under spine switches. Same-leaf traffic is one hop; cross-
  leaf goes up to a spine and back down. **Oversubscription** at the spine (more downlink than uplink
  bandwidth) means cross-leaf collectives contend — a comms tax on every step that crosses it.
- **Rail-optimized** — each GPU in a node connects via its own NIC to a dedicated "rail" of the fabric;
  GPU *i* on every node attaches to rail *i*. Same-rail traffic stays leaf-local and parallel across
  rails. **Placement must respect rails**: a collective that keeps each rank on its own rail runs at
  full parallel bandwidth; one that forces cross-rail hops congests and serializes.

### Placement principles

- **Contain the high-frequency groups.** TP group → one node (NVLink). MoE expert group → minimize
  cross-node all-to-all spread. Pipeline stages → adjacent in the fabric so P2P is cheap. DP/FSDP → the
  axis you let span nodes, because it's the most overlap-friendly.
- **Gang-schedule within a high-bandwidth block.** All ranks of a job should land in the same
  IB island / rail-aligned block, not scattered across the spine. On K8s this is compact placement /
  topology-aware scheduling — see [[jobset-leaderworkerset]], [[gke-master]],
  [[kubernetes-internals-expert]]. Give NCCL/XLA the placement that matches the fabric's hierarchy.
- **Align ranks to rails.** Ensure rank↔NIC↔rail mapping is consistent so same-index ranks share a
  rail; mismatched mapping silently forces cross-rail congestion.
- **Mind oversubscription & congestion.** Know your spine oversubscription ratio. Adaptive routing
  (IB) / good ECMP hashing (Ethernet) spreads flows; without it, multiple collectives can hash onto the
  same link and serialize.

### Comm/compute overlap

Even perfectly placed, an exposed collective on the critical path wastes the GPU. Overlap is what
converts comm cost from additive to (partially) free:

- **DDP** — gradients are bucketed and each bucket's all-reduce launches as soon as its grads are ready
  during backward, overlapping with the rest of backprop.
- **FSDP/ZeRO-3** — **prefetch** the next layer's parameter all-gather while computing the current
  layer; reduce-scatter grads while continuing backward. If prefetch depth is too shallow, the
  all-gather is exposed.
- **TP** — async/overlapped TP (and sequence parallel) hide the all-reduce/RS+AG behind the GEMMs.
- **Pipeline** — schedule (1F1B / interleaved) hides send/recv behind other micro-batches' compute,
  shrinking the bubble.
- **Diagnostic:** if a timeline shows GPU idle (no kernels) during a NCCL kernel, the collective is
  *exposed* — fix overlap (bucket/prefetch settings) or reduce the collective (bigger messages, better
  placement, SHARP).

---

## 6. Diagnosing comms bottlenecks

A repeatable method: **measure the fabric in isolation, then measure it inside the job, then attribute.**

### Step 1 — nccl-tests (fabric in isolation)

NVIDIA's **nccl-tests** (`all_reduce_perf`, `all_gather_perf`, `alltoall_perf`, …) measure raw
collective bandwidth/latency across a size sweep, independent of your training code. The key metric is
**busbw (bus bandwidth)** — the effective per-link bandwidth the collective achieves, which you compare
to the fabric's theoretical ceiling for that hardware.

- **Low busbw at all sizes** → wrong transport/topology (check `NCCL_DEBUG`: Socket vs IB, GDR
  off, too few channels), or NUMA/PCIe misaffinity.
- **busbw climbs with size then plateaus below ceiling** → algo/channel limited, or fabric oversub.
- **Cliff exactly at the node boundary** (intra-node fine, multi-node poor) → the inter-node fabric:
  IB/RoCE config, GDR, or RoCE PFC/ECN.
- Run it on the *same placement* your job will use, so the test exercises the same rails/hops.

See `examples.md` for an `all_reduce_perf` invocation and how to read busbw.

### Step 2 — inside the job (where is the time going?)

- **PyTorch profiler / Kineto + NVTX** and **Nsight Systems (nsys)** timelines: see whether NCCL
  kernels overlap compute or sit exposed; measure per-collective duration.
- **`torch.distributed` flight recorder / NCCL trace** *(verify current name)* — dumps in-flight
  collectives when a hang occurs; invaluable for "which rank is stuck."
- **Per-rank step time / kernel time** — collect across all ranks and look at the *distribution*, not
  the mean. The slowest rank is the one to chase (collectives are barriers).

### Step 3 — straggler & slow-link hunting

A single slow rank/link stalls the whole collective. Look for:

- A rank with consistently higher compute time (thermal throttling, ECC errors, a noisy neighbor).
- A NIC/link with errors or reduced speed (`ibstat`/`ibdiagnet` on IB; switch counters; link flaps in
  `dmesg`). A cable negotiating a lower rate, or one congested spine link, taxes every step.
- GPU clock throttling (`nvidia-smi -q -d CLOCK,PERFORMANCE`), ECC error counts, Xid errors in
  `dmesg`/syslog.

### Step 4 — MFU / goodput attribution

Compute achieved MFU/goodput and attribute the gap: `ideal_step_time` (pure compute) vs `actual`. The
difference, minus data-loading and bubble, is exposed communication. If exposed comm dominates, the fix
is one of: better placement (§5), bigger/fused messages, more overlap, RDMA/SHARP, or fixing a
straggler. Re-run nccl-tests after each change to confirm the *fabric* improved, not just luck.

---

## 7. Anti-patterns (the traps that halve your MFU)

- **Topology-blind placement.** Letting the scheduler scatter ranks across the spine, or putting a TP
  group across nodes over Ethernet. The fastest fix in this whole guide is usually *placement*.
- **Untuned NCCL by superstition.** Either never reading `NCCL_DEBUG` (so you don't know it picked
  Socket/no-GDR), or pasting a wall of NCCL_* exports cargo-culted from a blog that mismatch your
  fabric. Read the log; change one knob at a time; verify with nccl-tests.
- **RoCE without proper PFC/ECN.** Deploying RoCEv2 on a fabric that isn't actually lossless. You get
  intermittent slow all-reduces and pause storms that look like random "bad days." Configure DCQCN/PFC
  end-to-end or use IB.
- **Ignoring NUMA/PCIe.** GPU↔NIC across the wrong NUMA node / PCIe root complex; processes unpinned.
  Silent bandwidth loss and jitter.
- **Comm not overlapped with compute.** Exposed all-gather/all-reduce/all-to-all on the critical path
  (no prefetch, no bucketing, bad pipeline schedule). The GPU sits idle on the network.
- **Tiny messages.** Many small collectives instead of fused/bucketed large ones → stuck in the
  latency-bound regime, never reaching bandwidth efficiency. Fuse gradients, size buckets sensibly.
- **GDR/SHARP silently off.** Assuming GPUDirect RDMA or SHARP is active because the hardware supports
  it. Confirm in the NCCL log / counters; affinity or config often disables them.
- **Averaging away the straggler.** Tuning to mean step time while one rank/link drags the barrier.
  Collectives are as slow as the slowest participant.
- **Spanning the wrong axis across the slow fabric.** Crossing DCN (TPU) or inter-node (GPU) with TP/MoE
  instead of DP/FSDP. Cross the slow domain only with the most overlap-tolerant, least-frequent axis.

---

## 8. Troubleshooting — symptom → diagnosis → fix

| Symptom | Likely cause | Fix |
|---|---|---|
| Job hangs at init / first collective | Wrong `NCCL_SOCKET_IFNAME`/HCA, unreachable NIC, rank/world_size mismatch, firewall | `NCCL_DEBUG=INFO`; pin the right interface *(verify flag)*; confirm all ranks join |
| Hang mid-run, no progress | Straggler stuck in compute, dead process, link flap | Flight recorder / NCCL trace to find the rank not in the collective; check `dmesg` Xid, NIC link |
| Correct but slow; low busbw everywhere | Socket transport / GDR off / too few channels / NUMA misaffinity | Read `NCCL_DEBUG` transport lines; enable GDR; pin NUMA; raise channels *(verify)*; `nvidia-smi topo -m` |
| Intra-node fast, multi-node slow | Inter-node fabric: IB/RoCE config, GDR, RoCE PFC/ECN | nccl-test multi-node; IB diagnostics; verify DCQCN/PFC; confirm IB/RoCE transport in log |
| Intermittent slow all-reduce on RoCE | PFC pause storms / ECN not tuned / oversubscribed spine | Engineer lossless fabric end-to-end; check pause/ECN counters; adaptive routing/ECMP hashing |
| One rank always slower → whole step slow | Thermal throttle, ECC, bad cable, congested link | Per-rank kernel time distribution; `nvidia-smi` clocks/ECC; `ibdiagnet`; replace/reroute the link |
| MoE step dominated by comm | All-to-all spread across nodes / load imbalance | Contain expert group in fast domain; balance tokens; overlap dispatch/combine |
| GPU idle during NCCL kernel (timeline) | Exposed collective, no overlap | Increase FSDP prefetch depth / DDP bucket overlap; better pipeline schedule; fuse messages |
| busbw far below hardware spec | Wrong algo/proto for size, channel limit, oversub | Compare to spec; let NCCL auto-tune; size-sweep; check spine oversubscription ratio |

---

## 9. Version awareness

It is 2026 and this stack moves quickly. Concretely:

- **NCCL/RCCL releases** change algorithms, defaults, protocols, and env-var names/semantics between
  versions. Pin the version, read *that version's* docs, and re-benchmark when you upgrade.
- **NCCL_* knobs** named here are *(verify)* — confirm the exact name, default, and meaning against the
  current NCCL documentation before relying on them.
- **Cloud transports (GPUDirect-TCPX/TCPXO)** and their `libnccl-net` plugins evolve; follow the
  provider's current setup exactly.
- **Hardware bandwidths** (NVLink/NVSwitch/IB/RoCE/ICI generations) are per-generation — never quote a
  number from memory; check the spec sheet for your exact SKU.
- **SHARP / collnet** support and enablement depend on fabric, switch firmware, and library version.

When in doubt, measure (nccl-tests) rather than trust a remembered number.

---

## 10. Canonical references (verify current)

- **arXiv 2407.20018** — *"Understanding the network needs of LLM training"* (network requirements /
  collective traffic survey for LLM training). `https://arxiv.org/abs/2407.20018`
- **Meta Engineering (2024)** — *"How we built large-scale infrastructure to train LLMs"* / RoCE-vs-IP
  fabric design. `https://engineering.fb.com/2024/06/12/`
- **NVIDIA Developer Blog** — *"Scaling Deep Learning Training with NCCL."*
  `https://developer.nvidia.com/blog/scaling-deep-learning-training-nccl/`
- **NCCL** — docs, source, and env-var reference. `https://docs.nvidia.com/deeplearning/nccl/` ·
  `https://github.com/NVIDIA/nccl`
- **nccl-tests** — `https://github.com/NVIDIA/nccl-tests` (busbw definition in its README).
- **RCCL** (ROCm) — `https://github.com/ROCm/rccl`
- **PyTorch distributed** — `https://pytorch.org/docs/stable/distributed.html`
- **GPUDirect RDMA** — `https://docs.nvidia.com/cuda/gpudirect-rdma/`
- **NVIDIA SHARP** — in-network computing docs (verify current URL under NVIDIA networking docs).
- For parallelism strategy that emits these collectives: [[training-frameworks]]; framework/runtime
  (PyTorch/JAX/XLA, ICI/DCN): [[ml-frameworks]]; MPI/RDMA on Slurm/HPC: [[slurm-hpc-on-kubernetes]];
  cloud fabric provisioning & placement: [[gke-master]], [[jobset-leaderworkerset]],
  [[kubernetes-internals-expert]], [[aiml-on-kubernetes]].
