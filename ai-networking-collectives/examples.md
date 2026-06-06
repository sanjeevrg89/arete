# AI Networking & Collectives — Worked Examples

Concrete artifacts to imitate: a collective-bandwidth diagnostic, a topology-aware placement note, and
a comms-bottleneck triage checklist. Commands are correct in spirit; **flag names and bandwidth numbers
are fast-moving — verify against current NCCL/nccl-tests and your hardware docs before relying on them.**

---

## 1. All-reduce / NCCL-test diagnostic

Measure the *fabric in isolation* with **nccl-tests** before blaming your training code. The headline
metric is **busbw (bus bandwidth)** — compare it to the theoretical ceiling for your specific GPUs/NICs.

### Build and run a size sweep

```bash
# Build (once) — needs CUDA + an MPI if you launch multi-node via mpirun
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi   # adjust to your MPI path

# Single node, 8 GPUs: all-reduce from 8 MB up to 8 GB, doubling each step, 50 iters
./build/all_reduce_perf -b 8M -e 8G -f 2 -g 8

# Multi-node (e.g. 16 nodes x 8 GPUs = 128 ranks) via mpirun — one rank per GPU.
# Run it on the SAME placement your training job will use, so it exercises the same rails/hops.
mpirun --hostfile hosts -np 128 --map-by ppr:8:node \
  -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  ./build/all_reduce_perf -b 64M -e 8G -f 2 -g 1
```

### Read the output

`all_reduce_perf` prints per size: `time`, **`algbw`** (algorithm bandwidth = size/time) and
**`busbw`** (bus bandwidth = algbw scaled by the 2(N−1)/N all-reduce factor, i.e. effective per-link
bandwidth). **busbw is the number to judge** — it should approach the fabric's ceiling for large
messages.

```
#   size      count    type   redop    time   algbw   busbw  #wrong
# (bytes)   (elem)                     (us)  (GB/s)  (GB/s)
 67108864  16777216   float     sum    ...     ...     XX.X      0
   ...
8589934592 2147483648 float     sum    ...     ...     YY.Y      0
```

Interpretation:
- **busbw low at every size** → wrong transport/topology or affinity. Check the `NCCL_DEBUG` log for
  `NET/Socket` where you expected `NVL`/`IB`, GPUDirect RDMA not engaged, or too few channels; check
  `nvidia-smi topo -m` for NUMA/PCIe misaffinity.
- **busbw rises with size then plateaus below ceiling** → channel/algo limited or fabric oversubscribed.
- **Intra-node fine, multi-node poor (cliff at the node boundary)** → the inter-node fabric: IB/RoCE
  config, GDR, or RoCE PFC/ECN. Re-run the multi-node test and read the `IB`/`NET` lines.
- **`#wrong` non-zero** → data corruption; stop and fix the fabric/driver before anything else.

### Confirm what NCCL actually chose

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV   # rings/trees, per-connection transport, env overrides
```

In the log, verify: TP/intra-node connections show **`NVL`** (NVLink), inter-node shows **`NET/IB`** (or
RoCE) — **not** `NET/Socket` when you have RDMA NICs; GPUDirect RDMA is engaged (GDR level / `[GDRDMA]`
lines); channel count is plural; every NIC you expect is present in the topology. *These are diagnostic
reads, not permanent settings.*

---

## 2. Topology-aware placement note

A short design note of the kind you'd attach to a multi-host training job. Goal: keep the chattiest
collectives in the fastest domain. (Provisioning the node pool / compact placement is [[gke-master]];
gang placement is [[jobset-leaderworkerset]]; the parallelism degrees are [[training-frameworks]].)

```
Job: 512 GPUs = 64 nodes x 8 GPUs.  Parallelism: TP=8, PP=8, DP=8 (512 = 8*8*8).
Fabric: rail-optimized fat-tree, 8 rails (one NIC per GPU per node), IB inter-node with GPUDirect RDMA.

Mapping (chattiest collective -> fastest domain):
  TP=8  -> WITHIN a node.  TP emits an AllReduce twice per transformer layer = highest frequency,
           latency-critical.  Must ride NVLink/NVSwitch.  Never let a TP group span nodes.
  PP=8  -> adjacent nodes in the fabric.  PP is P2P send/recv between stages -> cheap, but keep
           neighboring stages leaf-local so the bubble-hiding schedule isn't fighting the network.
  DP=8  -> the axis that spans the fabric.  DP/FSDP grad collectives are the most overlap-friendly
           (bucketed all-reduce / reduce-scatter overlapping backward), so this is the one we let
           cross leaves/spine over IB.

Placement requirements:
  - Gang-schedule all 64 nodes into ONE high-bandwidth IB block (avoid scattering across the spine).
  - Align rank<->NIC<->rail so same-index ranks share a rail (same-rail traffic stays leaf-local and
    parallel across rails).  A mismatched mapping silently forces cross-rail congestion.
  - Pin each process to the NUMA node nearest its GPU and NIC (check `nvidia-smi topo -m`).
  - Confirm GPUDirect RDMA engaged in the NCCL_DEBUG log; enable SHARP for the DP all-reduce if the
    IB fabric/switches support it (verify current enablement).

Validate BEFORE the real run:
  - nccl-tests all_reduce_perf on this exact placement; busbw within range of the IB ceiling.
  - NCCL_DEBUG=INFO: TP connections show NVL, DP/inter-node show NET/IB (not NET/Socket), GDR on.
```

For TPU the same note reduces to: heavy collectives on **ICI** (within a slice/Pod), let the slowest,
least-frequent axis cross **DCN** (multi-slice). See [[ml-frameworks]] / [[maxtext-jax-llm]].

---

## 3. Comms-bottleneck triage checklist

Work top to bottom. Stop when busbw is healthy *and* the timeline shows comm overlapped with compute.

**A. Is it actually comms?**
- [ ] Compute achieved MFU/goodput; compare `ideal_step_time` (pure compute) to `actual`.
- [ ] Profiler/Nsight timeline: is the GPU idle (no kernels) during NCCL kernels? Idle = exposed comm.
- [ ] Rule out data loading and pipeline bubble as the gap before blaming the network.

**B. Measure the fabric in isolation**
- [ ] nccl-tests `all_reduce_perf` (and `alltoall_perf` for MoE) on the real placement, size sweep.
- [ ] Compare **busbw** to the hardware ceiling. Low everywhere vs cliff-at-node-boundary tells you
      intra- vs inter-node.

**C. What did NCCL choose? (`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH`)**
- [ ] TP/intra-node = `NVL`? Inter-node = `NET/IB` (or RoCE), not `NET/Socket`?
- [ ] GPUDirect RDMA engaged (GDR level / `[GDRDMA]`)? SHARP/collnet active where supported?
- [ ] Channel count plural? All expected NICs present in the topology graph?

**D. Placement & topology**
- [ ] High-frequency collectives (TP, MoE all-to-all) contained in the fast domain?
- [ ] Job gang-placed in one high-bandwidth block, not scattered across the spine?
- [ ] rank↔NIC↔rail aligned? Spine oversubscription ratio known? Adaptive routing / ECMP hashing ok?
- [ ] `nvidia-smi topo -m`: GPU↔NIC on the same NUMA node / PCIe root complex? Processes pinned?

**E. RoCE-specific (if Ethernet/RoCEv2)**
- [ ] PFC configured and ECN/DCQCN tuned end-to-end (NICs + switches)?
- [ ] Pause/ECN counters clean (no pause storms)? Drops zero?

**F. Straggler / slow link (collectives are barriers — slowest rank wins)**
- [ ] Per-rank kernel-time *distribution* (not mean) — any consistent outlier rank?
- [ ] `nvidia-smi -q -d CLOCK,PERFORMANCE` (thermal throttle), ECC counts, Xid errors in `dmesg`.
- [ ] IB health (`ibstat` / `ibdiagnet`), switch counters, link flaps, a cable negotiated low.

**G. Overlap & message size**
- [ ] FSDP prefetch depth / DDP bucket size so all-gather/all-reduce overlap backward?
- [ ] Pipeline schedule (1F1B / interleaved) hiding send/recv?
- [ ] Messages fused/bucketed large enough to leave the latency-bound regime?

**H. Re-validate**
- [ ] Re-run nccl-tests after each change — confirm the *fabric* improved, one variable at a time.
- [ ] Re-check the timeline: collectives now overlapped, GPU not idle on the network.
