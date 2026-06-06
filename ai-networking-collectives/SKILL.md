---
name: ai-networking-collectives
description: Expert knowledge of networking and collective communication for AI training/inference at
  scale — the competency that decides whether a 10,000-GPU job runs at 50% or 90% MFU. Use when
  configuring, tuning, or debugging the communication stack underneath distributed training/inference:
  collective operations (all-reduce, all-gather, reduce-scatter, broadcast, all-to-all, point-to-point)
  and where each appears (DP grad sync, FSDP shard/gather, TP, MoE all-to-all, pipeline send/recv);
  NCCL/RCCL algorithm and channel selection, NCCL_* tunables and NCCL_DEBUG; interconnects and
  transports (NVLink/NVSwitch, InfiniBand vs RoCEv2, GPUDirect RDMA, GPUDirect-TCPX/TCPXO, SHARP
  in-network reduction, TPU ICI vs DCN, PCIe/NUMA effects); topology-aware placement on rail-optimized
  fat-tree fabrics; comm/compute overlap; and diagnosing comms bottlenecks (nccl-tests all-reduce bench,
  Nsight, timeline traces, straggler/slow-link detection, MFU/goodput attribution). Triggers: NCCL hang
  or slow rank, low bus bandwidth, RoCE/PFC/ECN tuning, "training is comms-bound", topology-blind
  placement, NCCL_DEBUG=INFO output, choosing IB vs Ethernet. Parallelism strategy itself is
  [[training-frameworks]]; K8s/GKE fabric plumbing is [[gke-master]]/[[kubernetes-internals-expert]].
---

# AI Networking & Collective Communication

Apply the judgment of an engineer who has brought up and tuned the collective stack on
tens-of-thousands-of-accelerator clusters for years: **communication overhead is a dominant bottleneck
in large distributed training**, so map the chattiest collectives onto the fastest interconnect domain,
keep them overlapped with compute, and treat a tuned, topology-aware comm stack as a prerequisite for
high MFU — not an afterthought.

## How to use this skill

1. **Read `ai-networking-collectives-guide.md`** in this directory — the full reference (collective
   operations and where they appear, ring/tree/hierarchical algorithms, NCCL/RCCL internals and
   tunables, interconnects/transports, topology awareness, comm/compute overlap, diagnosis,
   anti-patterns, troubleshooting). Apply it to the task.
2. For concrete artifacts to imitate — an `all_reduce_perf` / nccl-tests diagnostic run, a
   topology-aware placement note, and a comms-bottleneck triage checklist — read **`examples.md`**.
3. Match the cluster's existing fabric (IB vs RoCE vs TCPX/ICI), NCCL version, and placement scheme;
   apply the correctness rules (keep collectives in the high-bandwidth domain, overlap comm with
   compute, validate the topology NCCL actually chose) regardless. **Never invent an NCCL_* flag,
   benchmark number, or hardware bandwidth figure — the stack moves fast; verify against current docs.**

## Essentials (full detail in `ai-networking-collectives-guide.md`)

- **Comms can dominate at scale.** At frontier scale, a topology-blind placement or an untuned
  collective stack silently halves goodput. The fix is structural (placement + algorithm + overlap),
  not a single magic flag.
- **Know which collective each parallelism axis emits.** DP grad sync = all-reduce (or
  reduce-scatter + all-gather); FSDP/ZeRO-3 = all-gather (params, fwd+bwd) + reduce-scatter (grads);
  TP = all-reduce (or reduce-scatter/all-gather with sequence parallel); MoE = all-to-all (dispatch +
  combine); pipeline = point-to-point send/recv. Tune the one that's on your critical path.
- **Match the collective to the interconnect.** Put the highest-volume, most-frequent collectives
  (TP all-reduce, MoE all-to-all) inside the NVLink/NVSwitch domain (intra-node); let DP/FSDP cross
  nodes over IB/RoCE. A TP group spanning nodes over Ethernet is a classic self-inflicted bottleneck.
- **Ring is bandwidth-optimal, tree is latency-optimal.** Large messages → ring/hierarchical
  (bandwidth-bound, ~2(N−1)/N of data moved); small messages/many ranks → tree (log-depth latency).
  NCCL auto-selects per size/topology; you mostly verify its choice rather than override it.
- **NCCL/RCCL is the engine.** It discovers topology, builds rings/trees across NVLink/PCIe/IB/RoCE,
  splits each collective across multiple **channels** for parallelism, and picks algo+protocol
  (LL/LL128/Simple) by message size. `NCCL_DEBUG=INFO` (plus `NCCL_DEBUG_SUBSYS=INIT,GRAPH`) prints the
  rings/trees and transports it chose — read it before touching any knob.
- **GPUDirect RDMA / SHARP are the big wins.** GPUDirect RDMA moves data NIC↔GPU without staging in
  host memory; SHARP does the reduction *in the switch* (offloads all-reduce from GPUs). On cloud
  Ethernet without RDMA NICs, GPUDirect-TCPX/TCPXO provide a GPU-direct datapath over TCP — verify the
  exact plugin/env for your platform.
- **RoCEv2 needs a lossless fabric.** RoCE over Ethernet requires PFC and ECN/DCQCN tuned end-to-end;
  unconfigured, you get pause storms, drops, and mysterious slow all-reduces. IB gives this with less
  hand-tuning. This is the #1 RoCE foot-gun.
- **Topology-aware placement is mandatory.** On rail-optimized fat-trees, place ranks so same-rail GPUs
  talk leaf-local; keep gang-scheduled jobs within a high-bandwidth block; minimize cross-spine hops.
  Oversubscription and congestion at the spine show up as a comms tax on every step.
- **Overlap comm with compute or pay full price.** FSDP prefetch, bucketed gradient all-reduce
  overlapping backward, async TP, pipeline that hides send/recv behind compute — if the collective is
  on the critical path (GPU idle waiting on the network), you've lost the overlap.
- **Mind NUMA/PCIe.** A GPU talking to a NIC on the wrong NUMA node or across the PCIe root complex
  loses bandwidth and adds latency. Pin processes, match GPU↔NIC affinity, and check `nvidia-smi topo
  -m` / the NCCL graph.
- **Diagnose with nccl-tests first.** `all_reduce_perf` reports **busbw** (bus bandwidth) — compare it
  to the fabric's expected ceiling. Low busbw at size = bad algo/topology/transport; a cliff at node
  boundary = the inter-node fabric. Then nvtx/Nsight/PyTorch-profiler timelines to find stragglers.
- **One slow rank stalls the whole collective.** Collectives are barriers; a single straggler GPU,
  hot NIC, throttling link, or bad cable drags every rank. Hunt the outlier (per-rank kernel time,
  ECC errors, link flaps), don't average it away.

## Related skills

- `[[training-frameworks]]` — choosing DP/FSDP/TP/PP/EP degrees and the parallelism that *emits* these
  collectives; this skill is the comm layer beneath it.
- `[[ml-frameworks]]` — PyTorch/JAX/XLA the collectives run under (process groups, XLA SPMD, ICI).
- `[[slurm-hpc-on-kubernetes]]` — MPI/RDMA on Slurm/Volcano, the HPC-side launcher and fabric.
- `[[gke-master]]` — provisioning RDMA/TCPX(O) node pools, compact placement, the cloud fabric.
- `[[jobset-leaderworkerset]]` — gang/topology-aware placement of multi-host workers.
- `[[aiml-on-kubernetes]]` — umbrella for running training/inference on K8s/GKE.
- `[[kubernetes-internals-expert]]` — scheduler/topology plumbing under topology-aware placement.
