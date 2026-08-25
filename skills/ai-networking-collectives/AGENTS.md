# AGENTS.md — AI Networking & Collective Communication

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`ai-networking-collectives-guide.md`** next to this file
> — read it before configuring, tuning, or debugging the comm stack, and apply it. Concrete artifacts
> to imitate (nccl-tests diagnostic, topology-aware placement note, triage checklist) are in
> **`examples.md`**. This file is the always-on summary.
>
> **Premise:** communication overhead is a dominant bottleneck in large distributed training. A
> topology-blind placement or untuned collective stack quietly turns ~90% MFU into ~50%. Parallelism
> *strategy* is [[training-frameworks]]; K8s/GKE fabric plumbing is [[gke-master]] /
> [[kubernetes-internals-expert]]. **Never invent an NCCL_* flag, bandwidth number, or benchmark —
> the stack moves fast; verify against current docs.**

## When working on the comm layer of distributed training/inference, apply by default:

- **Map each collective to the right interconnect domain.** DP grad sync = AllReduce (or
  ReduceScatter+AllGather); FSDP/ZeRO-3 = AllGather params + ReduceScatter grads; TP = AllReduce
  (RS+AG with sequence parallel); MoE = AllToAll; pipeline = P2P send/recv. Put the high-frequency
  ones (TP, MoE) on **NVLink/NVSwitch**; let DP/FSDP cross nodes over IB/RoCE. AllReduce =
  ReduceScatter + AllGather.
- **Algorithm by message size.** Ring = bandwidth-optimal (large messages); tree = latency-optimal
  (small messages / many ranks); hierarchical/SHARP for large multi-node. NCCL auto-selects — verify
  its choice; make messages big (fuse/bucket) to reach the bandwidth-efficient regime.
- **Read `NCCL_DEBUG=INFO` before touching any knob.** `NCCL_DEBUG_SUBSYS=INIT,GRAPH` prints the
  rings/trees and transport per connection (`NVL`/`PCI`/`IB`/`NET/Socket`). Most "comms slow" bugs are
  visible here: Socket instead of NVL, GDR off, too few channels, NIC missing from the graph.
- **NCCL is mostly self-tuning.** Change one knob at a time, only after the log shows a wrong choice,
  and re-verify with nccl-tests. A wall of cargo-culted NCCL_* exports is a regression source. Treat
  flag names as *(verify)* against the current NCCL docs.
- **Confirm GPUDirect RDMA / SHARP are actually engaged** — don't assume because the hardware supports
  them. On cloud Ethernet without RDMA NICs, GPUDirect-TCPX/TCPXO give a GPU-direct path; follow the
  provider's exact current setup.
- **RoCEv2 requires a lossless fabric:** PFC + ECN/DCQCN tuned end-to-end. Unconfigured → pause storms,
  drops, intermittently slow all-reduces. IB gives this with less hand-tuning. This is the top RoCE
  foot-gun.
- **Topology-aware placement is mandatory.** Rail-optimized fat-trees: keep same-rail ranks leaf-local,
  gang-schedule the job within one high-bandwidth block, minimize cross-spine hops, know your
  oversubscription ratio. On TPU: heavy collectives on ICI, slow axis on DCN.
- **Overlap comm with compute or pay full price.** FSDP prefetch, DDP gradient bucketing, async TP,
  1F1B/interleaved pipeline. GPU idle during a NCCL kernel on the timeline = exposed collective → fix
  overlap or reduce the collective.
- **Mind NUMA/PCIe.** Pin processes; match GPU↔NIC affinity; check `nvidia-smi topo -m`. Wrong NUMA
  node / PCIe root complex = silent bandwidth loss and jitter.
- **Diagnose with nccl-tests first.** `all_reduce_perf` busbw vs the fabric ceiling. Low everywhere =
  transport/topology/affinity; cliff at the node boundary = inter-node fabric. Then Nsight/profiler
  timelines and per-rank kernel-time *distribution*.
- **One slow rank stalls the whole collective** — it's a barrier. Hunt the outlier (thermal throttle,
  ECC/Xid, bad cable, congested link); don't average it away.

## Definition of done for comm-stack work
- `NCCL_DEBUG=INFO` log inspected; transports are what you expect (NVL/IB, GDR on where intended).
- nccl-tests busbw measured on the real placement and within range of the hardware ceiling.
- High-frequency collectives confined to the fast domain; job gang-placed in one high-bandwidth block.
- Comm overlapped with compute (no exposed collectives on the critical path in the timeline).
- For RoCE: lossless fabric (PFC/ECN) verified. No fabricated flags/numbers; fast-moving items verified
  against current docs.

## Triaging a comms bottleneck
Use the triage checklist at the end of `examples.md`.
