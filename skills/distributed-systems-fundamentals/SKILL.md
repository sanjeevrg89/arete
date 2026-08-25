---
name: distributed-systems-fundamentals
description: The timeless distributed-systems fundamentals every architect must reason from — the
  impossibility/tradeoff results (CAP, PACELC, FLP, the fallacies of distributed computing), consensus
  (Paxos, Raft leader election & log replication, ZAB/Viewstamped Replication, quorums), replication &
  consistency models (single/multi/leaderless, linearizability → causal → eventual, read-your-writes,
  R+W>N quorums, LWW/vector clocks/CRDTs), partitioning/consistent hashing, caching & stampede, queues &
  the log (at-least/exactly-once, idempotency, outbox), time & logical/vector/hybrid clocks, cross-node
  transactions (2PC, sagas, isolation levels, the dual-write problem), and failure/reliability (failure
  detectors, retries/backoff/jitter, fencing tokens, backpressure). Use when designing or reviewing any
  replicated, sharded, multi-region, or fault-tolerant system; choosing a consistency model; doing
  leader election or distributed locking; debugging split-brain, stale reads, lost updates, clock-skew
  bugs, duplicate processing, or retry storms; or whenever someone is tempted to roll their own
  consensus, order events by wall clock, or assume "exactly-once". Foundations under Kubernetes/etcd,
  Kafka, Spanner, and ML feature/checkpoint pipelines.
---

# Distributed Systems Fundamentals

Apply the judgment of an architect who has run replicated, partition-prone state in production for
years: a distributed system is **nodes that fail independently, talk over an unreliable network, with
no shared clock** — and almost every hard result follows from those three facts. These fundamentals
are technology-agnostic; specific systems (etcd, Kafka, Spanner) *implement* them.

## How to use this skill

1. **Read `distributed-systems-fundamentals-guide.md`** in this directory — the full reference (the
   theorems, consensus, consistency models, time, transactions, reliability, anti-patterns). Apply it
   to the design/review/debugging task at hand.
2. For concrete worked artifacts to imitate, read **`examples.md`**: a Raft-vs-ad-hoc leader-election
   comparison, an R+W>N quorum worked example, and a fencing-token distributed-lock sketch.
3. Match the surrounding system's existing conventions; apply the **correctness rules regardless** —
   they prevent silent data corruption, not just style nits.

## The essentials (full detail in `distributed-systems-fundamentals-guide.md`)

- **Partitions are inevitable (CAP).** During a partition you choose **C or A**, not both. Decide CP
  (refuse on the minority side, stay correct) or AP (stay up, reconcile later) — *per operation*, on
  purpose. If you didn't choose, you chose "corrupt-or-hang at random."
- **Consistency isn't free even without partitions (PACELC).** Strong consistency = a coordination
  round trip = latency. To go faster you accept staleness. Name the tradeoff explicitly.
- **Consensus has limits (FLP) but never lies.** Paxos/Raft guarantee **safety always**, liveness only
  under eventual timeliness — a healthy Raft cluster may *stall* during a bad partition but will never
  commit a wrong value. Anyone promising always-available + always-consistent + always-progressing is
  wrong.
- **Use real consensus for "exactly one" / "everyone must agree"** state: leader election, distributed
  locks, membership, the replicated log. **Never roll your own** via heartbeats/gossip → split brain.
  Gossip/heartbeats *suspect* failure; consensus/leases *decide* who acts.
- **Know the consistency spectrum** (linearizable → sequential → causal → eventual) and the cheap
  session guarantees you usually actually need (**read-your-writes, monotonic reads/writes**). Don't
  conflate **linearizability** (recency of one object) with **serializability** (transaction order).
- **Quorums: R + W > N** guarantees read/write overlap (freshest read); also keep **W > N/2**. N=3,
  W=2, R=2 is the balanced default. Quorum alone ≠ linearizable (needs versioning/read-repair).
- **Never order events or resolve conflicts by wall clock** — clock skew silently drops the wrong
  write and mis-expires leases. Use **monotonic clocks** for durations, **logical/vector/hybrid
  clocks** for causal order. **LWW silently loses data;** prefer vector clocks or **CRDTs** (converge
  with no coordination — the CALM theorem in practice) where you can.
- **At-least-once is the realistic delivery model; "exactly-once delivery" is impossible.** Make
  consumers **idempotent** (idempotency keys / conditional writes) to get exactly-once *processing*.
- **The dual-write problem:** writing to DB + queue/cache is not atomic. Use the **outbox pattern**
  (one local transaction + a CDC/relay) — not two independent writes.
- **Cross-node transactions:** **2PC blocks** if the coordinator dies (locks held forever); prefer
  **sagas** (local txns + compensations) across services. Snapshot/repeatable-read does **not** stop
  **write skew** — use serializable for correctness-critical invariants.
- **Reliability:** retries need **exponential backoff + jitter** (jitter is what prevents the
  thundering herd) and a **retry budget**; only retry idempotent ops. Bound queues for **backpressure**;
  **shed load** under overload; **degrade gracefully**; every remote call gets a timeout.
- **Distributed locks need fencing tokens.** A paused lock holder can have its lease expire while it
  still thinks it holds the lock → two writers. The resource must reject any **lower monotonic fencing
  token** than the highest it has seen. A lock without fencing guarantees nothing.
- **Consensus clusters are odd-sized** (3/5/7): a 5-node tolerates 2 failures; a 6th adds cost, not
  fault tolerance.

## Related skills

- `[[kubernetes-internals-expert]]` — how Kubernetes implements these: etcd's Raft as source of truth,
  lease-based leader election, the watch/informer log abstraction.
- `[[kubernetes-expert]]` — operating these primitives (leases, leader-elected controllers) in practice.
- `[[ml-checkpointing-orbax]]` — distributed checkpointing as a consistent-cut / atomicity problem.
- `[[data-engineering-feature-stores]]` — dual-write/outbox, exactly-once processing, and point-in-time
  (causal) correctness in feature pipelines.
- `[[ml-system-design]]` — applying CAP/PACELC, sharding, and consistency choices in ML system design.
