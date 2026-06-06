# AGENTS.md — Distributed Systems Fundamentals

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`distributed-systems-fundamentals-guide.md`** next to
> this file — read it before designing, reviewing, or debugging any replicated/sharded/multi-region/
> fault-tolerant system, and apply it. Worked artifacts to imitate (Raft-vs-ad-hoc leader election,
> R+W>N quorum, fencing-token lock) are in **`examples.md`**. This file is the always-on summary.
>
> Anchor: **a distributed system is nodes that fail independently, communicate over an unreliable
> network, with no shared clock.** These are technology-agnostic fundamentals; specific systems (etcd,
> Kafka, Spanner) implement them. The theorems are settled; verify product-specific behavior against
> current docs.

## Apply these by default on any distributed-systems work:

- **Partitions are inevitable (CAP).** During a partition choose **C or A**, per operation, on
  purpose: CP (refuse on the minority, stay correct) or AP (stay up, reconcile). PACELC: even without
  partitions, strong consistency costs latency — name the tradeoff.
- **Consensus guarantees safety always, liveness only under eventual timeliness (FLP).** A healthy
  Raft cluster may stall during a bad partition but never commits a wrong value. Reject any claim of
  always-available + always-consistent + always-progressing.
- **Use real consensus (Paxos/Raft/ZAB) for "exactly one" or "everyone agrees" state** — leader
  election, distributed locks, membership, the replicated log. **Never roll your own** via
  heartbeats/gossip (→ split brain, silent corruption). Gossip/heartbeats *suspect* failure;
  consensus/leases *decide*. Size consensus clusters **odd** (3/5/7).
- **Consistency spectrum:** linearizable → sequential → causal → eventual. Usually the real need is a
  **session guarantee** (read-your-writes, monotonic reads/writes), not full linearizability. Don't
  conflate **linearizability** (single-object recency) with **serializability** (transaction order).
- **Quorums: R + W > N** for overlapping read/write sets; also **W > N/2**. N=3/W=2/R=2 is the default.
  Quorum alone is not linearizable without versioning/read-repair.
- **Never order events or resolve conflicts by wall clock** — skew silently drops the wrong write and
  mis-expires leases. Monotonic clocks for durations; **logical/vector/hybrid clocks** for causality.
  **LWW silently loses data;** prefer vector clocks or **CRDTs** (coordination-free convergence) where
  applicable.
- **Delivery: at-least-once is reality; exactly-once *delivery* is impossible.** Make consumers
  **idempotent** (idempotency keys / conditional writes) for exactly-once *processing*.
- **Dual-write problem:** DB + queue/cache writes are not atomic → use the **outbox pattern** (one
  local transaction + CDC/relay), never two independent writes.
- **Cross-node transactions:** **2PC blocks** on coordinator failure (locks held forever); prefer
  **sagas** (local transactions + compensations) across services. Snapshot/repeatable-read does **not**
  prevent **write skew** — use serializable for correctness-critical invariants.
- **Partitioning:** prefer **consistent hashing** (only ~K/N keys move on resize) with virtual nodes;
  watch **hot shards/keys**. **Caching:** beware **stampede** (single-flight, jittered/early refresh,
  stale-while-revalidate) and stale-vs-DB drift.
- **Reliability:** retries need **exponential backoff + jitter** (jitter prevents the thundering herd)
  + a **retry budget**; only retry idempotent ops on retryable errors; use circuit breakers. **Bound
  queues for backpressure**, **shed load** under overload, **degrade gracefully**, timeout every
  remote call.
- **Distributed locks require fencing tokens.** A paused holder can keep acting after its lease expires
  → two writers. The protected resource must reject any **lower monotonic fencing token**. A lock
  without fencing guarantees nothing for correctness.

## Red flags to stop and fix
Rolling your own consensus; wall-clock ordering / LWW across machines; ignoring partitions;
non-idempotent retries; distributed locks without fencing tokens; dual writes without an outbox;
treating "exactly-once" as a transport guarantee; unbounded queues / retries without jitter;
even-numbered consensus clusters; synchronous unbounded fan-out.
