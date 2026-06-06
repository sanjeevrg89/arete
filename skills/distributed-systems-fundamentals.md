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

---

# Reference — distributed-systems-fundamentals

# Distributed Systems Fundamentals — Guide

The authoritative reference for this skill. These are the timeless results and design patterns every
distributed-systems architect must reason from. Higher-level systems (Kubernetes, etcd, Kafka,
Spanner, feature stores) are *built on* these ideas — they assume you already know them. This guide
is technology-agnostic; for how a specific system implements them, see the linked skills.

The mental anchor throughout: **a distributed system is a set of nodes that fail independently and
communicate over an unreliable network with no shared clock.** Almost every hard problem and every
counterintuitive result follows from those three facts.

## 1. The fallacies of distributed computing

Peter Deutsch / James Gosling's eight false assumptions. Every one of them has caused production
outages. Treat each as false and design around it.

1. The network is reliable. — It drops, reorders, duplicates, and partitions.
2. Latency is zero. — A cross-AZ round trip is ~1 ms; cross-region tens to hundreds of ms.
3. Bandwidth is infinite. — Large fan-out / large payloads saturate links.
4. The network is secure. — Assume eavesdropping and tampering; authenticate and encrypt.
5. Topology doesn't change. — Nodes, routes, and DNS move under you.
6. There is one administrator. — Coordinated change across teams/zones is itself a distributed problem.
7. Transport cost is zero. — Serialization, TLS, and syscalls cost CPU and money.
8. The network is homogeneous. — Mixed hardware, MTUs, and protocols behave differently.

A ninth practical fallacy worth adding: **clocks are synchronized and monotonic.** They are not
(see §8). Wall-clock comparisons across machines are a leading cause of subtle data corruption.

## 2. The core impossibility & tradeoff results

These are theorems, not opinions. You cannot engineer your way past them; you can only choose where
to sit.

### CAP

For a system that maintains a single logical piece of replicated state, during a **network
partition** you can guarantee at most one of:

- **Consistency (C)** — every read sees the most recent write (here, *linearizability* — see §6).
- **Availability (A)** — every non-failing node returns a non-error response.

Partition tolerance (P) is not a choice: in any real network, partitions *will* happen. So CAP
reduces to: **when (not if) a partition occurs, do you sacrifice consistency or availability?**

- **CP** — refuse/stall writes (or reads) on the minority side to preserve correctness. Consensus
  systems and strongly-consistent stores choose this (e.g. etcd, ZooKeeper, Spanner, a single-leader
  SQL primary). The minority partition becomes unavailable.
- **AP** — keep serving on both sides and reconcile later. Dynamo-style stores choose this (e.g.
  Cassandra, Riak, DNS). You get availability but must handle divergent/stale reads and conflicts.

CAP is widely misread. Crucial nuances:

- It only constrains behavior **during a partition**. When the network is healthy you can have both.
- "Consistency" in CAP specifically means linearizability, not ACID's "C", not "eventual consistency".
- It is not a global, static label on a database. The choice can be **per-operation**: the same
  system may serve some reads from any replica (AP) and route critical writes through a quorum (CP).

### PACELC — the result CAP omits

CAP says nothing about the common case (no partition). **PACELC** completes it:

> **if Partition: choose Availability or Consistency; Else (normal operation): choose Latency or
> Consistency.**

The "ELC" half is what you actually live with day to day. Strong consistency requires coordination
(a quorum round trip, a leader hop), which **costs latency**. To go faster you weaken consistency
(serve from a local/follower replica, accept staleness). Classify systems as e.g. PC/EC (always
consistent: Spanner-like), PA/EL (always favor latency/availability: Dynamo-like), PC/EL, PA/EC.
PACELC forces you to admit that even with a perfect network, **consistency is not free.**

### FLP impossibility

Fischer–Lynch–Paterson (1985): in an **asynchronous** network (no bound on message delay), **no
deterministic consensus protocol can guarantee both safety and liveness if even one node may crash.**
You cannot reliably distinguish a slow node from a dead one, so any protocol that always terminates
can be forced to either block or decide wrongly.

Why it matters: real consensus protocols (Paxos, Raft) **do not violate FLP** — they sidestep it.
They guarantee **safety always** (never two leaders' conflicting decisions committed) and provide
**liveness only under partial synchrony / eventual timeliness** (assuming the network *eventually*
behaves and timeouts are tuned sensibly). This is why a Raft cluster can stall (no progress) during a
bad partition but will **never return a wrong committed value** — that is the FLP tradeoff made
explicit. Anyone who claims a consensus system that is always available, always consistent, and
always makes progress is either ignoring partitions or wrong.

### Other foundational results to know by name

- **Two Generals Problem** — reliable agreement over a lossy channel is impossible with bounded
  certainty; the basis for "exactly-once delivery" being unattainable at the transport layer.
- **CALM theorem** — a computation can be coordination-free (and thus available/eventually
  consistent) **if and only if it is monotonic** (logic that only ever adds facts). This is the
  theory under CRDTs (§7) and a precise rule for *when* you can skip consensus.

## 3. Why consensus is required (and when)

**Consensus** = getting a set of nodes to agree on a single value (or a single ordered sequence of
values) such that: only a proposed value is chosen (validity), at most one value is chosen (agreement
/ safety), and a value is eventually chosen if the system is stable (termination / liveness).

You **need** real consensus whenever **correctness depends on exactly one party acting on shared
state**:

- **Leader election** — exactly one primary writer/coordinator at a time.
- **Distributed locking / leases** — exactly one holder of a mutex (with fencing — see §11).
- **Critical configuration / membership** — the cluster's source of truth (who's in the cluster,
  the current shard map, the current schema version).
- **An ordered, replicated log** — a totally-ordered sequence every replica applies identically
  (the foundation of state-machine replication and of "the log" abstraction in §5).

You **do not** need consensus for monotonic, commutative, or best-effort work (CALM, §2): metrics
aggregation, idempotent caches, eventually-consistent counters via CRDTs.

The Google SRE Book chapter *Managing Critical State* (sre.google/sre-book/managing-critical-state.html)
is the canonical operational treatment: use a proven consensus system for any "there can be only one"
or "everyone must agree" state, and resist the temptation to build your own.

### The protocols

| Protocol | One-liner | Notes |
|---|---|---|
| **Paxos** (single-decree) | Prove a single value can be agreed despite failures | Correct but famously hard to understand/implement; the theoretical bedrock. |
| **Multi-Paxos** | Paxos optimized for a stream of decisions with a stable leader | Powers many production systems; details vary widely between implementations. |
| **Raft** | Consensus designed for understandability: a strong leader + replicated log | The de-facto teaching and implementation standard; used by etcd, Consul, CockroachDB, TiKV. |
| **ZAB** | ZooKeeper Atomic Broadcast | Primary-backup atomic broadcast; powers ZooKeeper. |
| **Viewstamped Replication (VR)** | Leader-based replication via "views"; predates Paxos publication | Conceptually close to Raft/Multi-Paxos; worth reading for the lineage. |

These are **equivalent in power** — all solve the same atomic-broadcast/replicated-state-machine
problem and make the same FLP tradeoff. They differ in understandability and engineering ergonomics.

### Raft in detail (know this cold)

Raft decomposes consensus into three subproblems:

1. **Leader election.** Time is divided into **terms** (monotonically increasing integers; a logical
   clock for elections). Each server is *follower*, *candidate*, or *leader*. A follower that hears
   no heartbeat within its **randomized election timeout** becomes a candidate, increments the term,
   votes for itself, and requests votes. A candidate that wins a **majority** becomes leader. The
   randomized timeouts make split votes rare and self-healing. **At most one leader per term** is
   guaranteed because each server votes once per term and a majority is required.

2. **Log replication.** Clients send commands to the leader, which appends an entry and replicates it
   via `AppendEntries`. An entry is **committed** once stored on a **majority**; the leader then
   applies it to its state machine and tells followers the commit index. The leader forces follower
   logs to match its own (overwriting conflicting suffixes), enforcing the **Log Matching Property**:
   if two logs contain an entry with the same index and term, they are identical up to that index.

3. **Safety.** The **Election Restriction** ensures a candidate can only win if its log is at least as
   up-to-date as a majority — so a new leader always contains all committed entries. The **State
   Machine Safety** property: if any server applies an entry at a given index, no other server applies
   a *different* entry at that index. Together these guarantee that committed entries are durable and
   that all replicas apply the same sequence — even across leader changes.

Operational consequences to keep front of mind:
- A Raft cluster needs a **majority quorum** to commit. Size clusters **odd** (3, 5, 7): a 5-node
  cluster tolerates 2 failures; adding a 6th does **not** improve fault tolerance (still tolerates 2)
  but slows commits. This is why etcd is almost always 3 or 5 nodes.
- Reads can be served linearizably via the leader (with a quorum/lease check) or stale from followers
  if you accept staleness for latency (PACELC, §2).
- During a partition, the **minority side cannot elect a leader or commit** → it is unavailable (CP).

### Why ad-hoc heartbeats and gossip are not consensus

A common, dangerous shortcut: "I'll just have nodes heartbeat each other and whoever's alive longest
is the leader," or "I'll use gossip to elect a leader." These fail because:

- **No agreement guarantee → split brain.** Two nodes on opposite sides of a partition each conclude
  *they* are leader. Both accept writes. You now have divergent, conflicting state — silent data
  corruption that may surface days later.
- **Heartbeats detect liveness, not ground truth.** A GC pause, a long fsync, or a network blip makes
  a healthy leader look dead; a new leader is elected while the old one is still writing.
- **Gossip is eventually consistent by design** (anti-entropy, AP). It is excellent for membership,
  failure *detection*, and metadata dissemination — and useless for "exactly one" decisions, because
  it offers no total order and no commit point.

Rule: **use gossip/heartbeats to *suspect* failure; use a consensus system (or a lease backed by one)
to *decide* who acts.** See [[kubernetes-internals-expert]] for how Kubernetes does exactly this:
gossip-like watch/lease for liveness, but etcd's Raft for the authoritative state.

## 4. Replication & consistency models

### Replication topologies

| Topology | How writes flow | Strengths | Costs / hazards |
|---|---|---|---|
| **Single-leader** (primary/replica) | All writes to one leader; replicated to followers | Simple, no write conflicts, easy linearizable reads via leader | Leader is a bottleneck & SPOF; failover needs consensus to avoid split brain |
| **Multi-leader** | Multiple leaders accept writes, replicate to each other | Low-latency local writes, multi-region/offline | **Write conflicts are inevitable** → need conflict resolution (§7) |
| **Leaderless** (Dynamo-style) | Client/coordinator writes to many replicas; quorum reads | High availability, no failover step | Stale reads, conflicts, needs read-repair/anti-entropy |

### Synchronous vs asynchronous replication

- **Synchronous** — leader waits for the replica(s) to acknowledge before confirming the write.
  Stronger durability/consistency; **higher latency**; a slow/dead replica can stall writes.
- **Asynchronous** — leader confirms immediately, replicates in the background. Lower latency; **risk
  of data loss** on leader failure (the lost tail of un-replicated writes); replicas serve stale data.
- **Semi-synchronous** — wait for *at least one* replica; the common practical compromise.

This is PACELC's "EL vs EC" made concrete at the storage layer.

### The consistency spectrum (strongest → weakest)

| Model | Guarantee | Cost |
|---|---|---|
| **Linearizability** (atomic / strong) | Operations appear to take effect instantaneously at a single point between invocation and response; there is one global real-time order. The gold standard for "looks like a single copy." | Requires coordination on every op → highest latency; CP under partition. |
| **Sequential consistency** | All nodes see operations in the *same* total order, consistent with each process's program order — but not necessarily real-time order. | Cheaper than linearizable; no real-time guarantee. |
| **Causal consistency** | Operations causally related (happens-before, §8) are seen in order by everyone; concurrent ops may be seen in different orders. | Available under partition; the strongest model compatible with availability. |
| **Eventual consistency** | If writes stop, all replicas *eventually* converge. No ordering or recency guarantee in the meantime. | Cheapest, most available; hardest to reason about. |

**Client-centric (session) guarantees** — practical promises layered on weaker models so a *single
client* isn't surprised, even if global order is loose:

- **Read-your-writes (read-after-write)** — you always see your own prior writes.
- **Monotonic reads** — you never see time go backwards (a later read can't return older data than an
  earlier one).
- **Monotonic writes** — your writes are applied in the order you issued them.
- **Writes-follow-reads** — a write you make after reading X is ordered after the write that produced X.

These are usually the *actual* requirement behind a vague "it should be consistent" — pin down which
one before reaching for full linearizability (which is expensive).

### Quorums: R + W > N

For a leaderless/quorum system with **N** replicas, requiring **W** acknowledgments per write and
**R** replicas per read:

> If **R + W > N**, the read and write replica sets **must overlap by at least one node**, so a read
> is guaranteed to touch at least one replica holding the latest write → you can read the freshest
> value (a "strict quorum").

Also require **W > N/2** so two writes can't both succeed without overlapping (preventing concurrent
conflicting commits). Tuning:

- **W=N, R=1** — fast reads, slow/fragile writes (any down replica blocks writes).
- **W=1, R=N** — fast writes, slow reads; risky durability.
- **N=3, W=2, R=2** — the classic balanced strict quorum (tolerates 1 failure, still consistent).

Caveats: strict quorums still suffer edge cases (sloppy quorums + hinted handoff trade consistency
for availability; concurrent writes to the overlap need conflict resolution; a node returning a value
doesn't mean it's the latest unless versioning is correct). Quorum ≠ linearizable by itself — you
also need read-repair / versioning to make it so. Worked example in `examples.md`.

## 5. Partitioning / sharding, caching, queues & the log

### Partitioning (sharding)

Split data across nodes so each holds a subset. Two strategies:

- **Hash partitioning** — `shard = hash(key) mod N` (or a hash range). Even load distribution; **kills
  range scans** (adjacent keys scatter). Naive `mod N` is a trap: **changing N reshuffles almost every
  key.**
- **Range partitioning** — contiguous key ranges per shard. Great for range scans; prone to **hot
  spots** (e.g. timestamp or sequential-ID keys all hit the newest shard).

**Consistent hashing** solves the rehash-on-resize problem: map both nodes and keys onto a hash ring;
a key belongs to the next node clockwise. Adding/removing a node only moves the keys between adjacent
points — **~K/N keys move**, not all of them. Use **virtual nodes** (many ring points per physical
node) to smooth load and handle heterogeneous capacity. (Alternatives like rendezvous/HRW hashing and
jump consistent hash exist with different tradeoffs.)

**Rebalancing** — when you add/remove capacity. Prefer schemes where the number of partitions is
fixed and you move whole partitions (not rehash keys); avoid automatic rebalancing that can cascade
under load. **Hot shards / hot keys** (a celebrity user, a viral key) defeat uniform sharding;
mitigate with key salting/splitting, request coalescing, dedicated shards, or a cache layer.

### Caching

- **Cache-aside (lazy)** — app reads cache; on miss, loads from DB and populates cache. Simple,
  resilient; first request per key is slow; risk of serving stale data until TTL/invalidation.
- **Write-through** — writes go to cache and DB synchronously. Cache always fresh; higher write
  latency.
- **Write-behind (write-back)** — write to cache, flush to DB asynchronously. Fast writes; **data-loss
  risk** and complexity.
- **Invalidation** is the hard part ("there are only two hard problems..."). Prefer short TTLs +
  explicit invalidation on write; beware the **dual-write problem** (§9) between cache and DB.
- **Cache stampede / thundering herd** — a popular key expires and thousands of requests miss
  simultaneously, all hammering the DB. Mitigate with: a **single-flight / request-coalescing** lock
  (one loader, others wait), **probabilistic early expiration** (refresh slightly before TTL), and
  **stale-while-revalidate** (serve stale, refresh in background).

### Queues, the log, and delivery semantics

Asynchronous messaging decouples producers from consumers and absorbs load. Know the delivery
semantics precisely:

- **At-most-once** — deliver and forget. Messages may be lost, never duplicated. Fine for metrics.
- **At-least-once** — retry until acked. Never lost, **may duplicate**. The pragmatic default — which
  is why **consumers must be idempotent** (§11).
- **Exactly-once *delivery* is impossible** over an unreliable network (Two Generals, §2). What real
  systems provide is **exactly-once *processing*** = at-least-once delivery **+ idempotent/dedup
  consumers** or **transactional reads-process-writes** (offset commit + output in one atomic step).
  Never trust a vendor's bare "exactly-once" claim without understanding which mechanism backs it.

**The distributed log abstraction** — an append-only, totally-ordered, durable, replayable sequence
of records, partitioned for throughput and replicated for durability (Kafka, Pulsar, and the
replicated log inside consensus systems). It is a unifying primitive:

- Ordering within a partition is total; across partitions it is not (choose your partition key to
  group what must be ordered — e.g. by entity ID).
- Consumers track an **offset**; replay = reset offset. This enables event sourcing, CDC, and
  rebuilding derived state.
- It is the substrate for **state-machine replication** (apply the same ordered log → identical
  state) and for the **outbox pattern** (§9). See [[data-engineering-feature-stores]] for log-driven
  feature pipelines.

## 6. Linearizability vs serializability (don't conflate them)

- **Linearizability** is about **recency/ordering of single operations on a single object** in real
  time (a *consistency* model). "There appears to be one copy and you always read the latest."
- **Serializability** is about **transactions over multiple objects** appearing to run in *some*
  serial order (an *isolation* model). It says nothing about real-time recency.
- **Strict serializability** = serializable **and** linearizable (e.g. Spanner). The strongest
  combined guarantee — and the most expensive.

You need to know which one a given requirement actually needs; people say "consistent" when they mean
one of four different things.

## 7. Conflict resolution

When multiple writers can update the same datum concurrently (multi-leader/leaderless), you *will*
get conflicts. Options, weakest to strongest:

- **Last-Write-Wins (LWW)** — keep the write with the highest timestamp; discard the rest. Simple but
  **silently loses data**, and it depends on clocks — across machines that means lost updates from
  clock skew (§8). Acceptable only when losing concurrent writes is genuinely fine.
- **Version vectors / vector clocks** — detect concurrency precisely (§8); surface conflicts to the
  application (or client) to merge (Dynamo/Riak siblings). Correct, but pushes work to the app.
- **CRDTs (Conflict-free Replicated Data Types)** — data types (counters, sets, registers, maps,
  sequences/RGA for text) whose merge function is **commutative, associative, idempotent**, so
  replicas **converge automatically with no coordination** regardless of order. This is the CALM
  theorem (§2) in practice: monotonic structure ⇒ coordination-free. The backbone of collaborative
  editing and offline-first/local-first apps. Cost: metadata growth and restricted operations.

## 8. Time & ordering

**There is no global "now."** Each machine has its own clock; clocks drift, get stepped by NTP,
and even run backward. Two clock kinds, and the difference is load-bearing:

- **`time-of-day` / wall clock** — calendar time. Can jump (NTP correction, leap seconds), is **not
  monotonic**. **Never measure elapsed time or order events with it.**
- **Monotonic clock** — always increases, no absolute meaning; use it for **durations, timeouts,
  retries, and lease checks** within a single machine.

**Why wall-clock ordering across machines is dangerous:** two machines' clocks can differ by tens of
ms (sometimes far more). Ordering events by wall-clock timestamp → causally-later events look earlier,
LWW discards the wrong write, "expired" leases look valid, and you get lost updates with no error.
This is one of the most common silent-corruption bugs in distributed systems.

**Logical clocks** order events by causality without trusting wall clocks:

- **Lamport timestamps** — a single counter per node, incremented on every event and on receive set to
  `max(local, received) + 1`. Gives a **total order consistent with causality** (if A → B then
  L(A) < L(B)) but **cannot tell concurrency from causality** (L(A) < L(B) does not imply A → B).
- **Vector clocks** — a vector of counters, one entry per node. Compare element-wise: you can tell
  whether A happened-before B, B before A, or they are **concurrent**. The precise tool for detecting
  conflicts (§7). Cost: size grows with the number of writers.
- **Hybrid Logical Clocks (HLC)** — combine a physical-time component (so timestamps are close to wall
  time and human-meaningful) with a logical component (so causality is preserved even when wall clocks
  disagree). Used where you want both human-readable timestamps and causal correctness.
- **Bounded-uncertainty clocks** (e.g. Spanner's TrueTime) — expose a clock *interval* `[earliest,
  latest]` and **wait out the uncertainty** before committing, achieving external consistency. Requires
  tight clock infrastructure (GPS/atomic); verify against current sources for specifics.

Rule: **order by causality (logical clocks / consensus log), not by wall clock.** Use wall clocks only
for human-facing timestamps and coarse TTLs with generous margins.

## 9. Transactions across nodes

### Two-Phase Commit (2PC) and its blocking problem

2PC coordinates an atomic commit across multiple participants: a **coordinator** sends *prepare*; each
participant durably votes yes (promising it can commit) or no; if all vote yes the coordinator sends
*commit*, else *abort*. It provides atomicity across nodes — but:

- **It blocks.** If the coordinator crashes after participants have voted yes but before delivering the
  decision, participants are stuck holding locks **indefinitely** (they've promised to commit but don't
  know the outcome). A synchronous, coordination-heavy protocol that reduces availability.
- It is a **CP** mechanism with a single point of failure (the coordinator). 3PC tries to fix the
  blocking but breaks under network partitions and is rarely used.
- It is acceptable within a tightly-coupled, low-latency boundary; it scales poorly across services
  or regions. For cross-service atomicity, prefer sagas.

### Sagas

A **saga** is a sequence of local transactions, each in its own service, where each step has a
**compensating action** that semantically undoes it. There is no global lock: if step *k* fails, you
run compensations for steps *k-1 … 1*. Coordinated either by **choreography** (events trigger the next
step) or **orchestration** (a central coordinator drives steps). You trade atomicity+isolation for
availability and loose coupling — accepting **no isolation** (intermediate states are visible) and the
need to design idempotent steps and compensations. The standard pattern for long-lived, cross-service
business workflows.

### Isolation levels (the ANSI ladder + the gaps)

| Level | Prevents | Still allows |
|---|---|---|
| Read Uncommitted | (nothing) | dirty reads |
| Read Committed | dirty reads | non-repeatable reads, lost updates |
| Snapshot / Repeatable Read (MVCC) | dirty + non-repeatable reads | **write skew**, phantoms (impl-dependent) |
| Serializable | all anomalies | (correct, but costly) |

Know the anomalies by name: **dirty read, non-repeatable read, phantom, lost update, write skew, read
skew**. "Repeatable read" and "snapshot isolation" do **not** prevent write skew — a classic
correctness bug (e.g. two doctors both going off-call because each read the other as on-call). When in
doubt for correctness-critical invariants, use **serializable** (or explicit predicate locks /
`SELECT ... FOR UPDATE`).

### The dual-write problem

Writing to **two systems** in one logical operation (e.g. DB **and** a message queue, or DB **and**
cache) is **not atomic** — one can succeed and the other fail, leaving them inconsistent. There is no
distributed transaction across heterogeneous systems you can rely on. Solutions:

- **Outbox pattern** — write the business change **and** an "event to publish" row in the **same local
  DB transaction**. A separate relay (often via **CDC** tailing the DB log) reads the outbox and
  publishes to the queue at-least-once. The DB transaction is the single source of atomicity; the relay
  makes the second write reliable and idempotent. This is the correct, standard fix for "save to DB and
  emit an event."
- **Listen-to-yourself / event-sourcing** — make the log the source of truth and derive the DB from it.

## 10. Failure detection & reliability patterns

### Failure detectors

You cannot perfectly distinguish "crashed" from "slow." A failure detector is **suspicion**, defined by
**completeness** (every real failure is eventually suspected) and **accuracy** (no false suspicion).
FLP (§2) means you can't have both perfectly in an async system. Practical detectors:

- **Heartbeats + timeout** — simplest; tuning the timeout trades detection speed vs false positives.
- **Phi-accrual failure detector** — outputs a *suspicion level* φ (a probability) instead of a binary
  up/down, adapting to observed network variance; used by Cassandra/Akka. Lets callers pick their own
  threshold.

Use detection to *suspect*; use consensus/leases to *decide* (§3).

### Retries, backoff, jitter — and the thundering herd

Retries are mandatory but dangerous. Naive fixed-interval retries from many clients **synchronize**
and produce a **thundering herd / retry storm** that turns a blip into an outage (and can cause
*metastable failure* — the system stays down after the trigger is gone). Rules (per AWS Builders'
Library, *Timeouts, retries and backoff with jitter*):

- **Exponential backoff** — double the delay each attempt, with a cap.
- **Add jitter** — randomize the delay (e.g. *full jitter*: `sleep = random(0, min(cap, base·2^n))`).
  Jitter is what actually de-correlates clients; backoff alone still synchronizes. **Always add jitter.**
- **Cap total attempts / budget retries** — use a *retry budget* / token bucket so the system retries
  only a small fraction of total load; bound depth so retries don't amplify through every layer (a
  3-layer call stack each retrying 3× = 27× load).
- **Only retry idempotent / retry-safe operations**, and only on retryable errors (timeouts, 503),
  never on 4xx. Use **circuit breakers** to stop hammering a known-down dependency.

### Idempotency

The single most important reliability tool, because at-least-once is the realistic delivery model.
**An idempotent operation can be applied many times with the same effect as once.** Make writes
idempotent via an **idempotency key** (a client-supplied unique ID; the server records processed keys
and dedupes), natural idempotency (`SET x = 5` vs `x += 1`), or conditional writes
(compare-and-set / version checks). Required for safe retries, at-least-once queues, and exactly-once
*processing* (§5). See `examples.md`.

### Graceful degradation, backpressure, load shedding

- **Backpressure** — when a consumer can't keep up, propagate "slow down" upstream (bounded queues,
  flow control, blocking) rather than buffering unboundedly until OOM. Unbounded queues hide
  backpressure and convert latency into collapse.
- **Load shedding** — under overload, **reject excess work fast** (return 429/503) to protect the core
  rather than letting everything degrade. Prioritize critical traffic.
- **Graceful degradation** — serve a reduced but useful response when a dependency is down (stale
  cache, default recommendations, read-only mode) instead of failing hard.
- **Bulkheads & timeouts** — isolate resource pools so one slow dependency can't exhaust all threads;
  every remote call has a timeout (a call with no timeout is a latent hang).

## 11. Anti-patterns (the ones that cause real incidents)

- **Rolling your own consensus.** Ad-hoc leader election via heartbeats/gossip ⇒ split brain and
  silent corruption (§3). Use a proven consensus system (etcd/ZooKeeper) or a managed primitive.
- **Wall-clock ordering / LWW across machines.** Clock skew silently drops the "wrong" write and
  expires leases incorrectly (§8). Order by causality.
- **Ignoring partitions ("it'll be fine").** Partitions are inevitable (CAP). If you didn't *choose*
  CP or AP, you chose "corrupt-or-hang at random."
- **Non-idempotent retries.** Retrying a non-idempotent write (charge card, append row) double-applies
  it. Every retried operation needs an idempotency key (§10).
- **Distributed locks for correctness without fencing tokens.** A client can hold a lock, pause (GC),
  have its lease expire, then resume and act — while another client also holds the lock. **Two holders
  at once.** The fix: the lock service issues a **monotonically increasing fencing token** with each
  grant; every protected resource **rejects any write carrying a token lower than the highest it has
  seen.** Without fencing, a distributed lock guarantees nothing under pauses/partitions. (A lock for
  *efficiency*—avoiding duplicate work—can skip fencing; a lock for *correctness* cannot.) Sketch in
  `examples.md`.
- **Unbounded queues / no backpressure** → memory blowup and metastable failure.
- **Synchronized retries without jitter** → thundering herd.
- **Dual writes without an outbox** → DB and queue/cache drift apart (§9).
- **Treating "exactly-once" as a transport guarantee** rather than idempotent processing (§5).
- **Even-numbered consensus clusters** (4, 6) — extra cost, no extra fault tolerance (§3).
- **Synchronous, unbounded fan-out** — one request fanning to N services synchronously multiplies tail
  latency and failure probability (`p_fail ≈ N · per-call failure`).

## 12. How these show up in ML / platform infrastructure

- **Kubernetes control plane** — etcd is a **Raft** cluster (CP); it is the single source of truth for
  all cluster state. Controllers use **leases** (backed by etcd) for leader election, *not* ad-hoc
  heartbeats. The watch/informer model is a **log/event-stream** consumed with offsets (resource
  versions). See [[kubernetes-internals-expert]] and [[kubernetes-expert]].
- **Parameter / embedding sharding** — large embedding tables and model parameters are **partitioned**
  across servers/accelerators; lookups are quorum-free reads, but updates must handle **hot keys**
  (popular embeddings) exactly like hot shards (§5).
- **Distributed checkpointing** — a multi-host checkpoint must be a **consistent cut**: every shard
  reflects the *same* training step, written atomically (write-to-temp-then-rename, or a barrier) so a
  restore is coherent. This is a distributed-consistency problem, not just an I/O problem. See
  [[ml-checkpointing-orbax]].
- **Feature pipelines & training/serving skew** — feature stores must avoid the **dual-write problem**
  between offline and online stores (use an outbox/log so both derive from one ordered source) and need
  **exactly-once processing** (idempotent upserts keyed by entity+timestamp) to avoid double-counting.
  Point-in-time-correct joins are fundamentally a **causal-ordering** problem. See
  [[data-engineering-feature-stores]] and [[ml-system-design]].
- **Distributed training** — collective ops (all-reduce) assume a synchronized group; a single straggler
  or failed worker stalls the step (a liveness/failure-detection problem). Gang scheduling exists
  precisely because partial groups can't make progress.

## 13. Version awareness

The *theorems* here (CAP, PACELC, FLP, Two Generals, CALM) are settled and timeless. The *systems* that
embody them move fast: consensus-library defaults, etcd/Kafka/Spanner feature sets, TrueTime-style clock
bounds, and "exactly-once" implementations all change between versions. **Verify product-specific
behavior, default quorum sizes, isolation-level implementations, and delivery-semantics claims against
current official documentation** before relying on them. It is 2026; do not assume a blog post or this
guide reflects the latest release of any specific product.

## 14. Canonical references

- **Google SRE Book — Managing Critical State** (consensus for distributed state):
  https://sre.google/sre-book/managing-critical-state/
- **AWS Builders' Library — Leader election in distributed systems:**
  https://aws.amazon.com/builders-library/leader-election-in-distributed-systems/
- **AWS Builders' Library — Timeouts, retries, and backoff with jitter:**
  https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
- **Kleppmann, *Designing Data-Intensive Applications*** (2nd ed.) — the single best practitioner text
  on Parts II–III (replication, partitioning, transactions, consistency & consensus). Verify edition.
- **Raft** — Ongaro & Ousterhout, *In Search of an Understandable Consensus Algorithm*:
  https://raft.github.io/raft.pdf · interactive: https://raft.github.io/
- **Paxos** — Lamport, *Paxos Made Simple* (2001).
- **FLP** — Fischer, Lynch, Paterson, *Impossibility of Distributed Consensus with One Faulty Process*
  (1985).
- **CAP** — Brewer's conjecture; Gilbert & Lynch's proof (2002); Brewer, *CAP Twelve Years Later* (2012).
- **PACELC** — Abadi, *Consistency Tradeoffs in Modern Distributed Database System Design* (2012).
- **CALM** — Hellerstein & Alvaro, *Keeping CALM: When Distributed Consistency is Easy* (2020).
- **Vector clocks** — Fidge / Mattern (1988). **Lamport clocks** — Lamport, *Time, Clocks, and the
  Ordering of Events* (1978). **HLC** — Kulkarni et al., *Hybrid Logical Clocks* (2014).
- **Fencing tokens / locks** — Kleppmann, *How to do distributed locking* (2016).
- **CRDTs** — Shapiro et al., *Conflict-free Replicated Data Types* (2011).

---

# Distributed Systems Fundamentals — Worked Examples

Canonical patterns to imitate. Each shows the *wrong* shortcut and the *correct* fundamental, with
enough detail to reason about (not copy-paste production code). See
`distributed-systems-fundamentals-guide.md` for the full rationale.

---

## 1. Leader election: ad-hoc heartbeats vs Raft (a comparison note)

**The task:** exactly one node should be the active writer/coordinator at a time.

### The tempting-but-wrong approach: ad-hoc heartbeats

```
Every node broadcasts "I'm alive" every 1s.
A node that hasn't heard from the current leader in 3s declares itself leader.
```

Failure timeline that corrupts data:

```
t=0   Node A is leader, happily writing to the shared DB.
t=1   A enters a 5s stop-the-world GC pause (or a NIC blip drops its heartbeats).
t=4   B and C have heard nothing for 3s → B declares itself leader and starts writing.
t=5   A's GC finishes. A never saw a problem; A is STILL writing as "leader".
      → Two leaders. Interleaved/conflicting writes. Split brain. Silent corruption.
```

Why it's unfixable by tuning timeouts: you cannot distinguish *slow* from *dead* (FLP). Tighter
timeouts → more false positives (flapping); looser timeouts → slower failover. There is **no commit
point and no agreement** — two partitions can each "elect" a leader with full confidence.

Gossip-based election has the same flaw: gossip is eventually consistent (AP) and offers no total
order or single decision point. Gossip is great for *membership* and *failure suspicion* — not for
"exactly one."

### The correct approach: Raft (or a lease backed by a consensus system)

Raft makes leadership a **committed, agreed** decision:

- Time is divided into **terms** (a logical clock). A candidate must win votes from a **majority** to
  become leader, and each node votes at most once per term → **at most one leader per term**, by
  construction.
- The new leader's log must be **at least as up-to-date** as the majority (Election Restriction), so
  it contains every committed entry — no committed data is lost across a leadership change.
- During a partition, **only the majority side can elect a leader and commit.** The minority side
  cannot — it goes read-only/unavailable (CP). The old leader on the minority side **cannot commit**
  anything, so even if it still thinks it's leader, it does no harm.

What the GC-pause timeline looks like under Raft:

```
t=0   A is leader for term 5, replicating to a majority {A,B,C,D,E}.
t=1   A pauses. It stops sending heartbeats.
t=4   Followers time out, increment to term 6, elect B (majority votes). B is the only term-6 leader.
t=5   A resumes, still thinks it's term-5 leader, tries to AppendEntries.
      Followers reply "your term 5 < current term 6" → A steps down. No conflicting commit. Safe.
```

**Decision rule:** for any "there can be only one" state, use a proven consensus system (etcd,
ZooKeeper) — typically by holding a **lease/lock object** in that system rather than implementing
Raft yourself. See `[[kubernetes-internals-expert]]`: Kubernetes controllers use lease objects in
etcd for leader election, not ad-hoc heartbeats.

| | Ad-hoc heartbeats/gossip | Raft / consensus-backed lease |
|---|---|---|
| Agreement (one leader) | None — split brain | Guaranteed (majority + per-term vote) |
| Behavior under partition | Both sides "lead" | Minority is unavailable; never two committers |
| Data safety across failover | Lost/conflicting writes | Committed entries preserved |
| Liveness | "Always available" (falsely) | May stall during bad partition (FLP, on purpose) |
| Correct use | Membership, failure *suspicion* | Leader election, locks, critical state |

---

## 2. Quorum reads/writes: an R + W > N worked example

**Setup:** leaderless store, replication factor **N = 3** (replicas r1, r2, r3). Choose
**W = 2**, **R = 2**. Check the invariant: `R + W = 4 > N = 3` ✓ and `W = 2 > N/2 = 1.5` ✓.

**Write `x = "v2"` (overwriting old `x = "v1"`):**

```
Client writes x="v2" (version 2) to all 3 replicas, waits for W=2 acks.
  r1: x="v2" (v2)   ack ✓
  r2: x="v2" (v2)   ack ✓        ← 2 acks received → write succeeds (returns to client)
  r3: x="v1" (v1)   (slow/down — never acked; still holds the stale value)
```

**Read `x`:** client reads from any **R = 2** replicas. Because `R + W > N`, the read set of 2 and
the write set of 2 **must overlap in at least one replica**. Enumerate every possible read pair:

```
Read {r1, r2} → sees v2, v2 → returns v2 ✓
Read {r1, r3} → sees v2, v1 → versions differ; pick highest version = v2 ✓ (and read-repair r3)
Read {r2, r3} → sees v2, v1 → highest version = v2 ✓ (and read-repair r3)
```

Every read pair contains at least one replica that holds **v2** → the client can always recover the
freshest value. That overlap is the entire point of `R + W > N`.

**Why `W > N/2` also matters** — it prevents two concurrent writes from both "succeeding" on disjoint
quorums:

```
With N=3, if W=1 were allowed:
  Writer P writes a="pa" to r1 only (1 ack, succeeds).
  Writer Q writes a="qa" to r2 only (1 ack, succeeds).
  → Two conflicting "successful" writes with NO overlap. Lost update, no error.
W=2 forces any two writes to share a replica, surfacing the conflict.
```

**Critical caveats (quorum ≠ linearizable on its own):**

- You **must version values** (timestamp/version vector) so the reader can pick the newest and run
  **read-repair**; "the value a replica returns" is meaningless without ordering metadata.
- **Concurrent** writes to the overlap aren't ordered by R+W>N — you still need conflict resolution
  (vector clocks / CRDTs, not naive wall-clock LWW — see the guide §7–§8).
- **Sloppy quorums + hinted handoff** (writing to *any* N reachable nodes during a partition for
  availability) **break** the overlap guarantee — they're an AP choice, not strict-quorum consistency.

Common configs: `N=3,W=2,R=2` (balanced, tolerates 1 failure); `W=N,R=1` (fast reads, fragile
writes); `W=1,R=N` (fast writes, risky).

---

## 3. Distributed lock with fencing tokens (a sketch)

**The bug — a lock without fencing:**

```
1. Client C1 acquires lock L (lease, e.g. 30s TTL) from the lock service.
2. C1 prepares a write to storage, then enters a long GC pause (> 30s).
3. The lease expires. The lock service grants L to client C2.
4. C2 writes to storage.
5. C1 wakes up — still believes it holds L (it never saw the expiry) — and writes to storage.
   → Two clients wrote under "the same lock." The lock guaranteed NOTHING.
```

No lease TTL fixes this: a pause/partition can always exceed any timeout, and the holder doesn't know
its lease expired. The lock service alone cannot prevent a stale holder from acting on the resource.

**The fix — monotonic fencing tokens enforced at the resource:**

The lock service issues a **strictly increasing token** with every grant (it can be the consensus
log index / term — naturally monotonic). The **protected resource** records the highest token it has
accepted and **rejects any write carrying a lower-or-equal token.**

```
1. C1 acquires L  → token = 33.
2. C1 pauses (GC).
3. Lease expires; C2 acquires L → token = 34 (strictly greater — guaranteed by the lock service).
4. C2 writes(payload, fence=34).
     Storage: highest_seen (0) < 34 → ACCEPT, set highest_seen = 34.
5. C1 wakes and writes(payload, fence=33).
     Storage: 33 <= highest_seen (34) → REJECT.   ← the stale holder is fenced out. Safe.
```

Pseudocode the resource enforces (the token check must be **atomic** with the write — e.g. a
conditional/compare-and-set update):

```python
def handle_write(payload, fence_token):
    # Single atomic compare-and-set on the resource:
    #   apply the write IFF fence_token > stored highest_seen, and bump highest_seen.
    ok = storage.compare_and_apply(
        condition=lambda cur: fence_token > cur.highest_seen,
        mutate=lambda cur: cur.set(payload=payload, highest_seen=fence_token),
    )
    if not ok:
        raise FencedOut(f"token {fence_token} <= highest seen; stale lock holder rejected")
```

**Key points:**

- The token must come from a source that guarantees **monotonicity** (a consensus log index / term, or
  an atomic counter in the consensus system) — never a wall-clock timestamp (clock skew breaks
  monotonicity → see guide §8).
- The **enforcement lives at the resource**, not the lock client. A lock client that "checks if it
  still holds the lock" before writing still has a TOCTOU race across the pause; only the resource
  rejecting old tokens is safe.
- **Efficiency vs correctness:** a lock used only to *avoid duplicate work* (it's merely wasteful if
  two run) can skip fencing. A lock protecting a *correctness invariant* (exactly one writer) **must**
  fence. Be explicit about which you have.

(Reference: Kleppmann, *How to do distributed locking*, 2016.)
