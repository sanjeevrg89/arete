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
