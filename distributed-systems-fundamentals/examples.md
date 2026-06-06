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
