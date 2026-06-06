# RAG & Vector Databases — Worked Examples

Canonical, correct-in-shape artifacts to imitate. Verify exact image tags, API fields, and model
names/dims against current docs before deploying — this stack moves monthly (see the guide §13).

---

## 1. Qdrant StatefulSet on Kubernetes (sketch)

A vector DB is **stateful and RAM-bound** — run it as a `StatefulSet` with per-pod PVCs, a headless
Service for peer discovery, anti-affinity across nodes/zones, and RAM sized for the in-memory index.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: qdrant-headless          # headless service: stable per-pod DNS for the cluster
  labels: { app: qdrant }
spec:
  clusterIP: None
  selector: { app: qdrant }
  ports:
    - { name: http, port: 6333 }
    - { name: grpc, port: 6334 }
    - { name: p2p,  port: 6335 } # internal cluster (distributed mode)
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  serviceName: qdrant-headless
  replicas: 3                     # shard + replicate across nodes for scale & HA
  podManagementPolicy: Parallel
  selector: { matchLabels: { app: qdrant } }
  template:
    metadata:
      labels: { app: qdrant }
    spec:
      # Spread replicas across nodes/zones so one failure can't take the collection down.
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector: { matchLabels: { app: qdrant } }
              topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector: { matchLabels: { app: qdrant } }
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest   # PIN a concrete version in real use
          ports:
            - { containerPort: 6333, name: http }
            - { containerPort: 6334, name: grpc }
            - { containerPort: 6335, name: p2p }
          env:
            - name: QDRANT__CLUSTER__ENABLED
              value: "true"
          resources:
            requests:
              cpu: "4"
              memory: 32Gi          # size RAM for the in-memory HNSW index + working set
            limits:
              memory: 32Gi          # requests==limits for memory: avoid eviction surprises / OOM thrash
          volumeMounts:
            - { name: storage, mountPath: /qdrant/storage }
          readinessProbe:
            httpGet: { path: /readyz, port: 6333 }
            initialDelaySeconds: 10
          livenessProbe:
            httpGet: { path: /livez, port: 6333 }
            initialDelaySeconds: 20
  volumeClaimTemplates:            # stable per-pod persistent storage
    - metadata: { name: storage }
      spec:
        accessModes: [ "ReadWriteOnce" ]
        storageClassName: premium-rwo   # fast SSD-backed class (e.g. SSD PD on GKE)
        resources: { requests: { storage: 200Gi } }
```

Notes:
- **Sizing:** memory request must cover `n_vectors × dim × 4 bytes` (vectors) + HNSW graph overhead +
  OS page cache headroom (guide §6). Under-provision → OOMKill mid-build. If the index can't fit, enable
  quantization or move to a disk-based index.
- **Backups:** use Qdrant's native **collection snapshots**, not a raw PVC snapshot taken mid-write
  (which can capture a torn index). Test restore. Prefer the official Helm chart/operator and override
  resources for your sizing. See `[[kubernetes-expert]]`, `[[gke-master]]`.
- Put embedding/reranker GPU serving on a **separate node pool** from the RAM/CPU vector DB
  (`[[serving-frameworks]]`, `[[aiml-on-kubernetes]]`).

---

## 2. Hybrid retrieval (dense + sparse) → RRF fusion → cross-encoder rerank

The canonical two-stage pattern: over-retrieve cheaply with hybrid, fuse by **rank** (RRF), then re-score
the survivors with a cross-encoder and keep a small, high-precision set for the prompt. Illustrative
Python — adapt client/model APIs to your stack.

```python
def reciprocal_rank_fusion(ranked_lists, k: int = 60, top: int = 100):
    """Fuse multiple ranked result lists by rank (no score normalization needed)."""
    scores = {}
    for results in ranked_lists:                 # each list ordered best-first
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [doc_id for doc_id, _ in
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top]]


def retrieve(query: str, *, filters: dict, k: int = 100, n_final: int = 8):
    # --- Stage 1: hybrid retrieval (recall-oriented, large k) ---
    # IMPORTANT: embed the query with the SAME model+version as the documents,
    # and the correct asymmetric prefix if the model uses one (e.g. "query: ").
    q_vec = embed_query(query)                   # dense (metric MUST match the index, e.g. cosine)

    dense_hits = vector_db.search(               # ANN over HNSW, with metadata pre/filtered ANN
        vector=q_vec, limit=k, query_filter=filters,
    )
    sparse_hits = lexical_index.search(          # BM25 / SPLADE — exact terms, IDs, rare tokens
        text=query, limit=k, filters=filters,
    )

    fused_ids = reciprocal_rank_fusion(
        [[h.id for h in dense_hits], [h.id for h in sparse_hits]],
        k=60, top=k,
    )

    # --- Stage 2: cross-encoder rerank (precision-oriented, small n) ---
    candidates = fetch_documents(fused_ids)      # texts for the fused candidate ids
    pairs = [(query, doc.text) for doc in candidates]
    rel_scores = cross_encoder.predict(pairs)    # joint (query, doc) scoring — far more accurate than cosine
    reranked = [doc for _, doc in
                sorted(zip(rel_scores, candidates), key=lambda x: x[0], reverse=True)]

    return reranked[:n_final]                     # small, high-precision set for the prompt


def build_prompt(query: str, chunks: list) -> str:
    # "Lost in the middle": put the strongest chunks at the EDGES, dedup, carry citations,
    # and instruct the model to ABSTAIN if the context doesn't contain the answer.
    ctx = "\n\n".join(f"[{c.source_id}] {c.text}" for c in chunks)
    return (
        "Answer ONLY from the context below. Cite sources as [id]. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
```

Why each step: dense catches paraphrase/semantics, sparse catches exact tokens, **RRF** merges them
robustly without tuning weights, the **cross-encoder** fixes ranking precision, and the **prompt**
controls ordering + hallucination. Drop any one and a class of queries silently degrades.

---

## 3. HNSW / pgvector index configuration

### pgvector (HNSW inside Postgres)

Pick the operator class that **matches the embedding model's distance metric** — this is load-bearing.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id          bigserial PRIMARY KEY,
    doc_id      text NOT NULL,
    source      text NOT NULL,          -- metadata you will filter on
    lang        text NOT NULL,
    chunk       text NOT NULL,
    embedding   vector(768)             -- dim MUST equal the embedding model's output dim
);

-- HNSW index. The operator class MUST match how the model was trained:
--   vector_cosine_ops  -> cosine        (most dense text models; default choice)
--   vector_ip_ops      -> inner product (use ONLY if vectors are normalized / model trained for IP)
--   vector_l2_ops      -> Euclidean
CREATE INDEX ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);   -- build-time: higher m/ef_construction => better recall, slower build

-- B-tree on filter columns so metadata filtering doesn't force a full scan at scale.
CREATE INDEX ON chunks (source);
CREATE INDEX ON chunks (lang);

-- Query-time recall<->latency dial (per session/transaction):
SET hnsw.ef_search = 100;                  -- raise for recall, lower for latency

-- A cosine query. Use the operator that MATCHES the index (<=> cosine, <#> ip, <-> L2).
SELECT id, doc_id, source, chunk
FROM   chunks
WHERE  lang = 'en' AND source = 'docs'     -- metadata filter (indexed above)
ORDER  BY embedding <=> $1                  -- $1 = query embedding, same model+version as docs
LIMIT  100;                                 -- over-retrieve; rerank downstream
```

Gotchas:
- Operator class (`vector_cosine_ops`/`ip`/`l2`) **and** the query operator (`<=>`/`<#>`/`<->`) must both
  match the model's metric. Mixing them returns plausible-but-wrong neighbors with no error.
- The index dimension must equal the model output dim exactly; a model swap means re-embed + reindex.
- `ef_search` is the runtime recall/latency knob; `m` and `ef_construction` are fixed at build time.

### HNSW params, conceptually (any engine — Qdrant/Milvus/Weaviate/Faiss)

| Param | Phase | Effect | Direction |
|---|---|---|---|
| `M` | build | edges per node; recall + memory | higher → better recall, more RAM |
| `efConstruction` | build | candidate breadth while building | higher → better graph, slower build |
| `efSearch` / `ef` | query | candidate breadth while searching | **the recall↔latency dial** — tune per latency budget |

Tune `efSearch` against an eval set (recall@k vs. p99 latency); raise `M`/`efConstruction` only if recall
is still short after maxing reasonable `efSearch`. If the full-precision HNSW index won't fit in RAM,
switch to a quantized index (PQ/SQ/binary, with a full-vector rescore pass) or IVF-PQ / DiskANN
(guide §6).
