# AGENTS.md — RAG & Vector Databases

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`rag-vector-databases-guide.md`** next to this file —
> read it before building or debugging a RAG system, and apply it. Concrete artifacts to imitate
> (K8s StatefulSet, hybrid+rerank pipeline, HNSW/pgvector index config) are in **`examples.md`**.
> This file is the always-on summary.
>
> **Governing principle: retrieval quality is the ceiling.** The LLM can't answer from context it never
> got; it *will* hallucinate over irrelevant chunks. Treat RAG as IR + an LLM, and **measure every stage
> against an eval harness** — never tune by vibes.

## When working on RAG / vector search, apply these by default:

- **Match the distance metric to the embedding model.** Use the metric the model was trained with (model
  card) — usually cosine; normalize and use dot/IP only if intended. Mismatch degrades recall **silently**
  (no error). Set the index metric at creation; many DBs can't change it without a rebuild.
- **Use the same embedding model + version + prefix for queries and documents.** Asymmetric models need
  the right `query:`/`passage:` prefix. Changing the embedding model means a **full re-embed + reindex** —
  never mix versions in one index.
- **Chunk structurally, not blindly.** Default to recursive/structure-aware splitting on natural
  boundaries; tune size/overlap against eval. Prefer parent-document / auto-merging (retrieve small, feed
  large) and contextual retrieval for ambiguous corpora. Ensure chunk text == embedded text.
- **Hybrid by default.** Dense (semantic) + sparse (BM25/SPLADE for exact terms, IDs, rare tokens), fused
  with **RRF** (`1/(k+rank)`, k≈60 — no score normalization needed). Dense-only misses exact matches.
- **Always rerank.** Over-retrieve top-k (50–200) cheaply, cross-encoder rerank, keep top-n (5–20) for the
  prompt. Biggest precision win after hybrid.
- **HNSW is the default in-memory ANN.** Tune `M`/`efConstruction` at build, `efSearch` at query
  (the recall↔latency dial). RAM-bound: vectors + graph live in RAM. Exceed RAM → quantize (PQ/SQ/binary,
  +rescore) or use IVF-PQ / **DiskANN**. The triangle is recall ↔ latency ↔ memory — pick two.
- **Filter on indexed metadata.** Prefer integrated filtered ANN (Qdrant/Milvus/Weaviate); index filter
  fields. Watch pre-filter graph collapse and post-filter underflow at scale.
- **Pick the DB for the job:** pgvector/AlloyDB AI (data already in Postgres, moderate scale,
  transactional); Qdrant (filtering/hybrid, easy ops); Milvus (billion-scale); Weaviate (schema + hybrid +
  multi-tenancy); Elasticsearch/OpenSearch/Vespa (lexical + vector in one engine); Pinecone (managed).
- **Assemble context carefully:** most-relevant chunks at the **edges** (lost-in-the-middle), small n,
  dedup, carry citations, instruct the model to **abstain** when context lacks the answer.
- **Vector DBs on K8s are stateful + RAM-bound:** `StatefulSet` + `volumeClaimTemplates` (fast SSD),
  headless Service, anti-affinity across zones, replication for HA, RAM sized for the in-memory index
  (`n × dim × 4 bytes` + graph + headroom), native snapshots for backup (test restore). Embedding/reranker
  services are stateless — scale them with HPA/KEDA. See `[[gke-master]]`, `[[kubernetes-expert]]`.
- **Evaluate two layers:** retrieval (recall@k, MRR, nDCG on a labeled set) **and** end-to-end
  (faithfulness / answer-relevance / context-precision, RAGAS). Change one variable at a time; keep what
  moves the metric.

## Anti-patterns (reject these)
Naive fixed-size chunking as the final strategy · no reranking · cosine/dot/L2 metric mismatch ·
dense-only on keyword/ID-heavy corpora · no eval harness · stale index / un-re-embedded model upgrade ·
unindexed filter fields at scale · over-stuffed context burying the answer · mixed embedding-model
versions in one index · no abstain instruction.

## Definition of done for a RAG change
Report numbers, not impressions: retrieval metrics (recall@k, nDCG) on a labeled set **and** end-to-end
faithfulness/answer-relevance, before vs. after the change. Confirm the distance metric matches the model
and the index is fresh.

## Version awareness (2026)
Fast-moving: verify embedding dims/length/prefix/metric, reranker availability, DB index & filtering &
consistency semantics, pgvector/AlloyDB AI features, and RAGAS APIs against **current docs**. Principles
are stable; knobs and APIs are not.

## Related skills
`[[llm-app-agent-frameworks]]` · `[[serving-frameworks]]` · `[[gke-master]]` · `[[kubernetes-expert]]` ·
`[[aiml-on-kubernetes]]`
