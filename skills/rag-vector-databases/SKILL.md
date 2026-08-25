---
name: rag-vector-databases
description: Expert Retrieval-Augmented Generation (RAG) and vector-database engineering for production
  systems over large corpora. Use when building or debugging a RAG pipeline (ingestion, chunking,
  embeddings, indexing, retrieval, reranking, context assembly, generation), choosing or tuning a vector
  DB (Milvus, Qdrant, Weaviate, pgvector/AlloyDB AI, Pinecone, Vespa, Elasticsearch/OpenSearch), picking
  ANN indexes (HNSW, IVF, IVF-PQ/OPQ, ScaNN, DiskANN) and quantization (PQ/SQ/binary), implementing
  hybrid search (BM25/SPLADE + dense, RRF fusion), cross-encoder reranking, query rewriting/HyDE/multi-query/
  multi-hop/GraphRAG/contextual retrieval, metadata filtering, evaluation (recall@k, MRR, nDCG, RAGAS,
  faithfulness), or deploying a vector DB on Kubernetes/GKE (StatefulSet, sharding, replication, sizing,
  backups). Triggers on symptoms like poor recall, irrelevant chunks, hallucinated answers, slow ANN
  queries, OOM on in-memory indexes, or "cosine vs dot vs L2" distance-metric mismatch.
---

# RAG & Vector Databases

Apply the judgment of an engineer who has run production RAG over tens of millions of documents for
years: the retrieval quality ceiling, not the LLM, usually decides whether the system works. Optimize
the whole pipeline against an eval harness — never tune one stage by vibes.

## How to use this skill

1. **Read `rag-vector-databases-guide.md`** in this directory — the full reference (pipeline,
   retrieval methods, ANN internals, vector-DB selection, K8s deployment, evaluation). Apply it to the
   task at hand. For concrete artifacts to imitate (K8s StatefulSet, hybrid+rerank pipeline, HNSW/pgvector
   index config), read **`examples.md`**.
2. Match the surrounding stack's conventions (existing DB, embedding model, framework). Apply the
   correctness rules — distance-metric/embedding match, an eval harness, reranking — regardless.
3. Before declaring a RAG change "done," measure it: retrieval metrics (recall@k, nDCG) on a labeled
   set **and** end-to-end (faithfulness / answer-relevance). Report numbers, not impressions.

## Essentials (full detail in `rag-vector-databases-guide.md`)

- **The distance metric must match how the embedding model was trained.** Most modern dense models are
  trained for cosine; many are normalized so dot product == cosine. Mismatch silently wrecks recall.
  Verify the model card; normalize vectors if you use dot/IP.
- **Chunking is the highest-leverage knob.** Default to structure-aware recursive splitting on natural
  boundaries (headings, paragraphs, code blocks), not blind fixed-size cuts. Tune size/overlap against
  eval. Consider parent-document / auto-merging so you retrieve small but feed large.
- **Hybrid > dense-only for most corpora.** Combine dense (semantic) + sparse (BM25/SPLADE, exact terms,
  IDs, rare tokens) and fuse with **Reciprocal Rank Fusion (RRF)**. Dense alone misses exact matches.
- **Always rerank.** Retrieve top-k (50–200) cheaply, then re-score with a cross-encoder and keep the
  top-n (5–20) for the prompt. This is usually the single biggest precision win after hybrid.
- **HNSW is the default in-memory ANN index** (high recall, low latency, RAM-bound: vectors + graph in
  RAM). Tune `M` / `efConstruction` (build) and `efSearch` (query: recall↔latency). For huge corpora that
  exceed RAM, use IVF-PQ or **DiskANN** to trade recall/latency for memory.
- **Filter on metadata, and make it scale.** Pre-filtering is exact but can collapse the ANN graph;
  post-filtering is fast but may return too few. Use a DB with first-class filtered ANN (Qdrant, Milvus,
  Weaviate) and index the filtered fields.
- **Pick the DB for the job:** pgvector/AlloyDB AI when data already lives in Postgres and scale is
  moderate; Qdrant/Milvus/Weaviate for purpose-built scale, filtering, and hybrid; Vespa/Elasticsearch/
  OpenSearch when you need mature lexical + vector in one engine. Pinecone for fully-managed.
- **Vector DBs on K8s are stateful and RAM-bound.** Run as a `StatefulSet` with PVCs, anti-affinity,
  sized for the in-memory index (rule of thumb: vectors `n × dim × 4 bytes` + graph overhead + headroom),
  with replication for HA and a real backup/snapshot story. See `[[gke-master]]`, `[[kubernetes-expert]]`.
- **Build an eval harness before optimizing.** Labeled query→relevant-doc set for retrieval (recall@k,
  MRR, nDCG); RAGAS-style faithfulness / answer-relevance / context-precision for end-to-end. Iterate
  against numbers.
- **Watch the classic failure modes:** stale index (re-embed when the model changes), wrong distance
  metric, naive fixed chunks, no reranking, ignoring filters, oversized context that buries the answer
  ("lost in the middle"), and embedding queries with a different model/prompt than documents.
- **Advanced retrieval when basics plateau:** query rewriting/expansion, HyDE, multi-query, multi-hop,
  contextual retrieval (prepend doc context to each chunk before embedding), GraphRAG, agentic RAG.
- **Fast-moving ecosystem (2026):** embedding models, rerankers, and DB index features change monthly.
  Verify model dims/metric, index params, and DB API surface against current docs before relying on them.

## Related skills

- `[[llm-app-agent-frameworks]]` — agentic RAG, tool-calling retrieval, multi-step orchestration.
- `[[serving-frameworks]]` — serving the embedding/reranker/generator models (vLLM, Triton, KServe).
- `[[gke-master]]` — running the vector DB and embedding services on GKE (node pools, networking, storage).
- `[[kubernetes-expert]]` — StatefulSets, PVCs, scaling, backups for the vector DB.
- `[[aiml-on-kubernetes]]` — the broader AI/ML-on-K8s platform this RAG stack plugs into.
