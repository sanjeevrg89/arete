# RAG & Vector Databases — Deep Reference

The reference for building, debugging, and operating production Retrieval-Augmented Generation over
large corpora. The governing principle: **retrieval quality is the ceiling.** A frontier LLM cannot
answer from context it never received; it *can* hallucinate confidently over irrelevant chunks. Treat
RAG as an information-retrieval problem with an LLM on the end, and measure every stage.

---

## 1. Mental model: the pipeline is a funnel, measured end to end

```
                  ┌──────────── offline / ingestion ────────────┐
documents ──► parse ──► chunk ──► embed ──► index (ANN + metadata + sparse)
                                                          │
                  ┌──────────── online / query ───────────┼─────────────┐
query ──► (rewrite/expand/HyDE) ──► embed ──► ANN search ─┘──► hybrid fuse
        ──► rerank (cross-encoder) ──► assemble/pack context ──► generate ──► answer (+ citations)
```

Two independent loops:

- **Ingestion (offline, batch):** correctness here is permanent — a bad chunking or embedding choice is
  baked into the index until you re-ingest. Re-ingestion of a large corpus is expensive, so get parse /
  chunk / embedding-model / distance-metric decisions right *before* the first full build.
- **Query (online, latency-bound):** every stage adds latency and a chance to drop the right document.
  Recall is set at the ANN stage (you can't rerank a doc you didn't retrieve); precision is set by
  reranking and context assembly.

**The retrieval inequality:** `recall@k_ANN ≥ precision after rerank ≥ what the LLM actually uses.`
You lose documents at every stage. Over-retrieve early (high k), filter hard late (rerank → small n).

---

## 2. Ingestion & chunking

### Parsing

Garbage in, garbage out. PDFs, HTML, slides, and tables are where most real corpora bleed quality. Use
layout-aware extraction (preserve headings, lists, tables, code blocks); strip boilerplate (nav, headers,
footers). For tables and figures, extract structured text or a caption — embedding a mangled table as
prose poisons retrieval. Keep source metadata (URL, title, section path, page, timestamp, ACLs).

### Chunking strategies

Chunking decides what a "unit of retrieval" is. Strategies, roughly increasing in sophistication:

| Strategy | How | When |
|---|---|---|
| **Fixed-size** | N tokens, fixed overlap | Baseline only; cuts mid-sentence/mid-table. Avoid as final. |
| **Recursive / structural** | Split on a hierarchy of separators (¶, sentence, heading, code fence) down to a target size | **Default.** Respects natural boundaries; cheap; robust. |
| **Document-structure-aware** | Split by Markdown headings, HTML DOM, code AST, slide | Best for structured docs (docs sites, code, wikis). |
| **Semantic** | Embed sentences, cut where adjacent similarity drops | Coherent chunks for prose; costs embeddings at ingest; tune the breakpoint threshold. |
| **Parent-document / auto-merging** | Index small child chunks; at retrieval return the parent (or merge adjacent siblings) | Retrieve precisely, feed broad context. Very strong default. |
| **Contextual retrieval** | Prepend an LLM-generated doc-level summary/context to each chunk *before* embedding | Fixes "this chunk is meaningless without its document." Adds ingest cost; large recall gains on ambiguous corpora. |

**Size & overlap tradeoffs.** Smaller chunks → higher precision, more chunks, more index, risk of losing
context the answer needs. Larger chunks → more context per hit but diluted embeddings (one vector
averaging many topics → worse recall) and wasted prompt tokens. There is no universal number; common
starting points are a few hundred tokens with 10–20% overlap, **then tune against eval.** Overlap exists
to avoid cutting an answer across a boundary; too much overlap inflates the index and creates
near-duplicate hits.

**Failure modes:** mid-sentence/mid-table cuts; chunks too large (diluted vectors); chunks too small
(missing context); no overlap (answers split across the seam); chunk text differing from what you embed.

---

## 3. Embeddings

### Choosing a model

A dense embedding maps text → a vector in `R^dim` where semantic similarity ≈ geometric proximity. Choose by:

- **Task & domain:** general vs. code vs. multilingual vs. domain-specific. Consult current public
  retrieval leaderboards (e.g. MTEB-style) — but **validate on your own corpus**, leaderboards don't
  reflect your data. Verify against current docs; rankings churn monthly.
- **Dimension:** higher dim ≈ more capacity but more RAM/latency. Some models support **Matryoshka
  (MRL)** — truncate the vector to a smaller dim with graceful quality loss; useful to cut index size.
- **Max sequence length:** must exceed your chunk size or the tail is silently truncated.
- **Asymmetric models** use different prompts/prefixes for queries vs. documents (e.g. `query:` /
  `passage:`). Using the wrong prefix — or none — quietly tanks recall. Read the model card.
- **Distance metric** the model was trained for (§5). This is not optional.

### Serving embeddings

- **Batch ingestion** for throughput; embedding millions of chunks is the dominant ingest cost. Use a
  GPU-served embedding endpoint (vLLM / TEI / Triton — see `[[serving-frameworks]]`) with large batches.
- **Cache** query embeddings and re-use; embed queries with the *same model and version* as documents.
- **Versioning:** pin the model+version per index. Changing the embedding model **requires a full
  re-embed and re-index** — query and doc vectors from different models are not comparable.

**Failure modes:** query/document model mismatch; missing asymmetric prefix; chunk longer than model
context (silent truncation); mixing two embedding-model versions in one index (the #1 cause of a
mysteriously broken index after an "upgrade").

---

## 4. Retrieval methods

### Dense vector search

Embed the query, ANN-search the index for nearest neighbors (§6). Strong on paraphrase/semantic match;
**weak on exact terms** — IDs, SKUs, error codes, rare proper nouns, exact quotes — because those get
averaged into the embedding.

### Sparse retrieval

- **BM25** — classic lexical/term-frequency ranking. Exact, interpretable, no training, excellent on
  keywords, IDs, and rare tokens. The baseline you must beat, and often the half that saves dense search.
- **Learned sparse (SPLADE)** — a neural model produces a sparse term-weight vector with term expansion;
  combines lexical exactness with some semantic generalization. Heavier to compute/index than BM25.

### Hybrid search + fusion

Run dense and sparse, then merge. **Reciprocal Rank Fusion (RRF)** is the robust default — it fuses by
rank, not raw scores (which live on incomparable scales):

```
RRF(d) = Σ_retrievers  1 / (k + rank_retriever(d))     # k ≈ 60 by convention
```

RRF needs no score normalization and no per-corpus weight tuning, which is why it's the go-to. Weighted
score fusion (normalize then linearly combine) can edge it out but requires tuning. Hybrid beats
dense-only on the large majority of real corpora — adopt it as the default, not an optimization.

### Metadata filtering

Restrict by `source`, `date`, `tenant`, `lang`, ACLs, etc. Three implementation strategies, each with a trap:

- **Pre-filter** (filter, then ANN over the subset): exact results, but a restrictive filter can shred
  the HNSW graph's connectivity → poor recall or slow fallback to brute force.
- **Post-filter** (ANN, then drop non-matching): fast, but a selective filter can leave far fewer than k
  results ("k underflow").
- **Filtered/integrated ANN** (the index is filter-aware): Qdrant, Milvus, Weaviate do this well. **Index
  the fields you filter on.** At scale, this is the difference between sub-100ms and a timeout.

### Reranking (cross-encoders)

Bi-encoder retrieval (separate query/doc embeddings, cosine) is cheap but coarse. A **cross-encoder**
jointly encodes (query, document) and outputs a precise relevance score — far more accurate, far more
expensive, so you can't run it over the whole corpus. The canonical two-stage pattern:

```
ANN/hybrid: retrieve top-k (50–200, recall-oriented)  →  cross-encoder rerank  →  keep top-n (5–20)
```

Reranking is typically the **largest precision win after hybrid** and it lets you retrieve aggressively
(high k) without dumping junk into the prompt. Serve the reranker on GPU (`[[serving-frameworks]]`);
batch the (query,doc) pairs.

### Advanced retrieval (when the basics plateau)

- **Query rewriting / expansion** — clean up conversational/underspecified queries; expand with synonyms.
- **HyDE** (Hypothetical Document Embeddings) — have an LLM draft a hypothetical answer, embed *that*,
  and search with it; the answer's vector often lands closer to real passages than the bare question.
- **Multi-query** — generate several query variants, retrieve for each, fuse (RRF). Improves recall.
- **Multi-hop** — for questions needing chained facts, retrieve → reason → retrieve again.
- **GraphRAG** — build an entity/relationship graph from the corpus; retrieve subgraphs / community
  summaries. Strong for global "summarize across the whole corpus" and connected-fact questions; heavy
  to build and maintain.
- **Agentic RAG** — an LLM agent decides when/what/how to retrieve, calls retrieval as a tool, and
  iterates. See `[[llm-app-agent-frameworks]]`. Powerful but adds latency and failure surface; gate it.

**Failure modes:** dense-only on a keyword-heavy corpus; no reranking; k too small (right doc never
retrieved); filter applied wrong (underflow / graph collapse); HyDE/multi-query latency without measuring
that it helps.

---

## 5. Distance metrics (get this right or nothing works)

| Metric | Meaning | Use when |
|---|---|---|
| **Cosine** | Angle; magnitude-invariant | Most dense text models. Default unless the card says otherwise. |
| **Dot / inner product (IP)** | Angle × magnitude | Models trained for IP, or normalized vectors (then IP ≡ cosine). |
| **L2 (Euclidean)** | Straight-line distance | Some image/specialized models. |

**The rule:** use the metric the embedding model was *trained* with — it's on the model card. A
cosine-trained model queried with L2 returns plausible-but-wrong neighbors and **degrades silently** (no
error, just bad recall). If you normalize vectors to unit length, dot product equals cosine and is
faster — a common, valid optimization. **Configure the index's metric to match at creation time;** many
DBs can't change it without a rebuild. This single mismatch is one of the most common production RAG bugs.

---

## 6. ANN index algorithms & quantization

Exact nearest-neighbor search is O(n) per query — infeasible at scale. **Approximate** NN trades a little
recall for orders-of-magnitude speedup. The eternal triangle: **recall ↔ latency ↔ memory.** You pick two.

### HNSW (Hierarchical Navigable Small World) — the default

A multi-layer proximity graph; search greedily descends layers to the nearest neighborhood. High recall,
low latency, fast build — but **the entire graph + vectors live in RAM** (RAM-bound). Key params
(conceptual; names vary slightly by library):

- **`M`** — edges per node. Higher → better recall + more memory. Typical 16–64.
- **`efConstruction`** — candidate breadth at build. Higher → better graph quality, slower build.
- **`efSearch`** (a.k.a. `ef`) — candidate breadth at query. **The runtime recall↔latency dial:** raise
  for recall, lower for speed. Tune this per latency budget against eval.

### IVF (Inverted File)

Cluster vectors into `nlist` centroids; at query, probe the nearest `nprobe` clusters. Lower memory than
HNSW, but recall depends on `nprobe` (more probes → higher recall, slower) and on a representative
training set. Good when HNSW's RAM cost is prohibitive.

### Quantization (shrink the vectors)

- **PQ (Product Quantization)** — split each vector into sub-vectors, quantize each to a codebook;
  huge memory savings (often >10×), some recall loss. **OPQ** rotates first for a better fit.
- **SQ (Scalar Quantization)** — e.g. float32 → int8; ~4× smaller, small recall hit, simple.
- **Binary quantization** — 1 bit/dim; massive compression + Hamming-distance speed, larger recall hit;
  usually paired with a re-ranking/rescoring pass over full vectors to recover precision.

**IVF-PQ / OPQ** (IVF coarse partition + PQ-compressed residuals) is the workhorse for billion-scale
in-RAM-constrained indexes. A common high-quality pattern: quantized index for the fast first pass,
then **rescore** the top candidates with full-precision vectors.

### Disk-based & specialized

- **DiskANN** — graph index that keeps most data on SSD with a RAM cache; serves far larger-than-RAM
  indexes at the cost of latency. The answer when the corpus genuinely can't fit in RAM.
- **ScaNN** — anisotropic-quantization-based ANN, strong recall/latency on large datasets.

### Choosing

| Situation | Index |
|---|---|
| Fits in RAM, want best recall/latency | **HNSW** |
| Too big for full-precision RAM | **IVF-PQ / OPQ** (+ rescore) |
| Far larger than RAM | **DiskANN** |
| Billion-scale, max recall/$ | **ScaNN** / IVF-PQ |

**Sizing rule of thumb (full-precision HNSW):** RAM ≈ `n_vectors × dim × 4 bytes` (the vectors) + graph
overhead (roughly `n × M × ~8 bytes`) + process/OS headroom. 100M × 768-dim float32 ≈ ~300GB *just for
vectors* — at which point quantization or DiskANN stops being optional. Verify exact overheads against
your DB's current docs.

---

## 7. Vector databases

### The contenders

- **Milvus** — purpose-built, distributed; multiple index types (HNSW, IVF-PQ, DiskANN, GPU indexes),
  separates compute/storage, scales to billions. Heavier to operate; strong for very large scale.
- **Qdrant** — Rust, ergonomic; excellent **filtered ANN** and payload indexing, hybrid, quantization
  built in. Great default for filtered/metadata-heavy workloads; easy to run on K8s.
- **Weaviate** — schema/object model, built-in hybrid (BM25 + dense) and modules; multi-tenancy.
- **pgvector** (Postgres extension) / **AlloyDB AI** — vectors *inside* Postgres: keep relational data,
  joins, transactions, and one operational surface. HNSW & IVFFlat indexes. **Best choice when your data
  already lives in Postgres** and scale is moderate; AlloyDB AI adds managed scaling and optimized
  vector indexing for larger workloads. Verify current index/feature support against docs.
- **Pinecone** — fully managed, serverless; you trade control for zero ops.
- **Vespa** — heavyweight engine combining tensor/vector + lexical + ranking; excellent for complex
  ranking and hybrid at scale; steeper learning curve.
- **Elasticsearch / OpenSearch** — mature lexical (BM25) + added kNN/HNSW vector search. Compelling when
  you already run them and want lexical + vector in one place; vector ergonomics trail purpose-built DBs.

### How to pick

| Need | Lean toward |
|---|---|
| Data already in Postgres, moderate scale, transactional joins | **pgvector / AlloyDB AI** |
| Heavy metadata filtering, hybrid, easy ops | **Qdrant** |
| Very large scale (billions), tunable index zoo | **Milvus** |
| Schema/objects + built-in hybrid + multi-tenancy | **Weaviate** |
| Mature lexical + vector in one engine you already run | **Elasticsearch / OpenSearch / Vespa** |
| Zero ops, managed | **Pinecone** |

### Dimensions that actually differentiate them

- **Filtering:** quality of *filtered* ANN (pre/post/integrated) varies enormously — test on your filters.
- **Consistency:** most vector DBs are eventually consistent on inserts (a just-upserted vector may not be
  immediately searchable). pgvector inherits Postgres transactional semantics. Know your read-after-write needs.
- **Hybrid:** native BM25/sparse + dense + fusion vs. bolt-it-on-yourself.
- **Scale model:** sharding/replication, compute-storage separation, multi-tenancy isolation.
- **Quantization & index options:** what the engine supports natively.

---

## 8. Deploying a vector DB on Kubernetes / GKE

Vector DBs are **stateful and (for in-memory indexes) RAM-bound** — the two facts that drive the design.
See `[[kubernetes-expert]]` and `[[gke-master]]`.

- **`StatefulSet`, not Deployment.** Stable network identity + stable per-pod storage via
  `volumeClaimTemplates`. Use a headless `Service` for peer discovery.
- **Persistent storage:** fast SSD-backed `StorageClass` (e.g. SSD persistent disks / local SSD on GKE).
  For DiskANN-style on-disk indexes, storage IOPS/latency directly bound query latency.
- **Sharding & replication:** shard to scale data/throughput beyond one node; replicate each shard for HA
  and read scale. Spread replicas across zones/nodes with `podAntiAffinity` + topology spread.
- **Resource sizing:** size **RAM for the in-memory index** (§6 sizing rule) plus working set and OS page
  cache — under-provisioned RAM → OOMKill mid-build or thrash. Pin CPU for build/search; set requests=limits
  for memory to avoid eviction surprises. Distinguish ingest (CPU/throughput-heavy) from serve (latency).
- **Scaling:** these are stateful — naive HPA on a `StatefulSet` rebalances/reshards data, which is
  expensive and not instant. Plan capacity; scale deliberately. Embedding/reranker *services* are
  stateless and scale normally (HPA / KEDA — see `[[autoscaling-kubernetes]]`, `[[serving-frameworks]]`).
- **Backups:** use the DB's native snapshot/backup (collection snapshots, segment backups), not just a
  PVC snapshot mid-write (which can capture a torn index). Test restore. Re-embeddable source-of-truth
  corpus is your ultimate backup, but re-ingesting millions of docs is slow — keep DB snapshots too.
- **Operators/Helm:** most of these ship a Helm chart and/or operator — prefer them over hand-rolled
  manifests, but read what they configure (anti-affinity, PDBs, resources) and override for your sizing.
- **Co-locate or not:** embedding/reranker/LLM serving (GPU) vs. vector DB (RAM/CPU) want different node
  pools. Use separate GKE node pools and `nodeSelector`/taints. See `[[aiml-on-kubernetes]]`.

---

## 9. Context assembly / packing

Retrieved-and-reranked chunks still have to be turned into a prompt:

- **Order matters — "lost in the middle."** LLMs attend best to the start and end of long contexts.
  Put the most relevant chunks at the edges; don't bury the key passage in the middle of a huge context.
- **Don't overfill.** More context is not better — it dilutes attention, costs tokens/latency, and raises
  distraction/hallucination risk. A tight set of highly-relevant chunks beats a fat one. This is why
  rerank → small n exists.
- **Deduplicate** overlapping/near-duplicate chunks (overlap and multi-query create them).
- **Carry citations:** include source IDs/URLs with each chunk and instruct the model to cite, enabling
  attribution and verification.
- **Prompt the model to abstain** ("if the context doesn't contain the answer, say so") — the main lever
  against hallucination when retrieval misses.

**Failure modes:** context window stuffed to the brim; key chunk in the dead middle; duplicate chunks
crowding out diversity; no citations; no abstain instruction (model invents an answer).

---

## 10. Evaluation — the part that's usually skipped and shouldn't be

You cannot improve what you don't measure, and RAG has two layers to measure.

### Retrieval metrics (need a labeled query → relevant-doc set)

- **Recall@k** — fraction of relevant docs retrieved in the top k. The ceiling: if it's not in top-k,
  rerank and the LLM can't fix it. Watch this first.
- **Precision@k** — fraction of top-k that are relevant.
- **MRR** (Mean Reciprocal Rank) — rank of the first relevant doc; good when one right answer.
- **nDCG@k** — rank-weighted, graded relevance; the standard when ranking quality matters.

Build the labeled set from real queries (logs) + human or strong-LLM judgments. A few hundred labeled
queries already drives most decisions.

### End-to-end metrics (RAGAS-style)

- **Faithfulness / groundedness** — is the answer supported by the retrieved context? (hallucination check)
- **Answer relevance** — does it address the question?
- **Context precision / recall** — was the retrieved context relevant and sufficient?

Frameworks like **RAGAS** automate these with an LLM-as-judge; treat the numbers as directional and
spot-check with humans. Verify the framework's current metric definitions/API against its docs.

### How to iterate

1. Freeze an eval set (retrieval + end-to-end). 2. Change **one** variable (chunking, model, k,
reranker, fusion). 3. Re-run, compare. 4. Keep what moves the metric, revert what doesn't. Log every run.
Without this loop you are tuning by vibes and will regress silently.

---

## 11. Anti-patterns (the traps that bite in production)

- **Naive fixed-size chunking** as the final strategy — cuts answers, dilutes vectors. Use structural/recursive.
- **No reranking** — leaving the biggest precision win on the table; dumping coarse ANN hits into the prompt.
- **Distance-metric mismatch** — cosine model with L2 index (or unnormalized vectors with IP). Silent recall death.
- **Dense-only retrieval** on keyword/ID/code-heavy corpora — add sparse + RRF.
- **No eval harness** — optimizing by anecdote; can't tell improvement from regression.
- **Stale index** — corpus changed but not re-indexed; or embedding model upgraded without re-embedding
  (query and doc vectors now incompatible).
- **Ignoring metadata filters at scale** — unindexed filter fields → full scans/timeouts; or post-filter underflow.
- **Over-stuffed context** — max-filling the window, burying the answer ("lost in the middle"), paying
  tokens for noise.
- **Mixing embedding-model versions** in one index — the subtle "everything got worse after the upgrade" bug.
- **Query embedded differently than documents** — missing asymmetric prefix, different model/version.
- **No abstain path** — model hallucinates instead of saying "not in context."

---

## 12. Troubleshooting (symptom → likely cause → fix)

| Symptom | Likely cause | Fix |
|---|---|---|
| Right doc never appears in results | Recall@k low: k too small, wrong metric, bad chunking, dense-only | Raise k; verify metric/normalization; add hybrid; fix chunking; check recall@k |
| Relevant docs retrieved but answer is wrong | No reranking / poor context order / over-stuffed | Add cross-encoder rerank; put key chunks at edges; shrink n |
| Exact IDs/codes/quotes not found | Dense-only averages them out | Add BM25/SPLADE + RRF |
| Quality dropped after "upgrading" the model | Mixed embedding versions in one index | Full re-embed + reindex; pin version per index |
| Hallucinated answers | Retrieval miss + no abstain | Improve recall; instruct abstain; measure faithfulness |
| Slow queries / timeouts | efSearch/nprobe too high, no filter index, index > RAM (swapping) | Tune efSearch/nprobe; index filter fields; quantize or move to DiskANN; add RAM |
| OOMKilled pod | In-memory index exceeds RAM | Quantize (PQ/SQ/binary), shard, or size RAM via §6 rule; use DiskANN |
| Filtered queries return too few results | Post-filter underflow | Use integrated filtered ANN; raise k pre-filter; index the field |
| Just-inserted doc not searchable | Eventual-consistency indexing lag | Wait for index flush / use the DB's consistency knob; pgvector is transactional |

---

## 13. Version awareness (2026)

This stack moves monthly. **Verify against current docs before relying on specifics:**
embedding-model dimensions, max length, asymmetric prefixes, and trained distance metric; reranker model
availability; vector-DB index types, quantization options, filtering semantics, and consistency knobs;
RAGAS metric definitions and APIs; pgvector/AlloyDB AI index feature support. Retrieval *principles*
(hybrid, rerank, metric-match, eval-driven iteration) are stable; the knobs and APIs are not.

---

## Rationalizations & rebuttals

The excuses for skipping the right thing, each rebutted from the guide above:

- *"Naive fixed-size chunking is fine to ship."* It cuts answers mid-sentence/mid-table and dilutes
  vectors (one embedding averaging many topics → worse recall). It's a baseline, not a final strategy —
  use recursive/structural and tune against eval (§2).
- *"Skip reranking, the ANN hits are good enough."* Reranking is typically the largest precision win
  after hybrid. Bi-encoder ANN is coarse; a cross-encoder jointly scores (query, doc) and lets you
  over-retrieve (high k) without dumping junk into the prompt (§4).
- *"Cosine vs dot vs L2 doesn't really matter."* It does, and it fails silently — a cosine-trained model
  queried with L2 returns plausible-but-wrong neighbors with no error. Use the metric on the model card;
  the index metric is often fixed at creation (§5).
- *"No eval needed, the answers look good."* Looking good on a handful of queries is tuning by vibes;
  you can't distinguish improvement from regression. A few hundred labeled queries (recall@k, nDCG)
  drives most decisions and catches silent regressions (§10).
- *"Vector search alone is enough; skip hybrid."* Dense search averages out exact terms — IDs, SKUs,
  error codes, rare proper nouns, quotes. Hybrid (dense + BM25/SPLADE fused with RRF) beats dense-only
  on the large majority of real corpora; it's the default, not an optimization (§4).
- *"We'll index the corpus once and leave it."* The corpus drifts and the index goes stale; worse, an
  embedding-model "upgrade" without a full re-embed leaves query/doc vectors incompatible — the classic
  "everything got worse after the upgrade" bug (§3, §11).
- *"Metadata filters are just a flag, no need to index those fields."* At scale, unindexed filter fields
  → full scans/timeouts, and naive post-filtering underflows below k. Use integrated filtered ANN and
  index what you filter on (§4).

---

## Red flags (stop and reconsider)

- **No reranker** in the pipeline — coarse ANN hits go straight into the prompt; the biggest precision
  win is unclaimed.
- **Distance metric doesn't match the embedding model** (cosine model on an L2 index, or unnormalized
  vectors with inner product) — degrades silently, no error.
- **Dense-only retrieval** on a keyword/ID/code-heavy corpus — no sparse/BM25 leg, no RRF fusion.
- **Stale index** — corpus changed but not re-indexed, or embedding model swapped without a full
  re-embed (incompatible vectors), or two embedding-model versions mixed in one index.
- **No retrieval eval** — no labeled query set, no recall@k/nDCG; changes are accepted by anecdote.
- **No end-to-end / faithfulness measurement** — nobody checks whether answers are grounded in the
  retrieved context.
- **Metadata filter fields unindexed at scale** — full scans, timeouts, or post-filter k-underflow.
- **Context window stuffed to the brim** — key passage buried in the dead middle ("lost in the middle"),
  no dedup of overlapping chunks, no citations, no abstain instruction.

---

## Verification gate (definition of done)

Before the RAG system counts as done, confirm:

- [ ] **Chunking justified** — strategy chosen against the corpus (structural/recursive or better, not
  naive fixed-size as final); size/overlap tuned against eval, not guessed.
- [ ] **Hybrid + rerank in place** — dense + sparse (BM25/SPLADE) fused (RRF), followed by a
  cross-encoder rerank to a small top-n. Not dense-only, not unranked.
- [ ] **Distance metric matches the embeddings** — index metric == the metric on the model card (or
  vectors normalized and using inner product); set at index creation. Same model+version for queries and
  documents, with correct asymmetric prefixes.
- [ ] **Retrieval metrics measured** — recall@k and nDCG@k (plus MRR/precision@k as relevant) on a
  frozen labeled query set; recall@k is high enough that the right doc is reliably in top-k.
- [ ] **End-to-end faithfulness evaluated** — groundedness/faithfulness, answer relevance, and
  context precision/recall measured (RAGAS-style, spot-checked by humans); abstain path verified.
- [ ] **Index freshness handled** — re-ingestion/update path for corpus changes; embedding-model version
  pinned per index with a full-re-embed plan; consistency/read-after-write needs understood; filter
  fields indexed; backups (native snapshots) tested via restore.

---

## 14. Canonical references (real URLs only)

- HNSW paper — Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs": https://arxiv.org/abs/1603.09320
- Product Quantization — Jégou et al.: https://ieeexplore.ieee.org/document/5432202
- DiskANN — https://github.com/microsoft/DiskANN
- ScaNN — https://github.com/google-research/google-research/tree/master/scann
- SPLADE — https://github.com/naver/splade
- HyDE — "Precise Zero-Shot Dense Retrieval without Relevance Labels": https://arxiv.org/abs/2212.10496
- "Lost in the Middle" — Liu et al.: https://arxiv.org/abs/2307.03172
- RRF — Cormack et al.: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- MTEB embedding leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- RAGAS — https://github.com/explodinggradients/ragas
- Milvus — https://milvus.io/docs · Qdrant — https://qdrant.tech/documentation/
- Weaviate — https://weaviate.io/developers/weaviate · pgvector — https://github.com/pgvector/pgvector
- AlloyDB AI — https://cloud.google.com/alloydb/docs/ai · Vespa — https://docs.vespa.ai/
- Faiss (index/quantization reference) — https://github.com/facebookresearch/faiss/wiki
