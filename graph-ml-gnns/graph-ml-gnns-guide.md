# Graph ML & GNNs — The Guide

The deep reference for shipping graph machine learning in production. Read end to end the first time;
thereafter jump to the section you need. The two questions that decide a project's fate appear first
(do you need a graph; will it scale) and recur throughout.

---

## 1. Mental model: when graph structure matters

A graph is data where the **relationships between entities carry signal** beyond the entities'
own features. Formally a graph `G = (V, E)` with nodes `V`, edges `E`, optional node features `X ∈
R^{|V|×d}`, edge features, and (for heterogeneous graphs) node/edge **types**.

You should consider a graph model when at least one holds:

- **Homophily / relational dependence:** a node's label correlates with its neighbors' labels or
  features (fraud rings, citation topics, social communities). Plain i.i.d. models throw this away.
- **Multi-hop structure:** the useful signal lives 2–3 hops out (a buyer connected to a flagged seller
  through a shared device/payment instrument).
- **Structural roles:** the *shape* of a node's neighborhood matters (a money-laundering "mule" has a
  characteristic fan-in/fan-out), not just feature values.
- **New entities / inductive need:** you must score nodes unseen at training time and want to exploit
  their connections.

**Earn the graph.** The honest baseline is a gradient-boosted-trees or MLP model on node features
*plus hand-engineered neighborhood aggregates* (count of neighbors, mean neighbor feature, degree,
PageRank, triangle count). If that baseline isn't beaten by a GNN, you don't have a graph problem —
you have a tabular problem with extra infrastructure cost. This is the single most common GNN
anti-pattern. Run the baseline first, always.

### Problem types

| Task | Predict | Examples | Typical decoder/output |
| --- | --- | --- | --- |
| **Node classification/regression** | a label per node | fraud user, paper topic, customer segment | softmax/regression head on node embedding |
| **Link prediction** | does/should an edge exist | recommendation, KG completion, drug–target | score(emb_u, emb_v): dot, DistMult, MLP |
| **Graph classification/regression** | a label per *whole graph* | molecular property, toxicity, solubility | readout (pool nodes) → head |
| **Community detection / clustering** | node partitions | segmentation, dedup, structure discovery | unsupervised embeddings + clustering, or modularity |
| **Heterogeneous / KG learning** | typed nodes/edges, relations | knowledge graphs, e-commerce graphs | relation-aware GNN + typed decoder |

Transductive tasks score nodes present (unlabeled) at training time; inductive tasks score genuinely
new nodes/graphs. **This distinction governs both architecture and evaluation** (Sections 7, 8).

---

## 2. Foundations: representation, embeddings, and message passing

### Graph representation

- **Adjacency**: dense `A ∈ R^{n×n}` only for small graphs; production uses **sparse** formats —
  edge lists / CSR / COO. PyG stores `edge_index ∈ Z^{2×|E|}` (COO); DGL uses a `DGLGraph` backed by
  CSR/CSC structures.
- **Features** live on nodes (`x`) and optionally edges (`edge_attr`). Normalize and impute exactly as
  you would for tabular features; missing-feature handling matters because GNNs propagate it.
- **Degree skew is real.** Power-law graphs have hub nodes with millions of edges. Hubs dominate
  aggregation, blow up sampling fan-out, and cause memory spikes. Plan for them (sampling caps,
  degree normalization, hub-specific handling).

### Two paradigms: shallow embeddings vs end-to-end GNNs

- **Shallow node embeddings** — **DeepWalk** (Perozzi et al., 2014) and **node2vec** (Grover &
  Leskovec, 2016) run random walks and train skip-gram (word2vec-style) so co-occurring nodes get
  similar vectors. **Strengths:** simple, strong unsupervised baseline, capture community structure
  cheaply. **Hard limits:** *transductive* (one vector per node — no way to embed a new node without
  retraining), cannot use node features, no parameter sharing. Use as a baseline or a feature, not as
  the production model when entities churn.
- **GNNs** learn a parameterized function of features + structure, so they are **inductive** and
  **feature-aware** by construction. This is why GraphSAGE and successors replaced node2vec in most
  production stacks.

### The message-passing framework (the core abstraction)

Almost every modern GNN is a **Message-Passing Neural Network** (Gilmer et al., 2017). One layer
updates each node `v` from its 1-hop neighborhood `N(v)`:

```
m_v   = AGGREGATE_{u∈N(v)} ( MESSAGE(h_u, h_v, e_uv) )     # permutation-invariant: sum/mean/max/attn
h_v'  = UPDATE( h_v, m_v )                                  # combine self + aggregated message
```

- **Stacking L layers** gives each node a receptive field of L hops. Depth = hops, not "capacity."
- **AGGREGATE must be permutation-invariant** (a graph has no canonical neighbor order). The *choice*
  of aggregator determines expressivity (Section 4).
- The whole architecture zoo (Section 3) is just different `MESSAGE`/`AGGREGATE`/`UPDATE` choices.
  Absorb this and you stop memorizing models.

---

## 3. GNN architectures

Pick by constraint (inductive? heterogeneous? long-range? expressivity?), not by leaderboard rank.

### GCN — Graph Convolutional Network (Kipf & Welling, ICLR 2017, arXiv:1609.02907)
Mean-style aggregation with symmetric normalization: `H' = σ( Â H W )` where `Â = D̃^{-1/2} (A+I)
D̃^{-1/2}`. Simple, fast, strong transductive baseline. **Full-batch and transductive in its vanilla
form** — it normalizes over the whole graph, so it does not naturally handle new nodes or scale to
huge graphs without modification. Start here for a small/medium static graph; treat its number as the
bar to beat.

### GraphSAGE (Hamilton, Ying & Leskovec, NeurIPS 2017, arXiv:1706.02216)
The production default. Two ideas that matter:
1. **Inductive**: learns aggregator functions, not per-node vectors, so it embeds unseen nodes.
2. **Neighbor sampling**: at each layer sample a fixed number of neighbors (fan-out, e.g. `[15, 10]`
   for 2 layers) instead of using all of them → bounded compute/memory per node, mini-batch training,
   and the basis for scaling (Section 5). Aggregators: mean, max-pool, LSTM (mean/max in practice).
If you are unsure which architecture to use on a large or dynamic graph, the answer is usually
GraphSAGE with neighbor sampling.

### GAT — Graph Attention Network (Veličković et al., ICLR 2018, arXiv:1710.10903)
Learns **per-edge attention weights** so a node weights neighbors differently; multi-head for
stability. Helps when neighbor importance is heterogeneous and you can afford the extra compute. GATv2
(Brody et al., 2021, arXiv:2105.14491) fixes the "static attention" limitation of the original — prefer
GATv2 in new code; verify current API names.

### GIN — Graph Isomorphism Network (Xu et al., ICLR 2019, arXiv:1810.00826)
Designed for **maximal expressivity** among message-passing GNNs: a **sum** aggregator with an MLP
update is injective, matching the **1-dimensional Weisfeiler–Lehman (1-WL)** graph-isomorphism test —
the theoretical ceiling for standard MPNNs. Strong on graph classification (molecules) where
distinguishing structures matters. Key takeaway: **sum > mean > max** for distinguishing-power;
mean/max lose multiset information.

### R-GCN — Relational GCN (Schlichtkrull et al., ESWC 2018, arXiv:1703.06103)
For **heterogeneous / knowledge graphs**: a separate weight matrix per relation type, with
basis/block decomposition to control the parameter blow-up when relations are many. The canonical
starting point for KG node classification and entity-level tasks with typed edges.

### Heterogeneous graph transformers & metapath models
**HGT** (Hu et al., WWW 2020, arXiv:2003.01332) uses type-aware attention (per node-type/edge-type
projections) and scales to large heterogeneous graphs. **HAN** uses metapath-based attention. Use these
when you have many node/edge types and relation semantics matter. PyG (`HeteroData`,
`to_hetero`) and DGL both have first-class heterogeneous support.

### Graph transformers
Apply transformer-style global attention over nodes with structural/positional encodings (e.g.
**Graphormer**, Ying et al., NeurIPS 2021, arXiv:2106.05667; and the **GPS** general framework,
Rampášek et al., NeurIPS 2022, arXiv:2205.12454). They mitigate **over-squashing** by giving every node
direct access to every other, at `O(n²)` attention cost — practical mainly on **small graphs**
(molecules), not million-node graphs without linear-attention/sparsification tricks. Reach for them
when long-range interactions dominate and the graph is small enough.

### Depth, over-smoothing, over-squashing, expressivity
- **Over-smoothing**: stacking many layers makes all node representations converge to a similar value
  (repeated averaging is a low-pass filter), collapsing class separation. Symptom: deeper model →
  *worse* accuracy and near-identical embeddings.
- **Over-squashing** (Alon & Yahav, 2021, arXiv:2006.05205): information from an exponentially growing
  receptive field is compressed into fixed-size vectors through graph bottlenecks, so distant signal
  can't propagate. Adding layers makes this worse, not better.
- **Mitigations**: keep it shallow (2–3 layers is usually optimal); residual/skip connections;
  **Jumping Knowledge** (Xu et al., 2018) to combine representations from multiple depths; PairNorm /
  normalization; graph rewiring for over-squashing; or a graph transformer for genuine long-range needs.
- **Expressivity ceiling**: standard MPNNs are bounded by **1-WL** — they cannot distinguish certain
  non-isomorphic structures (e.g. some regular graphs, triangle counts). If your task needs that
  (counting substructures, some molecular properties), you need higher-order GNNs, positional/
  structural encodings, or subgraph-based methods. Know the ceiling before blaming your hyperparameters.

---

## 4. Aggregator choice and expressivity, concretely

| Aggregator | Distinguishing power | Use when |
| --- | --- | --- |
| **sum** | highest (injective on multisets) | graph classification, structure matters (GIN) |
| **mean** | loses multiset size/count info | homophilous node tasks, scale-invariant neighborhoods |
| **max** | keeps only the extremum | salient-neighbor signals, robustness to count |
| **attention** | learned, data-dependent | heterogeneous neighbor importance (GAT/GATv2/HGT) |

Rule of thumb: **node classification on homophilous graphs → mean/GCN/SAGE; graph classification where
structure is the label → sum/GIN; heterogeneous importance → attention.**

---

## 5. Scalability — the production hard part

Accuracy rarely kills GNN projects; **memory and throughput do.** A full-batch GCN forward pass
materializes activations for *every node at every layer* — `O(L · |V| · d)` — plus the sparse adjacency.
Fine on OGB-arxiv (~170K nodes); impossible on a billion-edge graph on one GPU.

### Training strategies (choose by graph size)

| Strategy | How it works | Scales to | Watch out for |
| --- | --- | --- | --- |
| **Full-batch** | whole graph per step (GCN-style) | ~10⁵–10⁶ nodes on a big GPU | OOM; can't mini-batch |
| **Neighbor sampling** (GraphSAGE) | sample fan-out neighbors per layer, build per-batch computation graph | very large, dynamic graphs | exponential blow-up of subgraph with depth × fan-out; redundant computation across batches |
| **Cluster-GCN** (Chiang et al., KDD 2019, arXiv:1905.07953) | partition graph into clusters (METIS), train on a few clusters per batch | large static graphs | cluster boundaries drop cross-cluster edges → bias; needs random cluster combination |
| **GraphSAINT** (Zeng et al., ICLR 2020, arXiv:1907.04931) | sample a *subgraph* (node/edge/random-walk), train full GNN on it with normalization | large graphs | sampler choice and normalization matter for unbiasedness |
| **Historical embeddings** (e.g. GAS / GNNAutoScale, arXiv:2106.05609) | cache stale neighbor embeddings to avoid recomputing the full neighborhood | deep models on large graphs | staleness bias; cache memory |

**Neighbor-sampling math you must respect:** with L layers and fan-out `f`, each seed pulls up to
`f^L` nodes. `[15,10]` over 2 hops ≈ 150 nodes/seed; a 3rd hop at `f=10` ≈ 1500. **Fan-out × depth is
the memory knob** — this is why depth is doubly punished (over-smoothing *and* blow-up). On power-law
graphs, cap neighbors of hubs.

### Distributed training & partitioning
- **Partition the graph** (e.g. METIS / DGL's partitioning) so each worker owns a subset of nodes and
  their local edges; minimize **edge cut** to reduce cross-worker communication. Cross-partition
  neighbors require remote feature fetches — the dominant cost at scale.
- **DGL** has a mature distributed stack (`DistGraph`, distributed samplers, partition tooling); **PyG**
  offers distributed/remote-backend support and integrates with samplers. Verify current APIs and
  recommended topologies against the framework docs — both move fast.
- Bottleneck is usually **feature/sampling I/O**, not GPU compute. Co-locate features with partitions,
  cache hot nodes, overlap sampling with compute. Run training infra on `[[aiml-on-kubernetes]]`.

### Serving GNNs (the underrated problem)
- **Precompute + serve embeddings**: for slow-changing graphs, run the GNN as a batch job, write node
  embeddings to a store, and serve them to a downstream ranker/classifier. Simplest and most common —
  pairs naturally with `[[recsys-ranking]]` retrieval.
- **Real-time inductive inference**: for fresh nodes, sample the k-hop subgraph at request time and run
  the GNN online — needs a low-latency graph store and bounded fan-out. Latency = sampling + feature
  fetch + forward; the graph fetch usually dominates.
- Keep **train/serve neighbor-sampling parity** (same fan-out semantics, same feature transforms) or
  you get train/serve skew. See `[[data-engineering-feature-stores]]`.

---

## 6. Tooling

- **PyTorch Geometric (PyG)** — the most widely used; Pythonic API, huge model/dataset zoo,
  `Data`/`HeteroData`, `NeighborLoader`/`LinkNeighborLoader`, `MessagePassing` base class. Best default
  for research velocity and most production PyTorch shops.
- **DGL (Deep Graph Library)** — framework-agnostic (PyTorch backend in practice), excellent
  **heterogeneous** and **distributed** support, strong sampling and partitioning tooling. Favored for
  the largest graphs.
- **Graph stores / sources**: graph databases (Neo4j, TigerGraph, Neptune) or columnar/edge-list stores
  feed the pipeline; for huge static graphs, a partitioned on-disk format + memory-mapped features is
  common. The feature pipeline (node/edge features, the graph topology itself, train/serve parity) is a
  data-engineering problem — see `[[data-engineering-feature-stores]]`.
- **Model/training stack** sits on `[[ml-frameworks]]` (PyTorch/CUDA); pin versions — PyG/DGL wheels are
  tightly coupled to specific PyTorch + CUDA versions and *will* break on mismatch. **Verify the
  compatibility matrix against current install docs** before pinning.

---

## 7. Applications

### Recommendation (user–item graphs)
Model users, items, and interactions as a (often bipartite, often heterogeneous) graph; a GNN produces
user/item embeddings used for **candidate generation/retrieval** (ANN over item embeddings) feeding a
downstream ranker. PinSAGE (Ying et al., KDD 2018, arXiv:1806.01973) is the canonical web-scale example
(GraphSAGE-style with importance sampling). LightGCN (He et al., SIGIR 2020, arXiv:2002.02126) is a
strong, simplified collaborative-filtering GNN (drops feature transform/nonlinearity). Most graph-recsys
value is at retrieval, not final ranking — pair with `[[recsys-ranking]]`.

### Fraud / anomaly detection on transaction graphs
Entities (accounts, devices, cards, merchants) and their interactions form a graph; fraud rings show up
as **structural anomalies and label homophily through shared attributes**. GNNs catch multi-hop
collusion that per-row models miss. Realities: **extreme class imbalance** (use ranking metrics,
PR-AUC, not accuracy), **adversarial adaptation** (fraudsters camouflage by connecting to legit nodes —
plain homophily assumptions break; consider heterogeneous/attention models), **label latency**, and
heavy **degree skew**. Inductive inference on fresh entities is essential.

### Molecular property prediction / drug discovery
Molecules are graphs (atoms = nodes, bonds = edges). **Graph classification/regression** for
properties (solubility, toxicity, binding). GIN-style sum aggregation and graph transformers shine
here because **structure is the label**. Benchmarks: OGB's `ogbg-molhiv`/`ogbg-molpcba`, MoleculeNet.
Watch scaffold splits (Section 8) — random splits leak structural families and overstate generalization.

### Knowledge-graph completion
A KG is a set of `(head, relation, tail)` triples; completion = **link prediction with relations**.
KG-embedding decoders: **TransE** (translational), **DistMult** (bilinear diagonal), **ComplEx**
(complex-valued, handles asymmetry), **RotatE** (rotation in complex space, models composition).
R-GCN/heterogeneous GNN encoders can feed these decoders. Evaluate with **filtered MRR and Hits@{1,3,10}**
(filtered = exclude other known-true triples from the ranking), the standard KG protocol.

### GNNs for time series (GNN4TS)
Many time-series problems are **multiple related series** (traffic sensors, power grids, sensor
networks, financial assets). A spatio-temporal GNN models **relations between series** with a GNN over
the series graph, composed with a temporal model (TCN/RNN/attention) — covering **forecasting,
classification, anomaly detection, and imputation**. The graph may be given (road network) or learned
(graph-structure learning). Canonical references: STGCN (Yu et al., 2018, arXiv:1709.04875), DCRNN
(Li et al., ICLR 2018, arXiv:1707.01926), Graph WaveNet (Wu et al., 2019, arXiv:1906.00121), and the
**GNN4TS survey** (Jin et al., *A Survey on Graph Neural Networks for Time Series: Forecasting,
Classification, Imputation, and Anomaly Detection*, IEEE TPAMI 2024, arXiv:2307.03759 — verify the
arXiv ID and venue against current records). The temporal half lives in `[[time-series-forecasting]]`.

---

## 8. Evaluation — where graph ML quietly lies to you

Graph evaluation has failure modes that *don't exist* in tabular ML. Getting the protocol wrong is the
most common cause of "great offline, useless online."

### Transductive vs inductive splits
- **Transductive**: all nodes (with edges) are present at training; only some labels are hidden. You
  measure label propagation over a fixed graph. Most citation-network benchmarks are transductive.
- **Inductive**: test nodes/graphs are *unseen during training* — the realistic production setting when
  new entities arrive. Inductive is strictly harder and more honest. **State which one you are reporting**;
  comparing across the two is meaningless.

### Edge leakage — the big one
- **Message-passing leakage**: in link prediction you must **remove the supervision/validation/test
  edges from the graph used for message passing.** If a test edge is in the adjacency the model
  aggregates over, you are predicting an edge while it's literally an input. Use separate
  message-passing vs supervision edge sets (PyG's `RandomLinkSplit` / `LinkNeighborLoader` are built for
  this — verify current API).
- **Negative-sampling leakage**: negatives must be genuine non-edges; sampling carelessly can include
  true edges, and a 1:1 balanced test set badly misrepresents the real (sparse) positive rate. Evaluate
  ranking, not balanced accuracy.
- **Temporal leakage**: on evolving graphs, split by **time** — never let future edges inform past
  predictions. Random edge splits on a temporal graph leak the future.
- **Feature leakage via neighbors**: a node's neighbor aggregate can encode the target label if labels
  leaked into features. Audit feature construction the same way you would for tabular.

### Metrics by task
- Node classification: accuracy / F1 (macro for imbalance), ROC-AUC.
- Link prediction / KG completion: **Hits@K, MRR** (filtered for KG), ROC-AUC/PR-AUC; **never** report
  accuracy on a balanced 1:1 set as if it reflects production.
- Graph classification/regression: ROC-AUC / AP (molecules), RMSE/MAE.
- Imbalanced (fraud): **PR-AUC**, precision@k, recall at fixed alert budget — accuracy is meaningless.

### Benchmarks & splits
- **OGB (Open Graph Benchmark)** — standardized large-scale datasets, *fixed splits*, and leaderboards
  across node/link/graph tasks. Use it to validate your *protocol* even on proprietary data: mirror its
  split discipline. **Don't invent your own random split for a dataset that ships an official one.**
- **Scaffold splits** (molecules) group by molecular scaffold so train/test share no structural family —
  far harder and more honest than random; report it.

See `[[ml-evaluation-evals]]` for general evaluation discipline; the items above are the graph-specific
additions.

---

## 9. Anti-patterns / gotchas (the traps that bite in production)

- **GNN where tabular suffices.** No graph signal beyond features → a GBDT/MLP with neighborhood
  aggregates wins at a fraction of the cost. *Always* run that baseline first. (The #1 mistake.)
- **Ignoring scalability until it explodes.** Prototyping full-batch on a sampled subgraph, then
  discovering at launch that the real graph won't fit. Decide the sampling/partitioning strategy up
  front, sized to the production graph.
- **Edge leakage in splits.** Test edges left in the message-passing graph, or temporal leakage on
  evolving graphs. Inflates offline metrics, collapses online. Use proper link splits.
- **Too-deep GNNs / over-smoothing.** "Add layers for more capacity" → embeddings collapse, accuracy
  drops. 2–3 layers; use JK/residuals if you genuinely need range.
- **Full-batch on a huge graph.** OOM, or silently training on a sampled subset you forgot you sampled.
- **Mean/max aggregation when structure is the label.** Graph classification with mean-pool can't
  distinguish multisets → use sum/GIN.
- **node2vec/DeepWalk in a churning entity space.** Transductive embeddings can't score new nodes;
  you'll quietly retrain constantly or serve stale/zero vectors. Use an inductive GNN.
- **Treating a heterogeneous graph as homogeneous.** Collapsing node/edge types loses the relation
  semantics that carry the signal. Use R-GCN/HGT and typed features.
- **Train/serve sampling skew.** Different fan-out or feature transforms offline vs online → the served
  model isn't the trained model.
- **Reading too much into 1-WL-bounded models.** Expecting a vanilla MPNN to count triangles or
  distinguish regular graphs — it provably can't. Add structural encodings or higher-order methods.
- **PyG/DGL ↔ PyTorch/CUDA version drift.** Mismatched wheels fail at import or silently fall back to
  slow paths. Pin to the compatibility matrix.

---

## 10. Version awareness

It is 2026 and graph ML moves fast. Treat as fast-moving and **verify against current docs**: PyG/DGL
APIs and sampler names (`NeighborLoader`, `LinkNeighborLoader`, `RandomLinkSplit`, distributed modules),
default architectures and SOTA on OGB leaderboards, graph-transformer state of the art, and the
GNN4TS literature. The *concepts* here (message passing, sampling, leakage, expressivity, depth limits)
are stable; specific signatures, version numbers, and benchmark figures are not — confirm before
relying on them. Where an arXiv ID is given, treat it as a pointer to verify, not gospel.

---

## 11. Canonical references (verify current)

**Foundations & architectures**
- GCN — Kipf & Welling, *Semi-Supervised Classification with GCNs*, ICLR 2017, arXiv:1609.02907.
- GraphSAGE — Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs*, NeurIPS
  2017, arXiv:1706.02216.
- GAT — Veličković et al., *Graph Attention Networks*, ICLR 2018, arXiv:1710.10903; GATv2 — Brody et
  al., arXiv:2105.14491.
- GIN — Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019, arXiv:1810.00826.
- R-GCN — Schlichtkrull et al., *Modeling Relational Data with GCNs*, ESWC 2018, arXiv:1703.06103.
- MPNN framework — Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017,
  arXiv:1704.01212.
- node2vec — Grover & Leskovec, KDD 2016, arXiv:1607.00653; DeepWalk — Perozzi et al., KDD 2014,
  arXiv:1403.6652.

**Scalability**
- Cluster-GCN — Chiang et al., KDD 2019, arXiv:1905.07953.
- GraphSAINT — Zeng et al., ICLR 2020, arXiv:1907.04931.
- GNNAutoScale (historical embeddings) — Fey et al., ICML 2021, arXiv:2106.05609 (verify).

**Depth / expressivity**
- Over-squashing & bottlenecks — Alon & Yahav, ICLR 2021, arXiv:2006.05205.
- Jumping Knowledge — Xu et al., ICML 2018, arXiv:1806.03536 (verify).
- Graph transformers — Graphormer, Ying et al., NeurIPS 2021, arXiv:2106.05667; GPS, Rampášek et al.,
  NeurIPS 2022, arXiv:2205.12454.

**Applications**
- PinSAGE — Ying et al., KDD 2018, arXiv:1806.01973. LightGCN — He et al., SIGIR 2020, arXiv:2002.02126.
- HGT — Hu et al., WWW 2020, arXiv:2003.01332.
- GNN4TS survey — Jin et al., IEEE TPAMI 2024, arXiv:2307.03759 (verify ID/venue). STGCN
  (arXiv:1709.04875), DCRNN (arXiv:1707.01926), Graph WaveNet (arXiv:1906.00121).

**Tooling & benchmarks**
- PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io
- DGL docs: https://docs.dgl.ai
- Open Graph Benchmark: https://ogb.stanford.edu (Hu et al., NeurIPS 2020, arXiv:2005.00687).
- Survey / textbook: Hamilton, *Graph Representation Learning* (2020); Bronstein et al., *Geometric
  Deep Learning* (arXiv:2104.13478) for the broader framing.
