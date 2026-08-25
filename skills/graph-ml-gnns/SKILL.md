---
name: graph-ml-gnns
description: World-class graph machine learning and Graph Neural Network (GNN) judgment for production —
  node classification, link prediction, graph classification/regression, community detection, and
  heterogeneous/knowledge-graph learning. Use when working with graph-structured data, relational data
  modeled as nodes/edges, or any GNN task; when choosing or implementing GCN, GraphSAGE, GAT, GIN, R-GCN,
  or graph transformers; when scaling GNNs (neighbor sampling, Cluster-GCN, GraphSAINT, distributed/
  partitioned training, serving); when using PyTorch Geometric (PyG) or DGL; or for graph-based
  recommendation, fraud/anomaly detection on transaction graphs, molecular property prediction/drug
  discovery, knowledge-graph completion, and GNNs for time series (GNN4TS). Covers message passing,
  over-smoothing/over-squashing, expressivity (1-WL), transductive vs inductive splits, edge leakage,
  and OGB benchmarks.
---

# Graph ML & GNNs

Apply the judgment of an engineer who has shipped GNNs on billion-edge graphs in production —
recommendations, fraud, molecules, knowledge graphs. The first decision is always **whether you even
need a graph model**; the second is **whether it will scale**. Most failed GNN projects die on one of
those two, not on architecture choice.

## How to use this skill

1. **Read `graph-ml-gnns-guide.md`** in this directory — the full reference (problem framing, message
   passing, architectures, scalability, tooling, applications, evaluation, anti-patterns). Apply it to
   the task at hand.
2. For concrete artifacts to imitate — a GraphSAGE node-classification training sketch with neighbor
   sampling, a link-prediction setup, and a sampling/scalability decision note — read **`examples.md`**.
3. Match the surrounding codebase and framework conventions (PyG vs DGL, the existing feature pipeline);
   apply the correctness and evaluation rules (split hygiene, edge leakage, scalability) regardless.
4. The field moves fast — APIs, default samplers, and SOTA architectures change. Where this skill flags
   "verify against current docs," do so before relying on a specific signature or benchmark number.

## The essentials (full rationale in `graph-ml-gnns-guide.md`)

- **Earn the graph.** Use a GNN only when the *relational structure carries signal* that tabular
  features can't recover (homophily, multi-hop dependence, structural roles). A strong GBDT/MLP on
  node features is the baseline you must beat — and often it wins. Don't reach for GNNs reflexively.
- **GNNs are message passing.** Every layer = *aggregate* neighbor messages (sum/mean/max/attention) +
  *update* (combine with self). Depth = receptive field in hops. Know this and the architecture zoo
  becomes a few knobs, not a menagerie.
- **Pick the architecture for the constraint, not the leaderboard.** GCN = simple transductive
  baseline; **GraphSAGE = inductive + neighbor sampling (the default for big/dynamic graphs)**; GAT =
  learned per-edge attention; GIN = max expressivity (sum aggregator, injective, 1-WL); **R-GCN/HGT for
  heterogeneous & knowledge graphs**; graph transformers when long-range interaction dominates.
- **Depth is a trap.** 2–3 layers is usually optimal. More layers → **over-smoothing** (node
  representations converge) and **over-squashing** (exponentially many messages crushed into fixed-size
  vectors across bottlenecks). Add residuals/jumping-knowledge before adding depth.
- **Scalability is the production problem, not accuracy.** Full-batch GCN materializes the whole graph
  in memory — fine for OGB-arxiv, impossible at billions of edges. Use **neighbor sampling
  (GraphSAGE)**, **subgraph sampling (Cluster-GCN, GraphSAINT)**, or historical embeddings. Plan
  partitioning and distributed training *before* the graph explodes, not after.
- **Embeddings vs end-to-end.** Shallow node embeddings (DeepWalk/node2vec) are transductive and
  cheap but can't use features or generalize to new nodes; GNNs are inductive and feature-aware.
  Don't ship node2vec where new entities appear daily.
- **Evaluation is where graph ML quietly lies to you.** **Transductive vs inductive** splits measure
  different things — be explicit. **Edge leakage** (test edges visible during message passing, or
  link-prediction negatives sampled from positives) inflates metrics catastrophically. Use **OGB**
  splits/leaderboards as a sanity reference for protocol.
- **Link prediction needs an honest negative-sampling and split protocol.** Remove supervision/test
  edges from the message-passing graph; score with a decoder (dot product / DistMult / MLP); evaluate
  with ranking metrics (Hits@K, MRR), not accuracy on a 1:1 balanced set.
- **Heterogeneous & knowledge graphs are first-class**, not a footnote. Multiple node/edge types →
  per-relation transforms (R-GCN), metapaths, or typed attention (HGT). KG completion = link
  prediction with relation-aware decoders (TransE/DistMult/ComplEx/RotatE).
- **GNNs for time series (GNN4TS)** model the *relations between series* (spatio-temporal: traffic
  sensors, grids, sensor networks) for forecasting/classification/anomaly/imputation — a GNN over the
  series graph composed with a temporal model. See the GNN4TS survey (verify current).
- **Tooling: PyG and DGL** are the two production frameworks. PyG = Pythonic, large model zoo; DGL =
  strong heterogeneous + distributed story. Both have mature samplers. Feed them from a real graph
  feature pipeline — see `[[data-engineering-feature-stores]]`.

## Related skills

- `[[recsys-ranking]]` — user–item interaction graphs; GNNs as candidate-generation/embedding models
  feeding a ranker. Most graph-recsys lives at the retrieval stage.
- `[[time-series-forecasting]]` — the temporal modeling half of GNN4TS (spatio-temporal forecasting).
- `[[ml-frameworks]]` — PyTorch/JAX, GPU memory and kernels underneath PyG/DGL.
- `[[data-engineering-feature-stores]]` — building and serving node/edge features and the graph itself.
- `[[ml-evaluation-evals]]` — general evaluation discipline; this skill adds graph-specific leakage traps.
- `[[aiml-on-kubernetes]]` — distributed/partitioned GNN training and GNN serving on clusters.
