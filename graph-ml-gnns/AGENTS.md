# AGENTS.md — Graph ML & GNN Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`graph-ml-gnns-guide.md`** next to this file — read it
> before designing, building, or reviewing graph-ML/GNN work, and apply it. Concrete artifacts to
> imitate (GraphSAGE node classification with neighbor sampling, a link-prediction setup, a
> sampling/scalability decision note) are in **`examples.md`**. This file is the always-on summary.
>
> The field moves fast (it is 2026). Concepts here are stable; specific PyG/DGL APIs, version pins,
> SOTA, and benchmark numbers are not — **verify against current docs** before relying on them.

## When working on graph data / GNNs, apply these by default:

- **Earn the graph first.** Use a GNN only when relational structure carries signal beyond node
  features. The baseline you must beat is a GBDT/MLP on node features + neighborhood aggregates
  (degree, neighbor means, PageRank). If it isn't beaten, it's a tabular problem — don't ship a GNN.
- **Decide scalability up front, sized to the production graph** — not after it explodes. Full-batch
  only for small graphs; **GraphSAGE neighbor sampling** is the default for large/dynamic graphs;
  Cluster-GCN / GraphSAINT for large static graphs; historical embeddings for depth at scale.
- **GNNs = message passing**: aggregate (sum/mean/max/attention) + update; **L layers = L hops** of
  receptive field. Architecture choice is just aggregator/update knobs.
- **Architecture by constraint:** GCN = small transductive baseline; **GraphSAGE = inductive + sampling
  (default at scale)**; GAT/GATv2 = learned neighbor attention; **GIN (sum aggregator) = max
  expressivity / graph classification**; **R-GCN / HGT = heterogeneous & knowledge graphs**; graph
  transformers only for genuine long-range needs on small graphs.
- **Keep it shallow: 2–3 layers.** More → **over-smoothing** (embeddings collapse) and **over-squashing**
  (bottlenecked long-range signal), *and* exponential neighbor blow-up (`fan-out^L`). Use
  residual/Jumping-Knowledge before adding depth.
- **Expressivity ceiling = 1-WL** for standard MPNNs — they can't count triangles or distinguish some
  regular graphs. Use sum/GIN, structural encodings, or higher-order methods if the task needs it.
- **Aggregator matters: sum > mean > max** for distinguishing power. Sum/GIN when structure is the
  label; mean/SAGE for homophilous node tasks; attention for heterogeneous neighbor importance.
- **Inductive vs transductive**: state which you are training/reporting. Prefer inductive (GraphSAGE
  family) when new entities appear; don't ship transductive node2vec/DeepWalk in a churning entity space.
- **Evaluation traps (graph-specific):** remove supervision/val/test edges from the message-passing
  graph (**edge leakage**); split temporally on evolving graphs; sample genuine negatives; report
  **Hits@K / MRR (filtered for KG)** and **PR-AUC** for imbalanced fraud — never balanced accuracy.
  Mirror **OGB** split discipline; use scaffold splits for molecules.
- **Link prediction / KG completion** = score node pairs with a decoder (dot / DistMult / ComplEx /
  RotatE / MLP); R-GCN/HGT encoders for typed graphs.
- **GNN4TS**: spatio-temporal GNN over the series graph + a temporal model for forecasting/
  classification/anomaly/imputation (see `[[time-series-forecasting]]`).
- **Tooling: PyG or DGL.** PyG = velocity + model zoo; DGL = heterogeneous + distributed. **Pin to the
  PyG/DGL ↔ PyTorch/CUDA compatibility matrix** — mismatches break at import. Partition large graphs
  (minimize edge cut); the bottleneck is feature/sampling I/O, not GPU compute.
- **Serving**: precompute + serve embeddings for slow graphs; online k-hop sampling for fresh nodes.
  Keep train/serve sampling and feature parity or you get skew.

## Definition of done for a GNN change
- Tabular baseline run and beaten (or graph use justified).
- Split protocol stated (transductive/inductive, temporal) and **edge leakage ruled out**.
- Scalability strategy matches the production graph size; train/serve parity holds.
- Metrics appropriate to the task (ranking/PR-AUC, not balanced accuracy where misleading).
- Framework/CUDA versions pinned to the compatibility matrix; fast-moving claims verified vs current docs.

## Related skills
`[[recsys-ranking]]` · `[[time-series-forecasting]]` · `[[ml-frameworks]]` ·
`[[data-engineering-feature-stores]]` · `[[ml-evaluation-evals]]` · `[[aiml-on-kubernetes]]`
