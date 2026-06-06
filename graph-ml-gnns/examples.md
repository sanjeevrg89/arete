# Graph ML & GNN — Worked Examples

Canonical artifacts to imitate. PyG-style (PyTorch Geometric) APIs shown — **verify exact signatures
against current PyG docs**, as loader/split APIs evolve. The patterns (sampling, leakage-safe splits,
sizing) are the durable part.

---

## 1. GraphSAGE node classification with neighbor sampling (PyG-style)

The production default for a large/dynamic graph: inductive, mini-batched via `NeighborLoader`, bounded
memory. Note the fan-out is the memory/throughput knob, and `n_id`/batch-size slicing extracts the seed
nodes' outputs from the sampled subgraph.

```python
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)            # aggregate(neighbors) + update(self)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x                                # logits per node

# data: a torch_geometric.data.Data with x, edge_index, y, train_mask
# num_neighbors = per-layer fan-out. 2 layers => up to 15*10 = 150 nodes per seed.
train_loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],                     # fan-out per hop; cap to bound f^L blow-up
    batch_size=1024,
    input_nodes=data.train_mask,                # seed nodes to compute loss for
    shuffle=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraphSAGE(data.num_features, 256, int(data.y.max()) + 1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

def train_epoch():
    model.train()
    total = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x, batch.edge_index)          # over the sampled subgraph
        # Loss ONLY on the seed nodes (first batch.batch_size rows are the seeds in PyG):
        seed_out = out[: batch.batch_size]
        seed_y   = batch.y[: batch.batch_size]
        loss = F.cross_entropy(seed_out, seed_y)
        loss.backward()
        opt.step()
        total += float(loss) * batch.batch_size
    return total / int(data.train_mask.sum())
```

Notes:
- **Inductive eval**: at inference, build a `NeighborLoader` over test seeds (or score brand-new nodes
  by sampling their k-hop subgraph) — the same learned weights generalize because GraphSAGE learns
  aggregators, not per-node vectors. Keep the *same fan-out semantics* at serve time as train time.
- **Hubs**: on power-law graphs, the `[15,10]` cap is what keeps a million-edge hub from blowing up the
  batch. Don't remove the cap "for accuracy" without measuring memory.
- **Depth**: this is 2 layers on purpose. Going to 3–4 multiplies both the subgraph size *and*
  over-smoothing risk — add Jumping-Knowledge/residuals instead if you need range.

---

## 2. Link prediction — leakage-safe setup (the part people get wrong)

The single most common link-prediction bug: leaving supervision/validation/test edges in the graph the
model passes messages over, so the model "predicts" an edge it was literally given as input. Use
disjoint **message-passing** and **supervision** edge sets, and ranking metrics.

```python
import torch
from torch_geometric.transforms import RandomLinkSplit

# Split edges into train/val/test, and within each, into message-passing vs supervision edges.
# is_undirected and negative sampling handled by the transform. VERIFY current arg names in PyG docs.
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,    # sampled NON-edges as negatives
    disjoint_train_ratio=0.3,           # hold out a fraction of train edges as supervision targets
)
train_data, val_data, test_data = transform(data)
# train_data.edge_index           -> message-passing graph (NO supervision/test edges in here)
# train_data.edge_label_index     -> the (pos+neg) pairs we actually score and compute loss on
# train_data.edge_label           -> 1 for positive, 0 for negative

def decode(z, edge_label_index):
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=-1)        # dot-product decoder; swap DistMult/MLP for KG/relations

# training step (encoder = the GraphSAGE/GCN above producing node embeddings z):
z = encoder(train_data.x, train_data.edge_index)         # messages over MP edges ONLY
scores = decode(z, train_data.edge_label_index)
loss = F.binary_cross_entropy_with_logits(scores, train_data.edge_label.float())
```

Protocol rules:
- **Never** include `edge_label_index`/val/test edges in the `edge_index` used for message passing.
- **Evaluate with ranking metrics — Hits@K, MRR** (filtered MRR for knowledge graphs: exclude other
  known-true tails when ranking) — *not* accuracy on a 1:1 balanced set, which misrepresents the sparse
  real positive rate.
- **Temporal graphs**: split by time, not randomly — random splits leak future edges into the past.
- **KG completion**: replace the dot product with a relation-aware decoder — **DistMult** (diagonal
  bilinear), **ComplEx** (asymmetric relations), or **RotatE** (composition); encode typed graphs with
  R-GCN/HGT.

---

## 3. Scalability / sampling decision note

Decide this *before* building, sized to the **production** graph (not your prototype subgraph).

| Production graph | Strategy | Why |
| --- | --- | --- |
| ≤ ~10⁵–10⁶ nodes, static, fits in GPU memory | **Full-batch GCN/GAT** | simplest, exact; no sampler variance |
| Large or **dynamic**, new nodes appear | **GraphSAGE + neighbor sampling** (`NeighborLoader`) | inductive, mini-batch, bounded memory; the default |
| Large **static**, want full-depth GNN | **Cluster-GCN** (METIS partitions) or **GraphSAINT** (subgraph sampler) | mini-batch on dense subgraphs; less redundant computation than naive neighbor sampling |
| Deep model on a large graph | **Historical embeddings** (GNNAutoScale) | cache stale neighbor states; trade memory/staleness for depth |
| Beyond a single machine | **Partition (minimize edge cut) + distributed (DGL DistGraph / PyG distributed)** | scale out; bottleneck is feature/sampling I/O, not GPU FLOPs |

Sizing the neighbor-sampling blow-up:

```
nodes_per_seed ≈ product(num_neighbors)          # fan-out per hop
[15, 10]        -> ~150 nodes/seed   (2 hops)     # typical, safe
[15, 10, 5]     -> ~750 nodes/seed   (3 hops)     # 3rd hop ~5x cost AND more over-smoothing
batch memory   ∝ batch_size * nodes_per_seed * feature_dim
```

Decision heuristics:
- If a **GBDT/MLP on node features + neighborhood aggregates** isn't beaten, stop — no graph model.
- Start at **2 layers, `[15,10]`**. Tune fan-out before depth. Cap hub neighbors on power-law graphs.
- If full-batch OOMs, switch to neighbor sampling *before* reaching for Cluster-GCN/GraphSAINT.
- For serving: **precompute embeddings** to a store for slow-changing graphs (pairs with
  `[[recsys-ranking]]` retrieval); **online k-hop sampling** only when you must score fresh nodes in
  real time — and keep train/serve fan-out + feature parity (`[[data-engineering-feature-stores]]`).
