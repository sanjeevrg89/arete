---
name: embedding-model-training
description: World-class guidance for TRAINING text/retrieval embedding and reranker models — bi-encoders
  (dual-encoders), cross-encoder rerankers, and the retrieve-then-rerank pattern. Use when training,
  fine-tuning, distilling, or evaluating an embedding/retriever/reranker (contrastive/InfoNCE loss,
  in-batch negatives, hard-negative mining, false-negative removal, cross-encoder distillation,
  Matryoshka/MRL nested dims, ColBERT multi-vector, instruction-tuned embeddings, multilingual/
  multimodal), or when picking embedding dimension/pooling/normalization and evaluating on MTEB/BEIR
  (nDCG@10, recall). This is how the MODELS are built; for serving embeddings, ANN indexes, and RAG
  retrieval infra see [[rag-vector-databases]].
---

# Embedding Model Training

Apply the judgment of someone who has trained state-of-the-art retrievers and rerankers and shipped
them into production search/RAG systems. The hard part is almost never the loss function — it is the
**data**: which positives, and above all which negatives, you put in front of the model. Get
hard-negative mining and false-negative removal right and a small model beats a big one.

## How to use this skill

1. **Read `embedding-model-training-guide.md`** in this directory — the full reference (architectures,
   contrastive learning, hard-negative mining, training recipes, evaluation, production). Apply it.
2. For concrete artifacts to imitate — an InfoNCE bi-encoder training sketch, a positive-aware
   hard-negative-mining config, and a cross-encoder distillation outline — read **`examples.md`**.
3. Match the surrounding codebase/framework conventions ([[ml-frameworks]], [[training-frameworks]]);
   apply the data-quality and evaluation rules regardless.

## The essentials (full rationale in the guide)

- **Pick the right architecture for the latency budget.** Bi-encoder (encode query and doc separately,
  compare with cosine/dot) is cheap and indexable — use it for first-stage retrieval over millions of
  docs. Cross-encoder (joint query+doc → score) is far more accurate but cannot be pre-indexed — use it
  as a **reranker** over the top-k. The default production shape is **retrieve-then-rerank**
  ([[rag-vector-databases]], [[recsys-ranking]]). If a reranker was the real need, do not train a
  bi-encoder.
- **Contrastive learning is the workhorse.** InfoNCE with **in-batch negatives**: every other doc in
  the batch is a negative for the current query. Large batches help because they give more negatives per
  step (DPR, E5). Tune **temperature** (often ~0.01–0.05 for normalized embeddings). Decide pooling
  (mean vs CLS/last-token) and **L2-normalize** when training with cosine.
- **Hard negatives are the crux.** Random/in-batch negatives are too easy and accuracy plateaus. Mine
  hard negatives with a retriever (top-ranked non-positives) — ANCE-style, refreshed periodically.
- **The false-negative problem dominates web data.** Naive top-k mining mislabels many *true* relevant
  passages as negatives. **Positive-aware / threshold-based mining** (NV-Retriever) anchors on the
  positive's score and drops candidates scoring too close to it → faster training and higher accuracy.
- **Recipe: contrastive pretrain on weakly-supervised pairs → fine-tune on labeled + mined-hard-negative
  data → optionally distill from a cross-encoder teacher** (soft scores / MarginMSE / KL). Instruction-
  prefixed embeddings let one model serve many tasks.
- **Matryoshka (MRL)** trains nested dimensions so you can truncate (e.g. 1024→256) at deploy time for
  cheaper storage/ANN with graceful accuracy loss. **ColBERT** multi-vector trades index size for
  accuracy via late interaction.
- **Evaluate on MTEB (retrieval subset = BEIR)** with **nDCG@10** and recall@k — but watch
  **contamination/overfitting** and treat leaderboard ranks as time-scoped (verify against current docs).
- **Production:** the embedding **dimension and distance metric must match the index** exactly; quantize
  embeddings (int8/binary) to cut cost; and **re-embed the whole corpus when the model version changes**
  — query and corpus embeddings must come from the same model. Version your indexes.

## Related skills
- `[[rag-vector-databases]]` — serving embeddings: ANN indexes (HNSW/IVF), hybrid search, RAG pipelines.
- `[[recsys-ranking]]` — two-tower retrieval and learning-to-rank; the same retrieve-then-rank shape.
- `[[multimodal-ml]]` — CLIP-style image/text and other multimodal embeddings (same contrastive core).
- `[[fine-tuning-peft]]` — LoRA/adapter fine-tuning of LLM-based embedding backbones.
- `[[ml-frameworks]]` / `[[training-frameworks]]` — PyTorch/JAX, FSDP/DeepSpeed, large-batch training.
- `[[ml-evaluation-evals]]` — building trustworthy eval sets and avoiding benchmark contamination.
- `[[aiml-on-kubernetes]]` — running the embedding/mining/eval jobs on GPUs at scale.

---

# Reference — embedding-model-training

# Embedding Model Training — Full Reference

How embedding and reranker **models** are built and trained. This is upstream of [[rag-vector-databases]]
(which consumes embeddings and runs ANN indexes) and shares its mental model with two-tower
[[recsys-ranking]]. The ecosystem moves fast — model rankings, leaderboard positions, and "best"
hyperparameters change monthly. Treat all leaderboard claims and exact numbers here as **time-scoped;
verify against current docs and papers before relying on them.**

---

## 1. Mental model: where embeddings come from, and why training matters

An embedding model maps a piece of content (query, passage, image) to a vector such that
**semantically related items are close** under a chosen similarity (cosine or dot product). A retriever
is "good" when, for a query, its true relevant documents are nearer than everything else in the corpus.

Pretrained LLMs and masked-LM backbones do **not** give you this for free. Their hidden states are
optimized for next-token / masked-token prediction, not for a metric space where relevance = proximity.
**Training is what shapes the geometry**: contrastive learning pulls positives together and pushes
negatives apart until nearest-neighbor search returns the right documents. The quality ceiling is set
far more by the *training pairs and negatives* than by the backbone or the loss.

### The two architectures (know which you are building)

| | **Bi-encoder (dual-encoder)** | **Cross-encoder (reranker)** |
|---|---|---|
| Shape | Encode query and doc **separately** → two vectors | Concatenate query+doc → encode **jointly** → 1 score |
| Similarity | cosine / dot of the two vectors | scalar relevance from a classification head |
| Pre-indexable? | **Yes** — embed corpus once, ANN search | **No** — must run the model per (query, doc) pair |
| Cost at query time | One query encode + ANN lookup (ms over millions) | One forward pass per candidate (k forwards) |
| Accuracy | Lower (no query↔doc token interaction) | **Higher** (full attention across query and doc) |
| Role | **First-stage retrieval** | **Reranking** the top-k |

These are complementary, not competing. **Retrieve-then-rerank** is the canonical production shape:
a bi-encoder retrieves top-100/1000 cheaply from the index, a cross-encoder reranks them to top-10. You
get the bi-encoder's scalability and most of the cross-encoder's accuracy. See [[rag-vector-databases]]
for the retrieval/index half and [[recsys-ranking]] for the same retrieve→rank decomposition in recsys.

**A common, expensive mistake:** burning weeks improving a bi-encoder when the real bottleneck is
ranking quality on a small candidate set — i.e. you needed a reranker. If candidate recall is already
high but the top results are wrong, train/deploy a cross-encoder before touching the retriever.

### Multi-vector (ColBERT): a middle ground

ColBERT (arXiv:2004.12832; v2 arXiv:2112.01488) keeps **one vector per token** and scores via **late
interaction** (sum of max query-token↔doc-token similarities). It is more accurate than single-vector
bi-encoders and far cheaper than a cross-encoder, at the cost of a much larger index (many vectors per
doc) and a specialized scoring path. Reach for it when single-vector recall is insufficient but
per-pair cross-encoding is too slow.

---

## 2. Contrastive learning (the core objective)

### InfoNCE / contrastive loss

Given a query `q`, one positive `d+`, and negatives `{d-}`, with similarity `s(·,·)` and temperature
`τ`, the InfoNCE loss is a softmax cross-entropy that maximizes the positive's relative score:

```
L = -log( exp(s(q, d+) / τ) / ( exp(s(q, d+) / τ) + Σ_j exp(s(q, d-_j) / τ) ) )
```

Intuition: it is a classification over "which of these documents is the true match," so **more and
harder negatives make the task harder and the gradient richer**. This single objective underlies DPR,
SimCSE, E5, GTE, and the NV-Retriever family.

### In-batch negatives, and why large batches help

The cheap trick that made dense retrieval practical (DPR, arXiv:2004.04906): within a batch of `B`
(query, positive) pairs, treat **every other pair's positive as a negative** for the current query.
One forward pass yields `B` queries × `B` docs of supervision essentially for free.

Because each step's negative count scales with batch size, **larger batches = more negatives per step =
better embeddings** — a key reason E5 (arXiv:2212.03533) and GTE (arXiv:2308.03281) train with very
large batches. Scaling techniques:

- **Cross-GPU negative sharing** ("gather"): collect embeddings from all data-parallel ranks so every
  query sees negatives from the whole global batch, not just its local shard. Standard in large runs.
- **GradCache** decouples the batch size from GPU memory by recomputing encoder activations, letting
  you reach huge effective batches on limited memory.
- See [[training-frameworks]] for FSDP/DeepSpeed/ZeRO and [[ml-frameworks]] for the PyTorch/JAX details
  of distributed gather and mixed precision.

### Temperature, similarity, pooling, normalization — the four knobs that quietly decide everything

- **Similarity space.** Train and serve with the **same** metric. Cosine (L2-normalized dot) is the
  common default and pairs with `IndexFlatIP`/cosine ANN; raw dot product (no normalization) keeps
  magnitude information. **The model's distance metric must match the index's metric** (see §6).
- **Normalization.** With cosine, **L2-normalize embeddings** at train and inference. Mismatched
  normalization between training and serving is a frequent silent accuracy killer.
- **Temperature `τ`.** Sharpens/softens the softmax. For normalized embeddings (where `s` ∈ [-1, 1]),
  values around **0.01–0.05** are common; too low destabilizes training, too high blunts the signal.
  Treat the exact value as a tuned hyperparameter, not a constant.
- **Pooling.** How token states become one vector:
  - **Mean pooling** over (mask-weighted) tokens — robust default for encoder backbones (E5, GTE).
  - **CLS / first-token** pooling — works when the backbone is trained for it.
  - **Last-token** pooling — common for **decoder/LLM-based** embedding models, where the final token
    attends to the whole sequence.
  Pooling, normalization, and the prompt/instruction must be **identical at train and inference** or
  the geometry you trained no longer holds.

### Instruction / prompt-based embeddings

Modern general-purpose embedders prepend a task instruction (e.g. `"query: ..."` / `"passage: ..."`,
or a natural-language task description) so one model serves retrieval, classification, clustering, and
STS. The model learns to condition its embedding on the instruction. **The serving side must use the
exact same prefixes** the model was trained with — a mismatch silently degrades retrieval.

---

## 3. Hard-negative mining (the crux)

### Why random and in-batch negatives plateau

In-batch negatives are mostly **easy**: a random corpus document is usually obviously irrelevant, so
the model quickly drives its score down and the gradient vanishes. To push the decision boundary you
need negatives that are **semantically close but not relevant** — "hard" negatives. Without them,
accuracy plateaus well below the achievable ceiling regardless of batch size.

### Mining hard negatives with a retriever

The standard procedure:

1. Embed the corpus with a current retriever (the model being trained, or a strong off-the-shelf one).
2. For each training query, retrieve the **top-k** documents.
3. Take the **non-positive** retrieved docs as hard-negative candidates (they look relevant to the
   model but are not labeled positive).
4. Mix some hard negatives with in-batch negatives in each training example.

**ANCE** (arXiv:2007.00808) made this *iterative*: periodically refresh an ANN index with the
in-training model and re-mine, so negatives stay "hard" as the model improves. This closes the gap
between training negatives and the irrelevant documents actually seen at test time. The cost is the
periodic re-encode + re-index of the corpus; trade refresh frequency against compute.

### The false-negative problem (this is where most of the accuracy is lost)

Top-k mining has a dangerous failure mode: **many of those top-ranked "non-positives" are actually
relevant** — they are just **unlabeled** positives. On web-scale data, relevance annotations are sparse
(a query may have one labeled positive but dozens of genuinely relevant passages), so naive mining
mislabels a **large fraction** of true positives as negatives. Training then explicitly pushes correct
answers away — actively teaching the model the wrong thing. This is **the** dominant error source in
naive hard-negative pipelines.

### Positive-aware / threshold-based mining (NV-Retriever)

**NV-Retriever** (arXiv:2407.15831) introduced **positive-aware** mining: use the **positive's
relevance score as an anchor** and discard any mined candidate whose score is too close to (or above) a
threshold derived from it — those are the likely false negatives. Two common forms:

- **Absolute threshold:** drop candidates with score above a fixed value.
- **Percentage / TopK-PercPos:** drop candidates scoring above some fraction (e.g. 95%) of the
  positive's score.

Removing these false negatives gives **faster training and more accurate retrievers**. NV-Retriever
reported a top MTEB-Retrieval (BEIR) score of ~60.9 and a #1 placement **at time of publication**
(July 2024) — treat that ranking as historical; **verify the current leaderboard**. The *technique*
(anchor on the positive, threshold out near-positives) is durable even as the numbers age.

**Practical mining checklist:**
- Use a **strong teacher/scorer** to mine (often a cross-encoder or a strong embedder) — better mining
  model → better negatives → better student.
- Apply **positive-aware filtering** always; never feed raw top-k as negatives on sparsely-labeled data.
- Sample negatives from a **range of ranks** (not only rank 1–5) to avoid over-concentrating on the
  very hardest, which can destabilize training.
- Keep some easy/in-batch negatives in the mix for stability.
- **Re-mine periodically** (ANCE-style) if you can afford it; the negatives a stale model finds are no
  longer hard for the improved model.

---

## 4. Training recipes

The reliable arc for a strong retriever is **weakly-supervised contrastive pretraining → supervised
fine-tuning on labeled + mined-hard-negative data → (optional) distillation from a cross-encoder.**

### Stage 1 — Contrastive pretraining on weakly-supervised pairs

Train on huge quantities of naturally-occurring "pairs" that are *probably* related — title↔body,
question↔answer (e.g. forum/QA dumps), citation pairs, click pairs, adjacent passages. No human labels;
in-batch negatives; very large batches. This teaches general semantic proximity. E5's CCPairs
(arXiv:2212.03533) and GTE's multi-stage mixture (arXiv:2308.03281) are canonical examples; SimCSE
(arXiv:2104.08821) showed even dropout-augmented self-pairs plus NLI pairs give strong sentence
embeddings.

### Stage 2 — Supervised fine-tuning on labeled + mined data

Fine-tune on task-relevant labeled pairs (MS MARCO, NQ, HotpotQA, domain data) **with positive-aware
mined hard negatives** (§3). This is where most retrieval-task accuracy is won. Keep batches large;
keep the instruction prefixes consistent with how you will serve.

### Stage 3 — Distillation from a cross-encoder (teacher → student)

A cross-encoder reranker is more accurate than any bi-encoder; **distill its knowledge into the
bi-encoder student**:

- Score (query, doc) pairs with the cross-encoder teacher (typically the mined positives + hard
  negatives).
- Train the student to match the teacher's relevance signal. Common objectives:
  - **MarginMSE** — match the teacher's *margin* between positive and negative scores.
  - **KL divergence** — match the teacher's softmax distribution over the candidate set per query.
- Result: the student bi-encoder inherits much of the teacher's ranking quality while staying cheap and
  indexable. This is one of the highest-leverage moves for production retrievers.

You can also distill into a **smaller, faster cross-encoder** for the rerank stage when latency matters
([[fine-tuning-peft]] for parameter-efficient backbone adaptation).

### Multi-stage and curriculum

Combining the above (sometimes with multiple mining rounds) is "multi-stage" training. A rough
curriculum — easy/weak pairs first, then harder labeled+mined data, then distillation — is the typical
shape. Order matters more than any single hyperparameter.

### Matryoshka Representation Learning (nested dimensions)

**MRL** (arXiv:2205.13147) adds contrastive losses at **multiple nested prefixes** of the embedding
(e.g. 64, 128, 256, 512, 1024 dims) simultaneously. The result: you can **truncate** the embedding at
serve time to a smaller dimension and still retain most of the accuracy — cheaper storage, smaller ANN
index, faster search, with graceful degradation. Many modern embedding APIs expose this as a
"dimensions" parameter. Train MRL in if you want deploy-time flexibility; it is nearly free to add.

### Single-vector vs multi-vector (ColBERT)

Decide early: single-vector (one embedding/doc, small index, simplest serving) vs ColBERT multi-vector
(one embedding/token, larger index, late-interaction scoring, higher accuracy). The training objective
is still contrastive; the difference is the scoring function and the index footprint (see §6 and
[[rag-vector-databases]]).

### Multilingual & multimodal

- **Multilingual:** a multilingual backbone + cross-lingual pairs (parallel/translation data, or
  English-aligned via distillation) yields embeddings where translations land near each other, enabling
  cross-lingual retrieval. Watch for high-resource languages dominating the batch.
- **Multimodal:** CLIP-style training is the *same InfoNCE contrastive recipe* applied to (image, text)
  pairs with two encoders projected into a shared space. See [[multimodal-ml]] for vision/text and
  audio embeddings; the negative-mining and temperature lessons here transfer directly.

---

## 5. Evaluation & SOTA

### MTEB and BEIR

- **MTEB** (Massive Text Embedding Benchmark, arXiv:2210.07316) evaluates one embedding model across
  many task types — retrieval, reranking, classification, clustering, STS, summarization — so you don't
  overfit to a single task. It is the de-facto general-purpose embedding benchmark.
- **BEIR** (arXiv:2104.08663) is the **zero-shot retrieval** benchmark and forms MTEB's **Retrieval**
  subset. It spans diverse domains/datasets to test generalization, not in-domain memorization.

### Metrics

- **nDCG@10** — the headline retrieval/ranking metric: rewards putting relevant docs high, discounted by
  rank. The standard BEIR metric.
- **Recall@k** — did the relevant docs make it into the top-k *at all* (e.g. recall@100)? This is the
  metric that matters for the **first-stage retriever** feeding a reranker — a reranker can only fix
  what retrieval surfaced.
- **MRR** — mean reciprocal rank of the first relevant result; common on QA-style data (MS MARCO).
- Choose the metric to match the role: optimize **recall@k** for the retriever, **nDCG@10/MRR** for the
  end-to-end / reranked result.

### Leaderboard dynamics & contamination (read this before trusting any rank)

- **Rankings change fast.** The MTEB top is reshuffled constantly; any "SOTA" claim is a snapshot.
  Treat every number in this guide and every leaderboard position as **time-scoped — verify current
  before quoting.**
- **Contamination / overfitting is real.** Public benchmark test data can leak into training corpora
  (web-scraped pairs), and models can be tuned *toward* the leaderboard. A high MTEB number does not
  guarantee performance on *your* data.
- **The defense:** build a **held-out, in-domain eval set** ([[ml-evaluation-evals]]) from your own
  queries/documents and treat it as the primary signal. Public benchmarks are for sanity-checking
  generalization, not for picking the production model.

---

## 6. Production concerns

### Dimension & cost tradeoffs

Embedding storage and ANN search cost scale with **dimension × corpus size**. A 4096-dim model over
100M docs is expensive to store and search. Levers: pick a smaller model dimension, or train **MRL**
and truncate, or quantize. Always measure the **accuracy-at-dimension** curve on *your* eval set, not
just MTEB.

### Quantization of embeddings

- **Scalar / int8** quantization: ~4× smaller than fp32 with usually small accuracy loss; widely
  supported by vector indexes.
- **Binary** quantization: ~32× smaller, search via Hamming distance; bigger accuracy hit, often used
  for a coarse first pass then re-scored with full-precision vectors.
- Validate the recall hit on your eval set; quantization interacts with the index and the distance
  metric. Index-side details (PQ, OPQ, HNSW params) live in [[rag-vector-databases]].

### Index interaction — match dimension AND metric exactly

The single most common production bug: the embedding model and the index disagree on **dimension** or
**distance metric**. If the model is trained for **cosine** but the index uses raw **L2/dot**, or you
forgot L2-normalization on one side, ranking silently degrades. Pin: same dim, same metric, same
normalization, same instruction prefixes — end to end. See [[rag-vector-databases]].

### Domain adaptation

Off-the-shelf embedders underperform on specialized domains (legal, biomedical, code, organization-
specific jargon). Adapt by fine-tuning on in-domain pairs with domain hard negatives, or by generating synthetic
queries for your documents (LLM-generated query↔passage pairs) and training on those. Re-mine negatives
from the *domain* corpus, not a generic one.

### Refresh & versioning (re-embed the corpus on every model change)

**Query and corpus embeddings must come from the same model version.** When you retrain or swap the
embedding model, the vector space changes — **you must re-embed the entire corpus**. Serving new-model
query vectors against an old-model index produces garbage that looks like a quiet quality regression,
not an error. Operational rules:

- **Version your indexes** alongside the model (e.g. `corpus@v3`); never mix versions.
- Plan re-embedding as a **batch job** — for large corpora this is a significant, schedulable compute
  cost ([[aiml-on-kubernetes]]); blue/green swap the index when done.
- Keep the embedding model, pooling, normalization, and instruction prefixes pinned together as one
  versioned unit.

---

## 7. Anti-patterns (the traps that bite in production)

- **Random / in-batch negatives only.** Accuracy plateaus. You must mine hard negatives to compete.
- **Ignoring false negatives.** Feeding raw top-k mined docs as negatives on sparsely-labeled data
  trains the model to reject correct answers. Always use positive-aware/threshold filtering (§3).
- **Evaluating on contaminated/overfit benchmarks.** Trusting an MTEB rank without a held-out in-domain
  eval. Public benchmarks generalize-check; they do not pick your model.
- **Dimension / distance-metric mismatch with the index.** Trained for cosine, indexed for L2 (or
  normalization applied on only one side) → silent ranking collapse.
- **Not re-embedding after a model change.** Mixing model versions across query and corpus. The most
  common silent regression in retrieval systems.
- **Train/serve skew in pooling, normalization, or instruction prefixes.** The geometry you trained no
  longer holds at inference.
- **Training a bi-encoder when a reranker was the real need.** If recall is fine but the top results are
  wrong, you needed a cross-encoder.
- **Tiny batches.** With contrastive loss, small batches starve the model of negatives; scale the batch
  (cross-GPU gather / GradCache) before blaming the architecture.
- **Mining negatives once and never refreshing.** Stale negatives stop being hard as the model improves.

---

## 8. Version awareness

It is 2026 and this field turns over fast. Embedding-model rankings, the "current best" open model, and
recommended hyperparameters change on the order of weeks. The **principles** here (architecture choice,
contrastive learning, hard-negative mining, false-negative removal, distillation, MRL, re-embedding
discipline) are stable; **specific model names, leaderboard positions, and exact numbers are not.**
Before quoting any benchmark figure or "SOTA" claim, **verify against the current MTEB leaderboard and
the latest papers.** Do not hardcode a hyperparameter from a paper without re-tuning on your data.

---

## 9. Canonical references (verify current; arXiv IDs checked at authoring time)

- **DPR — Dense Passage Retrieval** — arXiv:2004.04906 (in-batch negatives, dense retrieval baseline).
- **ANCE — Approximate Nearest Neighbor Negative Contrastive Learning** — arXiv:2007.00808 (iterative
  hard-negative mining via a refreshed ANN index).
- **SimCSE** — arXiv:2104.08821 (simple contrastive sentence embeddings; dropout + NLI hard negatives).
- **ColBERT** — arXiv:2004.12832; **ColBERTv2** — arXiv:2112.01488 (multi-vector late interaction).
- **E5 — Text Embeddings by Weakly-Supervised Contrastive Pre-training** — arXiv:2212.03533 (CCPairs,
  weakly-supervised contrastive pretraining, `query:`/`passage:` prefixes).
- **GTE — General Text Embeddings, multi-stage contrastive** — arXiv:2308.03281.
- **Matryoshka Representation Learning (MRL)** — arXiv:2205.13147 (nested truncatable dimensions).
- **NV-Retriever** — arXiv:2407.15831 (positive-aware / threshold-based hard-negative mining; #1 on
  MTEB-Retrieval *at publication*, July 2024 — verify current).
- **MTEB** — arXiv:2210.07316 (massive multi-task embedding benchmark; live leaderboard on Hugging Face).
- **BEIR** — arXiv:2104.08663 (zero-shot retrieval benchmark; MTEB's Retrieval subset).
- **Sentence-Transformers** — the de-facto open library for training/using bi-encoders and
  cross-encoders (`sentence-transformers` docs) — check the current version for loss/trainer APIs.

> Some arXiv IDs above were verified against arXiv at authoring time, but versions and follow-up work
> appear constantly. If an ID or claim is load-bearing for your decision, **open the paper and confirm.**

---

# Examples — Embedding & Reranker Training

Canonical, imitate-able sketches for the three highest-leverage tasks. These are correct *in spirit* and
use the standard PyTorch idioms; adapt to your framework (`sentence-transformers` has higher-level
trainers/losses that wrap exactly this — prefer them in production) and **verify API names against the
current library version** ([[ml-frameworks]], [[training-frameworks]]).

---

## 1. Contrastive bi-encoder training (InfoNCE + in-batch negatives)

A minimal-but-correct training step for a dual-encoder with mean pooling, L2 normalization, cosine
similarity, and temperature. In-batch negatives come for free from the cross-similarity matrix.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("intfloat/e5-base-v2")   # shared backbone for query & doc
tok   = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
TEMP  = 0.02                                               # tuned hyperparameter, not a constant

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def encode(texts, prefix):
    # E5-style instruction prefixes — MUST match at train and serve time.
    batch = tok([f"{prefix}{t}" for t in texts],
                padding=True, truncation=True, max_length=512, return_tensors="pt")
    out = model(**batch).last_hidden_state
    emb = mean_pool(out, batch["attention_mask"])
    return F.normalize(emb, p=2, dim=-1)                   # L2-normalize → cosine == dot

def contrastive_step(queries, positives):
    q = encode(queries,   prefix="query: ")                # [B, d]
    d = encode(positives, prefix="passage: ")              # [B, d]

    # Cosine similarity matrix; row i = query i vs every positive in the batch.
    # Diagonal = the true positive; every off-diagonal entry is an in-batch NEGATIVE.
    logits = (q @ d.T) / TEMP                              # [B, B]
    labels = torch.arange(q.size(0), device=q.device)      # correct doc index per query

    # Symmetric InfoNCE: query->doc and doc->query (helps with large batches).
    loss = 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels))
    return loss
```

Key points this sketch encodes:
- **In-batch negatives** = the off-diagonal of `q @ d.T`. Bigger batch → more negatives → better model.
  In a distributed run, **all-gather** `q` and `d` across ranks so each query sees the *global* batch's
  negatives (see [[training-frameworks]] for the gather + grad plumbing).
- **L2 normalization + temperature** must be applied identically at inference.
- **Instruction prefixes** (`query:` / `passage:`) are part of the model contract — pin them.
- To add **mined hard negatives**, extend `d` with extra hard-negative columns per query and adjust the
  logits/labels so the diagonal stays the positive (see §2 for where those negatives come from).

---

## 2. Positive-aware hard-negative mining (config note, NV-Retriever-style)

Mining turns a labeled `(query, positive)` set into `(query, positive, [hard_negatives])` training rows.
The critical step is **false-negative removal**: drop mined candidates whose relevance score is too
close to the positive's — they are probably unlabeled true positives, and training on them as negatives
teaches the model to reject correct answers.

```yaml
# hard_negative_mining.yaml  (positive-aware / threshold-based, NV-Retriever-style)
mining_model: "BAAI/bge-reranker-large"   # strong scorer (cross-encoder ideal); better scorer => better negatives
corpus_index:   "ann_index_v3"            # ANN over the corpus, encoded by the mining model
top_k_candidates: 100                      # retrieve top-100 non-positive docs per query as candidates

false_negative_removal:
  method: "positive_aware"                 # anchor on the POSITIVE's score
  # Drop any candidate scoring above this fraction of the positive's score — likely a true positive.
  # 'percentage_margin' (a.k.a. TopK-PercPos) is robust; 'absolute' uses a fixed score threshold instead.
  strategy: "percentage_margin"
  max_fraction_of_positive: 0.95           # e.g. drop candidates scoring > 0.95 * score(query, positive)
  # absolute_threshold: 0.55               # alternative: drop candidates with raw score above this

negative_sampling:
  negatives_per_query: 8
  # Sample across a RANK RANGE, not only the top — over-concentrating on the very hardest destabilizes.
  rank_range: [5, 100]
  keep_in_batch_negatives: true            # still mix easy/in-batch negatives for stability

refresh:
  iterative: true                          # ANCE-style: re-mine as the student improves
  every_n_epochs: 1                        # stale negatives stop being hard once the model improves
```

Rules of thumb (full rationale in `embedding-model-training-guide.md` §3):
- **Never** feed raw top-k as negatives on sparsely-labeled / web data — apply false-negative removal.
- Mine with a **stronger** model than the student (a cross-encoder is ideal); distill it in §3.
- **Re-mine periodically** (`iterative: true`) so negatives stay hard as the model improves.

---

## 3. Cross-encoder distillation (teacher → student bi-encoder)

The cross-encoder reranker is more accurate but cannot be indexed. Distill its scores into the cheap,
indexable bi-encoder so retrieval inherits the reranker's quality.

```python
import torch
import torch.nn.functional as F

# teacher: cross-encoder, scores a (query, doc) pair jointly -> scalar relevance (no grad).
# student: the bi-encoder from Example 1 (q and d encoders).

def distill_step(queries, docs_per_query):
    """
    docs_per_query[i] = [positive, hard_neg_1, ..., hard_neg_k]  (from Example 2's mining)
    """
    with torch.no_grad():
        teacher_scores = teacher.score_pairs(queries, docs_per_query)   # [B, 1+k]

    # Student scores via cosine of separately-encoded vectors (indexable at serve time).
    q = student.encode(queries,  prefix="query: ")                      # [B, d]
    student_scores = []
    for i, docs in enumerate(docs_per_query):
        de = student.encode(docs, prefix="passage: ")                   # [1+k, d]
        student_scores.append(q[i] @ de.T)                              # cosine (already normalized)
    student_scores = torch.stack(student_scores)                        # [B, 1+k]

    # --- Option A: KL — match the teacher's relevance DISTRIBUTION over the candidate set ---
    T = 1.0
    loss_kl = F.kl_div(
        F.log_softmax(student_scores / T, dim=-1),
        F.softmax(teacher_scores / T, dim=-1),
        reduction="batchmean",
    )

    # --- Option B: MarginMSE — match the teacher's positive-vs-negative MARGIN (per negative) ---
    s_pos, t_pos = student_scores[:, :1], teacher_scores[:, :1]
    s_neg, t_neg = student_scores[:, 1:], teacher_scores[:, 1:]
    loss_margin = F.mse_loss(s_pos - s_neg, t_pos - t_neg)

    # Often combined with the InfoNCE loss from Example 1.
    return loss_kl  # or loss_margin, or a weighted sum with contrastive loss
```

Why this is high-leverage:
- The student stays a **bi-encoder** (separate encodes, cosine) → fully **indexable** for first-stage
  retrieval, while inheriting much of the cross-encoder's ranking quality.
- **MarginMSE** is robust and widely used; **KL** transfers the full ranking distribution. Try both.
- Distill on the **mined positives + hard negatives** from §2 — the candidates where the teacher's
  judgment is most informative.
- You can also distill a large cross-encoder into a **smaller cross-encoder** for a cheaper rerank stage
  ([[fine-tuning-peft]] for LoRA/adapter backbones; [[recsys-ranking]] for the analogous rank stage).
