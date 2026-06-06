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
