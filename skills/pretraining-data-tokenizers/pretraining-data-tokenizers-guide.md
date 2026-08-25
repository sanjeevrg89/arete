# Pretraining Data & Tokenizers — Full Reference

The single source of truth for this skill. Two tightly coupled subsystems: the **data pipeline** that
turns raw web crawls into a clean, deduplicated, mixed token stream, and the **tokenizer** that defines
the vocabulary that stream is encoded in. Both are decided *before* the expensive training run and are
effectively immutable afterward, which is exactly why they deserve frontier-level care.

> Versions/numbers move fast (it is 2026). Treat specific duplicate fractions, dataset sizes, classifier
> recipes, and tokenizer-free results as **verify against current docs/papers** — the *methods* below
> are stable; the *figures* are not.

---

## 1. Mental model: data is the model

At a fixed training-compute budget (Chinchilla-style optimal token counts), the dominant lever on final
quality is **what tokens you train on**, not minor architecture tweaks. The same model architecture
trained on raw Common Crawl vs. a well-curated derivative differs dramatically on downstream evals. This
is the empirical lesson behind every modern open corpus (RefinedWeb, CCNet, the Pile, Dolma,
DataComp-LM, FineWeb). So:

- **Treat data work as experimentation, not ETL.** Each filter, threshold, dedup setting, and mixture
  weight is a hypothesis. Validate it by training a small "ablation" model on a fixed token budget and
  reading downstream metrics — *not* by eyeballing a few documents. DataComp-LM (arXiv:2406.11794) is
  built around exactly this: a fixed training recipe and eval suite so the *data* is the only variable.
- **Quality and quantity trade off.** Aggressive filtering raises average quality but shrinks the pool;
  past a point you must repeat data (epoching), which has diminishing and eventually negative returns
  (data-constrained scaling). Find the operating point empirically.
- **Cost compounds.** A bad tokenizer adds a multiplicative tax to *every* token in *every* run and
  *every* inference call forever. A contamination bug invalidates an entire eval campaign. Mistakes here
  are among the most expensive in the whole stack.

### The scale you are operating at

A frontier web corpus is built from **dozens of Common Crawl snapshots**, each on the order of a few
hundred TB of WARC, comprising **millions of WARC files** and **billions of documents**. After the full
pipeline you keep a small, high-value fraction. This is a petabyte-scale distributed-systems problem
before it is an ML problem.

---

## 2. The curation pipeline (stage by stage)

Order is load-bearing. Canonical sequence:

```
raw WARC ──▶ text extraction ──▶ language ID ──▶ quality filtering ──▶ toxicity/PII
       ──▶ deduplication ──▶ benchmark decontamination ──▶ tokenize + shuffle + pack ──▶ training
```

Do the cheap, high-recall stages first (extraction, language ID, heuristics) to shed the bulk volume,
then the expensive stages (model-based filters, fuzzy dedup) on the survivors. **Dedup after filtering**
(don't waste dedup compute on junk you'll drop) and **decontaminate last** (so nothing reintroduces eval
text). Carry a provenance record through every stage.

### 2.1 Text extraction (WARC → text)

WARC/WET records are HTML; you need clean main-content text, not nav bars, cookie banners, ads, and
footers. Two practical approaches:

- **WET files** (Common Crawl's pre-extracted plain text) — cheap but low quality; lots of boilerplate.
  CCNet (arXiv:1911.00359) builds from WET and compensates downstream.
- **Custom extraction from WARC** using **trafilatura** or **resiliparse** for boilerplate removal and
  main-content detection. This is what RefinedWeb and FineWeb do, and it is a major quality win over WET.
  resiliparse is notably faster; trafilatura tends to extract cleaner article text. Benchmark both on
  your traffic.

Strip markup, normalize Unicode, drop documents with almost no extractable content.

### 2.2 Language identification

Run a fast LID model (fastText `lid.176` is the common baseline; newer multilingual LID models exist —
verify current). Keep a per-document language label and a confidence score. **Do not hard-filter to one
language blindly**: language-blind English-only filters silently destroy multilingual capability and
mis-handle code (which LID often mislabels). Route documents by language so each can get
language-appropriate quality filters and thresholds.

### 2.3 Quality filtering (a cascade, cheapest first)

Layer filters from cheap/high-recall to expensive/high-precision:

1. **Heuristic rules** (Gopher/RefinedWeb-style): document length, mean word length, fraction of
   alphabetic characters, symbol-to-word ratio, bullet/ellipsis ratios, fraction of lines ending in
   punctuation, repetition metrics (duplicate line/paragraph fraction, top n-gram fraction), and
   bad-word/blocklist ratios. These remove machine-generated spam, listicles, and degenerate text.
2. **Repetition filters** specifically catch within-document boilerplate and SEO spam — high duplicate
   n-gram fractions are a strong junk signal.
3. **Perplexity filtering (KenLM)** — CCNet trains a KenLM n-gram LM on a clean reference corpus
   (e.g. Wikipedia) per language and scores documents by perplexity; low-perplexity = closer to the
   reference distribution. Bucket into head/middle/tail rather than hard-cutting.
4. **Classifier-based filtering** — train a lightweight classifier (logistic regression on n-grams, a
   fastText classifier, or a small transformer) to score "quality." FineWeb-Edu trains a classifier on
   LLM-generated *educational-value* labels and keeps high-scoring docs; this was a large quality win.
   DataComp-LM showed a well-tuned fastText quality classifier is a strong, cheap baseline.
5. **Model-based / LLM-as-judge filtering** — use an LLM to score or rewrite documents. Powerful but
   expensive and can bias the corpus toward the judge's preferences; use sparingly and ablate.

**Tune thresholds with ablations.** Over-filtering is a real failure mode: you can filter your way to a
small, homogeneous, lower-diversity corpus that underperforms. Measure, don't assume.

### 2.4 Toxicity and PII removal

- **Toxicity**: classifier-based filtering of hate/abuse/explicit content per policy. Be aware it
  interacts with bias (over-filtering can erase dialects/identities) — see `[[responsible-ai-governance]]`.
- **PII**: detect and redact emails, phone numbers, and similar (regex + NER). Reduces memorization and
  leakage of personal data and is increasingly a legal/compliance requirement.

### 2.5 Deduplication (see §3 for the math)

The single highest-leverage stage after extraction. Web data has enormous near-duplication (mirrors,
templated pages, reposts, syndicated news). Removing it reduces memorization, improves sample
efficiency, and prevents accidental eval leakage.

### 2.6 Benchmark decontamination

Remove training documents that overlap items in your evaluation suites (MMLU, GSM8K, HumanEval, etc.).
Match by n-gram overlap (e.g. flag a doc if it shares a long contiguous n-gram with an eval item) or
substring/suffix-array containment. **Do this last and against the exact eval set you will report on.**
Contamination is the most common way published numbers become meaningless.

### 2.7 Tokenize, shuffle, pack

Encode with the frozen tokenizer (§5–7), globally shuffle, and pack into fixed-length sequences for the
trainer. Sequence packing and the data loader belong to `[[training-frameworks]]`.

---

## 3. Deduplication in depth

### 3.1 Exact dedup

Hash each document (or normalized document) and drop collisions. Cheap, catches verbatim copies only.
Often done first as a coarse pass. Line-level exact dedup also removes ubiquitous boilerplate lines.

### 3.2 Fuzzy dedup with MinHash + LSH

Catches near-duplicates (small edits, different boilerplate around the same article). The standard
pipeline:

1. **Shingling**: represent each document as the set of its **n-gram shingles** (word n-grams; **5-gram
   is the common choice**). This turns "similar text" into "overlapping sets."
2. **MinHash signature**: apply *k* hash functions (or a single permutation scheme) and keep the minimum
   hash per function → a fixed-length **signature** of dimension *k* (e.g. 128–256). The probability two
   docs agree on any signature position equals their **Jaccard similarity** of shingle sets — so a short
   signature estimates Jaccard cheaply.
3. **LSH banding**: split the *k*-dim signature into *b* **bands** of *r* rows each (`k = b·r`). Two docs
   are *candidate* duplicates if they match exactly in **any** band. The probability of becoming a
   candidate is `1 − (1 − s^r)^b` where *s* is true Jaccard — an S-curve with threshold ≈ `(1/b)^(1/r)`.
   Choose *b*, *r* to put the steep part of the curve at your target similarity (e.g. ~0.8).
4. **Cluster & remove**: group candidates (union-find / connected components) and keep one
   representative per cluster.

This runs **batched over tens of millions to billions of documents** on Spark or Ray. The compute is
dominated by signature generation and the shuffle that brings same-band buckets together. Per-snapshot
duplicate fractions are frequently **large** (a substantial share of documents are near-dupes) — exact
fractions vary by snapshot and recipe, so **verify against current docs**.

Key knobs (see `examples.md` for a concrete config):

- **shingle/n-gram size** — smaller n (e.g. 3) raises recall but over-merges; 5-gram is the usual sweet
  spot for documents.
- **signature dimension *k*** — higher = more accurate Jaccard estimate, more memory/compute.
- **bands *b* and rows *r*** — set the similarity threshold and the precision/recall tradeoff.

### 3.3 Suffix-array / substring dedup

Instead of document-level similarity, find and remove **exact repeated substrings** above a length
threshold (e.g. 50+ tokens) across the whole corpus using a suffix array. Complementary to MinHash:
catches long copied spans embedded in otherwise different documents (the "exact-substring dedup"
technique from the deduplication-improves-LMs line of work). Memory-heavy; run on sharded data.

### 3.4 Semantic dedup

Embed documents and remove near-duplicates in embedding space (cluster, drop points within a cosine
threshold of a kept centroid). Catches paraphrases and translations that share little surface n-gram
overlap. More expensive (needs an embedding model + ANN index); use as a later, optional pass when
surface dedup has plateaued.

### 3.5 Why it matters

No/weak dedup → the model **memorizes** repeated text (privacy + regurgitation risk), wastes capacity
and compute on redundant tokens, and **leaks evals** when duplicated content overlaps benchmarks. Dedup
is consistently one of the highest-ROI stages.

---

## 4. Data mixtures, curricula, and repetition

A clean corpus is necessary but not sufficient — *how you compose it* is its own optimization.

- **Domain weighting**: web text alone is not enough. Mix in code, math, books, scientific text,
  curated/reference sources, and multilingual data. Weights are tuned by ablation (and methods like
  learned mixture optimization — verify current). Code and math notably improve reasoning even on
  non-code evals.
- **Upsampling / epoching**: high-value but scarce sources (curated reference, some code/math) are often
  repeated more than abundant web data. Track effective epochs per source.
- **Data-constrained scaling & repetition**: when you've exhausted unique high-quality tokens, repeating
  data trades off against fresh data. Up to a few epochs is roughly as good as new data; beyond that,
  returns fall sharply and overfitting/memorization rise (the data-constrained-scaling-laws result —
  verify current numbers).
- **Curriculum / phased training**: many recipes change the mixture over training (e.g. a final "anneal"
  or cooldown phase weighted toward the highest-quality and instruction-like data). This belongs jointly
  to data and `[[training-frameworks]]`.
- **Synthetic data**: LLM-generated text (rephrasing, textbook-style generation, problem/solution pairs)
  can boost quality and fill gaps, but risks **mode collapse / diversity loss** and propagating the
  generator's biases. Always ablate against a real-data baseline; never let synthetic dominate blindly.
- **Quality vs. quantity**: the central tension. More aggressive curation → higher per-token value but
  fewer tokens. The right point depends on model scale and compute budget; decide with ablations.
- **Provenance & licensing**: every document should carry source, snapshot, license signal, and consent
  status. This constrains *which* data can enter the mixture and is an audit/compliance requirement, not
  an afterthought — see `[[responsible-ai-governance]]`.

---

## 5. Reference pipelines — what each taught

| Pipeline | Core idea | What it taught |
|---|---|---|
| **The Pile** | Curated mixture of 22 diverse sources (web, books, code, academic) | Diverse, *documented* domain mixtures beat undifferentiated web; provenance matters. |
| **CCNet** (arXiv:1911.00359) | WET → LID → KenLM perplexity buckets → dedup, per language | Reproducible, language-aware, perplexity-based filtering of Common Crawl at scale. |
| **RefinedWeb** | High-quality web-*only* via strong extraction (trafilatura) + heuristics + dedup | Web alone, properly filtered & deduped, can rival curated mixtures. |
| **Dolma** | Large open corpus **with an open toolkit** and documented decisions | Reproducibility and open tooling/provenance for the whole pipeline. |
| **DataComp-LM** (arXiv:2406.11794) | A **benchmark**: fixed training/eval, compete on data filtering | Data curation as a measurable, comparable science; fastText quality classifier is a strong baseline. |
| **FineWeb / FineWeb-Edu** | Carefully ablated CC pipeline; **edu-quality classifier** for FineWeb-Edu | Each filter validated by ablation; an educational-value classifier yields large downstream gains. |

A good cross-cutting reference on data curation methods at scale is **arXiv:2407.12481** (data curation
for large-scale training) — *verify the exact title/scope against the current listing.* Always confirm
arXiv IDs and dataset details against current sources before citing externally.

---

## 6. Tokenizers — algorithms and design

The tokenizer maps raw bytes/characters to integer IDs. It is **trained on a representative sample of
your corpus** and then **frozen for the entire pretraining run** — you cannot change the vocabulary
afterward without retraining embeddings (and effectively the model). Decisions here are permanent.

### 6.1 Algorithms

| Algorithm | How it builds vocab | Notes |
|---|---|---|
| **BPE** (byte-pair encoding) | Greedily merge the most frequent adjacent pair, repeatedly, until vocab size | Dominant for LLMs. Deterministic encoding. |
| **Byte-level BPE** | BPE over **raw bytes** (256 base symbols) instead of Unicode chars | **The default for general LLMs** (GPT-2 onward). No `<UNK>` ever — any byte sequence is representable. Robust to emoji, rare scripts, code. |
| **Unigram LM** | Start large, prune tokens to maximize corpus likelihood under a unigram model | SentencePiece default; supports probabilistic/subword-regularized segmentation. Often slightly better fertility than BPE. |
| **WordPiece** | Likelihood-based merges (BERT family) | Mostly encoder/legacy; rare for new decoder LLMs. |

For a new general-purpose LLM, **byte-level BPE** is the safe default (universal coverage, no UNK, good
on code and multilingual). Unigram is a reasonable alternative when you want subword regularization or
marginally better compression.

### 6.2 Libraries

- **SentencePiece** — trains BPE or Unigram directly on raw text (language-agnostic, treats input as a
  raw byte/char stream, encodes whitespace as `▁`). Common for multilingual models.
- **HF `tokenizers`** (Rust) — fast byte-level BPE/Unigram/WordPiece with explicit
  normalizer/pre-tokenizer/model/decoder pipeline; the standard for training and shipping tokenizers in
  the HF ecosystem.
- **tiktoken** — fast byte-level BPE *encoder* (the GPT-family tokenizers). Encode/decode at inference;
  not a trainer.

### 6.3 Vocabulary size tradeoffs

- **Larger vocab** → fewer tokens per document (lower fertility, longer effective context, cheaper
  train/inference per word) **but** a larger embedding + output softmax matrix (more parameters, more
  memory, rarer tokens seen less often). Typical modern LLMs use **~32k–256k+**; recent large/multilingual
  models trend bigger — **verify current** norms.
- The output softmax cost scales with vocab; very large vocabs may need tied embeddings and care in the
  loss. Pick vocab by ablating fertility vs. parameter/throughput cost on *your* data mix.

### 6.4 Normalization, digits, whitespace, special tokens

- **Normalization**: choose Unicode normalization (NFC/NFKC) deliberately; NFKC folds more aggressively
  (can merge visually-distinct chars) — be cautious with code and multilingual text. Lowercasing is
  almost never done for modern LLMs (loses information).
- **Digits**: split numbers into single digits or fixed 3-digit groups so arithmetic and numeric
  reasoning aren't fragmented by frequency-driven merges. A deliberate digit policy measurably helps math.
- **Whitespace**: keep detokenization **lossless** — SentencePiece's `▁` meta-symbol or byte-level
  space handling. Decide whether leading spaces attach to the following token (they usually do in
  byte-level BPE). Indentation/tabs matter enormously for **code** — don't collapse them.
- **Special / control tokens**: reserve BOS/EOS/PAD, chat/role markers, tool/FIM (fill-in-the-middle for
  code) tokens, and a block of spare reserved IDs **up front**. Adding special tokens after pretraining
  means untrained embeddings. Reserve generously.

### 6.5 Multilingual and code

- A tokenizer trained mostly on English fragments other languages into many tokens (high fertility),
  which wastes context and inflates cost for those users. For multilingual targets, **train on a
  language-balanced sample** (upsample low-resource languages in the tokenizer-training corpus) so
  fertility is acceptable across the board. A **custom tokenizer can cut fertility substantially** vs. a
  general English-centric one on the target languages.
- **Code** needs whitespace/indentation preservation, sensible handling of identifiers and operators,
  and often FIM tokens. Include representative code in tokenizer training if the model targets code.

### 6.6 Tokenizer-free / byte & patch models

Active research replaces subword tokenization with byte- or patch-level models (e.g. byte-level
transformers and dynamic-patching approaches like the byte-latent line of work) to avoid tokenizer
brittleness entirely. Promising but not yet the universal default; **verify the current state of the
art** before betting a frontier run on it.

---

## 7. Fertility — the metric that determines cost

**Fertility = tokens per word** (or tokens per character / per byte) when a tokenizer encodes a text.
It is the compression ratio of your vocabulary on real data, and it is **directly multiplicative on
cost**:

- More tokens per word → more positions per document → **more FLOPs in training and inference**, more KV
  cache, and **less real content per fixed context window**. A tokenizer with fertility 1.5 vs 1.2 on
  your traffic is ~25% more expensive per word, forever.
- Fertility is **per-language and per-domain**. An English-centric tokenizer can be 2–4x more fertile on
  some other languages and on code than on English. This is an equity and cost issue: users of
  under-tokenized languages pay more and get less context.
- **Measure before you freeze.** Encode held-out text per language/domain and compute tokens-per-word
  (and bytes-per-token) for each candidate tokenizer. Compare against the model's target language mix.
  Choose vocab size, training-corpus balance, and algorithm to keep fertility low across all target
  languages — see `examples.md` for the comparison procedure.

---

## 8. Tooling & scale

- **Distributed frameworks**: **datatrove** (purpose-built for LLM-data pipelines; the FineWeb tooling),
  **Spark**, and **Ray** are the workhorses for extraction, filtering, and MinHash dedup across millions
  of files. Stages should be idempotent and checkpointed so a failure doesn't restart petabytes.
- **Storage formats**: sharded columnar/row formats — **Parquet** for intermediate document tables,
  WARC/WET for raw input, and a packed binary format (e.g. memory-mappable token shards / WebDataset
  tar shards) for the final tokenized stream. Keep shards a sensible size (hundreds of MB–few GB) for
  parallel reads.
- **Throughput**: extraction (HTML parsing) and MinHash signature generation are usually the hotspots;
  the dedup shuffle is I/O-bound. Profile; don't guess.
- **Cost of getting it wrong**: a filter bug that drops good data, a dedup misconfiguration that merges
  distinct docs, or contamination discovered after training all cost from thousands of accelerator-hours
  to a full re-run. Gate stages with metrics and small-model ablations *before* the full run.

---

## 9. Anti-patterns (the traps that bite in production)

- **No / weak deduplication** → memorization, regurgitation, wasted compute, and accidental eval leakage
  via duplicated content. Always run at least exact + MinHash-LSH dedup.
- **Benchmark contamination** → inflated, meaningless eval numbers. Decontaminate against the *exact*
  eval suite you report, as the last data stage.
- **Language-blind filtering** → English-tuned heuristics/thresholds silently gut other languages and
  mislabel code. Route and filter per language.
- **High-fertility tokenizer for target languages** → permanent cost/quality tax and inequitable
  treatment of some languages. Measure fertility per language before freezing.
- **No provenance / licensing tracking** → you cannot audit, honor takedowns, or reproduce; you may
  train on data you're not allowed to. Carry provenance end-to-end.
- **Train/eval leakage** beyond named benchmarks (e.g. test sets of your own downstream tasks). Hold out
  and decontaminate everything you'll evaluate on.
- **Over-filtering to a small homogeneous corpus** → loses diversity and underperforms; tune with
  ablations, don't max out every filter.
- **Adding special tokens after pretraining** → untrained embeddings and degraded behavior. Reserve them
  up front.
- **Eyeballing instead of ablating** → human spot-checks miss distributional problems. Validate data
  changes by training small models on a fixed budget and reading metrics.
- **Naive whitespace/digit handling** → lossy detokenization, broken code indentation, fragmented
  arithmetic. Decide these policies deliberately.

---

## 10. Troubleshooting (symptom → likely cause → fix)

- **Model memorizes/regurgitates long passages** → insufficient dedup → add MinHash-LSH + substring
  dedup; check duplicate fractions per snapshot.
- **Suspiciously high benchmark scores / scores collapse on a fresh held-out set** → contamination →
  re-run decontamination against the eval suite; check n-gram/substring overlap.
- **Multilingual eval much worse than English at similar data share** → high tokenizer fertility on those
  languages and/or LID dropping them → measure fertility per language; rebalance tokenizer training and
  the data mix.
- **Inference/training cost higher than budgeted per document** → fertility higher than assumed → audit
  tokens-per-word on real traffic; consider a larger or rebalanced vocab.
- **Quality plateaus despite more data** → diminishing returns from low-quality tail or over-repetition →
  tighten quality filter / check effective epochs per source (data-constrained regime).
- **Corpus feels homogeneous / model lacks breadth** → over-filtering or synthetic-data overuse → relax
  thresholds; cap synthetic share; restore source diversity.
- **Pipeline runs out of memory on dedup** → MinHash bucket skew / suffix-array size → shard more
  finely; tune band count; bound cluster sizes.
- **Garbled detokenization or broken code** → normalization too aggressive (NFKC) or lossy whitespace →
  switch to NFC/byte-level lossless handling; preserve indentation.

---

## 11. Canonical references (verify current before citing)

- CCNet: *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data* — **arXiv:1911.00359**.
- DataComp-LM: *DataComp-LM: In search of the next generation of training sets for language models* —
  **arXiv:2406.11794**.
- Data curation at scale — **arXiv:2407.12481** (confirm exact title/scope against the current listing).
- FineWeb / FineWeb-Edu — HuggingFace dataset cards + the FineWeb technical report/blog (search current).
- RefinedWeb, The Pile, Dolma — each has a paper and/or dataset card; confirm current IDs/URLs.
- Deduplication-for-LMs and exact-substring dedup, data-constrained scaling laws, byte-/patch-level
  tokenizer-free models — **look up current arXiv IDs**; do not cite from memory.
- Tooling: SentencePiece, HF `tokenizers`, `tiktoken`, `datatrove`, `trafilatura`, `resiliparse`,
  KenLM, fastText LID — consult each project's current docs.

Always confirm arXiv IDs, dataset sizes, duplicate fractions, vocab norms, and tokenizer-free results
against current sources. The methods are durable; the numbers are not.
