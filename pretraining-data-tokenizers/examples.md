# Examples — Pretraining Data & Tokenizers

Canonical artifacts to imitate. Numbers are illustrative starting points to **tune by ablation**, not
universal constants. Library APIs move fast (it is 2026) — verify against current docs.

---

## 1. MinHash-LSH deduplication config sketch

Goal: remove near-duplicate documents (target Jaccard similarity ≈ 0.8) across tens of millions to
billions of documents. The S-curve probability that two docs become candidates is
`P(candidate) = 1 − (1 − s^r)^b`, with approximate threshold `s* ≈ (1/b)^(1/r)`. Choose `b` (bands) and
`r` (rows/band) so the steep part of the curve sits at your target similarity.

```yaml
# minhash_dedup.yaml — illustrative; tune b/r and threshold by ablation
shingling:
  unit: word            # word n-grams (not chars) for document-level dedup
  ngram_size: 5         # 5-gram shingles: standard sweet spot (3 = higher recall, over-merges)
  lowercase: true       # normalize before shingling so trivial case diffs collapse
  strip_whitespace: true

minhash:
  num_permutations: 128 # signature dimension k = b * r = 14 * 9 = 126 -> use 126 (or pad to 128)
  hash_bits: 64
  seed: 42

lsh:
  bands: 14             # b
  rows_per_band: 9      # r   => k = 126; threshold s* ≈ (1/14)^(1/9) ≈ 0.75  (near target 0.8)
  match_rule: any_band  # candidate if exact match in ANY band

clustering:
  algorithm: union_find # connected components over candidate pairs
  keep: one_representative_per_cluster
  representative: longest # or: highest-quality-score / earliest-snapshot

scale:
  engine: datatrove     # or spark / ray
  input: parquet_shards # sharded document tables
  parallelism: per_shard_signatures_then_shuffle_by_band
  checkpoint: true      # signatures and band-buckets are checkpointed; never restart petabytes
```

Notes:
- **Tune the threshold to your goal**: lower `s*` (more bands / fewer rows) merges more aggressively
  (higher recall, more false merges); higher `s*` is conservative. Validate by sampling merged clusters.
- The **shuffle that groups same-band buckets** is the I/O hotspot; signature generation is the CPU
  hotspot. Watch for **bucket skew** (a few huge buckets) — cap cluster sizes or sub-shard.
- Run **exact (hash) dedup first** as a cheap coarse pass, then MinHash-LSH, then **suffix-array /
  substring dedup** for long copied spans embedded in otherwise-distinct documents. Optional **semantic
  (embedding) dedup** last when surface dedup has plateaued.

---

## 2. Quality-filter cascade outline

Order: cheapest/highest-recall first, expensive/highest-precision last. Each stage emits a kept/dropped
decision **plus the reason**, recorded in provenance.

```
Stage 0 — Extraction sanity
  drop if: extracted_text empty / < N chars after boilerplate removal (trafilatura/resiliparse)

Stage 1 — Language ID (fastText lid.176 or current LID)
  tag: language, confidence
  route per language (do NOT hard-drop to English-only blindly); drop only very-low-confidence noise

Stage 2 — Heuristic rules (Gopher / RefinedWeb style), per language
  drop if:
    - doc length outside [min, max] words
    - mean_word_length outside ~[3, 10]
    - symbol_to_word_ratio  > ~0.10   (# / … heavy)
    - fraction_alpha_chars  < ~0.80
    - fraction_lines_ending_in_punctuation < ~0.30   (listicle/nav signal)
    - duplicate_line_fraction or top_2gram_fraction high   (repetition / SEO spam)
    - bad_word / blocklist ratio over threshold

Stage 3 — Perplexity bucketing (CCNet: per-language KenLM trained on a clean reference corpus)
  bucket: head / middle / tail  (keep head+middle; tail often dropped or downweighted — ablate)

Stage 4 — Classifier-based quality (DataComp-LM fastText classifier / FineWeb-Edu edu-value classifier)
  score in [0,1]; keep score >= threshold (threshold set by ablation, NOT max precision)

Stage 5 — Toxicity / PII
  drop or redact: hate/abuse/explicit per policy; redact emails/phones/PII (regex + NER)

# THEN (separate stages, after filtering):
#   -> deduplication (exact + MinHash-LSH + substring)
#   -> benchmark decontamination (n-gram/substring overlap vs. the EXACT eval suite)
```

Each threshold above is a **knob to ablate**, not a constant. Over-filtering yields a small, homogeneous
corpus that underperforms — validate the whole cascade by training a small model on a fixed token budget
and reading downstream metrics.

---

## 3. Tokenizer-fertility comparison note

**Fertility = tokens per word.** It is multiplicative on train + inference FLOPs, KV cache, and how much
real content fits in the context window — so measure it **per language and per domain** on held-out text
*before freezing the tokenizer*.

Procedure (illustrative):

```python
# fertility.py — compare candidate tokenizers on held-out text, per language/domain
from transformers import AutoTokenizer  # or sentencepiece / tiktoken

def fertility(tok, texts):
    n_tokens = sum(len(tok.encode(t)) for t in texts)
    n_words  = sum(len(t.split()) for t in texts)     # whitespace words; report bytes/token too
    return n_tokens / max(n_words, 1)

candidates = {
    "english_centric_32k": AutoTokenizer.from_pretrained("..."),
    "multilingual_balanced_128k": AutoTokenizer.from_pretrained("..."),
}
heldout = {                # representative held-out samples per target slice
    "en": [...], "de": [...], "hi": [...], "code": [...],
}
for name, tok in candidates.items():
    for lang, texts in heldout.items():
        print(name, lang, round(fertility(tok, texts), 3))
```

Illustrative shape of results (**verify on your own data — do not treat as fixed**):

| Language / domain | English-centric vocab | Balanced multilingual vocab |
|---|---|---|
| English | low (≈ baseline)           | low (≈ baseline) |
| High-resource non-English | moderately higher | close to English |
| Low-resource language | **much higher (2–4x)** | substantially reduced |
| Code | higher (indentation/identifiers fragmented) | reduced if code in tokenizer-training mix |

How to act on it:
- If fertility on a **target** language/domain is 2–4x the English baseline, that slice pays a permanent
  cost tax and gets less effective context. **Rebalance the tokenizer-training corpus** (upsample those
  languages) and/or **increase vocab size**, then re-measure.
- Trade fertility against the embedding/softmax parameter and memory cost of a bigger vocab — pick the
  point that minimizes total cost across your **target** language mix, not just English.
- Confirm digit policy (split digits) and whitespace handling (`▁` / byte-level, lossless detokenize,
  preserved code indentation), and that special/control tokens (BOS/EOS/PAD, chat roles, FIM, spare
  reserved IDs) are present — **before** freezing for the run.
