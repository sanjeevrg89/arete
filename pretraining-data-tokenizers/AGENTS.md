# AGENTS.md — Pretraining Data & Tokenizers

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`pretraining-data-tokenizers-guide.md`** next to this
> file — read it before building or reviewing a pretraining-data pipeline or a tokenizer, and apply it.
> Concrete artifacts to imitate (MinHash-LSH config, quality-filter cascade, fertility comparison) are
> in **`examples.md`**. This file is the always-on summary.
>
> **Premise: data is the model.** At fixed compute, the corpus and the tokenizer explain most of the
> variance in final quality and a large share of serving cost. This field moves fast (it is 2026) —
> treat specific duplicate fractions, dataset sizes, vocab norms, and tokenizer-free results as
> **verify against current docs/papers**; never fabricate numbers.

## When working on pretraining data or tokenizers, apply by default:

- **Validate data changes by ablation, not by eyeballing.** Each filter/threshold/mixture/dedup setting
  is a hypothesis: train a small model on a fixed token budget and read downstream metrics.
- **Pipeline order is load-bearing:** extract (WARC→text, boilerplate removal via trafilatura/
  resiliparse) → language ID → quality filter (heuristic → perplexity/classifier/model-based) →
  toxicity/PII → **dedup** → **decontaminate** → tokenize/shuffle/pack. Dedup *after* filtering;
  decontaminate *last*, against the exact eval suite you report on.
- **Deduplicate always.** Exact (hash) + **MinHash-LSH** over **5-gram** shingles (signature dim ~128–256,
  banded to put the S-curve at your target similarity) + suffix-array/substring dedup for long repeated
  spans; semantic dedup as a later optional pass. No dedup → memorization, wasted compute, eval leakage.
- **Quality filtering is a cascade, cheapest first:** heuristics (length, symbol/word ratio, repetition,
  blocklists) → KenLM perplexity buckets (CCNet) → fastText/edu quality classifier (DataComp-LM,
  FineWeb-Edu). Tune thresholds with ablations; over-filtering to a small homogeneous corpus is a real
  failure mode.
- **Mixtures, upsampling, and repetition are first-class knobs.** Weight code/math/multilingual/curated
  sources; track effective epochs per source; respect data-constrained scaling (a few epochs ≈ new data,
  beyond that returns fall). Ablate synthetic data — it can collapse diversity.
- **The tokenizer is frozen at pretraining.** Choose algorithm (**byte-level BPE** is the general
  default; Unigram/SentencePiece for subword regularization or marginal compression), vocab size,
  normalization, digit and whitespace policy, and **reserve special/control tokens up front**. Always
  have a byte fallback so nothing is unrepresentable (no `<UNK>`).
- **Fertility (tokens-per-word) is multiplicative cost — measure before freezing.** Compute
  tokens-per-word per language and domain on held-out text for each candidate tokenizer. Multilingual
  targets often need a language-balanced custom tokenizer to avoid 2–4x fertility blowup vs. an
  English-centric one. High fertility = permanent train + inference tax and inequitable treatment.
- **Handle digits and whitespace deliberately:** split digits (single or 3-digit groups) for arithmetic;
  keep detokenization lossless (SentencePiece `▁` / byte-level space); preserve code indentation
  (avoid aggressive NFKC).
- **Track provenance and licensing per document end-to-end** (source URL, snapshot, filter decisions,
  license/consent). Required for audits, takedowns, reproducibility, and policy — see
  `[[responsible-ai-governance]]`.
- **Treat scale as a distributed-systems problem:** datatrove/Spark/Ray, sharded Parquet + packed token
  shards, idempotent checkpointed stages. A stage bug costs thousands of accelerator-hours or a re-run.

## Anti-patterns to flag
No/weak dedup · benchmark contamination · language-blind filtering · high-fertility tokenizer for target
languages · no provenance/licensing tracking · train/eval leakage · over-filtering to homogeneity ·
adding special tokens after pretraining · naive whitespace/digit handling · deciding by eyeball not ablation.

## Definition of done for a corpus/tokenizer change
Dedup + decontamination run and quantified · per-language/domain fertility measured against target mix ·
filter/mixture change validated by a small-model ablation on a fixed budget · provenance/license carried
through · special tokens reserved before freeze · numbers cited only from current sources (else flagged
"verify"). Full rationale and references in `pretraining-data-tokenizers-guide.md`.
