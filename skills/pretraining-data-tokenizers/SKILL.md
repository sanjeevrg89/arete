---
name: pretraining-data-tokenizers
description: World-class guidance for building petabyte-scale LLM pretraining-data pipelines and the
  tokenizers that sit in front of them — the curation and vocabulary decisions that quietly set model
  quality and serving cost. Use when curating web-scale corpora (Common Crawl / WARC extraction, language
  ID, quality filtering, dedup, decontamination), deciding data mixtures/curricula/repetition, or
  designing/evaluating a tokenizer (BPE/Unigram/WordPiece, byte-level BPE, SentencePiece/tiktoken/HF
  tokenizers, vocab size, fertility/tokens-per-word, multilingual & code). Covers FineWeb/FineWeb-Edu,
  RefinedWeb, CCNet, DataComp-LM, Dolma, The Pile, MinHash-LSH dedup, datatrove/Spark/Ray at scale, and
  the anti-patterns (no dedup, benchmark contamination, language-blind filtering, high-fertility
  tokenizers, no provenance) that waste compute and leak evals.
---

# Pretraining Data & Tokenizers

Apply the judgment of an engineer who has shipped multiple frontier pretraining corpora and the
tokenizers in front of them: **data is the model.** Architecture and the training loop matter, but at a
fixed compute budget the corpus and the vocabulary explain most of the variance in final quality and a
large fraction of serving cost. Treat every stage as a measurable experiment, not a one-time ETL job.

## How to use this skill

1. **Read `pretraining-data-tokenizers-guide.md`** in this directory — the full reference (pipeline
   stages, dedup math, mixtures, tokenizer design, scale, troubleshooting). Apply it to the task.
2. For concrete artifacts to imitate — a MinHash-LSH dedup config, a quality-filter cascade, and a
   tokenizer-fertility comparison — read **`examples.md`**.
3. Match the surrounding pipeline's conventions (storage layout, framework, naming) where they differ
   on style; apply the correctness rules — dedup before training, decontaminate against your eval
   suite, track provenance, freeze the tokenizer only after fertility checks — regardless.

This is a fast-moving field (it is 2026). Reference-pipeline details, exact duplicate fractions, and
tokenizer-free results change; treat specific numbers as "verify against current docs/papers."

## The essentials (full rationale in `pretraining-data-tokenizers-guide.md`)

- **The corpus dominates.** Spend ablation budget on data, not just architecture. Every filter and
  mixture change is a hypothesis you validate by training a small model and reading downstream metrics —
  never by eyeballing samples alone.
- **Pipeline order matters.** Extract (WARC→text, boilerplate removal) → language ID → quality filter
  (heuristic → classifier/perplexity/model-based) → toxicity/PII → **dedup** → **decontaminate** →
  tokenize/shuffle/pack. Dedup *after* filtering, decontaminate *last*.
- **Deduplicate aggressively.** Web snapshots are heavily duplicated. Exact (hash) catches verbatim
  copies; **MinHash-LSH** over 5-gram shingles catches near-duplicates; suffix-array/substring dedup
  catches repeated spans. No dedup → memorization, wasted compute, and benchmark leakage.
- **Decontaminate against *your* eval set.** Remove train docs overlapping eval items (n-gram or
  substring match). Contamination silently inflates scores and invalidates every comparison.
- **Quality filtering is a cascade, cheapest first.** Heuristics (length, symbol/word ratios,
  repetition, bad-word lists) → language ID → classifier (e.g. an edu/quality classifier as in
  FineWeb-Edu) or perplexity (KenLM, as in CCNet). Tune thresholds with ablations; over-filtering hurts.
- **Mixtures and repetition are first-class knobs.** Domain weights, upsampling of high-value sources
  (code/math/multilingual), and how many epochs you repeat scarce data (data-constrained scaling) change
  results as much as raw token count. Synthetic data helps but can collapse diversity — verify.
- **Tokenizer choice is frozen at pretraining.** You cannot change vocabulary after the run without
  retraining embeddings. Pick algorithm (almost always **byte-level BPE** for general LLMs), vocab size,
  and normalization deliberately. Always include a byte fallback so nothing is unrepresentable.
- **Fertility = tokens-per-word; it is throughput and cost.** A tokenizer with high fertility on your
  target languages/code wastes context window and inflates train *and* inference cost linearly. Measure
  fertility per language/domain on held-out text before freezing; multilingual targets often need a
  custom tokenizer to avoid 2–4x fertility blowup vs. an English-centric one.
- **Handle digits and whitespace deliberately.** Digit splitting (single digits or 3-digit groups) helps
  arithmetic; explicit whitespace handling (SentencePiece `▁`, byte-level space) keeps detokenization
  lossless. Reserve special/control tokens up front — adding them later is painful.
- **Track provenance and licensing per document.** Source URL, snapshot, filter decisions, and license
  signal must survive the whole pipeline. Required for audits, takedowns, and reproducibility — see
  `[[responsible-ai-governance]]`.
- **Scale is a distributed-systems problem.** Millions of WARC files / tens of TB demand
  Spark/Ray/datatrove, sharded columnar storage, streaming, and checkpointed stages. Getting a stage
  wrong is measured in thousands of GPU/TPU-hours, not minutes.

## Related skills

- `[[training-frameworks]]` — once data is tokenized and packed, this is the training loop (FSDP,
  Megatron, MaxText) that consumes it; data loaders, sharding, and sequence packing live there.
- `[[data-engineering-feature-stores]]` — classical/structured/feature data pipelines; orthogonal to
  unstructured web-text curation here.
- `[[ml-frameworks]]` — PyTorch/JAX/XLA primitives underneath tokenizer and dataloader code.
- `[[maxtext-jax-llm]]` — JAX/TPU LLM training that ingests these tokenized, packed datasets.
- `[[responsible-ai-governance]]` — provenance, licensing, consent, PII/toxicity policy, and audit
  obligations that this pipeline must satisfy.
- `[[fine-tuning-peft]]` — post-pretraining adaptation; reuses the frozen tokenizer and very different
  (small, curated/instruction) data.
- `[[aiml-on-kubernetes]]` — running these distributed curation and tokenizer-training jobs on clusters.
