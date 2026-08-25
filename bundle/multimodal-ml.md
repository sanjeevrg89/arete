---
name: multimodal-ml
description: World-class guidance for building production multimodal AI systems spanning vision, language,
  audio/speech, and video. Use when training or serving models that combine modalities — contrastive
  vision-language (CLIP/SigLIP), Vision-Language Models / multimodal LLMs (ViT encoder + projector/connector
  + LLM, à la LLaVA/Flamingo), ASR/TTS (Whisper), audio encoders, video temporal modeling/frame sampling,
  image/video generation (latent diffusion, DiT, flow matching), cross-modal embeddings and multimodal RAG.
  Covers the shared-representation mental model, early/late fusion, native-multimodal vs bolt-on, staged
  training (pretrain → align → instruction-tune), the serving cost of variable-length visual tokens on
  context/KV-cache, multimodal evaluation/hallucination/grounding, and the multimodal anti-patterns.
  Reach for it whenever a model takes images, audio, or video as input or output — not pure-text LLMs.
---

# Multimodal ML

Apply the judgment of an engineer who has shipped production multimodal systems — vision-language
retrieval, VLM assistants, ASR/TTS pipelines, and diffusion image generation — at scale for years.
The whole field rests on one idea: **align different modalities into a shared representation space** so
a model can reason across them. Everything below is downstream of getting that alignment, the data, and
the token/preprocessing economics right.

## How to use this skill

1. **Read `multimodal-ml-guide.md`** in this directory — the full reference (mental model, the VLM
   architecture pattern, per-modality detail, training stages, serving, evaluation, anti-patterns).
   Apply it to the task. For concrete walk-throughs (a VLM encoder→projector→LLM trace, a multimodal-RAG
   pipeline, and a serving token-budget note) read **`examples.md`**.
2. This skill is multimodal-specific. For the underlying LLM/training/serving machinery it builds on,
   defer to the sibling skills rather than duplicating them: `[[ml-frameworks]]`, `[[training-frameworks]]`,
   `[[serving-frameworks]]`, `[[inference-optimization]]`, `[[fine-tuning-peft]]`, `[[rag-vector-databases]]`,
   `[[ml-evaluation-evals]]`, `[[aiml-on-kubernetes]]`.
3. Match the surrounding codebase/stack conventions; apply the correctness and data-quality rules
   regardless. The ecosystem moves fast (it is 2026) — **verify model names, benchmark numbers, and
   engine feature support against current docs** before committing to them.

## The essentials (full detail in `multimodal-ml-guide.md`)

- **Shared embedding space is the core idea.** Contrastive training (CLIP/SigLIP) pulls matched
  image-text pairs together and pushes mismatches apart, giving you zero-shot classification and
  cross-modal retrieval for free. CLIP uses a softmax/InfoNCE loss over a batch; SigLIP swaps it for a
  pairwise sigmoid loss that scales to large batches without a global all-gather.
- **The dominant VLM pattern is encoder + projector + LLM.** A frozen-or-tuned **vision encoder (ViT)**
  produces patch embeddings; a **projector/connector** (MLP as in LLaVA, or cross-attention resampler
  as in Flamingo) maps them into the LLM's token space; the **LLM** consumes them as soft tokens
  alongside text. Know the two lineages: projection/prefix (LLaVA) vs gated cross-attention (Flamingo).
- **Early vs late fusion, native vs bolt-on.** Late/bolt-on (stitch a pretrained encoder to a pretrained
  LLM, align with an adapter) is cheap and dominant; native/early-fusion (trained multimodal from
  scratch, often with a shared tokenizer/transformer) generalizes better but costs far more. Any-to-any
  models output multiple modalities (text + image + audio), not just text.
- **Each modality has its own preprocessing reality.** Vision: patchify, **resolution and tiling** decide
  token count and OCR/document quality. Speech: log-mel features, Whisper-style encoder-decoder ASR,
  streaming/chunking for latency. Video: **frame sampling** is the whole game — naive dense sampling
  explodes the token budget; use temporal pooling/keyframes and long-context tricks.
- **Generation is diffusion, increasingly DiT + flow matching.** Latent diffusion denoises in a VAE
  latent space (cheap); DiT replaces the U-Net with a transformer; flow matching is a faster/cleaner
  training objective for the same goal. Treat as a systems problem: many denoising steps, scheduler
  choice, classifier-free guidance, distillation for few-step sampling.
- **Multimodal embeddings power retrieval and RAG.** Embed images, text (and audio/video) into one
  space; do cross-modal ANN search; ground a VLM on retrieved images/pages. See `[[rag-vector-databases]]`.
- **Training is staged and data-bound.** Typical recipe: pretrain/obtain encoder → **align** projector on
  image-text pairs (often freezing encoder+LLM) → **instruction-tune** on multimodal conversations.
  Data quality (caption quality, dedup, dedupe-near-duplicates, decontamination) dominates outcomes more
  than architecture. See `[[data-engineering-feature-stores]]`, `[[fine-tuning-peft]]`.
- **Serving multimodal ≠ serving an LLM.** A single image becomes hundreds-to-thousands of visual tokens,
  blowing up prefill, context, and KV cache; preprocessing (decode/resize/tile, mel-spectrogram, frame
  extraction) is a real CPU/GPU pipeline that bottlenecks throughput. Use multimodal-aware engines
  (vLLM and others — **verify current support**) and budget tokens explicitly. See `[[serving-frameworks]]`,
  `[[inference-optimization]]`.
- **Evaluate for the multimodal failure modes.** Text-only metrics miss hallucination (describing objects
  that aren't there), weak grounding, and OCR/spatial errors. Use multimodal benchmarks but know their
  ceilings and contamination; always include task-specific and hallucination/grounding evals.
  See `[[ml-evaluation-evals]]`.
- **Top anti-patterns:** treating images as "just more tokens" when costing/sizing; ignoring the
  preprocessing bottleneck; feeding low-quality/duplicated cross-modal pairs; shipping with text-only
  evals; mishandling resolution/tiling (downscaling away the text in a document).

## Related skills
- `[[ml-frameworks]]` — PyTorch/JAX/XLA, GPU/TPU; the substrate for the encoders, LLMs, and diffusion nets here.
- `[[training-frameworks]]` — FSDP/DeepSpeed/Megatron/NeMo for actually distributing multimodal pretraining.
- `[[serving-frameworks]]` — vLLM/SGLang/Triton/TensorRT-LLM/KServe; how multimodal inference is deployed.
- `[[inference-optimization]]` — KV cache, quantization, batching; doubly important once visual tokens balloon.
- `[[fine-tuning-peft]]` — LoRA/QLoRA/adapters for tuning VLMs without full fine-tunes.
- `[[rag-vector-databases]]` — vector stores and ANN for cross-modal embeddings and multimodal RAG.
- `[[ml-evaluation-evals]]` — eval harnesses, judging, contamination; the home of rigorous evaluation.
- `[[data-engineering-feature-stores]]` — building and cleaning the image/video-text data at scale.
- `[[aiml-on-kubernetes]]` — running multimodal training/serving on K8s/GKE with GPUs/TPUs.

---

# Reference — multimodal-ml

# Multimodal ML — The Reference

Production guidance for systems that take or produce more than text: vision, language, audio/speech,
video, and image/video generation. This is the level of an engineer who builds and operates these
systems, not a survey. It builds on the LLM/training/serving machinery in the sibling skills
(`[[ml-frameworks]]`, `[[training-frameworks]]`, `[[serving-frameworks]]`, `[[inference-optimization]]`)
— it does not re-explain transformers, FSDP, or KV cache from scratch; it explains what changes when a
modality other than text enters the picture.

> Fast-moving field (it is 2026). Model names, parameter counts, context lengths, benchmark scores, and
> engine feature matrices change monthly. Treat every such specific in this guide as "verify against
> current docs." The *concepts and architecture patterns* are stable; the *numbers and product names*
> are not.

## 1. Mental model: one shared representation space

The unifying idea across all of multimodal ML is **alignment**: project heterogeneous inputs (pixels,
waveforms, text tokens, video frames) into a common vector space where semantically related things are
near each other, and let a model reason over that space. Two broad ways to achieve and exploit alignment:

- **Contrastive alignment (no generation).** Train two encoders so that matched pairs (an image and its
  caption) have high similarity and mismatched pairs have low similarity. The output is a *metric space*:
  you get zero-shot classification, retrieval, and a clean embedding for downstream models. CLIP and
  SigLIP live here.
- **Generative fusion (a single model that reads/writes across modalities).** Convert non-text inputs
  into a sequence the transformer can attend over (soft "tokens"), interleave with text, and let a
  language model generate. VLMs / multimodal LLMs live here. Generation of images/video/audio is the
  inverse: condition a generator (usually diffusion) on text or other modalities.

Everything else — architecture choices, training stages, serving costs — is downstream of how well you
align modalities and how many tokens that alignment costs you at inference.

## 2. Contrastive vision-language (CLIP / SigLIP)

**CLIP** (Radford et al., 2021, arXiv:2103.00020): a vision encoder (ViT or ResNet) and a text encoder
(transformer) trained on hundreds of millions of web image–text pairs. For a batch of N pairs it computes
an N×N similarity matrix and applies a symmetric **InfoNCE / softmax contrastive loss**: each image's
correct caption is the positive against the other N−1 captions as negatives, and vice versa. A learned
temperature scales the logits.

What you get for free:
- **Zero-shot classification:** embed the image; embed candidate label strings as
  `"a photo of a {label}"`; pick the nearest. No task-specific training.
- **Cross-modal retrieval:** embed a text query, ANN-search an image index (or vice versa).
- **A frozen vision tower** that downstream VLMs reuse as their encoder (CLIP ViT and successors are the
  default LLaVA-style encoder).

**SigLIP** (Zhai et al., 2023, arXiv:2303.15343) replaces the softmax/InfoNCE loss with a **pairwise
sigmoid** loss: every image–text pair is an independent binary "match/no-match" logistic problem. This
removes the global all-gather of similarities that softmax needs across the batch, so it scales to very
large batches and shards cleanly. In practice SigLIP-family encoders are a strong, common choice for VLM
vision towers — **verify the current best encoder against up-to-date leaderboards**, this moves.

Practical notes:
- **Batch size is a hyperparameter that matters** — contrastive learning lives on negatives. Small
  batches starve the loss of hard negatives.
- The text encoder is usually short-context (a caption, not a document). Don't expect CLIP text towers to
  do document understanding.
- Embeddings are typically L2-normalized; similarity is cosine / dot product.

## 3. Vision-Language Models (VLMs / multimodal LLMs)

The dominant production pattern for "a chatbot that can see":

```
image ─▶ vision encoder (ViT) ─▶ patch embeddings ─▶ projector/connector ─▶ soft visual tokens
                                                                                   │
text  ─────────────────────────────── tokenizer ─▶ text tokens ──────────────────┤
                                                                                   ▼
                                                                       LLM (decoder) ─▶ text out
```

### The three components

1. **Vision encoder.** A ViT (CLIP/SigLIP-style) splits the image into patches and produces one embedding
   per patch (e.g., a 336px image at 14px patches → 24×24 = 576 patch tokens). Often frozen, especially
   in early training stages. The encoder's pretraining (contrastive vs. supervised) and resolution
   strongly shape downstream OCR/grounding quality.
2. **Projector / connector.** Maps patch embeddings into the LLM's embedding dimension and "token" space.
   Two lineages:
   - **Projection / prefix (LLaVA lineage; Liu et al., 2023, arXiv:2304.08485).** A simple linear or
     **MLP** projector turns each patch embedding into a soft token; these are prepended/interleaved with
     the text tokens and the LLM attends over them like any other token. Simple, cheap, and the default.
   - **Cross-attention resampler (Flamingo lineage; Alayrac et al., 2022, arXiv:2204.14198).** A
     Perceiver-style resampler compresses many patch tokens into a small fixed set of latents, and
     **gated cross-attention** layers inserted into the (frozen) LLM let text tokens attend to visual
     features. Keeps the visual token count bounded and the LLM mostly frozen; more parameters/complexity.
3. **LLM.** A standard decoder-only language model (`[[ml-frameworks]]`). It generates text conditioned
   on the interleaved visual + text tokens. May be frozen, partially tuned (LoRA), or fully tuned
   depending on stage and budget (`[[fine-tuning-peft]]`).

### Fusion taxonomy

- **Late fusion / bolt-on:** take a pretrained vision encoder and a pretrained LLM and connect them with
  an adapter, tuning mostly the connector. Cheap, fast, dominant. Most open VLMs are this.
- **Early fusion / native multimodal:** train a single model on interleaved modalities from (or near)
  scratch, frequently with a unified tokenizer that turns images into discrete tokens, or a single
  transformer over mixed embeddings. Better cross-modal generalization, far higher compute cost, harder
  data engineering. The trend in frontier models is toward more native-multimodal training — **verify
  which current models are natively multimodal vs. bolt-on; vendors describe this inconsistently.**
- **Any-to-any models:** output multiple modalities, not just text — e.g., text + images + audio from one
  model. Requires generation heads/decoders per output modality (a diffusion decoder for images, an audio
  codec decoder for speech) or discrete multimodal tokens decoded back to pixels/waveform.

### Resolution and tiling (the detail people get wrong)

A fixed low encoder resolution destroys fine detail — small text in a document, distant objects. The
standard fixes:
- **Tiling / "any-resolution":** slice a high-res image into a grid of tiles, encode each at the encoder's
  native resolution, plus a downsized global thumbnail, and concatenate the resulting tokens. This is how
  VLMs read documents and dense scenes — but **token count scales with the number of tiles**, so a
  full-page scan can cost thousands of visual tokens.
- **Native dynamic resolution:** newer encoders accept variable resolution/aspect ratio and emit a
  variable number of tokens. Cleaner, but the downstream token budget is now input-dependent.
- **Token compression / pooling:** pixel-shuffle, perceiver resamplers, or learned pooling to cut visual
  tokens before the LLM. A direct quality↔cost lever.

## 4. Modalities in depth

### 4.1 Vision

- **Image tokenization:** patchify (ViT) for understanding; or VQ/quantize to discrete codes (VQGAN-style)
  when you want images *in the token vocabulary* for native-multimodal or generation.
- **OCR / Document AI:** resolution and tiling dominate. Documents need high effective resolution; many
  VLMs add explicit OCR-heavy pretraining data. Evaluate on document/OCR benchmarks, not just natural
  images, if that's your use case.
- **Grounding / detection:** some VLMs emit bounding boxes or point coordinates as text. Grounding quality
  is a distinct axis — a model can describe an image well yet localize poorly.

### 4.2 Speech & audio

- **ASR (speech→text):** **Whisper** (Radford et al., 2022, arXiv:2212.04356) is the canonical open
  encoder-decoder ASR: log-mel spectrogram → transformer encoder → transformer decoder that emits text
  (with timestamps, language ID, translation as special tokens). Trained on large weakly-supervised
  multilingual audio. Robust, but offline-oriented (30s windows) — true low-latency streaming needs
  chunked/streaming ASR architectures or careful windowing.
- **TTS (text→speech):** modern systems are typically neural codec + LM or diffusion based: text →
  acoustic/codec tokens → vocoder/codec decoder → waveform. Streaming TTS emits audio incrementally for
  low time-to-first-audio.
- **Audio encoders for understanding:** to give a VLM/LLM "ears," an audio encoder (often a speech or
  general-audio transformer) produces embeddings projected into the LLM token space — same encoder +
  projector + LLM pattern as vision.
- **Streaming latency** is the hard part of production audio: measure **time-to-first-token/audio** and
  real-time factor, not just throughput. Chunk size trades latency against accuracy.

### 4.3 Video

- **Frame sampling is the entire ballgame.** Video is enormous; you cannot feed every frame. Strategies:
  uniform sampling at K fps, keyframe/scene-change selection, or learned frame selection. Each sampled
  frame costs a full image's worth of visual tokens.
- **Temporal modeling:** add temporal position/encoding so the model knows frame order; some models use
  spatiotemporal attention or temporal pooling across frames to bound tokens.
- **Long video** is a long-context problem on top of a token-explosion problem: an hour of video at even
  modest fps is millions of visual tokens. Mitigations: aggressive temporal pooling, memory/retrieval over
  frames (multimodal RAG over the video), hierarchical summarization. Treat long-video context jointly
  with `[[inference-optimization]]`.

### 4.4 Image / video generation (systems view)

- **Latent diffusion (Rombach et al., 2022, arXiv:2112.10752):** encode the image into a compressed VAE
  latent, run the diffusion (denoising) process in latent space (cheap), decode back to pixels. This is
  why high-res generation is tractable. The denoiser was originally a **U-Net**.
- **DiT (Diffusion Transformer; Peebles & Xie, 2022, arXiv:2212.09748):** replace the U-Net denoiser with
  a transformer over latent patches. Scales like transformers do and underpins many current
  image/video generators — **verify which products use DiT-style backbones; this is the prevailing trend
  but specifics change.**
- **Flow matching / rectified flow:** an alternative training objective that learns a (near-)straight
  probability-flow ODE from noise to data, often giving faster, more stable training and **few-step
  sampling**. Widely adopted in recent generators — verify current usage.
- **Systems levers that matter:** number of denoising steps (latency ∝ steps), scheduler/sampler choice,
  **classifier-free guidance** (a second forward pass per step → ~2× cost), and **distillation** (turning
  a 25–50 step model into a 1–4 step model) for real-time generation. Video adds temporal layers and a
  much larger latent — generation cost and memory dwarf images.

## 5. Multimodal embeddings & retrieval

- **Cross-modal embeddings:** a shared space (CLIP/SigLIP-style, or a dedicated multimodal embedding
  model) lets you search images with text, text with images, and increasingly audio/video too. L2-normalize;
  use cosine/dot ANN.
- **Multimodal RAG:** retrieve relevant images/pages/clips for a query and feed them (as pixels or as
  pre-extracted captions/OCR) to a VLM to ground its answer. Two common designs:
  - **Embed-the-media:** index image/page embeddings; retrieve top-k; pass the actual images to the VLM.
    Strong for visual/document QA (e.g., screenshot/page retrieval).
  - **Embed-the-text-proxy:** caption/OCR each item, index the text, retrieve, optionally re-fetch the
    image. Cheaper, but loses information the captioner didn't capture.
  Use a real vector database and ANN index — see `[[rag-vector-databases]]` — and watch the token budget:
  retrieved images are expensive context (Section 7).

## 6. Training multimodal models

### 6.1 Data — the dominant factor

Architecture matters less than data. The wins and losses are in the corpus:
- **Pairs at scale:** image–text and video–text pairs from the web are noisy. Filtering (e.g.,
  CLIP-score filtering), dedup, and **near-duplicate removal** materially change results.
- **Caption quality:** web alt-text is short and noisy; **re-captioning** with a strong model (synthetic
  high-quality captions) is now standard and a big lever. Mix synthetic and human captions deliberately.
- **Decontamination:** scrub eval/benchmark images and their near-duplicates from training data, or your
  evals are meaningless (Section 8).
- **Interleaved data:** for VLMs that handle interleaved image-text, you need interleaved documents (image
  and text in natural sequence), not just isolated pairs.
- Build and clean this with real data infra — `[[data-engineering-feature-stores]]`.

### 6.2 Staged training (canonical VLM recipe)

1. **Encoder pretraining** — usually reuse an existing contrastive/supervised vision encoder rather than
   train one (CLIP/SigLIP-family). Train your own only with strong reason.
2. **Alignment / connector pretraining** — freeze encoder and LLM, train the projector on large
   image–text (caption) data so visual features land in the LLM's space. Cheap, fast.
3. **Instruction / visual-instruction tuning** — train on multimodal conversations/instructions
   (often LLM-generated), unfreezing the LLM (full or LoRA) and possibly the encoder. This is where
   chat ability and task following come from. LLaVA established this two-stage align→instruct recipe.

For native/early-fusion models, replace stages 1–3 with large-scale interleaved multimodal pretraining,
which is far more compute-intensive and data-engineering-heavy.

### 6.3 Compute & memory profile vs. text-only

- **Sequences are longer and lumpier.** Visual tokens add hundreds–thousands of tokens per image; tiling
  and video make sequence length highly variable, which hurts batching/padding efficiency. Bucket by
  length or use packing.
- **The encoder is extra forward/backward compute** (and activations) on top of the LLM. Freezing it
  saves memory; precompute and cache encoder outputs for fixed-image datasets.
- **Distribution:** the LLM still wants FSDP/tensor/pipeline parallelism (`[[training-frameworks]]`); the
  encoder is usually small enough to replicate. Mind the heterogeneous component sizes.
- **Fine-tuning:** LoRA/QLoRA on the LLM (and optionally the projector) is the cost-effective default for
  adapting an open VLM — see `[[fine-tuning-peft]]`. Full fine-tunes risk catastrophic forgetting of the
  base LLM's text ability; keep text-only data in the mix.

## 7. Serving multimodal — what actually changes

This is where multimodal systems surprise teams who think of it as "an LLM that also takes images."

- **One image = many tokens.** A tiled high-res image or a sampled video can be **thousands of visual
  tokens**, all of which hit prefill and occupy the **KV cache**. A "small" multimodal request can be
  larger than a long text prompt. Size GPU memory and max-context for the *visual* worst case, and
  **budget visual tokens explicitly** per request (cap tiles/frames/resolution).
- **Preprocessing is a real pipeline, often the bottleneck.** Image decode + resize + tile, mel-spectrogram
  extraction, and video frame extraction/decoding are CPU/GPU-heavy and run *before* the model. If you do
  them inline on the inference host you serialize them against generation. Move them to a dedicated
  preprocessing stage/pool, do them async, and cache encoder outputs for repeated media.
- **Variable visual token counts wreck naive batching.** Input-dependent token counts (any-resolution,
  variable frames) make static batching and KV-cache planning hard. Continuous batching helps; still plan
  for the visual worst case.
- **Engine support is uneven and moving.** vLLM and other engines support multimodal inputs for many VLMs,
  but coverage varies by model and version — **verify which models, which connectors, and which features
  (e.g., prefix caching of image tokens, video) your engine supports before committing.** See
  `[[serving-frameworks]]` and apply `[[inference-optimization]]` (quantization, KV-cache management,
  chunked prefill) — doubly important once visual tokens balloon the prefill.
- **Streaming audio latency** is its own SLO. Optimize time-to-first-audio and real-time factor; chunk
  input; pipeline ASR→LLM→TTS so you don't wait for full utterances.
- **Generation serving (diffusion)** is throughput-bound on denoising steps × resolution; batch requests,
  use few-step/distilled models for interactive latency, and cache/reuse where possible. Very different
  profile from autoregressive decoding.
- Run all of this on GPUs/TPUs via `[[aiml-on-kubernetes]]` when on K8s/GKE.

## 8. Evaluation

- **Text-only metrics miss multimodal failure modes.** A VLM can produce fluent, well-formed answers that
  are visually wrong. You must evaluate the *grounding*, not just the language.
- **Hallucination in VLMs** is the signature failure: describing objects, attributes, counts, or text that
  aren't in the image (object hallucination). Measure it explicitly; it worsens with longer generations
  and weaker visual grounding.
- **Grounding / localization** is a separate axis from description quality — test spatial reasoning,
  counting, OCR, and bounding-box accuracy if they matter to you.
- **Benchmarks and their limits:** there are many multimodal QA/reasoning/OCR/document/video benchmarks.
  Use them, but know their ceilings — **contamination** (benchmark images leaking into training),
  multiple-choice gameability, and narrow coverage. **Always add task-specific and hallucination/grounding
  evals on your own data**, and prefer judged/open-ended eval where multiple-choice hides failures.
  Verify which benchmarks are current and uncontaminated; rankings churn. See `[[ml-evaluation-evals]]`.
- **Per-modality evals:** ASR → WER/CER and latency/RTF; TTS → MOS/intelligibility + latency; generation →
  FID/CLIP-score and human preference (know FID's limitations). Don't reuse one modality's metric for
  another.

## 9. Anti-patterns (the traps that bite in production)

- **Treating images as "just more tokens" for cost/sizing.** One image can be thousands of tokens; your
  context, KV cache, and bill are set by the *visual* worst case, not the text. Budget tiles/frames/res.
- **Ignoring the preprocessing bottleneck.** Decode/resize/tile, mel extraction, and frame sampling run
  before the model and routinely dominate latency/throughput. Profile and isolate them; cache encoder
  outputs.
- **Poor cross-modal data quality.** Noisy alt-text, near-duplicates, and unfiltered pairs cap your
  ceiling. Re-caption, filter, dedup, decontaminate — this beats architecture tweaks.
- **No multimodal-specific eval.** Shipping with text-only metrics hides hallucination and grounding
  failures. If you don't measure object hallucination and grounding, you don't know your model works.
- **Resolution / tiling mishandling.** Downscaling away the text in a document, or over-tiling and
  blowing the token budget. Match resolution/tiling to the task (high-res for documents, capped for
  natural images/video).
- **Unbounded video frames.** Dense frame sampling explodes tokens and cost for little quality gain.
  Sample smartly (keyframes/scene changes) and pool temporally.
- **Reinventing the LLM/serving stack.** The LLM, training distribution, and serving engine are solved by
  the sibling skills — defer to them; only the *multimodal-specific* deltas belong here.
- **Full fine-tuning a VLM carelessly.** Unfreezing everything on a small dataset forgets the base LLM's
  text skills. Prefer LoRA and keep text data in the mix.

## 10. Version awareness

The architecture *patterns* (contrastive alignment; encoder→projector→LLM; latent diffusion / DiT / flow
matching; staged align→instruct training; the serving token-budget reality) are durable. The *specifics*
are not. Before you rely on any of these, **verify against current docs/leaderboards**: which vision
encoder is best; which models are native-multimodal vs. bolt-on; which engines support which VLMs and
features (image prefix caching, video, audio); current SOTA benchmark numbers and which benchmarks are
uncontaminated; whether DiT/flow-matching/few-step distillation is the prevailing choice for a given
generator. Do not quote a benchmark score or product capability you have not re-verified.

## 11. Canonical references (verify currency)

- CLIP — Radford et al., "Learning Transferable Visual Models From Natural Language Supervision,"
  arXiv:2103.00020.
- SigLIP — Zhai et al., "Sigmoid Loss for Language Image Pre-Training," arXiv:2303.15343.
- LLaVA — Liu et al., "Visual Instruction Tuning," arXiv:2304.08485.
- Flamingo — Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning," arXiv:2204.14198.
- ViT — Dosovitskiy et al., "An Image is Worth 16x16 Words," arXiv:2010.11929.
- Whisper — Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," arXiv:2212.04356.
- Latent Diffusion — Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models,"
  arXiv:2112.10752.
- DiT — Peebles & Xie, "Scalable Diffusion Models with Transformers," arXiv:2212.09748.
- Flow Matching — Lipman et al., "Flow Matching for Generative Modeling," arXiv:2210.02747.
  (Rectified flow: Liu et al., arXiv:2209.03003.)

> These arXiv IDs are the originating papers and are stable; the *state of the art they describe is not*.
> For current models, encoders, engines, and benchmarks, consult up-to-date project docs and leaderboards.

---

# Multimodal ML — Worked Examples

Canonical, imitable walk-throughs. Code is illustrative-but-correct in spirit (PyTorch-flavored
pseudocode where a real API would pin you to a fast-moving library version). The *shapes, token math, and
pipeline structure* are the point. Verify library/model specifics against current docs.

---

## 1. VLM architecture walk-through (encoder → projector → LLM)

The LLaVA-style projection lineage, traced with explicit tensor shapes so the token math is concrete.

```python
# Components (all reused from sibling stacks; see [[ml-frameworks]], [[fine-tuning-peft]]).
vision_encoder   # e.g. a SigLIP/CLIP ViT, often FROZEN. patch=14, img=384 -> grid 27x27.
projector        # MLP: Linear(d_vis, d_llm) -> GELU -> Linear(d_llm, d_llm)
llm              # decoder-only LLM with embed_tokens(); may be LoRA-tuned or frozen-then-tuned.

def forward(image, prompt_text):
    # 1) Encode image -> one embedding per patch.
    #    image: [B, 3, 384, 384]
    patch_embeds = vision_encoder(image)        # [B, 729, d_vis]   (27*27 = 729 patches)
    #    729 visual tokens PER IMAGE before any text. This is the cost center.

    # 2) Project patch embeddings into the LLM's token space (soft tokens).
    visual_tokens = projector(patch_embeds)     # [B, 729, d_llm]

    # 3) Embed the text and splice visual tokens in where the <image> placeholder sits.
    #    Prompt: "<image>\nWhat is written on the sign?"
    text_ids   = tokenizer(prompt_text)         # includes a reserved <image> position
    text_embeds = llm.embed_tokens(text_ids)    # [B, T_text, d_llm]
    inputs_embeds = splice(text_embeds, visual_tokens, at="<image>")
    #    final length = T_text - 1 + 729  -> visual tokens dominate the sequence.

    # 4) The LLM generates conditioned on the interleaved sequence. Visual tokens are
    #    just more positions in the KV cache from here on.
    return llm.generate(inputs_embeds=inputs_embeds)
```

Key takeaways to imitate:
- **Token math is explicit.** 729 visual tokens for a single 384px image; **tiling a high-res document
  into 9 tiles + a thumbnail ≈ 7,000+ visual tokens** before the user's question. This sets your context
  length, KV-cache size, and prefill cost.
- **The projector is the cheap, high-leverage part.** In alignment-stage training you freeze the encoder
  and LLM and train only this MLP on caption data — fast and cheap (guide §6.2).
- **Cross-attention variant (Flamingo lineage):** instead of splicing 729 tokens into the sequence, a
  Perceiver resampler compresses them to a small fixed set (e.g., 64 latents) and **gated cross-attention**
  layers in the LLM attend to them — bounds the visual token count at the cost of more parameters and
  complexity. Pick projection for simplicity, cross-attention to cap tokens. (See guide §3.)
- **To add audio**, swap an audio encoder + its own projector in for the vision path — identical pattern.

---

## 2. Multimodal RAG pipeline sketch

Grounding a VLM on retrieved images/pages. Two-stage: build a cross-modal index, then retrieve + answer.
Uses a real vector DB and ANN index — see `[[rag-vector-databases]]`.

```python
# ---- Indexing (offline) ----
# Embed every page/image into a SHARED space with a multimodal embedding model.
for page in corpus:                       # e.g. scanned document pages, product images
    emb = mm_embedder.embed_image(page.image)   # L2-normalized vector, shared with text
    index.upsert(id=page.id, vector=emb, meta={"doc": page.doc, "page": page.n})
# Optionally also store OCR/caption text per page for hybrid retrieval + citation.

# ---- Query time (online) ----
def answer(question: str, k=4):
    q = mm_embedder.embed_text(question)        # SAME space as the image embeddings
    hits = index.search(q, top_k=k)             # cross-modal ANN: text query -> images

    # Design choice A (embed-the-media): pass the actual page images to the VLM.
    images = [load(h.id) for h in hits]
    return vlm.generate(
        prompt=f"Using only these pages, answer: {question}",
        images=images,                          # WARNING: each image = hundreds–thousands of tokens
    )
    # Design choice B (text-proxy): retrieve OCR/captions, answer text-only or re-fetch top-1 image.
    # Cheaper, but loses anything the captioner/OCR missed (layout, figures). See guide §5.
```

Decisions to imitate:
- **Embed media vs. text-proxy.** Embed-the-media (pass real pages) wins on visual/document QA — the VLM
  sees layout, figures, and handwriting OCR drops. Text-proxy is cheaper but lossy. Choose per use case.
- **Token budget is the constraint, not just recall.** `k=4` retrieved high-res pages can be **tens of
  thousands of visual tokens**. Lower k, downscale non-critical pages, or retrieve a region/crop instead
  of the full page. This is a retrieval *and* a serving decision (guide §7).
- **Keep citations.** Store doc/page metadata so the answer can point back to the source page.
- **Decontaminate / dedup the index** the same way you would training data (guide §6.1).

---

## 3. Multimodal serving: preprocessing & token-budget note

The two things that surprise teams treating a VLM like "an LLM that also takes images." Structure the
serving path around them. Apply `[[serving-frameworks]]` and `[[inference-optimization]]`; on K8s/GKE see
`[[aiml-on-kubernetes]]`.

**Preprocessing is a real pipeline and usually the bottleneck.** Don't run it inline on the GPU host where
it serializes against generation.

```
request (raw bytes)
   │
   ├─ PREPROCESS POOL (CPU/GPU, separate from inference, async, cached) ───────────────┐
   │     image  -> decode -> resize -> tile (any-res grid + thumbnail)                  │
   │     audio  -> resample -> log-mel spectrogram                                      │
   │     video  -> demux -> SAMPLE FRAMES (keyframes/uniform, NOT every frame) -> decode│
   │     (cache encoder outputs keyed by media hash to skip repeat work)                │
   │                                                                                    ▼
   └─ INFERENCE ENGINE (vLLM/etc.): vision/audio encoder -> projector -> LLM, continuous batching
```

**Budget the visual token count explicitly — size for the visual worst case.** Rough planning numbers
(verify against your encoder/config; these are illustrative):

| Input                          | Approx. visual tokens | Note                                            |
|--------------------------------|-----------------------|-------------------------------------------------|
| One 384px image, no tiling     | ~700–800              | grid² patches; encoder-dependent                |
| One high-res page, 9 tiles + thumbnail | ~7,000+        | tiling multiplies tokens linearly with tiles    |
| 1 min video @ 1 fps, 1 img/frame | ~40,000+            | frame count × per-frame tokens — explodes fast  |

Rules to imitate:
- **Cap tiles, frames, and resolution per request.** These are your direct cost/latency levers. A
  per-request visual-token ceiling prevents one document or video from OOMing the KV cache.
- **Move preprocessing off the inference host** (dedicated pool, async) and **cache encoder outputs** by
  media hash — repeated/popular media should never be re-encoded.
- **Plan KV cache and max-context for the visual worst case**, not the average text prompt. Variable visual
  token counts (any-resolution, variable frames) defeat static batching — rely on continuous batching and
  still leave headroom.
- **Verify engine support before committing:** which VLMs your engine (vLLM and others) handles, and which
  features (image prefix caching, video, audio, chunked prefill) it actually supports — this changes
  frequently. (Guide §7.)
- **Streaming audio is a separate SLO:** pipeline ASR → LLM → TTS, chunk inputs, and optimize
  time-to-first-audio / real-time factor rather than batch throughput.
