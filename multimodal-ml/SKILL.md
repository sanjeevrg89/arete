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
