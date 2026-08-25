# AGENTS.md — Multimodal ML

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`multimodal-ml-guide.md`** next to this file — read it
> before building or reviewing multimodal work, and apply it. Concrete walk-throughs (VLM
> encoder→projector→LLM trace, multimodal-RAG pipeline, serving token budget) are in **`examples.md`**.
> This file is the always-on summary.
>
> Scope: systems that span vision, language, audio/speech, and video — training and serving. The
> underlying LLM/training/serving machinery belongs to the sibling skills (`[[ml-frameworks]]`,
> `[[training-frameworks]]`, `[[serving-frameworks]]`, `[[inference-optimization]]`,
> `[[fine-tuning-peft]]`, `[[rag-vector-databases]]`, `[[ml-evaluation-evals]]`) — reference, don't
> duplicate. The field moves fast (it is 2026): **verify model names, benchmark numbers, and engine
> feature support against current docs.** Never quote a score or capability you have not re-verified.

## When a model takes/produces images, audio, or video, apply these by default:

- **The core idea is one shared representation space.** Contrastive training (CLIP, arXiv:2103.00020;
  SigLIP sigmoid loss, arXiv:2303.15343) aligns image-text pairs → zero-shot classification and
  cross-modal retrieval. CLIP = softmax/InfoNCE over the batch; SigLIP = pairwise sigmoid, scales to
  large batches.
- **The VLM pattern is vision encoder (ViT) → projector/connector → LLM.** Know the two lineages:
  projection/prefix MLP (LLaVA, arXiv:2304.08485) vs. gated cross-attention resampler (Flamingo,
  arXiv:2204.14198). Distinguish early vs. late fusion and native-multimodal vs. bolt-on; any-to-any
  models output multiple modalities.
- **Resolution & tiling decide quality and token count.** High-res/tiling for documents/OCR; cap tiles to
  bound tokens. Downscaling away document text is a classic bug.
- **Per-modality reality:** vision = patchify + tiling; speech = log-mel + Whisper-style ASR
  (arXiv:2212.04356), streaming for latency; video = frame sampling is everything (sample keyframes, pool
  temporally, never dense-sample); generation = latent diffusion (arXiv:2112.10752), DiT, flow matching —
  cost ∝ denoising steps × resolution.
- **Train in stages, and data dominates.** Encoder → align projector (freeze encoder+LLM) →
  visual-instruction tune (LoRA or full). Re-caption, filter, dedup, **decontaminate** the corpus — data
  quality beats architecture. Use `[[fine-tuning-peft]]` (LoRA/QLoRA) and `[[data-engineering-feature-stores]]`.
- **Serving multimodal ≠ serving an LLM.** One image = hundreds–thousands of visual tokens hitting prefill
  and KV cache — size for the visual worst case and budget tokens. Preprocessing (decode/resize/tile, mel,
  frame extraction) is a real, often-bottleneck pipeline — isolate it, run it async, cache encoder
  outputs. Verify engine multimodal support (vLLM et al.). See `[[serving-frameworks]]`,
  `[[inference-optimization]]`.
- **Evaluate the multimodal failure modes.** Measure object **hallucination** and **grounding**, not just
  text fluency. Use multimodal benchmarks but assume contamination and gameability; add task-specific
  and hallucination/grounding evals on your own data. Per-modality: WER/RTF (ASR), MOS/latency (TTS),
  FID/CLIP-score + human pref (generation). See `[[ml-evaluation-evals]]`.

## Anti-patterns to flag in review
- Costing/sizing images as "just more tokens" (they're thousands).
- Ignoring the preprocessing bottleneck; inline preprocessing serialized against generation.
- Low-quality / duplicated / undecontaminated cross-modal data.
- Shipping with text-only evals (no hallucination/grounding check).
- Resolution/tiling mishandling; unbounded video frames.
- Reinventing the LLM/serving stack instead of deferring to the sibling skills.
- Careless full fine-tune of a VLM forgetting the base LLM's text ability.

## Definition of done for multimodal work
Visual token budget per request is explicit and bounded; preprocessing is profiled and isolated;
data is filtered/dedup'd/decontaminated; evals include hallucination + grounding (not just text);
engine multimodal feature support is verified against current docs; LoRA preferred over reckless full
fine-tunes. Report honestly anything unverified.
