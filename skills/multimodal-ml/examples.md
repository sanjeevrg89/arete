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
