# Inference Optimization — Worked Examples

Canonical, correct-in-shape flows to imitate. **APIs and flags in this area change quarterly (2026)** —
every command below is illustrative of the *workflow*; **verify exact argument names, dtypes, and
hardware support against current docs** (TensorRT Model Optimizer, TensorRT-LLM, your AWQ/GPTQ library,
your serving engine) before running. Do not treat any flag here as guaranteed-current.

---

## 1. INT4 / AWQ weight-only quantization flow

Goal: take an FP16 base model to **INT4 weight-only (W4A16)** for a bandwidth-bound decode win, with a
real eval gate. Shape of the flow with `autoawq` (one common open path; `llm-compressor` and TensorRT
Model Optimizer follow the same *shape*).

```python
# pip install autoawq  (verify package name/version + GPU support before running)
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "org/base-model-fp16"
quant_path = "org/base-model-awq-int4"

# Quant config: 4-bit, group size 128, GEMM kernel. zero_point/version names change — verify.
quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path)

# CALIBRATION DATA IS THE WHOLE GAME: draw ~128-512 samples from YOUR real distribution,
# formatted with the SAME chat template / languages / domains you serve. A generic web
# corpus here is the usual cause of a "quantization hurt quality" surprise.
calib = load_my_representative_samples()   # list[str], already chat-templated

model.quantize(tok, quant_config=quant_config, calib_data=calib)
model.save_quantized(quant_path)
tok.save_pretrained(quant_path)
```

**Mandatory eval gate** — perplexity is NOT sufficient; INT4 can leave perplexity flat while code/math/
long-context/multilingual quality drops:

```bash
# Run a task suite (lm-eval-harness-style) on BOTH fp16 and the int4 export, compare per task.
# Verify current harness name/flags.
lm_eval --model hf --model_args pretrained=org/base-model-fp16 \
        --tasks gsm8k,arc_challenge,<your-domain-task> --batch_size auto
lm_eval --model hf --model_args pretrained=org/base-model-awq-int4 \
        --tasks gsm8k,arc_challenge,<your-domain-task> --batch_size auto
```

Ship only if the quality delta on *your* tasks is within tolerance. If not: recovery ladder →
finer group size → keep sensitive layers (first/last, embeddings) higher precision → try GPTQ →
escalate to QAT. The AWQ checkpoint is then loadable by a serving engine that supports AWQ
(`[[serving-frameworks]]`); pass the engine's quantization flag (verify name, e.g. `quantization=awq`).

---

## 2. TensorRT-LLM: build-then-serve note

TensorRT-LLM is an **ahead-of-time engine build** — it fuses layers, applies precision, and emits a
**GPU-arch- and shape-specific** engine. The two-phase shape (quantize/convert → build engine → serve):

```bash
# Phase A — produce a quantized checkpoint (e.g. FP8 or INT4-AWQ) with TensorRT Model Optimizer
#   (github.com/NVIDIA/TensorRT-Model-Optimizer). The exact CLI/script names and the FP8/INT4
#   support matrix change across releases — VERIFY against the repo's current examples.
#   Conceptually: load base model -> run PTQ (FP8 or AWQ-INT4, with YOUR calibration set) ->
#   export a TensorRT-LLM checkpoint directory.

# Phase B — build the engine for the TARGET GPU and a chosen shape profile (max batch, max input/
#   output len). The engine is specific to this GPU generation + these shapes.
trtllm-build --checkpoint_dir ./ckpt_fp8 \
             --output_dir ./engine_fp8 \
             --gemm_plugin auto \
             --max_batch_size 64 \
             --max_input_len 8192 \
             --max_seq_len 9216
#   (flag names/values illustrative — verify with `trtllm-build --help` for your version.)

# Phase C — serve the engine (Triton with the TensorRT-LLM backend, or trtllm-serve, or via an
#   engine that consumes TRT-LLM engines). Routing/batching is the serving layer's job.
```

Key correctness points:

- **Rebuild per GPU generation and per shape profile.** An engine built for one arch (or a different
  `max_batch_size`/`max_seq_len`) will underperform or fail to load elsewhere. Do not reuse blindly.
- Choose precision at build time to match the bound: **FP8** (weights+activations) on FP8-capable GPUs
  for compute *and* bandwidth; **INT4-AWQ** weight-only when memory/bandwidth dominate.
- Eval the *built engine's* outputs, not just the pre-build checkpoint — the build (fusion, precision,
  plugins) is part of what can move quality.
- The engine pairs with the serving engine's paged-KV / in-flight batching — see `[[serving-frameworks]]`.

---

## 3. Speculative decoding config note

Speculative decoding is **exact** (rejection-sampling verification reproduces the target's
distribution) and trades **spare compute for latency**. It pays only when **acceptance is high** and the
**draft is cheap** relative to the target.

Illustrative serving config (shape only — the actual keys differ per engine and change; **verify**):

```yaml
# Conceptual — map to your engine's real speculative-decoding options.
target_model: org/base-model-fp8        # the big, accurate model
speculative:
  method: eagle                          # or: draft_model | medusa | ngram | lookahead
  # draft_model: org/base-model-draft    # for vanilla draft-model spec: MUST share the target's
  #                                       #   tokenizer; quantize/align it to the (quantized) target.
  num_speculative_tokens: 5              # k candidates proposed per target verification step
```

Operate it correctly:

- **Pick the variant for the workload.** `ngram`/prompt-lookup is near-free and excellent when output
  echoes input (summarization, code edit, RAG quotation). `eagle`/`medusa`/self-speculative need no
  separate aligned draft model and tend to give high acceptance. A separate `draft_model` must share
  the tokenizer and be aligned (ideally quantized to match the target).
- **Measure realized acceptance rate AND end-to-end tokens/s on real traffic**, not a demo prompt. A
  config that wins on one dataset can be net-negative on another.
- **It helps most with spare compute** (low/medium batch, latency-sensitive). At max-throughput batch
  sizes the GPU is already compute-saturated and spec-decode can *reduce* throughput — benchmark at
  your real load and turn it off where it loses.
- Tune `num_speculative_tokens` (`k`): too high wastes verification compute on rejected tails; sweep it.

---

## Stacking note

These compose. A typical high-performance stack: **FP8 (or INT4) weights + FP8/INT8 KV cache +
FlashAttention + engine build / `torch.compile` + CUDA graphs + speculative decoding**, served by a
paged/continuous-batching engine (`[[serving-frameworks]]`). **Add one lever at a time and re-run the
eval gate** — interactions (aggressive weight quant + KV quant + spec-decode) can compound quality loss
that no single lever caused.
