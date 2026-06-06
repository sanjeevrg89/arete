# Fine-Tuning & PEFT — Worked Examples

Canonical config sketches to imitate. API names/flags move between library versions (it is 2026) — **verify
against the current `peft`/`trl`/Axolotl docs** before relying on a specific keyword. Imitate the shape and
the choices (target modules, masking, memory knobs), not the version-specific spelling.

---

## 1. QLoRA SFT with HF PEFT + TRL (Python)

A 4-bit NF4 base + LoRA adapters, instruction tuning with completion-only loss. This is the reference idiom.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

BASE = "meta-llama/Llama-3.1-8B"  # a *base* model for instruction tuning; respect its license

# --- 4-bit NF4 quantization for the FROZEN base (the "Q" in QLoRA) ---
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                 # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,     # compute in bf16
    bnb_4bit_use_double_quant=True,            # quantize the quant constants too
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:                # avoid pad==eos surprises during masking
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto",
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# --- LoRA: cover attention AND MLP projections, not just q/v ---
peft_config = LoraConfig(
    r=16,                      # rank = capacity; start 8-16, raise only if it underfits
    lora_alpha=32,            # effective scale = alpha/r = 2.0 here
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[          # or simply target_modules="all-linear"
        "q_proj", "k_proj", "v_proj", "o_proj",      # attention
        "gate_proj", "up_proj", "down_proj",         # MLP / FFN
    ],
    # use_dora=True,          # enable DoRA if plain LoRA underfits at this rank
    # modules_to_save=["embed_tokens", "lm_head"],  # ONLY if you added/resized tokens
)

# --- Data: format with the model's OWN chat template; never hand-roll role markers ---
ds = load_dataset("your-org/your-sft-data", split="train")  # columns -> chat "messages"

def to_text(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(to_text, remove_columns=ds.column_names)

cfg = SFTConfig(
    output_dir="out/llama3.1-8b-qlora",
    num_train_epochs=2,                  # 1-3; small data overfits fast
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch = 2 * 8 * num_gpus
    learning_rate=2e-4,                  # LoRA LR (full FT would be ~1e-5-2e-5)
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=2048,
    packing=True,                        # pack short samples; keep per-sample masking intact
    # completion_only_loss=True,         # mask the prompt -> loss only on the answer.
    #   (exact knob varies by TRL version: DataCollatorForCompletionOnlyLM /
    #    completion_only_loss / assistant-only masking — VERIFY in your version.)
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     peft_config=peft_config, processing_class=tokenizer)
trainer.train()
trainer.save_model("out/llama3.1-8b-qlora")   # saves the small ADAPTER (MBs), not the full model
```

**Before training, sanity-check two things on a real batch:**
1. Print a decoded sample — confirm the chat template, special tokens, and BOS/EOS look right.
2. Confirm some `labels != -100` (completions unmasked) and the prompt tokens *are* `-100` (masked).

---

## 2. Equivalent Axolotl-style YAML (config-driven)

Same run expressed declaratively. Field names track Axolotl's schema — **check current docs**; treat this as
the shape to imitate.

```yaml
base_model: meta-llama/Llama-3.1-8B
load_in_4bit: true            # QLoRA: 4-bit NF4 frozen base
adapter: qlora                # vs `lora` (16-bit base) or unset (full fine-tune)

# LoRA hyperparameters
lora_r: 16
lora_alpha: 32                # effective scale = alpha/r
lora_dropout: 0.05
lora_target_linear: true      # == target all linear layers (attention + MLP)
# lora_target_modules:        # or list them explicitly
#   [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

datasets:
  - path: your-org/your-sft-data
    type: chat_template        # use the model's chat template
chat_template: llama3
train_on_inputs: false         # <-- completion-only loss (mask the prompt). Critical.

sequence_len: 2048
sample_packing: true           # pack short samples for throughput
pad_to_sequence_len: true

# Memory / speed
bf16: true
gradient_checkpointing: true
flash_attention: true

# Schedule
micro_batch_size: 2
gradient_accumulation_steps: 8 # effective batch = micro * accum * num_gpus
num_epochs: 2
learning_rate: 0.0002
lr_scheduler: cosine
warmup_ratio: 0.03
optimizer: paged_adamw_8bit     # paged optimizer (the QLoRA spill-to-CPU trick)
```

Key parity points with the Python version: **4-bit NF4 base, LoRA over attention + MLP, completion-only
loss (`train_on_inputs: false`), packing, paged optimizer, bf16 + gradient checkpointing.**

---

## 3. Memory-budget note (single-GPU, order-of-magnitude)

VRAM ≈ base weights + adapter (grads+optimizer) + activations. Rough per-param weight costs:

| Component | Full FT (Adam, bf16) | LoRA (16-bit base) | QLoRA (NF4 base) |
|---|---|---|---|
| Base weights | ~2 B/param (trainable) | ~2 B/param (frozen) | **~0.5 B/param (frozen)** |
| Grads + Adam optimizer | ~14 B/param (master + 2 moments) | only on adapter (<1% params) | only on adapter (<1% params) |
| **Rough total for 8B (pre-activations)** | **~110GB+ → needs multi-GPU** | ~16GB+ | **~6–10GB** |

So **QLoRA puts a 7–13B fine-tune on a single 24GB GPU and ~70B on a single 80GB GPU** (QLoRA paper scale;
seq-length/batch dependent — **verify against current docs**). Activations scale with `batch × seq_len`;
gradient checkpointing slashes them. If you OOM, in order: lower `micro_batch_size` (raise grad-accum to
keep effective batch), shorten `sequence_len`, ensure gradient checkpointing is on, go 4-bit + paged
optimizer, then try Unsloth kernels. Full fine-tuning the same models needs FSDP/ZeRO across GPUs
([[training-frameworks]]).

---

## 4. Merge & serve note

**Option A — merge for a single dedicated model** (zero adapter overhead at inference):

```python
from peft import AutoPeftModelForCausalLM
# Load the base in 16-bit (NOT 4-bit) for a clean merge, attach the adapter, fold in ΔW:
model = AutoPeftModelForCausalLM.from_pretrained(
    "out/llama3.1-8b-qlora", torch_dtype="bfloat16")
merged = model.merge_and_unload()              # W' = W + (alpha/r)·B·A
merged.save_pretrained("out/llama3.1-8b-merged")
tokenizer.save_pretrained("out/llama3.1-8b-merged")
# Then optionally re-quantize the merged model for serving.
```

**Gotcha:** merging a LoRA *in place into a 4-bit base* can degrade quality. Merge into a 16-bit base, then
re-quantize for deployment. **Pin the chat template** used at serving to exactly the one used in training.

**Option B — keep adapters separate: one base + many LoRAs.** Don't merge. Ship the small adapter (MBs) and
let the serving stack load/route adapters per request, hosting one copy of the base weights for many cheap
fine-tunes. vLLM/SGLang and the GKE Inference Gateway support multi-LoRA serving and LoRA-aware routing —
see [[serving-frameworks]] and [[gke-inference-gateway]]. This is the right pattern when you have many
task-specific adapters over a shared base.
