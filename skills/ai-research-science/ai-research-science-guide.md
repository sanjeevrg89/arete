# AI Research Science — Deep Reference

This is the research-science layer: theory, mechanism, evidence, and open problems across the LLM
lifecycle. It deliberately stops at the engineering boundary — for *how to run* the thing, follow the
Wiki-style `[[<name>]]` wiki-links connect the sibling skills. The field moves weekly (it is 2026); every benchmark number,
default hyperparameter, and arXiv ID below is provisional. **Verify against current papers/docs**, and
where a citation ID is not certain it is named by authors/title with "(verify the citation)".

---

## 1. Research methodology & mindset

The job is to convert intuition into **falsifiable hypotheses** and then to design the experiment that
could kill the hypothesis cheaply. A research scientist's edge is not ideas (everyone has them) but
**experimental hygiene** and **taste about what is worth measuring**.

**Hypothesis → prediction → controlled test.** State the mechanism, state what you'd see if it's true
*and* what you'd see if it's false, then build the smallest experiment that discriminates. "X improves
loss" is not a hypothesis; "X improves loss *because* it reduces gradient variance, so the gain should
grow with batch size and vanish at batch size 1" is — and it's refutable.

**Ablations and controls.** Change one variable; hold everything else fixed — same tokens seen, same
compute budget, same data order where feasible, same eval. The cardinal sin is the **unmatched
baseline**: comparing your method trained for more steps / more data / a bigger model against a weaker
control. Report multiple seeds and the seed-to-seed variance; a 0.3% benchmark delta inside the noise
band is nothing. Prefer **learning-curve comparisons** (loss vs tokens/FLOPs) over single end-point
numbers, because they reveal whether a gain is a constant offset or a change in scaling slope.

**The reproducibility crisis.** Much of the literature does not replicate: undisclosed hyperparameter
tuning on the test set, cherry-picked seeds, baselines run at a disadvantage, and "improvements" that
are within noise. Treat a single paper's headline result as a *prior*, not a fact, until it survives an
independent reproduction or an apples-to-apples bake-off. When you reproduce, reproduce the **baseline**
first and confirm you can match its reported number — if you can't, you cannot trust your delta.

**Eval rigor & contamination.** This dominates everything. A capability gain that is really
train/test overlap is worse than no gain — it actively misleads. Benchmark contamination (the eval set,
or near-duplicates, in pretraining data), prompt-format sensitivity, and answer-extraction bugs routinely
swamp real effects. Decontaminate, hold out fresh evals, measure variance, and prefer evals that are
hard to game. The full discipline lives in [[ml-evaluation-evals]]; decontamination of corpora in
[[pretraining-data-tokenizers]].

**Compute-optimal experimentation.** Do research at the smallest scale where the effect is visible, then
confirm the **trend** holds with scale rather than re-running everything large. Use scaling laws to plan:
fit a small ladder of model sizes, predict the large run, and only spend the big compute when the
extrapolation justifies it. A negative result at small scale that you *expect* to flip with scale is a
hypothesis about an emergent effect — design for it explicitly.

**The bitter lesson** (Sutton, 2019): methods that scale with compute and data generally beat methods
that bake in human structure. The corollary is not "never use inductive bias" but "be suspicious of
cleverness that doesn't ride the compute curve." Architecture tricks that help at 100M params and vanish
at 10B are the canonical trap.

**Staying current with the arXiv firehose.** You cannot read everything. Read **abstracts and figures
first**; the experiments section tells you whether to trust the claim; the related-work tells you the
real novelty. Track a small set of credible groups and reproduction threads. Weight results by whether
the baselines are honest and the ablations are matched, not by the size of the claimed number.

---

## 2. Architecture science

### The transformer, viewed as a residual stream
The most useful mental model (popularized by the interpretability literature, Elhage et al. 2021,
"A Mathematical Framework for Transformer Circuits") is the **residual stream**: a per-token vector that
every layer *reads from and adds to*. Attention moves information *between* token positions; the MLP
processes information *within* a position. Because contributions are additive, layers compose, and you
can reason about features as directions in the stream.

**Attention variants** trade quality against the KV-cache memory bandwidth that bottlenecks decode:
- **MHA** — H independent query/key/value heads. Maximum expressivity, maximum KV cache.
- **MQA** (Shazeer, 2019) — all query heads share *one* K/V head. Shrinks KV cache ~H×, small quality hit.
- **GQA** (Ainslie et al., 2023, arXiv:2305.13245 — verify) — G key/value groups; interpolates MHA↔MQA.
  The default in most modern open models because it recovers nearly all quality at a fraction of the cache.
- **MLA** (multi-head latent attention, DeepSeek-V2) — compresses K/V into a low-rank latent that is
  cached, then reconstructs per head. Cuts cache further while preserving multi-head expressivity.

Mechanism detail — KV cache is the decode bottleneck; the rest is [[inference-optimization]]/
[[serving-frameworks]]. The research point: these are all ways to reduce the rank/footprint of cached
state with minimal loss, and the empirical lesson is that attention is **over-parameterized** at the K/V
side.

**Positional encoding.** Self-attention is permutation-equivariant, so position must be injected.
- **Absolute** (sinusoidal/learned, original transformer) — simple, extrapolates poorly.
- **RoPE** (Su et al., 2021, RoFormer, arXiv:2104.09864 — verify) — rotates Q/K by a position-dependent
  angle so the dot product depends on *relative* position. Now near-universal. Per-dimension frequencies
  set the effective wavelength.
- **ALiBi** (Press et al., 2021) — adds a linear distance penalty to attention scores; cheap, decent
  length extrapolation, but largely superseded by RoPE+extension methods.
- **Context extension**: RoPE breaks beyond its trained length. **Position Interpolation** (Chen et al.,
  2023) compresses positions; **NTK-aware / YaRN** (Peng et al., 2023, arXiv:2309.00071 — verify) scale
  frequencies non-uniformly so high-frequency (local) detail survives while low-frequency (global) range
  extends. These usually need a short fine-tune at the target length.

**Normalization.** **Pre-LN** (norm *before* the sublayer, inside the residual branch) is the stable
default — it keeps the residual identity path clean and tames gradients at depth, at a small quality cost
vs post-LN. **RMSNorm** (Zhang & Sennrich, 2019) drops the mean-centering and bias, cheaper and
empirically as good. **QK-norm** (normalize queries/keys before the dot product) is a key stability lever
at scale — it caps attention-logit growth, a common cause of loss spikes. Some recent models add a final
norm and/or "sandwich" norms; verify per architecture.

**Activations.** GeLU replaced ReLU; **SwiGLU** (Shazeer, 2020, "GLU Variants Improve Transformer") — a
gated linear unit with a SiLU/Swish gate — is the modern default in the MLP and reliably improves loss
per FLOP (at the cost of a third weight matrix, so the hidden dim is scaled to keep params matched).

### Mixture-of-Experts (MoE)
Replace the dense MLP with N expert MLPs and a **router** that sends each token to top-k experts (k=1–2).
This decouples **parameters** (total capacity) from **FLOPs per token** (only k experts fire), so you
buy capacity cheaply. Foundational: **Switch Transformer** (Fedus et al., 2021, top-1 routing) and
**GShard**; **Mixtral 8×7B** (Jiang et al., 2024) made sparse MoE mainstream in open models; DeepSeek-MoE
added fine-grained + shared experts.
- **Load-balancing loss**: routers collapse toward a few experts; an auxiliary balancing loss (or
  DeepSeek's loss-free, bias-based balancing — verify) pushes uniform utilization. Too much balancing
  pressure hurts quality; it's a tension, not a free lunch.
- **Open research**: token-dropping vs capacity factor, router instability, expert specialization (do
  experts learn interpretable roles? mostly weakly), and whether routing should be token-, sequence-, or
  layer-level. Expert *parallelism* and all-to-all comm are [[training-frameworks]]/[[serving-frameworks]].

### State-space models, Mamba, and hybrids
Attention is O(L²) compute and O(L) cache. **SSMs** model sequences as a linear recurrence
(continuous-time state space, discretized), giving O(L) compute and O(1) per-step state. **S4** (Gu et
al., 2021) and **Mamba** (Gu & Dao, 2023, arXiv:2312.00752 — verify) added input-dependent (selective)
state transitions plus a hardware-aware scan, making SSMs competitive with transformers at moderate
scale. The fundamental tradeoff: a **fixed-size state cannot losslessly recall arbitrary past tokens**,
so pure SSMs underperform attention on copying/retrieval ("needle in a haystack"). **Hybrids** (interleave
a few attention layers among many SSM/Mamba layers, e.g. Jamba-style) get linear-ish cost with attention's
exact recall where it matters — the current practical sweet spot.

### Long-context science & "is architecture mostly plumbing?"
Long context is partly architecture (positional extension, attention/SSM cost) and partly **data and
objective** — models must be *trained* on long sequences to use them. Evaluate with retrieval/recall
probes, not just perplexity, because perplexity hides "lost in the middle" failures (Liu et al., 2023).
The recurring empirical finding: **above a competence threshold, many architectural changes wash out** —
data quality, scale, and optimization often dominate. The honest position is that a *handful* of changes
(GQA, RoPE+extension, RMSNorm/pre-LN, SwiGLU, FlashAttention as an exact-attention enabler) are robust
wins, while most novel attention variants fail to replicate their gains at scale.

---

## 3. Training science

### Optimizers
- **SGD(+momentum)** dominates vision but is hard to tune for transformers (loss landscape is badly
  conditioned, heavy-tailed gradients).
- **Adam / AdamW** is the workhorse. Adam is per-parameter adaptive (first/second moment estimates ≈ a
  diagonal preconditioner). **AdamW** (Loshchilov & Hutter, 2017) decouples weight decay from the
  gradient update — the correct way to regularize with adaptive optimizers; coupling decay into the
  gradient (classic "Adam + L2") interacts badly with the adaptive scaling.
- **Adafactor** (Shazeer & Stern, 2018) factorizes the second-moment matrix to save optimizer memory —
  relevant at scale and on TPUs.
- **Lion** (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms") — sign-based update,
  cheaper state; sometimes matches AdamW, sensitive to LR/decay.
- **Muon** (recent; verify current evidence) — orthogonalizes the momentum update for 2D weight matrices
  (a Newton-Schulz step approximating a matrix preconditioner), reported to improve compute-efficiency on
  some LLM training; **fast-moving, verify against current results before relying on it.**
- **Second-order intuition**: full Newton/K-FAC/Shampoo precondition by curvature and converge in fewer
  steps but cost memory/compute; Adam is a cheap diagonal approximation. The frontier debate is how much
  of Shampoo/Muon's benefit survives at frontier scale.

### Learning-rate schedule
Warmup (linearly ramp LR over the first fraction of steps) is near-mandatory for transformer stability —
it avoids early large updates while moment estimates are unreliable. Then **cosine decay** to a small
floor is the classic default; the LR floor and total-steps assumption matter. **WSD / trapezoidal**
(warmup–stable–decay) schedules decouple the decay from a pre-committed horizon, enabling continued
training and clean scaling-law ladders — increasingly popular; verify current consensus.

### Batch size & the gradient noise scale
**Larger batches reduce gradient noise** but with diminishing returns. The **gradient noise scale**
(McCandlish et al., 2018, "An Empirical Model of Large-Batch Training") predicts the **critical batch
size**: below it, doubling batch ≈ halving steps (data-parallel scaling is "free"); above it, you mostly
waste compute for little step-count reduction. The noise scale grows during training (and with task
difficulty), so the optimal batch size *increases* over a run. This is the theory under "how big should
my global batch be" — the parallelism mechanics are [[training-frameworks]].

### Initialization & stability
Init controls the scale of activations/gradients at step 0. Scale residual-branch outputs down (e.g.
1/√(2·n_layers)-style factors) so the residual stream doesn't blow up with depth. **Loss spikes** are the
characteristic pathology of large runs: a sudden divergence, often from attention-logit blow-up or
fp-range overflow. Defenses, in order of leverage: **QK-norm** (caps attention logits), **z-loss** (a
small penalty on the softmax normalizer / logsumexp that keeps logits bounded — from the PaLM line of
work), careful init, gradient clipping, LR/warmup tuning, and **bf16 with fp32 master weights**. A spike
that doesn't recover usually means rolling back to a checkpoint, skipping the offending data batch, and
lowering LR — checkpoint hygiene is [[ml-checkpointing-orbax]]/[[training-frameworks]].

### Numerics: bf16 vs fp8
**bf16** (8-bit exponent, 7-bit mantissa) is the de facto training precision — same dynamic range as fp32,
so it rarely overflows, at the cost of precision (hence fp32 master weights + fp32 accumulation). **fp8**
training (E4M3 for forward, E5M2 for gradients, with per-tensor/per-block scaling) is the frontier
efficiency play but has **tighter dynamic range** → it needs careful scaling/delayed-scaling and is more
prone to instability; the research question is which tensors tolerate fp8 and which must stay higher
precision. Substrate detail (tensor cores, accumulation) is [[ml-frameworks]]; squeezing inference
precision is [[inference-optimization]].

### The pretraining objective
- **Next-token (causal LM)** — the dominant objective; dense self-supervision, scales cleanly.
- **MLM** (BERT-style masked LM) — bidirectional, great for encoders/representations, not for generation.
- **UL2 / mixture-of-denoisers** (Tay et al., 2022) — mix span-corruption and causal objectives to get
  both worlds; influential, niche in practice.
- **FIM** (fill-in-the-middle, Bavarian et al., 2022) — reorder documents (prefix–suffix–middle) so a
  causal model learns infilling; standard for code models, essentially free quality if the doc fraction
  is tuned.

### Scaling laws
- **Kaplan et al. (2020)** established power-law loss-vs-(params, data, compute) and argued for *large
  models, modest data*. Their token/parameter recommendation was later shown to be biased by a fixed/short
  LR schedule.
- **Chinchilla** (Hoffmann et al., 2022, arXiv:2203.15556 — verify) re-derived the **compute-optimal**
  frontier: for a fixed FLOP budget, scale params and tokens **together** (~20 tokens/param). Gopher/GPT-3
  were badly *undertrained*; Chinchilla (70B, 1.4T tokens) beat Gopher (280B) at less compute.
- **Inference-aware scaling**: Chinchilla minimizes *training* compute. If you'll serve the model a lot,
  it's rational to train a **smaller model on far more tokens** (well past 20:1) — you pay more training
  to save inference forever. This is why modern open models are "overtrained" relative to Chinchilla.
- **Data-constrained scaling** (Muennighoff et al., 2023) — when unique data runs out, repeating data has
  diminishing (eventually negative) returns; up to a few epochs is roughly as good as fresh data, then it
  decays. Sets the value of data curation/dedup ([[pretraining-data-tokenizers]]).
- **Emergent abilities debate**: Wei et al. (2022) reported sharp capability jumps at scale; Schaeffer et
  al. (2023, "Are Emergent Abilities a Mirage?") argued many "emergences" are artifacts of
  **discontinuous metrics** (exact-match), and smooth under continuous metrics. The research stance:
  predictability of capability is the goal; pick metrics that don't manufacture discontinuities.

---

## 4. Inference & test-time-compute science

### Decoding / sampling theory
A trained LM defines a distribution; decoding chooses how to sample from it.
- **Temperature** rescales logits: T<1 sharpens (more greedy), T>1 flattens (more diverse). T→0 is greedy.
- **Top-k / top-p (nucleus)** truncate the tail to cut incoherent low-probability tokens; **min-p**
  thresholds relative to the top token, adapting the cutoff to the distribution's peakiness.
- **Beam search** approximates the MAP sequence; great for low-entropy tasks (translation), but for
  open-ended generation it produces bland, repetitive text — the **likelihood-quality mismatch**: the
  most probable sequence is often not the best one.
- **Calibration view**: a well-calibrated model's token probabilities match empirical frequencies. RLHF
  tends to **mis-calibrate** confidence (the model becomes overconfident / mode-collapsed); base models
  are often better calibrated than their aligned descendants. Sampling choices interact with calibration
  and with downstream verification.

### The test-time-compute paradigm
Accuracy can be bought with **inference FLOPs**, a scaling axis orthogonal to model size:
- **Chain-of-thought** (Wei et al., 2022) — let the model emit intermediate steps; works because it
  conditions later tokens on its own reasoning (more serial computation).
- **Self-consistency** (Wang et al., 2022) — sample many CoT paths, majority-vote the answer. Cheap,
  strong on tasks with a checkable final answer.
- **Search + verification** — generate many candidates and select with a **verifier/reward model**;
  **best-of-N** (rejection sampling) is the simplest; tree search (e.g. beam/MCTS over reasoning steps)
  spends more. The verifier's quality caps the gain — and an imperfect verifier is itself hackable.
- **o1 / DeepSeek-R1-style long reasoning** — models **RL-trained** to produce long
  chains-of-thought (a private reasoning trace) that self-correct and backtrack, then emit a final answer. The key shift: the
  reasoning is *learned via RL on verifiable rewards*, not merely prompted. DeepSeek-R1 (2025) showed that
  large-scale RL on math/code with rule-based rewards can induce reasoning, and that an intermediate
  "R1-Zero" trained with RL *without* SFT develops reasoning but with readability/language-mixing issues.
- **Process- vs outcome-reward-guided search** — an **ORM** scores only the final answer; a **PRM**
  scores each step ("Let's Verify Step by Step", Lightman et al., 2023). PRMs give denser signal and
  better search/credit assignment but require step-level labels (human or automatically via rollouts,
  e.g. Math-Shepherd — verify). The mechanics of speculative/efficient decoding are
  [[inference-optimization]]/[[serving-frameworks]]; the *why test-time compute scales* is here.

---

## 5. Fine-tuning & adaptation science

### SFT / instruction-tuning data science
SFT teaches **format and behavior**, not (much) new knowledge — the model mostly surfaces capabilities it
already learned in pretraining. The dominant lesson (LIMA, Zhou et al., 2023, "Less Is More for
Alignment"): a **small set of high-quality, diverse** instruction examples beats a large noisy set. Data
*diversity and quality* dominate quantity. Loss-masking (train only on the completion, not the prompt),
chat templates, and packing are practice → [[fine-tuning-peft]].

### Why PEFT / LoRA works — intrinsic dimensionality
Aghajanyan et al. (2020, "Intrinsic Dimensionality Explains the Effectiveness of Language Model
Fine-Tuning") showed task adaptation lives in a **very low-dimensional subspace** — you can fine-tune by
optimizing a few hundred/thousand parameters in a random subspace. **LoRA** (Hu et al., 2021,
arXiv:2106.09685 — verify) operationalizes this: freeze W, learn a low-rank update ΔW = BA (rank r ≪ d),
so the *fine-tuning update*, not the weights, is assumed low-rank. This is why LoRA recovers most of full
fine-tuning quality at a fraction of the trainable params/memory, why merging the adapter back is exact,
and why **DoRA** (decompose into magnitude+direction) and **QLoRA** (LoRA over a 4-bit-quantized base)
extend it. Rank/alpha/target-modules tuning and the memory math are [[fine-tuning-peft]].

### Catastrophic forgetting & continual learning
Fine-tuning on a narrow task degrades unrelated capabilities — gradient updates overwrite shared
features. Mitigations: replay/rehearsal (mix in pretraining/general data), regularization toward the base
(EWC-style, or just a KL/L2 anchor), low-rank/adapter isolation (PEFT forgets less because most weights
are frozen), and **lower learning rates / fewer epochs**. The research framing connects to the
plasticity–stability tradeoff and to **why RLHF uses a KL-to-reference penalty** (§6): keep the new policy
near the capable base.

### Model merging
Combine fine-tuned models in **weight space** without retraining — surprisingly effective because
fine-tunes of a shared base stay in a connected, near-linear region (mode connectivity / linear mode
connectivity).
- **Task arithmetic** (Ilharco et al., 2022) — a **task vector** = (fine-tuned − base) weights; adding
  vectors composes skills, negating one *removes* a behavior.
- **Model soups** (Wortsman et al., 2022) — average the weights of many fine-tunes of the same base;
  often beats the best single model and ensembling at *zero* extra inference cost.
- **TIES** (Yadav et al., 2023) — **T**rim small deltas, **E**lect a sign per parameter, merge only
  agreeing signs — resolves interference between task vectors.
- **DARE** (Yu et al., 2023) — randomly **drop** and **rescale** deltas before merging; removes redundancy
  so many models merge with less interference.
- **SLERP** — spherical interpolation between two models' weights, preserving norm; common for 2-way merges.
Open question: merging is empirically robust but theoretically under-explained; it works best for models
sharing a base and breaks across very different fine-tunes.

### Knowledge distillation theory
Train a **student** on a **teacher**'s outputs. **Soft targets** (Hinton et al., 2015) carry "dark
knowledge" — the relative probabilities of wrong classes encode similarity structure the hard label
lacks; a temperature on the teacher logits exposes it. For LLMs, **sequence-level** distillation (train on
teacher-generated sequences) and **on-policy** distillation (student generates, teacher scores its own
samples) reduce exposure bias and outperform naive token-level KL. Distillation is also how reasoning is
cheaply propagated (R1-distilled small models). Distillation *mechanics/efficiency* → [[inference-optimization]].

---

## 6. RL & RLHF / post-training science (deepest)

Post-training turns a capable but unsteerable base model into a helpful, harmless, honest assistant — or
into a reasoner. The unifying frame: **optimize the policy against a reward signal while staying close to
a reference policy.** Engineering (frameworks, memory budgeting, rollout/learner split, weight resharding)
is [[rl-rlhf-frameworks]]; the science is here.

### The alignment-from-feedback pipeline (InstructGPT)
Ouyang et al. (2022, InstructGPT, arXiv:2203.02155 — verify) is the canonical recipe:
1. **SFT** — supervised fine-tune on demonstrations (§5) to get a reasonable policy.
2. **Reward model (RM)** — collect human **preferences** (A vs B), fit an RM that scores responses.
3. **RL (PPO)** — optimize the SFT policy to maximize RM reward, penalized by KL divergence from the SFT
   reference so it doesn't drift into degenerate, reward-hacking text.

### Reward modeling
Humans are far better at **comparing** two responses than at scoring one. The **Bradley-Terry** model
turns pairwise preferences into a scalar reward: P(A≻B) = σ(r(A) − r(B)); train the RM by maximizing the
log-likelihood of observed preferences. The RM is typically the LM with a scalar head.
- **Reward hacking / overoptimization / Goodhart's law**: "when a measure becomes a target, it ceases to
  be a good measure." The RM is a *proxy* for human preference; push the policy too hard and it exploits
  RM idiosyncrasies (verbosity, formatting, flattery) — true quality rises then **falls** as RM score
  keeps climbing. Gao et al. (2022, "Scaling Laws for Reward Model Overoptimization") quantified this:
  measured-vs-proxy reward diverges as KL from the reference grows, and the gap shrinks with RM size/data.
- **RM ensembles / uncertainty** — averaging multiple RMs (or penalizing RM disagreement) curbs
  overoptimization by not trusting any single proxy in regions where they disagree.
- The KL-to-reference penalty (below) is the primary leash on hacking; conservatism in the RM is the other.

### PPO for LLMs
The RL objective, concretely: maximize **E[ r(x,y) − β·KL(π_θ(·|x) ‖ π_ref(·|x)) ]**. The KL term keeps
the policy near the SFT reference (prevents collapse and limits hacking); β controls the leash. PPO
(Schulman et al., 2017) optimizes this with a **clipped surrogate** that bounds the policy ratio per
update for stability, plus:
- a **value/critic** network estimating expected return (a second large model in memory), and
- **GAE** (generalized advantage estimation, Schulman et al., 2015) to trade bias/variance in the
  advantage.
Practical instabilities: reward/advantage normalization is finicky; the critic is hard to fit on sparse
sequence-level reward; KL can be estimated several ways (and the choice matters); value-function init and
the four model copies (policy, reference, reward, critic) make it **expensive and brittle**. Much of the
field's energy since has gone into removing pieces of this machinery.

### The direct-alignment family (skip the RL loop)
- **DPO** (Rafailov et al., 2023, "Direct Preference Optimization", arXiv:2305.18290 — verify). The key
  insight: the **RLHF objective has a closed-form optimal policy**, and you can **reparameterize the
  reward in terms of the policy itself** — r(x,y) = β·log[π_θ(y|x)/π_ref(y|x)] + const. Substituting into
  the Bradley-Terry likelihood gives a **simple classification loss on preference pairs**, with *no
  reward model and no sampling*. DPO is stable, cheap, and offline. Caveats: it is **off-policy** (trains
  on a fixed preference dataset, not on its own generations), can over-optimize the chosen/reject margin,
  and is sensitive to the reference and to distribution shift between the preference data and the policy.
- **IPO** (Azar et al., 2023) — replaces DPO's logistic loss with a squared loss on the implicit reward
  gap; fixes DPO's tendency to overfit to (near-)deterministic preferences.
- **KTO** (Ethayarajh et al., 2024) — uses **prospect-theory** human-utility shaping and needs only
  **binary** good/bad labels (not pairs), which are far cheaper to collect.
- **ORPO** (Hong et al., 2024) — folds preference optimization **into SFT** with an odds-ratio penalty,
  removing the separate reference model / second stage entirely.
- **SimPO** (Meng et al., 2024) — **reference-free**, uses **length-normalized** average log-prob as the
  implicit reward plus a margin; cheaper and counters length bias, but loses the reference anchor.

| Method | Needs RM? | Needs ref? | Data | Notable property |
|---|---|---|---|---|
| PPO | yes | yes | online rollouts | on-policy, strongest ceiling, brittle/expensive |
| DPO | no | yes | offline pairs | stable, simple, off-policy |
| IPO | no | yes | offline pairs | fixes DPO overfit on hard prefs |
| KTO | no | yes | binary labels | cheap labels, prospect-theory loss |
| ORPO | no | no | pairs (+SFT) | single-stage, no ref |
| SimPO | no | no | pairs | length-normalized, reference-free |

### GRPO & RL with verifiable rewards (RLVR)
**GRPO** (Group Relative Policy Optimization, DeepSeek; popularized by DeepSeekMath/DeepSeek-R1, 2024–25)
**drops the critic**: for each prompt, sample a **group** of G responses, score each, and use the
**group-normalized reward** (subtract the group mean, divide by std) as the advantage. This kills the
value-network cost/instability and is a natural fit when reward is sparse and verifiable. **RLVR** is the
broader paradigm: the reward is a **rule/verifier**, not a learned RM — `==` on a math answer, unit-tests
passing for code, a parser checking format. Verifiable rewards **cannot be hacked the way a learned RM
can** (the answer is either right or not), which is why RLVR drove the o1/R1 reasoning breakthroughs.
- **Outcome vs process**: RLVR usually uses **outcome** reward (final answer). **PRMs** ("Let's Verify
  Step by Step") reward each step — denser credit assignment, better for search, but step labels are
  expensive and PRMs can themselves be gamed. R1 notably got far with *outcome-only, rule-based* rewards.
- **Rule-based rewards** also encode format/safety constraints cheaply (e.g. reward correct
  answer-tag structure), avoiding a learned proxy for objectively checkable properties.

### RLAIF / Constitutional AI
**RLAIF** replaces (some) human preference labels with **AI feedback** — a model judges responses,
massively cheaper and more scalable. **Constitutional AI** (Bai et al., 2022, Anthropic) uses a written
**constitution** (a set of principles): the model **critiques and revises** its own responses against the
principles (the SL stage), then preferences are **generated by a model** against the constitution to train
the RM (the RL stage, "RLAIF"). The research point: alignment signal can be **bootstrapped** from
principles + a capable model, reducing the human-label bottleneck — at the risk of inheriting/amplifying
the judge model's biases.

### Online vs offline, on- vs off-policy
- **On-policy** (PPO, GRPO) — train on the policy's *own current* generations; better exploration and
  ceiling, but you pay for generation every step (the rollout throughput bottleneck → [[rl-rlhf-frameworks]]).
- **Off-policy / offline** (DPO and kin) — train on a fixed dataset; cheap and stable, but suffers
  **distribution shift** — the preference data stops matching what the improving policy actually
  generates, so quality plateaus. **Iterative DPO** (regenerate preferences from the current policy each
  round) partially closes the gap, blurring the online/offline line. The empirical consensus (2024–25):
  on-policy generally beats offline at the top end, but the gap depends heavily on data and tuning —
  **verify against current bake-offs.**

### Self-play / self-improvement / iterated amplification
- **Rejection sampling fine-tuning / STaR** (Zelikman et al., 2022) — generate, **filter to correct**
  (by a verifier), fine-tune on the survivors; repeat. Simple, strong for verifiable tasks.
- **Iterated amplification / debate** (Christiano et al.; Irving et al.) — use models to *help supervise*
  models on tasks humans can't directly judge — a scalable-oversight research agenda.
- **Self-rewarding / self-improvement loops** — a model generates *and* judges its own training data;
  promising but risks **reward drift** and collapse without an external anchor.

### Open problems (the honest list)
- **Reward hacking / overoptimization** — the central failure; mitigated, not solved, by KL leashes, RM
  ensembles, verifiable rewards, and early stopping on a held-out judge.
- **Length & format bias** — RMs and judges prefer longer, well-formatted answers regardless of quality;
  this leaks into the policy (verbosity inflation). Length-debiasing (e.g. length-normalized rewards,
  controlled-length RMs) is active work.
- **Sycophancy** — models learn to tell users what they want to hear, because humans reward agreement;
  measured and partially mitigable, not eliminated.
- **Mode collapse / diversity loss** — RLHF sharpens the distribution; aligned models are less diverse and
  worse calibrated than their base. A real cost of alignment.
- **Distribution shift** — offline methods degrade as the policy moves away from the preference data.
- **Evaluating alignment** — there is **no ground-truth reward**; we evaluate with the same fallible
  judges/RMs we train against, and benchmarks saturate or get gamed. This circularity is the deepest open
  problem; rigorous, contamination-controlled, multi-judge evaluation ([[ml-evaluation-evals]]) is the
  current best answer, alongside red-teaming and human studies.

---

## 7. Frontier directions & open problems

**Reasoning & agents (research view).** RLVR-trained long reasoning (§4/§6) is the live frontier: how to
get reliable multi-step reasoning, when test-time compute beats more pretraining, and how to do **credit
assignment over long horizons** (PRMs, value learning over trajectories). Agentic settings extend this to
*tool use and environment interaction* — the reward is task success over many steps, exploration and
sparse reward become acute, and evaluation of trajectories (not just answers) is hard. Agent frameworks
are [[llm-app-agent-frameworks]]; the science is reward design and long-horizon RL.

**Multimodal fusion research.** Open questions: early vs late fusion, native-multimodal pretraining vs
bolt-on encoders, tokenizing continuous modalities, and whether a shared representation actually
transfers across modalities or just co-locates them. Cross-modal training dynamics and modality balancing
are [[multimodal-ml]].

**Mechanistic interpretability — the science of understanding models.** Reverse-engineer the computation:
features as directions, **superposition** (more features than neurons → polysemantic neurons; Elhage et
al., 2022), **sparse autoencoders** to extract monosemantic features (Anthropic/others, 2023–24),
circuits, and **induction heads** as a mechanism for in-context learning (Olsson et al., 2022). This is
how we move from "it works" to "we know *why*", and it underpins safety (detecting deception, auditing).

**Data efficiency & the data wall.** High-quality tokens are finite; data-constrained scaling (§3),
synthetic data (with the **model-collapse** risk from training on too much self-generated data — Shumailov
et al., 2023), curriculum, and active selection are the levers. Curation quality increasingly substitutes
for raw scale ([[pretraining-data-tokenizers]]).

**Scaling vs algorithmic progress — the central strategic debate.** Does capability come mainly from more
compute (the bitter lesson) or from better algorithms/data? The honest answer: **both compound** — measured
"algorithmic efficiency" doublings (effective-compute gains at fixed quality) are large, and they multiply
the returns to scale. The strategic question for a lab is where the next marginal FLOP/researcher-hour
buys the most effective compute.

**Safety & alignment research.** Beyond RLHF: **scalable oversight** (supervising superhuman models —
debate, amplification, weak-to-strong generalization, Burns et al. 2023), **robustness** to jailbreaks and
adversarial inputs ([[adversarial-ml-robustness]]), **interpretability-for-safety**, and evaluation of
dangerous capabilities. This is unsolved research, not a compliance checkbox; governance and policy are
[[responsible-ai-governance]].

---

## Canonical references (verify IDs/versions against current sources)

- Vaswani et al., **"Attention Is All You Need"**, 2017 (arXiv:1706.03762).
- Kaplan et al., **"Scaling Laws for Neural Language Models"**, 2020 (arXiv:2001.08361).
- Hoffmann et al., **"Training Compute-Optimal LLMs" (Chinchilla)**, 2022 (arXiv:2203.15556 — verify).
- McCandlish et al., **"An Empirical Model of Large-Batch Training" (gradient noise scale)**, 2018.
- Su et al., **RoFormer / RoPE**, 2021 (arXiv:2104.09864 — verify); Press et al., **ALiBi**, 2021;
  Peng et al., **YaRN**, 2023 (arXiv:2309.00071 — verify).
- Ainslie et al., **GQA**, 2023 (arXiv:2305.13245 — verify); Shazeer, **MQA**, 2019.
- Shazeer, **"GLU Variants Improve Transformer" (SwiGLU)**, 2020; Zhang & Sennrich, **RMSNorm**, 2019.
- Fedus et al., **Switch Transformer**, 2021; Jiang et al., **Mixtral of Experts**, 2024.
- Gu & Dao, **Mamba**, 2023 (arXiv:2312.00752 — verify); Gu et al., **S4**, 2021.
- Dao et al., **FlashAttention**, 2022 (arXiv:2205.14135 — verify).
- Loshchilov & Hutter, **AdamW / decoupled weight decay**, 2017; Chen et al., **Lion**, 2023.
- Elhage et al., **"A Mathematical Framework for Transformer Circuits"**, 2021; **Superposition**, 2022;
  Olsson et al., **"In-context Learning and Induction Heads"**, 2022.
- Wei et al., **Chain-of-Thought**, 2022; Wang et al., **Self-Consistency**, 2022; Schaeffer et al.,
  **"Are Emergent Abilities a Mirage?"**, 2023.
- Hu et al., **LoRA**, 2021 (arXiv:2106.09685 — verify); Aghajanyan et al., **Intrinsic Dimensionality**, 2020.
- Ilharco et al., **Task Arithmetic / editing with task vectors**, 2022; Wortsman et al., **Model Soups**,
  2022; Yadav et al., **TIES-Merging**, 2023; Yu et al., **DARE**, 2023.
- Hinton et al., **Distilling the Knowledge in a Neural Network**, 2015.
- Ouyang et al., **InstructGPT**, 2022 (arXiv:2203.02155 — verify); Bai et al., **Constitutional AI**, 2022.
- Schulman et al., **PPO**, 2017; **GAE**, 2015; Gao et al., **RM Overoptimization scaling laws**, 2022.
- Rafailov et al., **DPO**, 2023 (arXiv:2305.18290 — verify); Azar et al., **IPO**, 2023;
  Ethayarajh et al., **KTO**, 2024; Hong et al., **ORPO**, 2024; Meng et al., **SimPO**, 2024.
- Lightman et al., **"Let's Verify Step by Step" (PRMs)**, 2023; DeepSeek-AI, **DeepSeek-R1 / GRPO**, 2025
  (verify the citation).
- Zhou et al., **LIMA**, 2023; Muennighoff et al., **Data-Constrained Scaling**, 2023; Sutton,
  **"The Bitter Lesson"**, 2019.
