# Skill Authoring Spec (read before writing any skill)

This is a cross-agent skill library, open for anyone to use. Each skill is a self-contained directory consumed by
coding assistants (Claude Code via `SKILL.md`, Codex/Cursor/etc. via `AGENTS.md`, Gemini via imports).
Author every skill to this exact standard so the library is consistent.

The reference exemplar is `go-best-practices/` (see its `SKILL.md`, `AGENTS.md`, `go-guidelines.md`).
Mirror its structure and voice.

## Files every skill directory MUST contain

1. **`SKILL.md`** — Claude Code entry point. YAML frontmatter + concise body.
   ```
   ---
   name: <exact-kebab-slug-matching-the-directory-name>
   description: <2–4 dense sentences. THIS IS THE ROUTER — it decides when the skill loads.
     Lead with WHEN to use it, pack in concrete trigger terms (tool names, API kinds, file types,
     error symptoms, tasks), and what it covers. Mention the specific technologies by name.>
   ---

   # <Skill Title>

   <1–2 line statement of the expertise and the bar (e.g. "Apply the judgment of an engineer who has
   run this in production at scale for years.")>

   ## How to use this skill
   1. Read `<slug>-guide.md` in this directory — the full reference. Apply it to the task.
   2. <any examples.md / sub-references>
   3. Match the surrounding codebase/cluster conventions; apply correctness/safety rules regardless.

   ## Essentials (full detail in `<slug>-guide.md`)
   - <8–15 of the highest-value, most load-bearing bullets a top engineer would insist on>

   ## Related skills
   - `[[other-slug]]` — when to reach for it instead / in addition.
   ```

2. **`<slug>-guide.md`** — THE deep reference. This is the meat. **250–450 lines.** Sectioned with
   `##` headings. Must include, adapted to the topic:
   - **Mental model / architecture** — how the thing really works, not a feature list.
   - **Core concepts** — the objects/APIs/abstractions, with precise definitions.
   - **Hands-on** — real, correct artifacts: `kubectl`/YAML/Go/Python/CLI. No pseudo-code where real
     code is possible. Show the canonical idiom.
   - **Best practices** — opinionated, with the *why*.
   - **Anti-patterns / gotchas** — the traps that bite people in production. Be specific.
   - **Performance / scale** — where relevant (throughput, latency, memory, large clusters, big models).
   - **Troubleshooting** — concrete symptoms → diagnosis → fix.
   - **Security / multi-tenancy** — where relevant.
   - **Version awareness** — note that the ecosystem moves fast (it is 2026); flag where APIs/versions
     matter and tell the reader to verify current docs. Don't invent version numbers you're unsure of.
   - **Canonical references** — authoritative links (project docs, KEPs, papers, source). Real URLs only.

3. **`AGENTS.md`** — cross-tool always-on summary. Short. Header pointing to `<slug>-guide.md` as the
   authoritative source, then a condensed always-on checklist of the highest-value rules. Keep it small
   (it is loaded into context every turn for tools that use it) — point to the guide for depth.

4. **`examples.md`** (optional but encouraged where patterns help) — before/after or canonical
   worked examples (YAML/code) the agent can imitate.

## Quality bar

- Write as a **top-5-in-the-world practitioner with ~10 years of production experience** in the topic.
  Dense, concrete, opinionated, correct. Signal over volume.
- **Accuracy over completeness.** If you are unsure whether a detail is current/correct, say so or omit
  it — never fabricate API fields, flags, version numbers, or benchmark figures.
- Prefer the canonical/idiomatic approach; call out common-but-wrong patterns explicitly.
- Real commands and manifests must be runnable-in-spirit and correct (right apiVersion/kind/fields).
- Cross-link related skills by slug using `[[slug]]` so the library forms a graph.

## The 14 skills in this library (use these exact slugs for cross-links)

- `go-best-practices`
- `kubernetes-expert` — end-to-end practitioner mastery (using K8s)
- `kubernetes-controller-expert` — writing controllers (controller-runtime, client-go, reconcile)
- `kubernetes-operator-expert` — operator pattern, CRDs, webhooks, kubebuilder/OLM
- `kubernetes-internals-expert` — apiserver/etcd/scheduler/kubelet/kube-proxy internals
- `aiml-on-kubernetes` — training/inference/fine-tuning/RL/RLHF/agentic on K8s & GKE (umbrella)
- `kueue-advanced` — Kueue batch/quota/gang/MultiKueue/TAS
- `jobset-leaderworkerset` — JobSet + LeaderWorkerSet for multi-host training/inference
- `ml-frameworks` — PyTorch, JAX, XLA, GPU & TPU
- `serving-frameworks` — vLLM, SGLang, Dynamo, Triton, TensorRT-LLM, Ray Serve, KServe
- `training-frameworks` — DDP/FSDP, DeepSpeed, Megatron, NeMo, Ray Train, Kubeflow Trainer, MaxText
- `slurm-hpc-on-kubernetes` — Slurm/HPC, Slinky, Volcano, MPI, RDMA, Slurm-vs-K8s
- `gke-master` — GKE Standard/Autopilot, TPU/GPU node pools, networking, security, autoscaling
- `autoscaling-kubernetes` — HPA/VPA/Cluster Autoscaler/Karpenter/KEDA/NAP

## Voice / formatting

- Markdown. `##`/`###` headings, tight bullets, fenced code with language tags.
- No marketing language, no "in today's fast-paced world" filler. Engineer-to-engineer.
- Tables for comparisons. Keep line length readable (~100 cols, wrap prose).
