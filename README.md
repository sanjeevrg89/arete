# Arete

*är-ə-tā* (Greek, ἀρετή) — excellence; the habit of meeting your full standard.

[![validate-skills](https://github.com/sanjeevrg89/arete/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjeevrg89/arete/actions/workflows/ci.yml)

**Your agent has read all the docs. It has never been paged at 3 a.m.**

Arete closes that gap: 59 skills of production knowledge for Kubernetes, GKE, and the ML-infrastructure
stack — failure signatures, review rules, debugging methods, war stories — plus 25 curated process
skills from [mattpocock/skills](https://github.com/mattpocock/skills). Any agent loads what it needs,
when it needs it.

## Install (~30 seconds)

**Claude Code**

```
/plugin marketplace add sanjeevrg89/arete
/plugin install arete@arete
```

**Codex, Cursor, Gemini CLI, and 40+ other agents**

```bash
npx skills add sanjeevrg89/arete
```

**Clone it and own it**

```bash
git clone https://github.com/sanjeevrg89/arete.git && cd arete
./install.sh claude     # or: ./install.sh list, ./install.sh flat <dest>
```

## What docs-knowledge gets wrong

### It writes infra that passes CI and fails production

Plausible ≠ survivable. Every arete skill carries **non-negotiables** and a **reject-in-review** list
distilled from incidents:

> - **No naked Pods** — always a Deployment/StatefulSet/DaemonSet/Job.
> - **Set resource `requests`; memory `limit == request`** (memory is incompressible → OOMKilled).
> - **A liveness probe that checks a dependency causes self-inflicted CrashLoopBackOff.**
> - **Tolerations don't attract — pair them with affinity.**
>
> — `kubernetes-expert`, non-negotiables

### "Check the logs" is not debugging

Practitioners pattern-match failure signatures; agents guess. The skills encode the tables:

| Pod state | Actual meaning | First move |
|---|---|---|
| `CrashLoopBackOff` | app crashes **or a bad liveness probe restarting a healthy app** | `logs --previous`; check probe config |
| `OOMKilled` (exit 137) | memory limit hit | raise limit / fix leak; `request == limit` |
| `CreateContainerConfigError` | missing ConfigMap/Secret key | `describe` Events |
| `Pending` | nothing fits: resources, quota, PV, taints | `describe` → scheduler events |

— `kubernetes-expert`, §9

And at GPU scale, the truth that isn't in any quickstart:

> One thermal-throttled straggler rank makes every rank's nvidia-smi read ~100% — they're all busy
> *waiting in the all-reduce*. The fingerprint: the straggler shows **less** collective-wait than
> everyone else, because everyone waits on it.
>
> — `gpu-performance-engineering`, straggler differential method

### Expertise dies when you switch tools

Skills locked into one assistant vanish when you switch. Arete keeps **one source of truth per skill**
and ships it everywhere agents look: `SKILL.md` (Claude Code + the open standard), `AGENTS.md`
(Codex/Cursor/IDEs), `GEMINI.md` (Gemini CLI), and a flat bundle for anything else.

### Skill libraries rot

Most collections are frozen PDFs of prompts. Arete runs a loop: failures feed
[`feedback/log.jsonl`](feedback/README.md) → ranked candidates → a reviser opens PRs behind CI and human
review → lessons become regression checks. Green CI ≠ validated either — see the
[5-layer validation harness](tests/VALIDATION.md).

## Browse the library

Full index in [REGISTRY.md](REGISTRY.md). The shape of it:

| Domain | Skills (selection) |
|---|---|
| Kubernetes | use · controller · operator · source-level internals |
| GKE & compute | GKE masterclass · autoscaling · Kueue · JobSet/LWS · Slurm-on-K8s |
| ML training | frameworks · training at scale · checkpointing · tokenizers/data · RLHF/DPO/GRPO |
| ML serving | vLLM/SGLang/TensorRT-LLM · inference optimization · GKE inference gateway |
| ML craft | system design · evals · RAG/vector DBs · embeddings · multimodal · graph ML · recsys |
| Engineering discipline | lifecycle · spec-first · TDD · code review · verification/debugging · Staff-plus craft |

Process layer vendored verbatim from mattpocock/skills (MIT): grilling, to-spec/to-tickets, TDD,
code-review, diagnosing-bugs, wayfinder, and more — see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Where topics overlap, both descriptions are scoped so
your agent routes correctly.

## Usage & contribution

- [USAGE.md](USAGE.md) — getting 10–100x out of the library
- [SKILL-AUTHORING-SPEC.md](SKILL-AUTHORING-SPEC.md) — write a skill to the house bar
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR flow, CI gates
- `python scripts/validate.py` — stdlib-only validator, run on every push

## Design notes

- **On-demand loading makes a big library viable.** Agents see only router descriptions and load the
  one relevant guide — never everything.
- **Depth lives in `<name>-guide.md`.** Entry files defer to it; AGENTS/GEMINI files stay small because
  they're always-on.
- **Version honesty.** K8s/GKE/frameworks move fast; guides flag version-sensitive claims and tell you
  to verify against current upstream docs.

Apache-2.0 for first-party content; vendored skills keep their upstream licenses.
