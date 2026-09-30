# Kaushal

*kow-shul* (Sanskrit, कौशल) — skill; excellence in action. From the Gita: *yogaḥ karmasu kauśalam*,
"yoga is skill in action" (2.50).

[![validate-skills](https://github.com/sanjeevrg89/kaushal/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjeevrg89/kaushal/actions/workflows/ci.yml)

**Your agent has read all the docs. It has never been paged at 3 a.m.**

Kaushal closes that gap: 59 skills of production knowledge for Kubernetes, GKE, and the ML-infrastructure
stack — failure signatures, review rules, debugging methods, war stories — plus 25 curated process
skills from [mattpocock/skills](https://github.com/mattpocock/skills). Any agent loads what it needs,
when it needs it.

## Install (~30 seconds)

**Claude Code**

```
/plugin marketplace add sanjeevrg89/kaushal
/plugin install kaushal@kaushal
```

**Codex, Cursor, Gemini CLI, OpenCode, Qwen Code, and 70+ other agents**

```bash
npx skills add sanjeevrg89/kaushal
```

**Clone it and own it**

```bash
git clone https://github.com/sanjeevrg89/kaushal.git && cd kaushal
./install.sh all        # every agent on this machine, as symlinks — git pull updates them
```

### Tested agents

"Works everywhere" is checked, not assumed — each agent looks in different folders and parses
frontmatter its own way.

| Agent | Where it finds the skills | How that was checked |
|---|---|---|
| Claude Code | plugin, or `~/.claude/skills` | a real session's skill listing: 70 / 70 auto-loadable skills show their description |
| Codex CLI | `~/.agents/skills` | `codex debug prompt-input`: 70 / 70 auto-loadable skills in the prompt |
| Gemini CLI | `~/.agents/skills` | `gemini skills list`: 84 / 84 |
| OpenCode | `~/.agents/skills`, `~/.claude/skills` | `opencode debug skill`: 84 / 84 |
| Cursor agent | `~/.agents/skills` | its loader reads that folder and follows links (source check) |
| Qwen Code | `~/.qwen/skills` — its only folder | its loader follows links (source check); its frontmatter parser, run on all 84 files |
| 70+ others | `npx skills add` puts them in each agent's folder | the installer finds and installs 84 / 84 |

`./install.sh all` sets up every row but the last. Re-run the checks on your own machine with
`python scripts/agent_check.py`. Tested 2026-09-30 with Claude Code 2.1.283, Codex CLI 0.159.2, Gemini
CLI 0.62.0, OpenCode 1.18.30, Cursor agent 2026.08.11, and Qwen Code 0.12.6; old Gemini CLI releases
(0.17, for one) have no skills support at all.

## What docs-knowledge gets wrong

### It writes infra that passes CI and fails production

Plausible ≠ survivable. Every Kaushal skill carries **non-negotiables** and a **reject-in-review** list
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

Skills locked into one assistant vanish when you switch. Kaushal keeps **one source of truth per skill**
and ships it everywhere agents look: `SKILL.md` (the open standard — Claude Code, Codex, Gemini CLI,
Cursor), `AGENTS.md` (always-on rules for any agent), `GEMINI.md` (Gemini CLI imports), and a flat
bundle for anything else.

### Big skill libraries go invisible

An agent picks a skill from one line of description, and every installed skill shares one listing
budget. Past it, Claude Code lists skills by name only and Codex truncates descriptions. Kaushal's
routers are one line, ≤ 250 characters, strict YAML, and CI-checked, so the whole library stays visible
next to your other skills. Measured on Claude Code's real skill listing (30,000-character budget, Kaushal
installed alone), with [one discriminating prompt per skill](tests/skill-routing-checklist.md):

| | v1.0 | v1.1 |
|---|---|---|
| Skills the agent can see a description for | 29 / 70 | **70 / 70** |
| Routing accuracy, top-1, 59 prompts (Haiku · Sonnet) | 75% · 85% | **100% · 98%** |
| Skills installed by `npx skills add` (Codex, Cursor, Gemini CLI…) | 59 / 84 | **84 / 84** |

Short didn't cost accuracy: shown in full with no budget, the old ~1,000-character routers scored 98%
on Haiku; the new ones score 100% in a fifth of the space. Re-run it: `python scripts/routing_eval.py`.

### Skill libraries rot

Most collections are frozen PDFs of prompts. Kaushal ships the parts of an improvement loop: a feedback
log ([`feedback/log.jsonl`](feedback/README.md)) for what a skill got wrong, a ranker that turns it
into improvement candidates, and CI that gates every change. Nothing edits a skill by itself — using a
skill does not change it, and the reviser that turns feedback into pull requests is off until you
enable it. Green CI ≠ validated either — see the [5-layer validation harness](tests/VALIDATION.md).

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

## Make it yours

Fork it. Delete the skills you don't need, add your own under `skills/<name>/`, and replace
[`rules/AGENTS.md`](rules/AGENTS.md) with the rules you want every agent to follow. Then:

```bash
./install.sh all                     # every skill → Claude Code, Codex, Gemini CLI, OpenCode, Cursor, Qwen Code
./install.sh rules rules/AGENTS.md   # one rules file → each agent's global instructions (CLAUDE.md, AGENTS.md, GEMINI.md, QWEN.md)
```

Change a skill or a rule once and every agent picks it up; `./update.sh` pulls and re-links, and
`./install.sh uninstall` removes only what it linked. CI keeps a fork honest: strict YAML, router
limits, the Agent Skills spec, and a routing case for every skill. The one-folder-for-every-agent
pattern comes from [steipete/agent-scripts](https://github.com/steipete/agent-scripts).

## Usage & contribution

- [USAGE.md](USAGE.md) — getting 10–100x out of the library
- [SKILL-AUTHORING-SPEC.md](SKILL-AUTHORING-SPEC.md) — write a skill to the house bar
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR flow, CI gates
- `python scripts/validate.py` — strict frontmatter + router-limit validator, run on every push
- `python scripts/routing_eval.py` — routing eval: does the agent pick the right skill from descriptions alone?
- `python scripts/agent_check.py` — which agents on this machine can actually see the skills

## Design notes

- **On-demand loading makes a big library viable.** Agents see only router descriptions and load the
  one relevant guide — never everything. Routers stay ≤ 250 characters because they all share one
  listing budget; the long form lives in each skill's `## Scope and triggers`.
- **Depth lives in `<name>-guide.md`.** Entry files defer to it; AGENTS/GEMINI files stay small because
  they're always-on.
- **Version honesty.** K8s/GKE/frameworks move fast; guides flag version-sensitive claims and tell you
  to verify against current upstream docs.

Apache-2.0 for first-party content; vendored skills keep their upstream licenses.
