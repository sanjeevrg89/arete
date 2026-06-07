# Validating the skills

A skill is **not** "good" because it exists or reads well. It's validated when it (1) is structurally
sound, (2) routes for the right tasks, (3) produces **correct, expert-grade output on a real repo/task**,
and (4) its load-bearing claims hold up against current authoritative sources. The guides are dense
starting points authored from public sources — they explicitly say "verify against current docs." This
is how you do that.

Validation has five layers, cheapest → most valuable. The first three are automatable; the last two
need an agent and human judgment, and are where real confidence comes from.

---

## Layer 1 — Structural (automated, in CI)
Metadata, frontmatter, unique names, required files, resolvable `[[cross-links]]`, no banned framing.
```bash
python scripts/validate.py        # must be 0 errors
```

## Layer 2 — Functional spec lint (automated, in CI)
The functional check specs are well-formed and reference real skills.
```bash
python scripts/functional_test.py --lint
```

## Layer 3 — Routing (semi-automated)
Confirm the agent picks the **right** skill for a task. In Claude Code (skills installed):
- `/skills` lists every skill.
- Work through [`skill-routing-checklist.md`](skill-routing-checklist.md): paste each prompt prefixed
  with *"Pick the right skill(s), name them and why, then answer:"* and confirm the expected skill is named.
- A miss → sharpen that skill's `SKILL.md` `description` (the router) with the missing trigger terms, re-test.

## Layer 4 — Functional behavior (agent-in-the-loop)
Does the output satisfy concrete assertions?
```bash
AGENT_CMD='claude -p' python scripts/functional_test.py     # runs tests/functional/checks.json
```
Add a check per skill you care about (prompt + `must_contain`/`must_not_contain` regexes). This catches
"sounds right but says the wrong thing."

## Layer 5 — Real-repo / real-task validation (the gold standard)
Everything above is necessary but not sufficient. The real test: **use the skill on an actual codebase
and judge whether it produced correct, senior-level output.**

### Procedure (per skill)
1. **Pick a representative repo + task** that exercises the skill. Use your own or a real OSS repo. Examples:
   | Skill | Repo to point at | Task |
   |-------|------------------|------|
   | `go-best-practices` | any Go service | "Review this package for the issues in the skill." |
   | `kubernetes-controller-expert` | a controller-runtime operator | "Review the reconciler for hot-loops, finalizers, idempotency." |
   | `serving-frameworks` / `inference-optimization` | a model-serving repo | "Recommend an engine + optimization for this model & SLO." |
   | `rag-vector-databases` | a RAG app | "Find retrieval-quality problems and fix them." |
   | `k8s-manifest-scaffolder` (doer) | empty dir | "Generate manifests for spec X" → apply `--dry-run=server`. |
   | `accelerator-memory-estimator` (doer) | n/a | "Estimate memory for model X" → compare to a real run's `torch.cuda.max_memory_allocated()`. |
2. **Run it two ways (A/B)** for a fair read:
   - **Baseline:** an agent **without** the skill (e.g. a checkout with no skills installed, or a vanilla model).
   - **With skill:** the same agent + skills installed, *"pick the right skill(s) and do the task."*
3. **Judge the delta against a rubric** (1–5 each):
   - **Correctness** — is the advice/output actually right? (the decisive axis)
   - **Depth** — did it catch real, non-obvious issues a senior would?
   - **Discipline** — did it follow the skill's verification gate and avoid the listed anti-patterns/rationalizations?
   - **Accuracy of claims** — spot-check specifics (flags, APIs, formulas, version notes) against the cited docs.
   - **Lift over baseline** — is the with-skill answer meaningfully better?
4. **Record the result** in [`validation-log.md`](validation-log.md) and **feed failures back**:
   - Routing miss → fix the `description`.
   - Wrong/outdated content → fix the guide; add a `must_*` functional check so it can't regress.
   - Right but shallow → strengthen the guide's practices/red-flags.
5. **Verify against a real outcome where possible** — doer skills especially: did the estimate match the
   real memory? did the manifest `kubectl apply --dry-run=server` cleanly and run? did the kernel pass
   `torch.allclose` and beat the baseline?

### A capture helper
Save evidence for the log:
```bash
AGENT_CMD='claude -p'
echo "<your task prompt>" | eval "$AGENT_CMD" | tee tests/evidence/$(date +%s)-<skill>.md
```
(Run the baseline in a checkout without skills to capture the A side.)

## Content-accuracy pass (do this periodically)
The ecosystem moves fast. For each skill, spot-check the **load-bearing** claims (formulas, API/flag
names, version notes, benchmark mentions) against the cited authoritative sources. Anything marked
"verify against current docs" in a guide is a deliberate flag to check before you rely on it.

---

## When is a skill "validated"?
Mark it validated in `validation-log.md` when:
- [ ] Layer 1–2 pass (CI green).
- [ ] It routes correctly (Layer 3).
- [ ] It has ≥1 functional check that passes with a real agent (Layer 4).
- [ ] It produced **correct, expert-level** output on ≥1 real repo/task with lift over baseline (Layer 5).
- [ ] Its load-bearing claims were spot-checked against current sources.

Re-validate fast-moving skills (serving, frameworks, GKE/SKUs, benchmarks) on a cadence — "validated"
has a shelf life.

## Honest limits
- CI proves *structure*, not *correctness*. A guide can be perfectly formatted and wrong.
- Functional checks prove the output contains expected tokens, not that the reasoning is sound.
- Only Layer 5 + the accuracy pass give real confidence, and they need your judgment (or a domain
  expert's). Budget for it; don't treat green CI as "validated."
